# ARD-Gen 파이프라인 설계 문서

> 이 문서는 설계 전체를 미리 정리해둔 참고 자료다. **아직 구현되지 않은
> 단계가 대부분이며, 실제 구현은 이 문서를 보고 그때그때 요청받은 단계만
> 진행한다.** 각 단계 구현 후에는 이 문서의 해당 단계 상태를 갱신한다.

## 배경

**ARD-VLA**는 양팔 로봇의 두 팔에 서로 다른 역할을 부여하는 아키텍처다.

- **오른팔 (Actuator)**: 도구를 쥐고 미세조작을 수행한다.
- **왼팔 (Stabilizer)**: 물체를 고정하는 역할을 한다.

이 모델을 학습시킬 데이터가 필요하지만, 텔레오퍼레이션 장비도 GPU 학습
자원도 없는 환경(군 복무 중, Colab + Claude Code만 사용 가능)이라
**시뮬레이션만으로 데이터를 생성하는 파이프라인**을 설계했다 — 그게
ARD-Gen이다.

## 핵심 설계 원칙

1. **두 팔은 증강 방식이 다르다.** Stabilizer는 기하학적 변환(MimicGen
   스타일)으로, Actuator는 물리 기반 방식(diffusion)으로 증강한다.
2. **계산 비용이 싼 방법부터 쓴다.** RL처럼 정책을 처음부터 학습시키는
   무거운 방법 대신, 이미 설계된 admittance controller의 파라미터만
   탐색하는 가벼운 방법을 우선한다.
3. **두 팔의 데이터는 물리적으로 일관돼야 한다.** Actuator의 반작용이
   Stabilizer에도 전달되므로, 공유된 씬 조건 위에서 Actuator를 먼저 생성하고
   그 결과를 참고해 Stabilizer를 생성한다.

## 전체 파이프라인 (6단계)

### 0단계 — Seed 확보 ✅ 완료

CMA-ES로 admittance controller의 게인(`Kp_xy`, `Kd_xy`)을 탐색해서,
peg-in-hole 태스크를 최초로 성공시킨 궤적을 확보했다.

- 구현 위치: `optimize/cma_search.py`, `sim/peg_in_hole_sim.py`
- 산출물: `seed_trajectory.npz` (게인 + 궤적 + force/torque)
- 검증: `evaluate_generalization.py`로 무작위 시나리오 500건+ 대상 성공률
  99~100% 확인 (게인=0 베이스라인은 70~75%)
- 부가 작업(스코프 밖이었지만 먼저 해봄): `bootstrap/`에 이 seed를
  행동 복제(behavior cloning)로 신경망 정책에 재현하는 실험 — 구조적
  수정(bias-free, f(0)=0 보장) 후 성공률 100% 확인. **2-A 구현 결과,
  이 정책은 2단계와 목적이 달라 재사용하지 않기로 함** (아래 2-A 참고).

### 1단계 — 공유 씬 설정 ✅ 완료

매 에피소드마다 다음을 무작위로 샘플링한다:
- hole 위치/자세: `hole_pose = (dx, dy, theta)`, dx/dy는 ±1cm, theta는
  ±0.1rad(현재 sim은 아직 회전을 지원하지 않아 소비되지 않음 — 3단계용
  선반영)
- 마찰계수: 기준값(0.5)의 0.7~1.3배
- clearance: 2mm~5mm (friction보다 넓게 — margin 무작위화가 더 안정적인
  신호라는 원칙)
- peg 초기 오프셋: ±1.5cm

- 구현 위치: `pipeline/scene_sampler.py` (`sample_scene_config()`,
  `to_sim_scene_config()`)
- `sim/peg_in_hole_sim.py`의 기존 `sample_scene_config()`/
  `_default_scene_config()`는 CMA-ES 전용으로 남겨두고 건드리지 않음 —
  파이프라인 공식 스키마는 `pipeline/scene_sampler.py` 쪽. `run_episode()`에
  넘기려면 `to_sim_scene_config()`로 변환.

### 2단계 — Actuator(오른팔) 증강: Diffusion

seed 하나만으로는 diffusion을 학습시킬 수 없으므로 두 단계로 나눈다.

- **2-A 부트스트래핑 ✅ 완료**: seed 게인 주변에 넓은 랜덤 노이즈를 주고
  시뮬레이션을 반복 실행해서 `(씬 조건, 게인, 성공여부, force_profile)`
  기록을 대량으로 쌓는다.
  - 구현 위치: `pipeline/bootstrap.py`
  - 방식: `pipeline.scene_sampler.sample_scene_config()`로 씬 샘플링,
    seed 게인(Kp_xy, Kd_xy) 각각에 독립적으로 [0.5, 2.0]배 균등분포
    노이즈를 곱해서 `sim/peg_in_hole_sim.py`의 `run_episode()` 그대로 실행.
    성공/실패 모두 저장한다 — 2-B 구현 결과 diffusion 자체는 성공
    샘플만으로 학습시켰지만(아래 참고), 실패 샘플도 조건(scene_config)
    정규화 통계 계산에는 쓰인다.
  - 산출물: `data/bootstrap/bootstrap_dataset.npz` (1000건 기본)
  - **실제 실행 결과(N=1000, seed=0)**: 전체 성공률 76.0%. 성공 게인
    분포는 `Kp_xy` 평균 0.000649 (seed 0.000515보다 약간 큼), `Kd_xy`
    평균 3.19e-05. clearance로 4분위 나눠 보면 2.0~2.76mm 구간
    성공률 64.4% → 4.22~5.00mm 구간 83.2%로 뚜렷하게 clearance가
    넓을수록 성공률이 오른다. offset 크기도 성공군 평균 10.6mm vs
    실패군 평균 14.7mm로 명확히 갈리는 반면, friction은 성공군 0.498 vs
    실패군 0.507로 거의 차이가 없다 — 이 태스크에서 friction보다
    clearance/offset이 성공/실패를 가르는 훨씬 강한 신호라는 게 실측으로
    확인됨(설계 원칙 "margin 무작위화가 더 안정적인 신호"와 일치).
- **2-B diffusion 가동 ✅ 완료**: 축적된 기록으로 diffusion 모델을
  학습시켜서, 새로운 씬 조건이 주어지면 랜덤 샘플링 대신 diffusion이
  성공 확률 높은 게인 조합을 직접 생성하도록 전환한다.
  - 구현 위치: `pipeline/diffusion_gains.py`
  - 모델: 조건(7차원: hole_pose 3 + friction 1 + clearance_m 1 +
    peg_init_offset 2) + timestep을 받아 노이즈를 예측하는 작은 MLP
    (`GainDiffusionNet`, hidden=128) + 표준 DDPM 스케줄
    (`GaussianDiffusion`, T=100).
  - **학습 데이터는 성공 샘플만 사용**(1000건 중 760건). 조건 정규화
    통계(평균/표준편차)만 전체 1000건으로 계산 — 성공 샘플만으로
    계산하면 "쉬운 씬" 쪽으로 치우친 통계가 나와서다. 실패까지 포함해서
    diffusion을 학습시키는 classifier-guidance 방식도 검토했지만, 조건
    7차원/출력 2차원짜리 저차원 문제에는 과한 복잡도라 판단해
    "성공 사례만 보고 그 분포를 재현"하는 단순한 방식을 택함(자세한 이유는
    스크립트 docstring 참고).
  - 생성된 게인은 `optimize/cma_search.py`와 동일한 탐색 범위
    (`Kp_xy∈[2e-5,3e-3]`, `Kd_xy∈[0,5e-4]`)로 clip해서 물리적으로 말이
    안 되는 값(음수 등)을 방지.
  - **실제 검증 결과**: 새 무작위 씬 100개(2-A 데이터 생성에 쓰지 않은
    시드)에 대해 diffusion이 생성한 게인으로 직접 시뮬레이션 실행.
    - 시드 999: **86.0%** (86/100)
    - 시드 1234: **84.0%** (84/100)
    - 시드 7777: **84.0%** (84/100)
    - 2-A 부트스트래핑(무작위 노이즈) 베이스라인 **76.0%** 대비 세 시드
      모두에서 **+8~10%p 일관되게 개선** — 우연이 아님을 확인.
    - diffusion이 생성한 게인 분포는 `Kp_xy≈0.00064±0.00024`,
      `Kd_xy≈3.2e-05±1.1e-05`로, 2-A의 성공 게인 분포(`Kp_xy` 평균
      0.000649)와 비슷한 영역이지만 씬 조건에 따라 값을 조절해서 뽑는다는
      점이 다르다(무작위 노이즈는 조건과 무관하게 넓게 뿌리기만 함).

> `bootstrap/`(구 디렉토리, 행동 복제 MLP 정책)와 `pipeline/bootstrap.py`
> (2-A)는 이름은 비슷하지만 다른 것이다 — 전자는 "force 상태 → 행동"을
> 흉내내는 정책이고, 후자는 "씬 조건 → 게인이 성공하는지"를 기록해 2-B
> diffusion의 학습 데이터를 만드는 것. 실제 구현해보니 서로 재사용할
> 부분이 없어 별도로 유지한다.

### 3단계 — Stabilizer(왼팔) 증강: 기하 변환 ⬜ 미구현

MimicGen 방식:
- seed의 Stabilizer 궤적을 물체 기준 좌표계로 저장
- 새 씬의 물체 포즈에 맞춰 SE(3) 변환을 적용해 왼팔 궤적을 재생성
- 물리 시뮬레이션이 필요 없어 계산 비용이 가장 낮음
- 단, 2단계에서 나온 접촉력이 threshold를 넘는 구간에서는 Stabilizer
  목표 위치에 작은 반발 방향 보정을 추가해 "버티는" 반응을 표현

### 4단계 — 동시 실행 & 필터링 ⬜ 미구현

- Actuator + Stabilizer 궤적을 같은 시뮬레이션에서 동시에 재생
- 태스크 성공 조건 확인
- 실패 시 에피소드 전체 폐기
- 성공한 것만 **LeRobotDataset** 포맷으로 저장 (role 메타데이터 포함)

### 5단계 — LLM 기반 언어 라벨링 ⬜ 미구현

성공한 궤적의 force/torque profile을 LLM에 넘겨 자연어 지시문으로 변환:
- 토크 부호 → 회전 방향
- 회전수 → 수량
- 최대 접촉력 → 강도

최종 산출물: **행동 + force/torque + 언어**가 모두 포함된 VLA 학습용
데이터셋.

## 검증 태스크

- **1차**: peg-in-hole(위치보정), 뚜껑돌리기(토크제어) — 두 단순 태스크로
  파이프라인 자체가 작동하는지 검증
- **추후 확장**: GPU/실물 로봇 확보 시 0단계를 RL(RLDC 스타일)로 교체해서
  드릴 조작 같은 복잡한 다단계 태스크로 확장

## 현재 저장소 상태와의 매핑

| 파이프라인 단계 | 관련 코드 | 상태 |
|---|---|---|
| 0단계 (Seed 확보) | `optimize/cma_search.py`, `sim/peg_in_hole_sim.py` | ✅ 완료 |
| (스코프 외) 부트스트랩 정책 실험 | `bootstrap/` | ✅ 완료 (2단계와는 무관, 별도 유지) |
| 1단계 (공유 씬 설정) | `pipeline/scene_sampler.py` | ✅ 완료 |
| 2-A (게인 부트스트래핑) | `pipeline/bootstrap.py`, `data/bootstrap/` | ✅ 완료 (N=1000, 성공률 76.0%) |
| 2-B (diffusion) | `pipeline/diffusion_gains.py`, `data/bootstrap/diffusion_gains.pt` | ✅ 완료 (검증 성공률 84~86%, 베이스라인 76% 대비 +8~10%p) |
| 3단계 (Stabilizer 기하 변환) | — | ⬜ 미구현 |
| 4단계 (동시 실행 & 필터링) | — | ⬜ 미구현 |
| 5단계 (언어 라벨링) | — | ⬜ 미구현 |

지금 저장소(`ard-gen/`)는 **오른팔(Actuator) 단일 팔, 0~2단계(0, 1, 2-A,
2-B)까지** 구현된 상태다. 왼팔(Stabilizer) 자체가 아직 없고, 3단계
이후는 전부 새로 설계/구현해야 한다.
