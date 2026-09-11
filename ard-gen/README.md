# ard-gen

MuJoCo 기반 peg-in-hole 태스크에서 admittance controller 게인을 CMA-ES로 찾는
파이프라인. **ARD-Gen 파이프라인의 0단계(Seed 확보)**에 해당한다 — 여기서 찾은
"성공하는 게인 + 그 궤적"(`seed_trajectory.npz`)이 다음 단계(2단계, seed를 이용한
부트스트래핑)의 입력이 된다.

## 디렉토리 구조

```
ard-gen/
├── assets/peg_in_hole.xml       # MuJoCo 모델
├── sim/peg_in_hole_sim.py       # 시뮬레이션 실행 + admittance controller
├── optimize/cma_search.py       # CMA-ES 최적화 스크립트
├── render_result.py             # seed_trajectory.npz의 게인으로 롤아웃을 mp4로 렌더링
├── requirements.txt
└── README.md
```

## 모델 (assets/peg_in_hole.xml)

- **팔**: 3-DoF slide(x,y,z) + 1 hinge(wrist). 각 조인트는 서로 결합되지 않은
  독립 position actuator로 구동된다 (다른 프로젝트(rldc-lite)에서 SCARA형
  회전 조인트 팔을 쓰다가 "회전각이 조인트별로 누적돼서 관측이 꼬이는" 버그를
  겪었던 적이 있어서, 이번엔 일부러 단순한 슬라이드 구성을 유지했다).
- **peg**: wrist의 자식이지만 자체 조인트가 없음 → 그리퍼에 강체로 고정된
  것으로 취급. **tip이 구형(반지름 = peg half-width 10mm)** — 이유는 아래
  "검증하며 알아낸 것들" 참고.
- **hole**: 4개 벽 세그먼트 + 바닥으로 이루어진 소켓. nominal clearance
  3mm(편측 1.5mm)지만, `scene_config['clearance_m']`에 따라 `sim`이 런타임에
  벽의 `geom_pos`/`geom_size`를 직접 덮어써서 바꾼다 (XML의 값은 초기값일 뿐).
- **F/T 센서**: `peg_tip_site`에 3축 force + 3축 torque.
- 목표 삽입 깊이는 0.04m(hole 전체 깊이는 0.055m로, 바닥까지 1.5cm 여유).

## 시뮬레이션 + admittance controller (sim/peg_in_hole_sim.py)

```
F_error_xy = F_ext_xy - F_desired_xy        (F_desired_xy = 0)
dF_error_xy/dt ≈ (F_error_xy - F_error_xy_prev) / dt
Δx_xy = -Kp_xy * F_error_xy - Kd_xy * dF_error_xy/dt
```

삽입 방향(z)은 게인과 무관하게 매 스텝 일정 속도(`Z_RATE`)로 내려간다. wrist는
이 단계에서는 능동 보정하지 않고 초기값을 유지한다(peg가 이미 회전 정렬돼
있다고 가정 — 아래 한계 섹션 참고).

- **함수 시그니처**: `run_episode(gains: dict, scene_config: dict) -> dict`
  - `gains`: `{"Kp_xy": float, "Kd_xy": float}`
  - `scene_config`: `hole_pos_xy`, `friction`, `clearance_m`,
    `peg_init_offset_xy`, `peg_init_wrist`, `target_insertion_depth`
  - 반환: `ee_poses (T+1,4)`, `actions (T,4)`, `forces/force_profile (T,3)`,
    `torques/torque_profile (T,3)`, `insertion_depth`, `success`, `reward`,
    `max_force`, `step_count`
- 리워드는 **에피소드가 끝난 뒤 한 번** 계산되는 스칼라다 (CMA-ES 적합도로
  바로 쓰기 위함 — 매 스텝 누적하는 RL식 리워드가 아니다):
  ```
  reward = -2.0*dist + 20.0*insertion_depth - 0.001*max(0, max_force-5.0) - 0.01*step_count
  (+ 50.0 if insertion_depth >= target_insertion_depth)
  ```
  `dist`는 에피소드 종료 시점의 3D 거리, `max_force`는 에피소드 전체에서
  관측된 힘 크기의 **최댓값**(피크), `step_count`는 종료까지 걸린 스텝 수.
- `insertion_depth`는 z 깊이만 보지 않고, peg tip이 hole 벽 구조물의 실제
  xy footprint 안에 있을 때만 인정한다(rldc-lite에서 겪은 "정렬 없이 허공으로
  내리꽂아도 성공으로 인정되는" 리워드 해킹을 처음부터 막기 위함).

## 검증하며 알아낸 것들 (실제로 돌려보고 고친 것)

1. **평평한 peg 바닥은 admittance control이 아예 안 됨**: 처음엔 peg tip이
   평평한 박스였다. peg 바닥과 벽 윗면이 둘 다 수평면이라, 오프셋을 아무리
   줘도 접촉 normal이 항상 수직이었다(기하학적으로 당연함) — `F_ext_xy`가
   **항상 정확히 0**이라 Kp_xy를 얼마로 줘도 결과가 완전히 똑같았다. 실제
   커넥터/핀에 흔한 라운드팁과 같은 이유로 peg tip을 구형(반지름=peg
   half-width)으로 바꿔서 오프셋 상태의 접촉에 xy 성분이 생기게 했다
   (`assets/peg_in_hole.xml`의 `peg_tip_ball`).
2. **부호가 반대였음**: 구형 tip으로 고친 뒤, `Δx = +Kp*F_error`로 처음
   구현했더니 오프셋 (7mm,-5mm)에서 게인을 어떤 값으로 줘도(양수든 음수든
   무관하게 처음엔 양수로만 테스트) 오히려 게인이 없을 때보다 안 좋아지거나
   말이 안 되는 결과가 나왔다. 부호를 뒤집어(`Δx = -Kp*F_error`) 다시 테스트하니
   최대 접촉력이 60.5N(게인 0) → 6.8N(좋은 게인)까지 줄고 안정적으로
   성공했다. 최종 코드는 이 검증된 부호로 구현돼 있다.
3. **오프셋 난이도 구간 확인**: 오프셋이 2~6mm면 peg tip의 곡률만으로
   게인 없이도 대부분 자가정렬(캠 효과)되고, 12mm 이상이면 게인을 아무리
   키워도 평평한 면끼리 걸려서 안 됨(admittance controller가 반응할 신호가
   생기기도 전에 막힘). 6~10mm가 "게인이 있어야만 성공"이 실제로 갈리는
   구간이라, CMA-ES 평가 시나리오와 `sample_scene_config()`의 기본 오프셋
   범위를 여기에 맞췄다.

## CMA-ES 최적화 (optimize/cma_search.py)

- `cma` 패키지로 `[Kp_xy, Kd_xy]` 2차원을 탐색한다. 스케일이 크게 다른 두
  파라미터라 `CMA_stds` 옵션으로 좌표별 초기 스텝 크기를 따로 준다.
- 매 세대, 실측으로 확인한 대표 시나리오 3개(오프셋 (9,-6), (9,7), (9.5,-7)mm)
  각각에 대해 `run_episode()`를 돌려 리워드를 평균낸다 — 시나리오 하나에만
  과적합하지 않도록.
- 종료 조건: 평균 리워드가 `--threshold`에 도달 **하거나** `--max-generations`
  (기본 100)에 도달하면 멈춘다.
- 종료 시: 최종 게인으로 대표 시나리오(가장 어려운 오프셋)를 다시 실행해
  그 궤적을 `seed_trajectory.npz`로 저장하고, 세대별 최고 리워드로
  수렴 곡선(`convergence.png`)을 그린다.

### seed_trajectory.npz 데이터 구조 (2단계 부트스트래핑 입력)

| 키 | shape | 설명 |
|---|---|---|
| `ee_poses` | `(T+1, 4)` | (x, y, z, wrist) 시퀀스, 초기 상태 포함 |
| `actions` | `(T, 4)` | 매 스텝 실제로 적용한 (Δx, Δy, Δz, Δwrist) |
| `forces` | `(T, 3)` | F/T 센서 force |
| `torques` | `(T, 3)` | F/T 센서 torque |
| `gains` | `(2,)` | `[Kp_xy, Kd_xy]` |
| `scene_config` | 스칼라(문자열) | `json.dumps()`된 scene_config — `json.loads(str(arr))`로 복원 |
| `success` | 스칼라(bool) | 목표 삽입 깊이 도달 여부 |
| `insertion_depth` | 스칼라(float) | 최종 삽입 깊이 |
| `reward` | 스칼라(float) | 최종 리워드 |
| `best_reward_per_gen` | `(G,)` | CMA-ES 수렴 곡선 원본 데이터 |

## 로컬 실행 (Colab 불필요, CPU면 충분)

```bash
pip install -r requirements.txt

python optimize/cma_search.py \
    --max-generations 100 \
    --popsize 10 \
    --threshold 49.0 \
    --out-path ./seed_trajectory.npz \
    --curve-path ./convergence.png
```

## 실제 실행 결과

- **수렴 여부**: 수렴함. 일부러 나쁜 초기 게인(`Kp_xy=0.0001`)과 작은 초기
  스텝 크기에서 시작해서, 5세대 동안은 대표 시나리오 전부 실패(평균
  리워드 -3.56)했다가 **6세대째에 리워드 49.14로 돌파**하며 threshold(49.0)에
  도달해 조기 종료했다.
- **최종 게인**: `Kp_xy ≈ 0.00282`, `Kd_xy ≈ 4.5e-06`
- **성공 여부**: 최종 게인으로 메인 시나리오(오프셋 9mm/-6mm) 재실행 결과
  `success=True`, `insertion_depth=0.0404m`(목표 0.04m 초과 달성),
  `reward=49.16`
- 수렴 곡선(`convergence.png`)은 5세대까지 평평하다가 6세대에서 수직으로
  치솟는 형태로, CMA-ES가 초기엔 실패만 하다가 스텝 크기를 키워가며 실제로
  좋은 영역을 찾아내는 과정을 보여준다.
- `render_result.py`로 최종 게인의 실제 롤아웃을 mp4로 렌더링해서 육안으로도
  확인함 (헤드리스 환경은 `MUJOCO_GL=osmesa` 필요, rldc-lite 프로젝트에서
  검증한 것과 동일한 소프트웨어 렌더러):
  ```bash
  MUJOCO_GL=osmesa python render_result.py \
      --seed-path ./seed_trajectory.npz --out-path ./result.mp4
  ```
  172스텝 만에 peg가 실제로 hole에 삽입되는 것을 영상으로 확인.

## 지금 임시로 되어있는/한계인 부분

- **wrist(회전) misalignment는 보정하지 않음**: admittance controller가 xy
  위치만 보정하고 wrist는 초기값 유지다. peg 초기 자세가 이미 정렬돼 있다고
  가정한 것 — 실제로는 회전 오차도 힘/토크 피드백으로 보정해야 할 수 있다.
- **오프셋 6~10mm 범위에서만 검증됨**: 12mm 이상은 이 반응형(reactive) 제어
  방식으로는 원래 못 푸는 문제다(스파이럴 서치 등 별도 탐색 동작이 필요).
  이건 버그가 아니라 "순수 힘 피드백 admittance control"의 실제 한계다.
- **hole clearance/actuator 게인/질량은 대략적인 값**: rldc-lite와 마찬가지로
  물리적으로 캘리브레이션된 값이 아니라 시뮬레이션이 안정적으로 도는 선에서
  잡은 값이다.
- **대표 시나리오 3개로만 평균냄**: 매 세대 무작위로 다시 뽑지 않고 고정된
  3개 시나리오를 쓴다. 더 견고한(generalize하는) 게인을 원하면
  `sim.sample_scene_config()`로 매 세대 새로 샘플링하도록 바꿀 수 있다(이미
  구현돼 있음, `cma_search.py`에서 아직 안 쓰고 있을 뿐).
