# rldc-lite

MuJoCo 기반 강화학습 데이터 수집 파이프라인의 축소 재현체.
[RLDC (Reinforcement Learning as Data Collector), arXiv:2606.22471] 의 핵심 아이디어인
"RL 정책을 학습시켜 그 롤아웃을 로봇 데이터로 재사용한다"를, peg-in-hole 삽입이라는
단일 태스크에 대해 최소 구성으로 검증하기 위한 프로젝트입니다.

## 1단계 태스크: peg-in-hole

- 단일 팔, 그리퍼는 peg를 이미 강체로 고정하고 있다고 가정 (grasp 자체는 다루지 않음)
- 검증 대상은 순수 삽입 미세조작(insertion fine-manipulation)뿐

## 디렉토리 구조

```
rldc-lite/
├── assets/peg_in_hole.xml     # MuJoCo 모델 (4-DoF 팔 + peg + hole 소켓)
├── envs/peg_in_hole_env.py    # gymnasium 환경
├── scripts/train_ppo.py       # PPO 학습 스크립트 (SubprocVecEnv, 체크포인트/재개 지원)
├── scripts/collect_data.py    # 학습된 정책으로 롤아웃 데이터 수집
├── scripts/render_video.py    # 학습된(또는 랜덤) 정책 롤아웃을 mp4로 렌더링
├── requirements.txt
└── README.md
```

## 모델 개요 (assets/peg_in_hole.xml)

- **팔**: slide(x) → slide(y) → slide(z) → hinge(wrist, z축 회전) 순서로 연결된
  단순화된 4-DoF 팔. 각 조인트는 position actuator로 구동되며, RL action은 이
  actuator의 목표값에 대한 델타(증분)로 들어간다.
- **peg**: wrist 바디의 자식이지만 그 자체는 조인트가 없음 → 그리퍼에 강체로
  고정된 것으로 취급. 사각 단면, half-width 10mm.
- **hole**: 4개의 벽 세그먼트(picture-frame 형태) + 바닥으로 이루어진 소켓.
  개구부 half-width 11.5mm → peg 대비 편측 clearance 1.5mm (총 지름 clearance
  약 3mm)로, 실제 접촉이 발생할 만큼 타이트하게 잡음.
- **F/T 센서**: `peg_tip_site`에 3축 force + 3축 torque 센서. peg가 wrist에
  조인트 없이 붙어 있으므로, 이 센서는 peg-wrist 간 구속력(=실제 접촉/무게 반력)을
  그대로 읽는다. 정지 상태에서도 peg 자중(~0.49N)은 기본으로 읽히고, 벽에
  부딪히면 그 위로 접촉력이 스파이크된다 (검증 섹션 참고).
- 팔의 각 바디에는 `gravcomp="1"`을 줘서 actuator가 자체 중력과 싸우지 않도록
  했다. 이는 제어 단순화를 위한 트릭이며, F/T 센서 값 자체(물리적 반력)에는
  영향을 주지 않는다.

## gymnasium 환경 (envs/peg_in_hole_env.py)

- **action** `Box(-1, 1, shape=(4,))`: delta position(x,y,z) + delta wrist
  rotation. 내부적으로 스텝당 최대 이동량(위치 3mm, 회전 0.05rad)으로 스케일된
  뒤 actuator ctrl에 누적/클리핑된다.
- **observation** `Box(shape=(19,))` — 아래 벡터들을 실제로 concat한 뒤
  `len()`으로 shape을 계산하므로(하드코딩 아님) 필드를 추가/삭제해도 shape이
  자동으로 맞는다:
  - ee_pos (3) — wrist 바디 world position
  - wrist (1) — wrist hinge 각도
  - peg_tip_pos (3)
  - hole_center_pos (3)
  - relative_vec (3) = hole_center_pos - peg_tip_pos
  - force (3), torque (3) — F/T 센서
- **reward** (범용, 태스크별 커스텀 최소화):
  `-distance_penalty + insertion_reward - force_penalty - step_penalty (+ success_bonus)`
  - 거리 페널티: `-20 * ||relative_vec||`
  - 삽입 보상: `+10 * clip(insertion_depth / target_depth, 0, 1)`
  - 과다 접촉력 페널티: 접촉력이 안전 임계값(15N)을 넘는 초과분에만 페널티
  - 스텝 페널티: 매 스텝 `-0.05` (빠른 해결 유도)
  - 목표 삽입 깊이(25mm) 도달 시 `success=True`, terminated, `+20` 보너스
- **도메인 무작위화** (`reset()`마다):
  - hole 위치: xy ±4mm, z ±2mm 지터 (조인트가 없는 hole 바디의 `model.body_pos`를
    직접 덮어쓰고 `mj_forward`로 재계산)
  - peg 마찰계수: sliding friction을 [0.2, 0.8] 범위에서 균일 샘플링
  - 초기 peg 오프셋: xy ±8mm(= clearance보다 훨씬 큰 정렬 오차), z 지터,
    wrist 각도 ±0.15rad

## 학습 스크립트 (scripts/train_ppo.py)

- stable-baselines3 PPO + `SubprocVecEnv` (n-envs > 1일 때) 병렬화
- `--checkpoint-freq`(전체 timestep 기준) 마다 `CheckpointCallback`으로 저장 →
  Colab 세션이 끊겨도 마지막 체크포인트부터 재개 가능
- `--save-dir`에 Google Drive 마운트 경로를 그대로 넘기면 체크포인트가 Drive에
  저장됨
- `--resume <checkpoint.zip>`으로 이어서 학습 (timestep 카운터도 이어짐)

## 데이터 수집 스크립트 (scripts/collect_data.py)

- 학습된 정책으로 롤아웃하며, 기본적으로 **성공한 에피소드만** `.npz`로 저장
  (`--keep-failures`로 실패 에피소드도 저장 가능)
- 각 에피소드 npz의 키: `ee_poses (T,3)`, `actions (T,4)`, `forces (T,3)`,
  `torques (T,3)`, `rewards (T,)`, `success ()`
- 모든 배열의 0번째 축이 스텝(T)이므로, 나중에 LeRobotDataset의
  프레임 단위 필드로 그대로 매핑하기 쉽다.

## 영상 렌더링 스크립트 (scripts/render_video.py)

- `--model-path`를 주면 학습된 정책, 안 주면 랜덤 정책 롤아웃을 mp4로 렌더링한다
  (baseline 비교용).
- 디스플레이가 없는(headless) 환경에서는 소프트웨어 렌더러가 필요하므로 기본적으로
  `MUJOCO_GL=osmesa`를 쓴다. GPU + EGL이 있는 환경이면
  `MUJOCO_GL=egl python scripts/render_video.py ...`로 바꿔서 더 빠르게 렌더링할 수 있다.
- **중요한 주의사항**: OSMesa 렌더러를 만든 *뒤에* `torch`(stable-baselines3가 의존)를
  import하면 세그폴트가 난다 (OSMesa와 torch의 OpenMP 스레드 초기화 충돌로 추정).
  그래서 이 스크립트는 항상 정책(또는 torch)을 먼저 로드하고 나서 `mujoco.Renderer`를
  생성한다 — 이 스크립트를 참고해서 직접 렌더링 코드를 짤 때도 같은 순서를 지킬 것.

```bash
# osmesa 소프트웨어 렌더러에 필요한 시스템 라이브러리 (Colab/Ubuntu 기준)
!apt-get -qq install -y libosmesa6 libgl1

MUJOCO_GL=osmesa python scripts/render_video.py \
    --model-path ./checkpoints/ppo_peg_in_hole_final.zip \
    --out-path ./videos/success.mp4 --seed 42
```

## Colab 실행 순서

```bash
# 1. 패키지 설치
!pip install -q -r requirements.txt

# 2. Google Drive 마운트 (체크포인트를 세션 끊김에도 보존하기 위함)
from google.colab import drive
drive.mount('/content/drive')

# 3. 학습 (Colab 무료 등급은 보통 CPU 코어 2개 → n-envs는 기본값(min(4, cpu_count))이
#    알아서 낮게 잡히지만, 명시적으로 지정해도 됨)
!python scripts/train_ppo.py \
    --n-envs 2 \
    --total-timesteps 2000000 \
    --save-dir /content/drive/MyDrive/rldc-lite/checkpoints

# 4. 세션이 끊겼다면 마지막 체크포인트에서 재개
!python scripts/train_ppo.py \
    --n-envs 2 \
    --total-timesteps 2000000 \
    --save-dir /content/drive/MyDrive/rldc-lite/checkpoints \
    --resume /content/drive/MyDrive/rldc-lite/checkpoints/ppo_peg_in_hole_1000000_steps.zip

# 5. 학습된 정책으로 데이터 수집
!python scripts/collect_data.py \
    --model-path /content/drive/MyDrive/rldc-lite/checkpoints/ppo_peg_in_hole_final.zip \
    --n-episodes 500 \
    --out-dir /content/drive/MyDrive/rldc-lite/data/peg_in_hole

# 6. (선택) 학습된 정책 롤아웃을 영상으로 확인
!apt-get -qq install -y libosmesa6 libgl1
!MUJOCO_GL=osmesa python scripts/render_video.py \
    --model-path /content/drive/MyDrive/rldc-lite/checkpoints/ppo_peg_in_hole_final.zip \
    --out-path /content/drive/MyDrive/rldc-lite/videos/success.mp4
```

## 검증 (직접 실행 결과)

env 로드/스텝, obs shape, PPO 학습/체크포인트/재개, 데이터 수집까지 실제로
python으로 실행해서 확인했다.

1. **환경 로드 & 랜덤 정책 50스텝**: obs shape `(19,)`, action shape `(4,)`,
   NaN/Inf 없음, 매 스텝 정상적으로 reward/done 계산됨.
2. **스크립트(P-제어) 정책으로 실제 삽입 성공 확인**: xy 오차를 비례 제어로
   보정하며 z를 계속 내리는 간단한 정책으로 57스텝 만에 삽입 성공
   (`insertion_depth=0.0285m ≥ target 0.025m`, `success=True`, `terminated=True`).
   이 과정에서 접촉력이 정지 시 기준치(peg 자중 0.49N) 대비 최대 132.8N까지
   스파이크되는 것을 확인 → F/T 센서가 실제 접촉을 제대로 반영함.
3. **PPO 학습(SubprocVecEnv, n_envs=2) 스모크 테스트**: 4096 스텝 학습 →
   체크포인트 파일 생성 확인 → `--resume`으로 이어서 2048 스텝 추가 학습 →
   `total_timesteps` 카운터가 2048에서 4096으로 정상 이어짐.
4. **collect_data.py 스모크 테스트**: 학습 초기 정책(성공 못 함)으로
   `--keep-failures` 사용 시 5개 에피소드 모두 저장(`ee_poses (100,3)`,
   `actions (100,4)`, `forces (100,3)`, `torques (100,3)`, `rewards (100,)`,
   `success ()` 확인), `--keep-failures` 없이는 실패 에피소드가 저장되지
   않음을 확인.
5. **실제 PPO 학습 200,000 스텝 (n_envs=4, CPU 4코어)**: `ep_rew_mean`이
   -155 → +220으로, `ep_len_mean`이 200(타임아웃)에서 ~100~110스텝으로
   개선됨(삽입 성공 시 조기 종료되므로 짧아지는 것은 좋은 신호). 학습 후
   별도 시드로 50 에피소드 평가 시 **성공률 43/50 (86%)**, 성공 에피소드는
   평균 약 45~50스텝 만에 삽입 완료.
6. **render_video.py로 실제 영상 렌더링 확인**: 학습된 정책은 42스텝 만에
   삽입 성공(`insertion_depth=0.0250~0.0285m`), 랜덤 정책은 150스텝 내내
   삽입 깊이 0으로 실패 — 두 mp4를 육안으로 비교해 정책이 실제로 정렬 후
   삽입 동작을 학습했음을 확인.

## 지금 임시로 되어있는 부분 (TODO)

- **팔 모델이 실제 로봇이 아님**: `assets/peg_in_hole.xml`의 4-DoF
  slide+hinge 팔은 실제 자작 로봇의 기구학을 전혀 반영하지 않은 placeholder다.
  나중에 실제 로봇의 URDF(또는 MJCF로 변환한 버전)로 교체해야 하며, 그에 맞춰
  observation의 `ee_pos`/`wrist` 정의와 action의 delta 스케일도 다시 잡아야 한다.
- **hole clearance(3mm)는 임의값**: 실제 태스크의 공차와 무관하게 "접촉이
  발생할 만큼 타이트한" 정도로 임의로 정한 수치다. 실제 하드웨어의 peg/hole
  공차에 맞게 다시 설정해야 한다.
- **actuator 게인(kp)·질량·마찰 계수도 대략적인 값**: 물리적으로 캘리브레이션된
  값이 아니라 시뮬레이션이 안정적으로 도는 선에서 대충 잡은 값이다.
- **gravcomp로 팔 자체 중력을 상쇄**: 제어를 단순화하기 위한 트릭이며, 실제
  로봇에는 이런 보상이 없을 수 있다(로봇 자체 컨트롤러가 이미 중력보상을
  하는 경우도 많으므로, 실제 하드웨어 교체 시 이 가정이 맞는지 확인 필요).
- **리워드 가중치(거리/삽입/힘/스텝 페널티 계수)는 미세 튜닝되지 않음**:
  스크립트 정책으로 태스크가 "풀 수 있다"는 것만 확인했고, PPO로 안정적으로
  수렴하는지에 대한 하이퍼파라미터 튜닝은 하지 않았다.
- **비전 관측 없음**: force/torque와 기구학적 위치만 사용하므로, 카메라 기반
  도메인 무작위화(조명, 텍스처 등)는 대상이 아니다.
