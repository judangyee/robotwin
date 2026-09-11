# ard-gen

MuJoCo 기반 peg-in-hole 태스크에서 admittance controller 게인을 CMA-ES로 찾는
파이프라인. **ARD-Gen 파이프라인의 0단계(Seed 확보)**에 해당한다 — 여기서 찾은
"성공하는 게인 + 그 궤적"(`seed_trajectory.npz`)이 다음 단계(2단계, seed를 이용한
부트스트래핑)의 입력이 된다.

팔은 실제 **ALOHA 팔로워 팔인 Trossen ViperX 300 6DOF(VX300s)** 를 그대로 쓴다
(mujoco_menagerie의 공식 MJCF, 메쉬 포함). 처음엔 단순화된 3-DoF 슬라이드 +
1-hinge 플레이스홀더 팔로 만들었다가, "이거 무슨 로봇이냐"는 질문을 받고
실제 로봇으로 교체했다 — 그 과정에서 진짜 로봇 팔(회전 조인트 6개, 관성/게인/
마찰이 실측값)을 쓸 때만 드러나는 문제 3가지를 실제로 겪고 고쳤다. 아래
"팔을 ALOHA(VX300s)로 교체하며 겪은 것들" 섹션 참고.

## 디렉토리 구조

```
ard-gen/
├── assets/
│   ├── peg_in_hole.xml          # MuJoCo 모델 (VX300s + hole + peg)
│   └── vx300s/                  # mujoco_menagerie의 VX300s 메쉬/라이선스
├── sim/peg_in_hole_sim.py       # 시뮬레이션 실행 + admittance controller
├── optimize/cma_search.py       # CMA-ES 최적화 스크립트
├── render_result.py             # seed_trajectory.npz의 게인으로 롤아웃을 mp4로 렌더링
├── requirements.txt
└── README.md
```

## 모델 (assets/peg_in_hole.xml, assets/vx300s/)

- **팔**: 실제 ALOHA 팔로워 팔(VX300s, 6개 회전 조인트: waist, shoulder, elbow,
  forearm_roll, wrist_angle, wrist_rotate + 손가락 2개). 메쉬/관성/조인트
  범위/게인은 [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  의 `trossen_vx300s` 그대로이며, `assets/vx300s/LICENSE`(BSD-3-Clause,
  Trossen Robotics)를 함께 포함한다.
- **홈 자세**: `waist=0, shoulder=1.0513, elbow=-1.3961, forearm_roll=0,
  wrist_angle=1.9154, wrist_rotate=0` — 그리퍼(peg 방향)가 정확히 수직 아래를
  향하도록(world (0,0,-1)과 오차 0.0003 이내) 그리드 서치로 찾았다.
- **peg**: `gripper_link`의 자식이지만 자체 조인트가 없음 → 그리퍼에 강체로
  고정된 것으로 취급(grasp 자체는 다루지 않음). `pinch` site(VX300s 기본
  제공 TCP)보다 0.07m 더 뻗어나가며, **tip이 구형**(반지름 = peg half-width
  10mm) — 이유는 아래 "검증하며 알아낸 것들" 참고.
- **hole**: 4개 벽 세그먼트 + 바닥으로 이루어진 소켓, world (0.572, 0, 0.056)
  위치(홈 자세 peg tip 기준 호버 간격 3cm). nominal clearance 3mm(편측
  1.5mm)지만, `scene_config['clearance_m']`에 따라 `sim`이 런타임에 벽의
  `geom_pos`/`geom_size`를 직접 덮어써서 바꾼다.
- **F/T 센서**: `peg_tip_site`에 3축 force + 3축 torque.
- 목표 삽입 깊이는 0.04m.
- fingers(그리퍼 손가락)는 grasp를 다루지 않으므로 키프레임 기본값 근처로
  고정만 해둔다(peg는 gripper_link에 용접되듯 붙어 있어서 실제로 손가락이
  peg를 쥐는 물리는 없음).

## 시뮬레이션 + admittance controller (sim/peg_in_hole_sim.py)

```
F_error_xy = F_ext_xy - F_desired_xy        (F_desired_xy = 0, world-frame)
dF_error_xy/dt ≈ (F_error_xy - F_error_xy_prev) / dt
Δx_xy = -Kp_xy * F_error_xy - Kd_xy * dF_error_xy/dt   (world-frame)
```

VX300s는 6개 조인트가 서로 결합돼 있어서(단순 슬라이드처럼 x/y/z가 독립
조인트가 아님), "action의 xy = 조인트에 직접 더하기"가 안 통한다. 대신 매
스텝 peg tip의 위치 Jacobian(`mujoco.mj_jacSite`)을 구해서, world-frame 위치
델타 `(Δx, Δy, -Z_RATE)`를 만들어내는 조인트각 델타를 damped least-squares로
역산하는 **리졸브드-레이트(resolved-rate) 오퍼레이셔널 스페이스 제어**를
쓴다. wrist_angle/wrist_rotate/forearm_roll(peg 자세)은 능동 보정하지 않고
초기값을 유지한다.

- **함수 시그니처**: `run_episode(gains: dict, scene_config: dict) -> dict`
  - `gains`: `{"Kp_xy": float, "Kd_xy": float}`
  - `scene_config`: `hole_pos_xy`, `friction`, `clearance_m`,
    `peg_init_offset_xy`, `peg_init_wrist`, `target_insertion_depth`
  - 반환: `ee_poses (T+1,4)`(x,y,z,wrist_rotate), `actions (T,3)`(world-frame
    위치 델타), `forces/force_profile (T,3)`, `torques/torque_profile (T,3)`,
    `insertion_depth`, `success`, `reward`, `max_force`, `step_count`
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

## 검증하며 알아낸 것들 (단순 팔 버전에서, 실제로 돌려보고 고친 것)

1. **평평한 peg 바닥은 admittance control이 아예 안 됨**: peg 바닥과 벽 윗면이
   둘 다 수평면이면 접촉 normal이 항상 수직이라(기하학적으로 당연함)
   `F_ext_xy`가 **항상 정확히 0**이라 Kp_xy를 얼마로 줘도 결과가 완전히
   똑같았다. 실제 커넥터/핀에 흔한 라운드팁과 같은 이유로 peg tip을 구형으로
   바꿔서 오프셋 상태의 접촉에 xy 성분이 생기게 했다.
2. **부호가 반대였음**: `Δx = +Kp*F_error`로 처음 구현했더니 오히려 안
   좋아졌다. 부호를 뒤집어(`Δx = -Kp*F_error`) 다시 테스트하니 최대 접촉력이
   60.5N(게인 0) → 6.8N(좋은 게인)까지 줄고 안정적으로 성공했다.
3. **오프셋 난이도 구간 확인**: 2~6mm는 peg tip의 곡률만으로 게인 없이도
   자가정렬되고, 12mm 이상은 게인을 아무리 키워도 안 됨. "게인이 있어야만
   성공"하는 구간을 찾아 대표 시나리오로 썼다.

## 팔을 ALOHA(VX300s)로 교체하며 겪은 것들 (진짜 로봇 팔이라 새로 드러난 문제)

플레이스홀더 팔(단순 슬라이드)에서는 안 보이던 문제 3개가, 실제 6-DOF 팔의
실측 관성/게인/마찰을 쓰자마자 나타났다. 전부 실제로 시뮬레이션을 돌려보고
찾았다.

1. **F/T 센서가 peg의 로컬 프레임 기준**: 단순 팔은 우연히 peg 로컬 프레임이
   world와 정렬돼 있어서 문제가 안 됐는데, VX300s는 팔 자체가 기울어 있어서
   peg 로컬 x,y,z가 world x,y,z와 다르다 — 정지 상태에서도 로컬 xy에 중력
   성분이 섞여 나오는 것으로 확인했다. `get_force_torque()`에서
   `site_xmat`으로 world 프레임으로 회전시켜서 고쳤다. (admittance
   controller는 world-frame lateral(x,y) 힘을 가정하므로 이 변환이 없으면
   애초에 말이 안 됨.)
2. **gravcomp 없이는 팔이 처짐**: VX300s의 실제 게인(예: shoulder kp=76)은
   원래 keyframe 근처 자세에서는 안정적이지만, 우리가 쓰는(그리퍼가 수직
   아래를 향하는) 완전히 뻗은 자세에서는 순수 P 제어로 중력을 못 버텨서
   200스텝 만에 6.6cm나 처졌다(실측 확인). 이 프로젝트의 다른 모델들과
   같은 방식으로 팔 바디마다 `gravcomp="1"`을 줘서 해결했다.
3. **elbow의 정지마찰(frictionloss) 데드존**: admittance controller 한 스텝의
   목표 변화량이 아주 작은데(수 mm 스케일), elbow의 실제 정지마찰
   (frictionloss=1.74)보다 그 스텝의 유효 토크가 작아서 **조인트가 아예 안
   움직였다**(qpos가 1e-6 단위로만 변함). 컨트롤 목표를 "이전 목표 +
   델타"로 계속 누적하는 방식(실제 오퍼레이셔널 스페이스 컨트롤러가 하는
   방식)으로 두고, elbow/shoulder의 frictionloss만 0으로 뺐다 — 반대로
   "현재 실제 qpos + 델타"로 매번 다시 고정하는 방식도 시도해봤는데, 이건
   목표가 항상 조인트 바로 앞에만 있어서 정지마찰을 절대 못 넘고 아예
   정지해버렸다(둘 다 실측으로 비교해서 확인).

이 세 가지를 고친 뒤 다시 스크립트로 오프셋을 훑어서, "게인이 있어야만
성공하는" 구간을 다시 찾았다 — 팔이 바뀌면서 정확한 mm 값은 이전(단순 팔,
6~10mm)과 달라졌다(5.5~8mm). peg-hole 접촉 물리 자체(구형 tip, clearance)는
그대로지만 팔의 실제 조인트 특성이 다르므로 재보정이 필요했다.

## CMA-ES 최적화 (optimize/cma_search.py)

- `cma` 패키지로 `[Kp_xy, Kd_xy]` 2차원을 탐색한다. 스케일이 크게 다른 두
  파라미터라 `CMA_stds` 옵션으로 좌표별 초기 스텝 크기를 따로 준다.
- 매 세대, 실측으로 확인한 대표 시나리오 3개(오프셋 (6,6), (7.5,0),
  (8,-2)mm — "게인 0이면 실패, Kp_xy=0.008이면 성공"을 확인한 조합) 각각에
  대해 `run_episode()`를 돌려 리워드를 평균낸다.
- 종료 조건: 평균 리워드가 `--threshold`에 도달 **하거나** `--max-generations`
  (기본 100)에 도달하면 멈춘다.
- 종료 시: 최종 게인으로 대표 시나리오(첫 번째)를 다시 실행해 그 궤적을
  `seed_trajectory.npz`로 저장하고, 세대별 최고 리워드로 수렴 곡선
  (`convergence.png`)을 그린다.

### seed_trajectory.npz 데이터 구조 (2단계 부트스트래핑 입력)

| 키 | shape | 설명 |
|---|---|---|
| `ee_poses` | `(T+1, 4)` | (x, y, z, wrist_rotate) 시퀀스, 초기 상태 포함 |
| `actions` | `(T, 3)` | 매 스텝 실제로 적용한 world-frame 위치 델타 (Δx, Δy, Δz) |
| `forces` | `(T, 3)` | F/T 센서 force (world frame) |
| `torques` | `(T, 3)` | F/T 센서 torque (world frame) |
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

# 결과 영상 확인 (헤드리스 환경은 MUJOCO_GL=osmesa 필요)
MUJOCO_GL=osmesa python render_result.py \
    --seed-path ./seed_trajectory.npz --out-path ./result.mp4
```

## 실제 실행 결과 (VX300s 팔, 위 버그 3개 수정 후)

- **수렴 여부**: 수렴함. 일부러 나쁜 초기 게인(`Kp_xy=0.00012`)에서 시작해서
  1세대 47.84 → 2세대 48.09 → 3세대 48.42 → 4세대 48.93 → 5세대 48.91 →
  **6세대 49.11**로 점진적으로 개선되며 threshold(49.0)에 도달해 조기 종료.
- **최종 게인**: `Kp_xy ≈ 0.00118`, `Kd_xy ≈ 2.8e-06`
- **성공 여부**: 최종 게인으로 메인 시나리오(오프셋 6mm/6mm) 재실행 결과
  `success=True`, `insertion_depth=0.0405m`(목표 0.04m 초과 달성),
  `max_force=6.6N`, `reward=49.46`
- `render_result.py`로 148스텝 만에 peg가 실제로 hole에 삽입되는 것을
  영상으로 확인함 — 실제 ALOHA 팔로워 팔(VX300s) 형상으로 렌더링됨.

## 지금 임시로 되어있는/한계인 부분

- **wrist(회전) misalignment는 보정하지 않음**: admittance controller가 xy
  위치만 보정하고 wrist_angle/wrist_rotate/forearm_roll은 초기값 유지다.
  peg 초기 자세가 이미 정렬돼 있다고 가정한 것 — 실제로는 회전 오차도
  힘/토크 피드백으로 보정해야 할 수 있다.
- **오프셋 5.5~8mm 범위에서만 검증됨**: 그 이상은 이 반응형(reactive) 제어
  방식으로는 원래 못 푸는 문제다(스파이럴 서치 등 별도 탐색 동작이 필요).
  이건 버그가 아니라 "순수 힘 피드백 admittance control"의 실제 한계다.
- **hole clearance/actuator 게인/질량은 대략적인 값**: hole 위치·clearance·
  peg 치수는 이 태스크가 풀리는 선에서 임의로 잡았다(팔 자체의 관성/게인은
  VX300s 실측값을 그대로 씀). gravcomp와 shoulder/elbow의 frictionloss 제거는
  admittance controller를 순수하게 테스트하기 위한 단순화이며, 실제 로봇에는
  둘 다 존재한다(실제 컨트롤러는 보통 별도의 중력보상/마찰보상 항으로 이를
  다룬다).
- **대표 시나리오 3개로만 평균냄**: 매 세대 무작위로 다시 뽑지 않고 고정된
  3개 시나리오를 쓴다. 더 견고한(generalize하는) 게인을 원하면
  `sim.sample_scene_config()`로 매 세대 새로 샘플링하도록 바꿀 수 있다(이미
  구현돼 있음, `cma_search.py`에서 아직 안 쓰고 있을 뿐).
