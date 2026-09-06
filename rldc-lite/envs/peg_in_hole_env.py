"""단순화된 4-DoF 팔의 peg-in-hole 삽입 태스크를 위한 gymnasium 환경.

Task: 그리퍼에 강체로 고정된 사각 peg를, 4벽 소켓 형태의 hole에 삽입한다.
(grasp 자체는 다루지 않음 - peg는 항상 그리퍼에 고정되어 있다고 가정)
"""
from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "peg_in_hole.xml")

# 제어 스텝당 최대 이동량 (action=[-1,1] 정규화 값에 곱해지는 스케일)
_POS_STEP = 0.003  # m / control step
_ROT_STEP = 0.05   # rad / control step

# SCARA 2-link 기구학 상수. assets/peg_in_hole.xml에서 upper_arm/forearm 길이나
# arm_base(shoulder mount) 위치를 바꾸면 여기도 반드시 같이 고칠 것.
_L1 = 0.08  # upper_arm 길이 (m)
_L2 = 0.08  # forearm 길이 (m)
_SHOULDER_MOUNT_XY = np.array([-0.11, 0.0])  # world (x,y), arm_base 위치와 일치해야 함

# action의 xy 성분은 "shoulder 기준 상대좌표"로 유지되는 내부 목표점
# (_target_xy_world)에 누적된 뒤, 매 스텝 closed-form 2-link 역기구학(IK)으로
# (shoulder, elbow) 목표각으로 변환된다 — 실제 오퍼레이셔널 스페이스 컨트롤러와
# 같은 방식. 이 목표점은 특이점(팔이 완전히 펴지거나 완전히 접히는 지점) 근처로
# 가지 않도록 아래 범위로 매 스텝 클리핑한다 (RL 탐색 중 다이버전스 방지).
_WORKSPACE_X_RANGE = (0.07, 0.15)  # shoulder 기준 상대 x
_WORKSPACE_Y_RANGE = (-0.04, 0.04)  # shoulder 기준 상대 y

# peg의 절대 회전각(월드 기준 yaw)은 shoulder+elbow+hinge_wrist의 합이다 (세
# 조인트 모두 같은 z축 회전이라 누적됨). action의 wrist 성분은 "peg 절대 각도"
# 기준 목표(_target_wrist_absolute)에 누적되고, 매 스텝 그 값에서 현재
# shoulder+elbow 합을 뺀 값을 hinge_wrist ctrl로 내보낸다 — xy와 같은 방식의
# 오퍼레이셔널 스페이스 제어. 이렇게 해야 RL이 shoulder/elbow 각도를 몰라도
# (obs에 없음) peg의 실제 절대 방향을 일정하게 제어할 수 있다.
_TARGET_WRIST_RANGE = (-0.5, 0.5)  # rad, peg 절대 각도 목표의 클리핑 범위

# 목표 삽입 깊이 (hole opening 평면 기준, hole_socket 로컬 z=0)
_TARGET_INSERTION_DEPTH = 0.025  # m
_SUCCESS_TOLERANCE = 0.0  # depth >= target 이면 성공 (목표 자체가 이미 여유를 둔 값)

# hole 벽 구조물의 실제 바깥쪽 footprint half-width. assets/peg_in_hole.xml의
# hole_wall_* geom들의 size(outer_half=0.0195)와 반드시 일치해야 한다 — 이 값
# 바깥은 벽이 전혀 없는 완전한 허공이므로, insertion_depth를 xy 정렬 없이도
# 인정해버리는 리워드 해킹 구멍이 된다 (자세한 설명은 _get_info 참고).
_HOLE_OUTER_HALF_WIDTH = 0.0195  # m

# 도메인 무작위화 범위
_HOLE_POS_JITTER_XY = 0.004   # m
_HOLE_POS_JITTER_Z = 0.002    # m
_INIT_XY_OFFSET = 0.008       # m, peg 초기 측면 오프셋 (clearance 1.5mm보다 훨씬 큼)
_INIT_Z_NOMINAL = -0.065      # slide_z(z_carriage) 초기값 (peg tip이 opening 위 ~2.5cm)
_INIT_Z_JITTER = 0.01
_INIT_WRIST_JITTER = 0.15     # rad
_FRICTION_RANGE = (0.2, 0.8)  # peg geom sliding friction 무작위화 범위

# 리워드 가중치 (범용: 거리 페널티 + 삽입 보상 + 과다접촉력 페널티 + 스텝 페널티 + 성공 보너스)
_W_DIST = 20.0
_W_INSERT = 10.0
_W_FORCE = 0.02
_FORCE_SAFE_THRESHOLD = 15.0  # N, 이 이상 넘는 접촉력만 페널티
_W_STEP = 0.05
_SUCCESS_BONUS = 20.0


def _ik_2link(x: float, y: float) -> tuple[float, float]:
    """평면 2-link(shoulder-elbow) closed-form 역기구학.

    (x, y)는 shoulder 관절 기준 상대좌표. elbow-up 브랜치(theta2 > 0) 고정.
    도달 불가능한 거리(너무 멀거나 완전히 접힌 특이점 근처)는 방향은 유지한 채
    도달 가능한 최대/최소 거리로 투영해서 항상 유효한 각도를 반환한다.
    """
    d = float(np.hypot(x, y))
    d_safe = float(np.clip(d, abs(_L1 - _L2) + 1e-3, _L1 + _L2 - 1e-3))
    if d > 1e-9:
        scale = d_safe / d
        x, y = x * scale, y * scale
    else:
        x, y = d_safe, 0.0

    cos_t2 = (x * x + y * y - _L1 ** 2 - _L2 ** 2) / (2 * _L1 * _L2)
    cos_t2 = float(np.clip(cos_t2, -1.0, 1.0))
    t2 = float(np.arccos(cos_t2))
    t1 = float(np.arctan2(y, x) - np.arctan2(_L2 * np.sin(t2), _L1 + _L2 * np.cos(t2)))
    return t1, t2


class PegInHoleEnv(gym.Env):
    """단순화된 peg-in-hole 삽입 환경 (단일 팔, grasp 미포함)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_path: str | None = None,
        max_episode_steps: int = 200,
        n_substeps: int = 5,
        domain_randomization: bool = True,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.xml_path = xml_path or _DEFAULT_XML
        self.max_episode_steps = max_episode_steps
        self.n_substeps = n_substeps
        self.domain_randomization = domain_randomization
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # id 캐싱
        self._wrist_body_id = self.model.body("wrist").id
        self._hole_body_id = self.model.body("hole_socket").id
        self._hole_site_id = self.model.site("hole_center_site").id
        self._peg_tip_site_id = self.model.site("peg_tip_site").id
        self._peg_geom_id = self.model.geom("peg_geom").id

        self._joint_qposadr = {
            name: self.model.joint(name).qposadr[0]
            for name in ("shoulder", "elbow", "slide_z", "hinge_wrist")
        }
        self._actuator_ids = {
            "shoulder": self.model.actuator("act_shoulder").id,
            "elbow": self.model.actuator("act_elbow").id,
            "slide_z": self.model.actuator("act_z").id,
            "hinge_wrist": self.model.actuator("act_wrist").id,
        }

        self._nominal_hole_pos = self.model.body_pos[self._hole_body_id].copy()
        self._nominal_friction = self.model.geom_friction[self._peg_geom_id].copy()

        force_sensor_adr = self.model.sensor("peg_force").adr[0]
        torque_sensor_adr = self.model.sensor("peg_torque").adr[0]
        self._force_slice = slice(force_sensor_adr, force_sensor_adr + 3)
        self._torque_slice = slice(torque_sensor_adr, torque_sensor_adr + 3)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 관측 공간 shape은 실제 concat 결과 길이로부터 계산한다 (하드코딩 금지).
        mujoco.mj_forward(self.model, self.data)
        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self._np_random_seed = None
        self._target_xy_world = np.zeros(2)  # ee의 내부 목표 (x,y), reset()에서 초기화
        self._target_wrist_absolute = 0.0  # peg 절대 각도 목표, reset()에서 초기화

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.domain_randomization:
            hole_pos = self._nominal_hole_pos + np.array(
                [
                    self.np_random.uniform(-_HOLE_POS_JITTER_XY, _HOLE_POS_JITTER_XY),
                    self.np_random.uniform(-_HOLE_POS_JITTER_XY, _HOLE_POS_JITTER_XY),
                    self.np_random.uniform(-_HOLE_POS_JITTER_Z, _HOLE_POS_JITTER_Z),
                ]
            )
            self.model.body_pos[self._hole_body_id] = hole_pos

            friction = self._nominal_friction.copy()
            friction[0] = self.np_random.uniform(*_FRICTION_RANGE)
            self.model.geom_friction[self._peg_geom_id] = friction

            init_x = self.np_random.uniform(-_INIT_XY_OFFSET, _INIT_XY_OFFSET)
            init_y = self.np_random.uniform(-_INIT_XY_OFFSET, _INIT_XY_OFFSET)
            init_z = _INIT_Z_NOMINAL + self.np_random.uniform(-_INIT_Z_JITTER, _INIT_Z_JITTER)
            init_wrist_absolute = self.np_random.uniform(-_INIT_WRIST_JITTER, _INIT_WRIST_JITTER)
        else:
            hole_pos = self._nominal_hole_pos
            self.model.body_pos[self._hole_body_id] = hole_pos
            self.model.geom_friction[self._peg_geom_id] = self._nominal_friction
            init_x, init_y, init_z, init_wrist_absolute = 0.0, 0.0, _INIT_Z_NOMINAL, 0.0

        # xy는 hole의 world (x,y) + 무작위 오프셋을 목표점으로 잡고, shoulder 기준
        # 상대좌표로 변환해 IK로 (shoulder, elbow) 각도를 구한다.
        self._target_xy_world = hole_pos[:2] + np.array([init_x, init_y])
        shoulder_theta, elbow_theta = self._solve_ik(self._target_xy_world)

        # wrist는 "peg 절대 각도" 기준으로 목표를 잡고, shoulder+elbow가 이미
        # 기여한 회전을 상쇄하도록 hinge_wrist qpos를 역산한다.
        self._target_wrist_absolute = init_wrist_absolute
        wrist_theta = self._solve_wrist(shoulder_theta, elbow_theta)

        init_qpos = {
            "shoulder": shoulder_theta,
            "elbow": elbow_theta,
            "slide_z": init_z,
            "hinge_wrist": wrist_theta,
        }
        for name, value in init_qpos.items():
            self.data.qpos[self._joint_qposadr[name]] = value
            # 액추에이터 목표를 현재 qpos와 맞춰서, reset 직후 서보가 갑자기
            # 원점으로 되돌아가려 하지 않도록 한다.
            self.data.ctrl[self._actuator_ids[name]] = value

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        step_scale = np.array([_POS_STEP, _POS_STEP, _POS_STEP, _ROT_STEP])
        delta = action * step_scale

        # xy는 내부 목표점(_target_xy_world)에 델타를 누적한 뒤 IK로 (shoulder,
        # elbow) 각도를 구해서 액추에이터에 넣는다 (오퍼레이셔널 스페이스 제어).
        self._target_xy_world = self._target_xy_world + delta[:2]
        shoulder_theta, elbow_theta = self._solve_ik(self._target_xy_world)
        self.data.ctrl[self._actuator_ids["shoulder"]] = shoulder_theta
        self.data.ctrl[self._actuator_ids["elbow"]] = elbow_theta

        # z(삽입 깊이)는 이전과 동일하게 델타를 ctrl에 직접 누적.
        act_id = self._actuator_ids["slide_z"]
        lo, hi = self.model.actuator_ctrlrange[act_id]
        self.data.ctrl[act_id] = np.clip(self.data.ctrl[act_id] + delta[2], lo, hi)

        # wrist는 "peg 절대 각도" 목표(_target_wrist_absolute)에 델타를 누적한
        # 뒤, 지금 shoulder+elbow가 이미 기여한 회전을 상쇄하도록 hinge_wrist
        # ctrl을 역산한다 (xy와 같은 방식의 오퍼레이셔널 스페이스 제어).
        self._target_wrist_absolute = float(
            np.clip(self._target_wrist_absolute + delta[3], *_TARGET_WRIST_RANGE)
        )
        self.data.ctrl[self._actuator_ids["hinge_wrist"]] = self._solve_wrist(shoulder_theta, elbow_theta)

        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self._step_count += 1

        obs = self._get_obs()
        info = self._get_info()
        reward, success = self._compute_reward(info)
        info["success"] = success

        terminated = bool(success)
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------
    def _solve_ik(self, target_xy_world: np.ndarray) -> tuple[float, float]:
        """world (x,y) 목표점을 받아 workspace 범위로 클리핑하고 (shoulder, elbow)
        각도를 계산한다. `self._target_xy_world`도 클리핑된 값으로 갱신해서,
        RL 탐색 중 목표점이 도달 불가능한 먼 곳까지 드리프트하지 않게 한다."""
        rel = target_xy_world - _SHOULDER_MOUNT_XY
        rel_x = float(np.clip(rel[0], *_WORKSPACE_X_RANGE))
        rel_y = float(np.clip(rel[1], *_WORKSPACE_Y_RANGE))
        self._target_xy_world = np.array([rel_x, rel_y]) + _SHOULDER_MOUNT_XY

        shoulder_theta, elbow_theta = _ik_2link(rel_x, rel_y)
        s_lo, s_hi = self.model.actuator_ctrlrange[self._actuator_ids["shoulder"]]
        e_lo, e_hi = self.model.actuator_ctrlrange[self._actuator_ids["elbow"]]
        shoulder_theta = float(np.clip(shoulder_theta, s_lo, s_hi))
        elbow_theta = float(np.clip(elbow_theta, e_lo, e_hi))
        return shoulder_theta, elbow_theta

    def _solve_wrist(self, shoulder_theta: float, elbow_theta: float) -> float:
        """peg 절대 각도 목표(_target_wrist_absolute)에서 shoulder+elbow가 이미
        기여한 회전을 뺀 값을 hinge_wrist ctrl로 반환한다 (다시 강조: shoulder,
        elbow, hinge_wrist 모두 같은 z축 회전이라 절대 각도는 셋의 합이다)."""
        wrist_theta = self._target_wrist_absolute - shoulder_theta - elbow_theta
        lo, hi = self.model.actuator_ctrlrange[self._actuator_ids["hinge_wrist"]]
        return float(np.clip(wrist_theta, lo, hi))

    def _get_obs(self) -> np.ndarray:
        ee_pos = self.data.xpos[self._wrist_body_id]
        # peg의 절대 회전각(월드 기준 yaw). hinge_wrist qpos만으로는 알 수 없다 —
        # shoulder+elbow+hinge_wrist 세 z축 회전 조인트가 누적된 값이 실제 peg의
        # 절대 방향이므로, wrist 바디의 실제 회전행렬에서 직접 뽑아낸다.
        wrist_xmat = self.data.xmat[self._wrist_body_id]
        wrist_angle = np.array([np.arctan2(wrist_xmat[3], wrist_xmat[0])])
        peg_tip_pos = self.data.site_xpos[self._peg_tip_site_id]
        hole_center_pos = self.data.site_xpos[self._hole_site_id]
        relative_vec = hole_center_pos - peg_tip_pos
        force = self.data.sensordata[self._force_slice]
        torque = self.data.sensordata[self._torque_slice]

        obs = np.concatenate(
            [ee_pos, wrist_angle, peg_tip_pos, hole_center_pos, relative_vec, force, torque]
        )
        return obs.astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        peg_tip_pos = self.data.site_xpos[self._peg_tip_site_id]
        hole_center_pos = self.data.site_xpos[self._hole_site_id]
        relative_vec = hole_center_pos - peg_tip_pos
        # 삽입 깊이: peg tip이 hole opening 평면(hole_center z)보다 얼마나 아래 있는지.
        # 주의: z 깊이만 보면 xy 정렬 없이 hole 벽 구조물 바깥(허공)으로 그냥
        # z를 내리꽂아도 "삽입"으로 잘못 인정되는 리워드 해킹이 가능하다 (실제로
        # PPO가 이 구멍을 찾아냈다). 그래서 peg tip이 hole 벽 구조물의 실제
        # 바깥쪽 footprint(_HOLE_OUTER_HALF_WIDTH, assets/peg_in_hole.xml의 벽
        # geom size와 일치해야 함) 안에 있을 때만 depth를 인정한다. 이 footprint
        # 안쪽(hole 내부거나 벽 위)에서는 실제 벽 충돌 물리가 이미 잘못된 진입을
        # 막아주므로, 이 게이트는 "벽 구조물이 아예 없는 완전한 허공"만 걸러낸다.
        dx = float(hole_center_pos[0] - peg_tip_pos[0])
        dy = float(hole_center_pos[1] - peg_tip_pos[1])
        xy_within_hole_footprint = abs(dx) < _HOLE_OUTER_HALF_WIDTH and abs(dy) < _HOLE_OUTER_HALF_WIDTH
        raw_depth = max(0.0, float(hole_center_pos[2] - peg_tip_pos[2]))
        insertion_depth = raw_depth if xy_within_hole_footprint else 0.0
        force = self.data.sensordata[self._force_slice].copy()
        torque = self.data.sensordata[self._torque_slice].copy()
        return {
            "distance": float(np.linalg.norm(relative_vec)),
            "insertion_depth": insertion_depth,
            "force": force,
            "torque": torque,
            "force_norm": float(np.linalg.norm(force)),
        }

    def _compute_reward(self, info: dict[str, Any]) -> tuple[float, bool]:
        distance_penalty = -_W_DIST * info["distance"]
        insertion_progress = min(1.0, info["insertion_depth"] / _TARGET_INSERTION_DEPTH)
        insertion_reward = _W_INSERT * insertion_progress
        excess_force = max(0.0, info["force_norm"] - _FORCE_SAFE_THRESHOLD)
        force_penalty = -_W_FORCE * excess_force
        step_penalty = -_W_STEP

        success = info["insertion_depth"] >= _TARGET_INSERTION_DEPTH - _SUCCESS_TOLERANCE
        reward = distance_penalty + insertion_reward + force_penalty + step_penalty
        if success:
            reward += _SUCCESS_BONUS
        return float(reward), success
