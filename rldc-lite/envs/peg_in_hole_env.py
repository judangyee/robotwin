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

# 목표 삽입 깊이 (hole opening 평면 기준, hole_socket 로컬 z=0)
_TARGET_INSERTION_DEPTH = 0.025  # m
_SUCCESS_TOLERANCE = 0.0  # depth >= target 이면 성공 (목표 자체가 이미 여유를 둔 값)

# 도메인 무작위화 범위
_HOLE_POS_JITTER_XY = 0.004   # m
_HOLE_POS_JITTER_Z = 0.002    # m
_INIT_XY_OFFSET = 0.008       # m, peg 초기 측면 오프셋 (clearance 1.5mm보다 훨씬 큼)
_INIT_Z_NOMINAL = -0.065      # carriage_z 초기값 (peg tip이 opening 위 ~2.5cm)
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
        self._wrist_joint_qposadr = self.model.joint("hinge_wrist").qposadr[0]

        self._joint_qposadr = {
            name: self.model.joint(name).qposadr[0]
            for name in ("slide_x", "slide_y", "slide_z", "hinge_wrist")
        }
        self._actuator_ids = {
            name: self.model.actuator(f"act_{name.split('_')[-1]}").id
            for name in ("slide_x", "slide_y", "slide_z", "hinge_wrist")
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
            init_wrist = self.np_random.uniform(-_INIT_WRIST_JITTER, _INIT_WRIST_JITTER)
        else:
            self.model.body_pos[self._hole_body_id] = self._nominal_hole_pos
            self.model.geom_friction[self._peg_geom_id] = self._nominal_friction
            init_x, init_y, init_z, init_wrist = 0.0, 0.0, _INIT_Z_NOMINAL, 0.0

        init_qpos = {
            "slide_x": init_x,
            "slide_y": init_y,
            "slide_z": init_z,
            "hinge_wrist": init_wrist,
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

        for i, name in enumerate(("slide_x", "slide_y", "slide_z", "hinge_wrist")):
            act_id = self._actuator_ids[name]
            lo, hi = self.model.actuator_ctrlrange[act_id]
            new_ctrl = np.clip(self.data.ctrl[act_id] + delta[i], lo, hi)
            self.data.ctrl[act_id] = new_ctrl

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
    def _get_obs(self) -> np.ndarray:
        ee_pos = self.data.xpos[self._wrist_body_id]
        wrist_angle = np.array([self.data.qpos[self._wrist_joint_qposadr]])
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
        insertion_depth = max(0.0, float(hole_center_pos[2] - peg_tip_pos[2]))
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
