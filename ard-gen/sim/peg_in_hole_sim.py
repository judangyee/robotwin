"""ARD-Gen 0단계(Seed 확보)용 peg-in-hole 시뮬레이션 + admittance controller.

이 버전은 실제 ALOHA 팔로워 팔인 Trossen ViperX 300 6DOF(VX300s)를 그대로
쓴다(assets/vx300s/). 이전 버전(단순 슬라이드 3개+wrist 1개)과 달리 6개
회전 조인트가 서로 결합돼 있어서, "action의 xy 성분 = 슬라이드 조인트에 직접
더하기"가 안 통한다 — peg tip의 위치 Jacobian을 매 스텝 구해서 damped
least-squares로 조인트 각도 델타를 역산하는 리졸브드-레이트(resolved-rate)
방식의 오퍼레이셔널 스페이스 제어를 쓴다.

admittance control 수식 (force-error에 대한 PD, xy는 이전 버전과 동일):
    F_error_xy = F_ext_xy - F_desired_xy         (F_desired_xy = 0)
    dF_error_xy/dt ≈ (F_error_xy - F_error_xy_prev) / dt
    Δx_xy = -Kp_xy * F_error_xy - Kd_xy * dF_error_xy/dt

    부호(-)와 peg tip을 구형으로 만든 이유는 이전(단순 팔) 버전에서 실제로
    돌려보고 검증한 그대로다 — README 검증 섹션 참고. 팔 기구학만 바뀌었지
    peg-hole 접촉 물리/컨트롤 법칙 자체는 동일해서 그 결론이 그대로 적용된다.

    삽입 방향(z축, world -z)은 게인과 무관하게 매 스텝 일정 속도(Z_RATE)로
    내려간다. wrist_rotate/forearm_roll(peg 자세)는 능동 보정하지 않고
    초기값을 유지한다.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "peg_in_hole.xml")

N_SUBSTEPS = 5  # mj_step 호출당 substep 수 (timestep=0.002 -> 제어 주기 dt=0.01s)
DT = N_SUBSTEPS * 0.002

Z_RATE = 0.0006  # m / control step, 삽입 방향 일정 속도
MAX_STEPS = 400

TARGET_INSERTION_DEPTH = 0.04  # m (scene_config로 덮어쓸 수 있음)

# hole 벽 nominal 치수
_PEG_HALF_WIDTH = 0.010
_WALL_HALF_THICKNESS = 0.004
_WALL_HALF_HEIGHT = 0.0275
_WALL_CENTER_Z = -0.0275
_FLOOR_HALF_THICKNESS = 0.002
_NOMINAL_CLEARANCE_M = 0.003  # 총 지름 clearance (편측 1.5mm)

# VX300s 6개 팔 조인트 (fingers 제외). Jacobian 열 순서/ctrl 매핑에 이 순서를 쓴다.
_ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

# "홈" 자세: 그리퍼(peg 방향)가 정확히 수직 아래를 향하도록 그리드 서치로 찾은
# 값 (오차 0.0003 이내로 world (0,0,-1)과 일치). assets/peg_in_hole.xml의
# hole_socket 위치가 이 자세의 peg tip 바로 아래(호버 간격 3cm)에 오도록
# 맞춰져 있다 — 팔 mount 위치나 hole 위치를 바꾸면 이 값도 다시 찾아야 한다.
_HOME_QPOS = {
    "waist": 0.0,
    "shoulder": 1.05128,
    "elbow": -1.39615,
    "forearm_roll": 0.0,
    "wrist_angle": 1.91538,
    "wrist_rotate": 0.0,
}

_JAC_DAMPING = 1e-4  # damped least-squares 정규화 계수
_IK_MAX_ITERS = 200  # reset() 시 초기 자세를 구하는 반복 IK의 최대 반복 횟수
_IK_STEP_SCALE = 0.5  # 반복 IK 한 스텝당 오차의 몇 %를 보정할지

_GRIPPER_CLOSED_CTRL = 0.024  # 손가락은 grasp를 다루지 않으므로 고정값 (VX300s 기본 키프레임 값)


def _default_scene_config() -> dict[str, Any]:
    return {
        "hole_pos_xy": (0.0, 0.0),
        "friction": 0.5,
        "clearance_m": _NOMINAL_CLEARANCE_M,
        "peg_init_offset_xy": (0.0, 0.0),
        "peg_init_wrist": 0.0,
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


def sample_scene_config(rng: np.random.Generator) -> dict[str, Any]:
    """CMA-ES 평가/견고성 테스트용 무작위 scene_config 샘플러.

    peg_init_offset_xy 범위(5.5~8mm)는 VX300s 팔로 바꾼 뒤 다시 실측해서 정한
    값이다. 팔을 단순 슬라이드에서 VX300s(Jacobian 기반 오퍼레이셔널 스페이스
    제어)로 바꾸면서 "게인이 있어야만 성공하는" 오프셋 구간이 이전(6~10mm)과
    조금 달라졌다 — peg-hole 접촉 물리(구형 tip 등)는 그대로지만, 팔의 실제
    조인트 게인/역기구학 특성이 달라지면서 정확히 같은 수치가 재현되지는
    않았다(README 검증 섹션 참고)."""
    return {
        "hole_pos_xy": tuple(rng.uniform(-0.004, 0.004, size=2)),
        "friction": float(rng.uniform(0.2, 0.8)),
        "clearance_m": float(rng.uniform(0.0025, 0.0035)),
        "peg_init_offset_xy": tuple(
            rng.choice([-1.0, 1.0], size=2) * rng.uniform(0.0055, 0.008, size=2)
        ),
        "peg_init_wrist": 0.0,
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


class PegInHoleSim:
    """MjModel/MjData를 재사용하는 시뮬레이션 래퍼. run_episode()가 매번 이걸
    새로 만들지 않도록 CMA-ES 쪽에서 인스턴스 하나를 재사용하면 훨씬 빠르다."""

    def __init__(self, xml_path: str | None = None):
        self.xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self._hole_body_id = self.model.body("hole_socket").id
        self._hole_site_id = self.model.site("hole_center_site").id
        self._peg_tip_site_id = self.model.site("peg_tip_site").id
        self._peg_geom_ids = [self.model.geom("peg_shaft").id, self.model.geom("peg_tip_ball").id]

        self._wall_geom_ids = {
            name: self.model.geom(f"hole_wall_{name}").id for name in ("px", "nx", "py", "ny")
        }
        self._floor_geom_id = self.model.geom("hole_floor").id

        self._arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _ARM_JOINTS}
        self._arm_dofadr = {name: self.model.joint(name).dofadr[0] for name in _ARM_JOINTS}
        self._arm_actuator_ids = {name: self.model.actuator(name).id for name in _ARM_JOINTS}
        self._gripper_actuator_id = self.model.actuator("gripper").id
        self._left_finger_qposadr = self.model.joint("left_finger").qposadr[0]

        force_adr = self.model.sensor("peg_force").adr[0]
        torque_adr = self.model.sensor("peg_torque").adr[0]
        self._force_slice = slice(force_adr, force_adr + 3)
        self._torque_slice = slice(torque_adr, torque_adr + 3)

        self._nominal_hole_pos = self.model.body_pos[self._hole_body_id].copy()

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    # ------------------------------------------------------------------
    def _apply_clearance(self, clearance_m: float) -> float:
        """hole 벽 4개의 위치/크기를 원하는 clearance(총 지름, m)에 맞게
        런타임으로 덮어쓴다. 반환값은 실제 적용된 hole_outer_half_width."""
        inner_half = _PEG_HALF_WIDTH + clearance_m / 2.0
        outer_half = inner_half + _WALL_HALF_THICKNESS * 2.0

        wall_center = inner_half + _WALL_HALF_THICKNESS
        self.model.geom_pos[self._wall_geom_ids["px"]] = [wall_center, 0, _WALL_CENTER_Z]
        self.model.geom_pos[self._wall_geom_ids["nx"]] = [-wall_center, 0, _WALL_CENTER_Z]
        self.model.geom_pos[self._wall_geom_ids["py"]] = [0, wall_center, _WALL_CENTER_Z]
        self.model.geom_pos[self._wall_geom_ids["ny"]] = [0, -wall_center, _WALL_CENTER_Z]

        self.model.geom_size[self._wall_geom_ids["px"]] = [_WALL_HALF_THICKNESS, outer_half, _WALL_HALF_HEIGHT]
        self.model.geom_size[self._wall_geom_ids["nx"]] = [_WALL_HALF_THICKNESS, outer_half, _WALL_HALF_HEIGHT]
        self.model.geom_size[self._wall_geom_ids["py"]] = [outer_half, _WALL_HALF_THICKNESS, _WALL_HALF_HEIGHT]
        self.model.geom_size[self._wall_geom_ids["ny"]] = [outer_half, _WALL_HALF_THICKNESS, _WALL_HALF_HEIGHT]

        self.model.geom_size[self._floor_geom_id] = [inner_half, inner_half, _FLOOR_HALF_THICKNESS]
        return outer_half

    def _jac_solve(self, delta_pos_world: np.ndarray) -> np.ndarray:
        """peg tip에 대한 3xnv 위치 Jacobian을 구해서, world-frame 위치 델타를
        만들어내는 조인트각 델타(길이 nv)를 damped least-squares로 푼다.
        finger 조인트는 peg tip 위치에 영향이 없어(peg는 gripper_link 자식이라
        finger 서브트리와 무관) 자동으로 ~0이 나온다."""
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self._peg_tip_site_id)
        jjt = self._jacp @ self._jacp.T + _JAC_DAMPING * np.eye(3)
        return self._jacp.T @ np.linalg.solve(jjt, delta_pos_world)

    def _solve_initial_pose(self, target_pos_world: np.ndarray, target_wrist: float) -> None:
        """HOME_QPOS에서 시작해 반복적으로 Jacobian을 이용해 target_pos_world에
        수렴하는 팔 조인트각을 구하고, qpos/ctrl에 반영한다 (물리 시뮬레이션이
        아니라 순수 기구학 반복 계산)."""
        for name, value in _HOME_QPOS.items():
            self.data.qpos[self._arm_qposadr[name]] = value
        self.data.qpos[self._arm_qposadr["wrist_rotate"]] += target_wrist
        mujoco.mj_forward(self.model, self.data)

        for _ in range(_IK_MAX_ITERS):
            current = self.data.site_xpos[self._peg_tip_site_id]
            err = target_pos_world - current
            if np.linalg.norm(err) < 1e-5:
                break
            dq = self._jac_solve(err * _IK_STEP_SCALE)
            for name in _ARM_JOINTS:
                dof = self._arm_dofadr[name]
                qadr = self._arm_qposadr[name]
                lo, hi = self.model.jnt_range[self.model.joint(name).id]
                self.data.qpos[qadr] = np.clip(self.data.qpos[qadr] + dq[dof], lo, hi)
            mujoco.mj_forward(self.model, self.data)

        for name in _ARM_JOINTS:
            self.data.ctrl[self._arm_actuator_ids[name]] = self.data.qpos[self._arm_qposadr[name]]

    def reset(self, scene_config: dict[str, Any]) -> float:
        """scene_config를 적용하고 초기 상태로 되돌린다. hole의 실제 xy
        footprint half-width(outer_half)를 반환한다 (reward 게이팅에 사용)."""
        mujoco.mj_resetData(self.model, self.data)

        hole_pos = self._nominal_hole_pos.copy()
        hole_pos[0] += scene_config["hole_pos_xy"][0]
        hole_pos[1] += scene_config["hole_pos_xy"][1]
        self.model.body_pos[self._hole_body_id] = hole_pos

        for geom_id in self._peg_geom_ids:
            self.model.geom_friction[geom_id][0] = scene_config["friction"]

        outer_half = self._apply_clearance(scene_config["clearance_m"])

        # 목표 초기 peg tip 위치: hole 중심 + xy 오프셋, z는 "홈" 자세의 호버
        # 높이를 그대로 쓴다 (assets/peg_in_hole.xml 설계상 hole opening 위
        # 약 3cm).
        home_qpos_arr = np.array([_HOME_QPOS[n] for n in _ARM_JOINTS])
        for name, value in _HOME_QPOS.items():
            self.data.qpos[self._arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, self.data)
        home_tip_z = self.data.site_xpos[self._peg_tip_site_id][2]

        target_pos = np.array(
            [
                hole_pos[0] + scene_config["peg_init_offset_xy"][0],
                hole_pos[1] + scene_config["peg_init_offset_xy"][1],
                home_tip_z,
            ]
        )
        self._solve_initial_pose(target_pos, scene_config["peg_init_wrist"])

        self.data.qpos[self._left_finger_qposadr] = _GRIPPER_CLOSED_CTRL
        self.data.ctrl[self._gripper_actuator_id] = _GRIPPER_CLOSED_CTRL

        mujoco.mj_forward(self.model, self.data)
        return outer_half

    def get_force_torque(self) -> tuple[np.ndarray, np.ndarray]:
        """F/T 센서는 peg_tip_site의 로컬 프레임 기준으로 읽힌다. 단순
        슬라이드 팔 버전에서는 그 로컬 프레임이 우연히 world와 정렬돼 있었지만,
        VX300s는 팔 자체가 회전돼 있어서 peg의 로컬 x,y,z가 world x,y,z와
        다르다(정지 상태에서도 로컬 xy에 중력 성분이 섞여 나오는 것으로 실제
        확인함). admittance controller는 world-frame lateral(x,y) 힘을
        가정하므로, site_xmat으로 world 프레임으로 회전시켜서 반환한다."""
        site_rot = self.data.site_xmat[self._peg_tip_site_id].reshape(3, 3)
        force_local = self.data.sensordata[self._force_slice]
        torque_local = self.data.sensordata[self._torque_slice]
        return site_rot @ force_local, site_rot @ torque_local

    def get_ee_pose(self) -> np.ndarray:
        """(x, y, z, wrist_rotate) — peg tip world position + wrist_rotate
        조인트값(이 단계에서는 능동 보정하지 않으므로 초기값에서 고정)."""
        pos = self.data.site_xpos[self._peg_tip_site_id]
        wrist = self.data.qpos[self._arm_qposadr["wrist_rotate"]]
        return np.array([pos[0], pos[1], pos[2], wrist])

    def get_peg_tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._peg_tip_site_id].copy()

    def get_hole_center_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._hole_site_id].copy()

    def step(self, delta_pos_world: np.ndarray) -> None:
        """delta_pos_world = (dx, dy, dz) world-frame 위치 델타를 Jacobian
        기반 리졸브드-레이트 제어로 팔 6개 조인트 ctrl에 반영하고, n_substeps
        만큼 물리를 진행한다. ctrl은 이전 ctrl에 dq를 누적한다 (실제 오퍼레이셔널
        스페이스 컨트롤러처럼 이동 목표를 계속 앞으로 밀어준다 — "qpos + dq"로
        매번 현재 위치에 재고정하는 방식은 elbow의 정지 마찰력(frictionloss)
        보다 한 스텝의 목표 변화량이 작아서 조인트가 아예 안 움직이는 문제가
        있었다 (아래 elbow frictionloss=0 처리 참고)."""
        dq = self._jac_solve(delta_pos_world)
        for name in _ARM_JOINTS:
            act_id = self._arm_actuator_ids[name]
            dof = self._arm_dofadr[name]
            lo, hi = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = np.clip(self.data.ctrl[act_id] + dq[dof], lo, hi)
        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)


def run_episode(gains: dict[str, float], scene_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """admittance controller로 한 에피소드를 실행한다.

    Args:
        gains: {"Kp_xy": float, "Kd_xy": float}
        scene_config: hole 위치/마찰/clearance/peg 초기 오프셋. 없으면 기본값.

    Returns:
        dict with: trajectory(ee_poses, (T+1,4)), actions((T,3), world-frame
        위치 델타), forces((T,3)), torques((T,3)), insertion_depth(float),
        success(bool), reward(float), step_count(int), max_force(float).
    """
    cfg = _default_scene_config()
    if scene_config:
        cfg.update(scene_config)

    sim = PegInHoleSim()
    return _run_episode_with_sim(sim, gains, cfg)


def _run_episode_with_sim(
    sim: PegInHoleSim, gains: dict[str, float], cfg: dict[str, Any]
) -> dict[str, Any]:
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    kp_xy = float(gains["Kp_xy"])
    kd_xy = float(gains["Kd_xy"])

    ee_poses = [sim.get_ee_pose()]
    actions = []
    forces = []
    torques = []

    prev_force_error_xy = np.zeros(2)
    max_force_mag = 0.0
    insertion_depth = 0.0
    success = False
    step_count = 0

    for step_count in range(1, MAX_STEPS + 1):
        force, torque = sim.get_force_torque()
        force_mag = float(np.linalg.norm(force))
        max_force_mag = max(max_force_mag, force_mag)

        force_error_xy = force[:2]  # F_desired_xy = 0 (정지 시 xy 힘은 항상 0)
        d_force_error_xy = (force_error_xy - prev_force_error_xy) / DT
        prev_force_error_xy = force_error_xy

        # 부호: -Kp*F_error (이전(단순 팔) 버전에서 실제로 돌려보고 검증된 방향)
        delta_xy = -kp_xy * force_error_xy - kd_xy * d_force_error_xy
        delta = np.array([delta_xy[0], delta_xy[1], -Z_RATE])

        sim.step(delta)

        actions.append(delta.copy())
        forces.append(force)
        torques.append(torque)
        ee_poses.append(sim.get_ee_pose())

        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_within_hole_footprint = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = max(0.0, float(hole_center[2] - peg_tip[2]))
        insertion_depth = raw_depth if xy_within_hole_footprint else 0.0

        if insertion_depth >= target_depth:
            success = True
            break

    final_peg_tip = sim.get_peg_tip_pos()
    final_hole_center = sim.get_hole_center_pos()
    dist = float(np.linalg.norm(final_hole_center - final_peg_tip))

    reward = (
        -2.0 * dist
        + 20.0 * insertion_depth
        - 0.001 * max(0.0, max_force_mag - 5.0)
        - 0.01 * step_count
    )
    if success:
        reward += 50.0

    return {
        "trajectory": {"ee_poses": np.stack(ee_poses).astype(np.float32)},
        "ee_poses": np.stack(ee_poses).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "force_profile": np.stack(forces).astype(np.float32),
        "torque_profile": np.stack(torques).astype(np.float32),
        "forces": np.stack(forces).astype(np.float32),
        "torques": np.stack(torques).astype(np.float32),
        "insertion_depth": float(insertion_depth),
        "final_distance": dist,
        "max_force": max_force_mag,
        "step_count": step_count,
        "success": success,
        "reward": float(reward),
        "gains": dict(gains),
        "scene_config": cfg,
    }
