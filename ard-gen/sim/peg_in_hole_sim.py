"""ARD-Gen 0단계(Seed 확보)용 peg-in-hole 시뮬레이션 + admittance controller.

admittance control 수식 (force-error에 대한 PD):
    F_error_xy = F_ext_xy - F_desired_xy         (F_desired_xy는 보통 0)
    dF_error_xy/dt ≈ (F_error_xy - F_error_xy_prev) / dt
    Δx_xy = -Kp_xy * F_error_xy - Kd_xy * dF_error_xy/dt

    부호는 손으로 미리 정한 게 아니라 실제로 돌려보고 정했다: peg tip을
    구형(라운드)으로 만들어야 애초에 xy 방향 힘이 생긴다는 것부터가 실험으로
    알아낸 사실이고(아래 참고), 그 상태에서 오프셋을 준 채 Kp 부호를 바꿔가며
    실행해보니 **음수 부호일 때만** 실제로 접촉력이 줄고 삽입에 성공했다
    (README 검증 섹션 1번 — Kp>0은 오히려 벽 쪽으로 더 파고들어 정체됨).
    그래서 F_error 방향의 "반대"로 위치를 보정하도록 최종 구현했다.

    tip이 구형인 이유: peg 바닥과 벽 윗면이 둘 다 평평하면 접촉 normal이
    항상 수직이라 xy 힘이 전혀 안 생긴다(기하학적으로 당연함) — 게인이
    아무리 커도 admittance controller가 반응할 신호 자체가 없었다. 실제
    핀/커넥터에 흔한 라운드 tip과 같은 이유로 peg 끝을 구형으로 바꿔서,
    오프셋 상태로 벽에 닿을 때 접촉 normal에 xy 성분이 생기게 했다
    (assets/peg_in_hole.xml의 peg_tip_ball).

    삽입 방향(z축)은 게인과 무관하게 매 스텝 일정 속도(Z_RATE)로 내려간다.
    wrist는 이 단계에서는 능동 보정하지 않고 초기값을 그대로 유지한다
    (peg 초기 자세가 이미 정렬돼 있다고 가정 — 회전 misalignment 보정은
    이 seed 컨트롤러의 범위 밖이다. README 한계 섹션 참고).
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "peg_in_hole.xml")

N_SUBSTEPS = 5  # mj_step 호출당 substep 수 (timestep=0.002 -> 제어 주기 dt=0.01s)
DT = N_SUBSTEPS * 0.002

Z_RATE = 0.0004  # m / control step, 삽입 방향 일정 속도
MAX_STEPS = 350

TARGET_INSERTION_DEPTH = 0.04  # m (scene_config로 덮어쓸 수 있음)

# hole 벽 nominal 치수 (scene_config['clearance_m']이 없을 때의 기본값과,
# 벽 두께처럼 clearance와 무관하게 고정으로 취급하는 값들)
_PEG_HALF_WIDTH = 0.010
_WALL_HALF_THICKNESS = 0.004
_WALL_HALF_HEIGHT = 0.0275
_WALL_CENTER_Z = -0.0275
_FLOOR_HALF_THICKNESS = 0.002
_FLOOR_CENTER_Z = -0.057
_NOMINAL_CLEARANCE_M = 0.003  # 총 지름 clearance (편측 1.5mm)

_INIT_Z_QPOS = -0.06  # peg tip이 opening 위 약 2cm에서 시작 (구형 팁으로 바뀌며 재계산됨)


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

    peg_init_offset_xy 범위(6~10mm)는 임의로 잡은 게 아니라 실측으로 정했다:
    2~6mm는 peg tip의 구형 곡률만으로 게인 없이도 대부분 자가정렬되고(검증
    섹션 1번), 12mm 이상은 admittance controller가 반응할 정도로 xy 접촉력이
    생기기도 전에 평평한 면끼리 걸려버려 게인을 아무리 키워도 소용없었다.
    6~10mm가 "게인 있어야 성공, 없으면 실패"가 실제로 갈리는 구간이다."""
    return {
        "hole_pos_xy": tuple(rng.uniform(-0.004, 0.004, size=2)),
        "friction": float(rng.uniform(0.2, 0.8)),
        "clearance_m": float(rng.uniform(0.0025, 0.0035)),
        "peg_init_offset_xy": tuple(
            rng.choice([-1.0, 1.0], size=2) * rng.uniform(0.006, 0.010, size=2)
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
        self._wrist_body_id = self.model.body("wrist").id
        self._peg_tip_site_id = self.model.site("peg_tip_site").id
        self._peg_geom_ids = [self.model.geom("peg_shaft").id, self.model.geom("peg_tip_ball").id]

        self._wall_geom_ids = {
            name: self.model.geom(f"hole_wall_{name}").id for name in ("px", "nx", "py", "ny")
        }
        self._floor_geom_id = self.model.geom("hole_floor").id

        self._joint_qposadr = {
            name: self.model.joint(name).qposadr[0]
            for name in ("slide_x", "slide_y", "slide_z", "hinge_wrist")
        }
        self._actuator_ids = {
            name: self.model.actuator(f"act_{name.split('_')[-1]}").id
            for name in ("slide_x", "slide_y", "slide_z", "hinge_wrist")
        }

        force_adr = self.model.sensor("peg_force").adr[0]
        torque_adr = self.model.sensor("peg_torque").adr[0]
        self._force_slice = slice(force_adr, force_adr + 3)
        self._torque_slice = slice(torque_adr, torque_adr + 3)

        self._nominal_hole_pos = self.model.body_pos[self._hole_body_id].copy()

    # ------------------------------------------------------------------
    def _apply_clearance(self, clearance_m: float) -> float:
        """hole 벽 4개의 위치/크기를 원하는 clearance(총 지름, m)에 맞게
        런타임으로 덮어쓴다. 벽 두께는 고정이고, inner_half(개구부 반폭)만
        바뀐다. 반환값은 실제로 적용된 hole_outer_half_width (리워드의 xy
        footprint 게이팅에 쓰임)."""
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

        init_x = hole_pos[0] + scene_config["peg_init_offset_xy"][0]
        init_y = hole_pos[1] + scene_config["peg_init_offset_xy"][1]
        init_wrist = scene_config["peg_init_wrist"]

        init_qpos = {
            "slide_x": init_x,
            "slide_y": init_y,
            "slide_z": _INIT_Z_QPOS,
            "hinge_wrist": init_wrist,
        }
        for name, value in init_qpos.items():
            self.data.qpos[self._joint_qposadr[name]] = value
            self.data.ctrl[self._actuator_ids[name]] = value

        mujoco.mj_forward(self.model, self.data)
        return outer_half

    def get_force_torque(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.sensordata[self._force_slice].copy(),
            self.data.sensordata[self._torque_slice].copy(),
        )

    def get_ee_pose(self) -> np.ndarray:
        """(x, y, z, wrist) — wrist 바디 world position + hinge_wrist 각도.
        회전 조인트가 결합돼 있지 않으므로 hinge_wrist qpos가 곧 절대 각도다."""
        pos = self.data.xpos[self._wrist_body_id]
        wrist = self.data.qpos[self._joint_qposadr["hinge_wrist"]]
        return np.array([pos[0], pos[1], pos[2], wrist])

    def get_peg_tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._peg_tip_site_id].copy()

    def get_hole_center_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._hole_site_id].copy()

    def step(self, delta: np.ndarray) -> None:
        """delta = (dx, dy, dz, dwrist) 를 각 actuator ctrl에 더하고(클리핑),
        n_substeps만큼 물리를 진행한다."""
        for i, name in enumerate(("slide_x", "slide_y", "slide_z", "hinge_wrist")):
            act_id = self._actuator_ids[name]
            lo, hi = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = np.clip(self.data.ctrl[act_id] + delta[i], lo, hi)
        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)


def run_episode(gains: dict[str, float], scene_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """admittance controller로 한 에피소드를 실행한다.

    Args:
        gains: {"Kp_xy": float, "Kd_xy": float}
        scene_config: hole 위치/마찰/clearance/peg 초기 오프셋. 없으면 기본값.

    Returns:
        dict with: trajectory(ee_poses, (T+1,4)), actions((T,4)), forces((T,3)),
        torques((T,3)), insertion_depth(float), success(bool), reward(float),
        step_count(int), max_force(float).
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

        # 부호: -Kp*F_error (경험적으로 검증된 방향 — 모듈 docstring 참고)
        delta_xy = -kp_xy * force_error_xy - kd_xy * d_force_error_xy
        delta = np.array([delta_xy[0], delta_xy[1], -Z_RATE, 0.0])

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
