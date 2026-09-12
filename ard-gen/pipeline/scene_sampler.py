"""ARD-Gen 1단계: 공유 씬 설정.

매 에피소드마다 hole 위치/자세, 마찰계수, clearance, peg 초기 오프셋을
무작위로 샘플링한다. 이 함수가 반환하는 값을 2단계(Actuator)와 3단계
(Stabilizer)에 동일하게 전달해서, 두 팔이 물리적으로 같은 씬을 공유하도록
한다(PIPELINE.md의 핵심 설계 원칙 3).

주의: 이 스키마는 sim/peg_in_hole_sim.py의 run_episode()가 바로 받는
scene_config 스키마와 다르다(그쪽은 이 파이프라인이 생기기 전, CMA-ES
탐색 하나만을 위해 만든 것). run_episode()에 넘기려면 to_sim_scene_config()로
변환해야 한다.

clearance를 friction보다 넓게(2~5mm, friction은 ±30%) 무작위화하는 이유:
margin(여유 공간) 관련 무작위화가 마찰 같은 재질 파라미터보다 실제
성공/실패에 더 안정적인 신호를 준다는 원칙을 반영한 것 — 0단계 CMA-ES
검증 과정에서도 friction보다 clearance/오프셋 쪽 변화가 성공률에 더 뚜렷한
영향을 줬다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# "기준값" — sim/peg_in_hole_sim.py의 _default_scene_config()가 쓰는 nominal
# 마찰계수와 동일하게 맞췄다. 그쪽 nominal이 바뀌면 이 값도 같이 바꿔야 한다.
_BASE_FRICTION = 0.5

_HOLE_XY_RANGE_M = 0.01  # 기준 위치에서 ±1cm
_HOLE_THETA_RANGE_RAD = 0.1  # ±0.1 rad(약 ±5.7˚). 주의: 아래 "구현 상태" 참고
_FRICTION_SCALE_RANGE = (0.7, 1.3)
_CLEARANCE_RANGE_M = (0.002, 0.005)  # 2mm~5mm
_PEG_OFFSET_RANGE_M = 0.015  # ±1.5cm


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """무작위 공유 씬 설정을 하나 반환한다.

    Args:
        rng: 재현 가능한 시퀀스가 필요하면 np.random.default_rng(seed)를
            넘긴다. 생략하면 호출마다 새 전역 무작위성을 쓴다(재현 불가).

    Returns:
        {
            "hole_pose": (dx, dy, theta) -- 기준 hole 위치/자세 대비 오프셋.
                dx, dy는 ±1cm, theta는 ±0.1rad. **theta는 현재 sim이 아직
                소비하지 않는다**(assets/peg_in_hole.xml의 hole에 회전
                자유도가 없음) -- 3단계(Stabilizer 기하 변환)에서 SE(3)
                변환에 필요해질 값이라 스키마에는 미리 넣어뒀다. sim으로
                넘길 때는 to_sim_scene_config()가 이 필드를 버리고 dx,dy만
                쓴다.
            "friction": float -- 기준 마찰계수(0.5)의 0.7~1.3배.
            "clearance_m": float -- 2mm~5mm.
            "peg_init_offset": (dx, dy) -- ±1.5cm.
        }
    """
    if rng is None:
        rng = np.random.default_rng()

    hole_dx, hole_dy = rng.uniform(-_HOLE_XY_RANGE_M, _HOLE_XY_RANGE_M, size=2)
    hole_theta = rng.uniform(-_HOLE_THETA_RANGE_RAD, _HOLE_THETA_RANGE_RAD)

    friction_scale = rng.uniform(*_FRICTION_SCALE_RANGE)
    friction = _BASE_FRICTION * friction_scale

    clearance_m = rng.uniform(*_CLEARANCE_RANGE_M)

    peg_offset = tuple(rng.uniform(-_PEG_OFFSET_RANGE_M, _PEG_OFFSET_RANGE_M, size=2))

    return {
        "hole_pose": (float(hole_dx), float(hole_dy), float(hole_theta)),
        "friction": float(friction),
        "clearance_m": float(clearance_m),
        "peg_init_offset": (float(peg_offset[0]), float(peg_offset[1])),
    }


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    """공유 씬 설정을 sim/peg_in_hole_sim.py의 run_episode()가 받는
    scene_config 스키마로 변환한다. hole_pose의 theta는 현재 sim이
    회전을 지원하지 않아 버려진다(위 docstring 참고)."""
    from sim.peg_in_hole_sim import TARGET_INSERTION_DEPTH

    hole_dx, hole_dy, _theta = shared_cfg["hole_pose"]
    return {
        "hole_pos_xy": (hole_dx, hole_dy),
        "friction": shared_cfg["friction"],
        "clearance_m": shared_cfg["clearance_m"],
        "peg_init_offset_xy": shared_cfg["peg_init_offset"],
        "peg_init_wrist": 0.0,
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }
