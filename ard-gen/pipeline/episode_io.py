"""4/5단계 에피소드 저장 포맷(data/episodes/*.npz) 공통 입출력.

에피소드는 논리적으로 다음 구조를 가진다 (지금은 Stabilizer가 없으므로
right_arm만 채워진다 -- PIPELINE.md의 "지금은 Actuator 단일 팔" 참고):

    {
        "right_arm": {
            "traj": (T+1, 4) ndarray,   # ee_poses (x,y,z,wrist_rotate)
            "gains": (2,) ndarray,      # [Kp_xy, Kd_xy]
            "force": (T, 3) ndarray,
            "torque": (T, 3) ndarray,
            "role": "actuator",
        },
        "scene_config": {...},          # pipeline.scene_sampler 스키마
        "success": True,
        "insertion_depth": float,
        "force_max": float,
        "language": str | None,         # 5단계에서 채워짐, 그 전엔 없음
    }

npz는 중첩 dict를 그대로 못 담으므로 "right_arm_" 접두사로 평탄화해서
저장하고, load_episode()가 다시 위 구조로 복원한다.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np


def save_episode(path: str, episode: dict[str, Any]) -> None:
    right = episode["right_arm"]
    kwargs: dict[str, Any] = {
        "right_arm_traj": np.asarray(right["traj"], dtype=np.float32),
        "right_arm_gains": np.asarray(right["gains"], dtype=np.float32),
        "right_arm_force": np.asarray(right["force"], dtype=np.float32),
        "right_arm_torque": np.asarray(right["torque"], dtype=np.float32),
        "right_arm_role": np.array(right["role"]),
        "scene_config": np.array(json.dumps(episode["scene_config"])),
        "success": np.array(bool(episode["success"])),
    }
    if episode.get("insertion_depth") is not None:
        kwargs["insertion_depth"] = np.array(episode["insertion_depth"], dtype=np.float32)
    if episode.get("force_max") is not None:
        kwargs["force_max"] = np.array(episode["force_max"], dtype=np.float32)
    if episode.get("language") is not None:
        kwargs["language"] = np.array(episode["language"])
    np.savez(path, **kwargs)


def load_episode(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    episode: dict[str, Any] = {
        "right_arm": {
            "traj": data["right_arm_traj"],
            "gains": data["right_arm_gains"],
            "force": data["right_arm_force"],
            "torque": data["right_arm_torque"],
            "role": str(data["right_arm_role"]),
        },
        "scene_config": json.loads(str(data["scene_config"])),
        "success": bool(data["success"]),
    }
    if "insertion_depth" in data.files:
        episode["insertion_depth"] = float(data["insertion_depth"])
    if "force_max" in data.files:
        episode["force_max"] = float(data["force_max"])
    if "language" in data.files:
        episode["language"] = str(data["language"])
    return episode


def save_language(path: str, language: str) -> None:
    """기존 에피소드 npz를 읽어서 language 필드만 채워 다시 저장한다."""
    episode = load_episode(path)
    episode["language"] = language
    save_episode(path, episode)
