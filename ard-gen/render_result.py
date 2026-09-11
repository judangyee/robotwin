"""cma_search.py가 찾은 게인(seed_trajectory.npz)으로 admittance controller
롤아웃을 mp4로 렌더링한다. 헤드리스 환경에서는 MUJOCO_GL=osmesa가 필요하다
(rldc-lite/scripts/render_video.py에서 확인한 것과 동일한 소프트웨어 렌더러).

사용 예:
    MUJOCO_GL=osmesa python render_result.py \
        --seed-path ./seed_trajectory.npz --out-path ./result.mp4
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco
import numpy as np

from sim.peg_in_hole_sim import PegInHoleSim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=str, default="./seed_trajectory.npz")
    parser.add_argument("--out-path", type=str, default="./result.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--freeze-frames", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.seed_path, allow_pickle=False)
    gains = {"Kp_xy": float(data["gains"][0]), "Kd_xy": float(data["gains"][1])}
    scene_config = json.loads(str(data["scene_config"]))
    print(f"[render_result] gains={gains} scene_config={scene_config}")

    sim = PegInHoleSim()
    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [0, 0, 0.05]
    cam.distance = 0.4
    cam.azimuth = 130
    cam.elevation = -18

    from sim.peg_in_hole_sim import DT, MAX_STEPS, Z_RATE  # noqa: F401

    outer_half = sim.reset(scene_config)
    target_depth = scene_config["target_insertion_depth"]

    frames = []
    renderer.update_scene(sim.data, camera=cam)
    frames.append(renderer.render().copy())

    kp_xy, kd_xy = gains["Kp_xy"], gains["Kd_xy"]
    prev_force_error_xy = np.zeros(2)
    success = False
    insertion_depth = 0.0

    for step in range(1, MAX_STEPS + 1):
        force, _torque = sim.get_force_torque()
        force_error_xy = force[:2]
        d_force_error_xy = (force_error_xy - prev_force_error_xy) / DT
        prev_force_error_xy = force_error_xy
        delta_xy = -kp_xy * force_error_xy - kd_xy * d_force_error_xy
        delta = np.array([delta_xy[0], delta_xy[1], -Z_RATE, 0.0])

        sim.step(delta)
        renderer.update_scene(sim.data, camera=cam)
        frames.append(renderer.render().copy())

        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_ok = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = max(0.0, float(hole_center[2] - peg_tip[2]))
        insertion_depth = raw_depth if xy_ok else 0.0
        if insertion_depth >= target_depth:
            success = True
            break

    for _ in range(args.freeze_frames):
        frames.append(frames[-1])

    imageio.mimsave(args.out_path, frames, fps=args.fps, quality=8)
    print(
        f"[render_result] steps={len(frames)} success={success} "
        f"insertion_depth={insertion_depth:.4f} -> {args.out_path}"
    )
    sim = None
    renderer.close()


if __name__ == "__main__":
    main()
