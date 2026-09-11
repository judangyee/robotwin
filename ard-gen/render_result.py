"""cma_search.py가 찾은 게인(seed_trajectory.npz)으로 admittance controller
롤아웃을 mp4로 렌더링한다. 헤드리스 환경에서는 MUJOCO_GL=osmesa가 필요하다
(rldc-lite/scripts/render_video.py에서 확인한 것과 동일한 소프트웨어 렌더러).

카메라는 assets/peg_in_hole.xml에 정의된 이름을 쓴다:
  - wrist_cam: gripper_link에 고정된 손목 카메라 (peg가 뻗어나가는 방향을 봄)
  - top_cam:   hole 위 30cm에서 수직으로 내려다보는 탑 뷰 (world 고정)
--cameras에 콤마로 여러 개를 주면 시뮬레이션은 한 번만 돌리고(물리는
결정적이므로) 카메라별로 각각 다른 mp4를 만든다.

사용 예:
    MUJOCO_GL=osmesa python render_result.py \
        --seed-path ./seed_trajectory.npz --out-dir . \
        --cameras wrist_cam,top_cam
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco
import numpy as np

from sim.peg_in_hole_sim import DT, MAX_STEPS, Z_RATE, PegInHoleSim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=str, default="./seed_trajectory.npz")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument(
        "--cameras",
        type=str,
        default="wrist_cam,top_cam",
        help="콤마로 구분된 카메라 이름들 (assets/peg_in_hole.xml에 정의됨), 또는 'free'로 자유 시점",
    )
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

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]

    sim = PegInHoleSim()
    renderers = {name: mujoco.Renderer(sim.model, height=args.height, width=args.width) for name in camera_names}

    free_cam = mujoco.MjvCamera()
    free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    free_cam.lookat = [0.1, 0, 0.1]
    free_cam.distance = 0.4
    free_cam.azimuth = 130
    free_cam.elevation = -18

    def camera_arg(name: str):
        return free_cam if name == "free" else name

    outer_half = sim.reset(scene_config)
    target_depth = scene_config["target_insertion_depth"]

    frames_by_cam = {name: [] for name in camera_names}

    def capture():
        for name, renderer in renderers.items():
            renderer.update_scene(sim.data, camera=camera_arg(name))
            frames_by_cam[name].append(renderer.render().copy())

    capture()

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
        delta = np.array([delta_xy[0], delta_xy[1], -Z_RATE])

        sim.step(delta)
        capture()

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

    os.makedirs(args.out_dir, exist_ok=True)
    for name, frames in frames_by_cam.items():
        for _ in range(args.freeze_frames):
            frames.append(frames[-1])
        out_path = os.path.join(args.out_dir, f"result_{name}.mp4")
        imageio.mimsave(out_path, frames, fps=args.fps, quality=8)
        print(f"[render_result] {name}: {len(frames)} frames -> {out_path}")

    print(
        f"[render_result] steps={step} success={success} insertion_depth={insertion_depth:.4f}"
    )
    for renderer in renderers.values():
        renderer.close()


if __name__ == "__main__":
    main()
