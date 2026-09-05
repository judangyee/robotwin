"""학습된 정책(또는 랜덤 정책)의 롤아웃을 mp4 영상으로 렌더링한다.

헤드리스(디스플레이 없는) 환경에서는 소프트웨어 렌더러가 필요하므로 기본적으로
MUJOCO_GL=osmesa를 사용한다 (GPU가 있고 EGL이 설치돼 있다면 MUJOCO_GL=egl로
바꿔서 실행해도 된다).

주의 (중요): stable-baselines3/torch를 OSMesa 렌더러 생성 *이후에* import하면
세그폴트가 난다 (OSMesa와 torch의 OpenMP 스레드 초기화가 충돌하는 것으로 보임).
그래서 이 스크립트는 항상 정책(또는 torch)을 먼저 로드한 뒤에 mujoco.Renderer를
만든다. 이 순서를 바꾸지 말 것.

사용 예:
    # 학습된 정책으로 렌더링
    MUJOCO_GL=osmesa python scripts/render_video.py \
        --model-path ./checkpoints/ppo_peg_in_hole_final.zip \
        --out-path ./videos/success.mp4 --seed 42

    # 랜덤 정책(비교용 baseline)으로 렌더링
    MUJOCO_GL=osmesa python scripts/render_video.py \
        --out-path ./videos/random.mp4 --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="학습된 PPO 모델(.zip) 경로. 지정하지 않으면 랜덤 정책으로 렌더링한다.",
    )
    parser.add_argument("--out-path", type=str, default="./videos/rollout.mp4")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-domain-randomization",
        dest="domain_randomization",
        action="store_false",
        help="지정하면 도메인 무작위화 없이(고정된 hole 위치/마찰) 렌더링한다.",
    )
    parser.set_defaults(domain_randomization=True)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="정책을 샘플링(확률적)으로 실행. 기본은 --deterministic.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--freeze-frames", type=int, default=20, help="에피소드 종료 후 마지막 프레임을 반복해 잠깐 정지시키는 프레임 수")
    parser.add_argument("--cam-distance", type=float, default=0.35)
    parser.add_argument("--cam-azimuth", type=float, default=130.0)
    parser.add_argument("--cam-elevation", type=float, default=-25.0)
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=(0.0, 0.0, 0.05))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    # 정책(또는 torch)을 먼저 로드/import한 뒤에 mujoco.Renderer를 만들어야
    # OSMesa 렌더러와의 세그폴트를 피할 수 있다 (모듈 docstring 참고).
    model = None
    if args.model_path:
        from stable_baselines3 import PPO

        model = PPO.load(args.model_path)
    else:
        import torch  # noqa: F401

    import imageio
    import mujoco
    import numpy as np

    from envs.peg_in_hole_env import PegInHoleEnv

    env = PegInHoleEnv(
        max_episode_steps=args.max_episode_steps,
        domain_randomization=args.domain_randomization,
    )
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = list(args.cam_lookat)
    cam.distance = args.cam_distance
    cam.azimuth = args.cam_azimuth
    cam.elevation = args.cam_elevation

    rng = np.random.default_rng(args.seed)
    obs, info = env.reset(seed=args.seed)

    frames = []
    renderer.update_scene(env.data, camera=cam)
    frames.append(renderer.render().copy())

    success = False
    for _ in range(args.max_episode_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=args.deterministic)
        else:
            action = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render().copy())

        if info["success"]:
            success = True
        if terminated or truncated:
            break

    for _ in range(args.freeze_frames):
        frames.append(frames[-1])

    imageio.mimsave(args.out_path, frames, fps=args.fps, quality=8)

    print(
        f"[render_video] steps={len(frames)} success={success} "
        f"final_depth={info['insertion_depth']:.4f} -> {args.out_path}"
    )

    env.close()
    renderer.close()


if __name__ == "__main__":
    main()
