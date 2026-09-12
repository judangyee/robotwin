"""ARD-Gen 4단계: 궤적 실행 & 필터링 (지금은 Actuator 단독).

Stabilizer(왼팔)는 이번 검증 범위에서 완전히 제외한다 -- PIPELINE.md
"지금은 Actuator 단일 팔로만 파이프라인 검증 중" 참고. 원래 4단계는
"Actuator + Stabilizer 궤적을 같은 시뮬레이션에서 동시에 재생"하는
단계지만, 왼팔이 아직 없으므로 지금은 Actuator 궤적만 실행한다.

흐름: 1단계(scene_sampler)로 씬을 뽑고 -> 2-B(diffusion_gains)로 그 씬에
맞는 게인을 생성 -> sim.run_episode()로 Actuator를 실행 -> insertion_depth
기준 성공 여부(run_episode가 이미 판정)로 필터링 -> 성공한 에피소드만
data/episodes/에 개별 npz로 저장한다.

2-A/2-B와 달리 여기서부터는 **실패 에피소드를 보관하지 않는다** -- 이 뒤
5단계(언어 라벨링)의 출력물이 VLA 학습 데이터 자체이므로, 실패한 시도를
남겨봐야 학습에 쓸 수 없어서다.

사용 예:
    python pipeline/filter_episodes.py --n-scenes 100 --scene-seed 42 \
        --diffusion-path ./data/bootstrap/diffusion_gains.pt \
        --out-dir ./data/episodes
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from pipeline.diffusion_gains import load_checkpoint, sample_gains
from pipeline.episode_io import save_episode
from pipeline.scene_sampler import sample_scene_config, to_sim_scene_config
from sim.peg_in_hole_sim import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-scenes", type=int, default=100)
    parser.add_argument("--scene-seed", type=int, default=42)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="diffusion reverse-sampling(torch)의 노이즈 시드 -- 안 고정하면 씬 시드가 같아도 "
        "매 실행마다 뽑히는 게인이 달라져 성공/실패 결과가 재현되지 않는다",
    )
    parser.add_argument("--diffusion-path", type=str, default="./data/bootstrap/diffusion_gains.pt")
    parser.add_argument("--out-dir", type=str, default="./data/episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 이전 실행의 episode_*.npz가 남아있으면 이번 결과와 섞여서 개수/내용이
    # 헷갈리므로, 매 실행마다 out-dir을 비우고 새로 채운다.
    for stale in glob.glob(os.path.join(args.out_dir, "episode_*.npz")):
        os.remove(stale)

    torch.manual_seed(args.sample_seed)
    model, diffusion, cond_mean, cond_std, gain_mean, gain_std = load_checkpoint(args.diffusion_path)
    rng = np.random.default_rng(args.scene_seed)

    n_success = 0
    for i in range(args.n_scenes):
        scene_cfg = sample_scene_config(rng)
        gains = sample_gains(model, diffusion, cond_mean, cond_std, gain_mean, gain_std, scene_cfg)
        sim_cfg = to_sim_scene_config(scene_cfg)
        result = run_episode(gains, sim_cfg)

        if not result["success"]:
            continue

        force_mag = np.linalg.norm(result["forces"], axis=1)
        episode = {
            "right_arm": {
                "traj": result["ee_poses"],
                "gains": [gains["Kp_xy"], gains["Kd_xy"]],
                "force": result["forces"],
                "torque": result["torques"],
                "role": "actuator",
            },
            "scene_config": scene_cfg,
            "success": True,
            "insertion_depth": result["insertion_depth"],
            "force_max": float(force_mag.max()),
        }
        out_path = os.path.join(args.out_dir, f"episode_{n_success:04d}.npz")
        save_episode(out_path, episode)
        n_success += 1

    print(
        f"[filter_episodes] 씬 {args.n_scenes}개 중 {n_success}개 성공 "
        f"({n_success / args.n_scenes:.1%}) -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
