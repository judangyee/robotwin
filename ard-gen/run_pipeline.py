"""ARD-Gen 0->1->2-A->2-B->4->5단계를 한 번에 실행해서, Stabilizer 없이도
파이프라인이 끝까지 도는지 확인한다. 3단계(Stabilizer)는 이번 검증에서
완전히 제외한다 (PIPELINE.md 참고).

0단계(CMA-ES seed)와 2단계(부트스트래핑+diffusion 학습)는 비용이 드는
단계라서, 산출물(seed_trajectory.npz / bootstrap_dataset.npz /
diffusion_gains.pt)이 이미 있으면 재사용하고 --force-* 플래그로만 다시
만든다. 4단계(episode 생성)와 5단계(언어 라벨링)는 지금 실제로 검증하려는
부분이므로 매번 새로 실행한다.

사용 예:
    python run_pipeline.py --n-scenes 100
    python run_pipeline.py --n-scenes 100 --force-seed --force-bootstrap --force-diffusion
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from pipeline.episode_io import load_episode

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=str, default="./seed_trajectory.npz")
    parser.add_argument("--bootstrap-path", type=str, default="./data/bootstrap/bootstrap_dataset.npz")
    parser.add_argument("--diffusion-path", type=str, default="./data/bootstrap/diffusion_gains.pt")
    parser.add_argument("--episodes-dir", type=str, default="./data/episodes")
    parser.add_argument("--n-scenes", type=int, default=100)
    parser.add_argument("--scene-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=0, help="diffusion 샘플링(torch) 시드, 재현성용")
    parser.add_argument("--n-show", type=int, default=3, help="마지막에 자세히 출력할 샘플 에피소드 수")
    parser.add_argument("--force-seed", action="store_true", help="0단계(CMA-ES)를 강제로 다시 실행")
    parser.add_argument("--force-bootstrap", action="store_true", help="2-A단계를 강제로 다시 실행")
    parser.add_argument("--force-diffusion", action="store_true", help="2-B단계를 강제로 다시 학습")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("0단계 -- Seed 확보 (CMA-ES)")
    print("=" * 70)
    if args.force_seed or not os.path.exists(args.seed_path):
        _run([sys.executable, "optimize/cma_search.py", "--out-path", args.seed_path])
    else:
        print(f"[run_pipeline] 이미 있음, 재사용: {args.seed_path}")

    print()
    print("=" * 70)
    print("1단계 -- 공유 씬 설정 (pipeline/scene_sampler.py)")
    print("=" * 70)
    print("[run_pipeline] 별도 실행 단계 없음 -- 4단계 내부에서 sample_scene_config()로 매 씬마다 호출됨")

    print()
    print("=" * 70)
    print("2-A단계 -- 게인 부트스트래핑")
    print("=" * 70)
    if args.force_bootstrap or not os.path.exists(args.bootstrap_path):
        _run(
            [
                sys.executable,
                "pipeline/bootstrap.py",
                "--seed-path",
                args.seed_path,
                "--out-path",
                args.bootstrap_path,
            ]
        )
    else:
        print(f"[run_pipeline] 이미 있음, 재사용: {args.bootstrap_path}")

    print()
    print("=" * 70)
    print("2-B단계 -- Diffusion 게인 생성 모델 학습")
    print("=" * 70)
    if args.force_diffusion or not os.path.exists(args.diffusion_path):
        _run(
            [
                sys.executable,
                "pipeline/diffusion_gains.py",
                "--dataset-path",
                args.bootstrap_path,
                "--out-path",
                args.diffusion_path,
            ]
        )
    else:
        print(f"[run_pipeline] 이미 있음, 재사용: {args.diffusion_path}")

    print()
    print("=" * 70)
    print("4단계 -- 궤적 실행 & 필터링 (Actuator 단독, Stabilizer 제외)")
    print("=" * 70)
    _run(
        [
            sys.executable,
            "pipeline/filter_episodes.py",
            "--n-scenes",
            str(args.n_scenes),
            "--scene-seed",
            str(args.scene_seed),
            "--sample-seed",
            str(args.sample_seed),
            "--diffusion-path",
            args.diffusion_path,
            "--out-dir",
            args.episodes_dir,
        ]
    )

    print()
    print("=" * 70)
    print("5단계 -- 언어 라벨링")
    print("=" * 70)
    _run([sys.executable, "pipeline/language_labeling.py", "--episodes-dir", args.episodes_dir])

    print()
    print("=" * 70)
    print("최종 결과")
    print("=" * 70)
    paths = sorted(glob.glob(os.path.join(args.episodes_dir, "episode_*.npz")))
    episodes = [load_episode(p) for p in paths]
    complete = [e for e in episodes if e.get("language") is not None]
    print(
        f"씬 {args.n_scenes}개 시도 -> 성공(4단계 통과) {len(episodes)}개 -> "
        f"언어 라벨까지 완성된 에피소드 {len(complete)}개"
    )
    print(f"(행동 + force/torque + 언어가 모두 포함된 최종 에피소드 수: {len(complete)})")

    print()
    print(f"-- 샘플 {min(args.n_show, len(complete))}개 --")
    rng = np.random.default_rng(0)
    show_idx = rng.choice(len(complete), size=min(args.n_show, len(complete)), replace=False)
    for idx in sorted(show_idx):
        ep = complete[idx]
        right = ep["right_arm"]
        print()
        print(f"[episode #{idx}] {os.path.basename(paths[idx])}")
        print(f"  language: {ep['language']}")
        print(f"  role: {right['role']}, gains(Kp,Kd)={right['gains']}")
        print(f"  scene_config: {ep['scene_config']}")
        print(
            f"  traj shape={right['traj'].shape}, force shape={right['force'].shape}, "
            f"torque shape={right['torque'].shape}"
        )
        print(f"  insertion_depth={ep['insertion_depth']:.4f}, force_max={ep['force_max']:.2f}N")


if __name__ == "__main__":
    main()
