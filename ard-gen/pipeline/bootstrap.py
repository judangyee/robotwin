"""ARD-Gen 2-A단계: Actuator(오른팔) 부트스트래핑.

0단계에서 CMA-ES로 찾은 seed 게인(Kp_xy, Kd_xy) 하나만으로는 2-B단계의
diffusion 모델을 학습시킬 수 없다 — 학습 데이터가 1개뿐이기 때문이다.
그래서 이 스크립트가 seed 게인 주변에 넓은 무작위 노이즈를 줘서 시뮬레이션을
수백~수천 번 반복 실행하고, (씬 조건, 게인, 성공여부, force_profile) 기록을
대량으로 쌓는다. 이 기록이 2-B단계 diffusion 모델의 학습 데이터가 된다.

성공/실패 에피소드를 **둘 다** 저장한다 — diffusion이 "어떤 게인이
실패하는지"도 학습해야, 새 씬 조건에서 성공 확률 높은 쪽으로 샘플링할 수
있기 때문이다.

사용 예:
    python pipeline/bootstrap.py --seed-path ./seed_trajectory.npz \
        --n-trials 1000 --seed 0 --out-path ./data/bootstrap/bootstrap_dataset.npz
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pipeline.scene_sampler import sample_scene_config, to_sim_scene_config
from sim.peg_in_hole_sim import run_episode

# seed 게인에 곱할 노이즈 배율 범위. Kp_xy, Kd_xy에 각각 독립적으로 적용한다
# (같은 배율을 두 게인에 동시에 곱하면 탐색이 seed 방향의 1차원 직선으로만
# 퍼져서, diffusion 학습 데이터로 쓰기엔 다양성이 부족해진다).
_GAIN_NOISE_RANGE = (0.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=str, default="./seed_trajectory.npz")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="씬/게인 노이즈 샘플링용 RNG 시드")
    parser.add_argument("--out-path", type=str, default="./data/bootstrap/bootstrap_dataset.npz")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_data = np.load(args.seed_path, allow_pickle=False)
    seed_kp = float(seed_data["gains"][0])
    seed_kd = float(seed_data["gains"][1])
    print(f"[bootstrap] seed 게인: Kp_xy={seed_kp:.6f}, Kd_xy={seed_kd:.6e}")
    print(f"[bootstrap] 노이즈 범위: seed x [{_GAIN_NOISE_RANGE[0]}, {_GAIN_NOISE_RANGE[1]}] (Kp/Kd 독립)")

    rng = np.random.default_rng(args.seed)

    hole_poses = np.zeros((args.n_trials, 3), dtype=np.float32)
    frictions = np.zeros(args.n_trials, dtype=np.float32)
    clearances = np.zeros(args.n_trials, dtype=np.float32)
    peg_offsets = np.zeros((args.n_trials, 2), dtype=np.float32)
    kp_values = np.zeros(args.n_trials, dtype=np.float32)
    kd_values = np.zeros(args.n_trials, dtype=np.float32)
    successes = np.zeros(args.n_trials, dtype=bool)
    force_max = np.zeros(args.n_trials, dtype=np.float32)
    force_mean = np.zeros(args.n_trials, dtype=np.float32)
    insertion_depths = np.zeros(args.n_trials, dtype=np.float32)

    n_success_so_far = 0
    for i in range(args.n_trials):
        shared_cfg = sample_scene_config(rng)
        sim_cfg = to_sim_scene_config(shared_cfg)

        kp = seed_kp * rng.uniform(*_GAIN_NOISE_RANGE)
        kd = seed_kd * rng.uniform(*_GAIN_NOISE_RANGE)
        gains = {"Kp_xy": kp, "Kd_xy": kd}

        result = run_episode(gains, sim_cfg)

        hole_poses[i] = shared_cfg["hole_pose"]
        frictions[i] = shared_cfg["friction"]
        clearances[i] = shared_cfg["clearance_m"]
        peg_offsets[i] = shared_cfg["peg_init_offset"]
        kp_values[i] = kp
        kd_values[i] = kd
        successes[i] = result["success"]
        force_mag = np.linalg.norm(result["forces"], axis=1)
        force_max[i] = float(force_mag.max())
        force_mean[i] = float(force_mag.mean())
        insertion_depths[i] = result["insertion_depth"]

        if result["success"]:
            n_success_so_far += 1

        if (i + 1) % args.log_every == 0:
            print(f"[bootstrap] {i + 1}/{args.n_trials}  누적 성공률={n_success_so_far / (i + 1):.1%}")

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    np.savez(
        args.out_path,
        seed_gains=np.array([seed_kp, seed_kd], dtype=np.float32),
        hole_pose=hole_poses,
        friction=frictions,
        clearance_m=clearances,
        peg_init_offset=peg_offsets,
        kp_xy=kp_values,
        kd_xy=kd_values,
        success=successes,
        force_max=force_max,
        force_mean=force_mean,
        insertion_depth=insertion_depths,
    )
    print(f"[bootstrap] 저장: {args.out_path} ({args.n_trials}건)")

    _print_summary(successes, kp_values, kd_values, clearances, frictions, peg_offsets)


def _print_summary(
    successes: np.ndarray,
    kp_values: np.ndarray,
    kd_values: np.ndarray,
    clearances: np.ndarray,
    frictions: np.ndarray,
    peg_offsets: np.ndarray,
) -> None:
    n = len(successes)
    n_success = int(successes.sum())
    print()
    print("=" * 60)
    print(f"전체 성공률: {n_success / n:.1%} ({n_success}/{n})")

    if n_success > 0:
        print()
        print("성공 게인 분포:")
        print(f"  Kp_xy: 평균={kp_values[successes].mean():.6f}  표준편차={kp_values[successes].std():.6f}")
        print(f"  Kd_xy: 평균={kd_values[successes].mean():.6e}  표준편차={kd_values[successes].std():.6e}")

    print()
    print("실패가 어떤 조건에서 몰리는지 (clearance 중심):")
    fail = ~successes
    if fail.sum() > 0 and n_success > 0:
        # clearance를 4분위로 나눠 구간별 성공률을 본다.
        quartile_edges = np.quantile(clearances, [0.0, 0.25, 0.5, 0.75, 1.0])
        for lo, hi in zip(quartile_edges[:-1], quartile_edges[1:]):
            mask = (clearances >= lo) & (clearances <= hi)
            if mask.sum() == 0:
                continue
            rate = successes[mask].mean()
            print(f"  clearance [{lo * 1000:.2f}~{hi * 1000:.2f}]mm ({mask.sum()}건): 성공률 {rate:.1%}")

        print(f"  성공군 clearance 평균: {clearances[successes].mean() * 1000:.2f}mm")
        print(f"  실패군 clearance 평균: {clearances[fail].mean() * 1000:.2f}mm")
        print(f"  성공군 friction 평균: {frictions[successes].mean():.3f}  실패군 friction 평균: {frictions[fail].mean():.3f}")
        offset_mag = np.linalg.norm(peg_offsets, axis=1)
        print(
            f"  성공군 offset 크기 평균: {offset_mag[successes].mean() * 1000:.2f}mm  "
            f"실패군 offset 크기 평균: {offset_mag[fail].mean() * 1000:.2f}mm"
        )
    else:
        print("  실패(또는 성공) 사례가 없어 비교 불가.")
    print("=" * 60)


if __name__ == "__main__":
    main()
