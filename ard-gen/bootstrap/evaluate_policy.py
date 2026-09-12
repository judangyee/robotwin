"""ARD-Gen 2단계(부트스트래핑) 3/3: 행동 복제로 학습한 정책을 검증한다.

train_policy.py가 만든 policy.pt를, evaluate_generalization.py와 똑같은
무작위 시나리오 분포/시드로 돌려서 성공률을 잰다. admittance controller
(원본 공식) 및 게인=0 베이스라인과 나란히 비교해서, "시연 데이터로 학습한
신경망이 손으로 짠 공식만큼 일반화하는지"를 직접 확인한다.

사용 예:
    python bootstrap/evaluate_policy.py --policy-path ./policy.pt \
        --seed-path ../seed_trajectory.npz --n-trials 200 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from bootstrap.train_policy import AdmittancePolicy
from sim.peg_in_hole_sim import DT, MAX_STEPS, Z_RATE, PegInHoleSim, sample_scene_config


def load_policy(path: str) -> tuple[AdmittancePolicy, np.ndarray, np.ndarray]:
    ckpt = torch.load(path, weights_only=False)
    policy = AdmittancePolicy()
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, ckpt["obs_mean"], ckpt["obs_std"]


def run_episode_with_policy(sim: PegInHoleSim, policy: AdmittancePolicy, obs_mean, obs_std, cfg) -> dict:
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    prev_force_error_xy = np.zeros(2)
    max_force_mag = 0.0
    insertion_depth = 0.0
    success = False
    step_count = 0

    for step_count in range(1, MAX_STEPS + 1):
        force, _torque = sim.get_force_torque()
        force_mag = float(np.linalg.norm(force))
        max_force_mag = max(max_force_mag, force_mag)

        force_error_xy = force[:2]
        d_force_error_xy = (force_error_xy - prev_force_error_xy) / DT
        prev_force_error_xy = force_error_xy

        obs = np.concatenate([force_error_xy, d_force_error_xy]).astype(np.float32)
        obs_norm = (obs - obs_mean) / obs_std
        with torch.no_grad():
            delta_xy = policy(torch.from_numpy(obs_norm.astype(np.float32))).numpy()

        delta = np.array([delta_xy[0], delta_xy[1], -Z_RATE])
        sim.step(delta)

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

    return {
        "success": success,
        "insertion_depth": insertion_depth,
        "max_force": max_force_mag,
        "step_count": step_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=str, default="./policy.pt")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="evaluate_generalization.py와 동일한 시드를 써야 직접 비교 가능")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    policy, obs_mean, obs_std = load_policy(args.policy_path)
    sim = PegInHoleSim()
    rng = np.random.default_rng(args.seed)

    results = []
    for _ in range(args.n_trials):
        cfg = sample_scene_config(rng)
        results.append(run_episode_with_policy(sim, policy, obs_mean, obs_std, cfg))

    n = len(results)
    successes = [r for r in results if r["success"]]
    success_rate = len(successes) / n
    mean_force = float(np.mean([r["max_force"] for r in results]))
    mean_steps_success = float(np.mean([r["step_count"] for r in successes])) if successes else float("nan")

    print(
        f"[evaluate_policy] 학습된 정책: success_rate={success_rate:.1%} ({len(successes)}/{n})  "
        f"mean_max_force={mean_force:.2f}  mean_steps(success only)={mean_steps_success:.1f}"
    )


if __name__ == "__main__":
    main()
