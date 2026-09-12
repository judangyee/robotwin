"""CMA-ES가 찾은 게인이 3개 대표 시나리오 밖에서도 통하는지 검증한다.

cma_search.py는 고정된 시나리오 3개(오프셋 14mm/0, 9.9mm/9.9mm, 9.9mm/-9.9mm)의
평균 리워드만으로 게인을 골랐다. 이 스크립트는 그 게인을
sim.sample_scene_config()로 매번 새로 뽑은 무작위 시나리오(오프셋 반경
9~14mm 전체, 각도 ±45° 전체, 마찰/clearance도 랜덤) N개에 대해 실행해서
실제 성공률을 측정한다. 비교를 위해 게인=0(피드백 없이 그냥 수직으로만
내려가는 경우) 베이스라인도 같이 돌린다 — 게인이 실제로 기여하는지 보여주기
위함이다.

사용 예:
    python evaluate_generalization.py --n-trials 200 --seed-path ./seed_trajectory.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sim.peg_in_hole_sim import PegInHoleSim, _run_episode_with_sim, sample_scene_config


def run_trials(sim: PegInHoleSim, gains: dict[str, float], n_trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        cfg = sample_scene_config(rng)
        result = _run_episode_with_sim(sim, gains, cfg)
        results.append(
            {
                "success": bool(result["success"]),
                "insertion_depth": float(result["insertion_depth"]),
                "max_force": float(result["max_force"]),
                "step_count": int(result["step_count"]),
                "reward": float(result["reward"]),
                "offset_xy": cfg["peg_init_offset_xy"],
                "friction": cfg["friction"],
                "clearance_m": cfg["clearance_m"],
            }
        )
    return results


def summarize(name: str, results: list[dict]) -> dict:
    n = len(results)
    successes = [r for r in results if r["success"]]
    success_rate = len(successes) / n
    mean_reward = float(np.mean([r["reward"] for r in results]))
    mean_force = float(np.mean([r["max_force"] for r in results]))
    mean_steps_success = float(np.mean([r["step_count"] for r in successes])) if successes else float("nan")
    print(
        f"[{name}] success_rate={success_rate:.1%} ({len(successes)}/{n})  "
        f"mean_reward={mean_reward:.2f}  mean_max_force={mean_force:.2f}  "
        f"mean_steps(success only)={mean_steps_success:.1f}"
    )
    return {
        "n_trials": n,
        "success_rate": success_rate,
        "n_success": len(successes),
        "mean_reward": mean_reward,
        "mean_max_force": mean_force,
        "mean_steps_success": mean_steps_success,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=str, default="./seed_trajectory.npz")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="시나리오 샘플링용 RNG 시드 (CMA-ES 시드와 무관)")
    parser.add_argument("--out-path", type=str, default="./generalization_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.seed_path, allow_pickle=False)
    found_gains = {"Kp_xy": float(data["gains"][0]), "Kd_xy": float(data["gains"][1])}
    zero_gains = {"Kp_xy": 0.0, "Kd_xy": 0.0}
    print(f"[evaluate_generalization] 검증 대상 게인: {found_gains}")
    print(f"[evaluate_generalization] N={args.n_trials}, scenario seed={args.seed}")

    sim_found = PegInHoleSim()
    sim_zero = PegInHoleSim()

    results_found = run_trials(sim_found, found_gains, args.n_trials, args.seed)
    results_zero = run_trials(sim_zero, zero_gains, args.n_trials, args.seed)  # 같은 시드 -> 같은 시나리오 시퀀스

    summary_found = summarize("CMA-ES 게인", results_found)
    summary_zero = summarize("게인=0 (베이스라인)", results_zero)

    report = {
        "found_gains": found_gains,
        "summary_found": summary_found,
        "summary_zero_baseline": summary_zero,
        "trials_found": results_found,
        "trials_zero_baseline": results_zero,
    }
    with open(args.out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[evaluate_generalization] 상세 결과 저장: {args.out_path}")


if __name__ == "__main__":
    main()
