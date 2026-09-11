"""ARD-Gen 0단계(Seed 확보): CMA-ES로 admittance controller 게인(Kp_xy, Kd_xy)을 찾는다.

각 세대마다 CMA-ES가 제안하는 게인 후보들을, 미리 정한 몇 가지 대표 시나리오
(scene_config)에 대해 sim/peg_in_hole_sim.run_episode()로 실행하고 평균 리워드로
평가한다. 목표 리워드(threshold)에 도달하거나 최대 세대 수에 도달하면 멈추고,
최종 게인값 + 그 게인으로 다시 실행한 대표 시나리오의 궤적을 seed_trajectory.npz로
저장한다.

사용 예:
    python optimize/cma_search.py --max-generations 100 --threshold 40.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cma
import numpy as np

from sim.peg_in_hole_sim import PegInHoleSim, _run_episode_with_sim, _default_scene_config

# 대표 평가 시나리오 3개: sim/peg_in_hole_sim.py에서 실측으로 확인한, "게인이
# 있어야만 성공하는" 난이도의 오프셋들 (게인=0이면 전부 실패하거나 힘이 큼).
EVAL_SCENE_CONFIGS: list[dict] = []
for _offset in [(0.009, -0.006), (0.009, 0.007), (0.0095, -0.007)]:
    _cfg = _default_scene_config()
    _cfg["peg_init_offset_xy"] = _offset
    EVAL_SCENE_CONFIGS.append(_cfg)

# 대표 시나리오 중 seed_trajectory.npz에 저장할 "메인" 시나리오 (가장 어려운 것).
PRIMARY_SCENE_CONFIG = EVAL_SCENE_CONFIGS[0]

KP_BOUNDS = (0.0001, 0.03)
KD_BOUNDS = (0.0, 0.002)


def evaluate_gains(sims: list[PegInHoleSim], kp_xy: float, kd_xy: float) -> float:
    """대표 시나리오들에 대한 평균 리워드를 계산한다."""
    gains = {"Kp_xy": kp_xy, "Kd_xy": kd_xy}
    rewards = []
    for sim, cfg in zip(sims, EVAL_SCENE_CONFIGS):
        result = _run_episode_with_sim(sim, gains, cfg)
        rewards.append(result["reward"])
    return float(np.mean(rewards))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-generations", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=49.3, help="이 평균 리워드에 도달하면 조기 종료")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-path", type=str, default="./seed_trajectory.npz")
    parser.add_argument("--curve-path", type=str, default="./convergence.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 일부러 나쁜 초기 추정값 + 작은 초기 스텝 크기에서 시작한다 (Kp가 거의
    # 0에 가까움 -> 대표 시나리오들에서 전부 실패) — CMA-ES가 몇 세대에 걸쳐
    # 스텝 크기를 키워가며 실제로 좋은 게인을 찾아가는 과정을 보여주기 위함.
    # sim 모듈 검증 결과상 진짜 좋은 값은 Kp_xy~0.003~0.012 부근이었다.
    x0 = [0.0001, 0.000001]
    sigma0 = 1.0  # CMA_stds로 좌표별 스케일을 따로 주므로 sigma0 자체는 1.0

    opts = {
        "bounds": [[KP_BOUNDS[0], KD_BOUNDS[0]], [KP_BOUNDS[1], KD_BOUNDS[1]]],
        "CMA_stds": [0.0006, 0.00002],
        "popsize": args.popsize,
        "maxiter": args.max_generations,
        "seed": args.seed,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # run_episode()가 매번 MjModel을 새로 컴파일하지 않도록, 대표 시나리오 개수만큼
    # PegInHoleSim을 한 번만 만들어 재사용한다 (훨씬 빠름).
    sims = [PegInHoleSim() for _ in EVAL_SCENE_CONFIGS]

    best_reward_per_gen = []
    best_overall = {"reward": -np.inf, "gains": None}

    generation = 0
    while not es.stop() and generation < args.max_generations:
        generation += 1
        candidates = es.ask()
        rewards = [evaluate_gains(sims, kp, kd) for kp, kd in candidates]
        es.tell(candidates, [-r for r in rewards])  # cma는 최소화하므로 부호 반전

        gen_best_idx = int(np.argmax(rewards))
        gen_best_reward = rewards[gen_best_idx]
        best_reward_per_gen.append(gen_best_reward)

        if gen_best_reward > best_overall["reward"]:
            best_overall["reward"] = gen_best_reward
            best_overall["gains"] = {
                "Kp_xy": float(candidates[gen_best_idx][0]),
                "Kd_xy": float(candidates[gen_best_idx][1]),
            }

        print(
            f"[cma_search] gen {generation:3d}: best_reward_this_gen={gen_best_reward:7.3f} "
            f"best_overall={best_overall['reward']:7.3f} "
            f"gains(Kp,Kd)={candidates[gen_best_idx][0]:.5f},{candidates[gen_best_idx][1]:.5f}"
        )

        if best_overall["reward"] >= args.threshold:
            print(f"[cma_search] 목표 리워드({args.threshold}) 도달 -> 조기 종료 (gen {generation})")
            break
    else:
        if generation >= args.max_generations:
            print(f"[cma_search] 최대 세대 수({args.max_generations}) 도달 -> 종료")

    print(f"[cma_search] 최종 게인: {best_overall['gains']}, 평균 리워드: {best_overall['reward']:.3f}")

    # 최종 게인으로 대표(가장 어려운) 시나리오를 다시 실행해 궤적을 뽑는다.
    primary_sim = PegInHoleSim()
    final_result = _run_episode_with_sim(primary_sim, best_overall["gains"], PRIMARY_SCENE_CONFIG)
    print(
        f"[cma_search] 메인 시나리오 재실행: success={final_result['success']} "
        f"insertion_depth={final_result['insertion_depth']:.4f} "
        f"max_force={final_result['max_force']:.2f} reward={final_result['reward']:.2f}"
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    np.savez(
        args.out_path,
        ee_poses=final_result["ee_poses"],
        actions=final_result["actions"],
        forces=final_result["forces"],
        torques=final_result["torques"],
        gains=np.array([best_overall["gains"]["Kp_xy"], best_overall["gains"]["Kd_xy"]], dtype=np.float32),
        scene_config=np.array(json.dumps(PRIMARY_SCENE_CONFIG)),
        success=np.array(final_result["success"]),
        insertion_depth=np.array(final_result["insertion_depth"], dtype=np.float32),
        reward=np.array(final_result["reward"], dtype=np.float32),
        best_reward_per_gen=np.array(best_reward_per_gen, dtype=np.float32),
    )
    print(f"[cma_search] seed_trajectory 저장: {args.out_path}")

    # 수렴 곡선 (세대별 최고 리워드)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(best_reward_per_gen) + 1), best_reward_per_gen, marker="o", markersize=3)
        plt.axhline(args.threshold, color="r", linestyle="--", label=f"threshold={args.threshold}")
        plt.xlabel("generation")
        plt.ylabel("best reward (this generation)")
        plt.title("CMA-ES convergence")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.curve_path, dpi=120)
        print(f"[cma_search] 수렴 곡선 저장: {args.curve_path}")
    except ImportError:
        pass

    print("[cma_search] 세대별 최고 리워드:", [round(r, 2) for r in best_reward_per_gen])


if __name__ == "__main__":
    main()
