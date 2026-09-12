"""ARD-Gen 2단계(부트스트래핑) 1/3: seed 게인으로 시연 데이터를 모은다.

evaluate_generalization.py에서 검증한 admittance controller 게인으로
sim.sample_scene_config()가 뽑는 무작위 시나리오들을 실행하고, 성공한
에피소드들의 (force 상태, 행동) 스텝별 쌍을 모아 demonstrations.npz로
저장한다. 이 데이터가 train_policy.py의 행동 복제(behavior cloning)
학습 데이터가 된다.

state(obs) = [force_error_x, force_error_y, d_force_error_x, d_force_error_y]
action     = [delta_x, delta_y]  (admittance controller가 실제로 적용한 값)

_run_episode_with_sim()이 이미 매 스텝의 forces/actions를 기록해서
반환하므로, 새로 시뮬레이션 루프를 짜는 대신 그 결과에서 상태를 재구성한다
(force_error_xy = forces[:, :2], d_force_error_xy는 유한차분으로 재계산 —
admittance controller가 실제로 썼던 것과 동일한 값이 나온다).

사용 예:
    python bootstrap/collect_demonstrations.py --seed-path ../seed_trajectory.npz \
        --n-episodes 300 --seed 7 --out-path ./demonstrations.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.peg_in_hole_sim import DT, PegInHoleSim, _run_episode_with_sim, sample_scene_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=str, default="../seed_trajectory.npz")
    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7, help="시나리오 샘플링용 RNG 시드")
    parser.add_argument("--out-path", type=str, default="./demonstrations.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.seed_path, allow_pickle=False)
    gains = {"Kp_xy": float(data["gains"][0]), "Kd_xy": float(data["gains"][1])}
    print(f"[collect_demonstrations] admittance 게인: {gains}")

    rng = np.random.default_rng(args.seed)
    sim = PegInHoleSim()

    obs_chunks = []
    act_chunks = []
    n_success = 0

    for ep in range(args.n_episodes):
        cfg = sample_scene_config(rng)
        result = _run_episode_with_sim(sim, gains, cfg)
        if not result["success"]:
            continue  # 실패(잼밍 등) 에피소드는 나쁜 시연이므로 제외
        n_success += 1

        forces_xy = result["forces"][:, :2]
        prev = np.vstack([np.zeros((1, 2), dtype=np.float32), forces_xy[:-1]])
        d_forces_xy = (forces_xy - prev) / DT
        obs = np.concatenate([forces_xy, d_forces_xy], axis=1)  # (T, 4)
        actions_xy = result["actions"][:, :2]  # (T, 2)

        obs_chunks.append(obs)
        act_chunks.append(actions_xy)

    obs_arr = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    act_arr = np.concatenate(act_chunks, axis=0).astype(np.float32)

    print(
        f"[collect_demonstrations] episodes={args.n_episodes} success={n_success} "
        f"({n_success / args.n_episodes:.1%})  steps_collected={len(obs_arr)}"
    )

    np.savez(args.out_path, obs=obs_arr, actions=act_arr, gains=np.array([gains["Kp_xy"], gains["Kd_xy"]]))
    print(f"[collect_demonstrations] 저장: {args.out_path}")


if __name__ == "__main__":
    main()
