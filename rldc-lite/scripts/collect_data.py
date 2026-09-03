"""학습된 PPO 정책으로 PegInHoleEnv를 롤아웃하며 데이터를 수집한다.

RLDC(Reinforcement Learning as Data Collector) 아이디어의 핵심은 "학습된 정책을
데이터 수집기로 재사용"하는 것이므로, 기본 동작은 성공한 에피소드만 저장한다.
--keep-failures를 주면 실패한 에피소드도 함께 저장한다 (예: 실패 사례 분석, 실패까지
포함한 오프라인 RL 등에 쓰기 위함).

각 에피소드는 하나의 .npz 파일로 저장되며, 나중에 LeRobotDataset 포맷으로 옮길 때
그대로 프레임 단위 필드(ee_pose, action, ...)로 매핑하기 쉽도록 스텝 축이 항상
0번째 축(길이 T)이 되게 저장한다.

사용 예:
    python scripts/collect_data.py --model-path ./checkpoints/ppo_peg_in_hole_final.zip \
        --n-episodes 200 --out-dir ./data/peg_in_hole
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stable_baselines3 import PPO

from envs.peg_in_hole_env import PegInHoleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, required=True, help="학습된 PPO 모델(.zip) 경로")
    parser.add_argument("--n-episodes", type=int, default=100, help="시도할 에피소드 수")
    parser.add_argument("--out-dir", type=str, default="./data/peg_in_hole")
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="지정하면 실패한(성공하지 못한) 에피소드도 저장한다. 기본값은 성공 에피소드만 저장.",
    )
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="정책을 샘플링(확률적)으로 실행. 기본은 --deterministic.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def rollout_episode(env: PegInHoleEnv, model: PPO, deterministic: bool, seed: int):
    obs, _ = env.reset(seed=seed)

    ee_poses, actions, forces, torques, rewards = [], [], [], [], []
    success = False

    for _ in range(env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        ee_pos_before = obs[0:3].copy()
        obs, reward, terminated, truncated, info = env.step(action)

        ee_poses.append(ee_pos_before)
        actions.append(np.asarray(action, dtype=np.float32))
        forces.append(info["force"].astype(np.float32))
        torques.append(info["torque"].astype(np.float32))
        rewards.append(np.float32(reward))

        if info["success"]:
            success = True
        if terminated or truncated:
            break

    episode = {
        "ee_poses": np.stack(ee_poses).astype(np.float32),  # (T, 3)
        "actions": np.stack(actions).astype(np.float32),  # (T, 4)
        "forces": np.stack(forces).astype(np.float32),  # (T, 3)
        "torques": np.stack(torques).astype(np.float32),  # (T, 3)
        "rewards": np.stack(rewards).astype(np.float32),  # (T,)
        "success": np.array(success, dtype=bool),  # scalar
    }
    return episode


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env = PegInHoleEnv(max_episode_steps=args.max_episode_steps, domain_randomization=True)
    model = PPO.load(args.model_path)

    n_saved = 0
    n_success = 0
    for ep_idx in range(args.n_episodes):
        episode = rollout_episode(env, model, args.deterministic, seed=args.seed + ep_idx)
        if episode["success"]:
            n_success += 1

        should_save = bool(episode["success"]) or args.keep_failures
        if should_save:
            out_path = os.path.join(args.out_dir, f"episode_{ep_idx:05d}.npz")
            np.savez(out_path, **episode)
            n_saved += 1

        print(
            f"[collect_data] episode {ep_idx:05d}: success={bool(episode['success'])} "
            f"steps={len(episode['rewards'])} saved={should_save}"
        )

    env.close()
    print(
        f"[collect_data] 완료: {args.n_episodes}개 시도 중 성공 {n_success}개, "
        f"저장 {n_saved}개 -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
