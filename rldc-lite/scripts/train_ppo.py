"""PegInHoleEnv에서 PPO(stable-baselines3)를 학습시키는 스크립트.

Colab 세션 끊김에 대비해 CheckpointCallback으로 주기적으로 저장하고,
--resume으로 저장된 체크포인트에서 이어서 학습할 수 있다.

사용 예:
    python scripts/train_ppo.py --n-envs 4 --total-timesteps 2_000_000 \
        --save-dir /content/drive/MyDrive/rldc-lite/checkpoints

    # 이어서 학습:
    python scripts/train_ppo.py --n-envs 4 --total-timesteps 2_000_000 \
        --save-dir /content/drive/MyDrive/rldc-lite/checkpoints \
        --resume /content/drive/MyDrive/rldc-lite/checkpoints/ppo_peg_in_hole_1000000_steps.zip
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.peg_in_hole_env import PegInHoleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="병렬 환경 개수 (SubprocVecEnv). Colab 무료 등급은 보통 CPU 코어가 2개뿐이므로 "
        "기본값은 os.cpu_count()와 4 중 작은 값으로 잡는다. 코어보다 크게 잡으면 오히려 느려질 수 있다.",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="체크포인트 저장 경로. Colab에서는 Google Drive 마운트 경로를 넘기면 된다 "
        "(예: /content/drive/MyDrive/rldc-lite/checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="이 timestep 간격마다(전체 스텝 기준, n_envs로 자동 환산) 체크포인트를 저장한다.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="이어서 학습할 체크포인트(.zip) 경로. 지정하지 않으면 새로 학습을 시작한다.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout n_steps (env당)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def make_env(rank: int, seed: int, max_episode_steps: int):
    def _init():
        env = PegInHoleEnv(max_episode_steps=max_episode_steps, domain_randomization=True)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env_fns = [make_env(i, args.seed, args.max_episode_steps) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    # CheckpointCallback의 save_freq는 "callback 호출 횟수" 기준이고, 벡터 환경에서는
    # 한 번 호출될 때마다 n_envs 스텝씩 진행되므로 전체 timestep 기준 주기를 맞추려면 나눠줘야 한다.
    save_freq = max(args.checkpoint_freq // args.n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=args.save_dir,
        name_prefix="ppo_peg_in_hole",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    if args.resume:
        print(f"[train_ppo] 체크포인트에서 이어서 학습: {args.resume}")
        model = PPO.load(args.resume, env=vec_env)
        reset_num_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            verbose=1,
        )
        reset_num_timesteps = True

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps,
    )

    final_path = os.path.join(args.save_dir, "ppo_peg_in_hole_final")
    model.save(final_path)
    print(f"[train_ppo] 최종 모델 저장: {final_path}.zip")

    vec_env.close()


if __name__ == "__main__":
    main()
