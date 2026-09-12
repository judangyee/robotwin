"""ARD-Gen 2단계(부트스트래핑) 2/3: 시연 데이터로 정책을 행동 복제(behavior
cloning)한다.

collect_demonstrations.py가 모은 (force 상태 -> admittance controller가 낸
행동) 쌍으로 작은 MLP를 지도학습시킨다. 목표는 손으로 짠 PD 공식
(Δx = -Kp*F_error - Kd*dF_error/dt)을 신경망이 데이터만 보고 재현할 수
있는지 확인하는 것 — 이게 되면 이 네트워크를 이후 RL 파인튜닝(PPO 등)의
초기 정책으로 쓸 수 있다(ARD-Gen 2단계의 취지).

사용 예:
    python bootstrap/train_policy.py --data-path ./demonstrations.npz \
        --out-path ./policy.pt --epochs 200
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn


class AdmittancePolicy(nn.Module):
    """obs=[Fx, Fy, dFx/dt, dFy/dt] -> action=[dx, dy].

    모든 Linear에 bias=False를 주고 Tanh(홀함수, tanh(0)=0)만 활성화로 써서,
    입력이 정확히 0이면 출력도 정확히 0이 되도록 구조적으로 보장한다(f(0)=0).
    이게 왜 중요한지는 bootstrap/README 참고 — 이 컨트롤러는 매 스텝 ctrl에
    행동을 "누적"하므로, 힘이 0인데도 출력이 미세하게 0이 아니면(bias 학습)
    그 작은 오차가 수백 스텝에 걸쳐 적분(누적)되어 치명적인 위치 drift가
    된다. 실제로 최초 버전(bias 있는 일반 MLP + 평균-중심화 정규화)에서
    이 문제로 성공률이 99%->0.5%로 붕괴하는 걸 확인했다."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, hidden, bias=False),
            nn.Tanh(),
            nn.Linear(hidden, 2, bias=False),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default="./demonstrations.npz")
    parser.add_argument("--out-path", type=str, default="./policy.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data = np.load(args.data_path)
    obs = data["obs"]
    actions = data["actions"]

    # 관측값 스케일이 서로 크게 달라서(힘 vs 힘의 변화율) 정규화해야 학습이
    # 안정적으로 된다. 단, 평균을 빼면(mean-centering) "힘=0"이 정규화된
    # 공간에서 0이 아닌 값으로 옮겨가 버려서, f(0)=0 구조를 갖는 네트워크의
    # 의미가 사라진다 -> 표준편차로 스케일만 하고 평균은 빼지 않는다.
    obs_mean = np.zeros_like(obs.mean(axis=0))
    obs_std = obs.std(axis=0) + 1e-6
    obs_norm = obs / obs_std

    n = len(obs_norm)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    obs_t = torch.from_numpy(obs_norm.astype(np.float32))
    act_t = torch.from_numpy(actions.astype(np.float32))

    train_obs, train_act = obs_t[train_idx], act_t[train_idx]
    val_obs, val_act = obs_t[val_idx], act_t[val_idx]

    policy = AdmittancePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n_train = len(train_obs)
    for epoch in range(1, args.epochs + 1):
        perm_epoch = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, args.batch_size):
            idx = perm_epoch[start : start + args.batch_size]
            pred = policy(train_obs[idx])
            loss = loss_fn(pred, train_act[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        if epoch % 20 == 0 or epoch == 1:
            with torch.no_grad():
                val_loss = loss_fn(policy(val_obs), val_act).item()
            print(f"[train_policy] epoch {epoch:4d}: train_mse={epoch_loss:.4e}  val_mse={val_loss:.4e}")

    with torch.no_grad():
        final_val_loss = loss_fn(policy(val_obs), val_act).item()
    print(f"[train_policy] 최종 val_mse={final_val_loss:.4e} (n_train={n_train}, n_val={n_val})")

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
        },
        args.out_path,
    )
    print(f"[train_policy] 저장: {args.out_path}")


if __name__ == "__main__":
    main()
