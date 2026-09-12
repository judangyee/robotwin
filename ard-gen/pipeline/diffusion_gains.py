"""ARD-Gen 2-B단계: conditional diffusion으로 씬 조건에 맞는 게인을 생성한다.

2-A단계(pipeline/bootstrap.py)가 모은 bootstrap_dataset.npz(seed 게인 주변
무작위 노이즈로 1000회 실행한 기록)를 학습 데이터로 써서, "씬 조건이
주어지면 성공 확률 높은 (Kp_xy, Kd_xy)를 생성하는" conditional diffusion
모델을 학습한다. 학습 후에는 새로운 무작위 씬 100개에 대해 실제로
시뮬레이션을 돌려서, 2-A단계의 무작위 노이즈 방식(성공률 76%)보다
diffusion이 뽑은 게인이 더 잘 통하는지 직접 검증한다.

## 학습 데이터: 성공 샘플만 쓴다 (실패 샘플은 조건 정규화 통계에만 씀)

이유:
1. 우리가 실제로 원하는 건 "주어진 씬에서 성공하는 게인의 분포"를 모델링해서
   거기서 샘플링하는 것이다. Diffusion은 학습 데이터의 분포를 그대로
   재현하도록 배우므로, 성공/실패가 섞인 데이터로 학습하면 모델이 "실패하는
   게인"도 똑같이 그럴듯하게 생성하게 된다 — 우리가 원하는 게 아니다.
2. Classifier guidance(실패도 학습해서 그쪽을 피하도록 유도)는 별도의
   성공/실패 분류기 + guidance 가중치 튜닝이 필요해서, 저차원(조건 7차원,
   출력 2차원) 문제에 비해 과한 복잡도다. 이 스케일에서는 "성공 사례만
   보고 그 분포를 재현"하는 게 더 간단하고 안정적이다.
3. 표본 크기 문제도 크지 않다 — bootstrap_dataset.npz의 성공률이 76%라
   1000개 중 약 760개가 성공 샘플로 남는데, 조건 7차원/출력 2차원짜리
   저차원 회귀형 생성 문제에는 충분한 양이다.
4. 다만 조건(scene_config) 자체의 정규화 통계(평균/표준편차)는 성공 여부와
   무관하게 **전체 1000개**로 계산한다 — 성공 샘플만으로 계산하면 애초에
   "쉬운 씬"쪽으로 치우친 통계가 나와서, 실전에서 마주칠 어려운 씬(좁은
   clearance, 큰 오프셋)의 조건 벡터가 정규화 공간에서 이상한 위치로
   밀려날 수 있다.

## 모델

조건(7차원: hole_pose 3 + friction 1 + clearance_m 1 + peg_init_offset 2)과
diffusion timestep을 함께 받아 노이즈(eps)를 예측하는 작은 MLP
(GainDiffusionNet) + 표준 DDPM 스케줄(GaussianDiffusion). 게인이 2차원뿐이라
대형 U-Net 등은 불필요하다.

사용 예:
    python pipeline/diffusion_gains.py \
        --dataset-path ./data/bootstrap/bootstrap_dataset.npz \
        --out-path ./data/bootstrap/diffusion_gains.pt \
        --epochs 500 --n-eval 100 --eval-seed 999
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from pipeline.scene_sampler import sample_scene_config, to_sim_scene_config
from sim.peg_in_hole_sim import run_episode

_COND_DIM = 7  # hole_pose(3) + friction(1) + clearance_m(1) + peg_init_offset(2)
_GAIN_DIM = 2  # Kp_xy, Kd_xy

# optimize/cma_search.py의 CMA-ES 탐색 범위와 동일 -- diffusion이 뽑은
# 샘플이 물리적으로 말이 안 되는 값(음수 게인 등)으로 튀는 걸 막기 위한 clip.
_KP_BOUNDS = (0.00002, 0.003)
_KD_BOUNDS = (0.0, 0.0005)


# ---------------------------------------------------------------------------
# 조건 벡터 조립
# ---------------------------------------------------------------------------
def _condition_from_arrays(
    hole_pose: np.ndarray, friction: np.ndarray, clearance_m: np.ndarray, peg_init_offset: np.ndarray
) -> np.ndarray:
    """bootstrap_dataset.npz의 컬럼들(N,...)을 (N, 7) 조건 행렬로 합친다."""
    return np.concatenate(
        [hole_pose, friction[:, None], clearance_m[:, None], peg_init_offset], axis=1
    ).astype(np.float32)


def _condition_from_scene_config(cfg: dict) -> np.ndarray:
    """pipeline.scene_sampler.sample_scene_config()가 반환하는 dict 하나를
    (7,) 조건 벡터로 변환한다 (추론 시 사용)."""
    hole_pose = np.array(cfg["hole_pose"], dtype=np.float32)
    peg_offset = np.array(cfg["peg_init_offset"], dtype=np.float32)
    return np.concatenate([hole_pose, [cfg["friction"]], [cfg["clearance_m"]], peg_offset]).astype(np.float32)


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GainDiffusionNet(nn.Module):
    """(noisy_gains, condition, t) -> 예측 노이즈(eps). 조건부 score network."""

    def __init__(self, cond_dim: int = _COND_DIM, gain_dim: int = _GAIN_DIM, hidden: int = 128, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.Mish(), nn.Linear(time_dim, time_dim))
        self.net = nn.Sequential(
            nn.Linear(gain_dim + cond_dim + time_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, gain_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))
        h = torch.cat([x, cond, t_embed], dim=-1)
        return self.net(h)


class GaussianDiffusion:
    """표준 DDPM(선형 beta 스케줄) forward(q_sample)/reverse(p_sample_loop)."""

    def __init__(self, timesteps: int = 100, device: str = "cpu"):
        self.T = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.device = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ac = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_1mac = (1.0 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return sqrt_ac * x0 + sqrt_1mac * noise

    @torch.no_grad()
    def p_sample_loop(self, model: GainDiffusionNet, cond: torch.Tensor, gain_dim: int = _GAIN_DIM) -> torch.Tensor:
        batch = cond.shape[0]
        x = torch.randn(batch, gain_dim, device=self.device)
        for t_step in reversed(range(self.T)):
            t = torch.full((batch,), t_step, dtype=torch.long, device=self.device)
            eps_pred = model(x, cond, t)
            alpha = self.alphas[t_step]
            alpha_cumprod = self.alphas_cumprod[t_step]
            beta = self.betas[t_step]
            mean = (1.0 / alpha.sqrt()) * (x - (beta / (1.0 - alpha_cumprod).sqrt()) * eps_pred)
            if t_step > 0:
                noise = torch.randn_like(x)
                x = mean + beta.sqrt() * noise
            else:
                x = mean
        return x


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    data = np.load(args.dataset_path)

    cond_all = _condition_from_arrays(data["hole_pose"], data["friction"], data["clearance_m"], data["peg_init_offset"])
    gains_all = np.stack([data["kp_xy"], data["kd_xy"]], axis=1).astype(np.float32)
    success = data["success"]

    print(f"[diffusion_gains] 데이터셋: {len(success)}건, 성공 {success.sum()}건 ({success.mean():.1%})")

    # 조건 정규화 통계는 전체(성공+실패)로 계산 -- docstring 참고.
    cond_mean = cond_all.mean(axis=0)
    cond_std = cond_all.std(axis=0) + 1e-6

    gains_succ = gains_all[success]
    cond_succ = cond_all[success]

    gain_mean = gains_succ.mean(axis=0)
    gain_std = gains_succ.std(axis=0) + 1e-8

    cond_norm = (cond_succ - cond_mean) / cond_std
    gains_norm = (gains_succ - gain_mean) / gain_std

    cond_t = torch.from_numpy(cond_norm.astype(np.float32))
    gains_t = torch.from_numpy(gains_norm.astype(np.float32))

    model = GainDiffusionNet()
    diffusion = GaussianDiffusion(timesteps=args.timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = len(gains_t)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            x0 = gains_t[idx]
            cond = cond_t[idx]
            t = torch.randint(0, diffusion.T, (len(idx),))
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise)
            eps_pred = model(x_t, cond, t)
            loss = nn.functional.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        if epoch % 50 == 0 or epoch == 1:
            print(f"[diffusion_gains] epoch {epoch:4d}: loss={epoch_loss:.4f}")

    return {
        "model": model,
        "diffusion": diffusion,
        "cond_mean": cond_mean,
        "cond_std": cond_std,
        "gain_mean": gain_mean,
        "gain_std": gain_std,
        "bootstrap_success_rate": float(success.mean()),
    }


def save_checkpoint(state: dict, out_path: str, timesteps: int) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": state["model"].state_dict(),
            "cond_mean": state["cond_mean"],
            "cond_std": state["cond_std"],
            "gain_mean": state["gain_mean"],
            "gain_std": state["gain_std"],
            "timesteps": timesteps,
        },
        out_path,
    )
    print(f"[diffusion_gains] 저장: {out_path}")


def load_checkpoint(path: str) -> tuple[GainDiffusionNet, GaussianDiffusion, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ckpt = torch.load(path, weights_only=False)
    model = GainDiffusionNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    diffusion = GaussianDiffusion(timesteps=ckpt["timesteps"])
    return model, diffusion, ckpt["cond_mean"], ckpt["cond_std"], ckpt["gain_mean"], ckpt["gain_std"]


def sample_gains(
    model: GainDiffusionNet,
    diffusion: GaussianDiffusion,
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
    gain_mean: np.ndarray,
    gain_std: np.ndarray,
    scene_cfg: dict,
) -> dict[str, float]:
    """씬 조건 하나에 대해 diffusion으로 게인 하나를 샘플링한다."""
    cond = _condition_from_scene_config(scene_cfg)
    cond_norm = (cond - cond_mean) / cond_std
    cond_tensor = torch.from_numpy(cond_norm.astype(np.float32)).unsqueeze(0)

    sample = diffusion.p_sample_loop(model, cond_tensor)
    gains_norm = sample.squeeze(0).numpy()
    gains = gains_norm * gain_std + gain_mean

    kp = float(np.clip(gains[0], *_KP_BOUNDS))
    kd = float(np.clip(gains[1], *_KD_BOUNDS))
    return {"Kp_xy": kp, "Kd_xy": kd}


# ---------------------------------------------------------------------------
# 검증: 새 무작위 씬에서 실제로 시뮬레이션을 돌려 성공률을 잰다.
# ---------------------------------------------------------------------------
def evaluate(
    model: GainDiffusionNet,
    diffusion: GaussianDiffusion,
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
    gain_mean: np.ndarray,
    gain_std: np.ndarray,
    n_eval: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    successes = []
    kp_samples = []
    kd_samples = []

    for _ in range(n_eval):
        scene_cfg = sample_scene_config(rng)
        gains = sample_gains(model, diffusion, cond_mean, cond_std, gain_mean, gain_std, scene_cfg)
        sim_cfg = to_sim_scene_config(scene_cfg)
        result = run_episode(gains, sim_cfg)

        successes.append(result["success"])
        kp_samples.append(gains["Kp_xy"])
        kd_samples.append(gains["Kd_xy"])

    successes = np.array(successes)
    kp_samples = np.array(kp_samples)
    kd_samples = np.array(kd_samples)

    return {
        "n_eval": n_eval,
        "success_rate": float(successes.mean()),
        "n_success": int(successes.sum()),
        "kp_mean": float(kp_samples.mean()),
        "kp_std": float(kp_samples.std()),
        "kd_mean": float(kd_samples.mean()),
        "kd_std": float(kd_samples.std()),
    }


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=str, default="./data/bootstrap/bootstrap_dataset.npz")
    parser.add_argument("--out-path", type=str, default="./data/bootstrap/diffusion_gains.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=100, help="DDPM 스텝 수")
    parser.add_argument("--seed", type=int, default=0, help="학습(torch) 시드")
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=999, help="검증용 씬 샘플링 시드 (학습 데이터 생성 시드=0과 겹치지 않도록)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state = train(args)
    save_checkpoint(state, args.out_path, args.timesteps)

    print()
    print(f"[diffusion_gains] 검증: 새 무작위 씬 {args.n_eval}개 (시드={args.eval_seed})에서 diffusion 게인으로 실제 실행")
    eval_result = evaluate(
        state["model"],
        state["diffusion"],
        state["cond_mean"],
        state["cond_std"],
        state["gain_mean"],
        state["gain_std"],
        args.n_eval,
        args.eval_seed,
    )

    print()
    print("=" * 60)
    print(f"2-A 부트스트래핑(무작위 노이즈) 성공률: {state['bootstrap_success_rate']:.1%}")
    print(
        f"2-B diffusion(조건부 생성) 성공률:      {eval_result['success_rate']:.1%} "
        f"({eval_result['n_success']}/{eval_result['n_eval']})"
    )
    delta = eval_result["success_rate"] - state["bootstrap_success_rate"]
    verdict = "개선됨" if delta > 0 else ("동일" if delta == 0 else "악화됨")
    print(f"차이: {delta:+.1%}p ({verdict})")
    print(
        f"diffusion이 생성한 게인 분포: Kp_xy={eval_result['kp_mean']:.6f}±{eval_result['kp_std']:.6f}  "
        f"Kd_xy={eval_result['kd_mean']:.6e}±{eval_result['kd_std']:.6e}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
