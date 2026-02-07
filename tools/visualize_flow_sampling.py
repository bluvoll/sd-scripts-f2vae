#!/usr/bin/env python3
"""
Quick visualization of Rectified Flow timestep (sigma) sampling and optional loss weighting.

This mirrors diffusion-trainer/utils/sampler_utils.py so you can see how different configs
shape the sampled u/sigma distribution and the weight curves.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def _load_sampler_utils():
    """Load diffusion-trainer/utils/sampler_utils.py without requiring installation."""
    repo_root = Path(__file__).resolve().parents[1]
    sampler_utils_path = repo_root / "diffusion-trainer" / "utils" / "sampler_utils.py"
    if not sampler_utils_path.exists():
        raise FileNotFoundError(f"Cannot find sampler_utils.py at {sampler_utils_path}")
    spec = importlib.util.spec_from_file_location("dt_sampler_utils", sampler_utils_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sampler_utils = _load_sampler_utils()
get_flowmatch_inputs = sampler_utils.get_flowmatch_inputs
get_loss_weighting = sampler_utils.get_loss_weighting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Rectified Flow sampling and loss weighting.")
    parser.add_argument("--samples", type=int, default=16384, help="Number of samples to draw.")
    parser.add_argument("--num-train-timesteps", type=int, default=1000, help="Total training steps used for scaling.")
    parser.add_argument(
        "--weighting-scheme",
        type=str,
        default="mode",
        choices=["uniform", "logit_normal", "lognorm", "mode"],
        help="Sampler weighting scheme (see sampler_utils.py).",
    )
    parser.add_argument("--logit-mean", type=float, default=0, help="Mean for logit_normal sampling.")
    parser.add_argument("--logit-std", type=float, default=1, help="Std for logit_normal sampling.")
    parser.add_argument("--mode-scale", type=float, default=0.64, help="Mode scaling for 'mode' sampling.")
    parser.add_argument("--shift", type=float, default=2.5, help="Uniform shift applied to u/sigma.")
    parser.add_argument("--timestep-bias", type=float, default=0.5, help="Bias added to u (scaled by 1/num_train_timesteps).")
    parser.add_argument(
        "--loss-weighting",
        type=str,
        default="none",
        choices=["none", "sigma", "sigma_pi", "sigma_sqrt_clamp", "sigma_sqrt", "cosmap"],
        help="Optional loss weighting mode to plot vs sigma.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def sample_sigmas(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Latent shape kept tiny to avoid memory pressure; only the batch dimension matters for sampling u/sigma.
    latents = torch.zeros((args.samples, 4, 1, 1), device=device, dtype=torch.float32)
    sampler_config = {
        "weighting_scheme": args.weighting_scheme,
        "logit_mean": args.logit_mean,
        "logit_std": args.logit_std,
        "mode_scale": args.mode_scale,
        "shift": args.shift,
        "timestep_bias": args.timestep_bias,
    }
    _, timesteps, _, sigmas, _ = get_flowmatch_inputs(
        latents=latents,
        device=device,
        sampler_config=sampler_config,
        num_train_timesteps=args.num_train_timesteps,
    )
    u = sigmas.view(-1).detach().cpu()
    t = timesteps.detach().cpu()
    return u, t


def compute_weights(sigmas_1d: torch.Tensor, mode: str) -> torch.Tensor:
    sigmas_4d = sigmas_1d.view(-1, 1, 1, 1)
    ones = torch.ones_like(sigmas_4d)
    weighted_pred, _ = get_loss_weighting(mode, ones, ones, sigmas_4d)
    return weighted_pred.view(-1).detach().cpu()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    u, timesteps = sample_sigmas(args, device)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(u.numpy(), bins=100, density=True, color="#4c72b0")
    axes[0].set_title(f"u / sigma distribution ({args.weighting_scheme})")
    axes[0].set_xlabel("u (sigma)")
    axes[0].set_ylabel("density")

    axes[1].hist(timesteps.numpy(), bins=100, density=True, color="#dd8452")
    axes[1].set_title("timesteps (float)")
    axes[1].set_xlabel("t")

    if args.loss_weighting != "none":
        u_sorted, _ = torch.sort(u)
        weights = compute_weights(u_sorted, args.loss_weighting)
        axes[2].plot(u_sorted.numpy(), weights.numpy(), color="#55a868")
        axes[2].set_title(f"loss weighting vs sigma ({args.loss_weighting})")
        axes[2].set_xlabel("sigma")
        axes[2].set_ylabel("weight")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "loss weighting: none", ha="center", va="center")

    fig.suptitle("Rectified Flow sampling overview", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
