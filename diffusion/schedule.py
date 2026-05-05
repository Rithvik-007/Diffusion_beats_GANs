"""
Noise schedules for the DDPM diffusion process.

Provides linear and cosine beta schedules with all precomputed quantities
needed for forward/reverse diffusion.

Reference:
  - Ho et al., "Denoising Diffusion Probabilistic Models" (2020) — linear schedule
  - Nichol & Dhariwal, "Improved DDPM" (2021) — cosine schedule
"""

import torch
import numpy as np


def linear_schedule(timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> dict:
    """
    Linear beta schedule from DDPM.

    Args:
        timesteps: Number of diffusion steps T
        beta_start: Starting noise level
        beta_end: Ending noise level

    Returns:
        Dictionary of precomputed schedule tensors
    """
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    return _compute_schedule(betas)


def cosine_schedule(timesteps: int = 1000, s: float = 0.008) -> dict:
    """
    Cosine beta schedule from "Improved DDPM" (Nichol & Dhariwal).

    Produces a smoother noise schedule that avoids the sharp noise increase
    at the end of the linear schedule.

    Args:
        timesteps: Number of diffusion steps T
        s: Small offset to prevent beta from being too small near t=0

    Returns:
        Dictionary of precomputed schedule tensors
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alpha_bars = torch.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]  # Normalize so alpha_bar_0 = 1

    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    betas = torch.clamp(betas, min=0, max=0.999)  # Clip for numerical stability

    return _compute_schedule(betas)


def _compute_schedule(betas: torch.Tensor) -> dict:
    """
    Precompute all quantities needed for DDPM forward/reverse processes.

    Args:
        betas: (T,) tensor of noise levels

    Returns:
        Dictionary with all precomputed tensors
    """
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alpha_bars[:-1]])

    # Forward process quantities
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    # Reverse process quantities
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_log_variance = torch.log(
        torch.clamp(posterior_variance, min=1e-20)
    )
    posterior_mean_coeff1 = torch.sqrt(alpha_bars_prev) * betas / (1.0 - alpha_bars)
    posterior_mean_coeff2 = torch.sqrt(alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    return {
        "betas": betas.float(),
        "alphas": alphas.float(),
        "alpha_bars": alpha_bars.float(),
        "alpha_bars_prev": alpha_bars_prev.float(),
        "sqrt_alpha_bars": sqrt_alpha_bars.float(),
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars.float(),
        "sqrt_recip_alphas": sqrt_recip_alphas.float(),
        "posterior_variance": posterior_variance.float(),
        "posterior_log_variance": posterior_log_variance.float(),
        "posterior_mean_coeff1": posterior_mean_coeff1.float(),
        "posterior_mean_coeff2": posterior_mean_coeff2.float(),
    }


def get_schedule(name: str, timesteps: int = 1000) -> dict:
    """
    Get a noise schedule by name.

    Args:
        name: "linear" or "cosine"
        timesteps: Number of diffusion steps

    Returns:
        Dictionary of precomputed schedule tensors
    """
    if name == "linear":
        return linear_schedule(timesteps)
    elif name == "cosine":
        return cosine_schedule(timesteps)
    else:
        raise ValueError(f"Unknown schedule: {name}. Choose 'linear' or 'cosine'.")
