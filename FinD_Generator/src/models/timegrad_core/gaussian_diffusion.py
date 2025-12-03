"""Gaussian diffusion implementation for TimeGrad-style models.

This module is adapted to be a faithful, self-contained PyTorch version of the
PTS implementation used by TimeGrad. It supports training-time loss
computation with optional conditioning routed through the provided denoising
network.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def _linear_beta_schedule(diff_steps: int, beta_end: float) -> torch.Tensor:
    """Create a linear beta schedule from a small starting value to ``beta_end``.

    The starting value is fixed to 1e-4 to mirror the default DDPM schedule used
    by the original TimeGrad codebase, while the end value remains configurable
    for experimentation.
    """

    return torch.linspace(1e-4, beta_end, diff_steps, dtype=torch.float32)


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract coefficients for a batch of indices ``t`` and reshape for broadcast."""

    out = a.gather(-1, t.to(a.device))
    return out.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        input_size: int,
        diff_steps: int = 100,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
    ) -> None:
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.diff_steps = diff_steps

        if beta_schedule != "linear":
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        betas = _linear_beta_schedule(diff_steps, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    @torch.no_grad()
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diffuse the data for a given number of timesteps ``t``.

        Args:
            x_start: Clean input of shape ``[B, C, T]``.
            t: Timesteps tensor of shape ``[B]``.
            noise: Optional pre-sampled noise; if ``None`` standard normal noise
                is drawn.
        """

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute the diffusion training loss.

        This follows the original DDPM objective used by TimeGrad: sample a
        random timestep ``t``, corrupt the input with Gaussian noise, and train
        the denoiser to predict that noise.
        """

        if x_start.dim() != 3:
            raise ValueError(
                f"Expected x_start with shape [B, C, T], got {tuple(x_start.shape)}"
            )

        batch_size = x_start.size(0)
        device = x_start.device

        t = torch.randint(0, self.diff_steps, (batch_size,), device=device)
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Support conditioned denoisers by forwarding the conditioning dict when provided.
        if cond is None:
            pred_noise = self.denoise_fn(x_noisy, t)
        else:
            pred_noise = self.denoise_fn(x_noisy, t, cond)

        if pred_noise.shape != noise.shape:
            raise RuntimeError(
                "Denoiser output shape does not match noise shape: "
                f"pred={tuple(pred_noise.shape)} vs noise={tuple(noise.shape)}"
            )

        loss = nn.functional.mse_loss(pred_noise, noise)
        return loss
