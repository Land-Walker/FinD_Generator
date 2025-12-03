"""Faithful PyTorch clone of TimeGrad (without GluonTS/PTS estimator wrappers).

The module wraps ``GaussianDiffusion`` and ``EpsilonTheta`` in a minimal
``nn.Module`` so it can be integrated into standard PyTorch training loops.
Replace the placeholder implementations in ``gaussian_diffusion.py`` and
``epsilon_theta.py`` with the originals from PTS to obtain a working model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gaussian_diffusion import GaussianDiffusion
from .epsilon_theta import EpsilonTheta


class TimeGradBase(nn.Module):
    """Pure-PyTorch faithful clone of TimeGrad.

    Args:
        target_dim: Number of target variables.
        prediction_length: Forecast horizon (time dimension of x_future).
        diff_steps: Number of diffusion steps.
        beta_end: Final beta value for the diffusion schedule.
        beta_schedule: Schedule name (e.g., "linear").
        residual_layers: Number of residual blocks in EpsilonTheta.
        residual_channels: Channels in each residual block.
    """

    def __init__(
        self,
        target_dim: int,
        prediction_length: int,
        diff_steps: int = 100,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
        residual_layers: int = 6,
        residual_channels: int = 32,
    ) -> None:
        super().__init__()

        self.target_dim = target_dim
        self.prediction_length = prediction_length

        self.epsilon_theta = EpsilonTheta(
            input_size=target_dim,
            prediction_length=prediction_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
        )

        self.diffusion = GaussianDiffusion(
            denoise_fn=self.epsilon_theta,
            input_size=target_dim,
            diff_steps=diff_steps,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

    def forward(self, x_future: torch.Tensor) -> torch.Tensor:
        """Compute diffusion loss for the provided future window.

        Args:
            x_future: Tensor shaped ``[B, horizon, target_dim]``.

        Returns:
            Scalar diffusion loss tensor.
        """
        # Convert to channel-first format expected by the diffusion model.
        x = x_future.transpose(1, 2)  # -> [B, target_dim, horizon]

        # Compute diffusion loss
        loss = self.diffusion.p_losses(x_start=x)

        return loss
