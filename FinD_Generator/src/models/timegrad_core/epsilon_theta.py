"""TimeGrad epsilon network implemented purely in PyTorch.

The architecture mirrors the diffusion denoiser used in the original
TimeGrad implementation: a stack of dilated causal convolutions with residual
and skip connections, conditioned on a sinusoidal time embedding.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal time embeddings as in the original DDPM paper."""

    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad if needed
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dilated_conv = nn.Conv1d(
            channels, 2 * channels, kernel_size, padding=padding, dilation=dilation
        )
        self.time_proj = nn.Linear(time_emb_dim, 2 * channels)
        self.residual = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dilated_conv(x)
        time_part = self.time_proj(time_emb).unsqueeze(-1)
        h = h + time_part

        gate, filter = torch.chunk(h, 2, dim=1)
        h = torch.tanh(filter) * torch.sigmoid(gate)

        residual_out = self.residual(h) + x
        skip_out = self.skip(h)
        return residual_out, skip_out


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        input_size: int,
        prediction_length: int,
        residual_layers: int = 6,
        residual_channels: int = 32,
        kernel_size: int = 3,
        time_emb_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.prediction_length = prediction_length

        self.input_projection = nn.Conv1d(input_size, residual_channels, kernel_size=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        dilations: List[int] = [2 ** i for i in range(residual_layers)]
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=residual_channels,
                    time_emb_dim=time_emb_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                )
                for d in dilations
            ]
        )

        self.output_projection = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(residual_channels, residual_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(residual_channels, input_size, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected x with shape [B, C, T], got {tuple(x.shape)}"
            )

        if t.dim() != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(
                f"Expected t with shape [B], got {tuple(t.shape)} for batch {x.shape[0]}"
            )

        # Project inputs
        h = self.input_projection(x)

        # Time embedding
        time_emb = sinusoidal_time_embedding(t, self.time_mlp[0].in_features)
        time_emb = self.time_mlp(time_emb)

        skip_connections = []
        for block in self.residual_blocks:
            h, skip = block(h, time_emb)
            skip_connections.append(skip)

        h = sum(skip_connections)
        h = self.output_projection(h)

        # Ensure output matches input shape
        if h.shape[-1] != self.prediction_length:
            h = F.interpolate(h, size=self.prediction_length, mode="nearest")

        return h
