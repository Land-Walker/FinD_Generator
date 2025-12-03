"""Conditioned wrapper around the base TimeGrad epsilon network."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..timegrad_core.epsilon_theta import EpsilonTheta


class ConditionedEpsilonTheta(nn.Module):
    """Injects dynamic and static conditioning into ``EpsilonTheta``.

    The design keeps the base network untouched, forwarding a conditioned
    tensor while preserving interface compatibility with ``GaussianDiffusion``.
    """

    def __init__(
        self,
        base_epsilon: EpsilonTheta,
        cond_dynamic_dim: int,
        cond_static_dim: int,
        seq_len: int,
        prediction_length: int,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()

        self.base = base_epsilon
        self.seq_len = seq_len
        self.prediction_length = prediction_length

        self.dynamic_encoder = nn.Sequential(
            nn.Linear(cond_dynamic_dim * seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(cond_static_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.cond_project = nn.Linear(embed_dim * 2, prediction_length)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Dict[str, torch.Tensor]):
        """Forward pass with conditioning.

        Args:
            x: Noisy input, shape ``[B, C, horizon]``.
            t: Diffusion timestep tensor ``[B]``.
            cond: Dict with keys ``"dynamic"`` -> ``[B, seq_len, cond_dim]`` and
                ``"static"`` -> ``[B, static_dim]``.
        """

        batch_size = x.size(0)

        dyn = cond["dynamic"].reshape(batch_size, -1)
        dyn_emb = self.dynamic_encoder(dyn)

        static_emb = self.static_encoder(cond["static"])

        joint = torch.cat([dyn_emb, static_emb], dim=-1)
        cond_vec = self.cond_project(joint)

        # Expand to match x's channel dimension
        cond_vec = cond_vec.unsqueeze(1)  # [B, 1, horizon]

        x_cond = x + cond_vec

        return self.base(x_cond, t)
