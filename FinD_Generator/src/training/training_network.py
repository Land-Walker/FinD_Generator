"""Training-ready conditional TimeGrad network.

This module wraps ``ConditionalTimeGrad`` with the pieces typically handled by
the GluonTS/PyTorchTS estimator: scale normalization, conditioning prep, and
forecast sampling utilities. It is intentionally self contained so it can be
plugged into a standard PyTorch training loop without any external estimator
machinery.

Design goals for this training network:
- Full compatibility with the cross-attention conditioning stack
- Safe normalization of both targets and conditioning features
- Fast sampling utilities for validation/inference
- Minimal assumptions about the upstream DataLoader (accepts tensors directly)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, StudentT

from src.models.conditional_timegrad import ConditionalTimeGrad


class ConditionalTimeGradTrainingNetwork(nn.Module):
    """High-level training wrapper for the conditional TimeGrad model."""

    def __init__(
        self,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        cond_dynamic_dim: int,
        cond_static_dim: int,
        *,
        diff_steps: int = 100,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
        residual_layers: int = 6,
        residual_channels: int = 32,
        cond_embed_dim: int = 64,
        cond_attn_heads: int = 4,
        cond_attn_dropout: float = 0.1,
        cond_strategy: str = "fast",
        rnn_type: str = "lstm",
        scale_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.scale_eps = scale_eps

        # Keep track of inbound conditioning dimensions before augmentation.
        self.input_cond_dynamic_dim = cond_dynamic_dim
        self.input_cond_static_dim = cond_static_dim
        self.cond_embed_dim = cond_embed_dim

        # History encoder turns past targets into conditioning tokens so the
        # denoiser can leverage cross-attention, relative position bias, causal
        # masking, FiLM modulation, and learned alignment modules downstream.
        self.history_encoder = nn.Sequential(
            nn.Conv1d(
                target_dim,
                cond_embed_dim,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            nn.LeakyReLU(0.4, inplace=True),
            nn.Conv1d(
                cond_embed_dim,
                cond_embed_dim,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            nn.LeakyReLU(0.4, inplace=True),
        )

        # Summary pooling feeds into static FiLM modulation alongside provided
        # static features.
        self.history_pool = nn.AdaptiveAvgPool1d(1)

        # Augment conditioning dims with learned history tokens so the model's
        # cross-attention stack can consume both exogenous signals and encoded
        # past targets.
        combined_cond_dynamic_dim = cond_dynamic_dim + cond_embed_dim
        combined_cond_static_dim = cond_static_dim + cond_embed_dim

        self.model = ConditionalTimeGrad(
            target_dim=target_dim,
            prediction_length=prediction_length,
            seq_len=context_length,
            cond_dynamic_dim=combined_cond_dynamic_dim,
            cond_static_dim=combined_cond_static_dim,
            diff_steps=diff_steps,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            cond_embed_dim=cond_embed_dim,
            cond_attn_heads=cond_attn_heads,
            cond_attn_dropout=cond_attn_dropout,
            cond_strategy=cond_strategy,
            rnn_type=rnn_type,
        )

    def _fit_student_t(self, x_hist: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Fit per-series Student-t parameters from historical windows.
        Args:
            x_hist: Historical target window ``[B, context_length, target_dim]``.

        Returns:
            Tuple of ``(df, loc, scale)`` each shaped ``[B, 1, target_dim]`` suitable for
            broadcasting across time.
        """

        loc = x_hist.mean(dim=1, keepdim=True)
        centered = x_hist - loc
        m2 = centered.pow(2).mean(dim=1, keepdim=True).clamp_min(self.scale_eps)
        m4 = centered.pow(4).mean(dim=1, keepdim=True)

        # Method-of-moments estimate for degrees of freedom via excess kurtosis.
        excess_kurtosis = m4 / (m2.pow(2) + self.scale_eps) - 3.0
        positive_excess = excess_kurtosis.clamp_min(1e-3)
        df = 6.0 / positive_excess + 4.0
        df = torch.where(torch.isfinite(df), df, torch.full_like(df, 30.0))
        df = df.clamp(min=3.0, max=200.0)

        # Variance of Student-t: df/(df-2) * scale^2  -> scale = sqrt(var * (df-2)/df)
        scale = torch.sqrt(m2 * (df - 2.0) / df).clamp_min(self.scale_eps)
        return df, loc, scale

    def _to_gaussian(
        self, x: torch.Tensor, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Map Student-t returns to Gaussian space via CDF and probit."""

        student = StudentT(df=df, loc=loc, scale=scale)
        u = student.cdf(x).clamp(min=1e-6, max=1 - 1e-6)
        z = Normal(0.0, 1.0).icdf(u)
        return z

    def _from_gaussian(
        self, z: torch.Tensor, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Invert Gaussian samples back to the Student-t space."""

        u = Normal(0.0, 1.0).cdf(z).clamp(min=1e-6, max=1 - 1e-6)
        student = StudentT(df=df, loc=loc, scale=scale)
        x = student.icdf(u)
        return x

    def _normalize_cond(
        self, cond_dynamic: torch.Tensor, cond_static: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize conditioning signals to stabilize training."""

        dyn_loc = cond_dynamic.mean(dim=1, keepdim=True)
        dyn_scale = cond_dynamic.std(dim=1, keepdim=True).clamp_min(self.scale_eps)
        cond_dynamic_norm = (cond_dynamic - dyn_loc) / dyn_scale

        static_loc = cond_static.mean(dim=1, keepdim=True)
        static_scale = cond_static.std(dim=1, keepdim=True).clamp_min(self.scale_eps)
        cond_static_norm = (cond_static - static_loc) / static_scale

        return cond_dynamic_norm, cond_static_norm

    def _prepare_conditioning(
        self,
        x_hist_norm: torch.Tensor,
        cond_dynamic_norm: torch.Tensor,
        cond_static_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse history-derived tokens with provided conditioning signals."""

        if cond_dynamic_norm.shape[-1] != self.input_cond_dynamic_dim:
            raise ValueError(
                f"Expected cond_dynamic dim {self.input_cond_dynamic_dim}, got {cond_dynamic_norm.shape[-1]}"
            )
        if cond_static_norm.shape[-1] != self.input_cond_static_dim:
            raise ValueError(
                f"Expected cond_static dim {self.input_cond_static_dim}, got {cond_static_norm.shape[-1]}"
            )

        # Encode history as dynamic tokens so downstream cross-attention and
        # causal masking can exploit temporal structure and relative biases.
        hist_tokens = self.history_encoder(x_hist_norm.transpose(1, 2))  # [B, E, T]
        hist_tokens = hist_tokens.transpose(1, 2)  # [B, T, E]

        # Concatenate learned history tokens with exogenous dynamic features.
        cond_dynamic_aug = torch.cat([cond_dynamic_norm, hist_tokens], dim=-1)

        # Static channel incorporates pooled history for FiLM modulation.
        hist_summary = self.history_pool(hist_tokens.transpose(1, 2)).squeeze(-1)
        cond_static_aug = torch.cat([cond_static_norm, hist_summary], dim=-1)

        return cond_dynamic_aug, cond_static_aug

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the diffusion training loss for a batch.

        Args:
            x_hist: Historical target window ``[B, context_length, target_dim]``.
            x_future: Forecast target window ``[B, prediction_length, target_dim]``.
            cond_dynamic: Dynamic conditioning features aligned to history
                ``[B, context_length, cond_dynamic_dim]``.
            cond_static: Static conditioning features ``[B, cond_static_dim]``.
        Returns:
            Scalar diffusion training loss.
        """

        if x_hist.shape[1] != self.context_length:
            raise ValueError(
                f"Expected x_hist length {self.context_length}, got {x_hist.shape[1]}"
            )
        if x_future.shape[1] != self.prediction_length:
            raise ValueError(
                f"Expected x_future length {self.prediction_length}, got {x_future.shape[1]}"
            )
        if cond_dynamic.shape[1] != self.context_length:
            raise ValueError(
                f"Expected cond_dynamic length {self.context_length}, got {cond_dynamic.shape[1]}"
            )

        df, loc, scale = self._fit_student_t(x_hist)
        x_hist_norm = self._to_gaussian(x_hist, df, loc, scale)
        x_future_norm = self._to_gaussian(x_future, df, loc, scale)
        cond_dynamic_norm, cond_static_norm = self._normalize_cond(
            cond_dynamic, cond_static
        )

        cond_dynamic_aug, cond_static_aug = self._prepare_conditioning(
            x_hist_norm, cond_dynamic_norm, cond_static_norm
        )

        loss = self.model(
            x_future=x_future_norm,
            cond_dynamic=cond_dynamic_aug,
            cond_static=cond_static_aug,
        )
        return loss

    @torch.no_grad()
    def sample_forecast(
        self,
        x_hist: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor,
        *,
        num_samples: int = 1,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Generate forecast samples using the trained diffusion model.

        Args:
            x_hist: Historical target window ``[B, context_length, target_dim]``.
            cond_dynamic: Dynamic conditioning features ``[B, context_length, cond_dynamic_dim]``.
            cond_static: Static conditioning features ``[B, cond_static_dim]``.
            num_samples: Number of forecast samples per series.
        Returns:
            Forecast samples with shape ``[num_samples, B, prediction_length, target_dim]``.
        """

        batch_size = x_hist.size(0)

        # Fit Student-t marginals on history then transform to Gaussian space.
        df, loc, scale = self._fit_student_t(x_hist)
        x_hist_norm = self._to_gaussian(x_hist, df, loc, scale)
        cond_dynamic_norm, cond_static_norm = self._normalize_cond(cond_dynamic, cond_static)

        cond_dynamic_aug, cond_static_aug = self._prepare_conditioning(
            x_hist_norm, cond_dynamic_norm, cond_static_norm
        )

        # Repeat conditioning for multiple samples.
        cond_dynamic_rep = cond_dynamic_aug.repeat(num_samples, 1, 1)
        cond_static_rep = cond_static_aug.repeat(num_samples, 1)

        cond = {"dynamic": cond_dynamic_rep, "static": cond_static_rep}
        z_samples = self.model.diffusion.sample(
            batch_size=batch_size * num_samples,
            horizon=self.prediction_length,
            cond=cond,
            clip_denoised=clip_denoised,
        )

        z_samples = z_samples.view(
            num_samples, batch_size, self.target_dim, self.prediction_length
        )
        z_samples = z_samples.permute(0, 1, 3, 2)  # -> [S, B, horizon, target_dim]

        # Invert marginal transform back to real returns.
        loc_rep = loc.unsqueeze(0).expand(num_samples, -1, -1, -1)
        scale_rep = scale.unsqueeze(0).expand_as(loc_rep)
        df_rep = df.unsqueeze(0).expand_as(loc_rep)
        samples = self._from_gaussian(z_samples, df_rep, loc_rep, scale_rep)
        return samples


__all__ = ["ConditionalTimeGradTrainingNetwork"]