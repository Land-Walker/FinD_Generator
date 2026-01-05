"""Prediction-ready network for conditional TimeGrad.

This module mirrors the training wrapper architecture but performs strictly
autoregressive sampling with careful conditioning handling:
- The history encoder is recomputed at every forecast step so cross-attention
  and FiLM modules see the latest generated targets.
- Dynamic conditioning is only applied to the historical window; generated
  steps receive zeroed dynamic tokens to avoid leaking future information.
- Loc/scale statistics are frozen from the initial history window and reused
  for the entire forecast horizon.
- Causal mask shapes remain consistent by maintaining a fixed-length
  conditioning window that slides alongside the autoregressive loop.
"""
from __future__ import annotations

from typing import Tuple

import scipy.stats as stats

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.models.conditional_timegrad import ConditionalTimeGrad
from src.models.timegrad_core.timegrad_base import TimeGradBase


class StudentTMarginalMixin:
    """Shared Student-t marginal helpers for both conditional and vanilla paths."""

    scale_eps: float
    fixed_df: float
    ewma_alpha: float
    target_dim: int

    def _fit_student_t(self, x_hist: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        loc = x_hist.mean(dim=1, keepdim=True)
        centered = x_hist - loc

        df = torch.full(
            (x_hist.size(0), 1, self.target_dim),
            float(self.fixed_df),
            device=x_hist.device,
            dtype=x_hist.dtype,
        )

        alpha = self.ewma_alpha
        var = centered[:, 0:1, :].pow(2)
        vars_t = [var]
        for t in range(1, centered.size(1)):
            var = alpha * var + (1.0 - alpha) * centered[:, t : t + 1, :].pow(2)
            vars_t.append(var)

        ewma_var = torch.cat(vars_t, dim=1)
        scale_hist = torch.sqrt(ewma_var * (self.fixed_df - 2.0) / self.fixed_df).clamp_min(
            self.scale_eps
        )

        scale_future = scale_hist[:, -1:, :]
        return df, loc, scale_hist, scale_future

    def _student_t_cdf(
        self, x: torch.Tensor, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            u_np = stats.t.cdf(
                x.detach().cpu().numpy(),
                df=df.detach().cpu().numpy(),
                loc=loc.detach().cpu().numpy(),
                scale=scale.detach().cpu().numpy(),
            )
        u = torch.from_numpy(u_np).to(device=x.device, dtype=x.dtype)
        return u.clamp(min=1e-6, max=1 - 1e-6)

    def _student_t_ppf(
        self, u: torch.Tensor, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            x_np = stats.t.ppf(
                u.detach().cpu().numpy(),
                df=df.detach().cpu().numpy(),
                loc=loc.detach().cpu().numpy(),
                scale=scale.detach().cpu().numpy(),
            )
        return torch.from_numpy(x_np).to(device=u.device, dtype=u.dtype)

    def _to_gaussian(
        self, x: torch.Tensor, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        u = self._student_t_cdf(x, df, loc, scale)
        z = Normal(0.0, 1.0).icdf(u)
        return z

    def _from_gaussian(
        self, z: torch.Tensor, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        u = Normal(0.0, 1.0).cdf(z).clamp(min=1e-6, max=1 - 1e-6)
        x = self._student_t_ppf(u, df, loc, scale)
        return x

class ConditionalTimeGradPredictionNetwork(StudentTMarginalMixin, nn.Module):
    """Autoregressive inference wrapper for the conditional TimeGrad model."""

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
        fixed_df: float = 4.0,
        ewma_alpha: float = 0.94,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.scale_eps = scale_eps
        self.fixed_df = fixed_df
        self.ewma_alpha = ewma_alpha

        self.input_cond_dynamic_dim = cond_dynamic_dim
        self.input_cond_static_dim = cond_static_dim
        self.cond_embed_dim = cond_embed_dim

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

        self.history_pool = nn.AdaptiveAvgPool1d(1)

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

    def _normalize_cond(
        self, cond_dynamic: torch.Tensor, cond_static: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize conditioning signals to stabilize inference."""

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
        """Fuse history tokens with dynamic/static conditioning."""

        hist_tokens = self.history_encoder(x_hist_norm.transpose(1, 2))  # [B, E, T]
        hist_tokens = hist_tokens.transpose(1, 2)  # [B, T, E]

        # Ensure the historical token sequence matches the dynamic conditioning
        # length in case padding or circular convolutions alter the effective
        # width (e.g., very short contexts). This keeps cross-attention causal
        # masks aligned with the conditioning tokens used by the denoiser.
        if hist_tokens.size(1) != cond_dynamic_norm.size(1):
            hist_tokens = nn.functional.interpolate(
                hist_tokens.transpose(1, 2),
                size=cond_dynamic_norm.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        cond_dynamic_aug = torch.cat([cond_dynamic_norm, hist_tokens], dim=-1)

        hist_summary = self.history_pool(hist_tokens.transpose(1, 2)).squeeze(-1)
        cond_static_aug = torch.cat([cond_static_norm, hist_summary], dim=-1)

        return cond_dynamic_aug, cond_static_aug

    @torch.no_grad()
    def sample_autoregressive(
        self,
        x_hist: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor,
        *,
        num_samples: int = 1,
        clip_denoised: bool = True,
        sampling_strategy: str = "masked_step",
    ) -> torch.Tensor:
        """Generate forecasts autoregressively with fixed normalization.

        Dynamic conditioning is only drawn from the provided history window.
        Each forecast step recomputes the history encoder to keep conditioning
        aligned with newly generated targets.

        Args:
            sampling_strategy: ``"masked_step"`` uses the single-timestep
                masked sampler (fast, approximate; best for quick iterations or
                long horizons), while ``"full_horizon"`` runs the full DDPM
                reverse process each step (slower but mirrors the exact
                ancestral rollout).
        """

        batch_size = x_hist.size(0)
        device = x_hist.device

        if x_hist.shape[1] != self.context_length:
            raise ValueError(
                f"Expected x_hist length {self.context_length}, got {x_hist.shape[1]}"
            )
        if cond_dynamic.shape[1] != self.context_length:
            raise ValueError(
                f"Expected cond_dynamic length {self.context_length}, got {cond_dynamic.shape[1]}"
            )
        if cond_dynamic.shape[-1] != self.input_cond_dynamic_dim:
            raise ValueError(
                f"Expected cond_dynamic dim {self.input_cond_dynamic_dim}, got {cond_dynamic.shape[-1]}"
            )
        if cond_static.shape[-1] != self.input_cond_static_dim:
            raise ValueError(
                f"Expected cond_static dim {self.input_cond_static_dim}, got {cond_static.shape[-1]}"
            )

       # Backward compatibility for checkpoints saved prior to marginal attrs.
        if not hasattr(self, "fixed_df"):
            self.fixed_df = 4.0
        if not hasattr(self, "ewma_alpha"):
            self.ewma_alpha = 0.94

        df, loc, scale_hist, scale_future = self._fit_student_t(x_hist)
        x_hist_norm = self._to_gaussian(x_hist, df, loc, scale_hist)
        cond_dynamic_norm, cond_static_norm = self._normalize_cond(cond_dynamic, cond_static)

        # Freeze marginal params across samples and horizon.
        loc_rep = loc.repeat(num_samples, 1, 1)
        df_rep = df.repeat(num_samples, 1, 1)
        scale_rep = scale_future.repeat(num_samples, 1, 1)

        # Expand batch for multiple samples.
        hist_window = x_hist_norm.repeat(num_samples, 1, 1)
        cond_dynamic_window = cond_dynamic_norm.repeat(num_samples, 1, 1)
        cond_static_rep = cond_static_norm.repeat(num_samples, 1)

        forecasts_norm = []
        for _ in range(self.prediction_length):
            cond_dynamic_aug, cond_static_aug = self._prepare_conditioning(
                hist_window, cond_dynamic_window, cond_static_rep
            )

            cond = {"dynamic": cond_dynamic_aug, "static": cond_static_aug}

            if sampling_strategy == "masked_step":
                step = self.model.diffusion.sample_step(
                    batch_size=batch_size * num_samples,
                    horizon=self.prediction_length,
                    target_index=0,
                    cond=cond,
                    clip_denoised=clip_denoised,
                ).permute(0, 2, 1)  # [SB, 1, C]
            elif sampling_strategy == "full_horizon":
                full_block = self.model.diffusion.sample(
                    batch_size=batch_size * num_samples,
                    horizon=self.prediction_length,
                    cond=cond,
                    clip_denoised=clip_denoised,
                ).permute(0, 2, 1)  # [SB, horizon, C]
                step = full_block[:, :1, :]
            else:
                raise ValueError(
                    "sampling_strategy must be 'masked_step' or 'full_horizon'"
                )
            forecasts_norm.append(step)

            # Slide history and zero-fill dynamic conditioning for generated steps.
            hist_window = torch.cat([hist_window[:, 1:, :], step], dim=1)
            zero_dyn = torch.zeros(
                batch_size * num_samples,
                1,
                self.input_cond_dynamic_dim,
                device=device,
                dtype=cond_dynamic.dtype,
            )
            cond_dynamic_window = torch.cat([cond_dynamic_window[:, 1:, :], zero_dyn], dim=1)

        forecasts_norm = torch.cat(forecasts_norm, dim=1)  # [SB, horizon, C]
        forecasts = self._from_gaussian(forecasts_norm, df_rep, loc_rep, scale_rep)
        forecasts = forecasts.view(num_samples, batch_size, self.prediction_length, self.target_dim)
        return forecasts


class VanillaTimeGradPredictionNetwork(StudentTMarginalMixin, nn.Module):
    """Inference wrapper for vanilla TimeGrad with Student-t marginals."""

    def __init__(
        self,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        *,
        diff_steps: int = 100,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
        residual_layers: int = 6,
        residual_channels: int = 32,
        scale_eps: float = 1e-5,
        fixed_df: float = 4.0,
        ewma_alpha: float = 0.94,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.scale_eps = scale_eps
        self.fixed_df = fixed_df
        self.ewma_alpha = ewma_alpha

        self.model = TimeGradBase(
            target_dim=target_dim,
            prediction_length=prediction_length,
            diff_steps=diff_steps,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
        )

    @torch.no_grad()
    def sample_forecast(
        self, x_hist: torch.Tensor, *, num_samples: int = 1, clip_denoised: bool = True
    ) -> torch.Tensor:
        if x_hist.shape[1] != self.context_length:
            raise ValueError(
                f"Expected x_hist length {self.context_length}, got {x_hist.shape[1]}"
            )

        batch_size = x_hist.size(0)

        df, loc, _, scale_future = self._fit_student_t(x_hist)
        loc_rep = loc.expand(num_samples, -1, -1, -1)
        df_rep = df.expand_as(loc_rep)
        scale_rep = scale_future.expand_as(loc_rep)

        z_samples = []
        for _ in range(num_samples):
            z = self.model.diffusion.sample(
                batch_size=batch_size,
                horizon=self.prediction_length,
                cond=None,
                clip_denoised=clip_denoised,
            )
            z_samples.append(z.transpose(1, 2))

        z_stack = torch.stack(z_samples, dim=0)
        samples = self._from_gaussian(z_stack, df_rep, loc_rep, scale_rep)
        return samples


__all__ = [
    "ConditionalTimeGradPredictionNetwork",
    "VanillaTimeGradPredictionNetwork",
]