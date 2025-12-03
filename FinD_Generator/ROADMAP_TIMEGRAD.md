# TimeGrad Conditioning Roadmap

This document captures the actionable steps to stand up a faithful TimeGrad
baseline and extend it with dynamic/static conditioning for ``Clean-Refactor-1``.
Follow the phases sequentially to minimize integration risk and match the
original PTS TimeGrad performance.

## Phase 0 — Workspace prep
- Create module layout under ``src/models``: ``timegrad_core`` (vanilla) and
  ``conditional_timegrad`` (extensions).
- Install the PyTorch TS reference: ``pip install pytorchts``.
- Locate and inspect the reference sources:
  ```python
  import pts, pts.modules, pts.model.time_grad, inspect, os
  print(os.path.dirname(pts.modules.__file__))
  print(os.path.dirname(pts.model.time_grad.__file__))
  ```
- Copy the following into ``src/models/timegrad_core``:
  - ``gaussian_diffusion.py`` (required)
  - ``epsilon_theta.py`` (required)
  - Optionally ``scaler.py`` and ``distribution_output.py`` if you rely on them.

## Phase 1 — Port vanilla TimeGrad
- Remove GluonTS/PTS estimator dependencies; keep pure PyTorch ``nn.Module``
  components.
- Wire ``TimeGradBase`` around ``GaussianDiffusion`` and ``EpsilonTheta``.
- Validate on a known dataset (e.g., Electricity/Traffic) and ensure CRPS/MAE
  match the PTS implementation within noise.

## Phase 2 — Add conditioning
- Build compact encoders:
  - Dynamic: flatten ``cond_dynamic`` then ``Linear -> ReLU -> Linear``.
  - Static: ``Linear -> ReLU -> Linear``.
- Concatenate encodings, project to ``prediction_length``, broadcast, and add to
  the noisy input before calling the base epsilon network.
- Keep the diffusion schedule and loss identical to the vanilla model.

## Phase 3 — Integrate data pipeline
- Expect batch keys:
  - ``x_future``: ``[B, horizon, target_dim]``
  - ``cond_dynamic``: ``[B, seq_len, cond_dim]``
  - ``cond_static``: ``[B, static_dim]``
- Training loop example:
  ```python
  loss = model(
      x_future=batch["x_future"].to(device),
      cond_dynamic=batch["cond_dynamic"].to(device),
      cond_static=batch["cond_static"].to(device),
  )
  ```

## Phase 4 — Train and validate
- Add gradient clipping and an LR scheduler (e.g., OneCycleLR).
- Track CRPS, MAE, calibration, and sharpness; compare to the vanilla model.

## Phase 5 — Inference and scenarios
- Implement ``sample_paths``, ``mean_forecast``, and ``quantiles`` on top of the
  diffusion sampler.
- Hook static conditioning to macro regime indicators or other metadata.

## Phase 6 — Documentation
- Prepare an architecture overview, conditioning rationale, and performance
  comparison against the original TimeGrad.
