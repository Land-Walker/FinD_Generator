# FinD_Generator Architecture and Data Flow

FinD_Generator implements a full Conditional TimeGrad pipeline, separating data preparation, conditioning encoding, and the core diffusion process.

### Architecture Overview

The system design enforces a strict separation between **historical context (History Encoder)** and the **stochastic forecast process (Diffusion)**.

1.  **Input Data**: Cleaned financial time-series.
2.  **Preprocessing / Data Discipline**: Normalization (mean/std) and dimensionality reduction (PCA) are fit **only on training history** and frozen for all future steps. (See `data_loader.py` documentation for details).
3.  **Conditional Feature Extraction**: Features are split into two causal categories:

| Feature Type | Description (Source Files) | Role in TimeGrad |
| :--- | :--- | :--- |
| **Static Conditioning** | Single vector of features, typically regime labels (e.g., market state, volatility bucket) from the **end of the historical context window**. (From `data_loader.py`) | Encoded and used as a **FiLM-like modulation** signal across the entire denoising network (See `conditioned_epsilon_theta.py`'s use of `static_encoder`). This controls the overall *implied variance* of the output distribution. |
| **Dynamic Conditioning** | Time-series of features (e.g., macro signals, yield curve from `config.py`) corresponding *only* to the historical context window. | Encoded and integrated via a **Causal Cross-Attention** block (See `conditioned_epsilon_theta.py`). This allows the denoising process to be informed by the latest evolving context without violating the temporal causality. |

4.  **Core Diffusion (TimeGrad)**: The `ConditionalTimeGrad` model executes the standard DDPM ancestral sampling loop, using the conditioned features at every denoising step to guide the prediction toward the desired regime.

### B. Causal Leakage Rules & Conditioning Mapping

This table should be added to the `docs/conditioning.md` file (or the `data_loader.py` docstring, since it controls the indices). It directly addresses the leakage concerns of a quant team.

| Feature Name (Example) | Category | Known/Unknown | Lookahead Safety | Source Code Evidence |
| :--- | :--- | :--- | :--- | :--- |
| Asset Returns | **Target Sequence** | Unknown (Future) | **Strictly Forecasted** | Used in `x_future` only for loss computation (`training_network.py`). |
| Yield Curve (`T10Y2Y`) | **Dynamic Context** | Known (History) | **Safe** | Only included in the history window (`cond_dynamic` in `data_loader.py`). |
| Market Regime | **Static Context** | Known (Scenario) | **Safe** | Only the regime from the last historical step (`hist_end - 1`) is used for the entire future forecast window (`data_loader.py`). This acts as the **"start state instruction."** |
| Policy Stress | **Scenario Override** | Known (Future) | **Safe** | Used explicitly for **counterfactual** analysis in `scenario_generator.py`. This is safe because it is an explicit *assumption* for the scenario, not an observation. |