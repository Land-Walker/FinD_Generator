# FinD_Generator System Architecture

This document details the technical implementation of the FinD_Generator, a conditional diffusion model for financial time series forecasting. Unlike the README, which focuses on usage and high-level goals, this document explains the internal data flow, model components, and inference mechanics.

## 1. System Overview

The system is designed as a pipeline with three distinct stages:
1.  **Data Ingestion & Transformation**: Raw financial data is cleaned, transformed, and aligned.
2.  **Conditional Training**: A diffusion model learns to denoise target sequences conditioned on historical and regime contexts.
3.  **Autoregressive Inference**: A predictor network generates future trajectories step-by-step, updating its historical context dynamically.

### Component Map

| Component | Source File | Responsibility |
| :--- | :--- | :--- |
| **Orchestrator** | `run.py` | Wires data, training, and inference together. |
| **Data Module** | `src/preprocessor/data_loader.py` | Handles loading, wavelet denoising, PCA, regime labeling, and tensor creation. |
| **Training Net** | `src/training/training_network.py` | Wraps the diffusion model for the training loop (calculating loss). |
| **Prediction Net** | `src/predictor/prediction_network.py` | Wraps the diffusion model for inference (autoregressive sampling). |
| **Scenario Gen** | `src/scenario_generator.py` | Modifies static conditioning features for counterfactual analysis. |

---

## 2. Data Pipeline (`src/preprocessor`)

The data pipeline enforces strict causal safety to prevent look-ahead bias.

### A. Feature Engineering
The raw data is processed into three categories of features:

1.  **Target Series ($x$)**:
    *   **Source**: OHLCV data (Yahoo Finance).
    *   **Processing**: Wavelet denoising (db4, level 3) to remove high-frequency noise while preserving structural trends.
    *   **Dimensionality**: Reduced via PCA (95% variance retained) fitted *only* on the training split.

2.  **Dynamic Conditioning ($c_{dyn}$)**:
    *   **Source**: Daily macro (VIX, Yield Curve) and Monthly macro (CPI, Unemployment).
    *   **Alignment**: Lower frequency data is forward-filled (safe) or interpolated.
    *   **Usage**: These features align 1-to-1 with the historical time steps.

3.  **Static Conditioning ($c_{stat}$)**:
    *   **Source**: Derived Regime Labels (Market Bull/Bear, Inflation/Stagflation).
    *   **Usage**: A single vector representing the "state of the world" at the *end* of the historical window. This guides the generation of the entire forecast horizon.

### B. Causal Leakage Rules

| Feature Name | Category | Known/Unknown | Lookahead Safety | Source Code Evidence |
| :--- | :--- | :--- | :--- | :--- |
| Asset Returns | **Target Sequence** | Unknown (Future) | **Strictly Forecasted** | Used in `x_future` only for loss computation. |
| Yield Curve | **Dynamic Context** | Known (History) | **Safe** | Only included in the history window (`cond_dynamic`). |
| Market Regime | **Static Context** | Known (Scenario) | **Safe** | Only the regime from the last historical step (`hist_end - 1`) is used. |
| Policy Stress | **Scenario Override** | Known (Future) | **Safe** | Used explicitly for **counterfactual** analysis in `scenario_generator.py`. |

---

## 3. Model Architecture (`src/models`)

The core model is a **Conditional TimeGrad** network. It combines a Recurrent Neural Network (RNN) with a Denoising Diffusion Probabilistic Model (DDPM).

### A. The Backbone (RNN)
An LSTM or GRU processes the historical window $x_{hist}$ and dynamic conditioning $c_{dyn}$ to produce a hidden state $h_t$. This hidden state summarizes the temporal dependencies.

### B. The Diffusion Process
The model learns to approximate the data distribution $q(x_{future} | x_{hist})$ by reversing a Gaussian diffusion process.

1.  **Forward Process ($q$)**: Gradually adds Gaussian noise to $x_{future}$ over $K$ steps until it becomes pure noise $\mathcal{N}(0, I)$.
2.  **Reverse Process ($p_\theta$)**: A neural network $\epsilon_\theta(x_t, t, h, c_{stat})$ predicts the noise added at step $t$, allowing the model to denoise samples iteratively.

---

## 4. Inference & Autoregression (`src/predictor`)

Inference is handled by `ConditionalTimeGradPredictionNetwork`. Unlike standard diffusion generation, time-series forecasting requires **autoregression**.

### The Sliding Window Loop
To generate a forecast of length $L$:
1.  **Step 1**: The model takes the initial history $x_{hist}$ and generates the first future step $\hat{x}_{t+1}$.
2.  **Update**: The history window slides: $x_{hist} \leftarrow [x_{hist}[1:], \hat{x}_{t+1}]$.
3.  **Re-Encode**: The history encoder runs again on the new window to update the context for the next step.
4.  **Repeat**: This continues for $L$ steps.

### Student-t Marginals
Financial returns often have heavy tails (leptokurtic). The prediction network:
1.  Fits a Student-t distribution to the local history $x_{hist}$.
2.  Transforms the data to Gaussian space (via CDF/ICDF) before passing it to the diffusion model.
3.  Transforms the generated Gaussian samples back to the data space (Student-t) at the output.