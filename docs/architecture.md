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
An LSTM or GRU processes the historical window $x_{hist}$ and dynamic conditioning $c_{dyn}$ to produce a hidden state $h_t$. This hidden state summarizes the temporal dependencies:

$$
h_t = \text{RNN}(x_t, c_{dyn, t}, h_{t-1})
$$

This embedding $h_t$ serves as the primary conditioning input for the diffusion model at time step $t$.

### B. The Diffusion Process
The model learns to approximate the conditional data distribution $q(x_{future} | x_{hist})$ by reversing a Gaussian diffusion process. We denote diffusion steps by $k \in \{1, \dots, K\}$.

#### 1. Forward Process (Noise Injection)
We define a fixed Markov chain that gradually adds Gaussian noise to the data $x_0$ (representing the target slice $x_{future}$) over $K$ steps according to a variance schedule $\beta_1, \dots, \beta_K$:

$$
q(x_k | x_{k-1}) = \mathcal{N}(x_k; \sqrt{1 - \beta_k} x_{k-1}, \beta_k I)
$$

Using the property of Gaussians, we can sample $x_k$ at any arbitrary step $k$ directly from $x_0$:

$$
q(x_k | x_0) = \mathcal{N}(x_k; \sqrt{\bar{\alpha}_k} x_0, (1 - \bar{\alpha}_k) I)
$$

where $\alpha_k = 1 - \beta_k$ and $\bar{\alpha}_k = \prod_{s=1}^k \alpha_s$.

#### 2. Reverse Process (Denoising)
The generative process is defined as the reverse Markov chain, where a neural network approximates the true posterior $q(x_{k-1} | x_k)$. We model this as a Gaussian transition conditioned on the history embedding $h$ and static regimes $c_{stat}$:

$$
p_\theta(x_{k-1} | x_k, h, c_{stat}) = \mathcal{N}(x_{k-1}; \mu_\theta(x_k, k, h, c_{stat}), \tilde{\beta}_k I)
$$

The mean $\mu_\theta$ is parameterized by the noise prediction network $\epsilon_\theta$:

$$
\mu_\theta(x_k, k, h, c_{stat}) = \frac{1}{\sqrt{\alpha_k}} \left( x_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}} \epsilon_\theta(x_k, k, h, c_{stat}) \right)
$$

#### 3. Training Objective
The model is trained to minimize the simplified variational lower bound, which corresponds to the Mean Squared Error (MSE) between the actual noise $\epsilon$ and the predicted noise:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0, I), k} \left[ \| \epsilon - \epsilon_\theta(\underbrace{\sqrt{\bar{\alpha}_k} x_0 + \sqrt{1 - \bar{\alpha}_k} \epsilon}_{x_k}, k, h, c_{stat}) \|^2 \right]
$$

---

## 4. Inference & Autoregression (`src/predictor`)

Inference is handled by `ConditionalTimeGradPredictionNetwork`. Unlike standard diffusion generation, time-series forecasting requires **autoregression**.

### Probabilistic Factorization
The goal is to model the joint distribution of the future time series $x_{T+1:T+L}$ given history $x_{1:T}$. The model factorizes this joint distribution autoregressively:

$$
p_\theta(x_{T+1:T+L} | x_{1:T}) = \prod_{t=T+1}^{T+L} p_\theta(x_t | x_{1:t-1}, c_{dyn, t}, c_{stat})
$$

At each step $t$, the conditional distribution $p_\theta(x_t | \dots)$ is realized by sampling from the diffusion model described above.

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