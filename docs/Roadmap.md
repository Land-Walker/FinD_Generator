# Development Roadmap

## 1. System Architecture & Engineering
- **Configuration Management**: Migrate from `config.py` to **Hydra/OmegaConf** for hierarchical, type-safe configuration and easier hyperparameter sweeping.
- **CLI Stress-Test API**: Expose granular scenario controls (e.g., `python run.py --scenario "2008_crash" --shock_magnitude 2.0`) directly via CLI arguments to streamline risk reporting.
- **Inference Optimization**: Implement **DDIM (Denoising Diffusion Implicit Models)** sampling to reduce inference steps from 100 to 10-20, targeting sub-50ms generation for real-time applications.

## 2. Modeling & Research
- **Transformer Backbone**: Experiment with replacing the LSTM history encoder with a **Temporal Fusion Transformer (TFT)** or standard Transformer encoder to better capture long-range regime dependencies.
- **Classifier-Free Guidance**: Implement guidance terms to allow "knob-turning" of specific macro variables (e.g., "Generate a trajectory where Inflation > 5%") without retraining the model.
- **Alternative Noise Schedulers**: Benchmark Cosine vs. Linear noise schedules to improve sample quality in low-volatility regimes.

## 3. Evaluation & Downstream Validation
- **Downstream RL Testing**: Connect FinD_Generator to an RL Execution Agent to validate if training on synthetic "Counterfactual" data improves the agent's Sharpe Ratio on out-of-sample data.
- **Ablation Studies**: Systematically evaluate the contribution of "Dynamic Macro Features" vs. "Static Regime Labels" to the model's predictive NLL (Negative Log Likelihood).
- **Tail-Risk Calibration**: Compare the generated distribution's Kurtosis and VaR (Value at Risk) against historical 2000-2023 crash events to ensure heavy-tail fidelity.