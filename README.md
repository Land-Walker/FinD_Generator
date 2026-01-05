# FinD_Generator: Conditional Diffusion Model for Financial Scenario Generation

**Objective:** To serve as a high-fidelity, probabilistic infrastructure for generating **counterfactual financial time series scenarios** (e.g., asset returns, volatility) that are conditioned on past market history and exogenous, scenario-defined macro regimes.

This project is a sophisticated implementation of a conditional diffusion model, adapting the **TimeGrad** architecture to meet the strict requirements of quantitative finance.

---

### **Crucial Project Scope (What it IS and IS NOT)**

| **FinD_Generator IS...** | **FinD_Generator IS NOT...** | **Why This Matters to Quants/Researchers** |
| :--- | :--- | :--- |
| **A Probabilistic Scenario Generator** | A deterministic time-series predictor (like LSTM or Transformer). | Proves understanding that financial modeling requires capturing **uncertainty (risk)**, not just point estimates. |
| **An Infrastructure Tool** | A direct trading or execution bot. | Separates core modeling ability from speculative trading logic, positioning the project as a **research/risk management asset**. |
| **Regime-Aware & Stress-Test Ready** | A passively trained model. | Highlights the unique **scenario planning** capability unlocked by static conditioning (see `scenario_generator.py`). |
| **Leakage-Safe** | Using look-ahead bias during preprocessing/feature extraction. | Demonstrates strong **engineering and research discipline**, essential for backtesting credibility. |