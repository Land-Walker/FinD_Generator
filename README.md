# FinD Generator

A conditional TimeGrad implementation for financial time series forecasting.

**Objective:** To serve as a high-fidelity, probabilistic infrastructure for generating **counterfactual financial time series scenarios** (e.g., asset returns, volatility) that are conditioned on past market history and exogenous, scenario-defined macro regimes.

This project is a sophisticated implementation of a conditional diffusion model, adapting the **TimeGrad** architecture to meet the strict requirements of quantitative finance. For more details, check out architecture.md.

---

## Usage

The project includes a convenience script `run.py` to handle data collection, training, and inference in a single pipeline.

### 1. First Run (Download Data & Train)

To download fresh data, train the model, and generate forecasts:

```bash
python run.py --download --epochs 10 --batch-size 64
```

### 2. Subsequent Runs (Use Local Data)

Once data is downloaded to `data/raw`, you can run training without the `--download` flag to use the cached parquet files:

```bash
python run.py --epochs 10
```

### 3. Key Arguments

- `--download`: Fetch fresh data from Yahoo Finance and FRED.
- `--epochs`: Number of training epochs (default: 1).
- `--batch-size`: Batch size (default: 64).
- `--num-samples`: Number of forecast samples to generate per series (default: 2).
- `--device`: `cpu` or `cuda` (automatically detected if omitted).

Forecasts are saved to `data/processed/forecasts.pt`.