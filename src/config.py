from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILENAMES = {
    "target": "target.parquet",
    "market": "market.parquet",
    "daily_macro": "daily_macro.parquet",
    "monthly_macro": "monthly_macro.parquet",
    "quarterly_macro": "quarterly_macro.parquet",
}

# -----------------------------------------------------------------------------
# Data Collection
# -----------------------------------------------------------------------------
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"
MARKET_TICKER = "^GSPC"
TARGET_TICKER = "SPY"

FRED_DAILY_IDS = {
    "DGS10": "yield_curve_daily",
}
FRED_MONTHLY_IDS = {
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "FEDFUNDS": "interest_rate",
    "BOPGSTB": "trade_balance",
}
FRED_QUARTERLY_IDS = {
    "GDP": "gdp",
    "GFDEBTN": "gov_debt",
    "FYFSD": "gov_fiscal_balance",
    "W006RC1Q027SBEA": "tax_receipts",
    "FGEXPND": "gov_spending",
}

# -----------------------------------------------------------------------------
# Preprocessing & Model
# -----------------------------------------------------------------------------
WAVELET = "db4"
WAVELET_LEVEL = 3
PCA_VARIANCE = 0.95
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

DEFAULT_SEQ_LEN = 64
DEFAULT_HORIZON = 5
DEFAULT_BATCH = 64