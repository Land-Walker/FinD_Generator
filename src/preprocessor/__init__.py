"""Preprocessing package.

Note (W2.5): ``DataCollector`` is intentionally NOT imported or re-exported
here. Importing it eagerly pulled yfinance/pandas_datareader into every
local-data run. Import it directly from
``src.preprocessor.data_collector`` when downloading fresh data.
"""

from .data_loader import (
    TimeGradDataModule,
)

__all__ = [
    "TimeGradDataModule",
]
