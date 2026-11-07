from __future__ import annotations

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    ticker = ticker.strip()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise RuntimeError(
            f"No data returned for '{ticker}'. Check the ticker symbol, "
            "internet connection, and Yahoo Finance availability."
        )

    # Recent yfinance versions may return MultiIndex columns even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise RuntimeError(f"Downloaded data is missing required columns: {missing}")

    data = data[REQUIRED_COLUMNS].copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    if len(data) < 150:
        raise RuntimeError(
            f"Only {len(data)} usable rows were downloaded. "
            "Choose a longer period or another ticker (at least 150 rows recommended)."
        )

    return data
