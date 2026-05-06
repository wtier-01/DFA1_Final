"""
Alpha Vantage stock price client for the FDA1 Final Project.

This module downloads daily adjusted price data for the selected stocks using
Alpha Vantage's TIME_SERIES_DAILY_ADJUSTED endpoint.

The project requires adjusted closing prices for two eligible U.S. stocks over
at least ten full years ending December 31, 2025. We cache the downloaded data
locally so the final notebook can run reproducibly without repeatedly calling
the API.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

from src.config import (
    ALPHA_VANTAGE_API_KEY,
    ALPHA_VANTAGE_BASE_URL,
    ALPHA_VANTAGE_DAILY_ADJUSTED_FUNCTION,
    ALPHA_VANTAGE_ENTITLEMENT,
    ALPHA_VANTAGE_OUTPUTSIZE,
    END_DATE,
    RAW_PRICE_FILE,
    START_DATE,
    TICKERS,
)


def _validate_api_key(api_key: str) -> None:
    """
    Validate that the Alpha Vantage API key has been provided.
    """
    if not api_key or api_key == "PASTE_YOUR_KEY_HERE":
        raise ValueError(
            "Alpha Vantage API key is missing. "
            "Add your real key to the .env file as ALPHA_VANTAGE_API_KEY=..."
        )


def fetch_daily_adjusted(
    symbol: str,
    api_key: Optional[str] = None,
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    """
    Download daily adjusted OHLCV data for one stock from Alpha Vantage.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol, such as MSFT or LMT.
    api_key : str, optional
        Alpha Vantage API key. If None, the key is loaded from config.
    sleep_seconds : float
        Small pause after the request to avoid unnecessary API pressure.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with OHLCV fields and adjusted close.
    """
    api_key = api_key or ALPHA_VANTAGE_API_KEY
    _validate_api_key(api_key)

    params = {
        "function": ALPHA_VANTAGE_DAILY_ADJUSTED_FUNCTION,
        "symbol": symbol,
        "outputsize": ALPHA_VANTAGE_OUTPUTSIZE,
        "entitlement": ALPHA_VANTAGE_ENTITLEMENT,
        "apikey": api_key,
    }

    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage returned an error for {symbol}: {data['Error Message']}")

    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage API note for {symbol}: {data['Note']}")

    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage API information for {symbol}: {data['Information']}")

    time_series_key = "Time Series (Daily)"
    if time_series_key not in data:
        raise KeyError(
            f"Could not find '{time_series_key}' in Alpha Vantage response for {symbol}. "
            f"Returned keys: {list(data.keys())}"
        )

    raw = pd.DataFrame.from_dict(data[time_series_key], orient="index")
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "Date"

    raw = raw.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adjusted_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend_amount",
            "8. split coefficient": "split_coefficient",
        }
    )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividend_amount",
        "split_coefficient",
    ]

    for col in numeric_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_index()

    filtered = raw.loc[START_DATE:END_DATE].copy()

    if filtered.empty:
        raise ValueError(
            f"No data returned for {symbol} between {START_DATE} and {END_DATE}."
        )

    filtered["symbol"] = symbol

    time.sleep(sleep_seconds)

    return filtered


def download_adjusted_price_panel(
    tickers: Iterable[str] = TICKERS,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices for all selected tickers and combine them.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame indexed by Date with one adjusted-close column per ticker.
    """
    price_series: Dict[str, pd.Series] = {}

    for ticker in tickers:
        print(f"Downloading daily adjusted prices for {ticker}...")
        ticker_df = fetch_daily_adjusted(ticker, api_key=api_key)
        price_series[ticker] = ticker_df["adjusted_close"].rename(ticker)

    prices = pd.concat(price_series.values(), axis=1)
    prices.index.name = "Date"
    prices = prices.sort_index()
    prices = prices.dropna(how="any")

    return prices


def save_price_panel(prices: pd.DataFrame, path=RAW_PRICE_FILE) -> None:
    """
    Save adjusted close price panel to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    print(f"Saved adjusted price panel to: {path}")


def load_cached_price_panel(path=RAW_PRICE_FILE) -> pd.DataFrame:
    """
    Load the cached adjusted-close price panel from CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cached price file not found: {path}")

    prices = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    prices = prices.sort_index()
    return prices


def load_or_download_price_panel(
    tickers: Iterable[str] = TICKERS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load cached prices if available; otherwise download from Alpha Vantage.
    """
    if RAW_PRICE_FILE.exists() and not force_refresh:
        print(f"Loading cached adjusted prices from: {RAW_PRICE_FILE}")
        return load_cached_price_panel(RAW_PRICE_FILE)

    prices = download_adjusted_price_panel(tickers=tickers)
    save_price_panel(prices, RAW_PRICE_FILE)

    return prices


if __name__ == "__main__":
    prices = load_or_download_price_panel(force_refresh=True)

    print("\nAdjusted price panel preview:")
    print(prices.head())
    print("\nLast rows:")
    print(prices.tail())
    print("\nShape:", prices.shape)
    print("Start date:", prices.index.min().date())
    print("End date:", prices.index.max().date())