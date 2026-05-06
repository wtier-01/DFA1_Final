"""
Time-series analysis utilities for the FDA1 Final Project.

This module supports Section 3.1 and part of Section 3.4:
- daily return characterization
- rolling mean and volatility
- ADF stationarity tests
- descriptive statistics
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import adfuller

from src.config import (
    ROLLING_WINDOW_MEAN_VOL,
    TRADING_DAYS_PER_YEAR,
    TICKERS,
)
from src.data_pipeline import load_or_build_analysis_dataset


def get_price_columns(tickers=TICKERS) -> list[str]:
    """
    Return adjusted price column names.
    """
    return [f"{ticker}_price" for ticker in tickers]


def get_return_columns(tickers=TICKERS) -> list[str]:
    """
    Return daily simple return column names.
    """
    return [f"{ticker}_return" for ticker in tickers]


def compute_rolling_statistics(
    dataset: pd.DataFrame,
    tickers=TICKERS,
    window: int = ROLLING_WINDOW_MEAN_VOL,
) -> pd.DataFrame:
    """
    Compute rolling 60-day mean return and annualized volatility.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling mean and rolling annualized volatility columns.
    """
    rolling = pd.DataFrame(index=dataset.index)

    for ticker in tickers:
        return_col = f"{ticker}_return"

        rolling[f"{ticker}_rolling_{window}d_mean"] = dataset[return_col].rolling(window).mean()

        rolling[f"{ticker}_rolling_{window}d_ann_vol"] = (
            dataset[return_col].rolling(window).std() * (TRADING_DAYS_PER_YEAR ** 0.5)
        )

    return rolling


def run_adf_test(series: pd.Series, label: str) -> dict:
    """
    Run Augmented Dickey-Fuller stationarity test on a single series.

    Interpretation:
    - Small p-value, usually below 0.05, means reject unit-root null.
    - Rejecting unit-root null implies the series is stationary.
    """
    clean = series.dropna()

    if clean.empty:
        raise ValueError(f"Series is empty after dropping NaN values: {label}")

    result = adfuller(clean, autolag="AIC")

    adf_stat = result[0]
    p_value = result[1]
    used_lags = result[2]
    n_obs = result[3]
    critical_values = result[4]

    return {
        "series": label,
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "used_lags": used_lags,
        "n_observations": n_obs,
        "critical_1pct": critical_values.get("1%"),
        "critical_5pct": critical_values.get("5%"),
        "critical_10pct": critical_values.get("10%"),
        "stationary_at_5pct": p_value < 0.05,
    }


def run_price_return_adf_suite(dataset: pd.DataFrame, tickers=TICKERS) -> pd.DataFrame:
    """
    Run ADF tests on each stock's price series and return series.
    """
    rows = []

    for ticker in tickers:
        rows.append(run_adf_test(dataset[f"{ticker}_price"], f"{ticker} adjusted price"))
        rows.append(run_adf_test(dataset[f"{ticker}_return"], f"{ticker} daily return"))

    return pd.DataFrame(rows)


def compute_return_descriptive_statistics(dataset: pd.DataFrame, tickers=TICKERS) -> pd.DataFrame:
    """
    Compute descriptive statistics for daily returns.

    Includes above-and-beyond metrics:
    - annualized mean
    - annualized volatility
    - skewness
    - excess kurtosis
    - min/max daily return
    """
    rows = []

    for ticker in tickers:
        r = dataset[f"{ticker}_return"].dropna()

        rows.append(
            {
                "ticker": ticker,
                "mean_daily_return": r.mean(),
                "median_daily_return": r.median(),
                "annualized_mean_return": r.mean() * TRADING_DAYS_PER_YEAR,
                "daily_volatility": r.std(),
                "annualized_volatility": r.std() * (TRADING_DAYS_PER_YEAR ** 0.5),
                "min_daily_return": r.min(),
                "max_daily_return": r.max(),
                "skewness": skew(r),
                "excess_kurtosis": kurtosis(r, fisher=True),
                "observations": int(r.count()),
            }
        )

    return pd.DataFrame(rows)


def identify_high_volatility_periods(
    rolling_stats: pd.DataFrame,
    tickers=TICKERS,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identify highest rolling-volatility dates for each stock.

    This helps support the report requirement to identify visible volatility
    regime shifts and connect them to real-world events.
    """
    rows = []

    for ticker in tickers:
        vol_col = f"{ticker}_rolling_{ROLLING_WINDOW_MEAN_VOL}d_ann_vol"
        top_dates = rolling_stats[vol_col].dropna().sort_values(ascending=False).head(top_n)

        for date, value in top_dates.items():
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "rolling_annualized_volatility": value,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = load_or_build_analysis_dataset()

    adf_summary = run_price_return_adf_suite(data)
    rolling_stats = compute_rolling_statistics(data)
    desc_stats = compute_return_descriptive_statistics(data)
    high_vol_periods = identify_high_volatility_periods(rolling_stats)

    print("\nADF Summary:")
    print(adf_summary)

    print("\nReturn Descriptive Statistics:")
    print(desc_stats)

    print("\nHighest Rolling Volatility Periods:")
    print(high_vol_periods)