"""
Main data pipeline for the FDA1 Final Project.

This module combines:
1. Alpha Vantage adjusted stock prices
2. Daily simple stock returns
3. Daily Fama-French factors and risk-free rate
4. Daily stock excess returns

The final processed dataset is saved to:
data/processed/analysis_dataset.csv

This single file is designed to contain everything needed for the notebook:
- adjusted close prices
- simple returns
- Fama-French factors
- risk-free rate
- excess returns
"""

from __future__ import annotations

import pandas as pd

from src.alpha_vantage_client import load_or_download_price_panel
from src.config import (
    END_DATE,
    FACTOR_COLUMNS,
    PROCESSED_ANALYSIS_FILE,
    START_DATE,
    TICKERS,
)
from src.fama_french_loader import load_or_download_factors


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns using pct_change().
    """
    returns = prices.pct_change()
    returns = returns.rename(columns={ticker: f"{ticker}_return" for ticker in prices.columns})
    return returns


def build_analysis_dataset(force_refresh_prices: bool = False, force_refresh_factors: bool = False) -> pd.DataFrame:
    """
    Build the fully aligned analysis dataset.

    Returns
    -------
    pd.DataFrame
        Dataset indexed by Date with:
        - adjusted close price columns
        - return columns
        - Fama-French factor columns
        - excess return columns
    """
    prices = load_or_download_price_panel(force_refresh=force_refresh_prices)
    factors = load_or_download_factors(force_refresh=force_refresh_factors)

    price_cols = prices.rename(columns={ticker: f"{ticker}_price" for ticker in prices.columns})
    returns = compute_simple_returns(prices)

    dataset = price_cols.join(returns, how="inner")
    dataset = dataset.join(factors[FACTOR_COLUMNS], how="inner")

    for ticker in TICKERS:
        dataset[f"{ticker}_excess_return"] = dataset[f"{ticker}_return"] - dataset["RF"]

    dataset = dataset.loc[START_DATE:END_DATE].dropna().copy()
    dataset.index.name = "Date"

    if dataset.empty:
        raise ValueError("Processed analysis dataset is empty after merging and dropping NaN values.")

    return dataset


def save_analysis_dataset(dataset: pd.DataFrame, path=PROCESSED_ANALYSIS_FILE) -> None:
    """
    Save processed analysis dataset to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path)
    print(f"Saved processed analysis dataset to: {path}")


def load_analysis_dataset(path=PROCESSED_ANALYSIS_FILE) -> pd.DataFrame:
    """
    Load processed analysis dataset from CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Processed analysis dataset not found: {path}")

    dataset = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    dataset = dataset.sort_index()
    return dataset


def load_or_build_analysis_dataset(
    force_rebuild: bool = False,
    force_refresh_prices: bool = False,
    force_refresh_factors: bool = False,
) -> pd.DataFrame:
    """
    Load cached processed dataset if available; otherwise build it.
    """
    if PROCESSED_ANALYSIS_FILE.exists() and not force_rebuild:
        print(f"Loading processed analysis dataset from: {PROCESSED_ANALYSIS_FILE}")
        return load_analysis_dataset(PROCESSED_ANALYSIS_FILE)

    dataset = build_analysis_dataset(
        force_refresh_prices=force_refresh_prices,
        force_refresh_factors=force_refresh_factors,
    )
    save_analysis_dataset(dataset)

    return dataset


def validate_analysis_dataset(dataset: pd.DataFrame) -> None:
    """
    Run basic validation checks on the final dataset.
    """
    required_columns = []

    for ticker in TICKERS:
        required_columns.extend(
            [
                f"{ticker}_price",
                f"{ticker}_return",
                f"{ticker}_excess_return",
            ]
        )

    required_columns.extend(FACTOR_COLUMNS)

    missing = [col for col in required_columns if col not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if dataset.index.min() > pd.Timestamp(START_DATE):
        print(
            f"Note: first valid merged trading date is {dataset.index.min().date()}, "
            f"which is expected if {START_DATE} was not a trading day."
        )

    if dataset.index.max() < pd.Timestamp(END_DATE):
        raise ValueError(
            f"Dataset ends on {dataset.index.max().date()}, but must end at or near {END_DATE}."
        )

    if dataset.isna().sum().sum() > 0:
        raise ValueError("Dataset contains NaN values after cleaning.")

    print("\nDataset validation passed.")
    print("Shape:", dataset.shape)
    print("Start date:", dataset.index.min().date())
    print("End date:", dataset.index.max().date())
    print("Columns:")
    print(list(dataset.columns))


if __name__ == "__main__":
    analysis_data = load_or_build_analysis_dataset(
        force_rebuild=True,
        force_refresh_prices=False,
        force_refresh_factors=False,
    )

    validate_analysis_dataset(analysis_data)

    print("\nAnalysis dataset preview:")
    print(analysis_data.head())

    print("\nAnalysis dataset tail:")
    print(analysis_data.tail())