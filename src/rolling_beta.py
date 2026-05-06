"""
Rolling CAPM beta calculations for the FDA1 Final Project.

This module supports Section 3.4:
- 252-day rolling CAPM beta
- rolling beta summary statistics
- high/low beta regime identification

The assignment permits the covariance/variance method:
beta_t = rolling_cov(stock excess return, market excess return) /
         rolling_var(market excess return)
"""

from __future__ import annotations

import pandas as pd

from src.config import ROLLING_BETA_FILE, ROLLING_WINDOW_BETA, TICKERS
from src.data_pipeline import load_or_build_analysis_dataset


def compute_rolling_capm_beta(
    dataset: pd.DataFrame,
    tickers=TICKERS,
    window: int = ROLLING_WINDOW_BETA,
) -> pd.DataFrame:
    """
    Compute rolling CAPM beta for each stock using rolling covariance/variance.

    Parameters
    ----------
    dataset : pd.DataFrame
        Analysis dataset with stock excess returns and Mkt-RF.
    tickers : list
        Tickers to compute rolling beta for.
    window : int
        Rolling window length, default 252 trading days.

    Returns
    -------
    pd.DataFrame
        Rolling beta DataFrame indexed by Date.
    """
    rolling_betas = pd.DataFrame(index=dataset.index)

    market_excess = dataset["Mkt-RF"]

    for ticker in tickers:
        stock_excess = dataset[f"{ticker}_excess_return"]

        rolling_cov = stock_excess.rolling(window).cov(market_excess)
        rolling_var = market_excess.rolling(window).var()

        rolling_betas[f"{ticker}_rolling_beta"] = rolling_cov / rolling_var

    rolling_betas.index.name = "Date"

    return rolling_betas


def summarize_rolling_betas(rolling_betas: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize rolling beta behavior for each stock.

    These statistics help support the report's discussion of whether each
    stock's beta was stable or time-varying.
    """
    rows = []

    for col in rolling_betas.columns:
        clean = rolling_betas[col].dropna()

        rows.append(
            {
                "series": col,
                "mean_rolling_beta": clean.mean(),
                "median_rolling_beta": clean.median(),
                "min_rolling_beta": clean.min(),
                "max_rolling_beta": clean.max(),
                "std_rolling_beta": clean.std(),
                "first_valid_date": clean.index.min(),
                "last_valid_date": clean.index.max(),
                "observations": int(clean.count()),
            }
        )

    return pd.DataFrame(rows)


def identify_extreme_rolling_beta_dates(
    rolling_betas: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identify highest and lowest rolling beta dates for each stock.

    This helps connect beta regime shifts to real-world market periods.
    """
    rows = []

    for col in rolling_betas.columns:
        clean = rolling_betas[col].dropna()

        highest = clean.sort_values(ascending=False).head(top_n)
        lowest = clean.sort_values(ascending=True).head(top_n)

        for date, value in highest.items():
            rows.append(
                {
                    "series": col,
                    "date": date,
                    "rolling_beta": value,
                    "extreme_type": "highest",
                }
            )

        for date, value in lowest.items():
            rows.append(
                {
                    "series": col,
                    "date": date,
                    "rolling_beta": value,
                    "extreme_type": "lowest",
                }
            )

    return pd.DataFrame(rows)


def save_rolling_betas(rolling_betas: pd.DataFrame, path=ROLLING_BETA_FILE) -> None:
    """
    Save rolling beta DataFrame to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rolling_betas.to_csv(path)
    print(f"Saved rolling CAPM betas to: {path}")


def load_rolling_betas(path=ROLLING_BETA_FILE) -> pd.DataFrame:
    """
    Load rolling beta DataFrame from CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Rolling beta file not found: {path}")

    rolling_betas = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    rolling_betas = rolling_betas.sort_index()
    return rolling_betas


if __name__ == "__main__":
    data = load_or_build_analysis_dataset()

    betas = compute_rolling_capm_beta(data)
    beta_summary = summarize_rolling_betas(betas)
    extreme_beta_dates = identify_extreme_rolling_beta_dates(betas)

    save_rolling_betas(betas)

    print("\nRolling beta preview:")
    print(betas.dropna().head())

    print("\nRolling beta summary:")
    print(beta_summary)

    print("\nExtreme rolling beta dates:")
    print(extreme_beta_dates)