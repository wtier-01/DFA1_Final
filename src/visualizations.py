"""
Visualization utilities for the FDA1 Final Project.

This module generates all major required figures:
- adjusted price plots
- daily return plots
- rolling 60-day mean return plots
- rolling 60-day annualized volatility plots
- ACF/PACF plots for returns
- ACF plots for squared returns
- CAPM and FF3 residual diagnostics
- ACF/PACF plots for FF3 residuals
- ACF plots for squared FF3 residuals
- rolling 252-day CAPM beta plot

Every figure is saved to outputs/figures with clear titles, labeled axes,
and legends where appropriate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.config import ACF_PACF_LAGS, FIGURES_DIR, ROLLING_WINDOW_MEAN_VOL, TICKERS
from src.data_pipeline import load_or_build_analysis_dataset
from src.factor_models import extract_model_residuals, fit_all_capm_models, fit_all_ff3_models
from src.rolling_beta import compute_rolling_capm_beta
from src.time_series_analysis import compute_rolling_statistics


def _ensure_figures_dir() -> None:
    """
    Ensure the figures output directory exists.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save_current_figure(filename: str, dpi: int = 300) -> Path:
    """
    Save the current Matplotlib figure.
    """
    _ensure_figures_dir()
    path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_adjusted_prices(dataset: pd.DataFrame, tickers=TICKERS) -> Path:
    """
    Plot adjusted closing prices for both stocks on one chart.
    """
    plt.figure(figsize=(12, 6))

    for ticker in tickers:
        plt.plot(dataset.index, dataset[f"{ticker}_price"], label=ticker)

    plt.title("Adjusted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return _save_current_figure("01_adjusted_closing_prices.png")


def plot_daily_returns(dataset: pd.DataFrame, tickers=TICKERS) -> Path:
    """
    Plot daily simple returns for both stocks.
    """
    plt.figure(figsize=(12, 6))

    for ticker in tickers:
        plt.plot(dataset.index, dataset[f"{ticker}_return"], label=ticker, alpha=0.75)

    plt.axhline(0, linewidth=1, linestyle="--")
    plt.title("Daily Simple Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return _save_current_figure("02_daily_simple_returns.png")


def plot_rolling_mean_returns(
    rolling_stats: pd.DataFrame,
    tickers=TICKERS,
    window: int = ROLLING_WINDOW_MEAN_VOL,
) -> Path:
    """
    Plot rolling mean daily returns.
    """
    plt.figure(figsize=(12, 6))

    for ticker in tickers:
        col = f"{ticker}_rolling_{window}d_mean"
        plt.plot(rolling_stats.index, rolling_stats[col], label=ticker)

    plt.axhline(0, linewidth=1, linestyle="--")
    plt.title(f"Rolling {window}-Day Mean Daily Return")
    plt.xlabel("Date")
    plt.ylabel("Rolling Mean Daily Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return _save_current_figure("03_rolling_60d_mean_returns.png")


def plot_rolling_annualized_volatility(
    rolling_stats: pd.DataFrame,
    tickers=TICKERS,
    window: int = ROLLING_WINDOW_MEAN_VOL,
) -> Path:
    """
    Plot rolling annualized volatility.
    """
    plt.figure(figsize=(12, 6))

    for ticker in tickers:
        col = f"{ticker}_rolling_{window}d_ann_vol"
        plt.plot(rolling_stats.index, rolling_stats[col], label=ticker)

    plt.title(f"Rolling {window}-Day Annualized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return _save_current_figure("04_rolling_60d_annualized_volatility.png")


def plot_return_acf_pacf(dataset: pd.DataFrame, tickers=TICKERS, lags: int = ACF_PACF_LAGS) -> list[Path]:
    """
    Plot ACF and PACF of daily returns for each stock.
    """
    paths = []

    for ticker in tickers:
        series = dataset[f"{ticker}_return"].dropna()

        plt.figure(figsize=(10, 5))
        plot_acf(series, lags=lags, ax=plt.gca())
        plt.title(f"{ticker} Daily Returns ACF")
        paths.append(_save_current_figure(f"05_{ticker}_returns_acf.png"))

        plt.figure(figsize=(10, 5))
        plot_pacf(series, lags=lags, ax=plt.gca(), method="ywm")
        plt.title(f"{ticker} Daily Returns PACF")
        paths.append(_save_current_figure(f"06_{ticker}_returns_pacf.png"))

    return paths


def plot_squared_return_acf(dataset: pd.DataFrame, tickers=TICKERS, lags: int = ACF_PACF_LAGS) -> list[Path]:
    """
    Plot ACF of squared daily returns for volatility clustering checks.
    """
    paths = []

    for ticker in tickers:
        series = dataset[f"{ticker}_return"].dropna() ** 2

        plt.figure(figsize=(10, 5))
        plot_acf(series, lags=lags, ax=plt.gca())
        plt.title(f"{ticker} Squared Daily Returns ACF")
        paths.append(_save_current_figure(f"07_{ticker}_squared_returns_acf.png"))

    return paths


def plot_residual_time_series(residuals: pd.DataFrame, model_name: str) -> list[Path]:
    """
    Plot regression residuals over time.
    """
    paths = []

    for col in residuals.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(residuals.index, residuals[col], label=col)
        plt.axhline(0, linewidth=1, linestyle="--")
        plt.title(f"{col} Residuals Over Time")
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.legend()
        plt.grid(True, alpha=0.3)

        paths.append(_save_current_figure(f"08_{model_name}_{col}_residuals_time_series.png"))

    return paths


def plot_residual_histograms(residuals: pd.DataFrame, model_name: str) -> list[Path]:
    """
    Plot regression residual histograms.
    """
    paths = []

    for col in residuals.columns:
        clean = residuals[col].dropna()

        plt.figure(figsize=(9, 5))
        plt.hist(clean, bins=50, edgecolor="black", alpha=0.8)
        plt.axvline(clean.mean(), linewidth=1, linestyle="--", label="Mean")
        plt.title(f"{col} Residual Histogram")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        paths.append(_save_current_figure(f"09_{model_name}_{col}_residual_histogram.png"))

    return paths


def plot_residual_qq(residuals: pd.DataFrame, model_name: str) -> list[Path]:
    """
    Plot Q-Q plots of regression residuals against a normal distribution.
    """
    paths = []

    for col in residuals.columns:
        clean = residuals[col].dropna()

        plt.figure(figsize=(7, 7))
        stats.probplot(clean, dist="norm", plot=plt)
        plt.title(f"{col} Residual Q-Q Plot")

        paths.append(_save_current_figure(f"10_{model_name}_{col}_residual_qq.png"))

    return paths


def plot_ff3_residual_acf_pacf(
    ff3_residuals: pd.DataFrame,
    lags: int = ACF_PACF_LAGS,
) -> list[Path]:
    """
    Plot ACF and PACF of FF3 residuals.
    """
    paths = []

    for col in ff3_residuals.columns:
        series = ff3_residuals[col].dropna()

        plt.figure(figsize=(10, 5))
        plot_acf(series, lags=lags, ax=plt.gca())
        plt.title(f"{col} ACF")
        paths.append(_save_current_figure(f"11_{col}_acf.png"))

        plt.figure(figsize=(10, 5))
        plot_pacf(series, lags=lags, ax=plt.gca(), method="ywm")
        plt.title(f"{col} PACF")
        paths.append(_save_current_figure(f"12_{col}_pacf.png"))

    return paths


def plot_squared_ff3_residual_acf(
    ff3_residuals: pd.DataFrame,
    lags: int = ACF_PACF_LAGS,
) -> list[Path]:
    """
    Plot ACF of squared FF3 residuals.
    """
    paths = []

    for col in ff3_residuals.columns:
        series = ff3_residuals[col].dropna() ** 2

        plt.figure(figsize=(10, 5))
        plot_acf(series, lags=lags, ax=plt.gca())
        plt.title(f"{col} Squared Residual ACF")
        paths.append(_save_current_figure(f"13_{col}_squared_acf.png"))

    return paths


def plot_rolling_capm_betas(rolling_betas: pd.DataFrame) -> Path:
    """
    Plot both stocks' rolling 252-day CAPM betas on one chart.
    """
    plt.figure(figsize=(12, 6))

    for col in rolling_betas.columns:
        plt.plot(rolling_betas.index, rolling_betas[col], label=col.replace("_rolling_beta", ""))

    plt.axhline(1.0, linewidth=1.2, linestyle="--", label="Beta = 1")
    plt.title("Rolling 252-Day CAPM Beta")
    plt.xlabel("Date")
    plt.ylabel("Rolling CAPM Beta")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return _save_current_figure("14_rolling_252d_capm_beta.png")


def generate_all_figures(
    dataset: pd.DataFrame,
    capm_residuals: pd.DataFrame,
    ff3_residuals: pd.DataFrame,
    rolling_betas: pd.DataFrame,
) -> list[Path]:
    """
    Generate all required figures for the report and notebook.
    """
    figure_paths = []

    rolling_stats = compute_rolling_statistics(dataset)

    figure_paths.append(plot_adjusted_prices(dataset))
    figure_paths.append(plot_daily_returns(dataset))
    figure_paths.append(plot_rolling_mean_returns(rolling_stats))
    figure_paths.append(plot_rolling_annualized_volatility(rolling_stats))

    figure_paths.extend(plot_return_acf_pacf(dataset))
    figure_paths.extend(plot_squared_return_acf(dataset))

    figure_paths.extend(plot_residual_time_series(capm_residuals, model_name="capm"))
    figure_paths.extend(plot_residual_histograms(capm_residuals, model_name="capm"))
    figure_paths.extend(plot_residual_qq(capm_residuals, model_name="capm"))

    figure_paths.extend(plot_residual_time_series(ff3_residuals, model_name="ff3"))
    figure_paths.extend(plot_residual_histograms(ff3_residuals, model_name="ff3"))
    figure_paths.extend(plot_residual_qq(ff3_residuals, model_name="ff3"))

    figure_paths.extend(plot_ff3_residual_acf_pacf(ff3_residuals))
    figure_paths.extend(plot_squared_ff3_residual_acf(ff3_residuals))

    figure_paths.append(plot_rolling_capm_betas(rolling_betas))

    return figure_paths


if __name__ == "__main__":
    data = load_or_build_analysis_dataset()

    capm_models = fit_all_capm_models(data)
    ff3_models = fit_all_ff3_models(data)

    capm_residuals = extract_model_residuals(capm_models, suffix="capm")
    ff3_residuals = extract_model_residuals(ff3_models, suffix="ff3")

    rolling_betas = compute_rolling_capm_beta(data)

    paths = generate_all_figures(data, capm_residuals, ff3_residuals, rolling_betas)

    print("\nGenerated figures:")
    for path in paths:
        print(path)