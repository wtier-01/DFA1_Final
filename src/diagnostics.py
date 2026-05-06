"""
Residual and autocorrelation diagnostics for the FDA1 Final Project.

This module supports:
- CAPM residual diagnostics
- FF3 residual diagnostics
- residual stationarity testing
- residual autocorrelation checks
- squared residual volatility clustering checks

The project requires ADF tests on FF3 residuals, ACF/PACF plots of FF3
residuals, and ACF plots of squared FF3 residuals. This module computes the
tables behind those figures and interpretations.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller

from src.config import ACF_PACF_LAGS, TICKERS
from src.data_pipeline import load_or_build_analysis_dataset
from src.factor_models import (
    extract_model_residuals,
    fit_all_capm_models,
    fit_all_ff3_models,
)


def run_residual_adf_test(residual_series: pd.Series, label: str) -> dict:
    """
    Run Augmented Dickey-Fuller test on regression residuals.

    A well-specified regression model should generally produce residuals
    that are stationary around zero.
    """
    clean = residual_series.dropna()

    if clean.empty:
        raise ValueError(f"Residual series is empty: {label}")

    result = adfuller(clean, autolag="AIC")

    return {
        "series": label,
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lags": result[2],
        "n_observations": result[3],
        "critical_1pct": result[4].get("1%"),
        "critical_5pct": result[4].get("5%"),
        "critical_10pct": result[4].get("10%"),
        "stationary_at_5pct": result[1] < 0.05,
    }


def run_residual_adf_suite(residuals: pd.DataFrame) -> pd.DataFrame:
    """
    Run ADF tests for every residual series in a residual DataFrame.
    """
    rows = []

    for col in residuals.columns:
        rows.append(run_residual_adf_test(residuals[col], col))

    return pd.DataFrame(rows)


def compute_residual_distribution_stats(residuals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution diagnostics for residuals.

    Above-and-beyond metrics:
    - skewness
    - excess kurtosis
    - Jarque-Bera statistic and p-value
    - residual mean and standard deviation
    """
    rows = []

    for col in residuals.columns:
        clean = residuals[col].dropna()

        jb_result = jarque_bera(clean)

        rows.append(
            {
                "series": col,
                "mean": clean.mean(),
                "std_dev": clean.std(),
                "min": clean.min(),
                "max": clean.max(),
                "skewness": skew(clean),
                "excess_kurtosis": kurtosis(clean, fisher=True),
                "jarque_bera_statistic": jb_result.statistic,
                "jarque_bera_p_value": jb_result.pvalue,
                "approximately_normal_at_5pct": jb_result.pvalue >= 0.05,
                "observations": int(clean.count()),
            }
        )

    return pd.DataFrame(rows)


def compute_acf_snapshot(
    series: pd.Series,
    label: str,
    nlags: int = ACF_PACF_LAGS,
) -> pd.DataFrame:
    """
    Compute autocorrelation values for a series up to the requested lag.

    This is used for table-based support behind the visual ACF plots.
    """
    clean = series.dropna()

    if clean.empty:
        raise ValueError(f"Series is empty after dropping NaN values: {label}")

    values = acf(clean, nlags=nlags, fft=True)

    return pd.DataFrame(
        {
            "series": label,
            "lag": range(len(values)),
            "acf": values,
        }
    )


def compute_squared_acf_snapshot(
    series: pd.Series,
    label: str,
    nlags: int = ACF_PACF_LAGS,
) -> pd.DataFrame:
    """
    Compute ACF values for squared series.

    Persistent autocorrelation in squared returns or squared residuals is
    evidence of volatility clustering.
    """
    clean_squared = series.dropna() ** 2
    return compute_acf_snapshot(clean_squared, f"{label}_squared", nlags=nlags)


def compute_ljung_box_summary(
    residuals: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run Ljung-Box tests for autocorrelation in residuals.

    This is above-and-beyond. The assignment requires visual ACF/PACF checks,
    but this table gives a more formal diagnostic.
    """
    if lags is None:
        lags = [5, 10, 20, 40]

    rows = []

    for col in residuals.columns:
        clean = residuals[col].dropna()

        lb = acorr_ljungbox(clean, lags=lags, return_df=True)

        for lag, row in lb.iterrows():
            rows.append(
                {
                    "series": col,
                    "lag": lag,
                    "ljung_box_statistic": row["lb_stat"],
                    "p_value": row["lb_pvalue"],
                    "autocorrelation_detected_at_5pct": row["lb_pvalue"] < 0.05,
                }
            )

    return pd.DataFrame(rows)


def compute_squared_ljung_box_summary(
    residuals: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run Ljung-Box tests on squared residuals.

    This checks whether volatility clustering remains after the factor model.
    """
    if lags is None:
        lags = [5, 10, 20, 40]

    rows = []

    for col in residuals.columns:
        clean_squared = residuals[col].dropna() ** 2

        lb = acorr_ljungbox(clean_squared, lags=lags, return_df=True)

        for lag, row in lb.iterrows():
            rows.append(
                {
                    "series": f"{col}_squared",
                    "lag": lag,
                    "ljung_box_statistic": row["lb_stat"],
                    "p_value": row["lb_pvalue"],
                    "volatility_clustering_detected_at_5pct": row["lb_pvalue"] < 0.05,
                }
            )

    return pd.DataFrame(rows)


def build_residual_diagnostic_tables(
    capm_residuals: pd.DataFrame,
    ff3_residuals: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Build all residual diagnostic tables used in the report/notebook.
    """
    return {
        "capm_residual_distribution": compute_residual_distribution_stats(capm_residuals),
        "ff3_residual_distribution": compute_residual_distribution_stats(ff3_residuals),
        "ff3_residual_adf": run_residual_adf_suite(ff3_residuals),
        "ff3_residual_ljung_box": compute_ljung_box_summary(ff3_residuals),
        "ff3_squared_residual_ljung_box": compute_squared_ljung_box_summary(ff3_residuals),
    }


if __name__ == "__main__":
    data = load_or_build_analysis_dataset()

    capm_models = fit_all_capm_models(data, TICKERS)
    ff3_models = fit_all_ff3_models(data, TICKERS)

    capm_residuals = extract_model_residuals(capm_models, suffix="capm")
    ff3_residuals = extract_model_residuals(ff3_models, suffix="ff3")

    diagnostic_tables = build_residual_diagnostic_tables(capm_residuals, ff3_residuals)

    for name, table in diagnostic_tables.items():
        print(f"\n{name}:")
        print(table)