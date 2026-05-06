"""
CAPM and Fama-French three-factor model utilities for the FDA1 Final Project.

This module supports Sections 3.2 and 3.3:
- CAPM regression
- FF3 regression
- model summary tables
- VIF for factor multicollinearity
- residual extraction for diagnostics
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.config import FACTOR_COLUMNS, TICKERS
from src.data_pipeline import load_or_build_analysis_dataset


def fit_capm(dataset: pd.DataFrame, ticker: str):
    """
    Fit CAPM regression for one stock.

    Model:
    stock excess return = alpha + beta * Mkt-RF + error
    """
    y = dataset[f"{ticker}_excess_return"]
    X = dataset[["Mkt-RF"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X, missing="drop").fit()
    return model


def fit_ff3(dataset: pd.DataFrame, ticker: str):
    """
    Fit Fama-French three-factor regression for one stock.

    Model:
    stock excess return = alpha + beta_mkt*Mkt-RF + beta_smb*SMB + beta_hml*HML + error
    """
    y = dataset[f"{ticker}_excess_return"]
    X = dataset[["Mkt-RF", "SMB", "HML"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X, missing="drop").fit()
    return model


def fit_all_capm_models(dataset: pd.DataFrame, tickers=TICKERS) -> Dict[str, object]:
    """
    Fit CAPM models for all selected tickers.
    """
    return {ticker: fit_capm(dataset, ticker) for ticker in tickers}


def fit_all_ff3_models(dataset: pd.DataFrame, tickers=TICKERS) -> Dict[str, object]:
    """
    Fit FF3 models for all selected tickers.
    """
    return {ticker: fit_ff3(dataset, ticker) for ticker in tickers}


def summarize_capm_models(models: Dict[str, object]) -> pd.DataFrame:
    """
    Create side-by-side CAPM summary table.

    Required project metrics:
    - alpha
    - beta
    - R-squared
    - p-value of alpha
    - p-value of beta
    """
    rows = []

    for ticker, model in models.items():
        rows.append(
            {
                "ticker": ticker,
                "alpha": model.params.get("const"),
                "market_beta": model.params.get("Mkt-RF"),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "alpha_p_value": model.pvalues.get("const"),
                "market_beta_p_value": model.pvalues.get("Mkt-RF"),
                "n_observations": int(model.nobs),
            }
        )

    return pd.DataFrame(rows)


def summarize_ff3_models(models: Dict[str, object]) -> pd.DataFrame:
    """
    Create side-by-side FF3 summary table.

    Required project metrics:
    - alpha
    - market beta
    - SMB loading
    - HML loading
    - R-squared
    - adjusted R-squared
    - p-values of each coefficient
    """
    rows = []

    for ticker, model in models.items():
        rows.append(
            {
                "ticker": ticker,
                "alpha": model.params.get("const"),
                "market_beta": model.params.get("Mkt-RF"),
                "smb_loading": model.params.get("SMB"),
                "hml_loading": model.params.get("HML"),
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "alpha_p_value": model.pvalues.get("const"),
                "market_beta_p_value": model.pvalues.get("Mkt-RF"),
                "smb_p_value": model.pvalues.get("SMB"),
                "hml_p_value": model.pvalues.get("HML"),
                "n_observations": int(model.nobs),
            }
        )

    return pd.DataFrame(rows)


def compute_factor_vif(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for FF3 factor predictors.

    The constant column is excluded from VIF reporting, as required by the
    project technical reminders.
    """
    X = dataset[["Mkt-RF", "SMB", "HML"]].dropna().copy()

    rows = []

    for i, col in enumerate(X.columns):
        rows.append(
            {
                "factor": col,
                "vif": variance_inflation_factor(X.values, i),
            }
        )

    return pd.DataFrame(rows)


def extract_model_residuals(models: Dict[str, object], suffix: str) -> pd.DataFrame:
    """
    Extract residuals from fitted models into one DataFrame.

    Parameters
    ----------
    models : dict
        Dictionary mapping ticker to fitted statsmodels result.
    suffix : str
        Suffix describing model type, such as 'capm' or 'ff3'.
    """
    residuals = pd.DataFrame()

    for ticker, model in models.items():
        residuals[f"{ticker}_{suffix}_residual"] = model.resid

    residuals.index.name = "Date"
    return residuals


def compare_model_r_squared(
    capm_summary: pd.DataFrame,
    ff3_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare CAPM and FF3 explanatory power for each stock.
    """
    capm = capm_summary[["ticker", "r_squared", "adj_r_squared"]].rename(
        columns={
            "r_squared": "capm_r_squared",
            "adj_r_squared": "capm_adj_r_squared",
        }
    )

    ff3 = ff3_summary[["ticker", "r_squared", "adj_r_squared"]].rename(
        columns={
            "r_squared": "ff3_r_squared",
            "adj_r_squared": "ff3_adj_r_squared",
        }
    )

    comparison = capm.merge(ff3, on="ticker", how="inner")
    comparison["r_squared_improvement"] = (
        comparison["ff3_r_squared"] - comparison["capm_r_squared"]
    )
    comparison["adj_r_squared_improvement"] = (
        comparison["ff3_adj_r_squared"] - comparison["capm_adj_r_squared"]
    )

    return comparison


if __name__ == "__main__":
    data = load_or_build_analysis_dataset()

    capm_models = fit_all_capm_models(data)
    ff3_models = fit_all_ff3_models(data)

    capm_summary = summarize_capm_models(capm_models)
    ff3_summary = summarize_ff3_models(ff3_models)
    vif_summary = compute_factor_vif(data)
    comparison = compare_model_r_squared(capm_summary, ff3_summary)

    print("\nCAPM Summary:")
    print(capm_summary)

    print("\nFF3 Summary:")
    print(ff3_summary)

    print("\nFactor VIF:")
    print(vif_summary)

    print("\nModel R-squared Comparison:")
    print(comparison)