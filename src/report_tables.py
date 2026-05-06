"""
Report-ready table generation for the FDA1 Final Project.

This module creates all major CSV and Excel tables needed for the notebook
and Word report.

Tables generated:
- ADF tests for prices and returns
- return descriptive statistics
- high rolling-volatility periods
- CAPM summary
- FF3 summary
- factor VIF
- CAPM vs FF3 model comparison
- FF3 residual ADF
- residual distribution diagnostics
- residual autocorrelation diagnostics
- rolling beta summary
- extreme rolling beta dates
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import (
    ADF_SUMMARY_FILE,
    CAPM_SUMMARY_FILE,
    FF3_SUMMARY_FILE,
    TABLES_DIR,
    TICKERS,
    VIF_SUMMARY_FILE,
)
from src.data_pipeline import load_or_build_analysis_dataset
from src.diagnostics import build_residual_diagnostic_tables
from src.factor_models import (
    compare_model_r_squared,
    compute_factor_vif,
    extract_model_residuals,
    fit_all_capm_models,
    fit_all_ff3_models,
    summarize_capm_models,
    summarize_ff3_models,
)
from src.rolling_beta import (
    compute_rolling_capm_beta,
    identify_extreme_rolling_beta_dates,
    save_rolling_betas,
    summarize_rolling_betas,
)
from src.time_series_analysis import (
    compute_return_descriptive_statistics,
    compute_rolling_statistics,
    identify_high_volatility_periods,
    run_price_return_adf_suite,
)


def _round_table(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    """
    Round numeric columns for cleaner report output.
    """
    rounded = df.copy()

    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(decimals)

    return rounded


def save_table(df: pd.DataFrame, filename: str, decimals: int = 6) -> Path:
    """
    Save one table to outputs/tables.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    path = TABLES_DIR / filename
    _round_table(df, decimals=decimals).to_csv(path, index=False)

    return path


def build_all_report_tables(dataset: pd.DataFrame | None = None) -> Dict[str, pd.DataFrame]:
    """
    Build all report-ready tables.

    Parameters
    ----------
    dataset : pd.DataFrame, optional
        If None, loads or builds the processed analysis dataset.

    Returns
    -------
    dict
        Dictionary mapping table names to DataFrames.
    """
    if dataset is None:
        dataset = load_or_build_analysis_dataset()

    rolling_stats = compute_rolling_statistics(dataset)

    capm_models = fit_all_capm_models(dataset, TICKERS)
    ff3_models = fit_all_ff3_models(dataset, TICKERS)

    capm_summary = summarize_capm_models(capm_models)
    ff3_summary = summarize_ff3_models(ff3_models)

    capm_residuals = extract_model_residuals(capm_models, suffix="capm")
    ff3_residuals = extract_model_residuals(ff3_models, suffix="ff3")

    residual_tables = build_residual_diagnostic_tables(capm_residuals, ff3_residuals)

    rolling_betas = compute_rolling_capm_beta(dataset)
    save_rolling_betas(rolling_betas)

    tables = {
        "adf_price_return_summary": run_price_return_adf_suite(dataset),
        "return_descriptive_statistics": compute_return_descriptive_statistics(dataset),
        "high_rolling_volatility_periods": identify_high_volatility_periods(rolling_stats),
        "capm_summary": capm_summary,
        "ff3_summary": ff3_summary,
        "factor_vif": compute_factor_vif(dataset),
        "model_r_squared_comparison": compare_model_r_squared(capm_summary, ff3_summary),
        "rolling_beta_summary": summarize_rolling_betas(rolling_betas),
        "extreme_rolling_beta_dates": identify_extreme_rolling_beta_dates(rolling_betas),
    }

    tables.update(residual_tables)

    return tables


def save_all_report_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
    """
    Save all report tables as individual CSV files.
    """
    saved_paths = {}

    filename_map = {
        "adf_price_return_summary": "adf_price_return_summary.csv",
        "return_descriptive_statistics": "return_descriptive_statistics.csv",
        "high_rolling_volatility_periods": "high_rolling_volatility_periods.csv",
        "capm_summary": "capm_summary.csv",
        "ff3_summary": "ff3_summary.csv",
        "factor_vif": "factor_vif.csv",
        "model_r_squared_comparison": "model_r_squared_comparison.csv",
        "rolling_beta_summary": "rolling_beta_summary.csv",
        "extreme_rolling_beta_dates": "extreme_rolling_beta_dates.csv",
        "capm_residual_distribution": "capm_residual_distribution.csv",
        "ff3_residual_distribution": "ff3_residual_distribution.csv",
        "ff3_residual_adf": "ff3_residual_adf.csv",
        "ff3_residual_ljung_box": "ff3_residual_ljung_box.csv",
        "ff3_squared_residual_ljung_box": "ff3_squared_residual_ljung_box.csv",
    }

    for table_name, df in tables.items():
        filename = filename_map.get(table_name, f"{table_name}.csv")
        saved_paths[table_name] = save_table(df, filename)

    return saved_paths


def save_tables_to_excel_workbook(
    tables: Dict[str, pd.DataFrame],
    filename: str = "report_tables_workbook.xlsx",
) -> Path:
    """
    Save all report tables to one Excel workbook with separate sheets.

    This is useful for quickly copying tables into the Word report.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for table_name, df in tables.items():
            sheet_name = table_name[:31]
            _round_table(df).to_excel(writer, sheet_name=sheet_name, index=False)

    return path


def save_core_required_tables(
    tables: Dict[str, pd.DataFrame],
) -> None:
    """
    Save key required tables to the specific filenames already listed in config.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tables["adf_price_return_summary"].to_csv(ADF_SUMMARY_FILE, index=False)
    tables["capm_summary"].to_csv(CAPM_SUMMARY_FILE, index=False)
    tables["ff3_summary"].to_csv(FF3_SUMMARY_FILE, index=False)
    tables["factor_vif"].to_csv(VIF_SUMMARY_FILE, index=False)


if __name__ == "__main__":
    data = load_or_build_analysis_dataset()

    report_tables = build_all_report_tables(data)
    saved_csvs = save_all_report_tables(report_tables)
    workbook_path = save_tables_to_excel_workbook(report_tables)
    save_core_required_tables(report_tables)

    print("\nSaved report tables:")
    for name, path in saved_csvs.items():
        print(f"{name}: {path}")

    print(f"\nSaved Excel workbook: {workbook_path}")