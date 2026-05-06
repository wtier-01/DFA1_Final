"""
Main project runner for the FDA1 Final Project.

Run this file to generate:
- processed analysis dataset
- CAPM and FF3 model outputs
- rolling beta data
- report-ready tables
- all required figures

Command:
python main.py
"""

from __future__ import annotations

from src.config import (
    FIGURES_DIR,
    PROCESSED_ANALYSIS_FILE,
    RAW_PRICE_FILE,
    TABLES_DIR,
    TICKERS,
)
from src.data_pipeline import load_or_build_analysis_dataset, validate_analysis_dataset
from src.factor_models import extract_model_residuals, fit_all_capm_models, fit_all_ff3_models
from src.report_tables import (
    build_all_report_tables,
    save_all_report_tables,
    save_core_required_tables,
    save_tables_to_excel_workbook,
)
from src.rolling_beta import compute_rolling_capm_beta, save_rolling_betas
from src.visualizations import generate_all_figures


def run_project_pipeline() -> None:
    """
    Execute the full project pipeline from data to output.
    """
    print("=" * 80)
    print("FDA1 Final Project Pipeline")
    print("=" * 80)

    print("\nSelected tickers:")
    for ticker in TICKERS:
        print(f"- {ticker}")

    print("\nStep 1: Loading or building processed analysis dataset...")
    dataset = load_or_build_analysis_dataset(
        force_rebuild=False,
        force_refresh_prices=False,
        force_refresh_factors=False,
    )
    validate_analysis_dataset(dataset)

    print("\nStep 2: Fitting CAPM and Fama-French three-factor models...")
    capm_models = fit_all_capm_models(dataset)
    ff3_models = fit_all_ff3_models(dataset)

    capm_residuals = extract_model_residuals(capm_models, suffix="capm")
    ff3_residuals = extract_model_residuals(ff3_models, suffix="ff3")

    print("\nStep 3: Computing rolling 252-day CAPM betas...")
    rolling_betas = compute_rolling_capm_beta(dataset)
    save_rolling_betas(rolling_betas)

    print("\nStep 4: Building and saving report tables...")
    tables = build_all_report_tables(dataset)
    saved_table_paths = save_all_report_tables(tables)
    save_core_required_tables(tables)
    excel_workbook_path = save_tables_to_excel_workbook(tables)

    print("\nStep 5: Generating all required figures...")
    figure_paths = generate_all_figures(
        dataset=dataset,
        capm_residuals=capm_residuals,
        ff3_residuals=ff3_residuals,
        rolling_betas=rolling_betas,
    )

    print("\nPipeline complete.")
    print("=" * 80)

    print("\nKey files created/updated:")
    print(f"- Raw adjusted prices: {RAW_PRICE_FILE}")
    print(f"- Processed analysis dataset: {PROCESSED_ANALYSIS_FILE}")
    print(f"- Tables folder: {TABLES_DIR}")
    print(f"- Figures folder: {FIGURES_DIR}")
    print(f"- Excel workbook: {excel_workbook_path}")

    print("\nSaved tables:")
    for name, path in saved_table_paths.items():
        print(f"- {name}: {path}")

    print("\nGenerated figures:")
    for path in figure_paths:
        print(f"- {path}")

    print("\nNext step:")
    print("Use these tables and figures to build the Jupyter Notebook and Word report.")


if __name__ == "__main__":
    run_project_pipeline()