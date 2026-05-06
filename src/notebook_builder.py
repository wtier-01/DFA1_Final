"""
Notebook builder for the FDA1 Final Project.

This script programmatically creates the final Jupyter Notebook:
notebooks/FDA1_Final_Project.ipynb

The generated notebook is organized around the required project sections:
3.1 Time Series Characterization
3.2 CAPM
3.3 Fama-French Three-Factor Model
3.4 Rolling Beta and Residual Analysis

Run:
python -m src.notebook_builder
"""

from __future__ import annotations

import nbformat as nbf

from src.config import NOTEBOOKS_DIR


NOTEBOOK_PATH = NOTEBOOKS_DIR / "FDA1_Final_Project.ipynb"


def md(text: str):
    """
    Create a Markdown cell.
    """
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    """
    Create a code cell.
    """
    return nbf.v4.new_code_cell(text)


def build_notebook() -> nbf.NotebookNode:
    """
    Build the final project notebook.
    """
    nb = nbf.v4.new_notebook()

    cells = []

    cells.append(
        md(
            """
# FDA1 Final Project: Factor Model and Time Series Analysis of Two Stocks

**Course:** Financial Data Analytics I  
**Project:** Factor Model and Time Series Analysis of Two Stocks  
**Stocks Analyzed:** Microsoft Corporation (MSFT) and Lockheed Martin Corporation (LMT)  

## Research Question

What drives the returns of these two stocks, and what is left unexplained?

This notebook analyzes two U.S.-listed common stocks from different sectors using daily adjusted stock prices, daily simple returns, stationarity tests, autocorrelation diagnostics, CAPM regressions, Fama-French three-factor regressions, rolling CAPM beta estimates, and residual diagnostics.

## Stock Selection Rationale

- **MSFT — Technology:** Microsoft represents a large-cap technology company with exposure to software, cloud computing, enterprise productivity, and artificial intelligence.
- **LMT — Industrials / Aerospace & Defense:** Lockheed Martin represents a defense and aerospace business with a different revenue profile, customer base, and risk exposure than Microsoft.

Neither stock is on the excluded ticker list, and the two companies come from different sectors.
"""
        )
    )

    cells.append(
        md(
            """
## Setup

This cell sets the project root correctly whether the notebook is run from the root folder or from inside the `notebooks/` folder.
"""
        )
    )

    cells.append(
        code(
            """
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
"""
        )
    )

    cells.append(
        code(
            """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image, display, Markdown

from src.config import (
    FIGURES_DIR,
    TABLES_DIR,
    TICKERS,
    START_DATE,
    END_DATE,
    STOCKS,
)

from src.data_pipeline import load_or_build_analysis_dataset, validate_analysis_dataset
from src.time_series_analysis import (
    compute_rolling_statistics,
    compute_return_descriptive_statistics,
    run_price_return_adf_suite,
    identify_high_volatility_periods,
)
from src.factor_models import (
    fit_all_capm_models,
    fit_all_ff3_models,
    summarize_capm_models,
    summarize_ff3_models,
    compute_factor_vif,
    compare_model_r_squared,
    extract_model_residuals,
)
from src.diagnostics import build_residual_diagnostic_tables
from src.rolling_beta import (
    compute_rolling_capm_beta,
    summarize_rolling_betas,
    identify_extreme_rolling_beta_dates,
)
from src.visualizations import generate_all_figures
from src.report_tables import (
    build_all_report_tables,
    save_all_report_tables,
    save_tables_to_excel_workbook,
    save_core_required_tables,
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

print("Imports complete.")
"""
        )
    )

    cells.append(
        md(
            """
## Data Loading and Validation

The analysis uses Alpha Vantage daily adjusted close prices for the two selected stocks and daily Fama-French three-factor data from Ken French's data library.

The Fama-French factors are converted from percent units to decimal units so they align with daily simple stock returns computed using `pct_change()`.
"""
        )
    )

    cells.append(
        code(
            """
dataset = load_or_build_analysis_dataset(
    force_rebuild=False,
    force_refresh_prices=False,
    force_refresh_factors=False,
)

validate_analysis_dataset(dataset)

print("\\nDataset preview:")
display(dataset.head())

print("\\nDataset tail:")
display(dataset.tail())
"""
        )
    )

    cells.append(
        md(
            """
## Section 3.1 — Time Series Characterization of Returns

This section describes the return behavior of each stock before fitting any factor model.
"""
        )
    )

    cells.append(
        md(
            """
### 3.1.1 Prices and Returns

The adjusted price series show the long-run price paths of the two stocks. Daily returns remove the price level trend and show short-horizon variation around zero.
"""
        )
    )

    cells.append(
        code(
            """
rolling_stats = compute_rolling_statistics(dataset)
return_stats = compute_return_descriptive_statistics(dataset)
adf_price_return_summary = run_price_return_adf_suite(dataset)
high_vol_periods = identify_high_volatility_periods(rolling_stats)

print("Return descriptive statistics:")
display(return_stats)

print("\\nADF tests for prices and returns:")
display(adf_price_return_summary)

print("\\nHighest rolling volatility periods:")
display(high_vol_periods)
"""
        )
    )

    cells.append(
        md(
            """
### Required Figures: Prices, Returns, Rolling Mean, Rolling Volatility, ACF/PACF, and Squared Return ACF

The following cell generates and saves all required figures. These figures are also used in the written report.
"""
        )
    )

    cells.append(
        code(
            """
capm_models = fit_all_capm_models(dataset)
ff3_models = fit_all_ff3_models(dataset)

capm_residuals = extract_model_residuals(capm_models, suffix="capm")
ff3_residuals = extract_model_residuals(ff3_models, suffix="ff3")

rolling_betas = compute_rolling_capm_beta(dataset)

figure_paths = generate_all_figures(
    dataset=dataset,
    capm_residuals=capm_residuals,
    ff3_residuals=ff3_residuals,
    rolling_betas=rolling_betas,
)

print(f"Generated {len(figure_paths)} figures.")
"""
        )
    )

    cells.append(
        code(
            """
for filename in [
    "01_adjusted_closing_prices.png",
    "02_daily_simple_returns.png",
    "03_rolling_60d_mean_returns.png",
    "04_rolling_60d_annualized_volatility.png",
]:
    display(Markdown(f"### {filename}"))
    display(Image(filename=str(FIGURES_DIR / filename)))
"""
        )
    )

    cells.append(
        md(
            """
### 3.1 Interpretation Notes

In the written report, discuss:
- how the price plots differ from the return plots;
- which stock appears more volatile;
- whether volatility spikes occur around common market events;
- whether price series are non-stationary while return series are stationary;
- whether squared returns show volatility clustering.
"""
        )
    )

    cells.append(
        md(
            """
## Section 3.2 — Market Model / CAPM

This section estimates each stock's sensitivity to the overall market using the CAPM regression:

\\[
R_{i,t} - R_{f,t} = \\alpha_i + \\beta_i(R_{m,t} - R_{f,t}) + \\epsilon_{i,t}
\\]
"""
        )
    )

    cells.append(
        code(
            """
capm_summary = summarize_capm_models(capm_models)

print("CAPM summary table:")
display(capm_summary)

for ticker, model in capm_models.items():
    print("\\n" + "=" * 80)
    print(f"CAPM Regression Summary: {ticker}")
    print("=" * 80)
    print(model.summary())
"""
        )
    )

    cells.append(
        code(
            """
for filename in [
    "08_capm_MSFT_capm_residual_residuals_time_series.png",
    "09_capm_MSFT_capm_residual_residual_histogram.png",
    "10_capm_MSFT_capm_residual_residual_qq.png",
    "08_capm_LMT_capm_residual_residuals_time_series.png",
    "09_capm_LMT_capm_residual_residual_histogram.png",
    "10_capm_LMT_capm_residual_residual_qq.png",
]:
    display(Markdown(f"### {filename}"))
    display(Image(filename=str(FIGURES_DIR / filename)))
"""
        )
    )

    cells.append(
        md(
            """
### 3.2 Interpretation Notes

In the written report, interpret:
- each stock's beta as aggressive or defensive relative to the market;
- whether alpha is statistically different from zero at the 5% level;
- how much return variation the market explains using R-squared;
- whether residuals look approximately mean-zero and normal or show fat tails/structure;
- which stock is more market-sensitive and which is better explained by CAPM.
"""
        )
    )

    cells.append(
        md(
            """
## Section 3.3 — Fama-French Three-Factor Model

This section extends CAPM by adding the size and value factors:

\\[
R_{i,t} - R_{f,t} =
\\alpha_i +
\\beta_{m,i}(R_{m,t} - R_{f,t}) +
\\beta_{s,i}SMB_t +
\\beta_{h,i}HML_t +
\\epsilon_{i,t}
\\]
"""
        )
    )

    cells.append(
        code(
            """
ff3_summary = summarize_ff3_models(ff3_models)
factor_vif = compute_factor_vif(dataset)
model_comparison = compare_model_r_squared(capm_summary, ff3_summary)

print("Fama-French three-factor summary table:")
display(ff3_summary)

print("\\nFactor VIF:")
display(factor_vif)

print("\\nCAPM vs FF3 R-squared comparison:")
display(model_comparison)

for ticker, model in ff3_models.items():
    print("\\n" + "=" * 80)
    print(f"FF3 Regression Summary: {ticker}")
    print("=" * 80)
    print(model.summary())
"""
        )
    )

    cells.append(
        code(
            """
for filename in [
    "08_ff3_MSFT_ff3_residual_residuals_time_series.png",
    "09_ff3_MSFT_ff3_residual_residual_histogram.png",
    "10_ff3_MSFT_ff3_residual_residual_qq.png",
    "08_ff3_LMT_ff3_residual_residuals_time_series.png",
    "09_ff3_LMT_ff3_residual_residual_histogram.png",
    "10_ff3_LMT_ff3_residual_residual_qq.png",
]:
    display(Markdown(f"### {filename}"))
    display(Image(filename=str(FIGURES_DIR / filename)))
"""
        )
    )

    cells.append(
        md(
            """
### 3.3 Interpretation Notes

In the written report, interpret:
- whether the FF3 market beta is close to the CAPM beta;
- whether SMB is positive or negative and what that implies;
- whether HML is positive or negative and what that implies;
- whether each factor loading is statistically significant;
- whether VIF suggests multicollinearity is a concern;
- whether adding SMB and HML materially improves explanatory power over CAPM.
"""
        )
    )

    cells.append(
        md(
            """
## Section 3.4 — Synthesis: Rolling Beta and Residual Analysis

This section connects the time-series analysis to the factor model results by examining rolling 252-day CAPM beta and FF3 residual structure.
"""
        )
    )

    cells.append(
        code(
            """
rolling_beta_summary = summarize_rolling_betas(rolling_betas)
extreme_rolling_beta_dates = identify_extreme_rolling_beta_dates(rolling_betas)

residual_tables = build_residual_diagnostic_tables(
    capm_residuals=capm_residuals,
    ff3_residuals=ff3_residuals,
)

print("Rolling beta summary:")
display(rolling_beta_summary)

print("\\nExtreme rolling beta dates:")
display(extreme_rolling_beta_dates)

print("\\nFF3 residual ADF tests:")
display(residual_tables["ff3_residual_adf"])

print("\\nFF3 residual distribution diagnostics:")
display(residual_tables["ff3_residual_distribution"])

print("\\nFF3 residual Ljung-Box tests:")
display(residual_tables["ff3_residual_ljung_box"])

print("\\nFF3 squared residual Ljung-Box tests:")
display(residual_tables["ff3_squared_residual_ljung_box"])
"""
        )
    )

    cells.append(
        code(
            """
display(Markdown("### Rolling 252-Day CAPM Beta"))
display(Image(filename=str(FIGURES_DIR / "14_rolling_252d_capm_beta.png")))
"""
        )
    )

    cells.append(
        code(
            """
for filename in [
    "11_MSFT_ff3_residual_acf.png",
    "12_MSFT_ff3_residual_pacf.png",
    "13_MSFT_ff3_residual_squared_acf.png",
    "11_LMT_ff3_residual_acf.png",
    "12_LMT_ff3_residual_pacf.png",
    "13_LMT_ff3_residual_squared_acf.png",
]:
    display(Markdown(f"### {filename}"))
    display(Image(filename=str(FIGURES_DIR / filename)))
"""
        )
    )

    cells.append(
        md(
            """
### 3.4 Interpretation Notes

In the written report, discuss:
- whether each stock's rolling beta was stable or time-varying;
- whether the two rolling betas moved together or diverged;
- whether visible beta shifts align with COVID-19, the 2022 bear market, interest-rate shocks, AI/technology repricing, or defense/geopolitical periods;
- whether FF3 residuals are stationary;
- whether residual ACF/PACF suggests remaining serial correlation;
- whether squared residual ACF suggests remaining volatility clustering.
"""
        )
    )

    cells.append(
        md(
            """
## Discussion Section Support

The written report should directly answer the following:

1. Which stock is more market-sensitive based on full-sample CAPM beta and rolling 252-day beta?
2. Do SMB and HML loadings match expectations for each company?
3. Did FF3 materially improve explanatory power over CAPM?
4. Are FF3 residuals close to white noise, or is there remaining structure?
5. Did rolling beta change meaningfully over time?
6. What are the limitations of full-sample linear factor models when returns exhibit time-varying volatility and regime shifts?
"""
        )
    )

    cells.append(
        md(
            """
## Save Final Tables

This cell saves report-ready CSV tables and an Excel workbook.
"""
        )
    )

    cells.append(
        code(
            """
report_tables = build_all_report_tables(dataset)
saved_table_paths = save_all_report_tables(report_tables)
save_core_required_tables(report_tables)
excel_path = save_tables_to_excel_workbook(report_tables)

print("Saved table files:")
for name, path in saved_table_paths.items():
    print(f"{name}: {path}")

print("\\nExcel workbook:", excel_path)
"""
        )
    )

    cells.append(
        md(
            """
## Conclusion Placeholder

The final written conclusion should summarize:

- the main return behavior differences between MSFT and LMT;
- which stock is more market-sensitive;
- whether SMB/HML provide incremental explanatory power;
- whether residual structure remains after FF3;
- why CAPM and FF3 are more useful as descriptive risk models than precise forecasting tools.
"""
        )
    )

    nb["cells"] = cells

    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    }

    return nb


def save_notebook() -> None:
    """
    Save the generated notebook to the notebooks folder.
    """
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    nb = build_notebook()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Notebook created: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    save_notebook()