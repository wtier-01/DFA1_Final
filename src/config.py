"""
Project-wide configuration for the FDA1 Final Project.

This file centralizes all major project settings:
- file paths
- selected stocks
- analysis dates
- Alpha Vantage API settings
- rolling-window assumptions

The project requirements call for two U.S.-listed individual common stocks,
from different sectors, with at least ten full years of daily history ending
on December 31, 2025.
"""

from pathlib import Path
from dotenv import load_dotenv
import os


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# ---------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "PASTE_YOUR_KEY_HERE")


# ---------------------------------------------------------------------
# Stock selection
# ---------------------------------------------------------------------
# These are initial recommended selections. We can change them later.
# They satisfy the major project constraints:
# - U.S.-listed common stocks
# - different sectors
# - not on the excluded list
# - long daily price history

STOCKS = {
    "MSFT": {
        "company": "Microsoft Corporation",
        "sector": "Technology",
        "rationale": (
            "Microsoft is a large-cap technology company with strong exposure "
            "to software, cloud computing, and artificial intelligence themes."
        ),
    },
    "LMT": {
        "company": "Lockheed Martin Corporation",
        "sector": "Industrials / Aerospace & Defense",
        "rationale": (
            "Lockheed Martin is a defense and aerospace company with a very "
            "different operating profile from Microsoft, making it useful for "
            "cross-sector comparison."
        ),
    },
}

TICKERS = list(STOCKS.keys())

EXCLUDED_TICKERS = ["AAPL", "AMZN", "JNJ", "JPM", "KO", "NVDA", "PG", "XOM"]


# ---------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"


# ---------------------------------------------------------------------
# Alpha Vantage settings
# ---------------------------------------------------------------------

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

ALPHA_VANTAGE_DAILY_ADJUSTED_FUNCTION = "TIME_SERIES_DAILY_ADJUSTED"

ALPHA_VANTAGE_OUTPUTSIZE = "full"

# Your premium plan allows delayed U.S. market data.
# For historical daily adjusted data, this parameter is acceptable.
ALPHA_VANTAGE_ENTITLEMENT = "delayed"


# ---------------------------------------------------------------------
# Factor data settings
# ---------------------------------------------------------------------
# Fama-French factors are needed for:
# - Mkt-RF
# - SMB
# - HML
# - RF
#
# We will pull these separately from Ken French / pandas-datareader
# or optionally support the class-provided CSV later.

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RF"]


# ---------------------------------------------------------------------
# Analysis assumptions
# ---------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
ROLLING_WINDOW_MEAN_VOL = 60
ROLLING_WINDOW_BETA = 252
ACF_PACF_LAGS = 40
STATISTICAL_SIGNIFICANCE_LEVEL = 0.05


# ---------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------

RAW_PRICE_FILE = RAW_DATA_DIR / "alpha_vantage_adjusted_prices.csv"
PROCESSED_ANALYSIS_FILE = PROCESSED_DATA_DIR / "analysis_dataset.csv"

CAPM_SUMMARY_FILE = TABLES_DIR / "capm_summary.csv"
FF3_SUMMARY_FILE = TABLES_DIR / "ff3_summary.csv"
ADF_SUMMARY_FILE = TABLES_DIR / "adf_summary.csv"
VIF_SUMMARY_FILE = TABLES_DIR / "vif_summary.csv"
ROLLING_BETA_FILE = PROCESSED_DATA_DIR / "rolling_capm_betas.csv"