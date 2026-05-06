"""
Fama-French factor loader for the FDA1 Final Project.

This module downloads the daily Fama-French three-factor data directly from
Ken French's data library and converts the factor values from percent units
to decimal units.

Required columns:
- Mkt-RF
- SMB
- HML
- RF

Important:
Fama-French factors are published in percent units, while stock returns from
pct_change() are decimal returns. We divide all factor columns by 100 before
using them in regressions.
"""

from __future__ import annotations

from io import BytesIO, StringIO
from zipfile import ZipFile

import pandas as pd
import requests

from src.config import (
    END_DATE,
    FACTOR_COLUMNS,
    RAW_DATA_DIR,
    START_DATE,
)


KEN_FRENCH_DAILY_FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

RAW_FACTOR_FILE = RAW_DATA_DIR / "fama_french_daily_factors.csv"


def _extract_factor_csv_text(zip_bytes: bytes) -> str:
    """
    Extract the CSV text from the downloaded Ken French zip file.
    """
    with ZipFile(BytesIO(zip_bytes)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]

        if not csv_names:
            raise FileNotFoundError("No CSV file found inside the Ken French zip archive.")

        with zf.open(csv_names[0]) as csv_file:
            return csv_file.read().decode("utf-8", errors="replace")


def _parse_ken_french_daily_csv(csv_text: str) -> pd.DataFrame:
    """
    Parse the Ken French daily FF3 CSV text into a clean DataFrame.

    The downloaded file contains descriptive header/footer text around the
    actual data table, so we locate the true header row programmatically.
    """
    lines = csv_text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        normalized = line.replace(" ", "")
        if normalized.startswith(",Mkt-RF,SMB,HML,RF"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not locate the Fama-French daily factor header row.")

    data_end_idx = None
    for j in range(header_idx + 1, len(lines)):
        line = lines[j].strip()
        if line == "" or line.lower().startswith("copyright"):
            data_end_idx = j
            break

    if data_end_idx is None:
        data_end_idx = len(lines)

    table_text = "\n".join(lines[header_idx:data_end_idx])
    factors = pd.read_csv(StringIO(table_text))

    date_col = factors.columns[0]
    factors = factors.rename(columns={date_col: "Date"})

    factors["Date"] = pd.to_datetime(factors["Date"].astype(str), format="%Y%m%d", errors="coerce")
    factors = factors.dropna(subset=["Date"])
    factors = factors.set_index("Date").sort_index()

    for col in FACTOR_COLUMNS:
        factors[col] = pd.to_numeric(factors[col], errors="coerce")

    factors = factors[FACTOR_COLUMNS].dropna()

    # Convert percent units to decimal units.
    factors = factors / 100.0

    factors = factors.loc[START_DATE:END_DATE].copy()
    factors.index.name = "Date"

    if factors.empty:
        raise ValueError(
            f"No Fama-French factors found between {START_DATE} and {END_DATE}."
        )

    return factors


def download_fama_french_factors() -> pd.DataFrame:
    """
    Download and clean daily Fama-French three-factor data.

    Returns
    -------
    pd.DataFrame
        Daily factor DataFrame indexed by date, with decimal-unit columns:
        Mkt-RF, SMB, HML, RF.
    """
    print("Downloading daily Fama-French three-factor data from Ken French...")
    response = requests.get(KEN_FRENCH_DAILY_FF3_URL, timeout=60)
    response.raise_for_status()

    csv_text = _extract_factor_csv_text(response.content)
    factors = _parse_ken_french_daily_csv(csv_text)

    return factors


def save_factors(factors: pd.DataFrame, path=RAW_FACTOR_FILE) -> None:
    """
    Save cleaned Fama-French factors to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(path)
    print(f"Saved Fama-French factors to: {path}")


def load_cached_factors(path=RAW_FACTOR_FILE) -> pd.DataFrame:
    """
    Load cached Fama-French factors from CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cached Fama-French factor file not found: {path}")

    factors = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    factors = factors.sort_index()
    return factors


def load_or_download_factors(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load cached factors if available; otherwise download from Ken French.
    """
    if RAW_FACTOR_FILE.exists() and not force_refresh:
        print(f"Loading cached Fama-French factors from: {RAW_FACTOR_FILE}")
        return load_cached_factors(RAW_FACTOR_FILE)

    factors = download_fama_french_factors()
    save_factors(factors, RAW_FACTOR_FILE)

    return factors


if __name__ == "__main__":
    ff_factors = load_or_download_factors(force_refresh=True)

    print("\nFama-French factor preview:")
    print(ff_factors.head())
    print("\nLast rows:")
    print(ff_factors.tail())
    print("\nShape:", ff_factors.shape)
    print("Start date:", ff_factors.index.min().date())
    print("End date:", ff_factors.index.max().date())
    print("\nColumn means, decimal units:")
    print(ff_factors.mean())