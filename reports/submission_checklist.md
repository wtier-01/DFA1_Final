# FDA1 Final Project — Submission Checklist

## Required Blackboard Submission Files

Submit the following:

- [ ] Final Word report: `.docx`
- [ ] Final Jupyter Notebook: `notebooks/FDA1_Final_Project.ipynb`
- [ ] Local data file: `data/raw/alpha_vantage_adjusted_prices.csv`
- [ ] Local factor file: `data/raw/fama_french_daily_factors.csv`
- [ ] Processed analysis file: `data/processed/analysis_dataset.csv`

Because this project caches local files for reproducibility, include the local data files with the submission.

---

## Stock Selection Requirements

- [ ] Exactly two stocks selected
- [ ] Both are individual common stocks
- [ ] Both are listed on NYSE or NASDAQ
- [ ] Both are headquartered in the United States
- [ ] Neither is an ETF, mutual fund, index fund, SPAC, or ADR
- [ ] Stocks are from different sectors
- [ ] Neither stock is on excluded list:
  - AAPL
  - AMZN
  - JNJ
  - JPM
  - KO
  - NVDA
  - PG
  - XOM
- [ ] Data covers at least January 2016 through December 2025
- [ ] Analysis period ends on December 31, 2025, or the final valid trading date at year-end

---

## Data and Unit Checks

- [ ] Adjusted close prices are used
- [ ] Daily simple returns are computed with `pct_change()`
- [ ] Log returns are not used
- [ ] Fama-French factor columns are:
  - Mkt-RF
  - SMB
  - HML
  - RF
- [ ] Fama-French factors are converted from percent units to decimal units
- [ ] Excess stock return equals stock daily return minus RF
- [ ] Market excess return uses `Mkt-RF`
- [ ] RF is not subtracted from `Mkt-RF` again
- [ ] Stock returns and factors are merged by date
- [ ] Final regression dataset is NaN-free

---

## Section 3.1 — Time Series Characterization

Required:

- [ ] Adjusted price plot included
- [ ] Daily return plot included
- [ ] Rolling 60-day mean return plot included
- [ ] Rolling 60-day annualized volatility plot included
- [ ] At least one visible volatility regime shift identified
- [ ] ADF test on MSFT price
- [ ] ADF test on MSFT returns
- [ ] ADF test on LMT price
- [ ] ADF test on LMT returns
- [ ] ADF statistic and p-value reported for all four tests
- [ ] Stationarity interpreted correctly
- [ ] ACF and PACF of daily returns plotted up to 40 lags
- [ ] ACF of squared returns plotted
- [ ] Volatility clustering discussed
- [ ] Cross-stock time-series comparison included

---

## Section 3.2 — CAPM

Required:

- [ ] Daily excess returns computed correctly
- [ ] CAPM regression run for each stock
- [ ] `statsmodels.OLS` used
- [ ] Constant/intercept included using `sm.add_constant`
- [ ] CAPM summary table includes:
  - alpha
  - beta
  - R-squared
  - alpha p-value
  - beta p-value
- [ ] Beta interpreted for each stock
- [ ] Alpha statistical significance interpreted at 5%
- [ ] R-squared interpreted
- [ ] CAPM residual time plot included for each stock
- [ ] CAPM residual histogram included for each stock
- [ ] CAPM residual Q-Q plot included for each stock
- [ ] CAPM residual behavior discussed
- [ ] Cross-stock CAPM comparison included

---

## Section 3.3 — Fama-French Three-Factor Model

Required:

- [ ] FF3 regression run for each stock
- [ ] Constant/intercept included
- [ ] FF3 summary table includes:
  - alpha
  - market beta
  - SMB loading
  - HML loading
  - R-squared
  - adjusted R-squared
  - p-values of all coefficients
- [ ] Market beta compared to CAPM beta
- [ ] SMB loading interpreted
- [ ] HML loading interpreted
- [ ] Statistical significance of each loading discussed
- [ ] VIF computed for Mkt-RF, SMB, HML
- [ ] Multicollinearity discussed
- [ ] FF3 R-squared compared to CAPM R-squared
- [ ] FF3 residual time plot included for each stock
- [ ] FF3 residual histogram included for each stock
- [ ] FF3 residual Q-Q plot included for each stock
- [ ] FF3 residuals compared to CAPM residuals
- [ ] Cross-stock factor loading comparison included

---

## Section 3.4 — Rolling Beta and Residual Analysis

Required:

- [ ] Rolling 252-day CAPM beta computed for each stock
- [ ] Both rolling betas plotted on one chart
- [ ] Horizontal beta = 1 line included
- [ ] Time-varying beta interpreted
- [ ] Rolling beta shifts tied to real-world events where possible
- [ ] ADF test run on FF3 residuals for each stock
- [ ] Residual ADF statistic and p-value reported
- [ ] ACF and PACF of FF3 residuals plotted up to 40 lags
- [ ] ACF of squared FF3 residuals plotted
- [ ] Residual white-noise behavior discussed
- [ ] Remaining volatility clustering discussed
- [ ] Final stock-by-stock synthesis included

---

## Discussion Section

The report must substantively answer:

- [ ] Which stock is more market-sensitive?
- [ ] Do SMB and HML loadings match expectations?
- [ ] Did FF3 materially improve explanatory power over CAPM?
- [ ] Are FF3 residuals close to white noise?
- [ ] Did rolling beta change meaningfully over time?
- [ ] What are the limitations of full-sample linear factor models?

Avoid one-line answers. Each question should be answered with specific results from tables and figures.

---

## Report Quality Checks

- [ ] Report is a single `.docx`
- [ ] Report includes introduction
- [ ] Report includes methodology
- [ ] Report includes Sections 3.1 through 3.4
- [ ] Report includes discussion section
- [ ] Report includes conclusion
- [ ] Every figure has a clear title
- [ ] Every table is referenced and interpreted
- [ ] Every chart has labeled axes
- [ ] Multi-series charts have legends
- [ ] Writing explains results rather than narrating code
- [ ] Report is long enough to cover all required content

---

## Notebook Quality Checks

- [ ] Notebook runs top-to-bottom without errors
- [ ] Notebook has Markdown headings for Sections 3.1 through 3.4
- [ ] Code cells are readable
- [ ] Outputs reproduce tables and figures used in the report
- [ ] Notebook does not require manual fixes
- [ ] Notebook uses submitted local data files or can rebuild them from API/cache
- [ ] Final notebook has been restarted and run all before submission

---

## Final Pre-Submission Commands

Run these from the project root:

```bash
source .venv/bin/activate
python main.py
python -m src.notebook_builder
jupyter notebook notebooks/FDA1_Final_Project.ipynb