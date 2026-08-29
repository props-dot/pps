# One-Hour Stock Strategy — V9

V9 adds the dashboard refinements requested after V8.

## New in V9

- Custom backtest controls are inside a Streamlit form. Changing sliders/dates no longer runs the report.
- New **Generate Backtest** button starts processing only when requested.
- Visible percentage/status updates during custom backtests and both dashboard optimizers.
- New **S3 — LDV Scenario**:
  - Adjustable Top 1–25 performers.
  - Adjustable prior lookback 1–30 sessions.
  - Individual-stock Gain and LDV multipliers: 50%, 75%, 100%, 125%, 150%.
  - Each stock independently chooses its historical best average entry/exit pair.
  - Entry/exit search uses 5-minute timestamps with a minimum 20-minute hold.
  - Trade-window LDV is measured only after the automatic entry and through the automatic exit.
  - Equal-weight S3 portfolio compounds from subject day to subject day.
- S1, S2, and S3 appear together in the portfolio summary and daily result export.
- S3 selections and S3 trade logs have their own CSV downloads.
- Backtest results are kept in Streamlit session state so download interactions do not erase the completed report.

## Install

Replace the existing GitHub `app.py` with the V9 `app.py`. Existing Streamlit secrets, requirements, and `data/market_history.csv.gz` remain unchanged.
