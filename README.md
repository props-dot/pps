# One-Hour Stock Strategy V6 — Faster Backtest

This fixes the apparent hanging behavior in V5.

Changes:
- Bulk-loads the requested Alpaca historical range.
- Batches symbols and paginates data.
- Stops after repeated rate-limit responses instead of retrying forever.
- Indexes each (stock, date) session once.
- Reuses indexed data during all rolling calculations.
- Shows separate loading and calculation progress.
- Removes the Streamlit `use_container_width` deprecation warnings.

All customizable controls remain:
- Lookback: 0–30 sessions
- Best-window length: 15–120 minutes
- S1 profit/loss percentages
- S2 profit/loss percentages

Replace the existing GitHub `app.py` with this V6 file.
