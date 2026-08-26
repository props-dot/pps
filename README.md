# One-Hour Stock Strategy V3 — Main-Page Backtest

Replace the current files in your GitHub repo with these files.

Most importantly, replace the existing `app.py` with this V3 `app.py`.

The existing live dashboard remains on the main page. A new section named
**Rolling 15-Day Strategy Backtest** is added below the existing dashboard.

Default test dates:
- First subject day: August 11, 2026
- Last subject day: August 25, 2026

Click **Run rolling 15-day backtest** on the main dashboard.

This version reuses:
- the stock universe selected in the existing sidebar
- the Alpaca feed selected in the existing sidebar
- the existing Streamlit Alpaca secrets

No `pages` directory is needed.

Backtest assumptions:
- Each subject day uses only the immediately preceding 15 completed market sessions.
- Top 10 are ranked by average 8:30–10:30 AM Central return.
- Historical best hour is selected from 1-minute start times from 8:30–9:30 Central.
- S1 uses equal capital across Top 10.
- S2 uses a sequential 100%-capital round robin and compounds.
- If target and stop both occur inside the same 1-minute bar, stop is assumed first.
- No transaction costs, spread, slippage, taxes, or market impact are included.
