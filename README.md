# One-Hour Stock Strategy — V8

Changes implemented:

1. Historical Rank and Today Rank are separated. Daily rows are sorted by actual subject-day 8:30–10:30 gain.
2. Each backtest subject day independently recalculates its Top 10 from that day's own prior lookback sessions.
3. Adjustable starting capital (default $1,000) compounds independently through S1 and S2. S1 divides capital equally across all ten selected stocks; S2 uses the full current balance sequentially.
4. CSV exports include all selected parameters at the top, subject-date range/count, and the daily/portfolio outcomes below.
5. Added on-demand optimizer summaries. Trigger grid = 25%, 50%, 75%, 100%, 125%, 150%, 175%, 200%.
6. Historical optimizer evaluates the fixed Top 10 derived from the trailing 30-session set across that same 30-session period and is explicitly labeled hindsight.
7. Daily optimizer shows independent 1-, 5-, 15-, and 30-session lookback selections and the best hindsight S1/S2 trigger outcome for the latest complete day.

Deployment: replace only `app.py` in the existing GitHub repository. Existing Streamlit secrets and persistent data file remain unchanged.
