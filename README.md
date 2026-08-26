# One-Hour Stock Strategy V5 — Custom Backtest

This version adds fully adjustable backtest controls directly to the main Streamlit page.

## Adjustable settings

### Historical lookback
0–30 sessions.

- 1–30: uses only prior completed sessions.
- 0: same-day diagnostic mode. This has look-ahead bias and is not a valid forward backtest.

### Best-window length
15–120 minutes, adjustable in 5-minute increments.

The model searches every possible 1-minute start time inside the 8:30–10:30 AM Central analysis period.

### S1
- Profit trigger: 0–200% of median historical best-window gain.
- Loss trigger: 0–200% of median historical LDV.

### S2
- Profit trigger: 0–200% of median historical best-window gain.
- Loss trigger: 0–200% of median historical LDV.

Default values remain:
- Lookback = 15 sessions
- Best window = 60 minutes
- S1 profit = 90%
- S1 loss = 80%
- S2 profit = 90%
- S2 loss = 80%

Replace the current GitHub `app.py` with this version.
