# One-Hour Stock Strategy — V1

This is a working local dashboard for the strategy specification.

## What it calculates
- Analysis window: 8:30 AM–10:30 AM America/Chicago
- Trailing 30 market sessions
- Top 25 ranked primarily by average gain in the two-hour window
- 8:30 price / 10:30 price
- Best contiguous 1-hour interval and gain
- LDV: largest drop below the 8:30 starting price
- TDV: sum of all negative step-by-step percentage movements
- MUT: shortest sustained positive-price run, in seconds
- Positive-day percentage, median return, and return variability
- Interactive daily price chart and trailing daily-return chart

## Run
1. Install Python 3.10+
2. Open a terminal in this folder
3. Run:
   pip install -r requirements.txt
   streamlit run app.py

## Input data
The dashboard includes synthetic sample data so it runs immediately.

To use real data, upload a CSV with:
symbol,timestamp,price

Timestamp should be UTC or timezone-aware. Second-level or tick-level observations are preferred for TDV and MUT.

## Important data-source note
Minute bars can support the broader 8:30–10:30 and best-1-hour research, but they cannot faithfully measure a 10-second MUT. For the exact strategy definition, use trade/tick data or second-level aggregates from a licensed market-data provider.

## Ranking
V1 displays a consistency score, but the Top 25 ordering intentionally remains primarily based on average 8:30–10:30 gain, matching the stated strategy. Win rate, LDV, TDV, MUT and standard deviation are shown alongside it rather than silently overriding the user's definition.

## Next step
Connect a real historical/live provider, run the full US-stock universe (with liquidity/price filters), and host the Streamlit app.
