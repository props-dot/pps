# One-Hour Stock Strategy — V2

V2 adds direct Alpaca market-data support to the Streamlit dashboard.

## What V2 does
- Pulls real 1-minute historical stock bars from Alpaca
- Uses 8:30–10:30 AM America/Chicago
- Keeps the latest 30 complete market sessions
- Ranks Top 25 primarily by average two-hour gain
- Calculates daily gain, best 1-hour interval, LDV, TDV, MUT, win rate and volatility
- Adds an optional trade-level Precision Trade Check for a selected stock/day
- Keeps sample mode available for testing

## Streamlit setup

After replacing the V1 files in GitHub with these V2 files:

1. Open the deployed Streamlit app.
2. Click **Manage app**.
3. Open **Settings**.
4. Open **Secrets**.
5. Add:

```toml
ALPACA_API_KEY = "YOUR_ALPACA_KEY"
ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET"
```

6. Save and rerun the app.

Do NOT put API keys directly in GitHub or app.py.

## Data feed

The app offers:
- `iex` — useful for initial/free testing; only one exchange
- `sip` — consolidated US exchange data; appropriate subscription/access may be needed

For serious ranking, SIP is preferable because it represents the consolidated US market rather than one exchange.

## Precision

The broad universe scan uses 1-minute bars because scanning dozens/hundreds of symbols with every individual trade is expensive and slow.

The selected-symbol **Precision Trade Check** retrieves individual historical trades for one date and calculates:
- trade-level LDV
- trade-level total downward volatility
- minimum upswing time down to the timestamp precision of the feed

## Recommended next stage
After verifying the Alpaca connection:
1. Expand the stock universe.
2. Add price/liquidity filters.
3. Store each day's results persistently.
4. Schedule a post-10:30 CT refresh.
5. Add a morning pre-market candidate view separately from the historical ranking.
