# One-Hour Stock Strategy V7 — Persistent Data

V7 prevents repeated historical downloads when backtest parameters change.

## What it does

On app load:
1. Checking existing data
2. Loads `data/market_history.csv.gz` from the private GitHub repository
3. Compares the saved range/universe with what the app currently needs
4. Downloads only missing/new Alpaca data
5. Merges and saves the updated source back to GitHub
6. Uses the saved source for all dashboard and backtest calculations

Changing S1/S2 trigger percentages, lookback, best-window length, or subject dates
will reuse saved data. Alpaca is contacted only if the selected dates or stock universe
require data that are not already saved.

## Erase & Reconfirm All Data

A button on the main page fresh-downloads the entire saved period from Alpaca and
replaces the historical source.

## Required Streamlit Secrets

Keep the existing Alpaca secrets and add:

```toml
GITHUB_TOKEN = "YOUR_FINE_GRAINED_GITHUB_TOKEN"
GITHUB_REPO = "props-dot/pps"
GITHUB_BRANCH = "main"
GITHUB_DATA_PATH = "data/market_history.csv.gz"
```

The GitHub token should be a fine-grained token restricted to this repository with
**Contents: Read and write** permission.

Do not put the token in app.py or commit it to GitHub.

## Repository files

Your normal app files remain:
- app.py
- requirements.txt
- sample_intraday.csv
- README.md

After the app first saves successfully, GitHub will also contain:
- data/market_history.csv.gz

Replace your current `app.py` with the V7 app.py.
