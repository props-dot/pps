
import os
import time as time_module
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

CT = ZoneInfo("America/Chicago")
UTC = timezone.utc

START_TIME = time(8, 30)
END_TIME = time(10, 30)
TRAILING_DAYS = 30
TOP_N = 25

DEFAULT_SYMBOLS = """
AAPL MSFT NVDA AMZN META GOOGL GOOG AVGO TSLA BRK.B LLY JPM V MA
COST WMT NFLX ORCL HD PG JNJ ABBV BAC KO CRM AMD MRK CVX XOM
TMUS CSCO IBM GE CAT ADBE QCOM TXN INTU NOW AMAT PANW BKNG UBER
AXP GS MS PEP MCD DIS NKE LOW TMO UNH AMGN ISRG PLTR MU
""".split()

st.set_page_config(page_title="One-Hour Stock Strategy", layout="wide")
st.title("One-Hour Stock Strategy — V2")
st.caption("Real-market-data capable. Ranks repeatable gains in the 8:30–10:30 AM Central window.")

def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

ALPACA_KEY = get_secret("ALPACA_API_KEY")
ALPACA_SECRET = get_secret("ALPACA_SECRET_KEY")

def alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

def normalize_symbol(sym):
    # Alpaca uses BRK.B style for asset symbols; preserve punctuation.
    return sym.strip().upper()

@st.cache_data(ttl=60*60, show_spinner=False)
def fetch_alpaca_bars(symbols_tuple, start_iso, end_iso, feed):
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Alpaca API credentials are missing.")

    symbols = list(symbols_tuple)
    all_rows = []
    # Batch to keep request URLs reasonable and failures isolated.
    batch_size = 25

    for b0 in range(0, len(symbols), batch_size):
        batch = symbols[b0:b0+batch_size]
        page_token = None
        while True:
            params = {
                "symbols": ",".join(batch),
                "timeframe": "1Min",
                "start": start_iso,
                "end": end_iso,
                "limit": 10000,
                "adjustment": "split",
                "feed": feed,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token

            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/bars",
                headers=alpaca_headers(),
                params=params,
                timeout=45,
            )
            if r.status_code == 429:
                time_module.sleep(2)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Alpaca returned {r.status_code}: {r.text[:500]}")

            payload = r.json()
            bars = payload.get("bars", {})
            for sym, records in bars.items():
                for x in records:
                    all_rows.append({
                        "symbol": sym,
                        "timestamp": x["t"],
                        "open": x["o"],
                        "high": x["h"],
                        "low": x["l"],
                        "close": x["c"],
                        "volume": x.get("v", np.nan),
                    })

            page_token = payload.get("next_page_token")
            if not page_token:
                break

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(CT)
    return df

@st.cache_data(ttl=15*60, show_spinner=False)
def fetch_alpaca_trades(symbol, start_iso, end_iso, feed):
    """Optional precision view for a single symbol/day."""
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Alpaca API credentials are missing.")

    rows, token = [], None
    while True:
        params = {
            "start": start_iso,
            "end": end_iso,
            "limit": 10000,
            "feed": feed,
            "sort": "asc",
        }
        if token:
            params["page_token"] = token
        r = requests.get(
            f"https://data.alpaca.markets/v2/stocks/{symbol}/trades",
            headers=alpaca_headers(),
            params=params,
            timeout=45,
        )
        if r.status_code == 429:
            time_module.sleep(2)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca returned {r.status_code}: {r.text[:500]}")
        payload = r.json()
        for x in payload.get("trades", []):
            rows.append({
                "symbol": symbol,
                "timestamp": x["t"],
                "price": x["p"],
                "size": x.get("s", np.nan),
            })
        token = payload.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(CT)
    return df

def sample_data():
    path = os.path.join(os.path.dirname(__file__), "sample_intraday.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(CT)
    df["open"] = df["price"]
    df["high"] = df["price"]
    df["low"] = df["price"]
    df["close"] = df["price"]
    df["volume"] = np.nan
    return df

def filter_complete_sessions(df):
    if df.empty:
        return df
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    tt = df["timestamp"].dt.time
    df = df[(tt >= START_TIME) & (tt <= END_TIME)]

    # Only include sessions that have data near both ends of the window.
    good = []
    for (sym, d), g in df.groupby(["symbol", "date"]):
        mins = g["timestamp"].dt.hour * 60 + g["timestamp"].dt.minute
        if mins.min() <= 8*60+32 and mins.max() >= 10*60+28:
            good.append((sym, d))
    if not good:
        return df.iloc[0:0]
    good_df = pd.DataFrame(good, columns=["symbol","date"])
    df = df.merge(good_df, on=["symbol","date"], how="inner")
    return df.sort_values(["symbol","date","timestamp"])

def pct(a, b):
    return (b/a - 1.0) * 100.0 if a and np.isfinite(a) else np.nan

def daily_metrics_from_bars(g):
    g = g.sort_values("timestamp")
    if len(g) < 2:
        return None

    # Using first open and last close best approximates entry at 8:30 and exit at 10:30.
    start_price = float(g.iloc[0]["open"])
    end_price = float(g.iloc[-1]["close"])
    gain = pct(start_price, end_price)

    low = float(g["low"].min())
    ldv = max(0.0, (start_price-low)/start_price*100.0)

    closes = g["close"].astype(float).to_numpy()
    ts = list(g["timestamp"])
    step_ret = np.diff(closes) / closes[:-1] * 100.0
    tdv = float(np.abs(step_ret[step_ret < 0]).sum())

    # MUT at 1-minute resolution in the broad scan.
    dirs = step_ret > 0
    durations = []
    i = 0
    while i < len(dirs):
        if dirs[i]:
            j = i
            while j < len(dirs) and dirs[j]:
                j += 1
            durations.append((j-i) * 60.0)
            i = j
        else:
            i += 1
    if len(dirs) and np.all(dirs):
        mut = (ts[-1]-ts[0]).total_seconds()
    elif durations:
        mut = min(durations)
    else:
        mut = 0.0

    # Best rolling 60-minute interval using nearest bars.
    best_gain = -np.inf
    best_start = best_end = None
    tvals = np.array([x.value for x in pd.Series(ts)])
    for i, t0 in enumerate(ts):
        target = t0 + pd.Timedelta(minutes=60)
        j = np.searchsorted(tvals, target.value, side="right") - 1
        if j > i:
            g1 = pct(float(g.iloc[i]["open"]), float(g.iloc[j]["close"]))
            if g1 > best_gain:
                best_gain, best_start, best_end = g1, t0, ts[j]

    return {
        "start_price": start_price,
        "end_price": end_price,
        "gain_pct": gain,
        "ldv_pct": ldv,
        "tdv_pct": tdv,
        "mut_seconds": mut,
        "best_1h_gain_pct": best_gain if np.isfinite(best_gain) else np.nan,
        "best_1h_start": best_start,
        "best_1h_end": best_end,
        "volume": float(g["volume"].fillna(0).sum()),
    }

def analyze_bars(df):
    df = filter_complete_sessions(df)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), df

    # Keep the latest 30 distinct complete sessions globally.
    all_dates = sorted(df["date"].unique())
    keep_dates = all_dates[-TRAILING_DAYS:]
    df = df[df["date"].isin(keep_dates)]

    rows = []
    for (sym, d), g in df.groupby(["symbol","date"]):
        m = daily_metrics_from_bars(g)
        if m:
            rows.append({"symbol":sym, "date":d, **m})
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily, pd.DataFrame(), df

    summary = daily.groupby("symbol").agg(
        avg_gain_pct=("gain_pct","mean"),
        positive_days=("gain_pct", lambda s: int((s>0).sum())),
        days_observed=("gain_pct","count"),
        win_rate_pct=("gain_pct", lambda s: float((s>0).mean()*100)),
        avg_best_1h_gain_pct=("best_1h_gain_pct","mean"),
        avg_ldv_pct=("ldv_pct","mean"),
        avg_tdv_pct=("tdv_pct","mean"),
        avg_mut_seconds=("mut_seconds","mean"),
        median_gain_pct=("gain_pct","median"),
        gain_std_pct=("gain_pct","std"),
        avg_window_volume=("volume","mean"),
    ).reset_index()

    # Exclude symbols with too little history relative to the requested 30 sessions.
    max_days = max(1, daily["date"].nunique())
    summary = summary[summary["days_observed"] >= max(10, int(max_days*0.7))].copy()

    summary["consistency_score"] = (
        summary["avg_gain_pct"]
        + 0.012 * summary["win_rate_pct"]
        - 0.20 * summary["avg_ldv_pct"]
        - 0.05 * summary["avg_tdv_pct"]
        - 0.10 * summary["gain_std_pct"].fillna(0)
    )
    summary = summary.sort_values(
        ["avg_gain_pct","win_rate_pct","avg_ldv_pct"],
        ascending=[False,False,True]
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary)+1)
    return daily, summary, df

def precision_metrics_from_trades(df):
    if df.empty or len(df) < 2:
        return None
    df = df.sort_values("timestamp")
    prices = df["price"].astype(float).to_numpy()
    times = list(df["timestamp"])
    start_price = prices[0]
    low = prices.min()
    ldv = max(0.0, (start_price-low)/start_price*100.0)
    step_ret = np.diff(prices)/prices[:-1]*100.0
    tdv = float(np.abs(step_ret[step_ret<0]).sum())
    dirs = step_ret > 0
    durations = []
    i = 0
    while i < len(dirs):
        if dirs[i]:
            j = i
            while j < len(dirs) and dirs[j]:
                j += 1
            end_idx = min(j, len(times)-1)
            durations.append((times[end_idx]-times[i]).total_seconds())
            i = j
        else:
            i += 1
    mut = min([x for x in durations if x >= 0], default=0.0)
    if len(dirs) and np.all(dirs):
        mut = (times[-1]-times[0]).total_seconds()
    return {"ldv_pct":ldv, "tdv_pct":tdv, "mut_seconds":mut, "trade_count":len(df)}

# Sidebar
st.sidebar.header("Market Data")
mode = st.sidebar.radio("Source", ["Real Alpaca data", "Built-in sample data"])
feed = st.sidebar.selectbox("Alpaca feed", ["iex","sip"], index=0,
                            help="IEX works for initial/free testing. SIP is the consolidated US market feed and may require the appropriate Alpaca subscription.")
symbols_text = st.sidebar.text_area(
    "Stock universe",
    value=" ".join(DEFAULT_SYMBOLS),
    height=160,
    help="Space- or comma-separated tickers. V2 starts with a liquid-stock universe; expand this later after validating the model."
)
symbols = [normalize_symbol(x) for x in symbols_text.replace(","," ").split()]
symbols = list(dict.fromkeys([x for x in symbols if x]))

if mode == "Real Alpaca data":
    if not ALPACA_KEY or not ALPACA_SECRET:
        st.error("Alpaca credentials are not configured yet.")
        st.info("In Streamlit: Manage app → Settings → Secrets. Add ALPACA_API_KEY and ALPACA_SECRET_KEY, then rerun.")
        st.code('ALPACA_API_KEY = "your_key_here"\nALPACA_SECRET_KEY = "your_secret_here"', language="toml")
        st.stop()

    now_ct = datetime.now(CT)
    # Pull enough calendar days to yield >=30 market sessions.
    start_dt = (now_ct - timedelta(days=55)).astimezone(UTC)
    end_dt = now_ct.astimezone(UTC)
    with st.spinner(f"Loading real 1-minute market data for {len(symbols)} stocks..."):
        try:
            raw = fetch_alpaca_bars(tuple(symbols), start_dt.isoformat(), end_dt.isoformat(), feed)
        except Exception as e:
            st.error(str(e))
            if feed == "sip":
                st.info("If SIP access is rejected, switch the feed to IEX for initial testing or upgrade the Alpaca market-data plan.")
            st.stop()
else:
    raw = sample_data()

daily, summary, windowed = analyze_bars(raw)
if summary.empty:
    st.warning("No complete 8:30–10:30 AM Central sessions were found.")
    st.stop()

top = summary.head(TOP_N).copy()
latest_date = max(daily["date"])
latest = daily[(daily["date"]==latest_date) & (daily["symbol"].isin(top["symbol"]))].copy()
latest = latest.merge(top[["symbol","rank"]], on="symbol", how="inner").sort_values("rank")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Universe", f"{summary['symbol'].nunique()}")
c2.metric("Sessions", f"{daily['date'].nunique()}")
c3.metric("Top list", f"{len(top)}")
c4.metric("Latest complete", str(latest_date))
c5.metric("Feed", feed.upper() if mode.startswith("Real") else "SAMPLE")

st.subheader("Top 25 — trailing 30 complete sessions")
view = top[[
    "rank","symbol","avg_gain_pct","win_rate_pct","avg_best_1h_gain_pct",
    "avg_ldv_pct","avg_tdv_pct","avg_mut_seconds","median_gain_pct","gain_std_pct","days_observed"
]].copy()
view.columns = [
    "Rank","Symbol","Avg 8:30–10:30 Gain %","Positive-Day %","Avg Best 1-Hour Gain %",
    "Avg LDV %","Avg TDV %","Avg MUT (sec)*","Median Gain %","Gain Std Dev %","Days"
]
st.dataframe(view, use_container_width=True, hide_index=True)

st.caption("* Broad-scan MUT/TDV use 1-minute bars. Use the Precision Trade Check below for trade-level MUT/TDV on a selected stock/day.")

st.subheader(f"Daily Top-25 results — {latest_date}")
if not latest.empty:
    lv = latest[[
        "rank","symbol","start_price","end_price","gain_pct","best_1h_gain_pct",
        "best_1h_start","best_1h_end","ldv_pct","tdv_pct","mut_seconds","volume"
    ]].copy()
    lv["best_1h_start"] = pd.to_datetime(lv["best_1h_start"]).dt.strftime("%H:%M")
    lv["best_1h_end"] = pd.to_datetime(lv["best_1h_end"]).dt.strftime("%H:%M")
    lv.columns = [
        "Rank","Symbol","8:30 Price","10:30 Price","Daily Gain %","Best 1-Hour Gain %",
        "Best 1-Hour Start","Best 1-Hour End","LDV %","TDV %","MUT (sec)*","Window Volume"
    ]
    st.dataframe(lv, use_container_width=True, hide_index=True)

st.subheader("Intraday chart")
symbol = st.selectbox("Stock", top["symbol"].tolist())
date_choices = sorted(daily[daily["symbol"]==symbol]["date"].unique(), reverse=True)
sel_date = st.selectbox("Session", date_choices)
chart = windowed[(windowed["symbol"]==symbol) & (windowed["date"]==sel_date)]
fig = px.line(chart, x="timestamp", y="close", title=f"{symbol} — {sel_date} — 8:30–10:30 CT")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Trailing 30-session performance")
td = daily[daily["symbol"]==symbol].sort_values("date")
fig2 = px.bar(td, x="date", y="gain_pct", title=f"{symbol} daily 8:30–10:30 gain (%)")
st.plotly_chart(fig2, use_container_width=True)

if mode == "Real Alpaca data":
    st.subheader("Precision Trade Check")
    st.write("For the selected stock/day, this can pull individual trades to estimate MUT and TDV at trade-level resolution.")
    if st.button(f"Run precision check for {symbol} on {sel_date}"):
        start_local = datetime.combine(sel_date, START_TIME, tzinfo=CT)
        end_local = datetime.combine(sel_date, END_TIME, tzinfo=CT)
        with st.spinner("Loading trades..."):
            try:
                trades = fetch_alpaca_trades(
                    symbol,
                    start_local.astimezone(UTC).isoformat(),
                    end_local.astimezone(UTC).isoformat(),
                    feed,
                )
                pm = precision_metrics_from_trades(trades)
                if pm:
                    a,b,c,d = st.columns(4)
                    a.metric("Precision LDV", f"{pm['ldv_pct']:.3f}%")
                    b.metric("Precision TDV", f"{pm['tdv_pct']:.3f}%")
                    c.metric("Precision MUT", f"{pm['mut_seconds']:.3f} sec")
                    d.metric("Trades analyzed", f"{pm['trade_count']:,}")
                else:
                    st.warning("Not enough trade data returned for that interval.")
            except Exception as e:
                st.error(str(e))

st.download_button(
    "Download Top 25 CSV",
    top.to_csv(index=False).encode("utf-8"),
    "one_hour_stock_strategy_top25.csv",
    "text/csv"
)

st.caption(
    "Research/backtesting tool only; not investment advice. Historical intraday patterns may not persist. "
    "Spreads, slippage, liquidity, halts, fees, data-feed coverage and survivorship bias can materially change results."
)
