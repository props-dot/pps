
import os
import math
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

CT = ZoneInfo("America/Chicago")

st.set_page_config(page_title="One-Hour Stock Strategy", layout="wide")
st.title("One-Hour Stock Strategy — V1")
st.caption("Ranks stocks by repeatable intraday gains in the 8:30–10:30 AM Central window.")

# -----------------------------
# Configuration
# -----------------------------
START_TIME = time(8, 30)
END_TIME = time(10, 30)
TRAILING_DAYS = 30
TOP_N = 25

def load_sample_data():
    path = os.path.join(os.path.dirname(__file__), "sample_intraday.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(CT)
    return df

def load_csv(uploaded):
    df = pd.read_csv(uploaded, parse_dates=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(CT)
    else:
        ts = ts.dt.tz_convert(CT)
    df["timestamp"] = ts
    return df

def make_sample_data():
    rng = np.random.default_rng(42)
    symbols = ["AAPL","MSFT","NVDA","META","AMZN","GOOGL","AVGO","AMD","CRM","NFLX",
               "JPM","V","MA","COST","WMT","HD","LLY","UNH","XOM","CVX",
               "ORCL","ADBE","INTU","QCOM","TXN","PANW","BKNG","UBER","GE","CAT"]
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=35)
    rows = []
    for si, sym in enumerate(symbols):
        base = 40 + si*7
        drift = 0.00008 + (si % 9) * 0.000008
        smoothness = 0.00045 + (si % 5) * 0.00008
        for d in dates:
            start = pd.Timestamp.combine(d.date(), START_TIME).tz_localize(CT)
            price = base * (1 + rng.normal(0, 0.02))
            for sec in range(0, 2*60*60+1, 30):
                t = start + pd.Timedelta(seconds=sec)
                # gentle upward bias with intermittent down moves
                ret = drift + rng.normal(0, smoothness)
                price *= max(0.90, 1 + ret)
                rows.append((sym, t, price))
    return pd.DataFrame(rows, columns=["symbol","timestamp","price"])

def ensure_sample():
    path = os.path.join(os.path.dirname(__file__), "sample_intraday.csv")
    if not os.path.exists(path):
        make_sample_data().to_csv(path, index=False)

def filter_window(df):
    df = df.copy()
    df = df.dropna(subset=["symbol","timestamp","price"])
    df["date"] = df["timestamp"].dt.date
    tt = df["timestamp"].dt.time
    return df[(tt >= START_TIME) & (tt <= END_TIME)].sort_values(["symbol","date","timestamp"])

def _pct(a, b):
    return (b / a - 1.0) * 100.0 if a and not pd.isna(a) else np.nan

def contiguous_runs(direction, times):
    runs = []
    start_idx = None
    for i, val in enumerate(direction):
        if val:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                runs.append((start_idx, i))
                start_idx = None
    if start_idx is not None:
        runs.append((start_idx, len(direction)))
    durations = []
    for a,b in runs:
        if b < len(times):
            durations.append((times[b] - times[a]).total_seconds())
        else:
            durations.append((times[-1] - times[a]).total_seconds())
    return [x for x in durations if x >= 0]

def daily_metrics(g):
    g = g.sort_values("timestamp")
    prices = g["price"].astype(float).to_numpy()
    times = list(g["timestamp"])
    if len(prices) < 2:
        return None
    start_price, end_price = prices[0], prices[-1]
    gain = _pct(start_price, end_price)
    low = float(np.min(prices))
    ldv = max(0.0, (start_price - low)/start_price * 100.0)

    step_returns = np.diff(prices) / prices[:-1] * 100.0
    tdv = float(np.abs(step_returns[step_returns < 0]).sum())

    up_dirs = step_returns > 0
    up_durations = contiguous_runs(up_dirs, times[:-1])
    if np.all(up_dirs):
        mut = (times[-1] - times[0]).total_seconds()
    elif up_durations:
        mut = min(up_durations)
    else:
        mut = 0.0

    # Best 60-minute contiguous interval
    best_gain = -np.inf
    best_start = None
    best_end = None
    ts = pd.Series(times)
    for i, t0 in enumerate(times):
        target = t0 + pd.Timedelta(minutes=60)
        j = np.searchsorted(np.array([x.value for x in ts]), target.value, side="right") - 1
        if j > i:
            g1 = _pct(prices[i], prices[j])
            if g1 > best_gain:
                best_gain, best_start, best_end = g1, times[i], times[j]

    return {
        "start_price": start_price,
        "end_price": end_price,
        "gain_pct": gain,
        "ldv_pct": ldv,
        "tdv_pct": tdv,
        "mut_seconds": mut,
        "best_1h_gain_pct": best_gain if best_gain != -np.inf else np.nan,
        "best_1h_start": best_start,
        "best_1h_end": best_end,
    }

def analyze(df):
    df = filter_window(df)
    # last 30 distinct market days
    dates = sorted(df["date"].unique())[-TRAILING_DAYS:]
    df = df[df["date"].isin(dates)]

    daily = []
    for (sym, d), g in df.groupby(["symbol","date"]):
        m = daily_metrics(g)
        if m:
            daily.append({"symbol":sym, "date":d, **m})
    daily = pd.DataFrame(daily)
    if daily.empty:
        return daily, pd.DataFrame(), df

    summary = daily.groupby("symbol").agg(
        avg_gain_pct=("gain_pct","mean"),
        positive_days=("gain_pct", lambda s: int((s > 0).sum())),
        days_observed=("gain_pct","count"),
        win_rate_pct=("gain_pct", lambda s: float((s > 0).mean()*100)),
        avg_ldv_pct=("ldv_pct","mean"),
        avg_tdv_pct=("tdv_pct","mean"),
        avg_mut_seconds=("mut_seconds","mean"),
        avg_best_1h_gain_pct=("best_1h_gain_pct","mean"),
        median_gain_pct=("gain_pct","median"),
        gain_std_pct=("gain_pct","std"),
    ).reset_index()

    # Primary rank: average gain; secondary preference: repeatability/smoothness
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

ensure_sample()
st.sidebar.header("Data")
source = st.sidebar.radio("Source", ["Built-in sample data", "Upload CSV"])
st.sidebar.caption("CSV columns: symbol, timestamp, price. Second/tick-level data is preferred for MUT and TDV.")
if source == "Upload CSV":
    up = st.sidebar.file_uploader("Upload intraday CSV", type=["csv"])
    if up is None:
        st.info("Upload a CSV or switch to the built-in sample data.")
        st.stop()
    raw = load_csv(up)
else:
    raw = load_sample_data()

daily, summary, windowed = analyze(raw)
if summary.empty:
    st.warning("No usable observations in the 8:30–10:30 AM Central window.")
    st.stop()

top = summary.head(TOP_N).copy()
latest_date = max(daily["date"])
latest = daily[daily["date"] == latest_date].copy()
latest = latest.merge(top[["symbol","rank"]], on="symbol", how="inner").sort_values("rank")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Universe", f"{summary['symbol'].nunique()} stocks")
c2.metric("Trailing sessions", f"{daily['date'].nunique()}")
c3.metric("Top list", f"{len(top)}")
c4.metric("Latest session", str(latest_date))

st.subheader("Top 25 — trailing 30 sessions")
view = top[[
    "rank","symbol","avg_gain_pct","win_rate_pct","avg_best_1h_gain_pct",
    "avg_ldv_pct","avg_tdv_pct","avg_mut_seconds","median_gain_pct","gain_std_pct"
]].copy()
view.columns = [
    "Rank","Symbol","Avg 8:30–10:30 Gain %","Positive-Day %","Avg Best 1-Hour Gain %",
    "Avg LDV %","Avg TDV %","Avg MUT (sec)","Median Gain %","Gain Std Dev %"
]
st.dataframe(view, use_container_width=True, hide_index=True)

st.subheader(f"Daily results — {latest_date}")
if latest.empty:
    st.write("No Top-25 observations for the latest session.")
else:
    lv = latest[[
        "rank","symbol","start_price","end_price","gain_pct","best_1h_gain_pct",
        "best_1h_start","best_1h_end","ldv_pct","tdv_pct","mut_seconds"
    ]].copy()
    lv["best_1h_start"] = pd.to_datetime(lv["best_1h_start"]).dt.strftime("%H:%M:%S")
    lv["best_1h_end"] = pd.to_datetime(lv["best_1h_end"]).dt.strftime("%H:%M:%S")
    lv.columns = [
        "Rank","Symbol","8:30 Price","10:30 Price","Daily Gain %",
        "Best 1-Hour Gain %","Best 1-Hour Start","Best 1-Hour End",
        "LDV %","TDV %","MUT (sec)"
    ]
    st.dataframe(lv, use_container_width=True, hide_index=True)

st.subheader("Intraday chart")
symbol = st.selectbox("Stock", top["symbol"].tolist())
date_choices = sorted(daily[daily["symbol"] == symbol]["date"].unique(), reverse=True)
sel_date = st.selectbox("Session", date_choices)
chart = windowed[(windowed["symbol"] == symbol) & (windowed["date"] == sel_date)]
fig = px.line(chart, x="timestamp", y="price", title=f"{symbol} — {sel_date} — 8:30–10:30 CT")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Trailing performance")
td = daily[daily["symbol"] == symbol].sort_values("date")
fig2 = px.bar(td, x="date", y="gain_pct", title=f"{symbol} daily 8:30–10:30 gain (%)")
st.plotly_chart(fig2, use_container_width=True)

st.download_button(
    "Download Top 25 CSV",
    top.to_csv(index=False).encode("utf-8"),
    file_name="one_hour_stock_strategy_top25.csv",
    mime="text/csv",
)

st.caption(
    "Research/backtesting tool only. Rankings are historical and do not guarantee future returns. "
    "Transaction costs, spreads, slippage, halts, liquidity, survivorship bias and data-feed differences can materially change results."
)
