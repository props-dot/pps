
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

# Backtest aliases
START_CT = START_TIME
END_CT = END_TIME

@st.cache_data(ttl=60*60*24, show_spinner=False)
def fetch_session(symbols_tuple, session_date, feed="iex"):
    """Fetch only 8:30–10:30 Central for one date, 1-minute bars."""
    syms = list(symbols_tuple)
    d = pd.Timestamp(session_date).date()
    start_local = datetime.combine(d, START_CT, tzinfo=CT)
    # Alpaca bar end is inclusive enough for our close/end handling.
    end_local = datetime.combine(d, END_CT, tzinfo=CT)

    params = {
        "symbols": ",".join(syms),
        "timeframe": "1Min",
        "start": start_local.astimezone(UTC).isoformat(),
        "end": end_local.astimezone(UTC).isoformat(),
        "limit": 10000,
        "adjustment": "split",
        "feed": feed,
        "sort": "asc",
    }

    rows = []
    token = None
    while True:
        if token:
            params["page_token"] = token
        elif "page_token" in params:
            del params["page_token"]

        r = requests.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            headers=alpaca_headers(),
            params=params,
            timeout=60,
        )
        if r.status_code == 429:
            time_module.sleep(2.0)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca {r.status_code}: {r.text[:500]}")

        payload = r.json()
        for sym, recs in payload.get("bars", {}).items():
            for x in recs:
                rows.append({
                    "symbol": sym,
                    "timestamp": x["t"],
                    "open": float(x["o"]),
                    "high": float(x["h"]),
                    "low": float(x["l"]),
                    "close": float(x["c"]),
                    "volume": float(x.get("v", 0)),
                })
        token = payload.get("next_page_token")
        if not token:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(CT)
    df["date"] = df["timestamp"].dt.date
    df["minute"] = (
        (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute)
        - (8*60 + 30)
    )
    return df.sort_values(["symbol", "timestamp"])

def session_complete(g):
    if g.empty:
        return False
    mins = set(g["minute"].astype(int).tolist())
    return 0 in mins and 120 in mins and len(mins) >= 115

def session_metrics(g):
    g = g.sort_values("minute")
    start = float(g.iloc[0]["open"])
    end = float(g.iloc[-1]["close"])
    gain = (end/start - 1) * 100
    ldv = max(0.0, (start - float(g["low"].min())) / start * 100)
    return gain, ldv

def price_at_or_after(g, minute):
    z = g[g["minute"] >= minute].sort_values("minute")
    if z.empty:
        return None
    return z.iloc[0]

def bar_at(g, minute):
    z = g[g["minute"] == minute]
    if z.empty:
        return None
    return z.iloc[0]

def build_history(all_df, trailing_dates, symbols):
    rows = []
    for sym in symbols:
        sym_daily = []
        complete = True
        for d in trailing_dates:
            g = all_df[(all_df["symbol"] == sym) & (all_df["date"] == d)]
            if not session_complete(g):
                complete = False
                break
            gain, ldv = session_metrics(g)
            sym_daily.append((d, gain, ldv, g))
        if not complete or len(sym_daily) != 15:
            continue

        avg_gain = float(np.mean([x[1] for x in sym_daily]))
        avg_ldv = float(np.mean([x[2] for x in sym_daily]))

        # Exact rolling historical "best average 1-hour" on a 1-minute start grid.
        best_start = None
        best_avg_1h = -np.inf
        for start_min in range(0, 61):  # 8:30 through 9:30 CT
            end_min = start_min + 60
            returns = []
            valid = True
            for _, _, _, g in sym_daily:
                b0 = bar_at(g, start_min)
                b1 = bar_at(g, end_min)
                if b0 is None or b1 is None:
                    valid = False
                    break
                ret = (float(b1["close"]) / float(b0["open"]) - 1) * 100
                returns.append(ret)
            if valid and len(returns) == 15:
                av = float(np.mean(returns))
                if av > best_avg_1h:
                    best_avg_1h = av
                    best_start = start_min

        if best_start is None:
            continue

        rows.append({
            "symbol": sym,
            "avg_2h_gain_pct": avg_gain,
            "avg_ldv_pct": avg_ldv,
            "best_start_min": int(best_start),
            "best_end_min": int(best_start + 60),
            "best_avg_1h_gain_pct": float(best_avg_1h),
        })

    rank = pd.DataFrame(rows)
    if rank.empty:
        return rank

    # Top performers = highest average 8:30–10:30 return, matching project definition.
    rank = rank.sort_values(
        ["avg_2h_gain_pct", "best_avg_1h_gain_pct", "avg_ldv_pct"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank)+1)
    return rank

def time_label(minute):
    base = datetime(2000,1,1,8,30)
    return (base + timedelta(minutes=int(minute))).strftime("%-I:%M %p")

def simulate_one_trade(g, entry_min, end_min, gain_target_pct, stop_pct):
    """
    One-minute-bar estimate.
    Entry at bar open.
    If both target and stop appear inside same minute, stop is assumed first (conservative).
    Returns exit minute, return %, exit reason.
    """
    entry_bar = price_at_or_after(g, entry_min)
    if entry_bar is None or int(entry_bar["minute"]) > end_min:
        return None

    actual_entry_min = int(entry_bar["minute"])
    entry = float(entry_bar["open"])
    target = entry * (1 + max(0.0, gain_target_pct) / 100)
    stop = entry * (1 - max(0.0, stop_pct) / 100)

    scan = g[(g["minute"] >= actual_entry_min) & (g["minute"] <= end_min)].sort_values("minute")
    if scan.empty:
        return None

    for _, b in scan.iterrows():
        lo = float(b["low"])
        hi = float(b["high"])
        m = int(b["minute"])

        stop_hit = lo <= stop if stop_pct > 0 else False
        target_hit = hi >= target if gain_target_pct > 0 else False

        if stop_hit and target_hit:
            exit_px = stop
            return {
                "entry_min": actual_entry_min, "exit_min": m,
                "entry_price": entry, "exit_price": exit_px,
                "return_pct": (exit_px/entry - 1)*100,
                "reason": "STOP (both hit same bar)"
            }
        if stop_hit:
            exit_px = stop
            return {
                "entry_min": actual_entry_min, "exit_min": m,
                "entry_price": entry, "exit_price": exit_px,
                "return_pct": (exit_px/entry - 1)*100,
                "reason": "STOP"
            }
        if target_hit:
            exit_px = target
            return {
                "entry_min": actual_entry_min, "exit_min": m,
                "entry_price": entry, "exit_price": exit_px,
                "return_pct": (exit_px/entry - 1)*100,
                "reason": "TARGET"
            }

    last = scan.iloc[-1]
    exit_px = float(last["close"])
    return {
        "entry_min": actual_entry_min, "exit_min": int(last["minute"]),
        "entry_price": entry, "exit_price": exit_px,
        "return_pct": (exit_px/entry - 1)*100,
        "reason": "BEST-HOUR END"
    }

def run_s1(subject_df, top10):
    trades = []
    for _, r in top10.iterrows():
        g = subject_df[subject_df["symbol"] == r["symbol"]]
        tr = simulate_one_trade(
            g,
            int(r["best_start_min"]),
            int(r["best_end_min"]),
            0.50 * float(r["best_avg_1h_gain_pct"]),
            0.50 * float(r["avg_ldv_pct"]),
        )
        if tr:
            trades.append({
                "symbol": r["symbol"],
                "rank": int(r["rank"]),
                **tr,
            })

    if not trades:
        return np.nan, pd.DataFrame()

    tdf = pd.DataFrame(trades)
    # Equal capital in all Top 10.
    return float(tdf["return_pct"].mean()), tdf

def run_s2(subject_df, top10):
    """
    100% capital round-robin.
    #1 begins no earlier than its historical best start.
    After each exit, the next eligible name begins at the NEXT minute to avoid
    impossible within-minute sequencing. A name is skipped if its best hour ended.
    After #10, cycle back to #1 while any best-hour window remains open.
    Daily return compounds each sequential trade.
    """
    rows = top10.sort_values("rank").to_dict("records")
    if not rows:
        return np.nan, pd.DataFrame()

    current_min = int(rows[0]["best_start_min"])
    idx = 0
    equity = 1.0
    trades = []
    no_trade_pass = 0

    # Hard guard against pathological loops.
    for _ in range(200):
        r = rows[idx]
        start = int(r["best_start_min"])
        end = int(r["best_end_min"])

        entry_min = max(current_min, start)
        did_trade = False

        if entry_min <= end:
            g = subject_df[subject_df["symbol"] == r["symbol"]]
            tr = simulate_one_trade(
                g,
                entry_min,
                end,
                0.75 * float(r["best_avg_1h_gain_pct"]),
                0.30 * float(r["avg_ldv_pct"]),
            )
            if tr:
                did_trade = True
                equity *= (1 + tr["return_pct"]/100)
                trades.append({
                    "cycle_position": idx + 1,
                    "symbol": r["symbol"],
                    "rank": int(r["rank"]),
                    **tr,
                    "equity_after": equity,
                })
                current_min = int(tr["exit_min"]) + 1

        idx = (idx + 1) % len(rows)

        if did_trade:
            no_trade_pass = 0
        else:
            no_trade_pass += 1

        # If we've checked all 10 without finding any eligible remaining window, stop.
        if no_trade_pass >= len(rows):
            break
        # No strategy trading past 10:30 CT.
        if current_min > 120:
            break

    return (equity - 1) * 100, pd.DataFrame(trades)


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


st.divider()
st.header("Rolling 15-Day Strategy Backtest")
st.caption(
    "For each subject day, the model selects that day's Top 10 using only the previous "
    "15 completed market sessions, then simulates S1 and S2 on the actual subject-day prices."
)

with st.expander("Backtest definitions & assumptions", expanded=False):
    st.markdown(
        """
**S1 — Trigger Scenario**
- Equal capital across all Top 10.
- Enter each stock at its trailing-15 historical best-hour start.
- Profit exit: **50% of best average 1-hour gain**.
- Stop exit: **50% of trailing-15 average LDV**.
- If neither triggers, exit at that stock's historical best-hour end.

**S2 — Round Robin**
- Start with rank #1 and use one pool of capital sequentially.
- Profit exit: **75% of best average 1-hour gain**.
- Stop exit: **30% of trailing-15 average LDV**.
- After an exit, move to the next ranked stock if its historical best hour is still open.
- After #10, cycle back to #1 while eligible best-hour windows remain.
- The next trade begins on the next 1-minute bar after the prior exit.
- Returns compound sequentially.

**Estimation rule**
If both the target and stop are touched inside the same 1-minute candle, the backtest
assumes the **stop occurred first**. This is deliberately conservative because 1-minute
bars do not reveal the exact intrabar order.
"""
    )

bc1, bc2 = st.columns(2)
with bc1:
    backtest_start = st.date_input(
        "First subject day",
        value=pd.Timestamp("2026-08-11").date(),
        key="backtest_subject_start",
    )
with bc2:
    backtest_end = st.date_input(
        "Last subject day",
        value=pd.Timestamp("2026-08-25").date(),
        key="backtest_subject_end",
    )

st.caption(
    f"The backtest will use the current sidebar universe ({len(symbols)} symbols) "
    f"and the current **{feed.upper()}** market-data feed."
)

if st.button("Run rolling 15-day backtest", type="primary", key="run_rolling_15_backtest"):
    if not ALPACA_KEY or not ALPACA_SECRET:
        st.error("Alpaca credentials are missing from Streamlit Secrets.")
    elif backtest_end < backtest_start:
        st.error("The last subject day must be on or after the first subject day.")
    else:
        # Pull enough calendar history to obtain at least 15 completed trading sessions.
        calendar_start = pd.Timestamp(backtest_start) - pd.Timedelta(days=35)
        calendar_end = pd.Timestamp(backtest_end)
        candidate_dates = [d.date() for d in pd.bdate_range(calendar_start, calendar_end)]

        progress = st.progress(0)
        status = st.empty()
        frames = []

        for i, d in enumerate(candidate_dates):
            status.write(f"Loading market data for {d} ({i+1}/{len(candidate_dates)})...")
            try:
                x = fetch_session(tuple(symbols), str(d), feed)
            except Exception as e:
                st.error(f"{d}: {e}")
                st.stop()
            if not x.empty:
                frames.append(x)
            progress.progress((i + 1) / len(candidate_dates))

        status.empty()

        if not frames:
            st.error("No Alpaca market data was returned for the requested period.")
        else:
            all_bt_df = pd.concat(frames, ignore_index=True)
            available_dates = sorted(all_bt_df["date"].unique())
            subject_dates = [
                d for d in available_dates
                if backtest_start <= d <= backtest_end
            ]

            daily_results = []
            all_s1_trades = []
            all_s2_trades = []
            rankings = []

            for subject_day in subject_dates:
                previous_dates = [x for x in available_dates if x < subject_day]
                if len(previous_dates) < 15:
                    continue

                trailing_dates = previous_dates[-15:]
                hist = build_history(all_bt_df, trailing_dates, symbols)
                if hist.empty or len(hist) < 10:
                    continue

                top10 = hist.head(10).copy()
                top10["subject_date"] = subject_day
                top10["best_start_ct"] = top10["best_start_min"].map(time_label)
                top10["best_end_ct"] = top10["best_end_min"].map(time_label)
                rankings.append(top10)

                subject_df = all_bt_df[all_bt_df["date"] == subject_day]

                s1_ret, s1_trades = run_s1(subject_df, top10)
                s2_ret, s2_trades = run_s2(subject_df, top10)

                if not s1_trades.empty:
                    s1_trades["subject_date"] = subject_day
                    all_s1_trades.append(s1_trades)

                if not s2_trades.empty:
                    s2_trades["subject_date"] = subject_day
                    all_s2_trades.append(s2_trades)

                daily_results.append({
                    "date": subject_day,
                    "trailing_start": trailing_dates[0],
                    "trailing_end": trailing_dates[-1],
                    "S1_return_pct": s1_ret,
                    "S2_return_pct": s2_ret,
                    "S1_trades": len(s1_trades),
                    "S2_trades": len(s2_trades),
                    "top10": ", ".join(top10["symbol"].tolist()),
                })

            results = pd.DataFrame(daily_results)

            if results.empty:
                st.error(
                    "No subject-day results could be calculated. "
                    "Try a larger stock universe or confirm the selected Alpaca feed has historical coverage."
                )
            else:
                results["S1_equity_100"] = 100 * (1 + results["S1_return_pct"]/100).cumprod()
                results["S2_equity_100"] = 100 * (1 + results["S2_return_pct"]/100).cumprod()

                st.success("Rolling 15-day backtest complete.")

                mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                mc1.metric(
                    "S1 cumulative",
                    f"{(results['S1_equity_100'].iloc[-1]/100 - 1)*100:.2f}%"
                )
                mc2.metric(
                    "S2 cumulative",
                    f"{(results['S2_equity_100'].iloc[-1]/100 - 1)*100:.2f}%"
                )
                mc3.metric("S1 avg/day", f"{results['S1_return_pct'].mean():.3f}%")
                mc4.metric("S2 avg/day", f"{results['S2_return_pct'].mean():.3f}%")
                mc5.metric(
                    "S1 winning days",
                    f"{(results['S1_return_pct'] > 0).mean()*100:.1f}%"
                )
                mc6.metric(
                    "S2 winning days",
                    f"{(results['S2_return_pct'] > 0).mean()*100:.1f}%"
                )

                st.subheader("Estimated daily returns")
                st.dataframe(results, use_container_width=True, hide_index=True)

                st.download_button(
                    "Download daily backtest results CSV",
                    results.to_csv(index=False).encode("utf-8"),
                    "rolling_15_day_backtest_daily.csv",
                    "text/csv",
                    key="download_backtest_daily",
                )

                if rankings:
                    rank_df = pd.concat(rankings, ignore_index=True)
                    with st.expander("Top 10 selected for each subject day"):
                        st.dataframe(rank_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download Top-10 rankings CSV",
                            rank_df.to_csv(index=False).encode("utf-8"),
                            "rolling_15_day_rankings.csv",
                            "text/csv",
                            key="download_backtest_rankings",
                        )

                if all_s1_trades:
                    s1df = pd.concat(all_s1_trades, ignore_index=True)
                    with st.expander("S1 trade log"):
                        st.dataframe(s1df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download S1 trade log CSV",
                            s1df.to_csv(index=False).encode("utf-8"),
                            "S1_trade_log.csv",
                            "text/csv",
                            key="download_s1_log",
                        )

                if all_s2_trades:
                    s2df = pd.concat(all_s2_trades, ignore_index=True)
                    with st.expander("S2 trade log"):
                        st.dataframe(s2df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download S2 trade log CSV",
                            s2df.to_csv(index=False).encode("utf-8"),
                            "S2_trade_log.csv",
                            "text/csv",
                            key="download_s2_log",
                        )

st.caption(
    "Backtest estimates exclude commissions, bid/ask spread, slippage, taxes and market impact. "
    "Historical performance does not guarantee future results."
)
