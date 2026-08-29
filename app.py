
import os
import time as time_module
import base64
import gzip
import io
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
st.title("One-Hour Stock Strategy — V9")
st.caption("Persistent-data edition with click-to-generate backtests, live processing percentages, and the new S3 LDV Scenario.")

def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

ALPACA_KEY = get_secret("ALPACA_API_KEY")
ALPACA_SECRET = get_secret("ALPACA_SECRET_KEY")

GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO = get_secret("GITHUB_REPO", "props-dot/pps")
GITHUB_BRANCH = get_secret("GITHUB_BRANCH", "main")
GITHUB_DATA_PATH = get_secret("GITHUB_DATA_PATH", "data/market_history.csv.gz")

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


@st.cache_data(ttl=60*60*24, show_spinner=False)
def fetch_backtest_range(symbols_tuple, start_date, end_date, feed="iex"):
    """
    Fetch the full backtest range in bulk instead of one trading day at a time.
    Symbols are batched and Alpaca pagination is handled automatically.
    """
    symbols = list(symbols_tuple)
    start_local = datetime.combine(pd.Timestamp(start_date).date(), START_CT, tzinfo=CT)
    end_local = datetime.combine(pd.Timestamp(end_date).date(), END_CT, tzinfo=CT)

    all_rows = []
    batch_size = 20

    for b0 in range(0, len(symbols), batch_size):
        batch = symbols[b0:b0 + batch_size]
        token = None
        retry_count = 0

        while True:
            params = {
                "symbols": ",".join(batch),
                "timeframe": "1Min",
                "start": start_local.astimezone(UTC).isoformat(),
                "end": end_local.astimezone(UTC).isoformat(),
                "limit": 10000,
                "adjustment": "split",
                "feed": feed,
                "sort": "asc",
            }
            if token:
                params["page_token"] = token

            r = requests.get(
                "https://data.alpaca.markets/v2/stocks/bars",
                headers=alpaca_headers(),
                params=params,
                timeout=60,
            )

            if r.status_code == 429:
                retry_count += 1
                if retry_count > 8:
                    raise RuntimeError(
                        "Alpaca rate limit persisted after 8 retries. "
                        "Wait about a minute and run the backtest again."
                    )
                time_module.sleep(min(2 ** retry_count, 30))
                continue

            retry_count = 0

            if r.status_code >= 400:
                raise RuntimeError(f"Alpaca returned {r.status_code}: {r.text[:500]}")

            payload = r.json()

            for sym, recs in payload.get("bars", {}).items():
                for x in recs:
                    all_rows.append({
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

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(CT)
    df["date"] = df["timestamp"].dt.date
    df["minute"] = (
        (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute)
        - (8 * 60 + 30)
    )

    df = df[(df["minute"] >= 0) & (df["minute"] <= 120)]
    return df.sort_values(["symbol", "date", "minute"]).reset_index(drop=True)


def build_session_map(df):
    """Index each stock/session once for fast rolling-window calculations."""
    return {
        (sym, d): g.sort_values("minute").reset_index(drop=True)
        for (sym, d), g in df.groupby(["symbol", "date"], sort=False)
    }

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



def build_history(session_map, trailing_dates, symbols, window_minutes=60,
                  same_day_mode=False, subject_day=None):
    rows = []
    analysis_dates = [subject_day] if same_day_mode else list(trailing_dates)
    required_count = len(analysis_dates)

    if required_count < 1:
        return pd.DataFrame()

    max_start = 120 - int(window_minutes)
    if max_start < 0:
        return pd.DataFrame()

    for sym in symbols:
        sym_daily = []

        for d in analysis_dates:
            g = session_map.get((sym, d))
            if g is None or not session_complete(g):
                sym_daily = []
                break
            gain, ldv = session_metrics(g)
            sym_daily.append((d, gain, ldv, g))

        if len(sym_daily) != required_count:
            continue

        avg_gain = float(np.mean([x[1] for x in sym_daily]))
        median_ldv = float(np.median([x[2] for x in sym_daily]))

        # Convert each historical session to a minute lookup once.
        minute_maps = [
            {int(r["minute"]): r for _, r in g.iterrows()}
            for _, _, _, g in sym_daily
        ]

        best_start = None
        best_avg_window = -np.inf

        for start_min in range(0, max_start + 1):
            end_min = start_min + int(window_minutes)
            returns = []

            for mm in minute_maps:
                b0 = mm.get(start_min)
                b1 = mm.get(end_min)
                if b0 is None or b1 is None:
                    returns = []
                    break
                returns.append(
                    (float(b1["close"]) / float(b0["open"]) - 1) * 100
                )

            if len(returns) == required_count:
                av = float(np.mean(returns))
                if av > best_avg_window:
                    best_avg_window = av
                    best_start = start_min

        if best_start is None:
            continue

        daily_best_window_gains = []

        for mm in minute_maps:
            day_best = -np.inf

            for start_min in range(0, max_start + 1):
                end_min = start_min + int(window_minutes)
                b0 = mm.get(start_min)
                b1 = mm.get(end_min)

                if b0 is None or b1 is None:
                    continue

                ret = (float(b1["close"]) / float(b0["open"]) - 1) * 100
                if ret > day_best:
                    day_best = ret

            if np.isfinite(day_best):
                daily_best_window_gains.append(float(day_best))

        if len(daily_best_window_gains) != required_count:
            continue

        rows.append({
            "symbol": sym,
            "avg_2h_gain_pct": avg_gain,
            "median_ldv_pct": median_ldv,
            "best_start_min": int(best_start),
            "best_end_min": int(best_start + int(window_minutes)),
            "best_avg_window_gain_pct": float(best_avg_window),
            "median_daily_best_window_gain_pct": float(np.median(daily_best_window_gains)),
            "analysis_sessions": required_count,
            "window_minutes": int(window_minutes),
        })

    rank = pd.DataFrame(rows)
    if rank.empty:
        return rank

    rank = rank.sort_values(
        ["avg_2h_gain_pct", "best_avg_window_gain_pct", "median_ldv_pct"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    rank["rank"] = np.arange(1, len(rank) + 1)
    return rank

def time_label(minute):
    base = datetime(2000,1,1,8,30)
    return (base + timedelta(minutes=int(minute))).strftime("%-I:%M %p")

def simulate_one_trade(g, entry_min, end_min, gain_target_pct, stop_pct, end_reason="BEST-WINDOW END"):
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
        "reason": end_reason
    }


def run_s1(subject_df, top10, profit_trigger_pct=90.0, loss_trigger_pct=80.0):
    """
    Equal-weight S1 portfolio. Capital is divided across every selected name.
    If a selected stock cannot be traded because its subject-day data are missing,
    that allocation stays in cash (0% return) rather than being re-weighted.
    """
    trades = []
    selected_count = max(1, len(top10))

    for _, r in top10.iterrows():
        g = subject_df[subject_df["symbol"] == r["symbol"]]

        gain_target = (float(profit_trigger_pct) / 100.0) * float(r["median_daily_best_window_gain_pct"])
        stop_target = (float(loss_trigger_pct) / 100.0) * float(r["median_ldv_pct"])

        tr = simulate_one_trade(
            g,
            int(r["best_start_min"]),
            int(r["best_end_min"]),
            gain_target,
            stop_target,
        )

        if tr:
            trades.append({
                "symbol": r["symbol"],
                "rank": int(r["rank"]),
                "portfolio_weight_pct": 100.0 / selected_count,
                "profit_trigger_setting_pct": float(profit_trigger_pct),
                "loss_trigger_setting_pct": float(loss_trigger_pct),
                "profit_target_pct": gain_target,
                "stop_target_pct": stop_target,
                **tr,
            })

    if not trades:
        return 0.0, pd.DataFrame()

    tdf = pd.DataFrame(trades)
    # Equal allocation is based on the full selected list, not just names with usable bars.
    portfolio_return = float(tdf["return_pct"].sum() / selected_count)
    return portfolio_return, tdf


def run_s2(subject_df, top10, profit_trigger_pct=90.0, loss_trigger_pct=80.0):
    """
    Sequential round-robin using 100% of current capital per trade.
    """
    rows = top10.sort_values("rank").to_dict("records")
    if not rows:
        return np.nan, pd.DataFrame()

    current_min = int(rows[0]["best_start_min"])
    idx = 0
    equity = 1.0
    trades = []
    no_trade_pass = 0

    for _ in range(300):
        r = rows[idx]
        start = int(r["best_start_min"])
        end = int(r["best_end_min"])

        entry_min = max(current_min, start)
        did_trade = False

        if entry_min <= end:
            g = subject_df[subject_df["symbol"] == r["symbol"]]

            gain_target = (float(profit_trigger_pct) / 100.0) * float(r["median_daily_best_window_gain_pct"])
            stop_target = (float(loss_trigger_pct) / 100.0) * float(r["median_ldv_pct"])

            tr = simulate_one_trade(
                g,
                entry_min,
                end,
                gain_target,
                stop_target,
            )

            if tr:
                did_trade = True
                equity *= (1 + tr["return_pct"] / 100)

                trades.append({
                    "cycle_position": idx + 1,
                    "symbol": r["symbol"],
                    "rank": int(r["rank"]),
                    "profit_trigger_setting_pct": float(profit_trigger_pct),
                    "loss_trigger_setting_pct": float(loss_trigger_pct),
                    "profit_target_pct": gain_target,
                    "stop_target_pct": stop_target,
                    **tr,
                    "equity_after": equity,
                })

                current_min = int(tr["exit_min"]) + 1

        idx = (idx + 1) % len(rows)

        if did_trade:
            no_trade_pass = 0
        else:
            no_trade_pass += 1

        if no_trade_pass >= len(rows):
            break
        if current_min > 120:
            break

    return (equity - 1) * 100, pd.DataFrame(trades)




def build_s3_history(session_map, trailing_dates, symbols, top_n=10,
                     entry_step_minutes=5, minimum_hold_minutes=20):
    """
    Build the S3 profile using prior completed sessions only.

    1) Rank the universe by average full-window (8:30-10:30) gain.
    2) Keep Top N.
    3) For each selected stock independently, search entry/exit pairs on a
       5-minute grid with a minimum 20-minute holding period.
    4) Choose the pair with the highest average lookback return.
    5) Compute that stock's average LDV only inside its selected trade window.
    """
    trailing_dates = list(trailing_dates)
    required_count = len(trailing_dates)
    top_n = max(1, min(25, int(top_n)))
    if required_count < 1:
        return pd.DataFrame()

    session_cache = {}
    base_rows = []

    for sym in symbols:
        sessions = []
        gains = []
        full_ldvs = []
        valid = True

        for d in trailing_dates:
            g = session_map.get((sym, d))
            if g is None or not session_complete(g):
                valid = False
                break
            gain, ldv = session_metrics(g)
            sessions.append(g.sort_values("minute").reset_index(drop=True))
            gains.append(float(gain))
            full_ldvs.append(float(ldv))

        if not valid or len(sessions) != required_count:
            continue

        session_cache[sym] = sessions
        base_rows.append({
            "symbol": sym,
            "avg_2h_gain_pct": float(np.mean(gains)),
            "avg_2h_ldv_pct": float(np.mean(full_ldvs)),
        })

    base = pd.DataFrame(base_rows)
    if base.empty:
        return base

    base = base.sort_values(
        ["avg_2h_gain_pct", "avg_2h_ldv_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)
    base["rank"] = np.arange(1, len(base) + 1)
    selected = base.head(top_n).copy()

    detailed = []
    entry_step_minutes = max(1, int(entry_step_minutes))
    minimum_hold_minutes = max(entry_step_minutes, int(minimum_hold_minutes))

    for _, base_row in selected.iterrows():
        sym = base_row["symbol"]
        sessions = session_cache[sym]

        # Create compact exact-minute lookups once for each historical session.
        prepared = []
        for g in sessions:
            minute_rows = {int(r["minute"]): r for _, r in g.iterrows()}
            lows = np.full(121, np.nan, dtype=float)
            for _, r in g.iterrows():
                m = int(r["minute"])
                if 0 <= m <= 120:
                    lows[m] = float(r["low"])
            prepared.append((minute_rows, lows))

        best = None
        max_entry = 120 - minimum_hold_minutes

        for entry_min in range(0, max_entry + 1, entry_step_minutes):
            first_exit = entry_min + minimum_hold_minutes
            for exit_min in range(first_exit, 121, entry_step_minutes):
                returns = []
                ldvs = []
                valid_pair = True

                for minute_rows, lows in prepared:
                    b0 = minute_rows.get(entry_min)
                    b1 = minute_rows.get(exit_min)
                    if b0 is None or b1 is None:
                        valid_pair = False
                        break

                    entry_px = float(b0["open"])
                    exit_px = float(b1["close"])
                    if not np.isfinite(entry_px) or entry_px <= 0:
                        valid_pair = False
                        break

                    window_lows = lows[entry_min:exit_min + 1]
                    finite_lows = window_lows[np.isfinite(window_lows)]
                    if finite_lows.size == 0:
                        valid_pair = False
                        break

                    ret = (exit_px / entry_px - 1.0) * 100.0
                    min_low = float(np.min(finite_lows))
                    ldv = max(0.0, (entry_px - min_low) / entry_px * 100.0)
                    returns.append(ret)
                    ldvs.append(ldv)

                if not valid_pair or len(returns) != required_count:
                    continue

                avg_gain = float(np.mean(returns))
                avg_ldv = float(np.mean(ldvs))
                duration = int(exit_min - entry_min)

                candidate = {
                    "entry_min": int(entry_min),
                    "exit_min": int(exit_min),
                    "duration_minutes": duration,
                    "avg_gain_pct": avg_gain,
                    "avg_ldv_pct": avg_ldv,
                }

                if best is None:
                    best = candidate
                else:
                    # Highest average gain first; then lower LDV; then shorter window.
                    better_gain = candidate["avg_gain_pct"] > best["avg_gain_pct"] + 1e-12
                    tied_gain = abs(candidate["avg_gain_pct"] - best["avg_gain_pct"]) <= 1e-12
                    lower_ldv = candidate["avg_ldv_pct"] < best["avg_ldv_pct"] - 1e-12
                    tied_ldv = abs(candidate["avg_ldv_pct"] - best["avg_ldv_pct"]) <= 1e-12
                    shorter = candidate["duration_minutes"] < best["duration_minutes"]
                    if better_gain or (tied_gain and lower_ldv) or (tied_gain and tied_ldv and shorter):
                        best = candidate

        if best is None:
            continue

        detailed.append({
            "rank": int(base_row["rank"]),
            "symbol": sym,
            "avg_2h_gain_pct": float(base_row["avg_2h_gain_pct"]),
            "avg_2h_ldv_pct": float(base_row["avg_2h_ldv_pct"]),
            "s3_avg_gain_pct": float(best["avg_gain_pct"]),
            "s3_avg_ldv_pct": float(best["avg_ldv_pct"]),
            "s3_entry_min": int(best["entry_min"]),
            "s3_exit_min": int(best["exit_min"]),
            "s3_duration_minutes": int(best["duration_minutes"]),
            "analysis_sessions": required_count,
            "entry_step_minutes": entry_step_minutes,
            "minimum_hold_minutes": minimum_hold_minutes,
        })

    out = pd.DataFrame(detailed)
    if out.empty:
        return out
    return out.sort_values("rank").reset_index(drop=True)


def run_s3(subject_df, s3_selection, gain_multiplier_pct=100.0, ldv_multiplier_pct=100.0):
    """
    Equal-weight S3 portfolio.

    Each selected stock uses its own historical average gain, historical
    trade-window LDV, and automatically selected entry/exit window.
    """
    trades = []
    selected_count = max(1, len(s3_selection))

    for _, r in s3_selection.sort_values("rank").iterrows():
        g = subject_df[subject_df["symbol"] == r["symbol"]]

        historical_gain = float(r["s3_avg_gain_pct"])
        historical_ldv = float(r["s3_avg_ldv_pct"])
        gain_target = max(0.0, float(gain_multiplier_pct) / 100.0 * historical_gain)
        stop_target = max(0.0, float(ldv_multiplier_pct) / 100.0 * historical_ldv)

        tr = simulate_one_trade(
            g,
            int(r["s3_entry_min"]),
            int(r["s3_exit_min"]),
            gain_target,
            stop_target,
            end_reason="AUTO WINDOW END",
        )

        if tr:
            trades.append({
                "symbol": r["symbol"],
                "rank": int(r["rank"]),
                "portfolio_weight_pct": 100.0 / selected_count,
                "historical_avg_2h_gain_pct": float(r["avg_2h_gain_pct"]),
                "historical_s3_avg_gain_pct": historical_gain,
                "historical_s3_avg_ldv_pct": historical_ldv,
                "auto_entry_min": int(r["s3_entry_min"]),
                "auto_exit_min": int(r["s3_exit_min"]),
                "auto_entry_ct": time_label(int(r["s3_entry_min"])),
                "auto_exit_ct": time_label(int(r["s3_exit_min"])),
                "historical_window_minutes": int(r["s3_duration_minutes"]),
                "gain_multiplier_setting_pct": float(gain_multiplier_pct),
                "ldv_multiplier_setting_pct": float(ldv_multiplier_pct),
                "profit_target_pct": gain_target,
                "stop_target_pct": stop_target,
                **tr,
            })

    if not trades:
        return 0.0, pd.DataFrame()

    tdf = pd.DataFrame(trades)
    portfolio_return = float(tdf["return_pct"].sum() / selected_count)
    return portfolio_return, tdf


def csv_with_metadata(df, metadata):
    """Create a human-readable CSV with test parameters first, then the result table."""
    out = io.StringIO()
    out.write("BACKTEST PARAMETERS\n")
    for key, value in metadata.items():
        safe_key = str(key).replace('"', '""')
        safe_value = str(value).replace('"', '""')
        out.write(f'"{safe_key}","{safe_value}"\n')
    out.write("\nRESULTS\n")
    df.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")


def portfolio_stats(returns, starting_capital=1000.0):
    s = pd.Series(list(returns), dtype=float).dropna()
    if s.empty:
        return {
            "ending_value": float(starting_capital),
            "total_return_pct": 0.0,
            "dollar_gain_loss": 0.0,
            "avg_daily_return_pct": np.nan,
            "win_rate_pct": np.nan,
            "max_drawdown_pct": 0.0,
            "best_day_pct": np.nan,
            "worst_day_pct": np.nan,
        }
    equity = float(starting_capital) * (1 + s / 100.0).cumprod()
    equity_with_start = pd.concat(
        [pd.Series([float(starting_capital)], dtype=float), equity.reset_index(drop=True)],
        ignore_index=True,
    )
    running_peak = equity_with_start.cummax()
    drawdown = (equity_with_start / running_peak - 1.0) * 100.0
    ending = float(equity.iloc[-1])
    return {
        "ending_value": ending,
        "total_return_pct": (ending / float(starting_capital) - 1.0) * 100.0,
        "dollar_gain_loss": ending - float(starting_capital),
        "avg_daily_return_pct": float(s.mean()),
        "win_rate_pct": float((s > 0).mean() * 100.0),
        "max_drawdown_pct": float(drawdown.min()),
        "best_day_pct": float(s.max()),
        "worst_day_pct": float(s.min()),
    }


def optimize_triggers_for_subjects(session_map, subject_days, top10_by_day,
                                   scenario="S1", trigger_values=None,
                                   starting_capital=1000.0,
                                   progress_callback=None):
    """
    Hindsight optimizer over supplied subject days and precomputed selections.
    Optional progress_callback(completed_combinations, total_combinations) lets
    the dashboard show a real completion percentage.
    """
    if trigger_values is None:
        trigger_values = list(range(25, 201, 25))

    total_combinations = max(1, len(trigger_values) * len(trigger_values))
    completed_combinations = 0
    best = None

    for profit_pct in trigger_values:
        for loss_pct in trigger_values:
            daily_returns = []
            valid_days = 0
            for d in subject_days:
                top10 = top10_by_day.get(d)
                if top10 is None or top10.empty:
                    continue

                frames = [
                    session_map[(sym, d)]
                    for sym in top10["symbol"].tolist()
                    if (sym, d) in session_map
                ]
                subject_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

                if scenario == "S1":
                    ret, _ = run_s1(
                        subject_df, top10,
                        profit_trigger_pct=profit_pct,
                        loss_trigger_pct=loss_pct,
                    )
                else:
                    ret, _ = run_s2(
                        subject_df, top10,
                        profit_trigger_pct=profit_pct,
                        loss_trigger_pct=loss_pct,
                    )

                if pd.notna(ret):
                    daily_returns.append(float(ret))
                    valid_days += 1

            if daily_returns:
                stats = portfolio_stats(daily_returns, starting_capital=starting_capital)
                candidate = {
                    "profit_trigger_pct": profit_pct,
                    "loss_trigger_pct": loss_pct,
                    "subject_days": valid_days,
                    **stats,
                }
                if best is None or candidate["ending_value"] > best["ending_value"]:
                    best = candidate

            completed_combinations += 1
            if progress_callback:
                progress_callback(completed_combinations, total_combinations)

    return best

def prepare_daily_optimizer_selections(session_map, available_dates, symbols,
                                       subject_day, lookbacks, window_minutes=60):
    selections = {}
    previous_dates = [d for d in available_dates if d < subject_day]
    for lb in lookbacks:
        if len(previous_dates) < lb:
            continue
        trailing = previous_dates[-lb:]
        hist = build_history(
            session_map, trailing, symbols,
            window_minutes=window_minutes,
            same_day_mode=False,
            subject_day=None,
        )
        if not hist.empty and len(hist) >= 10:
            selections[lb] = hist.head(10).copy()
    return selections

def _history_columns():
    return ["symbol", "timestamp", "open", "high", "low", "close", "volume"]

def _normalize_history(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=_history_columns() + ["date", "minute"])

    df = df.copy()

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["symbol"] = df["symbol"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(CT)
    df["date"] = df["timestamp"].dt.date
    df["minute"] = (
        (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute)
        - (8 * 60 + 30)
    )

    df = df[(df["minute"] >= 0) & (df["minute"] <= 120)]
    df = df.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

def _github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def load_history_from_github():
    """
    Returns (history_df, sha).

    A missing, empty, placeholder, or corrupt gzip source is treated as empty history.
    The existing SHA is preserved so the next save can overwrite the bad file.
    """
    if not GITHUB_TOKEN:
        return pd.DataFrame(), None

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_DATA_PATH}"
    r = requests.get(
        url,
        headers=_github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=45,
    )

    if r.status_code == 404:
        return pd.DataFrame(), None

    if r.status_code >= 400:
        raise RuntimeError(
            f"GitHub history read failed ({r.status_code}): {r.text[:500]}"
        )

    payload = r.json()
    sha = payload.get("sha")

    try:
        raw_bytes = base64.b64decode(payload.get("content", ""))

        if not raw_bytes or len(raw_bytes) < 2:
            return pd.DataFrame(), sha

        # Valid gzip files begin with magic bytes 1f 8b.
        if raw_bytes[:2] != b"\x1f\x8b":
            return pd.DataFrame(), sha

        with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes), mode="rb") as gz:
            df = pd.read_csv(gz)

        return _normalize_history(df), sha

    except (gzip.BadGzipFile, EOFError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame(), sha

def save_history_to_github(df, sha=None, commit_message="Update market history"):
    """
    Persists the compressed source file back to the private GitHub repository.
    """
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "GITHUB_TOKEN is not configured, so persistent history cannot be saved."
        )

    clean = _normalize_history(df)
    export = clean[_history_columns()].copy()

    csv_bytes = export.to_csv(index=False).encode("utf-8")
    gz_bytes = gzip.compress(csv_bytes, compresslevel=6)

    body = {
        "message": commit_message,
        "content": base64.b64encode(gz_bytes).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        body["sha"] = sha

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_DATA_PATH}"
    r = requests.put(
        url,
        headers=_github_headers(),
        json=body,
        timeout=90,
    )

    if r.status_code >= 400:
        raise RuntimeError(
            f"GitHub history save failed ({r.status_code}): {r.text[:500]}"
        )

    return r.json()["content"]["sha"]

def _merge_history(old_df, new_df):
    if old_df is None or old_df.empty:
        return _normalize_history(new_df)
    if new_df is None or new_df.empty:
        return _normalize_history(old_df)

    merged = pd.concat([old_df, new_df], ignore_index=True)
    return _normalize_history(merged)

def _fetch_range_if_valid(symbols, start_date, end_date, feed):
    if start_date is None or end_date is None:
        return pd.DataFrame()
    if pd.Timestamp(start_date).date() > pd.Timestamp(end_date).date():
        return pd.DataFrame()

    return fetch_backtest_range(
        tuple(symbols),
        pd.Timestamp(start_date).date(),
        pd.Timestamp(end_date).date(),
        feed,
    )

def ensure_persistent_history(
    symbols,
    feed,
    required_start=None,
    required_end=None,
    force_reconfirm=False,
    status_box=None,
):
    """
    Persistent market-history manager.

    - Loads the saved GitHub source file once per Streamlit session.
    - Adds only missing dates/symbols.
    - Saves only when new data were actually added.
    - `force_reconfirm=True` re-downloads the entire currently saved period and replaces it.
    """
    symbols = list(dict.fromkeys(symbols))
    today_ct = datetime.now(CT).date()

    if required_end is None:
        required_end = today_ct
    required_end = min(pd.Timestamp(required_end).date(), today_ct)

    if required_start is None:
        required_start = (pd.Timestamp(required_end) - pd.Timedelta(days=75)).date()
    else:
        required_start = pd.Timestamp(required_start).date()

    if status_box:
        status_box.write("Checking existing data...")

    # Load persistent source once per browser/session.
    if "persistent_history_df" not in st.session_state or force_reconfirm:
        if force_reconfirm:
            history = pd.DataFrame()
            sha = None
        elif GITHUB_TOKEN:
            history, sha = load_history_from_github()
        else:
            history = pd.DataFrame()
            sha = None

        st.session_state["persistent_history_df"] = _normalize_history(history)
        st.session_state["persistent_history_sha"] = sha

    history = st.session_state["persistent_history_df"]
    sha = st.session_state.get("persistent_history_sha")

    if force_reconfirm:
        # If a source file already existed, preserve its earliest date;
        # otherwise use the required range.
        try:
            old_history, old_sha = load_history_from_github() if GITHUB_TOKEN else (pd.DataFrame(), None)
        except Exception:
            old_history, old_sha = pd.DataFrame(), None

        if not old_history.empty:
            fresh_start = min(required_start, min(old_history["date"]))
            sha = old_sha
        else:
            fresh_start = required_start

        if status_box:
            status_box.write("Erasing cached copy and reconfirming all saved data from Alpaca...")

        fresh = _fetch_range_if_valid(symbols, fresh_start, required_end, feed)
        history = _normalize_history(fresh)

        if GITHUB_TOKEN:
            sha = save_history_to_github(
                history,
                sha=sha,
                commit_message=f"Reconfirm market history through {required_end}",
            )

        st.session_state["persistent_history_df"] = history
        st.session_state["persistent_history_sha"] = sha

        if status_box:
            status_box.write(
                f"Fresh confirmation complete: {len(history):,} one-minute bars saved."
            )

        return history

    updates = []

    # Determine the currently stored range for requested symbols.
    requested_existing = (
        history[history["symbol"].isin(symbols)]
        if not history.empty else pd.DataFrame()
    )

    if requested_existing.empty:
        if status_box:
            status_box.write("No valid saved market history found. Creating a fresh source from Alpaca...")
        updates.append(
            _fetch_range_if_valid(symbols, required_start, required_end, feed)
        )
    else:
        existing_symbols = set(requested_existing["symbol"].unique())
        missing_symbols = [s for s in symbols if s not in existing_symbols]

        existing_min = min(requested_existing["date"])
        existing_max = max(requested_existing["date"])

        # Backfill earlier dates only if the requested analysis needs them.
        if required_start < existing_min:
            if status_box:
                status_box.write(
                    f"Updating new data: backfilling {required_start} through "
                    f"{existing_min - timedelta(days=1)}..."
                )
            updates.append(
                _fetch_range_if_valid(
                    symbols,
                    required_start,
                    existing_min - timedelta(days=1),
                    feed,
                )
            )

        # Add newer dates.
        if required_end > existing_max:
            if status_box:
                status_box.write(
                    f"Updating new data: adding sessions after {existing_max}..."
                )
            updates.append(
                _fetch_range_if_valid(
                    symbols,
                    existing_max + timedelta(days=1),
                    required_end,
                    feed,
                )
            )

        # If the universe has new symbols, backfill only those names.
        if missing_symbols:
            missing_start = min(required_start, existing_min)
            if status_box:
                status_box.write(
                    f"Updating new data: adding {len(missing_symbols)} new symbol(s)..."
                )
            updates.append(
                _fetch_range_if_valid(
                    missing_symbols,
                    missing_start,
                    required_end,
                    feed,
                )
            )

    updates = [u for u in updates if u is not None and not u.empty]

    if updates:
        if status_box:
            status_box.write("Merging new data into saved source...")

        for u in updates:
            history = _merge_history(history, u)

        if GITHUB_TOKEN:
            if status_box:
                status_box.write("Saving updated historical source to GitHub...")

            sha = save_history_to_github(
                history,
                sha=sha,
                commit_message=f"Update market history through {required_end}",
            )
        else:
            if status_box:
                status_box.write(
                    "Persistent GitHub saving is not configured; data will remain only "
                    "for this running Streamlit session."
                )

        st.session_state["persistent_history_df"] = history
        st.session_state["persistent_history_sha"] = sha
    else:
        if status_box:
            status_box.write("Existing saved data are current. No Alpaca download needed.")

    return _normalize_history(history)


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
        st.info(
            "In Streamlit: Manage app → Settings → Secrets. "
            "Add ALPACA_API_KEY and ALPACA_SECRET_KEY, then rerun."
        )
        st.stop()

    st.subheader("Historical Data Status")

    if not GITHUB_TOKEN:
        st.warning(
            "Persistent history is not yet enabled. Add GITHUB_TOKEN to Streamlit Secrets "
            "to preserve the historical source across app restarts and future logins."
        )

    data_status = st.status("Checking existing data...", expanded=True)

    now_ct = datetime.now(CT)
    main_required_start = (pd.Timestamp(now_ct.date()) - pd.Timedelta(days=75)).date()

    try:
        raw = ensure_persistent_history(
            symbols,
            feed,
            required_start=main_required_start,
            required_end=now_ct.date(),
            force_reconfirm=False,
            status_box=data_status,
        )
        data_status.update(label="Historical data ready", state="complete", expanded=False)
    except Exception as e:
        data_status.update(label="Historical data update failed", state="error")
        st.error(str(e))
        st.stop()

    d1, d2, d3 = st.columns([2, 2, 2])

    with d1:
        if not raw.empty:
            st.metric("Saved bars", f"{len(raw):,}")

    with d2:
        if not raw.empty:
            st.metric(
                "Saved date range",
                f"{min(raw['date'])} → {max(raw['date'])}"
            )

    with d3:
        reconfirm = st.button(
            "Erase & Reconfirm All Data",
            type="secondary",
            help=(
                "Fresh-downloads the entire saved history period from Alpaca and replaces "
                "the persistent source file."
            ),
            key="reconfirm_all_history",
        )

    if reconfirm:
        reconfirm_status = st.status(
            "Reconfirming historical source...", expanded=True
        )
        try:
            raw = ensure_persistent_history(
                symbols,
                feed,
                required_start=min(raw["date"]) if not raw.empty else main_required_start,
                required_end=now_ct.date(),
                force_reconfirm=True,
                status_box=reconfirm_status,
            )
            reconfirm_status.update(
                label="All historical data reconfirmed",
                state="complete",
                expanded=False,
            )
            st.success("Saved history was replaced with a fresh Alpaca download.")
        except Exception as e:
            reconfirm_status.update(label="Reconfirmation failed", state="error")
            st.error(str(e))
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
latest = latest.merge(
    top[["symbol", "rank", "avg_gain_pct"]].rename(
        columns={"rank": "historical_rank", "avg_gain_pct": "trailing_avg_gain_pct"}
    ),
    on="symbol",
    how="inner",
)
if not latest.empty:
    latest = latest.sort_values(["gain_pct", "trailing_avg_gain_pct"], ascending=[False, False]).reset_index(drop=True)
    latest["today_rank"] = np.arange(1, len(latest) + 1)

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
st.dataframe(view, width='stretch', hide_index=True)

st.caption("* Broad-scan MUT/TDV use 1-minute bars. Use the Precision Trade Check below for trade-level MUT/TDV on a selected stock/day.")

st.subheader(f"Daily ranking of the Historical Top 25 — {latest_date}")
st.caption(
    "Historical Rank is based on the trailing 30-session 8:30–10:30 average. "
    "Today Rank is based only on this subject day's actual 8:30–10:30 gain among these 25 stocks."
)
if not latest.empty:
    lv = latest[[
        "historical_rank", "today_rank", "symbol", "trailing_avg_gain_pct",
        "start_price", "end_price", "gain_pct", "best_1h_gain_pct",
        "best_1h_start", "best_1h_end", "ldv_pct", "tdv_pct", "mut_seconds", "volume"
    ]].copy()
    lv["best_1h_start"] = pd.to_datetime(lv["best_1h_start"]).dt.strftime("%H:%M")
    lv["best_1h_end"] = pd.to_datetime(lv["best_1h_end"]).dt.strftime("%H:%M")
    lv.columns = [
        "Historical Rank", "Today Rank", "Symbol", "Trailing Avg 8:30–10:30 Gain %",
        "8:30 Price", "10:30 Price", "Today 8:30–10:30 Gain %", "Best 1-Hour Gain %",
        "Best 1-Hour Start", "Best 1-Hour End", "LDV %", "TDV %", "MUT (sec)*", "Window Volume"
    ]
    st.dataframe(lv, width='stretch', hide_index=True)


st.subheader("Optimization summaries")
st.caption(
    "These optimizers are retrospective research tools. Trigger values are tested in 25% increments "
    "from 25% through 200%. They run only when you press the buttons so normal page loads stay fast."
)

opt_session_map = build_session_map(raw)
opt_available_dates = sorted(daily["date"].unique())
trigger_grid = list(range(25, 201, 25))

with st.expander("30-session Historical Top-10 optimizer", expanded=False):
    st.write(
        "Uses the Top 10 generated from the same trailing 30-session history and asks which S1/S2 "
        "profit/loss trigger pair would have produced the highest compounded return across that historical period. "
        "Because the same 30 sessions are used to identify the stocks and evaluate them, this is a hindsight benchmark, not a forward backtest."
    )
    if st.button("Run 30-session historical optimizer", key="run_hist_optimizer"):
        hist_dates = opt_available_dates[-TRAILING_DAYS:]
        hist_prog = st.progress(0)
        hist_prog_text = st.empty()

        def set_hist_progress(pct_value, message):
            pct_value = int(max(0, min(100, pct_value)))
            hist_prog.progress(pct_value / 100.0)
            hist_prog_text.markdown(f"**{pct_value}% — {message}**")

        set_hist_progress(5, "Building the 30-session historical selection...")
        hist_rank = build_history(
            opt_session_map, hist_dates, symbols,
            window_minutes=60,
            same_day_mode=False,
            subject_day=None,
        )
        if hist_rank.empty or len(hist_rank) < 10:
            st.warning("Not enough complete history to build a 10-stock historical optimizer set.")
        else:
            fixed_top10 = hist_rank.head(10).copy()
            fixed_map = {d: fixed_top10 for d in hist_dates}

            def h1_cb(done, total):
                set_hist_progress(10 + round(42 * done / max(1, total)), "Optimizing S1 trigger combinations...")

            h1 = optimize_triggers_for_subjects(
                opt_session_map, hist_dates, fixed_map,
                scenario="S1", trigger_values=trigger_grid, starting_capital=1000.0,
                progress_callback=h1_cb,
            )

            def h2_cb(done, total):
                set_hist_progress(52 + round(43 * done / max(1, total)), "Optimizing S2 trigger combinations...")

            h2 = optimize_triggers_for_subjects(
                opt_session_map, hist_dates, fixed_map,
                scenario="S2", trigger_values=trigger_grid, starting_capital=1000.0,
                progress_callback=h2_cb,
            )
            set_hist_progress(100, "Historical optimizer complete.")

            rows = []
            for name, r in [("S1", h1), ("S2", h2)]:
                if r:
                    rows.append({
                        "Scenario": name,
                        "Best Profit Trigger %": r["profit_trigger_pct"],
                        "Best Loss Trigger %": r["loss_trigger_pct"],
                        "Subject Days": r["subject_days"],
                        "Ending $ from $1,000": round(r["ending_value"], 2),
                        "Total Return %": round(r["total_return_pct"], 3),
                        "Win Rate %": round(r["win_rate_pct"], 2),
                        "Max Drawdown %": round(r["max_drawdown_pct"], 3),
                        "Best Day %": round(r["best_day_pct"], 3),
                        "Worst Day %": round(r["worst_day_pct"], 3),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

with st.expander(f"Daily optimizer — {latest_date}", expanded=False):
    st.write(
        "For the latest complete subject day, this creates four independent Top-10 selections from only the prior "
        "1, 5, 15, or 30 complete sessions, then shows the best same-day S1/S2 trigger outcome in hindsight."
    )
    if st.button("Run daily 1 / 5 / 15 / 30-day optimizer", key="run_daily_optimizer"):
        daily_prog = st.progress(0)
        daily_prog_text = st.empty()

        def set_daily_progress(pct_value, message):
            pct_value = int(max(0, min(100, pct_value)))
            daily_prog.progress(pct_value / 100.0)
            daily_prog_text.markdown(f"**{pct_value}% — {message}**")

        set_daily_progress(5, "Building 1 / 5 / 15 / 30-day selections...")
        selections = prepare_daily_optimizer_selections(
            opt_session_map, opt_available_dates, symbols,
            latest_date, [1, 5, 15, 30], window_minutes=60,
        )
        rows = []
        tasks = [(lb, scenario) for lb in [1, 5, 15, 30] for scenario in ["S1", "S2"]]
        task_index = 0

        for lb in [1, 5, 15, 30]:
            top10_lb = selections.get(lb)
            if top10_lb is None or top10_lb.empty:
                rows.append({"Lookback": f"{lb} day", "Scenario": "N/A", "Status": "Insufficient prior complete data"})
                task_index += 2
                set_daily_progress(5 + round(90 * task_index / len(tasks)), f"Completed lookback {lb}...")
                continue

            day_map = {latest_date: top10_lb}
            for scenario in ["S1", "S2"]:
                base_task = task_index

                def daily_cb(done, total, base_task=base_task, lb=lb, scenario=scenario):
                    fractional = (base_task + done / max(1, total)) / len(tasks)
                    set_daily_progress(
                        5 + round(90 * fractional),
                        f"Optimizing {scenario} with {lb}-day lookback..."
                    )

                r = optimize_triggers_for_subjects(
                    opt_session_map, [latest_date], day_map,
                    scenario=scenario, trigger_values=trigger_grid, starting_capital=1000.0,
                    progress_callback=daily_cb,
                )
                if r:
                    rows.append({
                        "Lookback": f"{lb} day" if lb == 1 else f"{lb} days",
                        "Scenario": scenario,
                        "Best Profit Trigger %": r["profit_trigger_pct"],
                        "Best Loss Trigger %": r["loss_trigger_pct"],
                        "Day Return %": round(r["total_return_pct"], 3),
                        "Ending $ from $1,000": round(r["ending_value"], 2),
                        "Top 10": ", ".join(top10_lb["symbol"].tolist()),
                        "Status": "Retrospective ideal for this completed day",
                    })
                task_index += 1

        set_daily_progress(100, "Daily optimizer complete.")
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

st.subheader("Intraday chart")
symbol = st.selectbox("Stock", top["symbol"].tolist())
date_choices = sorted(daily[daily["symbol"]==symbol]["date"].unique(), reverse=True)
sel_date = st.selectbox("Session", date_choices)
chart = windowed[(windowed["symbol"]==symbol) & (windowed["date"]==sel_date)]
fig = px.line(chart, x="timestamp", y="close", title=f"{symbol} — {sel_date} — 8:30–10:30 CT")
st.plotly_chart(fig, width='stretch')

st.subheader("Trailing 30-session performance")
td = daily[daily["symbol"]==symbol].sort_values("date")
fig2 = px.bar(td, x="date", y="gain_pct", title=f"{symbol} daily 8:30–10:30 gain (%)")
st.plotly_chart(fig2, width='stretch')

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
st.header("Custom Strategy Backtest")
st.caption(
    "Adjust the settings freely. The app will not regenerate the backtest until you click "
    "Generate Backtest. Processing then shows an estimated completion percentage based on the work completed."
)

with st.expander("How S1, S2, and S3 work", expanded=False):
    st.markdown(
        """
### S1 — Trigger Scenario
Each subject day independently selects the Top 10 from that day's own prior lookback sessions.
Capital is divided equally across those ten stocks. Profit and loss triggers are based on the selected
percentages of each stock's historical median best-window gain and median LDV.

### S2 — Round Robin
Uses the same independently generated Top 10, but cycles one pool of capital sequentially through the names.
The ending S2 balance carries to the next subject day.

### S3 — LDV Scenario
Each subject day independently ranks the universe from its own prior **1–30 completed sessions** and selects
an adjustable **Top 1 through Top 25**. Every selected stock is then analyzed separately.

For each stock, S3 automatically searches the 8:30–10:30 AM Central window on a **5-minute entry/exit grid**
with a **minimum 20-minute holding period**. It chooses the entry/exit pair with the highest average return
across the lookback. The stock's LDV is calculated only from that selected entry through exit.

- Gain multiplier: **50%, 75%, 100%, 125%, or 150%** of that stock's own historical average gain.
- LDV multiplier: **50%, 75%, 100%, 125%, or 150%** of that stock's own historical average trade-window LDV.
- If neither target nor stop is reached, the stock exits at its automatically selected historical exit time.
- S3 capital is split equally across the selected Top N, and the ending balance compounds into the next subject day.

### Lookback = 0 for S1/S2
Lookback 0 remains a same-day diagnostic with look-ahead bias. S3 always uses at least one prior completed session.
        """
    )

st.subheader("Backtest parameters")
st.info(
    "Settings inside this box are staged. Moving a slider or changing a date does not run the report. "
    "Click Generate Backtest when the setup is ready."
)

with st.form("custom_backtest_form_v9", clear_on_submit=False):
    st.markdown("#### Core settings")
    p1, p2, p3 = st.columns(3)
    with p1:
        lookback_sessions = st.slider(
            "S1/S2 historical lookback sessions",
            min_value=0, max_value=30, value=15, step=1,
            help="0 = same-day diagnostic (look-ahead). 1–30 = prior completed sessions only."
        )
    with p2:
        best_window_minutes = st.slider(
            "S1/S2 best-window length (minutes)",
            min_value=15, max_value=120, value=60, step=5,
        )
    with p3:
        starting_capital = st.number_input(
            "Starting capital ($)",
            min_value=1.0, value=1000.0, step=100.0, format="%.2f",
            help="Each scenario begins with this amount and compounds independently across subject days."
        )

    st.markdown("#### S1 — Trigger Scenario")
    s1c1, s1c2 = st.columns(2)
    with s1c1:
        s1_profit_pct = st.slider(
            "S1 profit trigger (% of median best-window gain)",
            min_value=0, max_value=200, value=90, step=5,
        )
    with s1c2:
        s1_loss_pct = st.slider(
            "S1 loss trigger (% of median LDV)",
            min_value=0, max_value=200, value=80, step=5,
        )

    st.markdown("#### S2 — Round Robin")
    s2c1, s2c2 = st.columns(2)
    with s2c1:
        s2_profit_pct = st.slider(
            "S2 profit trigger (% of median best-window gain)",
            min_value=0, max_value=200, value=90, step=5,
        )
    with s2c2:
        s2_loss_pct = st.slider(
            "S2 loss trigger (% of median LDV)",
            min_value=0, max_value=200, value=80, step=5,
        )

    st.markdown("#### S3 — LDV Scenario")
    s3c1, s3c2, s3c3, s3c4 = st.columns(4)
    with s3c1:
        s3_top_n = st.slider(
            "S3 Top performers",
            min_value=1, max_value=25, value=10, step=1,
            help="Number of highest average 8:30–10:30 performers selected independently for each subject day."
        )
    with s3c2:
        s3_lookback_sessions = st.slider(
            "S3 lookback sessions",
            min_value=1, max_value=30, value=5, step=1,
        )
    with s3c3:
        s3_gain_multiplier_pct = st.slider(
            "S3 Gain % multiplier",
            min_value=50, max_value=150, value=100, step=25,
            help="Percent of each stock's own average gain used as its profit target."
        )
    with s3c4:
        s3_ldv_multiplier_pct = st.slider(
            "S3 LDV % multiplier",
            min_value=50, max_value=150, value=100, step=25,
            help="Percent of each stock's own average trade-window LDV used as its stop."
        )

    st.caption(
        "S3 timing is automatic: entry/exit candidates are tested every 5 minutes from 8:30–10:30 CT, "
        "with a minimum 20-minute holding period."
    )

    st.markdown("#### Subject dates")
    dc1, dc2 = st.columns(2)
    with dc1:
        backtest_start = st.date_input(
            "First subject day",
            value=pd.Timestamp("2026-08-11").date(),
            key="backtest_subject_start_v9",
        )
    with dc2:
        backtest_end = st.date_input(
            "Last subject day",
            value=pd.Timestamp("2026-08-25").date(),
            key="backtest_subject_end_v9",
        )

    generate_backtest = st.form_submit_button(
        "Generate Backtest",
        type="primary",
        width='stretch',
    )


def _safe_apply_portfolio_return(balance, ret):
    if ret is None or pd.isna(ret):
        return float(balance), 0.0
    gain_loss = float(balance) * float(ret) / 100.0
    return float(balance) + gain_loss, gain_loss


def render_custom_backtest_v9(package):
    results = package["results"]
    summary_rows = package["summary_rows"]
    metadata = package["metadata"]

    st.success(
        f"Backtest complete — {len(results)} subject-date rows generated. "
        f"Generated {package['generated_at']}."
    )

    stats = package["stats"]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("S1 ending value", f"${stats['S1']['ending_value']:,.2f}", f"{stats['S1']['total_return_pct']:.2f}%")
    mc2.metric("S2 ending value", f"${stats['S2']['ending_value']:,.2f}", f"{stats['S2']['total_return_pct']:.2f}%")
    mc3.metric("S3 ending value", f"${stats['S3']['ending_value']:,.2f}", f"{stats['S3']['total_return_pct']:.2f}%")

    mc4, mc5, mc6 = st.columns(3)
    mc4.metric("S1 max drawdown", f"{stats['S1']['max_drawdown_pct']:.2f}%")
    mc5.metric("S2 max drawdown", f"{stats['S2']['max_drawdown_pct']:.2f}%")
    mc6.metric("S3 max drawdown", f"{stats['S3']['max_drawdown_pct']:.2f}%")

    st.subheader("Portfolio summary")
    st.dataframe(summary_rows, width='stretch', hide_index=True)

    st.subheader("Subject-day portfolio results")
    st.dataframe(results, width='stretch', hide_index=True)

    st.download_button(
        "Download daily results CSV",
        csv_with_metadata(results, metadata),
        "custom_backtest_daily_v9.csv",
        "text/csv",
        key="download_custom_daily_v9",
        on_click="ignore",
    )

    rankings = package.get("rankings")
    if rankings is not None and not rankings.empty:
        with st.expander("S1/S2 Top 10 selected for each subject day"):
            st.dataframe(rankings, width='stretch', hide_index=True)
            st.download_button(
                "Download S1/S2 rankings CSV",
                csv_with_metadata(rankings, metadata),
                "custom_backtest_s1_s2_rankings_v9.csv",
                "text/csv",
                key="download_custom_rankings_v9",
                on_click="ignore",
            )

    s3_rankings = package.get("s3_rankings")
    if s3_rankings is not None and not s3_rankings.empty:
        with st.expander("S3 independently selected stocks and automatic windows"):
            st.dataframe(s3_rankings, width='stretch', hide_index=True)
            st.download_button(
                "Download S3 selections CSV",
                csv_with_metadata(s3_rankings, metadata),
                "custom_backtest_s3_selections_v9.csv",
                "text/csv",
                key="download_custom_s3_rankings_v9",
                on_click="ignore",
            )

    for scenario, key_name, filename in [
        ("S1", "s1_trades", "custom_S1_trade_log_v9.csv"),
        ("S2", "s2_trades", "custom_S2_trade_log_v9.csv"),
        ("S3", "s3_trades", "custom_S3_trade_log_v9.csv"),
    ]:
        tdf = package.get(key_name)
        if tdf is not None and not tdf.empty:
            with st.expander(f"{scenario} trade log"):
                st.dataframe(tdf, width='stretch', hide_index=True)
                st.download_button(
                    f"Download {scenario} trade log CSV",
                    csv_with_metadata(tdf, metadata),
                    filename,
                    "text/csv",
                    key=f"download_{scenario.lower()}_trade_log_v9",
                    on_click="ignore",
                )


if generate_backtest:
    st.session_state.pop("custom_backtest_package_v9", None)

    if not ALPACA_KEY or not ALPACA_SECRET:
        st.error("Alpaca credentials are missing from Streamlit Secrets.")
    elif backtest_end < backtest_start:
        st.error("The last subject day must be on or after the first subject day.")
    else:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def set_backtest_progress(pct_value, message):
            pct_value = int(max(0, min(100, pct_value)))
            progress_bar.progress(pct_value / 100.0)
            progress_text.markdown(f"**{pct_value}% — {message}**")

        set_backtest_progress(2, "Checking saved market history...")
        max_lookback = max(int(lookback_sessions), int(s3_lookback_sessions))
        extra_days = max(10, max_lookback * 2 + 12)
        calendar_start = pd.Timestamp(backtest_start) - pd.Timedelta(days=extra_days)
        calendar_end = pd.Timestamp(backtest_end)

        load_status = st.status("Checking existing data...", expanded=True)
        backtest_failed = False

        try:
            all_bt_df = ensure_persistent_history(
                symbols,
                feed,
                required_start=calendar_start.date(),
                required_end=calendar_end.date(),
                force_reconfirm=False,
                status_box=load_status,
            )
        except Exception as e:
            backtest_failed = True
            load_status.update(label="Historical data check failed", state="error")
            st.error(str(e))

        if not backtest_failed:
            set_backtest_progress(12, "Saved market history ready; preparing working data...")
            all_bt_df = all_bt_df[
                (all_bt_df["date"] >= calendar_start.date()) &
                (all_bt_df["date"] <= calendar_end.date()) &
                (all_bt_df["symbol"].isin(symbols))
            ].copy()

            if all_bt_df.empty:
                backtest_failed = True
                load_status.update(label="No usable historical data", state="error")
                st.error("No saved or Alpaca market data were available for this period.")
            else:
                load_status.write(f"Using {len(all_bt_df):,} saved one-minute bars.")
                set_backtest_progress(17, "Indexing one-minute sessions...")
                session_map = build_session_map(all_bt_df)
                available_dates = sorted(all_bt_df["date"].unique())
                subject_dates = [d for d in available_dates if backtest_start <= d <= backtest_end]
                load_status.update(label="Saved historical data ready", state="complete", expanded=False)

                if not subject_dates:
                    backtest_failed = True
                    st.error("No complete subject trading dates were found inside the requested date range.")

        if not backtest_failed:
            daily_results = []
            all_s1_trades = []
            all_s2_trades = []
            all_s3_trades = []
            rankings = []
            s3_rankings = []

            s1_balance = float(starting_capital)
            s2_balance = float(starting_capital)
            s3_balance = float(starting_capital)

            n_subjects = max(1, len(subject_dates))
            set_backtest_progress(20, f"Beginning {len(subject_dates)} subject-day calculations...")

            for subject_index, subject_day in enumerate(subject_dates):
                day_base = 20 + 74 * subject_index / n_subjects
                day_span = 74 / n_subjects

                def day_progress(frac, message):
                    set_backtest_progress(round(day_base + day_span * frac), message)

                day_progress(0.05, f"{subject_day}: building S1/S2 historical ranking...")

                # S1/S2 selection for this subject day.
                same_day_mode = lookback_sessions == 0
                top10 = pd.DataFrame()
                s12_trailing_start = None
                s12_trailing_end = None

                if same_day_mode:
                    hist = build_history(
                        session_map, [], symbols,
                        window_minutes=best_window_minutes,
                        same_day_mode=True,
                        subject_day=subject_day,
                    )
                    s12_trailing_start = subject_day
                    s12_trailing_end = subject_day
                else:
                    previous_dates = [x for x in available_dates if x < subject_day]
                    if len(previous_dates) >= lookback_sessions:
                        trailing_dates = previous_dates[-lookback_sessions:]
                        hist = build_history(
                            session_map, trailing_dates, symbols,
                            window_minutes=best_window_minutes,
                            same_day_mode=False,
                            subject_day=None,
                        )
                        s12_trailing_start = trailing_dates[0]
                        s12_trailing_end = trailing_dates[-1]
                    else:
                        hist = pd.DataFrame()

                if not hist.empty and len(hist) >= 10:
                    top10 = hist.head(10).copy()
                    top10["subject_date"] = subject_day
                    top10["best_start_ct"] = top10["best_start_min"].map(time_label)
                    top10["best_end_ct"] = top10["best_end_min"].map(time_label)
                    top10["lookback_sessions"] = lookback_sessions
                    top10["window_minutes"] = best_window_minutes
                    rankings.append(top10)

                # S3 selection is always based strictly on prior completed sessions.
                day_progress(0.30, f"{subject_day}: finding S3 Top {s3_top_n} and automatic entry/exit windows...")
                previous_dates_s3 = [x for x in available_dates if x < subject_day]
                s3_selection = pd.DataFrame()
                s3_trailing_start = None
                s3_trailing_end = None

                if len(previous_dates_s3) >= s3_lookback_sessions:
                    s3_dates = previous_dates_s3[-s3_lookback_sessions:]
                    s3_selection = build_s3_history(
                        session_map,
                        s3_dates,
                        symbols,
                        top_n=s3_top_n,
                        entry_step_minutes=5,
                        minimum_hold_minutes=20,
                    )
                    s3_trailing_start = s3_dates[0]
                    s3_trailing_end = s3_dates[-1]

                    if not s3_selection.empty:
                        s3_selection = s3_selection.copy()
                        s3_selection["subject_date"] = subject_day
                        s3_selection["auto_entry_ct"] = s3_selection["s3_entry_min"].map(time_label)
                        s3_selection["auto_exit_ct"] = s3_selection["s3_exit_min"].map(time_label)
                        s3_selection["lookback_sessions"] = s3_lookback_sessions
                        s3_selection["gain_multiplier_pct"] = s3_gain_multiplier_pct
                        s3_selection["ldv_multiplier_pct"] = s3_ldv_multiplier_pct
                        s3_rankings.append(s3_selection)

                if top10.empty and s3_selection.empty:
                    day_progress(1.0, f"{subject_day}: insufficient prior data; skipped.")
                    continue

                subject_frames = [
                    session_map[(sym, subject_day)]
                    for sym in symbols
                    if (sym, subject_day) in session_map
                ]
                subject_df = pd.concat(subject_frames, ignore_index=True) if subject_frames else pd.DataFrame()

                # S1
                day_progress(0.58, f"{subject_day}: running S1...")
                if not top10.empty:
                    s1_ret, s1_trades = run_s1(
                        subject_df, top10,
                        profit_trigger_pct=s1_profit_pct,
                        loss_trigger_pct=s1_loss_pct,
                    )
                else:
                    s1_ret, s1_trades = np.nan, pd.DataFrame()

                # S2
                day_progress(0.70, f"{subject_day}: running S2...")
                if not top10.empty:
                    s2_ret, s2_trades = run_s2(
                        subject_df, top10,
                        profit_trigger_pct=s2_profit_pct,
                        loss_trigger_pct=s2_loss_pct,
                    )
                else:
                    s2_ret, s2_trades = np.nan, pd.DataFrame()

                # S3
                day_progress(0.82, f"{subject_day}: running S3 individual-stock LDV strategy...")
                if not s3_selection.empty:
                    s3_ret, s3_trades = run_s3(
                        subject_df,
                        s3_selection,
                        gain_multiplier_pct=s3_gain_multiplier_pct,
                        ldv_multiplier_pct=s3_ldv_multiplier_pct,
                    )
                else:
                    s3_ret, s3_trades = np.nan, pd.DataFrame()

                s1_start = s1_balance
                s2_start = s2_balance
                s3_start = s3_balance
                s1_balance, s1_gain_loss = _safe_apply_portfolio_return(s1_balance, s1_ret)
                s2_balance, s2_gain_loss = _safe_apply_portfolio_return(s2_balance, s2_ret)
                s3_balance, s3_gain_loss = _safe_apply_portfolio_return(s3_balance, s3_ret)

                if not s1_trades.empty:
                    s1_trades["subject_date"] = subject_day
                    s1_trades["subject_starting_capital"] = s1_start
                    s1_trades["allocated_dollars"] = s1_start / max(1, len(top10))
                    s1_trades["trade_dollar_gain_loss"] = s1_trades["allocated_dollars"] * s1_trades["return_pct"] / 100.0
                    all_s1_trades.append(s1_trades)

                if not s2_trades.empty:
                    s2_trades["subject_date"] = subject_day
                    s2_trades["subject_starting_capital"] = s2_start
                    all_s2_trades.append(s2_trades)

                if not s3_trades.empty:
                    s3_trades["subject_date"] = subject_day
                    s3_trades["subject_starting_capital"] = s3_start
                    s3_trades["allocated_dollars"] = s3_start / max(1, len(s3_selection))
                    s3_trades["trade_dollar_gain_loss"] = s3_trades["allocated_dollars"] * s3_trades["return_pct"] / 100.0
                    all_s3_trades.append(s3_trades)

                daily_results.append({
                    "date": subject_day,
                    "S1_S2_lookback_sessions": lookback_sessions,
                    "S1_S2_window_minutes": best_window_minutes,
                    "S1_S2_trailing_start": s12_trailing_start,
                    "S1_S2_trailing_end": s12_trailing_end,
                    "S3_lookback_sessions": s3_lookback_sessions,
                    "S3_top_n": s3_top_n,
                    "S3_trailing_start": s3_trailing_start,
                    "S3_trailing_end": s3_trailing_end,
                    "S3_gain_multiplier_pct": s3_gain_multiplier_pct,
                    "S3_ldv_multiplier_pct": s3_ldv_multiplier_pct,
                    "S1_starting_capital": s1_start,
                    "S1_return_pct": s1_ret,
                    "S1_dollar_gain_loss": s1_gain_loss,
                    "S1_ending_capital": s1_balance,
                    "S2_starting_capital": s2_start,
                    "S2_return_pct": s2_ret,
                    "S2_dollar_gain_loss": s2_gain_loss,
                    "S2_ending_capital": s2_balance,
                    "S3_starting_capital": s3_start,
                    "S3_return_pct": s3_ret,
                    "S3_dollar_gain_loss": s3_gain_loss,
                    "S3_ending_capital": s3_balance,
                    "S1_trades": len(s1_trades),
                    "S2_trades": len(s2_trades),
                    "S3_trades": len(s3_trades),
                    "S1_S2_top10": ", ".join(top10["symbol"].tolist()) if not top10.empty else "",
                    "S3_selected": ", ".join(s3_selection["symbol"].tolist()) if not s3_selection.empty else "",
                })

                day_progress(1.0, f"{subject_day}: complete.")

            set_backtest_progress(95, "Building portfolio summaries and export files...")
            results = pd.DataFrame(daily_results)

            if results.empty:
                st.error(
                    "No subject-day results could be calculated. Try smaller lookbacks, a larger stock universe, "
                    "or confirm the selected feed has coverage."
                )
            else:
                s1_stats = portfolio_stats(results["S1_return_pct"], starting_capital)
                s2_stats = portfolio_stats(results["S2_return_pct"], starting_capital)
                s3_stats = portfolio_stats(results["S3_return_pct"], starting_capital)

                summary_rows = pd.DataFrame([
                    {
                        "Scenario": name,
                        "Starting Capital": starting_capital,
                        "Ending Capital": stats["ending_value"],
                        "Dollar Gain/Loss": stats["dollar_gain_loss"],
                        "Total Return %": stats["total_return_pct"],
                        "Average Daily Return %": stats["avg_daily_return_pct"],
                        "Winning Days %": stats["win_rate_pct"],
                        "Max Drawdown %": stats["max_drawdown_pct"],
                        "Best Day %": stats["best_day_pct"],
                        "Worst Day %": stats["worst_day_pct"],
                    }
                    for name, stats in [("S1", s1_stats), ("S2", s2_stats), ("S3", s3_stats)]
                ])

                metadata = {
                    "S1/S2 historical lookback sessions": lookback_sessions,
                    "S1/S2 best-window length minutes": best_window_minutes,
                    "S1 profit trigger % of median best-window gain": s1_profit_pct,
                    "S1 loss trigger % of median LDV": s1_loss_pct,
                    "S2 profit trigger % of median best-window gain": s2_profit_pct,
                    "S2 loss trigger % of median LDV": s2_loss_pct,
                    "S3 Top performers": s3_top_n,
                    "S3 historical lookback sessions": s3_lookback_sessions,
                    "S3 gain multiplier % of each stock's own average gain": s3_gain_multiplier_pct,
                    "S3 LDV multiplier % of each stock's own average trade-window LDV": s3_ldv_multiplier_pct,
                    "S3 entry/exit grid minutes": 5,
                    "S3 minimum holding period minutes": 20,
                    "S3 timing rule": "Each selected stock independently uses the prior-lookback entry/exit pair with the highest average return",
                    "Starting capital for each scenario": f"${starting_capital:,.2f}",
                    "Requested first subject date": backtest_start,
                    "Requested last subject date": backtest_end,
                    "Subject trading dates in requested range": len(subject_dates),
                    "Calculated subject-date rows": len(results),
                    "Selection rule": "Each subject day independently recalculates its selections using only the applicable lookback sessions",
                }

                package = {
                    "generated_at": datetime.now(CT).strftime("%Y-%m-%d %I:%M:%S %p CT"),
                    "results": results,
                    "summary_rows": summary_rows,
                    "stats": {"S1": s1_stats, "S2": s2_stats, "S3": s3_stats},
                    "metadata": metadata,
                    "rankings": pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame(),
                    "s3_rankings": pd.concat(s3_rankings, ignore_index=True) if s3_rankings else pd.DataFrame(),
                    "s1_trades": pd.concat(all_s1_trades, ignore_index=True) if all_s1_trades else pd.DataFrame(),
                    "s2_trades": pd.concat(all_s2_trades, ignore_index=True) if all_s2_trades else pd.DataFrame(),
                    "s3_trades": pd.concat(all_s3_trades, ignore_index=True) if all_s3_trades else pd.DataFrame(),
                }
                st.session_state["custom_backtest_package_v9"] = package
                set_backtest_progress(100, "Backtest complete.")

package = st.session_state.get("custom_backtest_package_v9")
if package:
    render_custom_backtest_v9(package)

st.caption(
    "Backtest estimates exclude commissions, bid/ask spread, slippage, taxes and market impact. "
    "S1/S2 lookback=0 contains look-ahead bias and is diagnostic only. S3 always uses prior completed sessions. "
    "Optimizer summaries are retrospective/hindsight benchmarks."
)
