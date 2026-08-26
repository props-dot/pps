
import os
import time as time_module
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

CT = ZoneInfo("America/Chicago")
ET = ZoneInfo("America/New_York")
UTC = timezone.utc

START_CT = time(8, 30)
END_CT = time(10, 30)

DEFAULT_SYMBOLS = """
AAPL MSFT NVDA AMZN META GOOGL GOOG AVGO TSLA BRK.B LLY JPM V MA
COST WMT NFLX ORCL HD PG JNJ ABBV BAC KO CRM AMD MRK CVX XOM
TMUS CSCO IBM GE CAT ADBE QCOM TXN INTU NOW AMAT PANW BKNG UBER
AXP GS MS PEP MCD DIS NKE LOW TMO UNH AMGN ISRG PLTR MU
""".split()

st.set_page_config(page_title="Rolling 15-Day Strategy Backtest", layout="wide")
st.title("One-Hour Stock Strategy — Rolling 15-Day Backtest")
st.caption("Subject days begin Aug. 11, 2026. Each subject day uses only the prior 15 completed sessions.")

def get_secret(name):
    try:
        return st.secrets.get(name, None)
    except Exception:
        return os.getenv(name)

KEY = get_secret("ALPACA_API_KEY")
SECRET = get_secret("ALPACA_SECRET_KEY")

if not KEY or not SECRET:
    st.error("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in Streamlit Secrets.")
    st.stop()

HEADERS = {
    "APCA-API-KEY-ID": KEY,
    "APCA-API-SECRET-KEY": SECRET,
}

def norm_symbol(s):
    return s.strip().upper()

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
            headers=HEADERS,
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

st.sidebar.header("Backtest settings")
feed = st.sidebar.selectbox("Feed", ["iex", "sip"], index=0)
symbols_text = st.sidebar.text_area("Universe", " ".join(DEFAULT_SYMBOLS), height=180)
symbols = list(dict.fromkeys(norm_symbol(x) for x in symbols_text.replace(","," ").split() if x.strip()))

subject_start = st.sidebar.date_input("First subject day", pd.Timestamp("2026-08-11").date())
subject_end = st.sidebar.date_input("Last subject day", pd.Timestamp("2026-08-25").date())

st.info(
    "Backtest assumptions: prior 15 completed sessions only; Top 10 ranked by average 8:30–10:30 gain; "
    "best hour chosen from every 1-minute start between 8:30 and 9:30 CT; LDV is the 15-day average LDV. "
    "S1 is equal-weighted. S2 compounds one 100%-capital trade at a time."
)

if st.button("Run rolling 15-day backtest", type="primary"):
    # Pull enough prior business days. We fetch individual 2-hour windows to keep data volume low.
    calendar_start = pd.Timestamp(subject_start) - pd.Timedelta(days=35)
    calendar_end = pd.Timestamp(subject_end)
    candidate_dates = [d.date() for d in pd.bdate_range(calendar_start, calendar_end)]

    progress = st.progress(0)
    status = st.empty()
    frames = []

    for i, d in enumerate(candidate_dates):
        status.write(f"Loading {d} ({i+1}/{len(candidate_dates)})...")
        try:
            x = fetch_session(tuple(symbols), str(d), feed)
        except Exception as e:
            st.error(f"{d}: {e}")
            st.stop()
        if not x.empty:
            frames.append(x)
        progress.progress((i+1)/len(candidate_dates))

    if not frames:
        st.error("No market data returned.")
        st.stop()

    all_df = pd.concat(frames, ignore_index=True)
    available_dates = sorted(all_df["date"].unique())

    subject_dates = [
        d for d in available_dates
        if subject_start <= d <= subject_end
    ]

    daily_results = []
    all_s1_trades = []
    all_s2_trades = []
    rankings = []

    for d in subject_dates:
        prev = [x for x in available_dates if x < d]
        if len(prev) < 15:
            continue
        trailing = prev[-15:]
        hist = build_history(all_df, trailing, symbols)
        if hist.empty or len(hist) < 10:
            continue
        top10 = hist.head(10).copy()
        top10["subject_date"] = d
        top10["best_start_ct"] = top10["best_start_min"].map(time_label)
        top10["best_end_ct"] = top10["best_end_min"].map(time_label)
        rankings.append(top10)

        subject_df = all_df[all_df["date"] == d]
        s1_ret, s1_trades = run_s1(subject_df, top10)
        s2_ret, s2_trades = run_s2(subject_df, top10)

        if not s1_trades.empty:
            s1_trades["subject_date"] = d
            all_s1_trades.append(s1_trades)
        if not s2_trades.empty:
            s2_trades["subject_date"] = d
            all_s2_trades.append(s2_trades)

        daily_results.append({
            "date": d,
            "trailing_start": trailing[0],
            "trailing_end": trailing[-1],
            "S1_return_pct": s1_ret,
            "S2_return_pct": s2_ret,
            "S1_trades": len(s1_trades),
            "S2_trades": len(s2_trades),
            "top10": ", ".join(top10["symbol"].tolist()),
        })

    results = pd.DataFrame(daily_results)
    if results.empty:
        st.error("No subject-day results could be calculated.")
        st.stop()

    results["S1_equity_100"] = 100 * (1 + results["S1_return_pct"]/100).cumprod()
    results["S2_equity_100"] = 100 * (1 + results["S2_return_pct"]/100).cumprod()

    st.success("Backtest complete.")
    st.subheader("Daily returns")
    st.dataframe(results, use_container_width=True, hide_index=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("S1 cumulative", f"{(results['S1_equity_100'].iloc[-1]/100-1)*100:.2f}%")
    c2.metric("S2 cumulative", f"{(results['S2_equity_100'].iloc[-1]/100-1)*100:.2f}%")
    c3.metric("S1 avg/day", f"{results['S1_return_pct'].mean():.3f}%")
    c4.metric("S2 avg/day", f"{results['S2_return_pct'].mean():.3f}%")

    st.download_button(
        "Download daily results CSV",
        results.to_csv(index=False).encode(),
        "rolling_15_day_backtest_daily.csv",
        "text/csv"
    )

    if rankings:
        rank_df = pd.concat(rankings, ignore_index=True)
        st.subheader("Top 10 used for each subject day")
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download rankings CSV",
            rank_df.to_csv(index=False).encode(),
            "rolling_15_day_rankings.csv",
            "text/csv"
        )

    if all_s1_trades:
        s1df = pd.concat(all_s1_trades, ignore_index=True)
        st.subheader("S1 trade log")
        st.dataframe(s1df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download S1 trades CSV",
            s1df.to_csv(index=False).encode(),
            "S1_trade_log.csv",
            "text/csv"
        )

    if all_s2_trades:
        s2df = pd.concat(all_s2_trades, ignore_index=True)
        st.subheader("S2 trade log")
        st.dataframe(s2df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download S2 trades CSV",
            s2df.to_csv(index=False).encode(),
            "S2_trade_log.csv",
            "text/csv"
        )
