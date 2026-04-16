"""
fetchers.py — Data fetching layer
Primary source: yfinance (price history + fundamentals, free, no key needed)
Supplemental:  Massive API (real-time snapshots, short interest)
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

# Locate .env one level up from screener/
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from universe import COMBINED


# ── yfinance ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(period: str = "6mo") -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV history for all tickers via yfinance.
    Returns dict: ticker → DataFrame(Open, High, Low, Close, Volume).
    Cached 1 hour — daily bars don't change intraday.
    """
    results: Dict[str, pd.DataFrame] = {}

    def _fetch(ticker: str) -> tuple:
        try:
            t = yf.Ticker(ticker)
            h = t.history(period=period, interval="1d", auto_adjust=True)
            if h.empty:
                return ticker, None
            return ticker, h[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=20) as exe:
        futures = {exe.submit(_fetch, t): t for t in COMBINED}
        for f in as_completed(futures):
            ticker, df = f.result()
            if df is not None and len(df) >= 20:
                results[ticker] = df

    return results


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fundamentals() -> pd.DataFrame:
    """
    Fetch fundamental data for all tickers via yfinance.
    Cached 6 hours — fundamentals are point-in-time, not intraday.
    Returns DataFrame indexed by ticker.
    """
    FIELDS = [
        "shortName", "sector", "industry",
        "marketCap", "beta",
        "trailingPE", "forwardPE", "priceToBook",
        "trailingEps", "forwardEps",
        "revenueGrowth", "earningsGrowth",
        "profitMargins", "returnOnEquity", "debtToEquity",
        "recommendationMean", "targetMeanPrice",
        "shortPercentOfFloat",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "regularMarketPrice", "averageVolume",
        "dividendYield",
    ]

    rows = {}

    def _fetch(ticker: str) -> tuple:
        try:
            info = yf.Ticker(ticker).info
            return ticker, {f: info.get(f) for f in FIELDS}
        except Exception:
            return ticker, {f: None for f in FIELDS}

    with ThreadPoolExecutor(max_workers=20) as exe:
        futures = {exe.submit(_fetch, t): t for t in COMBINED}
        for f in as_completed(futures):
            ticker, row = f.result()
            rows[ticker] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    return df


# ── Massive API ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime_snapshots() -> pd.DataFrame:
    """
    Fetch real-time (15-min delayed) snapshots from Massive API.
    Supplements yfinance with current-session price/volume.
    Cached 5 minutes.
    """
    try:
        from massive import RESTClient
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            return pd.DataFrame()
        client = RESTClient(api_key=api_key)
        universe_set = set(COMBINED)
        snaps = client.get_snapshot_all("stocks")
        rows = []
        for s in snaps:
            sym = getattr(s, "ticker", None)
            if sym not in universe_set:
                continue
            d = s.day
            p = s.prev_day
            rows.append({
                "ticker":       sym,
                "rt_price":     getattr(d, "close", None) or getattr(d, "vwap", None),
                "rt_volume":    getattr(d, "volume", None),
                "rt_vwap":      getattr(d, "vwap", None),
                "rt_chg_pct":   getattr(s, "todays_change_percent", None),
                "rt_prev_close":getattr(p, "close", None) if p else None,
            })
        df = pd.DataFrame(rows).set_index("ticker")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_short_interest_bulk() -> pd.DataFrame:
    """
    Fetch most recent short interest reading from Massive API for each ticker.
    Cached 24 hours — short interest reports are bi-weekly.
    """
    try:
        from massive import RESTClient
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            return pd.DataFrame()
        client = RESTClient(api_key=api_key)
        rows = {}

        def _fetch(ticker: str) -> tuple:
            try:
                records = list(client.list_short_interest(
                    ticker=ticker, limit=1,
                    params={"order": "desc"}
                ))
                if records:
                    r = records[0]
                    return ticker, {
                        "short_interest":   getattr(r, "short_interest", None),
                        "days_to_cover":    getattr(r, "days_to_cover", None),
                        "avg_daily_vol_si": getattr(r, "avg_daily_volume", None),
                        "si_date":          getattr(r, "settlement_date", None),
                    }
            except Exception:
                pass
            return ticker, {}

        with ThreadPoolExecutor(max_workers=10) as exe:
            futures = {exe.submit(_fetch, t): t for t in COMBINED}
            for f in as_completed(futures):
                ticker, row = f.result()
                if row:
                    rows[ticker] = row

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "ticker"
        return df
    except Exception:
        return pd.DataFrame()
