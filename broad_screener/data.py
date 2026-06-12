"""
data.py — Data fetching for broad fundamental screener.

Universe:  ~5,300 CS tickers from Massive → filter to ~2,000 liquid names
Fundamentals: yfinance (P/E, EPS growth, revenue growth, ROE, margins)
Momentum:  yfinance price history (SMA, RSI, 3-month return)
Insider:   Massive Form 4 open-market P/S transactions last 90 days
"""

import ast
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass


# ── helpers ───────────────────────────────────────────────────────────────────

def _massive_client():
    try:
        from massive import RESTClient
        key = os.getenv("MASSIVE_API_KEY")
        return RESTClient(api_key=key) if key else None
    except ImportError:
        return None

_TICKER_RE = re.compile(r'^[A-Z]{1,5}$')

def _primary_ticker(tickers_field) -> Optional[str]:
    """Extract clean primary ticker from Form 4 tickers field (stored as list-string)."""
    try:
        if isinstance(tickers_field, str):
            lst = ast.literal_eval(tickers_field)
        else:
            lst = list(tickers_field)
        for t in lst:
            if _TICKER_RE.match(str(t)):
                return str(t)
    except Exception:
        pass
    return None


# ── universe ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_universe(min_volume: int = 200_000, min_price: float = 2.0) -> List[str]:
    """
    Pull all actively trading US common stocks from Massive snapshot.
    Filter to liquid names: day volume > min_volume and price > min_price.
    Returns sorted list of ticker strings.
    """
    client = _massive_client()
    if client is None:
        return []

    tickers = []
    try:
        snaps = client.get_snapshot_all("stocks")
        for s in snaps:
            sym = getattr(s, "ticker", None)
            if not sym or not _TICKER_RE.match(sym):
                continue
            d = getattr(s, "day", None)
            if d is None:
                continue
            price  = getattr(d, "close", None) or getattr(d, "vwap", None) or 0
            volume = getattr(d, "volume", None) or 0
            if price >= min_price and volume >= min_volume:
                tickers.append(sym)
    except Exception:
        pass

    return sorted(set(tickers))


# ── fundamentals ──────────────────────────────────────────────────────────────

_FUND_FIELDS = [
    "shortName", "sector", "industry",
    "marketCap", "beta",
    "trailingPE", "forwardPE",
    "trailingEps", "forwardEps",
    "earningsGrowth", "revenueGrowth",
    "profitMargins", "returnOnEquity",
    "debtToEquity", "currentRatio",
    "freeCashflow", "totalRevenue",
    "recommendationMean", "targetMeanPrice",
    "regularMarketPrice",
]

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fundamentals(tickers: tuple) -> pd.DataFrame:
    """
    Fetch yfinance fundamentals for all tickers in parallel.
    Pass tickers as tuple for cache key hashing.
    Cached 6 hours.
    """
    rows = {}

    def _fetch(ticker: str):
        try:
            info = yf.Ticker(ticker).info
            if not info or info.get("regularMarketPrice") is None:
                return ticker, None
            return ticker, {f: info.get(f) for f in _FUND_FIELDS}
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=25) as exe:
        futures = {exe.submit(_fetch, t): t for t in tickers}
        for f in as_completed(futures):
            ticker, row = f.result()
            if row and row.get("sector"):
                rows[ticker] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    return df


# ── price momentum ────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return float("nan")
    delta = close.diff().dropna()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return float((100 - 100 / (1 + rs)).iloc[-1])


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_momentum(tickers: tuple, batch_size: int = 100) -> pd.DataFrame:
    """
    Download 6-month daily price history in batches, compute momentum indicators.
    Returns DataFrame indexed by ticker with: vs_sma50, vs_sma200, ret_3m, rsi14.
    Cached 6 hours.
    """
    rows = {}
    ticker_list = list(tickers)

    for i in range(0, len(ticker_list), batch_size):
        batch = ticker_list[i : i + batch_size]
        try:
            raw = yf.download(
                " ".join(batch),
                period="9mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue

            # yf.download returns MultiIndex columns when >1 ticker
            if isinstance(raw.columns, pd.MultiIndex):
                close_df = raw["Close"]
            else:
                close_df = raw[["Close"]].rename(columns={"Close": batch[0]})

            for ticker in batch:
                if ticker not in close_df.columns:
                    continue
                c = close_df[ticker].dropna()
                if len(c) < 63:
                    continue
                price = float(c.iloc[-1])
                sma50  = float(c.iloc[-50:].mean())  if len(c) >= 50  else float("nan")
                sma200 = float(c.iloc[-200:].mean()) if len(c) >= 200 else float("nan")
                ret3m  = float((c.iloc[-1] / c.iloc[-63] - 1) * 100) if len(c) >= 63 else float("nan")
                rows[ticker] = {
                    "vs_sma50":  (price - sma50)  / sma50  * 100 if not np.isnan(sma50)  else float("nan"),
                    "vs_sma200": (price - sma200) / sma200 * 100 if not np.isnan(sma200) else float("nan"),
                    "ret_3m":    ret3m,
                    "rsi14":     _rsi(c, 14),
                }
        except Exception:
            continue

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    return df


# ── insider activity (Form 4) ─────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_insider_data(days: int = 90, max_pages: int = 80) -> pd.DataFrame:
    """
    Paginate Massive Form 4 filings for the last `days` days.
    Filters to:
      - non-derivative transactions only
      - transaction_code in ('P', 'S')  — open-market purchase or sale
      - is_director or is_officer        — actual corporate insiders
    Aggregates per ticker:
      - net_buy_value: total purchase $ minus total sale $ (positive = net buying)
      - buy_count, sell_count, total_transactions
    Cached 24 hours.
    """
    client = _massive_client()
    if client is None:
        return pd.DataFrame()

    since = (date.today() - timedelta(days=days)).isoformat()
    records = []
    cursor  = None
    page    = 0

    while page < max_pages:
        try:
            params = {
                "filing_date.gte": since,
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor

            resp = client.list_form4(params=params) if hasattr(client, "list_form4") else None
            if resp is None:
                # fall back to direct HTTP
                import urllib.request, json, urllib.parse
                key    = os.getenv("MASSIVE_API_KEY", "")
                qs     = urllib.parse.urlencode(params)
                url    = f"https://api.massive.com/stocks/filings/vX/form-4?{qs}"
                req    = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read())
                items  = data.get("results", [])
                cursor = data.get("next_cursor")
            else:
                items  = list(resp)
                cursor = None  # iterator exhausts itself

            if not items:
                break

            for item in items:
                # support both dict (raw JSON) and object (client model)
                def _g(obj, key, default=None):
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                    return getattr(obj, key, default)

                record_type = _g(item, "record_type", "")
                if record_type != "non_derivative":
                    continue

                code = _g(item, "transaction_code", "")
                if code not in ("P", "S"):
                    continue

                is_dir = bool(_g(item, "is_director", False))
                is_off = bool(_g(item, "is_officer",  False))
                if not (is_dir or is_off):
                    continue

                tickers_raw = _g(item, "tickers", None)
                ticker      = _primary_ticker(tickers_raw)
                if not ticker:
                    continue

                value  = _g(item, "transaction_value", None) or 0
                acq_or_disp = _g(item, "transaction_acquired_disposed", "")

                # positive = acquired (purchase), negative = disposed (sale)
                signed_value = abs(float(value)) if acq_or_disp == "A" else -abs(float(value))

                records.append({
                    "ticker":       ticker,
                    "code":         code,
                    "signed_value": signed_value,
                    "filing_date":  _g(item, "filing_date", ""),
                })

            page += 1
            if cursor is None:
                break

        except Exception:
            break

    if not records:
        return pd.DataFrame()

    raw = pd.DataFrame(records)
    agg = (
        raw.groupby("ticker")
           .agg(
               net_buy_value    = ("signed_value", "sum"),
               buy_count        = ("code", lambda x: (x == "P").sum()),
               sell_count       = ("code", lambda x: (x == "S").sum()),
               total_insider_tx = ("code", "count"),
           )
           .reset_index()
           .set_index("ticker")
    )
    return agg


# ── direct Form 4 fetch via REST (fallback if client has no list_form4) ───────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_insider_via_rest(days: int = 90, max_pages: int = 80) -> pd.DataFrame:
    """
    Fetch Form 4 open-market transactions via Massive REST API directly.
    Returns DataFrame indexed by ticker with net_buy_value and counts.
    Cached 24 hours.
    """
    key = os.getenv("MASSIVE_API_KEY", "")
    if not key:
        return pd.DataFrame()

    since   = (date.today() - timedelta(days=days)).isoformat()
    records = []
    cursor  = None
    page    = 0
    base    = "https://api.massive.com"

    import urllib.request, json, urllib.parse

    while page < max_pages:
        try:
            params = {"filing_date.gte": since, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            qs  = urllib.parse.urlencode(params)
            url = f"{base}/stocks/filings/vX/form-4?{qs}"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())

            results     = data.get("results", [])
            next_cursor = data.get("next_cursor") or data.get("cursor")

            for item in results:
                if item.get("record_type") != "non_derivative":
                    continue
                if item.get("transaction_code") not in ("P", "S"):
                    continue
                if not (item.get("is_director") or item.get("is_officer")):
                    continue
                ticker = _primary_ticker(item.get("tickers"))
                if not ticker:
                    continue
                value       = item.get("transaction_value") or 0
                acq         = item.get("transaction_acquired_disposed", "")
                signed_val  = abs(float(value)) if acq == "A" else -abs(float(value))
                records.append({
                    "ticker":       ticker,
                    "code":         item.get("transaction_code"),
                    "signed_value": signed_val,
                })

            page  += 1
            cursor = next_cursor
            if not cursor or not results:
                break

        except Exception:
            break

    if not records:
        return pd.DataFrame()

    raw = pd.DataFrame(records)
    return (
        raw.groupby("ticker")
           .agg(
               net_buy_value    = ("signed_value", "sum"),
               buy_count        = ("code", lambda x: (x == "P").sum()),
               sell_count       = ("code", lambda x: (x == "S").sum()),
               total_insider_tx = ("code", "count"),
           )
           .reset_index()
           .set_index("ticker")
    )
