"""
Massive API — after-hours REST data tests for QQQI.
Covers: snapshots, historical aggs, technical indicators,
        reference data, corporate actions, market status.

Usage:
    cd MarketWatch
    python3 test_rest_data.py
"""

import os
import sys
from datetime import date, timedelta
from pprint import pformat

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from massive import RESTClient

SYMBOL = "QQQI"
TODAY  = date.today()
ONE_YEAR_AGO = (TODAY - timedelta(days=365)).isoformat()
THIRTY_DAYS_AGO = (TODAY - timedelta(days=30)).isoformat()
YESTERDAY = (TODAY - timedelta(days=1)).isoformat()
TODAY_STR = TODAY.isoformat()


# ── helpers ──────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def ok(label: str, value):
    print(f"  {'[OK]':<6} {label}: {value}")

def fail(label: str, exc: Exception):
    print(f"  {'[ERR]':<6} {label}: {exc}")


# ── tests ─────────────────────────────────────────────────────────────────────

def test_market_status(client):
    section("Market Status")
    try:
        status = client.get_market_status()
        ok("market",       status.market)
        ok("server_time",  status.server_time)
        ok("exchanges",    status.exchanges)
    except Exception as e:
        fail("get_market_status", e)


def test_snapshot(client):
    section(f"Snapshot — {SYMBOL}")
    try:
        snap = client.get_snapshot_ticker("stocks", SYMBOL)
        d = snap.day
        p = snap.prev_day
        ok("ticker",         snap.ticker)
        ok("day  O/H/L/C",   f"{d.open} / {d.high} / {d.low} / {d.close}")
        ok("day  volume",    f"{d.volume:,.0f}")
        ok("day  vwap",      d.vwap)
        ok("prev C",         p.close)
        ok("day chg %",      f"{((d.close - p.close) / p.close * 100):.2f}%" if p.close else "N/A")
        ok("min  bid/ask",   f"{getattr(snap.min, 'open', 'N/A')} / {getattr(snap.min, 'close', 'N/A')}")
    except Exception as e:
        fail("get_snapshot_ticker", e)


def test_previous_close(client):
    section(f"Previous Close — {SYMBOL}")
    try:
        prev = client.get_previous_close_agg(SYMBOL)  # returns list directly
        for r in prev:
            ok("date",    r.timestamp)
            ok("O/H/L/C", f"{r.open} / {r.high} / {r.low} / {r.close}")
            ok("volume",  f"{r.volume:,.0f}")
    except Exception as e:
        fail("get_previous_close_agg", e)


def test_minute_aggs(client):
    section(f"Minute Aggregates — {SYMBOL} (last 5 trading days)")
    try:
        bars = list(client.list_aggs(
            ticker=SYMBOL,
            multiplier=1,
            timespan="minute",
            from_=THIRTY_DAYS_AGO,
            to=TODAY_STR,
            adjusted=True,
            sort="desc",
            limit=20,
        ))
        ok("bars returned", len(bars))
        if bars:
            b = bars[0]
            ok("most recent bar", f"O={b.open} H={b.high} L={b.low} C={b.close} V={b.volume} t={b.timestamp}")
        for b in bars[1:6]:
            print(f"         O={b.open:<8} H={b.high:<8} L={b.low:<8} C={b.close:<8} V={b.volume:<12,.0f} t={b.timestamp}")
    except Exception as e:
        fail("list_aggs (minute)", e)


def test_second_aggs(client):
    section(f"Second Aggregates — {SYMBOL} (last trading session sample)")
    try:
        bars = list(client.list_aggs(
            ticker=SYMBOL,
            multiplier=1,
            timespan="second",
            from_=YESTERDAY,
            to=TODAY_STR,
            adjusted=True,
            sort="desc",
            limit=10,
        ))
        ok("second bars returned", len(bars))
        for b in bars[:5]:
            print(f"         O={b.open:<8} C={b.close:<8} V={b.volume:<10} t={b.timestamp}")
    except Exception as e:
        fail("list_aggs (second)", e)


def test_daily_aggs(client):
    section(f"Daily Aggregates — {SYMBOL} (1 year)")
    try:
        bars = list(client.list_aggs(
            ticker=SYMBOL,
            multiplier=1,
            timespan="day",
            from_=ONE_YEAR_AGO,
            to=TODAY_STR,
            adjusted=True,
            sort="asc",
            limit=365,
        ))
        ok("trading days returned", len(bars))
        if len(bars) >= 2:
            first, last = bars[0], bars[-1]
            change = ((last.close - first.close) / first.close) * 100
            ok("1yr return",    f"{change:.2f}%  ({first.close} → {last.close})")
            avg_vol = sum(b.volume for b in bars) / len(bars)
            ok("avg daily vol", f"{avg_vol:,.0f}")
            high = max(bars, key=lambda b: b.high)
            low  = min(bars, key=lambda b: b.low)
            ok("52wk high",     f"{high.high} on {high.timestamp}")
            ok("52wk low",      f"{low.low} on {low.timestamp}")
    except Exception as e:
        fail("list_aggs (day)", e)


def test_sma(client):
    section(f"SMA — {SYMBOL}")
    try:
        for period in (20, 50, 200):
            result = client.get_sma(
                SYMBOL,
                timespan="day",
                adjusted=True,
                window=period,
                series_type="close",
                order="desc",
                limit=1,
            )
            values = result.values or []  # SingleIndicatorResults.values
            val = f"{values[0].value:.4f}" if values else "N/A"
            ok(f"SMA-{period}", val)
    except Exception as e:
        fail("get_sma", e)


def test_ema(client):
    section(f"EMA — {SYMBOL}")
    try:
        for period in (12, 26, 50):
            result = client.get_ema(
                SYMBOL,
                timespan="day",
                adjusted=True,
                window=period,
                series_type="close",
                order="desc",
                limit=1,
            )
            values = result.values or []
            val = f"{values[0].value:.4f}" if values else "N/A"
            ok(f"EMA-{period}", val)
    except Exception as e:
        fail("get_ema", e)


def test_rsi(client):
    section(f"RSI — {SYMBOL}")
    try:
        result = client.get_rsi(
            SYMBOL,
            timespan="day",
            adjusted=True,
            window=14,
            series_type="close",
            order="desc",
            limit=5,
        )
        values = result.values or []
        ok("RSI-14 (latest)", f"{values[0].value:.2f}" if values else "N/A")
        for v in values[1:]:
            print(f"           t={v.timestamp}: {v.value:.2f}")
    except Exception as e:
        fail("get_rsi", e)


def test_macd(client):
    section(f"MACD — {SYMBOL}")
    try:
        result = client.get_macd(
            SYMBOL,
            timespan="day",
            adjusted=True,
            short_window=12,
            long_window=26,
            signal_window=9,
            series_type="close",
            order="desc",
            limit=1,
        )
        values = result.values or []  # MACDIndicatorResults.values
        if values:
            v = values[0]
            ok("MACD value",  f"{v.value:.4f}")
            ok("MACD signal", f"{v.signal:.4f}")
            ok("MACD hist",   f"{v.histogram:.4f}")
            ok("trend",       "Bullish" if v.value > v.signal else "Bearish")
    except Exception as e:
        fail("get_macd", e)


def test_ticker_details(client):
    section(f"Ticker Details — {SYMBOL}")
    try:
        r = client.get_ticker_details(SYMBOL)  # returns TickerDetails directly
        ok("name",             r.name)
        ok("type",             r.type)
        ok("market",           r.market)
        ok("primary_exchange", r.primary_exchange)
        ok("currency",         r.currency_name)
        ok("active",           r.active)
        ok("list_date",        r.list_date)
        ok("description",      (r.description or "")[:120])
        ok("market_cap",       getattr(r, "market_cap", "N/A"))
        branding = getattr(r, "branding", None)
        if branding:
            ok("logo_url",     getattr(branding, "logo_url", "N/A"))
    except Exception as e:
        fail("get_ticker_details", e)


def test_dividends(client):
    section(f"Dividends — {SYMBOL} (recent)")
    try:
        divs = list(client.list_stocks_dividends(
            ticker=SYMBOL,
            sort="ex_dividend_date",  # sort param, not order
            limit=12,
        ))
        ok("dividend records", len(divs))
        for d in sorted(divs, key=lambda x: x.ex_dividend_date, reverse=True)[:8]:
            print(f"         ex={d.ex_dividend_date}  pay={d.pay_date}  "
                  f"amount=${d.cash_amount}  freq={d.frequency}")
    except Exception as e:
        fail("list_stocks_dividends", e)


def test_splits(client):
    section(f"Splits — {SYMBOL}")
    try:
        splits = list(client.list_stocks_splits(ticker=SYMBOL))
        ok("split records", len(splits))
        for s in splits:
            print(f"         {s.execution_date}  {s.split_from}:{s.split_to}")
        if not splits:
            print("         (no splits found)")
    except Exception as e:
        fail("list_stocks_splits", e)


def test_related(client):
    section(f"Related Companies — {SYMBOL}")
    try:
        result = client.get_related_companies(SYMBOL)  # returns list directly
        tickers = [r.ticker for r in result] if result else []
        ok("related tickers", tickers[:10] if tickers else "(none)")
    except Exception as e:
        fail("get_related_companies", e)


def test_grouped_daily(client):
    section(f"Grouped Daily (all US stocks — {YESTERDAY})")
    try:
        bars = client.get_grouped_daily_aggs(date=YESTERDAY, adjusted=True)  # returns list
        ok("tickers in response", len(bars))
        qqqi = next((b for b in bars if getattr(b, "ticker", None) == SYMBOL), None)
        if qqqi:
            ok(f"{SYMBOL} in group",
               f"O={qqqi.open} H={qqqi.high} L={qqqi.low} C={qqqi.close} V={qqqi.volume:,.0f}")
        else:
            ok(f"{SYMBOL} in group", "not found — check ticker attr name")
            if bars:
                print(f"         sample attrs: {[a for a in dir(bars[0]) if not a.startswith('_')]}")
    except Exception as e:
        fail("get_grouped_daily_aggs", e)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        print("ERROR: MASSIVE_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    client = RESTClient(api_key=api_key)

    print(f"\nMassive API — after-hours data tests")
    print(f"Symbol: {SYMBOL}   Date range: {ONE_YEAR_AGO} → {TODAY_STR}")

    test_market_status(client)
    test_snapshot(client)
    test_previous_close(client)
    test_daily_aggs(client)
    test_minute_aggs(client)
    test_second_aggs(client)
    test_sma(client)
    test_ema(client)
    test_rsi(client)
    test_macd(client)
    test_ticker_details(client)
    test_dividends(client)
    test_splits(client)
    test_related(client)
    test_grouped_daily(client)

    print(f"\n{'='*60}")
    print("  All tests complete.")
    print('='*60)


if __name__ == "__main__":
    main()
