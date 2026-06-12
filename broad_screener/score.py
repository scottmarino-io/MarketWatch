"""
score.py — Composite scoring for broad fundamental screener.

Three signal layers:
  Fundamental score  -6 → +6   (valuation + growth + profitability)
  Insider score      -3 → +3   (net open-market buy/sell $ last 90 days)
  Momentum score     -3 → +3   (price vs SMAs + 3-month return)

Composite = fundamental + insider + momentum  →  -12 to +12
"""

import numpy as np
import pandas as pd


# ── fundamental score ─────────────────────────────────────────────────────────

def score_fundamental(row: pd.Series) -> int:
    """
    Score a ticker on five fundamental dimensions.
    Range: -6 to +6.
    """
    score = 0

    # 1. Valuation: forward P/E
    fpe = row.get("forwardPE")
    if pd.notna(fpe) and fpe > 0:
        if   fpe < 15:  score += 2
        elif fpe < 25:  score += 1
        elif fpe < 40:  score += 0
        elif fpe < 60:  score -= 1
        else:           score -= 2

    # 2. Earnings growth (EPS YoY)
    eg = row.get("earningsGrowth")
    if pd.notna(eg):
        if   eg >  0.25: score += 2
        elif eg >  0.10: score += 1
        elif eg >  0.00: score += 0
        elif eg > -0.10: score -= 1
        else:            score -= 2

    # 3. Revenue growth
    rg = row.get("revenueGrowth")
    if pd.notna(rg):
        if   rg >  0.15: score += 1
        elif rg >  0.05: score += 0
        elif rg < -0.05: score -= 1

    # 4. Return on equity
    roe = row.get("returnOnEquity")
    if pd.notna(roe):
        if   roe > 0.20: score += 1
        elif roe > 0.00: score += 0
        else:            score -= 1

    # 5. Profit margin
    pm = row.get("profitMargins")
    if pd.notna(pm):
        if   pm > 0.20: score += 1
        elif pm > 0.00: score += 0
        else:           score -= 1

    # Hard cap at ±6
    return max(-6, min(6, score))


# ── insider score ─────────────────────────────────────────────────────────────

def score_insider(row: pd.Series) -> int:
    """
    Score based on net open-market insider buying/selling last 90 days.
    Range: -3 to +3.

    net_buy_value > 0: insiders net buying  → bullish
    net_buy_value < 0: insiders net selling → bearish
    """
    v = row.get("net_buy_value")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    v = float(v)
    if   v >  5_000_000: return  3   # >$5M net buying
    elif v >  1_000_000: return  2   # $1M–$5M
    elif v >    100_000: return  1   # $100K–$1M
    elif v > -  100_000: return  0   # neutral
    elif v > -1_000_000: return -1
    elif v > -5_000_000: return -2
    else:                return -3   # >$5M net selling


# ── momentum score ────────────────────────────────────────────────────────────

def score_momentum(row: pd.Series) -> int:
    """
    Score based on price momentum.
    Range: -3 to +3.
    """
    score = 0

    vs50 = row.get("vs_sma50")
    if pd.notna(vs50):
        score += 1 if vs50 > 0 else -1

    vs200 = row.get("vs_sma200")
    if pd.notna(vs200):
        score += 1 if vs200 > 0 else -1

    r3m = row.get("ret_3m")
    if pd.notna(r3m):
        if   r3m >  10: score += 1
        elif r3m < -10: score -= 1

    return max(-3, min(3, score))


# ── assemble screener DataFrame ───────────────────────────────────────────────

def build_screener(
    fundamentals:  pd.DataFrame,
    momentum:      pd.DataFrame,
    insider:       pd.DataFrame,
    min_market_cap: float = 0,
    include_insider: bool = True,
) -> pd.DataFrame:
    """
    Merge all data sources, compute composite score, return ranked DataFrame.

    Parameters
    ----------
    fundamentals   : indexed by ticker — yfinance fundamental fields
    momentum       : indexed by ticker — vs_sma50, vs_sma200, ret_3m, rsi14
    insider        : indexed by ticker — net_buy_value, buy_count, sell_count
    min_market_cap : filter to market caps >= this value
    include_insider: whether to include insider score in composite
    """
    df = fundamentals.copy()

    # merge momentum
    for col in ["vs_sma50", "vs_sma200", "ret_3m", "rsi14"]:
        if col in momentum.columns:
            df[col] = momentum[col]

    # merge insider
    if include_insider and not insider.empty:
        for col in ["net_buy_value", "buy_count", "sell_count", "total_insider_tx"]:
            if col in insider.columns:
                df[col] = insider[col]

    # market cap filter
    if min_market_cap > 0 and "marketCap" in df.columns:
        df = df[df["marketCap"].fillna(0) >= min_market_cap]

    # scores
    df["fund_score"]     = df.apply(score_fundamental, axis=1)
    df["momentum_score"] = df.apply(score_momentum,    axis=1)
    df["insider_score"]  = df.apply(score_insider,     axis=1) if include_insider else 0
    df["composite"]      = df["fund_score"] + df["momentum_score"] + df["insider_score"]

    df = df.reset_index()
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df
