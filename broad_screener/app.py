"""
Broad Fundamental Screener
==========================
~5,300 US common stocks → filtered to ~2,000 liquid names → scored and ranked.

Composite score (-12 to +12):
  Fundamental  -6 → +6   valuation, growth, profitability
  Momentum     -3 → +3   price vs SMA50/200, 3-month return
  Insider      -3 → +3   net open-market buy/sell $ last 90 days (Massive Form 4)

Data sources:
  Universe:      Massive API get_snapshot_all  (liquid CS tickers)
  Fundamentals:  yfinance                      (P/E, EPS growth, ROE, margins…)
  Momentum:      yfinance price history
  Insider:       Massive Form 4                (SEC-filed insider transactions)

Usage:
    cd MarketWatch/broad_screener
    streamlit run app.py --server.port 8503
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data  import fetch_universe, fetch_fundamentals, fetch_momentum, fetch_insider_via_rest
from score import build_screener


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Broad Fundamental Screener",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-card { background:#1e1e2e; border-radius:8px; padding:12px 16px; margin:4px 0; }
  .bull-chip { background:#1e3a2f; color:#a6e3a1; padding:2px 8px; border-radius:6px;
               font-size:.7rem; font-weight:700; display:inline-block }
  .bear-chip { background:#3a1e1e; color:#f38ba8; padding:2px 8px; border-radius:6px;
               font-size:.7rem; font-weight:700; display:inline-block }
  div[data-testid="stMetric"] label { font-size:.7rem !important }
</style>
""", unsafe_allow_html=True)


# ── display helpers ───────────────────────────────────────────────────────────

def _pct(v, decimals=1, fallback="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return fallback
    color = "#a6e3a1" if v > 0 else ("#f38ba8" if v < 0 else "#cdd6f4")
    arrow = "▲" if v > 0 else ("▼" if v < 0 else "─")
    return f"<span style='color:{color}'>{arrow}{abs(v):.{decimals}f}%</span>"

def _val(v, fmt="{:.1f}", fallback="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return fallback
    return fmt.format(v)

def _cap(v):
    if not v or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if v >= 1e12: return f"${v/1e12:.1f}T"
    if v >= 1e9:  return f"${v/1e9:.0f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:.0f}"

def _insider_fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<span style='color:#6c7086'>—</span>"
    v = float(v)
    color = "#a6e3a1" if v > 0 else ("#f38ba8" if v < 0 else "#6c7086")
    sign  = "+" if v > 0 else ""
    if   abs(v) >= 1e6: s = f"{sign}${v/1e6:.1f}M"
    elif abs(v) >= 1e3: s = f"{sign}${v/1e3:.0f}K"
    else:               s = f"{sign}${v:.0f}"
    return f"<span style='color:{color};font-weight:600'>{s}</span>"

def score_bar(score: int, lo: int = -12, hi: int = 12) -> str:
    pct   = max(0, min(100, (score - lo) / (hi - lo) * 100))
    color = "#a6e3a1" if score > 2 else ("#f38ba8" if score < -2 else "#f9e2af")
    return (
        f"<div style='background:#313244;border-radius:4px;height:14px;width:100%'>"
        f"<div style='background:{color};border-radius:4px;height:14px;width:{pct:.0f}%'></div>"
        f"</div>"
        f"<span style='font-size:.72rem;color:{color}'>{score:+d}</span>"
    )

def sub_bar(score: int, lo: int, hi: int, color: str) -> str:
    pct = max(0, min(100, (score - lo) / (hi - lo) * 100))
    return (
        f"<div style='background:#313244;border-radius:3px;height:10px;width:80px;display:inline-block'>"
        f"<div style='background:{color};border-radius:3px;height:10px;width:{pct:.0f}%'></div>"
        f"</div> <span style='font-size:.7rem;color:{color}'>{score:+d}</span>"
    )

def build_table(subset: pd.DataFrame, include_insider: bool) -> str:
    rows_html = ""
    for _, r in subset.iterrows():
        rank  = int(r.get("rank", 0))
        comp  = int(r.get("composite", 0))
        fs    = int(r.get("fund_score", 0))
        ms    = int(r.get("momentum_score", 0))
        ins   = int(r.get("insider_score", 0))
        rows_html += (
            f"<tr>"
            f"<td style='color:#6c7086;width:36px'>{rank}</td>"
            f"<td style='font-weight:700;color:#cdd6f4'>{r.get('ticker','')}</td>"
            f"<td style='color:#a6adc8;max-width:160px;overflow:hidden;white-space:nowrap'>"
            f"{(r.get('shortName') or '')[:22]}</td>"
            f"<td style='color:#6c7086;font-size:.75rem'>{(r.get('sector') or '')[:18]}</td>"
            f"<td style='min-width:90px'>{score_bar(comp)}</td>"
            f"<td style='min-width:80px'>{sub_bar(fs, -6, 6, '#89b4fa')}</td>"
            f"<td style='min-width:80px'>{sub_bar(ms, -3, 3, '#94e2d5')}</td>"
        )
        if include_insider:
            rows_html += f"<td style='min-width:80px'>{sub_bar(ins, -3, 3, '#cba6f7')}</td>"
            rows_html += f"<td>{_insider_fmt(r.get('net_buy_value'))}</td>"
        rows_html += (
            f"<td>{_val(r.get('forwardPE'), '{:.1f}×')}</td>"
            f"<td>{_pct(r.get('earningsGrowth', np.nan) * 100 if pd.notna(r.get('earningsGrowth')) else np.nan, 0)}</td>"
            f"<td>{_pct(r.get('revenueGrowth',  np.nan) * 100 if pd.notna(r.get('revenueGrowth'))  else np.nan, 0)}</td>"
            f"<td>{_pct(r.get('returnOnEquity', np.nan) * 100 if pd.notna(r.get('returnOnEquity')) else np.nan, 0)}</td>"
            f"<td>{_pct(r.get('ret_3m'), 1)}</td>"
            f"<td style='color:#a6adc8'>{_cap(r.get('marketCap'))}</td>"
            f"</tr>"
        )

    insider_headers = (
        "<th>Ins Score</th><th>Insider Net $</th>"
        if include_insider else ""
    )
    return (
        "<table style='width:100%;font-size:.78rem;border-collapse:collapse'>"
        "<thead><tr style='color:#a6adc8;border-bottom:1px solid #313244'>"
        "<th>#</th><th>Ticker</th><th>Company</th><th>Sector</th>"
        "<th>Composite</th><th>Fund</th><th>Momentum</th>"
        f"{insider_headers}"
        "<th>Fwd P/E</th><th>EPS Gr</th><th>Rev Gr</th><th>ROE</th>"
        "<th>3M Ret</th><th>Mkt Cap</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Broad Screener")
    st.caption("~5,300 US common stocks · filtered to liquid names")
    st.divider()

    st.subheader("Universe Filters")
    min_vol   = st.selectbox("Min daily volume", ["200K+", "500K+", "1M+"], index=0)
    min_price = st.number_input("Min price ($)", value=2.0, step=0.5, min_value=0.5)
    vol_map   = {"200K+": 200_000, "500K+": 500_000, "1M+": 1_000_000}

    st.divider()
    st.subheader("Score Filters")
    cap_map    = {"Any": 0, "$100M+": 1e8, "$500M+": 5e8, "$1B+": 1e9, "$5B+": 5e9, "$10B+": 1e10}
    min_cap    = st.selectbox("Min market cap", list(cap_map.keys()), index=0)
    sectors    = ["All Sectors","Technology","Financials","Healthcare","Consumer Cyclical",
                  "Consumer Defensive","Industrials","Energy","Communication Services",
                  "Utilities","Real Estate","Basic Materials"]
    sector_sel = st.selectbox("Sector", sectors)

    st.divider()
    st.subheader("Signals")
    use_insider = st.toggle("Include insider signal", value=True,
                            help="Massive Form 4 — net open-market buy/sell $ last 90 days. Cached 24h.")
    n_show = st.slider("Tickers per list", 25, 100, 100, step=25)

    st.divider()
    if st.button("🔄 Refresh all data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Fundamentals: 6h  ·  Momentum: 6h  ·  Insider: 24h  ·  Universe: 1h")


# ── load data ─────────────────────────────────────────────────────────────────

with st.spinner("Step 1/4 — Fetching liquid universe from Massive…"):
    universe = fetch_universe(
        min_volume=vol_map[min_vol],
        min_price=min_price,
    )

if not universe:
    st.error("Could not load universe. Check MASSIVE_API_KEY in .env")
    st.stop()

st.sidebar.caption(f"Universe: {len(universe):,} tickers")

with st.spinner(f"Step 2/4 — Loading fundamentals for {len(universe):,} tickers via yfinance (2–4 min first run, then cached)…"):
    fund_df = fetch_fundamentals(tuple(universe))

with st.spinner(f"Step 3/4 — Loading price momentum…"):
    mom_df = fetch_momentum(tuple(universe))

insider_df = pd.DataFrame()
if use_insider:
    with st.spinner("Step 4/4 — Loading insider transactions from Massive Form 4 (last 90 days, cached 24h)…"):
        insider_df = fetch_insider_via_rest(days=90, max_pages=80)

# ── build composite ───────────────────────────────────────────────────────────

df = build_screener(
    fundamentals   = fund_df,
    momentum       = mom_df,
    insider        = insider_df,
    min_market_cap = cap_map[min_cap],
    include_insider= use_insider,
)

if df.empty:
    st.error("No data after scoring. Try relaxing filters.")
    st.stop()

# sector filter
if sector_sel != "All Sectors":
    df = df[df["sector"] == sector_sel].reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

# ── summary banner ────────────────────────────────────────────────────────────

bull = (df["composite"] > 3).sum()
bear = (df["composite"] < -3).sum()
avg  = df["composite"].mean()
ins_active = df["total_insider_tx"].notna().sum() if "total_insider_tx" in df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stocks Screened",   f"{len(df):,}")
c2.metric("Bullish  (>+3)",    f"{bull:,}  ({bull/len(df)*100:.0f}%)")
c3.metric("Bearish  (<−3)",    f"{bear:,}  ({bear/len(df)*100:.0f}%)")
c4.metric("Avg Composite",     f"{avg:+.1f}")
c5.metric("Insider Coverage",  f"{ins_active:,}" if use_insider else "Off")

st.divider()


# ── tabs ──────────────────────────────────────────────────────────────────────

tab_bull, tab_bear, tab_all, tab_method = st.tabs([
    f"🟢 Top {n_show} Bullish",
    f"🔴 Top {n_show} Bearish",
    "📋 Full Rankings",
    "ℹ️ Methodology",
])


# ── TOP BULLISH ───────────────────────────────────────────────────────────────

with tab_bull:
    top_bull = df.head(n_show)
    st.caption(f"Highest composite scores · {len(top_bull)} stocks · "
               f"composite range {top_bull['composite'].max():+d} → {top_bull['composite'].min():+d}")

    # mini sector distribution
    if "sector" in top_bull.columns:
        sec_counts = top_bull["sector"].value_counts().head(6)
        cols = st.columns(min(6, len(sec_counts)))
        for i, (sec, cnt) in enumerate(sec_counts.items()):
            cols[i].markdown(
                f"<div style='text-align:center'>"
                f"<div style='color:#a6e3a1;font-size:1.1rem;font-weight:700'>{cnt}</div>"
                f"<div style='color:#6c7086;font-size:.68rem'>{sec[:14]}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.write("")

    st.markdown(build_table(top_bull, use_insider), unsafe_allow_html=True)

    csv = top_bull[["ticker","shortName","sector","composite","fund_score",
                    "momentum_score","insider_score","forwardPE","earningsGrowth",
                    "revenueGrowth","returnOnEquity","ret_3m","marketCap"]].to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "bullish_top.csv", "text/csv")


# ── TOP BEARISH ───────────────────────────────────────────────────────────────

with tab_bear:
    top_bear = df.tail(n_show).sort_values("composite", ascending=True)
    top_bear["rank"] = range(len(df), len(df) - n_show, -1)
    st.caption(f"Lowest composite scores · {len(top_bear)} stocks · "
               f"composite range {top_bear['composite'].min():+d} → {top_bear['composite'].max():+d}")

    if "sector" in top_bear.columns:
        sec_counts = top_bear["sector"].value_counts().head(6)
        cols = st.columns(min(6, len(sec_counts)))
        for i, (sec, cnt) in enumerate(sec_counts.items()):
            cols[i].markdown(
                f"<div style='text-align:center'>"
                f"<div style='color:#f38ba8;font-size:1.1rem;font-weight:700'>{cnt}</div>"
                f"<div style='color:#6c7086;font-size:.68rem'>{sec[:14]}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.write("")

    st.markdown(build_table(top_bear, use_insider), unsafe_allow_html=True)

    csv = top_bear[["ticker","shortName","sector","composite","fund_score",
                    "momentum_score","insider_score","forwardPE","earningsGrowth",
                    "revenueGrowth","returnOnEquity","ret_3m","marketCap"]].to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "bearish_top.csv", "text/csv")


# ── FULL RANKINGS ─────────────────────────────────────────────────────────────

with tab_all:
    st.caption(f"All {len(df):,} scored stocks — sorted by composite score")

    # score distribution chart
    fig = go.Figure(go.Histogram(
        x=df["composite"],
        nbinsx=25,
        marker_color=[
            "#a6e3a1" if v > 3 else "#f38ba8" if v < -3 else "#f9e2af"
            for v in df["composite"]
        ],
    ))
    fig.update_layout(
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"), height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Composite Score", gridcolor="#313244",
                   tickvals=list(range(-12, 13, 3))),
        yaxis=dict(title="# Stocks", gridcolor="#313244"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # sector breakdown
    if "sector" in df.columns:
        sec_agg = (
            df.dropna(subset=["sector"])
              .groupby("sector")
              .agg(count=("composite","count"), avg=("composite","mean"),
                   bull=("composite", lambda x: (x > 3).sum()),
                   bear=("composite", lambda x: (x < -3).sum()))
              .reset_index()
              .sort_values("avg", ascending=False)
        )
        sec_agg["avg"] = sec_agg["avg"].round(2)
        sec_agg.columns = ["Sector", "# Stocks", "Avg Score", "Bullish", "Bearish"]
        st.dataframe(sec_agg, use_container_width=True, hide_index=True)

    st.write("")
    st.markdown(build_table(df.head(200), use_insider), unsafe_allow_html=True)

    full_csv = df[["rank","ticker","shortName","sector","composite","fund_score",
                   "momentum_score","insider_score","forwardPE","earningsGrowth",
                   "revenueGrowth","returnOnEquity","profitMargins","ret_3m",
                   "vs_sma50","vs_sma200","rsi14","marketCap"]].to_csv(index=False)
    st.download_button("⬇️ Download full rankings CSV", full_csv, "full_rankings.csv", "text/csv")


# ── METHODOLOGY ───────────────────────────────────────────────────────────────

with tab_method:
    st.subheader("How Stocks Are Scored")
    st.markdown("""
    ### Composite Score  (−12 to +12)
    Three independent signal layers are summed to produce the composite score.
    Higher = more bullish. Lower = more bearish.

    ---

    ### 🔵 Fundamental Score  (−6 to +6)
    Sourced from **yfinance** (quarterly updated):

    | Signal | Bullish | Bearish |
    |--------|---------|---------|
    | Forward P/E | <15 → +2, 15–25 → +1 | 40–60 → −1, >60 → −2 |
    | EPS Growth (YoY) | >25% → +2, 10–25% → +1 | <−10% → −2, 0–−10% → −1 |
    | Revenue Growth | >15% → +1 | <−5% → −1 |
    | Return on Equity | >20% → +1 | <0% → −1 |
    | Profit Margin | >20% → +1 | <0% → −1 |

    ---

    ### 🟢 Momentum Score  (−3 to +3)
    Sourced from **yfinance** price history (6-month daily bars):

    | Signal | Bullish | Bearish |
    |--------|---------|---------|
    | Price vs SMA-50 | Above → +1 | Below → −1 |
    | Price vs SMA-200 | Above → +1 | Below → −1 |
    | 3-Month Return | >+10% → +1 | <−10% → −1 |

    ---

    ### 🟣 Insider Score  (−3 to +3)
    Sourced from **Massive API Form 4** — SEC insider transaction filings, last 90 days.

    Only counts:
    - **Open-market purchases (P)** and **sales (S)** — not grants, not option exercises
    - By **directors and officers** — not 10% owners (who may be activists)
    - **Non-derivative** transactions only — actual stock, not options

    | Net Buy $ (90 days) | Score |
    |---------------------|-------|
    | > $5M | +3 |
    | $1M–$5M | +2 |
    | $100K–$1M | +1 |
    | ±$100K | 0 |
    | −$100K to −$1M | −1 |
    | −$1M to −$5M | −2 |
    | < −$5M | −3 |

    **Why insider buying matters:** Studies consistently show that open-market purchases by
    corporate insiders (who have legal access to non-public information) predict 3–12 month
    outperformance. Insider *sales* are noisier (often for diversification or tax reasons),
    so they carry less weight here — but large sustained selling is still a warning signal.

    ---

    ### Universe Construction
    1. Massive `get_snapshot_all` returns all ~5,300 active US common stocks
    2. Filtered by: day volume ≥ threshold and price ≥ threshold
    3. Fundamentals fetched from yfinance — tickers with no sector data dropped
    4. Result: ~1,500–2,500 scoreable stocks depending on filters
    """)
