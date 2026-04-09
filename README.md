# MarketWatch

Real-time (15-min delayed) market monitoring and options screener built on the [Massive API](https://massive.com) (formerly Polygon.io).

## Apps

### `market_monitor.py` — Terminal Dashboard
Bloomberg-style terminal monitor. Tracks a configurable basket of tickers with live trend scoring, technical indicators, and a market breadth panel.

```bash
python3 market_monitor.py
python3 market_monitor.py --tickers SPY QQQ IWM DIA QQQI --interval 60
```

**Features**
- Per-ticker price, OHLC, volume vs 30-day average, 20-day sparkline
- Trend scoring (−5 to +5) from SMA 20/50/200, RSI-14, MACD
- Session badge: `OPEN` / `AFTER-HRS` / `PRE-MKT` / `CLOSED`
- Market Breadth TICK Proxy — All-Market, NYSE, Nasdaq advance/decline counts
- Auto-switches from REST polling to WebSocket streaming when market opens
- Keyboard: `Q` quit · `R` force refresh · `P` pause/resume

### `wheel_screener.py` — Streamlit Options Screener
Web-based screener for finding ideal strikes for cash-secured puts and covered calls in a wheel strategy.

```bash
streamlit run wheel_screener.py
```

**Features**
- Filter options chain by DTE range, |delta| range, min OI, min annualized yield
- Wheel zone highlighting (configurable delta + DTE sweet spot)
- IV smile chart, yield vs delta scatter, theta decay by strike
- Top picks ranked by wheel score (0–5)
- Market breadth TICK proxy cards + rolling history chart

## Setup

### Requirements
- Python 3.9+
- [Massive API](https://massive.com) account — Stocks Starter plan or higher (provides WebSockets, 15-min delayed data)

### Install dependencies
```bash
pip install massive rich streamlit plotly python-dotenv
```

### Configure API key
Create a `.env` file in the project directory:
```
MASSIVE_API_KEY=your_key_here
```

## Data Notes

| Feature | Source | Latency |
|---|---|---|
| Prices, OHLC, volume | REST snapshot | 15-min delayed |
| SMA, EMA, RSI, MACD | REST indicators | 15-min delayed |
| Intraday price updates | WebSocket (market hours) | 15-min delayed |
| Options chain | REST snapshot | 15-min delayed |
| Market breadth (TICK proxy) | `get_snapshot_all` | 15-min delayed |

**Note on Market Breadth:** The `$TICK` values shown are an advance/decline count (stocks up vs. down vs. previous close), not true NYSE $TICK which requires real-time tick-by-tick data. They are directionally useful but are labeled "Breadth TICK Proxy" throughout the UI.

## Files

| File | Purpose |
|---|---|
| `market_monitor.py` | Terminal dashboard |
| `wheel_screener.py` | Streamlit options screener |
| `breadth.py` | Shared market breadth module (TICK proxy) |
| `test_websocket.py` | WebSocket connectivity test |
| `test_rest_data.py` | REST API endpoint tests |

## Wheel Strategy Overview

The wheel strategy involves:
1. **Sell cash-secured puts** on a stock you want to own at a strike below the current price
2. If assigned, **sell covered calls** on the shares above your cost basis
3. Repeat — collecting premium throughout

The screener targets the typical wheel entry parameters:
- **DTE:** 30–45 days to expiration
- **Delta:** 0.20–0.30 (approximately 70–80% probability of expiring worthless)
- **Annualized yield:** varies by IV environment
