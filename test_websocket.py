"""
Massive API WebSocket test — real-time pricing data for QQQI.

Usage:
    python test_websocket.py
    (reads MASSIVE_API_KEY from .env file or environment)
"""

import os
import signal
import sys
from datetime import datetime
from typing import List

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # fallback to environment variable

from massive import WebSocketClient
from massive.websocket.models import WebSocketMessage


SYMBOL = "QQQI"

# Track basic stats per run
stats = {
    "trades": 0,
    "quotes": 0,
    "aggs": 0,
}


def handle_msg(msgs: List[WebSocketMessage]):
    for m in msgs:
        ev = getattr(m, "ev", None)
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if ev == "T":  # Trade
            stats["trades"] += 1
            print(
                f"[{ts}] TRADE  | price={m.p:<10.4f} size={m.s:<8} "
                f"conditions={getattr(m, 'c', [])} total_trades={stats['trades']}"
            )

        elif ev == "Q":  # Quote
            stats["quotes"] += 1
            print(
                f"[{ts}] QUOTE  | bid={getattr(m, 'bp', 'N/A'):<10} "
                f"ask={getattr(m, 'ap', 'N/A'):<10} "
                f"bid_sz={getattr(m, 'bs', 'N/A')} ask_sz={getattr(m, 'as', 'N/A')}"
            )

        elif ev in ("A", "AM"):  # Aggregate (per-second or per-minute)
            stats["aggs"] += 1
            label = "AGG/s " if ev == "A" else "AGG/m "
            print(
                f"[{ts}] {label} | o={m.o:<10.4f} h={m.h:<10.4f} "
                f"l={m.l:<10.4f} c={m.c:<10.4f} v={m.av:<12,.0f}"
            )

        else:
            # Status / auth messages and anything else
            print(f"[{ts}] {m}")


def main():
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        print("ERROR: Set MASSIVE_API_KEY environment variable before running.")
        print("  export MASSIVE_API_KEY='your_key_here'")
        sys.exit(1)

    print(f"Connecting to Massive WebSocket — symbol: {SYMBOL}")
    print("Subscriptions: Trades (T), Quotes (Q), Per-second Aggs (A)")
    print("Press Ctrl+C to stop.\n")

    ws = WebSocketClient(
        api_key=api_key,
        subscriptions=[
            f"T.{SYMBOL}",   # trades
            f"Q.{SYMBOL}",   # quotes
            f"A.{SYMBOL}",   # per-second aggregates
        ],
    )

    def shutdown(sig, frame):
        print(f"\n\nStopped. Summary: {stats}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    ws.run(handle_msg=handle_msg)


if __name__ == "__main__":
    main()
