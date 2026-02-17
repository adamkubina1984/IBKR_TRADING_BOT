from __future__ import annotations

from typing import Any

try:
    from ib_insync import Future, Stock
except Exception:
    Future = None
    Stock = None

def build_contract(symbol: str, expiry: str | None, exchange: str | None, currency: str = "USD"):
    """Return IB contract for futures (if expiry given) or stock otherwise.
    Safe to import even when ib_insync is missing (callers should catch)."""
    if Future is None:
        raise RuntimeError("ib_insync not available")
    if expiry:
        return Future(symbol, lastTradeDateOrContractMonth=expiry, exchange=exchange or "NYMEX", currency=currency)
    else:
        ex = exchange or "SMART"
        cur = currency or "USD"
        if Stock is None:
            # very defensive fallback: synthesize a future far in time
            return Future(symbol, lastTradeDateOrContractMonth="202512", exchange=ex, currency=cur)
        return Stock(symbol, ex, cur)

def map_bar_size_and_duration(label: str) -> tuple[str, str]:
    """Map UI label to (barSizeSetting, durationStr) for historical data requests."""
    key = (label or "1 min").strip().lower()
    mapping = {
        "1 min": ("1 min", "2 D"),
        "5 mins": ("5 mins", "5 D"),
        "15 mins": ("15 mins", "10 D"),
        "30 mins": ("30 mins", "20 D"),
        "1 hour": ("1 hour", "30 D"),
    }
    return mapping.get(key, ("1 min", "2 D"))

def bar_to_payload(b: Any) -> dict[str, float]:
    """Convert ib_insync BarData/BarDataList item to a serializable dict."""
    return {
        "time": getattr(b, "time", getattr(b, "date", None)),
        "open": float(getattr(b, "open", float("nan"))),
        "high": float(getattr(b, "high", float("nan"))),
        "low":  float(getattr(b, "low",  float('nan'))),
        "close": float(getattr(b, "close", float("nan"))),
        "volume": float(getattr(b, "volume", 0.0)),
    }
