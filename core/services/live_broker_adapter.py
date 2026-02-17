from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any


class LiveBrokerAdapter:
    """Tenký wrapper nad ib_insync pro real-time bary (bez Qt)."""
    def __init__(self, host: str='127.0.0.1', port: int=7497, client_id: int=199):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None

    def connect(self):
        from ib_insync import IB
        self.ib = IB()
        self.ib.connect(self.host, self.port, clientId=self.client_id)

    def disconnect(self):
        if self.ib:
            self.ib.disconnect()

    def subscribe_bars(self, symbol: str, exchange: str, currency: str, contract_month: str | None, what: str='TRADES', rth: bool=False) -> Generator[dict[str, Any], None, None]:
        """Yield 5s bary jako dict: {time, open, high, low, close, volume}."""
        from ib_insync import Future, Stock
        if not self.ib:
            self.connect()

        if contract_month:
            contract = Future(symbol, lastTradeDateOrContractMonth=contract_month, exchange=exchange, currency=currency)
        else:
            contract = Stock(symbol, exchange, currency)

        bars = self.ib.reqRealTimeBars(contract, 5, what, rth)
        last_ts = None
        for _ in bars.updateEvent:
            bar = bars[-1]
            d = {
                "time": bar.time,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            if last_ts is None or d["time"] != last_ts:
                last_ts = d["time"]
                yield d
            time.sleep(0.01)
