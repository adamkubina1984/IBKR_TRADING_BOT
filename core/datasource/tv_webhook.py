# ibkr_trading_bot/core/datasource/tv_webhook.py
from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request


class TVWebhookServer:
    """
    Spustí FastAPI server v samostatném vlákně a ukládá příchozí svíčky do fronty.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, secret: str = "CHANGE_ME"):
        self.host = host
        self.port = port
        self.secret = secret
        self._thread: threading.Thread | None = None
        self._app: FastAPI | None = None
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=10000)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        app = FastAPI(title="TV Webhook")
        self._app = app

        @app.post("/tv_ohlc")
        async def tv_ohlc(request: Request):
            # ověření secrettu přes query ?key=...
            key = request.query_params.get("key", "")
            if key != self.secret:
                raise HTTPException(status_code=403, detail="Forbidden")
            try:
                data = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Bad JSON")

            # očekáváme JSON: symbol, tf, time (epoch ms/sek), open, high, low, close, volume
            try:
                # čas může přijít v ms nebo sekundách
                t = data.get("time")
                if t is None:
                    raise ValueError("missing time")
                t = int(t)
                if t > 10_000_000_000:  # milisekundy
                    ts = pd.to_datetime(t, unit="ms", utc=True)
                else:
                    ts = pd.to_datetime(t, unit="s", utc=True)

                bar = {
                    "symbol": str(data.get("symbol", "")),
                    "tf": str(data.get("tf", "")),
                    "time": ts.to_pydatetime(),
                    "open": float(data.get("open")),
                    "high": float(data.get("high")),
                    "low": float(data.get("low")),
                    "close": float(data.get("close")),
                    "volume": int(data.get("volume", 0) or 0),
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid fields: {e}")

            try:
                self._queue.put_nowait(bar)
            except Exception:
                # fronta plná – nejstarší drop
                try:
                    _ = self._queue.get_nowait()
                    self._queue.put_nowait(bar)
                except Exception:
                    pass
            return {"status": "ok"}

        def _run():
            uvicorn.run(app, host=self.host, port=self.port, log_level="warning")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def get_next_bar(self, timeout: float = 0.1) -> dict[str, Any] | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None
