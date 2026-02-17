from __future__ import annotations

"""
LiveBotService (IBKR stream)
- OPRAVA: odstraněny kruhové importy a nepoužité TradingViewClient importy
- OPRAVA: žádný top-level kód s `self` (patřil do GUI)
- Soubor pojmenovat **live_bot_service.py** (ne "live_bod_service.py")
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ib_insync import IB, BarData, BarDataList, Contract

log = logging.getLogger(__name__)


# ------------------------
# Pomocné pro timeframe/IB
# ------------------------

def _norm_timeframe(tf: str) -> str:
    t = (tf or "").strip().lower().replace("mins", "min").replace("minutes", "min").replace("hours", "hour")
    aliases = {
        "1m": "1 min",
        "5m": "5 min",
        "15m": "15 min",
        "30m": "30 min",
        "60m": "1 hour",
        "1h": "1 hour",
        "4h": "4 hour",
        "1d": "1 day",
        "d": "1 day",
    }
    return aliases.get(t, t)


def _bar_size_setting(tf: str) -> str:
    t = _norm_timeframe(tf)
    mapping = {
        "1 min": "1 min",
        "5 min": "5 mins",
        "15 min": "15 mins",
        "30 min": "30 mins",
        "1 hour": "1 hour",
        "4 hour": "4 hours",
        "1 day": "1 day",
    }
    return mapping.get(t, t)


def _duration_for(bar_size: str) -> str:
    if bar_size in ("1 min", "5 mins", "15 mins", "30 mins"):
        return "2 D"
    if bar_size in ("1 hour", "4 hours"):
        return "1 W"
    return "1 M"


# ------------------------
# Konfigurace featur
# ------------------------

@dataclass
class FeatureConfig:
    ma_fast: int = 9
    ma_slow: int = 21
    atr_len: int = 14


# ------------------------
# Hlavní služba
# ------------------------

class LiveBotService:
    """IBKR live stream: udržuje OHLCV DataFrame, počítá featury a po uzavření baru vyhodnotí signál.

    POZN.: Tahle služba **nemá** žádnou GUI logiku ani `_predict_with_model` – to patří do GUI vrstvy.
    """

    def __init__(
        self,
        ib: IB,
        contract: Contract,
        timeframe: str,
        model_path: str | None = None,
        logger: logging.Logger | None = None,
        log_bars: bool = True,
        rth: bool = False,
        backfill_bars: int = 300,
        buffer_limit: int = 5000,
        feat_cfg: FeatureConfig = FeatureConfig(),
    ) -> None:
        self.ib = ib
        self.contract = contract
        self.timeframe = _norm_timeframe(timeframe)
        self.bar_size = _bar_size_setting(self.timeframe)
        self.duration = _duration_for(self.bar_size)
        self.useRTH = 1 if rth else 0

        # Info
        self.symbol = getattr(contract, "localSymbol", getattr(contract, "symbol", "UNKNOWN"))
        self.logger = logger or logging.getLogger("live_bot")
        self.log_bars = log_bars

        # Data
        self.df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype({"volume": "int64"})
        self.df.index.name = "date"
        self.buffer_limit = buffer_limit

        # Featury
        self.feat_cfg = feat_cfg

        # Model (volitelné)
        self.model_path = model_path
        self.model = None
        self._load_model_if_any()

        # IB subscription
        self.bars: BarDataList | None = None

        # Control
        self._stop_event = threading.Event()
        self._started = False

        # Freshness
        self._last_closed_ts: pd.Timestamp | None = None
        self.fresh_limit_sec = self._fresh_limit_seconds()

    # ------------------------
    # Public API
    # ------------------------
    def run(self) -> None:
        self._started = True
        self._subscribe_bars()
        try:
            while not self._stop_event.is_set():
                self._check_freshness()
                time.sleep(0.5)
        finally:
            self._unsubscribe_bars()
            self._started = False

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------
    # Interní pomocné metody
    # ------------------------
    def _load_model_if_any(self) -> None:
        if not self.model_path:
            return
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            self.logger.info(f"[INFO] Model načten z: {self.model_path}")
        except Exception as e:
            self.logger.exception(f"[WARN] Nepodařilo se načíst model '{self.model_path}': {e}")
            self.model = None

    def _fresh_limit_seconds(self) -> int:
        t = self.timeframe
        if t in ("1 min",):
            return 120
        if t in ("5 min", "5 mins"):
            return 600
        if t in ("15 min", "15 mins"):
            return 1800
        if t in ("30 min", "30 mins"):
            return 3600
        if t in ("1 hour",):
            return 90 * 60
        return 24 * 3600

    def _subscribe_bars(self) -> None:
        self.logger.info(
            f"[INFO] Subscribing {self.symbol} {self.bar_size} duration={self.duration} useRTH={self.useRTH}"
        )
        self.bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr=self.duration,
            barSizeSetting=self.bar_size,
            whatToShow="TRADES",
            useRTH=self.useRTH,
            keepUpToDate=True,
        )

        # Backfill: vše kromě poslední běžící svíčky
        try:
            initial: list[BarData] = list(self.bars)
            if len(initial) > 0:
                for b in initial[:-1]:
                    self._append_closed_bar(b, emit_log=False)
                if len(initial) >= 2:
                    self._last_closed_ts = pd.to_datetime(_to_datetime_like(initial[-2]))
        except Exception as e:
            self.logger.exception(f"[WARN] Nepodařilo se naplnit DF z backfillu: {e}")

        # Callback na aktualizace
        self.bars.updateEvent += self._on_bars_update

    def _unsubscribe_bars(self) -> None:
        try:
            if self.bars is not None:
                self.bars.updateEvent -= self._on_bars_update
                try:
                    self.ib.cancelHistoricalData(self.bars)
                except Exception:
                    pass
        finally:
            self.bars = None

    # ------------------------
    # Handlery
    # ------------------------
    def _on_bars_update(self, bars: BarDataList, hasNewBar: bool) -> None:
        try:
            if hasNewBar and len(bars) >= 2:
                closed = bars[-2]
                self._append_closed_bar(closed, emit_log=True)
                self._last_closed_ts = pd.to_datetime(_to_datetime_like(closed))
                self._evaluate_last_row()
                if len(self.df) > self.buffer_limit:
                    self.df = self.df.iloc[-self.buffer_limit :]
        except Exception as e:
            self.logger.exception(f"[ERROR] _on_bars_update selhalo: {e}")

    # ------------------------
    # Dataframe + featury + signál
    # ------------------------
    def _append_closed_bar(self, b: BarData, emit_log: bool = True) -> None:
        ts = pd.to_datetime(_to_datetime_like(b))
        row = pd.Series(
            {
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume or 0),
            },
            name=ts,
        )
        self.df.loc[ts, ["open", "high", "low", "close", "volume"]] = row

        if self.log_bars and emit_log:
            self.logger.info(
                f"[BAR] {self.symbol} {self.timeframe} {ts} "
                f"O={row['open']:.4f} H={row['high']:.4f} L={row['low']:.4f} "
                f"C={row['close']:.4f} V={int(row['volume'])}"
            )
        self._recalc_features_tail()

    def _recalc_features_tail(self) -> None:
        if len(self.df) == 0:
            return
        fc = self.feat_cfg
        self.df["ma_fast"] = self.df["close"].rolling(fc.ma_fast, min_periods=1).mean()
        self.df["ma_slow"] = self.df["close"].rolling(fc.ma_slow, min_periods=1).mean()
        h_l = self.df["high"] - self.df["low"]
        h_pc = (self.df["high"] - self.df["close"].shift(1)).abs()
        l_pc = (self.df["low"] - self.df["close"].shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        self.df["atr"] = tr.rolling(fc.atr_len, min_periods=1).mean()

    def _evaluate_last_row(self) -> None:
        if len(self.df) < 2:
            return
        last = self.df.iloc[-1]
        signal = None
        if self.model is not None:
            try:
                features = last[["close", "ma_fast", "ma_slow", "atr"]].to_frame().T.fillna(method="ffill").fillna(0.0)
                if hasattr(self.model, "predict_proba"):
                    proba_vec = self.model.predict_proba(features)[0]
                    classes = list(getattr(self.model, "classes_", []))
                    # najdi indexy LONG/SHORT (stringové nebo numerické mapované na LONG/SHORT)
                    idx = {str(c).upper(): i for i,c in enumerate(classes)}
                    if "LONG" in idx and "SHORT" in idx:
                        p_long = float(proba_vec[idx["LONG"]]); p_short = float(proba_vec[idx["SHORT"]])
                    else:
                        # fallback – když nejsou stringy
                        p_long = float(proba_vec[1]); p_short = float(proba_vec[0])
                    signal = "LONG" if p_long >= 0.5 else "SHORT"
                    self.logger.info(f"[SIG] classes={classes} proba={proba_vec.tolist()} -> pL={p_long:.3f} pS={p_short:.3f} -> {signal}")
                else:
                    pred = int(self.model.predict(features)[0])
                    signal = "LONG" if pred == 1 else "SHORT"
                    self.logger.info(f"[SIG] model pred={pred} -> {signal}")
            except Exception as e:
                self.logger.exception(f"[WARN] Model selhal při predikci: {e}")
        else:
            if np.isnan(last["ma_fast"]) or np.isnan(last["ma_slow"]):
                return
            signal = "LONG" if last["ma_fast"] > last["ma_slow"] else "SHORT"
            self.logger.info(f"[SIG] ma_fast={last['ma_fast']:.2f} ma_slow={last['ma_slow']:.2f} -> {signal}")
        # TODO: zde můžeš vyvolat callback do UI

    # ------------------------
    # Watchdog
    # ------------------------
    def _check_freshness(self) -> None:
        if self._last_closed_ts is None:
            return
        delta = pd.Timestamp.utcnow() - self._last_closed_ts.tz_localize(None)
        if delta.total_seconds() > self.fresh_limit_sec:
            self.logger.warning(
                f"[WARN] Poslední uzavřený bar je starý {int(delta.total_seconds())} s "
                f"(limit {self.fresh_limit_sec} s). Zvaž reconnect."
            )


# ------------------------
# Utility
# ------------------------

def _to_datetime_like(b: BarData):
    d = b.date
    try:
        return pd.to_datetime(d)
    except Exception:
        return d
