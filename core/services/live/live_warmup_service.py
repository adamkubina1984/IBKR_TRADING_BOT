"""
LiveWarmupService – drop‑in obálka pro stávající LiveBotService s warm‑up režimem,
diagnostikou mapování tříd a bezpečnostními pojistkami.

Použití (pseudokód):

    from live_warmup_service import LiveWarmupService

    warm = LiveWarmupService(
        base_service=LiveBotService(...),     # tvůj stávající servis; musí mít metody: predict(features)->(signal, proba, classes), execute(signal, bar), featurize(), fetch_history(), log
        threshold=0.50,
        warmup_bars=300,
        min_sim_trades=20,
        start_sharpe=0.0,
        max_dd=20.0,
        diag_first_n=10,
    )

    warm.start(symbol="GC1!", exchange="COMEX", timeframe="1 hour")

Požadavky na `base_service` (tvůj stávající LiveBotService nebo kompatibilní třída):
- musí poskytovat:
    - .log (logger se .info/.warn/.error)
    - .featurize(bars) -> features (dict/np.array)  # použito i na historická data
    - .predict(features) -> (signal: str, proba: list[float], classes: list[str])
        * signal v {"LONG","SHORT","FLAT"}
        * classes v pořadí odpovídajícím proba
    - .execute(signal, bar) -> realized_pnl (float)  # v live módu
    - .fetch_history(symbol, exchange, timeframe, n_bars: int) -> list[Bar]
        * Bar musí mít alespoň: .open, .high, .low, .close, .time

Pozn.: Pokud nemáš přesně tyto metody, uprav jednoduché adaptéry dole (viz třída SimpleAdapterTemplate).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ------------------------------ Pomocné výpočty metrik ------------------------------

def _safe_mean(xs: list[float]) -> float:
    return sum(xs)/len(xs) if xs else 0.0


def _rolling_sharpe(pnls: list[float]) -> float:
    """Sharpe na základě PnL (bez annualizace, jen signálová jednotka)."""
    if not pnls:
        return 0.0
    mu = _safe_mean(pnls)
    # sample std (ddof=1) s ochranou proti nulové varianci
    if len(pnls) >= 2:
        var = sum((x - mu) ** 2 for x in pnls) / (len(pnls) - 1)
    else:
        var = 0.0
    std = math.sqrt(max(var, 1e-12))
    return mu / std if std > 0 else 0.0


def _max_drawdown_from_equity(eqs: list[float]) -> float:
    """Max DD z průběhu equity (v absolutních jednotkách)."""
    if not eqs:
        return 0.0
    peak = eqs[0]
    max_dd = 0.0
    for x in eqs:
        peak = max(peak, x)
        dd = peak - x
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ------------------------------ Třída s warm‑up logikou ------------------------------

@dataclass
class WarmupConfig:
    threshold: float = 0.50
    warmup_bars: int = 300
    min_sim_trades: int = 20
    start_sharpe: float = 0.0
    max_dd: float = 20.0
    diag_first_n: int = 10
    # Pokud True, po warm-upu přepni do LIVE bez ohledu na metriky.
    force_live_after_warmup: bool = False
    # Počet obchodů používaných pro rolling metriky v warm‑upu
    roll_window: int = 20


class LiveWarmupService:
    """Obálka nad stávajícím live service pro bezpečný start.

    Stavový automat: IDLE -> WARMUP -> LIVE.
    Během WARMUP se neobchoduje: provádí se "paper" simulace PnL, počítá se rolling Sharpe
    a rolling max DD. Do LIVE se přejde po splnění podmínek.
    """

    def __init__(self, base_service, config: WarmupConfig | None = None, **kwargs):
        # config lze zadat buď objektem, nebo přes kwargs (kompatibilita s GUI)
        if config is None:
            config = WarmupConfig(**kwargs)
        self.cfg = config

        self.base = base_service
        self.log = getattr(base_service, "log", None)
        if self.log is None:
            raise ValueError("base_service musí mít .log (logger s metodami info/warn/error)")

        # stav
        self.state = "IDLE"  # IDLE | WARMUP | LIVE
        self._diag_printed = 0
        self._bars_seen = 0
        self._paper_equity = 0.0
        self._equity_series: list[float] = []
        self._rolling_pnl: deque[float] = deque(maxlen=self.cfg.roll_window)
        self._sim_trades = 0

        # Informativní
        self.symbol = None
        self.exchange = None
        self.timeframe = None

    # ------------------------------ Veřejné API ------------------------------

    def start(self, symbol: str, exchange: str, timeframe: str) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe

        self.log.info("[INFO] Start sezení…")
        self._prefetch_and_warmup()
        if self.state == "LIVE":
            self.log.info("[INFO] Warm-up OK → přecházím do LIVE režimu.")
        else:
            self.log.warn("[WARN] Warm-up podmínky nesplněny, setrvávám v WARMUP a neobchoduji.")

    def on_new_bar(self, bar) -> None:
        """Zavolej na každý nový bar. Podle stavu buď simuluje (WARMUP) nebo obchoduje (LIVE)."""
        self._bars_seen += 1

        # Feature engineering
        features = self.base.featurize_recent()
        signal, proba, classes = self.base.predict(features)

        # Diagnostika mapování tříd u prvních barů
        if self._diag_printed < self.cfg.diag_first_n:
            self.log.info(
                f"[DIAG] classes={classes} proba={list(map(lambda x: round(float(x), 4), proba))} chosen={signal} thr={self.cfg.threshold}"
            )
            self._diag_printed += 1

        if self.state != "LIVE":
            # WARMUP – jen simulace PnL (paper)
            pnl = self._simulate_paper(signal, bar)
            self._after_paper_trade(pnl)
            sharpe, maxdd = self._current_metrics()
            self.log.info(
                f"[WARMUP] bars={self._bars_seen} sim_trades={self._sim_trades} sharpe={sharpe:.2f} maxDD={maxdd:.1f}"
            )
            if self._can_go_live(sharpe, maxdd):
                self.state = "LIVE"
                self.log.info(
                    f"[INFO] Warm-up splněn → LIVE (sharpe={sharpe:.2f}, maxDD={maxdd:.1f}, trades={self._sim_trades})"
                )
            return

        # LIVE – skutečná exekuce
        realized = self.base.execute(signal, bar)
        # můžeš sem přidat živé safeguarding prahy (pauza při zhoršení metrik)

    # ------------------------------ Interní logika ------------------------------

    def _prefetch_and_warmup(self) -> None:
        """Nahrání historie a provedení warm-up simulace (bez exekuce)."""
        need = max(self.cfg.warmup_bars, self._min_feature_lookback())
        bars = self.base.fetch_history(self.symbol, self.exchange, self.timeframe, n_bars=need)
        if not bars:
            self.log.error("[ERROR] fetch_history vrátil prázdný seznam – warm-up nelze provést.")
            return

        self.state = "WARMUP"
        self._bars_seen = 0
        self._paper_equity = 0.0
        self._equity_series.clear()
        self._rolling_pnl.clear()
        self._sim_trades = 0

        for bar in bars:
            self._bars_seen += 1
            features = self.base.featurize_until(bar)
            signal, proba, classes = self.base.predict(features)

            if self._diag_printed < self.cfg.diag_first_n:
                self.log.info(
                    f"[DIAG] classes={classes} proba={list(map(lambda x: round(float(x), 4), proba))} chosen={signal} thr={self.cfg.threshold}"
                )
                self._diag_printed += 1

            pnl = self._simulate_paper(signal, bar)
            self._after_paper_trade(pnl)

        sharpe, maxdd = self._current_metrics()
        self.log.info(
            f"[WARMUP-END] bars={self._bars_seen} sim_trades={self._sim_trades} sharpe={sharpe:.2f} maxDD={maxdd:.1f}"
        )
        if self.cfg.force_live_after_warmup:
            self.state = "LIVE"
            self.log.info("[INFO] force_live_after_warmup=True → přepínám do LIVE režimu.")
            return
        if self._can_go_live(sharpe, maxdd):
            self.state = "LIVE"

    def _simulate_paper(self, signal: str, bar) -> float:
        """Jednoduchá paper-simulace: pokud FLAT, PnL=0. Jinak stub/volání base paper simulace.
        Uprav dle svých pravidel (SL/TP/ATR trailing)."""
        if signal == "FLAT":
            pnl = 0.0
        else:
            # Zkus použít metodu base_service, pokud ji poskytuje
            sim = getattr(self.base, "simulate_trade", None)
            if callable(sim):
                pnl = float(sim(signal, bar))
            else:
                # fallback: approximace – vstup na close, výstup na tomtéž baru (0 PnL)
                pnl = 0.0
        return pnl

    def _after_paper_trade(self, pnl: float) -> None:
        if pnl != 0.0:
            self._sim_trades += 1
        self._paper_equity += pnl
        self._equity_series.append(self._paper_equity)
        self._rolling_pnl.append(pnl)

    def _current_metrics(self) -> tuple[float, float]:
        sharpe = _rolling_sharpe(list(self._rolling_pnl))
        maxdd = _max_drawdown_from_equity(self._equity_series[-self.cfg.roll_window :])
        return sharpe, maxdd

    def _can_go_live(self, sharpe: float, maxdd: float) -> bool:
        if self._sim_trades < self.cfg.min_sim_trades:
            return False
        if sharpe < self.cfg.start_sharpe:
            return False
        if maxdd > self.cfg.max_dd:
            return False
        return True

    def _min_feature_lookback(self) -> int:
        """Pokud base_service umí sdělit minimální lookback featur, použij ho; jinak rozumný default."""
        get_lb = getattr(self.base, "min_feature_lookback", None)
        if callable(get_lb):
            try:
                return int(get_lb())
            except Exception:
                pass
        # fallback default
        return max(self.cfg.warmup_bars // 3, 100)


# ------------------------------ Adaptér – vzor, když se signatury liší ------------------------------

class SimpleAdapterTemplate:
    """Pokud tvůj LiveBotService nemá přesně vyžadované metody, obal ho tímto adaptérem
    a doplň TODO části podle tvého kódu. Tenhle kus slouží jako šablona – klidně smaž.
    """

    def __init__(self, raw, logger):
        self.raw = raw
        self.log = logger

    # === Historie ===
    def fetch_history(self, symbol, exchange, timeframe, n_bars: int):
        # TODO: deleguj na svůj datafeed
        return []

    # === Featurizace ===
    def featurize_until(self, bar):
        # TODO: vrať featury spočtené po zahrnutí `bar`
        return {}

    def featurize_recent(self):
        # TODO: vrať featury pro nejčerstvější stav okna
        return {}

    def min_feature_lookback(self) -> int:
        # TODO: vrať max lookback z indikátorů, např. načtené z configu
        return 300

    # === Model ===
    def predict(self, features) -> tuple[str, list[float], list[str]]:
        """
        Vrací (signal, proba, classes) robustně podle model.classes_.
        - do predict_proba() vždy posílá 1-řádkový DataFrame se jmény sloupců (kvůli SimpleImputeru ap.)
        - p(LONG)/p(SHORT) bere podle classes_, ne podle indexu 0/1
        - pokud máš číselné classes_ (0/1), použije mapu {1:'LONG', 0:'SHORT'} (nebo self.class_to_dir)
        """
        # -- 1) připrav Xrow jako 1-řádkový DataFrame se jmény sloupců
        if isinstance(features, pd.DataFrame):
            Xrow = features.tail(1)
        elif isinstance(features, pd.Series):
            Xrow = features.to_frame().T
        else:
            arr = np.asarray(features)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            cols = []
            if hasattr(self, "expected_features") and isinstance(self.expected_features, (list, tuple)):
                cols = list(self.expected_features)
            elif hasattr(self, "model") and hasattr(self.model, "feature_names_in_"):
                cols = list(self.model.feature_names_in_)
            Xrow = pd.DataFrame(arr, columns=cols if cols and len(cols) == arr.shape[1] else None)

        # -- 2) predikce pravděpodobností
        proba_vec = self.model.predict_proba(Xrow)[0]
        classes = list(getattr(self.model, "classes_", []))

        # -- 3) získej p(LONG) a p(SHORT) podle classes_
        p_long = p_short = None

        # a) stringové classes: ["LONG","SHORT"] apod.
        if classes:
            lut = {str(c).upper(): i for i, c in enumerate(classes)}
            if "LONG" in lut:
                p_long = float(proba_vec[lut["LONG"]])
            if "SHORT" in lut:
                p_short = float(proba_vec[lut["SHORT"]])

        # b) numerické classes: [0,1] s mapou 0→SHORT, 1→LONG (lze přepsat self.class_to_dir)
        if p_long is None or p_short is None:
            label_map = getattr(self, "class_to_dir", {1: "LONG", 0: "SHORT"})
            idx_long = next((i for i, c in enumerate(classes)
                             if str(label_map.get(int(c), "")).upper() == "LONG"), None) if classes else 1
            idx_short = next((i for i, c in enumerate(classes)
                              if str(label_map.get(int(c), "")).upper() == "SHORT"), None) if classes else 0
            if idx_long is not None:
                p_long = float(proba_vec[idx_long])
            if idx_short is not None:
                p_short = float(proba_vec[idx_short])

        # c) nouzové doplnění
        if p_long is None and p_short is not None:
            p_long = 1.0 - p_short
        if p_short is None and p_long is not None:
            p_short = 1.0 - p_long
        if p_long is None or p_short is None:
            # poslední fallback dle pořadí
            p_long = float(proba_vec[1] if len(proba_vec) > 1 else proba_vec[0])
            p_short = float(proba_vec[0])

        # -- 4) rozhodnutí se symetrickým prahem
        thr = float(getattr(self, "entry_thr", 0.5))
        if p_long >= thr and p_long > p_short:
            signal = "LONG"
        elif p_short >= thr and p_short > p_long:
            signal = "SHORT"
        else:
            signal = "FLAT"

        # vrať skutečné classes (pro logy/diag)
        return signal, [p_long, p_short], (classes if classes else ["SHORT", "LONG"])


    # === Live exekuce ===
    def execute(self, signal: str, bar) -> float:
        # TODO: provést reálnou exekuci u tvého brokera podle signálu
        return 0.0

    # === Paper simulace (volitelné) ===
    def simulate_trade(self, signal: str, bar) -> float:
        # TODO: simulace pravidel (TP/SL/trailing)
        return 0.0
