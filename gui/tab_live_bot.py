# ==============================================
# Záložka 4) Live trading bot – TradingView only
# + Ensemble AND (MA ∧ Model) + pojmenované vrstvy L0/L1/L2
# + Auto-align featur na expected_features/feature_names_in_ (vč. 'average')
# (CLEAN verze – odstraněn legacy single-model kód)
# ==============================================

from __future__ import annotations

import json as jsonlib
import os
import smtplib
import threading
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from PySide6.QtCore import QThread, QTimer, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ibkr_trading_bot.core.config.presets import PRESETS_BY_TF

try:
    from PySide6.QtCore import QUrl
    from PySide6.QtMultimedia import QSoundEffect
except Exception:
    QSoundEffect = None
    QUrl = None

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Warm-up service
try:
    from ibkr_trading_bot.core.services.live.live_warmup_service import LiveWarmupService, WarmupConfig
except Exception as e:
    raise ImportError(f"Nelze importovat LiveWarmupService: {e}")


# TradingView klient (různé fallback importy)
try:
    from ibkr_trading_bot.core.datasource.tradingview_client import TradingViewClient
except ModuleNotFoundError:
    try:
        from ibkr_trading_bot.core.data_sources.tradingview_client import TradingViewClient
    except ModuleNotFoundError:
        try:
            from core.datasource.tradingview_client import TradingViewClient
        except ModuleNotFoundError:
            from core.data_sources.tradingview_client import TradingViewClient

# Logger
try:
    from ibkr_trading_bot.core.utils.loggers import get_logger
except Exception:
    def get_logger(name: str):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

DEFAULT_MODEL_DIR = r"C:\Users\adamk\Můj disk\Trader\ibkr_trading_bot\model_outputs"

def _make_proxy_target_from_df(df):
    """
    Vytvoří 'y_proxy' z historických dat: LONG pokud příští close > aktuální close, jinak SHORT.
    Vrací numpy pole s hodnotami {"LONG","SHORT"} délky (len(df)-1) a indexy pro X[:-1].
    """
    import numpy as np
    closes = df["close"].to_numpy(dtype=float)
    # posun o -1: predikujeme pohyb následující svíčky
    up = closes[1:] > closes[:-1]
    y_proxy = np.where(up, "LONG", "SHORT")
    return y_proxy

def _auto_detect_label_polarity(model, X_df, raw_df, max_samples=200):
    """
    Zjistí, zda je 0=LONG/1=SHORT nebo 0=SHORT/1=LONG porovnáním s proxy cílem z cen.
    Vrací mapu {0:"LONG"/"SHORT", 1:"LONG"/"SHORT"}.
    """
    import numpy as np

    if X_df is None or len(X_df) < 5 or raw_df is None or len(raw_df) < 5:
        return {0: "SHORT", 1: "LONG"}

    # Poslední N vzorků
    X = X_df.tail(max_samples).copy()

    # Proxy cíl (o 1 kratší)
    raw_tail = raw_df.tail(len(X) + 1).copy()
    y_proxy = _make_proxy_target_from_df(raw_tail)  # len = len(X)

    # Hard align: když to přesto nesedí, ořež delší tak, aby délky byly stejné
    nX = len(X)
    ny = len(y_proxy)
    if nX > ny:
        X = X.iloc[-ny:, :].copy()
        nX = ny
    elif ny > nX:
        y_proxy = y_proxy[-nX:]

    # Predikce
    proba = model.predict_proba(X.to_numpy(dtype=float))
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != proba.shape[1] or len(proba) != len(y_proxy):
        return {0: "SHORT", 1: "LONG"}

    y_hat_idx = np.argmax(proba, axis=1)

    map_A = {0: "SHORT", 1: "LONG"}  # default
    map_B = {0: "LONG", 1: "SHORT"}  # inverze

    def acc_for_map(m):
        lab = np.array([m.get(int(classes[i]), "SHORT") for i in y_hat_idx], dtype=object)
        return float((lab == y_proxy).mean()) if len(lab) == len(y_proxy) else 0.0

    acc_A = acc_for_map(map_A)
    acc_B = acc_for_map(map_B)
    return map_A if acc_A >= acc_B else map_B


# --- robustní extrakce proba podle classes_ ---
def _extract_long_short_proba(model, df_row, label_map: dict | None = None):
    """
    Vrátí (p_long, p_short, classes_list, raw_proba_list) robustně podle model.classes_.
    df_row je 1řádkový DataFrame ve správném pořadí featur.
    label_map: např. {0: "SHORT", 1: "LONG"} – použije se pro číselné classes_.
    Default (bez label_map) je bezpečný {0:"SHORT", 1:"LONG"}.
    """
    # 1) Skutečný výpočet proba
    try:
        proba = model.predict_proba(df_row)[0]
    except Exception:
        proba = model.predict_proba(np.asarray(df_row, dtype=float))[0]

    classes = getattr(model, "classes_", None)

    # 2) Výchozí bezpečná mapa pro numerické třídy (pokud label_map není dána)
    #    - v praxi většina modelů: 0=SHORT, 1=LONG
    if label_map is None:
        label_map = {0: "SHORT", 1: "LONG"}

    p_long = p_short = None

    # 3) Přímo textové classes_ (["LONG","SHORT"] apod.)
    if classes is not None and any(isinstance(c, str) for c in classes):
        lut = {str(c).upper(): i for i, c in enumerate(classes)}
        if "LONG" in lut:
            p_long = float(proba[lut["LONG"]])
        if "SHORT" in lut:
            p_short = float(proba[lut["SHORT"]])

    # 4) Numerické classes_ s mapou
    if (p_long is None or p_short is None) and classes is not None:
        idx_long = next((i for i, c in enumerate(classes)
                         if str(label_map.get(int(c), "")).upper() == "LONG"), None)
        idx_short = next((i for i, c in enumerate(classes)
                          if str(label_map.get(int(c), "")).upper() == "SHORT"), None)
        if idx_long is not None:
            p_long = float(proba[idx_long])
        if idx_short is not None:
            p_short = float(proba[idx_short])

    # 5) Nouzová doplnění
    if p_long is None and p_short is not None:
        p_long = 1.0 - p_short
    if p_short is None and p_long is not None:
        p_short = 1.0 - p_long

    # 6) Poslední fallback – nehádej, ale vezmi maximum a druhou jako 1-p
    if p_long is None or p_short is None:
        # Pokud neznám pořadí, rozhodni podle maxima
        i_max = int(np.argmax(proba))
        p_max = float(proba[i_max])
        # Připusť, že i_max může znamenat LONG nebo SHORT – rozhodni podle label_map / classes
        dir_max = None
        if classes is not None and 0 <= i_max < len(classes):
            cls = classes[i_max]
            dir_max = str(label_map.get(int(cls), cls)).upper()
        # rozdělení
        if dir_max == "LONG":
            p_long, p_short = p_max, 1.0 - p_max
        elif dir_max == "SHORT":
            p_long, p_short = 1.0 - p_max, p_max
        else:
            # když fakt netuším: udrž symetrii
            p_long, p_short = p_max, 1.0 - p_max

    return float(p_long), float(p_short), (list(classes) if classes is not None else None), proba


# ==============================================
# Konfigurace záložky Live
# ==============================================
@dataclass
class LiveConfig:
    symbol: str = "GOLD"
    exchange: str = "TVC"
    mode: str = "live"
    bar_size: str = "1 hour"
    sensitivity: float = 0.5      # confidence threshold (0..1)
    dry_run: bool = True
    max_fresh_age_min: int = 5
    max_bars_buffer: int = 100
    use_ma_only: bool = False
    use_and_ensemble: bool = True  # MA ∧ Model
    alert_on_flip: bool = True
    alert_sound: str | None = r"C:\Users\adamk\Můj disk\Trader\ibkr_trading_bot\gui\assets\alert.wav"
    alert_cooldown_s: int = 5

    # E-mail
    alert_email_enabled: bool = (os.getenv("ALERT_EMAIL_ENABLED", "0").lower() in ("1","true","yes"))
    alert_email_to: str | None = os.getenv("ALERT_EMAIL_TO")
    smtp_host: str | None = os.getenv("SMTP_HOST")
    smtp_port: int = int(os.getenv("SMTP_PORT", "465"))
    smtp_user: str | None = os.getenv("SMTP_USER")
    smtp_password: str | None = os.getenv("SMTP_PASS")
    smtp_use_ssl: bool = os.getenv("SMTP_USE_SSL", "1").lower() not in ("0","false","no")
    smtp_from: str | None = os.getenv("SMTP_FROM")

    entry_thr: float = 0.60
    exit_thr: float = 0.45
    rounds_enabled: bool = False

# ==============================================
# TV Worker – polling posledních uzavřených barů
# ==============================================
class TVWorker(QThread):
    statusChanged = Signal(str)
    error = Signal(str)
    barClosed = Signal(dict)

    def __init__(self, cfg: LiveConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self._stop = False
        self._last_ns: int | None = None
        self.tv = TradingViewClient(
            username=os.getenv("TV_USERNAME"),
            password=os.getenv("TV_PASSWORD")
        )

    def stop(self):
        self._stop = True

    def _poll_interval_s(self) -> int:
        return {
            "5 min": 10,
            "15 min": 30,
            "30 min": 45,
            "1 hour": 60,
        }.get((self.cfg.bar_size or "1 hour"), 30)

    def run(self):
        try:
            self.statusChanged.emit("Connected")
            stale_count = 0
            while not self._stop:
                tf_label = (self.cfg.bar_size or "1 hour").replace("mins", "min")
                exchange = (getattr(self.cfg, "exchange", None) or "COMEX")
                df = self.tv.get_history(self.cfg.symbol, exchange, tf_label, limit=2)
                if df is not None and not df.empty:
                    last = df.iloc[-1]
                    ts = pd.to_datetime(last["time"], utc=True, errors="coerce")
                    if pd.isna(ts):
                        self.msleep(self._poll_interval_s() * 1000); continue
                    ts_ns = int(ts.value)
                    if ts_ns != self._last_ns:
                        self._last_ns = ts_ns
                        stale_count = 0
                        self.barClosed.emit({
                            "time": str(ts),
                            "open": float(last.get("open", 0)),
                            "high": float(last.get("high", 0)),
                            "low":  float(last.get("low",  0)),
                            "close":float(last.get("close",0)),
                            "volume": float(last.get("volume", 0) or 0),
                        })
                    else:
                        stale_count += 1
                else:
                    stale_count += 1

                if stale_count >= 30:
                    try:
                        self.statusChanged.emit("Reconnecting…")
                        self.tv = TradingViewClient(
                            username=os.getenv("TV_USERNAME"),
                            password=os.getenv("TV_PASSWORD")
                        )
                    except Exception:
                        pass
                    stale_count = 0
                    self._last_ns = None

                self.msleep(self._poll_interval_s() * 1000)
        except Exception as e:
            self.error.emit(f"TV worker error: {e}")
            self.msleep(2000)
        finally:
            self.statusChanged.emit("Disconnected")


# ==============================================
# Adaptér pro LiveWarmupService – napojí GUI + model
# ==============================================
class _WarmAdapter:
    """Adaptér, který napojí LiveWarmupService na LiveBotWidget."""
    def __init__(self, widget: LiveBotWidget):
        self.w = widget
        self.log = widget.logger
        self._hist_df = pd.DataFrame(columns=["date","open","high","low","close","volume"])
        self._pos = 0
        self._entry = None
        self.models: list[dict] = []   # list členů ensemble [{predictor, classes, exp_feats}]

    def fetch_history(self, symbol: str, exchange: str, timeframe: str, n_bars: int):
        tv_client = getattr(getattr(self.w, "worker", None), "tv", None) or TradingViewClient(
            username=os.getenv("TV_USERNAME"), password=os.getenv("TV_PASSWORD")
        )
        tf_label = (timeframe or "1 hour").replace("mins", "min")
        df = tv_client.get_history(symbol, exchange, tf_label, limit=int(n_bars))
        if df is None or df.empty:
            return []
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")

        out = []
        for _, r in df.iterrows():
            out.append({
                "time": r["time"],
                "open": float(r.get("open", np.nan)),
                "high": float(r.get("high", np.nan)),
                "low":  float(r.get("low",  np.nan)),
                "close":float(r.get("close",np.nan)),
                "volume": float(r.get("volume", 0) or 0),
            })
        self._hist_df = pd.DataFrame({
            "date":   df["time"].to_numpy(),
            "open":   df["open"].astype(float).to_numpy(),
            "high":   df["high"].astype(float).to_numpy(),
            "low":    df["low"].astype(float).to_numpy(),
            "close":  df["close"].astype(float).to_numpy(),
            "volume": df["volume"].astype(float).to_numpy(),
        })
        return out

    def featurize_until(self, bar: dict[str, Any]):
        row = pd.DataFrame([{
            "date":   pd.to_datetime(bar["time"], utc=True),
            "open":   float(bar["open"]),
            "high":   float(bar["high"]),
            "low":    float(bar["low"]),
            "close":  float(bar["close"]),
            "volume": float(bar.get("volume", 0) or 0),
        }])
        self._hist_df = pd.concat([self._hist_df, row], ignore_index=True)
        feat = self.w._compute_indicators(self._hist_df.rename(columns={"date": "date"}))
        last = feat.iloc[[-1]].copy()
        X = self.w._align_features_to_model(last)

        if getattr(self.w.config, "use_ma_only", False):
            for col in ("ma_fast", "ma_slow"):
                if col not in X.columns and col in last.columns:
                    X[col] = float(last[col].iloc[-1])

        return X

    def featurize_recent(self):
        return self.w._build_features_from_live()

    def min_feature_lookback(self) -> int:
        return 100

    @staticmethod
    def _ma_sig_from_features(features: pd.DataFrame) -> str | None:
        try:
            ma_f = float(features.get("ma_fast").iloc[-1])
            ma_s = float(features.get("ma_slow").iloc[-1])
        except Exception:
            return None
        if np.isfinite(ma_f) and np.isfinite(ma_s):
            if ma_f > ma_s:  return "LONG"
            if ma_f < ma_s:  return "SHORT"
        return None

    def _classes(self) -> list[str]:
        return ["LONG", "SHORT"]

    # ---- Predikce pro WarmupService (vrací L2_AND pokud zapnut AND, jinak model/MA podle nastavení) ----
    def predict(self, features: pd.DataFrame):
        # UI práh pro vstup/výstup (hystereze) – stejný jako v GUI
        thr_ui = float(self.w.dsb_sensitivity.value())

        classes = ["LONG", "SHORT"]

        # L0: MA
        l0 = self._ma_sig_from_features(features) or "FLAT"

        # MA-only režim -> vrať rovnou MA
        if getattr(self.w.config, "use_ma_only", False):
            probs = [1.0, 0.0] if l0 == "LONG" else [0.0, 1.0] if l0 == "SHORT" else [0.5, 0.5]
            return l0, probs, classes

        # Když není žádný model -> vrať MA
        if not self.w.models:
            probs = [1.0, 0.0] if l0 == "LONG" else [0.0, 1.0] if l0 == "SHORT" else [0.5, 0.5]
            return l0, probs, classes

        # L1: čistý AND přes modely (uvnitř bez prahu – jen směr a conf_min)
        label, conf_min, dirs, confs = self.w._predict_one_label_AND(features, thr=0.0)
        l1 = "LONG" if label == +1 else "SHORT" if label == -1 else "FLAT"

        # L2: (volitelně) MA ∧ L1 + aplikace prahu z UI (thr_ui) – stejná politika jako v _rescore_all
        if getattr(self.w.config, "use_and_ensemble", True):
            # nejdřív jen směrové „proposal“
            if l0 == "FLAT":
                proposal = l1 if (l1 in ("LONG", "SHORT")) else None
            else:
                proposal = l1 if (l1 == l0) else None
        else:
            proposal = l1 if (l1 in ("LONG", "SHORT")) else None

        # Hystereze (vstup/výstup) – stejně jako v _rescore_all
        final = None
        try:
            # k featurám se v adaptér módu nedostaneme k celé historii – použij poslední bar z features
            close = float(features.get("close").iloc[-1])
            atr   = float(features.get("atr").iloc[-1]) if "atr" in features.columns else np.nan
        except Exception:
            close, atr = np.nan, np.nan

        if self.w._live_pos == 0:
            if proposal in ("LONG","SHORT") and conf_min >= thr_ui and not self.w._near_round_level(close, atr):
                final = proposal
        else:
            want_dir = "LONG" if self.w._live_pos > 0 else "SHORT"
            if proposal == want_dir and conf_min >= max(0.0, thr_ui - 0.05):
                final = want_dir

        # log (už neodkazuje na neexistující 'thr' / 'thr_and')
        self.log.info(
            "[LAYER] L0_MA=%s | %s -> L1_AND=%s | thr_ui=%.2f -> L2=%s",
            l0,
            " ".join([f"L1{chr(97+i)}={d}({confs[i]:.2f})" for i,d in enumerate(dirs)]),
            l1, thr_ui, (final or "FLAT")
        )

        probs = [conf_min, 1.0 - conf_min] if final in ("LONG","SHORT") else [0.5, 0.5]
        return final, probs, classes



    def execute(self, signal: str, bar: dict[str, Any]) -> float:
        self.log.info(f"[LIVE] signal={signal} time={bar.get('time')} close={bar.get('close')}")
        return 0.0

    def simulate_trade(self, signal: str, bar: dict[str, Any]) -> float:
        px = float(bar.get("close", 0.0))
        pnl = 0.0
        def close_long(entry, price):  return price - entry
        def close_short(entry, price): return entry - price
        if signal == "LONG":
            if self._pos == -1 and self._entry is not None: pnl = close_short(self._entry, px)
            if self._pos != 1: self._pos, self._entry = 1, px
        elif signal == "SHORT":
            if self._pos == 1 and self._entry is not None: pnl = close_long(self._entry, px)
            if self._pos != -1: self._pos, self._entry = -1, px
        else:
            if self._pos == 1 and self._entry is not None: pnl = close_long(self._entry, px)
            elif self._pos == -1 and self._entry is not None: pnl = close_short(self._entry, px)
            self._pos, self._entry = 0, None
        return float(pnl)


# ==============================================
# Hlavní widget
# ==============================================
class LiveBotWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, config: LiveConfig | None = None) -> None:
        super().__init__(parent)
        self.logger = get_logger("live_bot.gui")
        self.config = config or LiveConfig()
        self.model = None                          # používá se jen pro feature_names_in_
        self.models: list[dict[str, Any]] = []     # členové ensemble
        self.class_to_dir = {1: "LONG", 0: "SHORT"}  # lze přepsat z meta
        self._diag_once = {}
        self.label_map_from_meta = False
        self.model_expected_features: list[str] | None = None
        self.worker: TVWorker | None = None
        self.warm: LiveWarmupService | None = None

        self.live_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self._bars: list[dict[str, Any]] = []
        self._bar_index: dict[int, int] = {}
        self._last_arrival_utc: pd.Timestamp | None = None

        self._build_ui()
        self._wire_basic_logic()
        self._last_alert_sig: str | None = None
        self._last_alert_bar_key: int | None = None
        self._se = None

        self._live_pos = 0       # -1 short, 0 flat, +1 long
        self._live_entry_px = None
        self._curr_entry_thr = self.config.entry_thr
        self._curr_exit_thr = self.config.exit_thr
        self._rounds = {"grid": [], "tol_atr": 0.0}
        self.class_to_dir = {0: "SHORT", 1: "LONG"}

        if QSoundEffect is not None and QUrl is not None and self.config.alert_sound:
            try:
                self._se = QSoundEffect(self)
                self._se.setSource(QUrl.fromLocalFile(self.config.alert_sound))
                self._se.setLoopCount(1)
                self._se.setVolume(0.9)
            except Exception:
                self._se = None
        self._last_alert_bar_ns = None
        self._last_beep_time = None
        self._last_signal = None

    def _apply_tf_presets(self):
        tf = self.cmb_interval.currentText()
        p = PRESETS_BY_TF.get(tf, PRESETS_BY_TF["1 hour"])

        # UI práh (hystereze)
        s = float(self.dsb_sensitivity.value())
        self._curr_entry_thr = s
        self._curr_exit_thr  = max(0.0, min(s - 0.05, s))

        # presetované kulatá čísla
        self._rounds = {"grid": [], "tol_atr": 0.0}

        # ---- přidej odtud: zmírnění pro 5 min ----
        if tf == "5 min":
            # buď úplně vypnout
            # self._rounds = {"grid": [], "tol_atr": 0.0}

            # ...nebo jen zmírnit:
            self._rounds["tol_atr"] = 0.05   # bylo 0.15, což je moc přísné pro 5m
            # volitelně zmenši grid, ať to nebere každou "pětku":
            self._rounds["grid"] = [1]       # dříve [1, 5]
        # ---- až sem ----

        self._append_log(
        f"[PRESET] TF='{tf}' | entry_thr={self._curr_entry_thr:.2f} "
        f"exit_thr={self._curr_exit_thr:.2f} | rounds={self._rounds}"
        )


    def _near_round_level(self, price: float, atr: float) -> bool:
        if not self.config or not self.config.__dict__.get("rounds_enabled", True):
            return False
        grid = self._rounds.get("grid") or []
        tol_atr = float(self._rounds.get("tol_atr", 0.0) or 0.0)
        if atr is None or atr <= 0 or not grid:
            return False
        tol_abs = tol_atr * float(atr)
        for g in grid:
            step = float(g)
            if step <= 0:
                continue
            # vzdálenost ceny od nejbližší hladiny step
            dist = abs(price - round(price / step) * step)
            if dist <= tol_abs:
                return True
        return False


    def export_live_features_df(self):
        return self._build_features_for_all()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # Status
        status_box = QGroupBox("Status")
        grid = QGridLayout()
        self.lbl_ib_status = QLabel("TV: Disconnected")
        self.lbl_time = QLabel("Time: --:--:--")
        self.lbl_fresh = QLabel("Freshness: --")
        grid.addWidget(self.lbl_ib_status, 0, 0)
        grid.addWidget(self.lbl_time,      0, 1)
        grid.addWidget(self.lbl_fresh,     0, 2)
        self.lbl_mode = QLabel("Mode: WARM-UP")
        grid.addWidget(self.lbl_mode,      0, 3)
        status_box.setLayout(grid)

        # Sezení
        session_box = QGroupBox("Sezení")
        h = QHBoxLayout()
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(["live"]); self.cmb_mode.setCurrentText("live"); self.cmb_mode.setEnabled(False)
        self.ed_symbol = QLineEdit(self.config.symbol)
        self.ed_expiry = QLineEdit(self.config.exchange)
        self.cmb_interval = QComboBox(); self.cmb_interval.addItems(["5 min", "15 min", "30 min", "1 hour"])
        self.cmb_interval.setCurrentText(self.config.bar_size)
        self.btn_start = QPushButton("Start"); self.btn_stop = QPushButton("Stop"); self.btn_reconnect = QPushButton("Reconnect")
        h.addWidget(QLabel("Režim:"));      h.addWidget(self.cmb_mode)
        h.addWidget(QLabel("Symbol:"));     h.addWidget(self.ed_symbol)
        h.addWidget(QLabel("Exchange:"));   h.addWidget(self.ed_expiry)
        h.addWidget(QLabel("Timeframe:"));  h.addWidget(self.cmb_interval)
        h.addWidget(self.btn_start);        h.addWidget(self.btn_stop); h.addWidget(self.btn_reconnect)
        session_box.setLayout(h)

        # Model
        model_box = QGroupBox("Model")
        g = QGridLayout()
        self.le_model_path = QLineEdit(DEFAULT_MODEL_DIR)
        self.btn_model = QPushButton("…")
        self.dsb_sensitivity = QDoubleSpinBox()
        self.dsb_sensitivity.setDecimals(2); self.dsb_sensitivity.setRange(0.00, 1.00); self.dsb_sensitivity.setSingleStep(0.01)
        self.dsb_sensitivity.setValue(self.config.sensitivity)
        self.chk_ma_only = QCheckBox("MA-only"); self.chk_ma_only.setChecked(self.config.use_ma_only)
        self.chk_and = QCheckBox("Ensemble AND (MA ∧ Model)")
        self.chk_invert_labels = QCheckBox("Invert labels 0↔1")
        self.chk_invert_labels.setToolTip("Ručně prohodí mapu tříd (0↔1). Použij jen pokud DIAG ukazuje opačnou polaritu.")
        self.chk_invert_labels.stateChanged.connect(self._on_toggle_invert_labels)
        g.addWidget(self.chk_invert_labels, 3, 0, 1, 2)
        self.chk_and.setChecked(self.config.use_and_ensemble)
        g.addWidget(QLabel("Cesta:"), 0, 0); g.addWidget(self.le_model_path, 0, 1); g.addWidget(self.btn_model, 0, 2)
        g.addWidget(QLabel("Citlivost:"), 1, 0); g.addWidget(self.dsb_sensitivity, 1, 1)
        g.addWidget(self.chk_ma_only, 2, 0); g.addWidget(self.chk_and, 2, 1)
        model_box.setLayout(g)

        # Log
        log_box = QGroupBox("Log")
        lv = QVBoxLayout()
        self.console = QPlainTextEdit(); self.console.setReadOnly(True)
        lv.addWidget(self.console)
        log_box.setLayout(lv)

        # Grafy
        center = self._create_center_charts()

        # Layout
        left = QVBoxLayout()
        left.addWidget(status_box)
        left.addWidget(session_box)
        left.addWidget(model_box)
        left.addWidget(log_box, 1)

        main = QHBoxLayout()
        main.addLayout(left, 2)
        main.addWidget(center, 8)
        root.addLayout(main)

    def _create_center_charts(self) -> QWidget:
        box = QGroupBox("Grafy")
        v = QVBoxLayout()
        self.fig = Figure(figsize=(8, 5), constrained_layout=True)
        gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 2])
        self.ax_price = self.fig.add_subplot(gs[0, 0])
        self.ax_macd  = self.fig.add_subplot(gs[1, 0], sharex=self.ax_price)
        self.canvas = FigureCanvas(self.fig)
        v.addWidget(self.canvas)
        box.setLayout(v)
        return box

    # ---------- Dráty ----------
    def _wire_basic_logic(self) -> None:
        self.btn_model.clicked.connect(self._on_choose_model)
        self.cmb_interval.currentTextChanged.connect(self._on_interval_changed)
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_reconnect.clicked.connect(self._on_reconnect)
        self.dsb_sensitivity.valueChanged.connect(self._on_sensitivity_changed)
        self.chk_ma_only.toggled.connect(lambda v: setattr(self.config, "use_ma_only", bool(v)))
        self.chk_and.toggled.connect(lambda v: setattr(self.config, "use_and_ensemble", bool(v)))

        self.fresh_timer = QTimer(self); self.fresh_timer.setInterval(1000)
        self.fresh_timer.timeout.connect(self._update_clock)
        self.fresh_timer.start()

    # ---------- Model (ensemble) ----------
    def _load_models(self) -> bool:
        """
        Načte 1..N modelů ze self.le_model_path (oddělené ; nebo novými řádky).
        Nastaví self.models a self.model_expected_features = průnik featur všech modelů (fallback na base cols).
        """
        text = (self.le_model_path.text() or "").strip()
        if not text:
            self._append_log("[ERROR] Zadej .pkl soubor(y) do pole Cesta.")
            return False

        # rozdělení vstupu
        parts = []
        for chunk in text.replace("\n", ";").split(";"):
            c = chunk.strip()
            if c:
                parts.append(c)
        if not parts:
            self._append_log("[ERROR] Nebyly nalezeny žádné cesty k modelům.")
            return False

        # pokud je zadaný adresář, necháme uživatele vybrat soubory
        if len(parts) == 1 and os.path.isdir(parts[0]):
            start_dir = parts[0]
            fnames, _ = QFileDialog.getOpenFileNames(self, "Vybrat modely", start_dir, "Pickle files (*.pkl);;All files (*)")
            parts = fnames

        loaded = []
        feats_sets = []
        feats_lists = []
        label_map_final = None

        def _extract_predictor(obj):
            meta = {}
            if hasattr(obj, "predict") or hasattr(obj, "predict_proba") or hasattr(obj, "decision_function"):
                return obj, meta
            if isinstance(obj, dict):
                for k in ("predictor", "model", "estimator"):
                    if k in obj:
                        return obj[k], (obj.get("metadata") or obj.get("meta") or {})
            if isinstance(obj, (tuple, list)) and len(obj) >= 1:
                pred = obj[0]; meta = obj[1] if len(obj) > 1 and isinstance(obj[1], dict) else {}
                return pred, meta
            raise TypeError("Objekt neobsahuje použitelný estimator.")

        from pathlib import Path
        for p in parts:
            if not os.path.exists(p):
                self._append_log(f"[ERROR] Soubor neexistuje: {p}")
                return False
            try:
                obj = joblib.load(p)
                pred, meta = _extract_predictor(obj)
                # načti meta z pkl-sidecaru, pokud chybí
                if not meta:
                    meta_path = Path(p).with_name(Path(p).stem + "_meta.json")
                    if meta_path.exists():
                        try:
                            with meta_path.open("r", encoding="utf-8") as fh:
                                meta = jsonlib.load(fh)
                        except Exception:
                            meta = {}

                # map tříd (poprvé převezmeme)
                if not label_map_final:
                    self.class_to_dir = {0: "SHORT", 1: "LONG"}
                    label_map_final = self.class_to_dir

                # featury
                exp = None
                if isinstance(meta, dict):
                    exp = meta.get("trained_features") or meta.get("expected_features")
                if exp is None and hasattr(pred, "feature_names_in_") and getattr(pred, "feature_names_in_", None) is not None:
                    exp = [str(c) for c in list(pred.feature_names_in_)]
                if exp:
                    exp_list = [str(c) for c in exp]
                    feats_sets.append(set(exp_list))
                    feats_lists.append(exp_list)

                loaded.append({
                    "predictor": pred,
                    "path": p,
                    "exp_feats": exp_list if exp else None,
                    "label_map": {0: "SHORT", 1: "LONG"},  # per-model default
                })

                loaded.append({"predictor": pred, "path": p})
                self._append_log(f"[INFO] Načten model: {os.path.basename(p)}")
            except Exception as e:
                self._append_log(f"[ERROR] Načtení modelu selhalo ({p}): {e}")
                return False

        # uložit členy ensemble
        self.models = loaded
        self.model = loaded[0]["predictor"] if loaded else None  # jen pro feature_names_in_

        # sjednotit feature-space → průnik, nebo fallback
        base_cols = ['close', 'ma_fast', 'ma_slow', 'atr', 'average']
        if feats_sets:
            inter = set.intersection(*feats_sets) if len(feats_sets) > 1 else list(feats_sets)[0]
            if not inter:
                self._append_log("[WARN] Průnik featur modelů je prázdný – padám na fallback (MA/ATR/average).")
                self.model_expected_features = base_cols
            else:
                if feats_lists:
                    base_order = feats_lists[0]
                    self.model_expected_features = [c for c in base_order if c in inter]
                else:
                    # kdyby nebyl k dispozici list, nesortovat: ponecháme pořadí tak, jak je v inter přes první nalezený set
                    self.model_expected_features = list(inter)
                self._append_log(f"[INFO] Průnik featur v ensemble: {len(self.model_expected_features)}")
        else:
            self.model_expected_features = base_cols

        # pojmenování vrstev
        self._append_log(f"[LAYERS] L0=MA | L1i=Model_i | L1_AND=AND přes {len(self.models)} modelů | L2_AND=(volitelně) MA ∧ L1_AND")
        return True

    # AND hlasování přes všechny modely
    def _predict_one_label_AND(self, Xrow: pd.DataFrame, thr: float) -> tuple[int, float, list[str], list[float]]:
        """
        Vrací (label {-1,0,+1}, conf_min, directions, confs)
        """
        if not self.models:
            return 0, 0.0, [], []

        dirs = []
        confs = []
        for m in self.models:
            mdl = m["predictor"]
            exp = m.get("exp_feats")
            X_use = Xrow
            if exp:
                cols = [c for c in exp if c in Xrow.columns]
                if cols:
                    X_use = Xrow[cols].astype(float)
            try:
                label_map = m.get("label_map") or self.class_to_dir  # per-model mapa 1st
                pL, pS, classes_i, _ = _extract_long_short_proba(mdl, X_use, label_map=label_map)
            except Exception:
                pL, pS, classes_i = 0.5, 0.5, None

            # DIAG logging (prvních 10 záznamů)
            if not hasattr(self, "_diag_counter"):
                self._diag_counter = 0
            if self._diag_counter < 10:
                self._append_log(f"[DIAG] classes={classes_i} pL={pL:.3f} pS={pS:.3f} from {type(mdl).__name__}")
                self._diag_counter += 1

        # výběr směru a konfidence
            if pL > pS:
                direction = "LONG"
                conf = float(pL)
            elif pS > pL:
                direction = "SHORT"
                conf = float(pS)
            else:
                direction = "FLAT"
                conf = float(pL)

            dirs.append(direction)
            confs.append(conf)


        conf_min = min(confs) if confs else 0.0
        if all(d == "LONG"  for d in dirs) and conf_min >= thr:
            return +1, conf_min, dirs, confs
        if all(d == "SHORT" for d in dirs) and conf_min >= thr:
            return -1, conf_min, dirs, confs
        return 0, conf_min, dirs, confs

    @Slot()
    def _on_choose_model(self) -> None:
        default_dir = DEFAULT_MODEL_DIR if os.path.isdir(DEFAULT_MODEL_DIR) else os.getcwd()
        fnames, _ = QFileDialog.getOpenFileNames(self, "Vybrat modely", default_dir, "Pickle files (*.pkl);;All files (*)")
        if fnames:
            self.le_model_path.setText(";".join(fnames))

    # ---------- Ovládání ----------
    @Slot()
    def _on_start(self):
        if not self._load_models():
            return
        self._apply_tf_presets()
        self._start_worker()
        self._append_log("[INFO] Start sezení…")
        self._append_log(f"[MODE] MA-only={self.config.use_ma_only} | AND={self.config.use_and_ensemble}")

        try:
            adapter = _WarmAdapter(self)
            self.warm = LiveWarmupService(
                base_service=adapter,
                config=WarmupConfig(
                    threshold=float(self.dsb_sensitivity.value()),
                    warmup_bars=100,
                    min_sim_trades=3,
                    start_sharpe=0.00,
                    max_dd=25.0,
                    diag_first_n=220,
                ),
            )
            cfg = getattr(self.warm, "config", None)
            self._append_log(f"[WARMUP-CONFIG] {cfg if cfg is not None else 'MISSING'}")
            self.warm.start(self.config.symbol, self.config.exchange, self.config.bar_size)
            self.lbl_mode.setText("Mode: LIVE" if self.warm.state == "LIVE" else "Mode: WARM-UP")
        except Exception as e:
            self._append_log(f"[WARN] Warm-up inicializace selhala: {e}")

    @Slot()
    def _on_stop(self) -> None:
        self._append_log("[INFO] Stop sezení.")
        self._stop_worker()

    @Slot()
    def _on_reconnect(self) -> None:
        self._append_log("[INFO] Reconnect…")
        self._stop_worker()
        self._start_worker()

    @Slot(str)
    def _on_mode_changed(self, mode: str) -> None:
        self.config.mode = mode

    @Slot(float)
    def _on_sensitivity_changed(self, val: float) -> None:
        self._append_log(f"[INFO] Citlivost (min conf) → {val:.2f} (přepočet signálů)")
        try:
            self._rescore_all()
            if self.warm is not None:
                self.lbl_mode.setText("Mode: LIVE" if self.warm.state == "LIVE" else "Mode: WARM-UP")
        except Exception as e:
            self._append_log(f"[WARN] Re-score selhal: {e}")
        self._render_charts()

    @Slot(str)
    def _on_interval_changed(self, val: str) -> None:
        self.config.bar_size = val
        self._apply_tf_presets()
        self._append_log(f"[INFO] Interval změněn na '{val}'.")
        if self.worker is not None:
            self._append_log("[INFO] Restart streamu kvůli změně intervalu…")
            self._stop_worker()
            self._start_worker()

    # ---------- Worker reakce ----------
    @Slot(dict)
    def _on_bar_closed(self, bar: dict) -> None:
        ts_raw = bar.get("time")
        close = float(bar.get("close", 0.0))
        self._append_log(f"[BAR] {ts_raw} close={close}")

        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        if pd.isna(ts):
            return
        key = int(ts.value)
        payload = {
            "time": ts,
            "open":  float(bar.get("open",  np.nan)),
            "high":  float(bar.get("high",  np.nan)),
            "low":   float(bar.get("low",   np.nan)),
            "close": close,
            "volume": float(bar.get("volume", 0) or 0),
        }
        idx = self._bar_index.get(key)
        if idx is None:
            self._bar_index[key] = len(self._bars)
            self._bars.append(payload)
        else:
            self._bars[idx] = payload

        if len(self._bars) > self.config.max_bars_buffer:
            self._bars = self._bars[-self.config.max_bars_buffer:]
            self._bar_index = {int(pd.to_datetime(x["time"]).value): i for i, x in enumerate(self._bars)}

        row = {
            "timestamp": ts,
            "open": payload["open"],
            "high": payload["high"],
            "low":  payload["low"],
            "close": payload["close"],
            "volume": payload["volume"],
        }
        self.live_df = pd.concat([self.live_df, pd.DataFrame([row])], ignore_index=True)
        self.live_df.dropna(subset=["timestamp"], inplace=True)
        self.live_df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        self.live_df.sort_values("timestamp", inplace=True)
        if len(self.live_df) > self.config.max_bars_buffer:
            self.live_df = self.live_df.tail(self.config.max_bars_buffer).reset_index(drop=True)

        try:
            if self.warm is not None:
                self.warm.on_new_bar(payload)
                self.lbl_mode.setText("Mode: LIVE" if self.warm.state == "LIVE" else "Mode: WARM-UP")
            self._rescore_all()
            self._maybe_alert_flip_on_last_bar()
        except Exception as e:
            self._append_log(f"[WARN] Re-score selhal: {e}")

        self._last_arrival_utc = pd.Timestamp.now(tz='UTC')
        self._update_freshness()
        self._render_charts()

        # email flip alert
        try:
            if self.config.alert_email_enabled and self._bars:
                last = self._bars[-1]
                sig = last.get("signal")
                if sig in ("LONG", "SHORT"):
                    bar_key = int(pd.to_datetime(last["time"]).value)
                    if self._last_alert_bar_key is None or bar_key > self._last_alert_bar_key:
                        prev = self._last_alert_sig
                        if prev in ("LONG", "SHORT") and prev != sig:
                            subj, body = self._format_flip_email(prev, sig, last["time"], float(last.get("close", float("nan"))))
                            for addr in (self.config.alert_email_to or "").split(","):
                                addr = addr.strip()
                                if addr:
                                    self._send_email_async(addr, subj, body)
                        self._last_alert_sig = sig
                        self._last_alert_bar_key = bar_key
        except Exception:
            pass

    @Slot(str)
    def _on_ib_status(self, status: str) -> None:
        self.lbl_ib_status.setText(f"TV: {status}")

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._append_log("[ERROR] " + message)

    # ---------- Worker lifecycle ----------
    def _start_worker(self) -> None:
        if self.worker is not None:
            self._stop_worker()
        self._bars.clear()
        self._bar_index.clear()
        self.live_df = self.live_df.iloc[0:0].copy()
        self._render_charts()

        self.config.mode = self.cmb_mode.currentText()
        self.config.symbol = (self.ed_symbol.text() or "GOLD").strip()
        self.config.exchange = (self.ed_expiry.text() or "TVC").strip()
        self.config.bar_size = self.cmb_interval.currentText()

        try:
            tv = TradingViewClient(username=os.getenv("TV_USERNAME"), password=os.getenv("TV_PASSWORD"))
            tf_label = (self.config.bar_size or "1 hour").replace("mins", "min")
            df = tv.get_history(self.config.symbol, self.config.exchange, tf_label, limit=int(self.config.max_bars_buffer))
            if df is not None and not df.empty:
                df = df.copy()
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = df.dropna(subset=["time"]).sort_values("time").tail(self.config.max_bars_buffer)

                self._bars = []
                self._bar_index = {}
                for _, r in df.iterrows():
                    ts = r["time"]
                    payload = {
                        "time": ts,
                        "open": float(r.get("open", np.nan)),
                        "high": float(r.get("high", np.nan)),
                        "low":  float(r.get("low",  np.nan)),
                        "close": float(r.get("close", np.nan)),
                        "volume": float(r.get("volume", 0) or 0),
                    }
                    self._bar_index[int(ts.value)] = len(self._bars)
                    self._bars.append(payload)

                self.live_df = pd.DataFrame({
                    "timestamp": df["time"].to_numpy(),
                    "open":  df["open"].astype(float).to_numpy(),
                    "high":  df["high"].astype(float).to_numpy(),
                    "low":   df["low"].astype(float).to_numpy(),
                    "close": df["close"].astype(float).to_numpy(),
                    "volume": df["volume"].astype(float).to_numpy(),
                })
                self.live_df = self.live_df.tail(self.config.max_bars_buffer).reset_index(drop=True)
                self._append_log(f"[INFO] Načten počáteční snapshot: {len(self.live_df)} barů.")
                self._last_arrival_utc = pd.Timestamp.now(tz='UTC')
                self._rescore_all()
                X_hist_all = self._build_features_for_all()
                raw_df = self.live_df.rename(columns={'timestamp': 'date'})[['date','open','high','low','close','volume']].copy()
                raw_df['date'] = pd.to_datetime(raw_df['date'], utc=True, errors='coerce')
                raw_df = raw_df.dropna(subset=['date']).sort_values('date')

                changed = 0
                for m in self.models:
                    mdl = m['predictor']
                    exp = m.get('exp_feats')
                    X_use = X_hist_all
                    if exp:
                        cols = [c for c in exp if c in X_hist_all.columns]
                        if cols:
                            X_use = X_hist_all[cols].astype(float)
                    auto_map = _auto_detect_label_polarity(mdl, X_use, raw_df)
                    if auto_map and set(auto_map.values()) == {"LONG","SHORT"}:
                        m['label_map'] = dict(auto_map)
                        changed += 1

                self._append_log(f"[AUTO] Per-model mapy tříd nastaveny pro {changed}/{len(self.models)} modelů.")
                self._rescore_all()
                self._update_freshness()
                self._render_charts()
            else:
                self._append_log("[WARN] Počáteční snapshot prázdný.")
        except Exception as e:
            self._append_log(f"[WARN] Počáteční snapshot selhal: {e}")

        self.worker = TVWorker(self.config, parent=self)
        self.worker.statusChanged.connect(self._on_ib_status)
        self.worker.error.connect(self._on_error)
        self.worker.barClosed.connect(self._on_bar_closed)
        self.worker.start()

    def _on_toggle_invert_labels(self, state):
        if not hasattr(self, "class_to_dir") or not self.class_to_dir:
            self.class_to_dir = {0: "SHORT", 1: "LONG"}
        # invertuj
        inv = {}
        for k, v in self.class_to_dir.items():
            inv[k] = "LONG" if str(v).upper() == "SHORT" else "SHORT"
        self.class_to_dir = inv
        self._append_log(f"[MANUAL] Invertuji mapu tříd: {self.class_to_dir}")
        # přepočítej okamžitě posledních N barů, ať vidíš efekt
        try:
            self._rescore_all()
        except Exception as e:
            self._append_log(f"[MANUAL] Rescore po invertu selhal: {e!r}")


    def _stop_worker(self) -> None:
        if self.worker is None:
            return
        try:
            self.worker.stop()
        except Exception:
            pass
        finally:
            self.worker = None

    # ---------- Feature engineering ----------
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ma_fast'] = df['close'].rolling(9, min_periods=1).mean()
        df['ma_slow'] = df['close'].rolling(21, min_periods=1).mean()
        h_l  = df['high'] - df['low']
        h_pc = (df['high'] - df['close'].shift(1)).abs()
        l_pc = (df['low']  - df['close'].shift(1)).abs()
        tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14, min_periods=1).mean()
        df['average'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
        return df

    def _get_raw_indicators(self) -> pd.DataFrame | None:
        if self.live_df is None or self.live_df.empty:
            return None
        df = self.live_df.rename(columns={"timestamp": "date"})
        df = df[["date","open","high","low","close","volume"]].dropna(subset=["close"]).copy()
        if df.empty:
            return None
        ind = self._compute_indicators(df)
        ind["date"] = pd.to_datetime(ind["date"], utc=True, errors="coerce")
        ind = ind.dropna(subset=["date"]).set_index("date").sort_index()
        return ind

    def _align_features_to_model(self, feat: pd.DataFrame) -> pd.DataFrame:
        base_cols = ['close', 'ma_fast', 'ma_slow', 'atr', 'average']
        df = feat.copy()
        if 'average' not in df.columns and all(c in df.columns for c in ['open','high','low','close']):
            df['average'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0

        # Kandidátní seznam požadovaných featur
        if self.model_expected_features and isinstance(self.model_expected_features, (list, tuple)):
            use_cols = [str(c) for c in self.model_expected_features]
        elif hasattr(self.model, "feature_names_in_") and getattr(self.model, "feature_names_in_", None) is not None:
            use_cols = [str(c) for c in list(self.model.feature_names_in_)]
        else:
            use_cols = list(base_cols)

        # Vezmi pouze ty, které v DF opravdu existují
        overlap = [c for c in use_cols if c in df.columns]

        # Pokud jich je málo (např. < 30 %), spadni na base_cols (ty si umíme spočítat)
        if not overlap or len(overlap) < max(1, int(0.2 * len(use_cols))):
            self._append_log(f"[WARN] Pouze {len(overlap)}/{len(use_cols)} požadovaných featur k dispozici – padám na base_cols.")
            overlap = [c for c in base_cols if c in df.columns]

        df = df[overlap].copy()

        # Číselná konverze + imputace mediánem (žádné doplňování *nových* sloupců nulami)
        for c in df.columns:
            if not pd.api.types.is_bool_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        med = df.median(numeric_only=True)
        df = df.fillna(med).fillna(0.0).astype('float32')
        return df

    def _build_features_from_live(self) -> pd.DataFrame | None:
        if self.live_df is None or self.live_df.empty:
            return None
        df = self.live_df.rename(columns={'timestamp': 'date'})
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])
        if df.empty:
            return None
        feat_df = self._compute_indicators(df)
        last = feat_df.iloc[[-1]].copy()
        return self._align_features_to_model(last)

    def _build_features_for_all(self) -> pd.DataFrame | None:
        if self.live_df is None or self.live_df.empty:
            return None
        df = self.live_df.rename(columns={"timestamp": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"]).copy()
        if df.empty:
            return None
        feat = self._compute_indicators(df)
        feat["date"] = pd.to_datetime(feat["date"], utc=True, errors="coerce")
        feat = feat.dropna(subset=["date"]).set_index("date").sort_index()
        return self._align_features_to_model(feat)

    # ---------- Pomocné mapování ----------
    def _label_to_dir(self, cls) -> str | None:
        if isinstance(cls, (np.generic,)):
            try:
                cls = cls.item()
            except Exception:
                pass
        if isinstance(cls, (int, np.integer)) and int(cls) in self.class_to_dir:
            return self.class_to_dir[int(cls)]
        if isinstance(cls, str) and cls.strip().isdigit():
            k = int(cls.strip())
            return self.class_to_dir.get(k)
        s = str(cls).strip().lower()
        if s in ("1", "+1", "long", "buy", "up"):
            return "LONG"
        if s in ("0", "-1", "short", "sell", "down"):
            return "SHORT"
        return None

    def _sign_to_dir(self, v) -> str | None:
        try:
            f = float(v)
        except Exception:
            return None
        if f > 0:
            return "LONG"
        if f < 0:
            return "SHORT"
        return None

    def _nearest_bar_index(self, ts) -> int | None:
        if not self._bars:
            return None
        t_target = int(pd.to_datetime(ts).value)
        arr = np.array([int(pd.to_datetime(b["time"]).value) for b in self._bars], dtype=np.int64)
        tf = (self.config.bar_size or "1 hour")
        sec = {"5 min":300, "15 min":900, "30 min":1800, "1 hour":3600}.get(tf, 3600)
        tol_ns = int(0.5 * sec * 1e9)
        i = int(np.argmin(np.abs(arr - t_target)))
        if abs(int(arr[i]) - t_target) <= tol_ns:
            return i
        return None

    # ---------- Hlavní re-score s AND logikou ----------
    def _rescore_all(self) -> None:
        self._append_log(f"[DBG] rescore: models={len(self.models) if hasattr(self,'models') else 0} "
                     f"MA_only={self.config.use_ma_only} tf='{self.config.bar_size}'")

        raw = self._get_raw_indicators()
        if raw is None or raw.empty:
            return

        feats = self._align_features_to_model(raw)
        thr = float(self.dsb_sensitivity.value())

        # L0: MA
        l0_series = np.sign((raw["ma_fast"] - raw["ma_slow"]).astype(float).to_numpy())
        l0_dir = np.array(["LONG" if v>0 else "SHORT" if v<0 else "FLAT" for v in l0_series], dtype=object)

        # Pokud MA-only nebo chybí ensemble → rovnou použij L0
        if self.config.use_ma_only or not self.models:
            reason = "MA-only=True" if self.config.use_ma_only else "no_models_loaded"
            n_total = len(feats.index)
            n_mapped = n_shown = n_long = n_short = 0
            for ts, d0 in zip(feats.index, l0_dir):
                idx = self._nearest_bar_index(ts)
                if idx is None: continue
                n_mapped += 1
                sig = None if d0 == "FLAT" else d0
                self._bars[idx]["signal"] = sig
                self._bars[idx]["proba"]  = 1.0 if sig else None
                self._bars[idx]["layers"] = {"L0_MA": d0, "L1_AND": None, "L2_AND": sig or "FLAT"}
                if sig == "LONG": n_long += 1; n_shown += 1
                elif sig == "SHORT": n_short += 1; n_shown += 1
            self._append_log(f"[RESCORE] (MA-only/model-missing) bars={len(self._bars)} feats={n_total} mapped={n_mapped} shown={n_shown} (LONG={n_long}, SHORT={n_short})")
            self._append_log(f"[RESCORE] ({reason}) bars={len(self._bars)} feats={n_total} mapped={n_mapped} ")
            return

        # Ensemble AND (volitelně MA ∧ L1_AND)
        use_ma_and = self.config.use_and_ensemble

        n_mapped = n_shown = n_long = n_short = 0
        for ts, d0 in zip(feats.index, l0_dir):
            idx = self._nearest_bar_index(ts)
            if idx is None:
                continue
            Xrow = feats.loc[[ts]]

            thr_model = 0.0  # nefiltruj směr uvnitř AND, prahy řeší hysterese
            label, conf_min, dirs, confs = self._predict_one_label_AND(Xrow, thr_model)
            l1 = "LONG" if label == +1 else "SHORT" if label == -1 else "FLAT"

            # --- MA∧AND nebo čistý AND: vytvoř "proposal" ---
            if use_ma_and:
                # jen shoda směru s MA, BEZ prahu conf_min
                if d0 == "FLAT":
                    proposal = l1 if (l1 in ("LONG","SHORT")) else None
                else:
                    proposal = l1 if (l1 == d0) else None
            else:
                # čistý výstup modelu L1, BEZ prahu conf_min
                proposal = l1 if (l1 in ("LONG","SHORT")) else None

            # --- Hysterese: tady teprve aplikuj prahy ---
            final = None
            curr_close = float(raw.loc[ts, "close"])
            curr_atr   = float(raw.loc[ts, "atr"])

            if self._live_pos == 0:
                if proposal in ("LONG","SHORT") and conf_min >= self._curr_entry_thr:
                    final = proposal
            else:
                want_dir = "LONG" if self._live_pos > 0 else "SHORT"
                if proposal == want_dir and conf_min >= self._curr_exit_thr:
                    final = want_dir

            proba = conf_min if final else None
            layers = {"L0_MA": d0, "L1_AND": l1, "L2_AND": (final or "FLAT"),
                      "L1_members": [{"dir": d, "conf": float(c)} for d,c in zip(dirs, confs)]}


            self._bars[idx]["signal"] = final
            self._bars[idx]["proba"]  = proba
            self._bars[idx]["layers"] = layers

            # --- Aktualizace stavu pozice pouze na posledním baru ---
            is_last_bar = (ts == feats.index[-1])
            if is_last_bar:
                if final == "LONG":
                    if self._live_pos <= 0:  # vstup/otočka
                        self._live_pos = +1
                        self._live_entry_px = float(raw.loc[ts, "close"])
                elif final == "SHORT":
                    if self._live_pos >= 0:
                        self._live_pos = -1
                        self._live_entry_px = float(raw.loc[ts, "close"])
                else:
                    # FLAT – uzavři případnou živou pozici
                    if self._live_pos != 0:
                        self._live_pos = 0
                        self._live_entry_px = None
            # --------------------------------------------------------

            n_mapped += 1
            if final == "LONG":  n_long += 1; n_shown += 1
            if final == "SHORT": n_short += 1; n_shown += 1

        self._append_log(
            f"[RESCORE] bars={len(self._bars)} mapped={n_mapped} shown={n_shown} "
            f"(LONG={n_long}, SHORT={n_short}) thr={thr:.2f} models={len(self.models)} AND_MA={use_ma_and}"
        )

    # ----------
    def _append_log(self, text: str) -> None:
        self.console.appendPlainText(text)
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.console.setTextCursor(cursor)

    def _play_alert(self) -> None:
        try:
            if self._se is not None:
                self._se.play()
            else:
                from PySide6.QtGui import QGuiApplication
                QGuiApplication.beep()
        except Exception:
            pass

    def _maybe_alert_flip_on_last_bar(self) -> None:
        if not getattr(self.config, "alert_on_flip", True) or not self._bars:
            return
        last = self._bars[-1]
        sig = last.get("signal")
        if sig not in ("LONG", "SHORT"): return
        ts = pd.to_datetime(last.get("time"), utc=True, errors="coerce")
        if pd.isna(ts): return
        ts_ns = int(ts.value)
        if self._last_alert_bar_ns == ts_ns: return

        prev = self._last_signal
        if prev is not None and prev != sig:
            now = pd.Timestamp.now(tz="UTC")
            cd = int(getattr(self.config, "alert_cooldown_s", 5))
            if self._last_beep_time is None or (now - self._last_beep_time).total_seconds() >= cd:
                self._append_log(f"[ALERT] Flip {prev} → {sig} @ {ts}")
                self._play_alert()
                try:
                    if getattr(self.config, "alert_email_enabled", False):
                        subject, body = self._format_flip_email(prev, sig, ts, float(last.get("close", float("nan"))))
                        to_addrs = [a.strip() for a in (self.config.alert_email_to or "").split(",") if a.strip()]
                        for addr in to_addrs:
                            self._send_email_async(addr, subject, body)
                except Exception as _e:
                    self._append_log(f"[EMAIL] Výjimka při přípravě zprávy: {_e}")
                self._last_beep_time = now
                self._last_alert_bar_ns = ts_ns
        self._last_signal = sig

    def _format_flip_email(self, prev_sig: str, new_sig: str, ts: pd.Timestamp, px: float) -> tuple[str, str]:
        sym = getattr(self.config, "symbol", "?")
        exch = getattr(self.config, "exchange", "?")
        tf = getattr(self.config, "bar_size", "?")
        when = str(ts)
        subject = f"[LIVE] Flip {prev_sig} → {new_sig} | {sym} {exch} {tf}"
        body = (
            f"Signal flip detected.\n\n"
            f"Symbol:   {sym}\n"
            f"Exchange: {exch}\n"
            f"TF:       {tf}\n"
            f"Time:     {when}\n"
            f"Close:    {px}\n"
            f"From → To: {prev_sig} → {new_sig}\n"
        )
        return subject, body

    def _send_email(self, to_addr: str, subject: str, body: str) -> None:
        cfg = self.config
        host = cfg.smtp_host
        port = int(cfg.smtp_port or (465 if cfg.smtp_use_ssl else 587))
        user = cfg.smtp_user
        password = cfg.smtp_password
        from_addr = cfg.smtp_from or user or "alerts@localhost"

        if not host or not to_addr:
            self._append_log("[EMAIL] SMTP_HOST/ALERT_EMAIL_TO chybí – e-mail se neodeslal.")
            return

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.set_content(body)

        if cfg.smtp_use_ssl:
            with smtplib.SMTP_SSL(host, port) as s:
                if user and password: s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port) as s:
                s.ehlo(); s.starttls()
                if user and password: s.login(user, password)
                s.send_message(msg)

    def _send_email_async(self, to_addr: str, subject: str, body: str) -> None:
        def _worker():
            try:
                self._send_email(to_addr, subject, body)
                self._append_log("[EMAIL] Flip notifikace odeslána.")
            except Exception as e:
                self._append_log(f"[EMAIL] Odeslání selhalo: {e}")
        threading.Thread(target=_worker, daemon=True).start()

    def _update_clock(self) -> None:
        from datetime import datetime
        self.lbl_time.setText("Time: " + datetime.now().strftime("%H:%M:%S"))
        self._update_freshness()

    def _update_freshness(self) -> None:
        try:
            ok_threshold = int(self.config.max_fresh_age_min) * 60
        except Exception:
            ok_threshold = 300
        now_utc = pd.Timestamp.now(tz='UTC')
        if self._last_arrival_utc is not None:
            age_s = max(0, int((now_utc - self._last_arrival_utc).total_seconds()))
        else:
            if self.live_df is None or self.live_df.empty:
                self.lbl_fresh.setText("Freshness: --"); self.lbl_fresh.setStyleSheet(""); return
            last_ts = pd.to_datetime(self.live_df["timestamp"].iloc[-1], utc=True, errors="coerce")
            if pd.isna(last_ts):
                self.lbl_fresh.setText("Freshness: --"); self.lbl_fresh.setStyleSheet(""); return
            age_s = max(0, int((now_utc - last_ts).total_seconds()))
        mins, secs = divmod(age_s, 60)
        self.lbl_fresh.setText(f"Freshness: {mins}m {secs}s")
        self.lbl_fresh.setStyleSheet("color: #119911;" if age_s <= ok_threshold else "color: #cc0000;")

    def _render_charts(self) -> None:
        if not self._bars:
            self.ax_price.cla(); self.ax_macd.cla(); self.canvas.draw_idle(); return
        df = pd.DataFrame(self._bars).reset_index(drop=True)

        # MACD 12-26-9
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig  = macd.ewm(span=9, adjust=False).mean()
        hist = macd - sig

        ax1, ax2 = self.ax_price, self.ax_macd
        ax1.cla(); ax2.cla()
        x = np.arange(len(df))

        # svíčky
        for i, row in df.iterrows():
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            color = 'g' if c >= o else 'r'
            ax1.vlines(i, l, h, linewidth=1, color=color)
            ax1.vlines(i, min(o, c), max(o, c), linewidth=6, color=color)

        # šipky (výsledek = L2_AND / FINAL)
        if 'signal' in df.columns:
            rng = (df['high'] - df['low']).replace(0, np.nan)
            pad = float(np.nanmedian(rng)) * 0.12 if not np.isnan(np.nanmedian(rng)) else 0.0
            pad = max(pad, 0.0001)
            long_x, long_y, short_x, short_y = [], [], [], []
            for i2, row2 in df.iterrows():
                s = row2.get('signal')
                if s == 'LONG':
                    long_x.append(i2); long_y.append(row2['low'] - pad)
                elif s == 'SHORT':
                    short_x.append(i2); short_y.append(row2['high'] + pad)
            if long_x:
                ax1.scatter(long_x, long_y, marker='^', s=90, color='green', zorder=5)
            if short_x:
                ax1.scatter(short_x, short_y, marker='v', s=90, color='red', zorder=5)

        ax1.set_ylabel('Price')
        ax2.plot(x, macd.values, label='MACD')
        ax2.plot(x, sig.values,  label='Signal')
        ax2.bar(x, hist.values, width=0.8, alpha=0.3)
        ax2.set_ylabel('MACD')
        ax2.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.2); ax2.grid(True, alpha=0.2)
        self.canvas.draw_idle()


# Kompatibilitní aliasy
class LiveTradingBotTab(LiveBotWidget):
    pass

class LiveBotTab(LiveBotWidget):
    pass
