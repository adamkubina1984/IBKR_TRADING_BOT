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
import warnings
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ibkr_trading_bot.core.config.presets import PRESETS_BY_TF
from ibkr_trading_bot.features.feature_engineering import compute_all_features

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


def _feature_names_for_model(model) -> list[str] | None:
    try:
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            return [str(c) for c in list(names)]
    except Exception:
        pass
    try:
        steps = getattr(model, "steps", None)
        if steps:
            last = steps[-1][1]
            names = getattr(last, "feature_names_in_", None)
            if names is not None:
                return [str(c) for c in list(names)]
    except Exception:
        pass
    return None


def _align_X_for_model(model, X):
    if isinstance(X, pd.DataFrame):
        Xdf = X.copy()
    else:
        Xdf = pd.DataFrame(X)

    names = _feature_names_for_model(model)
    if names:
        for c in names:
            if c not in Xdf.columns:
                Xdf[c] = 0.0
        Xdf = Xdf.reindex(columns=names, fill_value=0.0)

    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med).fillna(0.0)
    for c in Xdf.columns:
        if not pd.api.types.is_bool_dtype(Xdf[c]):
            Xdf[c] = Xdf[c].astype(float, copy=False)
    return Xdf


def _predict_proba_safely(model, X):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but .* was fitted with feature names",
            category=UserWarning,
        )
        return model.predict_proba(X)


def _infer_label_map_from_classes(classes, base_map: dict | None = None) -> dict[int, str]:
    """
    Vrátí robustní mapu numerických tříd na směr:
    - binární: {-1,+1} nebo {0,1}
    - ternární: {-1,0,+1} nebo {0,1,2}
    Pokud je base_map dodaná, má prioritu nad inferovanou mapou.
    """
    inferred: dict[int, str] = {}

    ints: list[int] = []
    if classes is not None:
        for c in list(classes):
            try:
                ints.append(int(c))
            except Exception:
                pass

    uniq = sorted(set(ints))
    if uniq:
        s = set(uniq)
        if s == {-1, 1}:
            inferred = {-1: "SHORT", 1: "LONG"}
        elif s == {0, 1}:
            inferred = {0: "SHORT", 1: "LONG"}
        elif s == {-1, 0, 1}:
            inferred = {-1: "SHORT", 0: "HOLD", 1: "LONG"}
        elif s == {0, 1, 2}:
            inferred = {0: "SHORT", 1: "HOLD", 2: "LONG"}
        else:
            if len(uniq) >= 2:
                inferred[uniq[0]] = "SHORT"
                inferred[uniq[-1]] = "LONG"
            for u in uniq[1:-1]:
                inferred[u] = "HOLD"

    if not inferred:
        inferred = {0: "SHORT", 1: "LONG"}

    if base_map:
        for k, v in base_map.items():
            try:
                inferred[int(k)] = str(v).upper()
            except Exception:
                continue

    return inferred

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
    X_pred = _align_X_for_model(model, X)
    proba = _predict_proba_safely(model, X_pred)
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
    X_pred = _align_X_for_model(model, df_row)
    try:
        proba = _predict_proba_safely(model, X_pred)[0]
    except Exception:
        proba = _predict_proba_safely(model, df_row)[0]

    classes = getattr(model, "classes_", None)

    # 2) Robustní mapa pro numerické třídy (včetně ternární klasifikace)
    label_map = _infer_label_map_from_classes(classes, base_map=label_map)

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
        elif dir_max in ("HOLD", "FLAT", "NONE"):
            p_long, p_short = 0.5, 0.5
        else:
            # když fakt netuším: udrž symetrii
            p_long, p_short = p_max, 1.0 - p_max

    return float(p_long), float(p_short), (list(classes) if classes is not None else None), proba


def _pick_direction_from_raw_proba(
    classes_i,
    raw_proba,
    label_map: dict,
    hold_block_thr: float = 0.78,
    hold_margin: float = 0.20,
):
    """
    Určí směr z raw pravděpodobností u multi-class modelu.
    HOLD blokuje směr jen při silné dominanci (thr + margin), jinak vrací LONG/SHORT.
    """
    try:
        if classes_i is None or raw_proba is None or len(raw_proba) != len(classes_i):
            return None, 0.0

        map_i = _infer_label_map_from_classes(classes_i, base_map=label_map)

        p_long = p_short = p_hold = 0.0
        for idx, cls in enumerate(classes_i):
            try:
                d = str(map_i.get(int(cls), "")).upper()
            except Exception:
                d = ""
            p = float(raw_proba[idx])
            if d == "LONG":
                p_long += p
            elif d == "SHORT":
                p_short += p
            elif d in ("HOLD", "FLAT", "NONE", "NEUTRAL"):
                p_hold += p

        dir_side = "LONG" if p_long >= p_short else "SHORT"
        p_side = p_long if dir_side == "LONG" else p_short

        if p_hold >= hold_block_thr and (p_hold - p_side) >= hold_margin:
            return "FLAT", float(p_hold)

        return dir_side, float(p_side)
    except Exception:
        return None, 0.0


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
    max_bars_buffer: int = 300  # Buffer pro live bars (po dropna z rolling bude ~200 validních)
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
        X = self.w._sanitize_feature_matrix(last)

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
        # Praktikovat práh z user_settings (nachází se z Tab 3)
        user_settings = self.w.user_settings or {}
        thr_ui = float(user_settings.get("entry_threshold", self.w._curr_entry_thr))

        classes = ["LONG", "SHORT"]

        # L0: MA
        l0 = self._ma_sig_from_features(features) or "FLAT"

        # MA-only režim -> vrať rovnou MA
        if user_settings.get("use_ma_only", False):
            probs = [1.0, 0.0] if l0 == "LONG" else [0.0, 1.0] if l0 == "SHORT" else [0.5, 0.5]
            return l0, probs, classes

        # Když není žádný model -> vrať MA
        if not self.w.models:
            probs = [1.0, 0.0] if l0 == "LONG" else [0.0, 1.0] if l0 == "SHORT" else [0.5, 0.5]
            return l0, probs, classes

        # L1: AND nebo VOTE podle nastavení (model-only = VOTE)
        if user_settings.get("use_and_ensemble", True):
            label, conf_min, dirs, confs = self.w._predict_one_label_AND(features, thr=0.0)
        else:
            label, conf_min, dirs, confs = self.w._predict_one_label_VOTE(features)
        l1 = "LONG" if label == +1 else "SHORT" if label == -1 else "FLAT"

        # L2: (volitelně) MA ∧ L1 + aplikace prahu z UI (thr_ui) – stejná politika jako v _rescore_all
        if user_settings.get("use_and_ensemble", True):
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
        
        # Nastavení modelu z Tab 3 (uložená v metadata)
        self.user_settings = {}  # dict se všemi thresholdy a flagy z Tab 3

        # Diagnostika degradace modelu (Reference vs Live)
        self.reference_metrics = {}  # Referenční metriky z metadata (train nebo holdout)
        self._prediction_buffer = []  # Buffer posledních N predictions (signály)
        self._price_buffer = []  # Buffer posledních N cen (close)
        self._y_true_buffer = []  # Buffer posledních N ground truth (pokud dostupné)
        self._tracked_timestamps = set()  # Set timestampů již trackovaných barů (pro deduplicitu)
        self.degradation_window_size = 500  # Počet barů pro recent window
        self._last_degradation_check = 0  # Index posledního checku
        self.live_metrics_recent = {}  # Aktuální metriky na recent window

        # Sledování obchodů
        self._trades: list[dict[str, Any]] = []  # seznam obchodů pro tabulku
        self._open_trade: dict[str, Any] | None = None  # otevřený obchod

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

        # UI práh (hystereze) - z user_settings (načteno z Tab 3)
        s = self.user_settings.get("entry_threshold", self.config.sensitivity)
        self._curr_entry_thr = float(s) if isinstance(s, (int, float)) else self.config.sensitivity
        self._curr_exit_thr  = max(0.0, min(self._curr_entry_thr - 0.05, self._curr_entry_thr))

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
        
        # Cesta k modelu
        g.addWidget(QLabel("Cesta:"), 0, 0); g.addWidget(self.le_model_path, 0, 1); g.addWidget(self.btn_model, 0, 2)
        
        # Nastavení modelu: READ-ONLY info panel (v samostatném GroupBoxu)
        settings_box = QGroupBox("⚙️ Nastavení z Tab 3 (read-only)")
        settings_layout = QVBoxLayout()
        
        self.lbl_decision_threshold = QLabel("Decision Threshold: –")
        self.lbl_entry_threshold = QLabel("Entry Threshold: –")
        self.lbl_exit_threshold = QLabel("Exit Threshold: –")
        self.lbl_ma_only = QLabel("MA-Only: –")
        self.lbl_and_ensemble = QLabel("AND Ensemble: –")
        
        # Styl info panelu - viditelný text s bordelem
        for lbl in [self.lbl_decision_threshold, self.lbl_entry_threshold, self.lbl_exit_threshold, 
                    self.lbl_ma_only, self.lbl_and_ensemble]:
            lbl.setStyleSheet(
                "color: #000; font-size: 9pt; font-weight: 500; "
                "background-color: #f5f5f5; padding: 6px 10px; "
                "border: 1px solid #999; border-radius: 4px;"
            )
            lbl.setMinimumHeight(26)
            settings_layout.addWidget(lbl)
        
        settings_box.setLayout(settings_layout)
        
        # Invert labels (diagnostic - zachová u sebe)
        invert_layout = QHBoxLayout()
        self.chk_invert_labels = QCheckBox("Invert labels 0↔1")
        self.chk_invert_labels.setToolTip("Ručně prohodí mapu tříd (0↔1). Použij jen pokud DIAG ukazuje opačnou polaritu.")
        self.chk_invert_labels.stateChanged.connect(self._on_toggle_invert_labels)
        invert_layout.addWidget(self.chk_invert_labels)
        invert_layout.addStretch()
        
        # Finální layout pro model_box
        model_layout = QVBoxLayout()
        model_layout.addLayout(g)
        model_layout.addWidget(settings_box)
        model_layout.addLayout(invert_layout)
        model_box.setLayout(model_layout)

        # Diagnostika degradace modelu
        degradation_box = QGroupBox("📊 Diagnostika degradace modelu")
        deg_layout = QVBoxLayout()
        self.degradation_console = QPlainTextEdit()
        self.degradation_console.setReadOnly(True)
        self.degradation_console.setMaximumHeight(120)
        self.degradation_console.setPlainText("(Žádný model načten)")
        deg_layout.addWidget(self.degradation_console)
        degradation_box.setLayout(deg_layout)

        # Log
        log_box = QGroupBox("Log")
        lv = QVBoxLayout()
        self.console = QPlainTextEdit(); self.console.setReadOnly(True)
        lv.addWidget(self.console)
        log_box.setLayout(lv)

        # Obchody
        trades_box = QGroupBox("Obchody")
        tv = QVBoxLayout()
        self.tbl_trades = QTableWidget()
        self.tbl_trades.setColumnCount(5)
        self.tbl_trades.setHorizontalHeaderLabels(["Čas", "Směr", "Vstup", "Výstup", "PnL"])
        self.tbl_trades.horizontalHeader().setStretchLastSection(True)
        tv.addWidget(self.tbl_trades)
        trades_box.setLayout(tv)

        # Grafy
        center = self._create_center_charts()

        # Layout
        left = QVBoxLayout()
        left.addWidget(status_box)
        left.addWidget(session_box)
        left.addWidget(model_box)
        left.addWidget(degradation_box)
        left.addWidget(log_box, 1)
        left.addWidget(trades_box, 1)

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
        self.chk_invert_labels.stateChanged.connect(self._on_toggle_invert_labels)

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
        classes_summary = []
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
            
            self._append_log(f"[MODEL] Načítám: {p}")
            
            try:
                obj = joblib.load(p)
                pred, meta = _extract_predictor(obj)
                
                # načti meta z pkl-sidecaru (VŽDY, i když _extract_predictor něco vrátil)
                meta_path = Path(p).with_name(Path(p).stem + "_meta.json")
                self._append_log(f"[META] Hledám metadata: {meta_path}")
                
                if meta_path.exists():
                    try:
                        with meta_path.open("r", encoding="utf-8") as fh:
                            loaded_meta = jsonlib.load(fh)
                        
                        # Merge metadata (sidecar má prioritu)
                        if isinstance(loaded_meta, dict):
                            meta.update(loaded_meta)  # Přepíše prázdné meta z PKL
                            self._append_log(f"[META] ✅ Načteno {len(loaded_meta)} klíčů z {meta_path.name}")
                            self._append_log(f"[META] Klíče: {list(loaded_meta.keys())[:10]}")
                        else:
                            self._append_log(f"[META] ⚠️ Metadata nejsou dict: {type(loaded_meta)}")
                    except Exception as ex:
                        self._append_log(f"[META] ❌ Chyba při čtení {meta_path.name}: {ex}")
                        import traceback
                        self._append_log(f"[DEBUG] {traceback.format_exc()[:500]}")
                else:
                    self._append_log(f"[META] ⚠️ Soubor neexistuje: {meta_path}")
                    self._append_log(f"[META] Zkouším absolutní cestu: {meta_path.absolute()}")
                    if not meta_path.absolute().exists():
                        self._append_log(f"[META] ❌ Ani absolutní cesta neexistuje")

                # map tříd (poprvé převezmeme)
                if not label_map_final:
                    meta_map = None
                    if isinstance(meta, dict):
                        meta_map = meta.get("class_to_dir")
                    parsed_map = {}
                    if isinstance(meta_map, dict):
                        for k, v in meta_map.items():
                            try:
                                parsed_map[int(k)] = str(v).upper()
                            except Exception:
                                pass
                    inferred_map = _infer_label_map_from_classes(getattr(pred, "classes_", None), base_map=None)
                    self.class_to_dir = parsed_map if parsed_map else inferred_map
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
                    sample = ", ".join(exp_list[:10])
                    more = "" if len(exp_list) <= 10 else f" (+{len(exp_list) - 10} more)"
                    self._append_log(f"[FEATS] Model expects {len(exp_list)} features: {sample}{more}")

                loaded.append({
                    "predictor": pred,
                    "path": p,
                    "exp_feats": exp_list if exp else None,
                    "label_map": dict(self.class_to_dir),
                    "metadata": meta,  # uložit metadata pro později
                })

                try:
                    classes_dbg = list(getattr(pred, "classes_", []))
                except Exception:
                    classes_dbg = []
                classes_summary.append({
                    "model": os.path.basename(p),
                    "classes": classes_dbg,
                })
                self._append_log(
                    f"[TAB4-DIAG] class_map model={os.path.basename(p)} classes={classes_dbg} map={loaded[-1]['label_map']}"
                )
                
                # Debug: Zkontroluj, že metadata obsahují očekávané klíče
                meta_keys = list(meta.keys()) if isinstance(meta, dict) else []
                has_train = "metrics_train" in meta
                has_holdout = "metrics_holdout" in meta
                self._append_log(f"[META] Model metadata obsahuje {len(meta_keys)} klíčů")
                self._append_log(f"[META] metrics_train: {has_train}, metrics_holdout: {has_holdout}")
                
                self._append_log(f"[INFO] ✅ Načten model: {os.path.basename(p)}")
            except Exception as e:
                self._append_log(f"[ERROR] Načtení modelu selhalo ({p}): {e}")
                return False

        # uložit členy ensemble
        self.models = loaded
        self.model = loaded[0]["predictor"] if loaded else None  # jen pro feature_names_in_

        if classes_summary:
            uniq_counts = {}
            for it in classes_summary:
                key = tuple(it.get("classes") or [])
                uniq_counts[key] = int(uniq_counts.get(key, 0)) + 1
            combos = " | ".join(
                [f"classes={list(k)} x{v}" for k, v in sorted(uniq_counts.items(), key=lambda kv: (len(kv[0]), str(kv[0])))]
            )
            self._append_log(
                f"[TAB4-DIAG] startup models={len(classes_summary)} unique={len(uniq_counts)} {combos}"
            )

        # Cada model usa sus propias features - sin intersección
        base_cols = ['close', 'ma_fast', 'ma_slow', 'atr', 'average']
        if feats_sets:
            self._append_log(f"[INFO] Ensemble mód: {len(feats_lists)} modelů, cada uno usa suas próprias features")
            for i, exp_list in enumerate(feats_lists):
                sample = ", ".join(exp_list[:5])
                more = "" if len(exp_list) <= 5 else f" (+{len(exp_list) - 5} mais)"
                self._append_log(f"  [M{i}] {len(exp_list)} features: {sample}{more}")
            # Cada modelo usará sus propias features, no la intersección
            self.model_expected_features = base_cols  # fallback only for MA etc
        else:
            self.model_expected_features = base_cols

        # pojmenování vrstev
        self._append_log(f"[LAYERS] L0=MA | L1i=Model_i | L1_AND=AND přes {len(self.models)} modelů | L2_AND=(volitelně) MA ∧ L1_AND")
        
        # Načti user_settings z prvního modelu (je-li k dispozici)
        self._load_user_settings_from_first_model()
        
        return True

    def _load_user_settings_from_first_model(self) -> None:
        """Načte user_settings z metadat prvního modelu a zobrazí je jako read-only info panel."""
        if not self.models or not self.models[0]:
            self._append_log("[SETTINGS] ❌ Žádné modely načteny")
            self._update_settings_display({})
            self._load_reference_metrics({})
            return
        
        try:
            # Vezmi metadata z prvního modelu
            first_model = self.models[0]
            metadata = first_model.get("metadata") or {}
            
            self._append_log(f"[SETTINGS] Metadata typ: {type(metadata)}, velikost: {len(metadata) if isinstance(metadata, dict) else 'N/A'}")
            
            if not isinstance(metadata, dict):
                self._append_log(f"[SETTINGS] ❌ Metadata nejsou dict: {type(metadata)}")
                metadata = {}
            
            if not metadata:
                self._append_log("[SETTINGS] ⚠️ Metadata jsou prázdný dict")
            else:
                self._append_log(f"[SETTINGS] Metadata klíče: {list(metadata.keys())[:15]}")
            
            user_settings = metadata.get("user_settings") or {}
            
            # Načti referenční metriky pro degradation diagnostics
            self._load_reference_metrics(metadata)
            
            # Načti historická data pro okamžitou degradation diagnostiku
            self._preload_historical_data_for_degradation()
            
            if not user_settings:
                self._append_log("[INFO] Žádná uložená nastavení v modelu - ponechávám defaults")
                self._update_settings_display({})
                return
            
            # Zobraz nastavení v info panelu
            self._update_settings_display(user_settings)
            self._append_log("[SETTINGS] ✅ Nastavení modelu načtena z Tab 3")
            
        except Exception as e:
            self._append_log(f"[WARN] Nelze načít user_settings: {e}")
            self._update_settings_display({})
            self._load_reference_metrics({})
    
    def _load_reference_metrics(self, metadata: dict) -> None:
        """Extrahuje referenční metriky z metadata modelu (holdout preferovaně, jinak train)."""
        if not metadata:
            msg = "❌ PRÁZDNÁ METADATA\n\nModel neobsahuje žádná metadata.\nOvěřte, že existuje soubor *_meta.json vedle .pkl souboru."
            self.degradation_console.setPlainText(msg)
            self._append_log("[DEGRADATION] ❌ Metadata jsou prázdná")
            self.reference_metrics = {}
            return
        
        # Debug: Co je v metadatech?
        self._append_log(f"[DEGRADATION] Metadata klíče: {list(metadata.keys())[:20]}")
        
        # Načti metriky
        holdout = metadata.get("metrics_holdout")
        train = metadata.get("metrics_train")
        
        # Fallback: hledej jiné klíče s metrikami
        if not holdout and not train:
            # Zkus generic "metrics" klíč
            generic_metrics = metadata.get("metrics")
            if generic_metrics and isinstance(generic_metrics, dict):
                # Pokud obsahuje nested strukturu (train/holdout)
                if "holdout" in generic_metrics:
                    holdout = generic_metrics["holdout"]
                elif "train" in generic_metrics:
                    train = generic_metrics["train"]
                else:
                    # Použij přímo jako train metriky
                    train = generic_metrics
        
        self._append_log(f"[DEGRADATION] metrics_train nalezeny: {bool(train)}")
        self._append_log(f"[DEGRADATION] metrics_holdout nalezeny: {bool(holdout)}")
        
        if holdout:
            self.reference_metrics = holdout
            ref_source = "holdout"
            self._append_log(f"[DEGRADATION] ✅ Použity HOLDOUT metriky ({len(holdout)} klíčů)")
        elif train:
            self.reference_metrics = train
            ref_source = "train (OOF)"
            self._append_log(f"[DEGRADATION] ✅ Použity TRAIN metriky ({len(train)} klíčů)")
        else:
            self.reference_metrics = {}
            available_keys = [k for k in metadata.keys() if "metric" in k.lower()]
            msg = (
                f"❌ Žádné referenční metriky v metadata\n\n"
                f"Hledal jsem: metrics_train, metrics_holdout\n"
                f"Metadata obsahuje {len(metadata)} klíčů\n"
                f"Klíče s 'metric': {available_keys if available_keys else 'žádné'}\n\n"
                f"Všechny klíče:\n" + "\n".join(f"  • {k}" for k in list(metadata.keys())[:20])
            )
            self.degradation_console.setPlainText(msg)
            self._append_log("[DEGRADATION] ❌ Reference metriky nenalezeny")
            return
        
        # Zobraz info o referenčních metrikách
        ref_f1 = self.reference_metrics.get("f1", "?")
        ref_acc = self.reference_metrics.get("accuracy", "?")
        ref_sharpe = self.reference_metrics.get("sharpe", "?")
        ref_profit = self.reference_metrics.get("profit_net", "?")
        
        # Formátuj hodnoty (nelze použít podmínky přímo v format specifieru)
        acc_str = f"{ref_acc:.4f}" if isinstance(ref_acc, (int, float)) else str(ref_acc)
        f1_str = f"{ref_f1:.4f}" if isinstance(ref_f1, (int, float)) else str(ref_f1)
        sharpe_str = f"{ref_sharpe:.4f}" if isinstance(ref_sharpe, (int, float)) else str(ref_sharpe)
        profit_str = f"{ref_profit:.2f}" if isinstance(ref_profit, (int, float)) else str(ref_profit)
        
        info_text = (
            f"📌 Referenční metriky ({ref_source}):\n"
            f"   Accuracy: {acc_str}\n"
            f"   F1: {f1_str}\n"
            f"   Sharpe: {sharpe_str}\n"
            f"   Profit Net: {profit_str}\n"
            f"\n⏳ Čekám na {self.degradation_window_size} barů pro live diagnostiku..."
        )
        self.degradation_console.setPlainText(info_text)
    
    def _preload_historical_data_for_degradation(self) -> None:
        """
        Načte historická data z TradingView a naplní buffery pro degradation diagnostiku.
        Umožní okamžitou diagnostiku bez čekání na 500+ nových barů.
        """
        if not self.models:
            self._append_log("[DEGRADATION] Přeskakuji preload - žádné modely načteny")
            self.degradation_console.setPlainText("(Načtěte model pro spuštění diagnostiky)")
            return
        
        if not self.reference_metrics:
            self._append_log("[DEGRADATION] Přeskakuji preload - žádné referenční metriky")
            self.degradation_console.setPlainText("(Model neobsahuje referenční metriky)")
            return
        
        try:
            from pathlib import Path
            
            self._append_log("[DEGRADATION] 🔄 Načítám historická data pro okamžitou diagnostiku...")
            self.degradation_console.setPlainText("⏳ Načítám historická data z TradingView...\nMůže trvat několik sekund...")
            
            # Načti potřebný počet barů (+ extra pro výpočet feature s rolling window)
            bars_needed = self.degradation_window_size + 200  # +200 pro MA/ATR warmup
            
            # Použij aktuální symbol/exchange z konfigurace (nebo z modelu)
            symbol = (self.ed_symbol.text() or "GOLD").strip()
            exchange = (self.ed_expiry.text() or "TVC").strip()
            timeframe = self.cmb_interval.currentText().replace("mins", "min")
            
            self._append_log(f"[DEGRADATION] Symbol={symbol}, Exchange={exchange}, TF={timeframe}, Bars={bars_needed}")
            
            from ibkr_trading_bot.core.datasource.tradingview_client import TradingViewClient
            tv = TradingViewClient(
                username=os.getenv("TV_USERNAME"),
                password=os.getenv("TV_PASSWORD")
            )
            
            self._append_log("[DEGRADATION] TradingView client vytvořen, stahuji data...")
            df = tv.get_history(symbol, exchange, timeframe, limit=bars_needed)
            
            if df is None or df.empty:
                msg = "❌ Nelze načíst historická data z TradingView"
                self._append_log(f"[WARN] {msg}")
                self.degradation_console.setPlainText(f"{msg}\nDiagnostika bude dostupná po načtení {self.degradation_window_size} live barů.")
                return
            
            self._append_log(f"[DEGRADATION] Staženo {len(df)} historických barů, připravuji data...")
            
            # Připrav data
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
            
            self._append_log(f"[DEGRADATION] Data připravena ({len(df)} barů po cleanupu), počítám features...")
            
            # Připrav index podle očekávání feature_engineering (timestamp v UTC)
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).set_index("timestamp")
            df.index.name = "timestamp"
            
            # Vypočítej features pomocí compute_all_features
            from ibkr_trading_bot.features.feature_engineering import compute_all_features
            
            df_feats = compute_all_features(df)
            
            if df_feats.empty:
                msg = "❌ Feature calculation selhala"
                self._append_log(f"[WARN] {msg}")
                self.degradation_console.setPlainText(f"{msg}\nDiagnostika bude dostupná po načtení {self.degradation_window_size} live barů.")
                return
            
            self._append_log(f"[DEGRADATION] Features vypočítány ({len(df_feats)} barů, {len(df_feats.columns)} sloupců)")
            
            # Vezmi posledních degradation_window_size barů
            df_recent = df_feats.tail(self.degradation_window_size).copy().reset_index(drop=True)
            
            self._append_log(f"[DEGRADATION] Používám posledních {len(df_recent)} barů, spouštím predikce...")
            
            # Vypočítej predikce modelu pro CELÝ DataFrame najednou (efektivnější a bez feature warnings)
            model = self.models[0]["predictor"]
            exp_feats = self.models[0].get("exp_feats")
            
            # Připrav celý DataFrame pro model
            X_prepared = df_recent
            if exp_feats:
                X_prepared = self._prepare_X_for_model(df_recent, exp_feats)
            
            try:
                # Vypočítej predikce pro všechny řádky najednou
                label_map = self.models[0].get("label_map") or _infer_label_map_from_classes(getattr(model, "classes_", None))
                
                # Batch prediction - rychlejší a bez warningů
                X_pred = _align_X_for_model(model, X_prepared)
                proba_all = _predict_proba_safely(model, X_pred)
                
                # Určení indexů LONG a SHORT v classes_
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    # Textové classes
                    if any(isinstance(c, str) for c in classes):
                        lut = {str(c).upper(): i for i, c in enumerate(classes)}
                        idx_long = lut.get("LONG")
                        idx_short = lut.get("SHORT")
                    # Numerické classes s label_map
                    else:
                        idx_long = next((i for i, c in enumerate(classes)
                                       if str(label_map.get(int(c), "")).upper() == "LONG"), None)
                        idx_short = next((i for i, c in enumerate(classes)
                                        if str(label_map.get(int(c), "")).upper() == "SHORT"), None)
                else:
                    # Fallback: předpokládej binární klasifikaci
                    idx_long = 1
                    idx_short = 0
                
                # Konverze proba → predictions (-1/0/+1)
                predictions = []
                for proba_row in proba_all:
                    pL = float(proba_row[idx_long]) if idx_long is not None else 0.5
                    pS = float(proba_row[idx_short]) if idx_short is not None else 0.5
                    
                    if pL > 0.5:
                        predictions.append(1)  # LONG
                    elif pS > 0.5:
                        predictions.append(-1)  # SHORT
                    else:
                        predictions.append(0)  # NEUTRAL
                
                prices = df_recent["close"].astype(float).tolist()
                timestamps = df_recent["time"].tolist()
                
                self._append_log(f"[DEGRADATION] Predikce dokončeny: {len(predictions)} barů zpracováno")
                
            except Exception as e:
                self._append_log(f"[ERROR] Batch predikce selhala: {e}")
                import traceback
                self._append_log(f"[DEBUG] {traceback.format_exc()}")
                
                # Fallback: prázdné buffery
                predictions = [0] * len(df_recent)
                prices = df_recent["close"].astype(float).tolist()
                timestamps = df_recent["time"].tolist()
            
            # Naplň buffery
            self._prediction_buffer = predictions
            self._price_buffer = prices
            self._y_true_buffer = [None] * len(predictions)
            
            # Označ všechny timestampy jako trackované
            self._tracked_timestamps = {str(ts) for ts in timestamps}
            
            self._append_log(f"[DEGRADATION] ✅ Načteno {len(predictions)} historických barů")
            
            # Spusť okamžitou diagnostiku
            if len(self._prediction_buffer) >= self.degradation_window_size:
                self._update_degradation_diagnostics()
                self._last_degradation_check = len(self._prediction_buffer)
                self._append_log("[DEGRADATION] ✅ Okamžitá diagnostika spuštěna")
            else:
                msg = f"📊 Načteno {len(self._prediction_buffer)} barů (potřeba {self.degradation_window_size})"
                self._append_log(f"[DEGRADATION] {msg}")
                self.degradation_console.setPlainText(msg)
            
        except Exception as e:
            error_msg = f"❌ Preload historických dat selhal: {e}"
            self._append_log(f"[ERROR] {error_msg}")
            import traceback
            traceback_str = traceback.format_exc()
            self._append_log(f"[DEBUG] {traceback_str}")
            
            # Zobraz chybu i v degradation konzoli pro uživatele
            self.degradation_console.setPlainText(
                f"{error_msg}\n\n"
                f"Detail:\n{traceback_str[:500]}\n\n"
                f"Diagnostika bude dostupná po načtení {self.degradation_window_size} live barů."
            )
    
    def _update_settings_display(self, user_settings: dict) -> None:
        """Aktualizuje display panelu s nastavením modelu (read-only) a uloží do self.user_settings."""
        self.user_settings = user_settings  # uložit pro použití v predikci
        
        decision_threshold = user_settings.get("decision_threshold", "–")
        entry_threshold = user_settings.get("entry_threshold", "–")
        exit_threshold = user_settings.get("exit_threshold", "–")
        use_ma_only = user_settings.get("use_ma_only", False)
        use_and_ensemble = user_settings.get("use_and_ensemble", True)
        
        self.lbl_decision_threshold.setText(f"Decision Threshold: {decision_threshold}")
        self.lbl_entry_threshold.setText(f"Entry Threshold: {entry_threshold}")
        self.lbl_exit_threshold.setText(f"Exit Threshold: {exit_threshold}")
        self.lbl_ma_only.setText(f"MA-Only: {'✓ zapnuto' if use_ma_only else '✗ vypnuto'}")
        self.lbl_and_ensemble.setText(f"AND Ensemble: {'✓ AND' if use_and_ensemble else '✗ VOTE'}")

        # Synchronizuj runtime chování Tab 4 s nastavením načteným z Tab 3
        self.config.use_ma_only = bool(use_ma_only)
        self.config.use_and_ensemble = bool(use_and_ensemble)
        
        # Aplikuj entry/exit thresholdy na aktivní konfiguraci
        if isinstance(entry_threshold, (int, float)):
            self._curr_entry_thr = float(entry_threshold)
        if isinstance(exit_threshold, (int, float)):
            self._curr_exit_thr = float(exit_threshold)
        
        # Log po aktualizaci
        if user_settings:
            self._append_log(f"[SETTINGS] Decision={decision_threshold}, Entry={entry_threshold}, Exit={exit_threshold}, MA-Only={use_ma_only}, AND={use_and_ensemble}")

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
                X_use = self._prepare_X_for_model(Xrow, exp)
            try:
                label_map = m.get("label_map") or self.class_to_dir  # per-model mapa 1st
                pL, pS, classes_i, raw_proba = _extract_long_short_proba(mdl, X_use, label_map=label_map)
            except Exception:
                pL, pS, classes_i, raw_proba = 0.5, 0.5, None, None

            # DIAG logging (prvních 10 záznamů)
            if not hasattr(self, "_diag_counter"):
                self._diag_counter = 0
            if self._diag_counter < 10:
                self._append_log(f"[DIAG] classes={classes_i} pL={pL:.3f} pS={pS:.3f} from {type(mdl).__name__}")
                self._diag_counter += 1

        # výběr směru a konfidence (u ternary respektuj i HOLD)
            direction, conf = _pick_direction_from_raw_proba(classes_i, raw_proba, label_map)

            if direction is None:
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

    # Majority-vote přes všechny modely (model-only, bez MA filtru)
    def _predict_one_label_VOTE(self, Xrow: pd.DataFrame) -> tuple[int, float, list[str], list[float]]:
        """
        Vrací (label {-1,0,+1}, conf_vote, directions, confs)
        label=0 při remíze nebo když není jasná většina.
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
                X_use = self._prepare_X_for_model(Xrow, exp)
            try:
                label_map = m.get("label_map") or self.class_to_dir
                pL, pS, classes_i, raw_proba = _extract_long_short_proba(mdl, X_use, label_map=label_map)
            except Exception:
                pL, pS, classes_i, raw_proba = 0.5, 0.5, None, None

            direction, conf = _pick_direction_from_raw_proba(classes_i, raw_proba, label_map)

            if direction is None:
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

        n_long = sum(1 for d in dirs if d == "LONG")
        n_short = sum(1 for d in dirs if d == "SHORT")
        if n_long == n_short:
            return 0, float(np.mean(confs)) if confs else 0.0, dirs, confs

        if n_long > n_short:
            conf_vote = float(np.mean([c for d, c in zip(dirs, confs) if d == "LONG"]))
            return +1, conf_vote, dirs, confs
        conf_vote = float(np.mean([c for d, c in zip(dirs, confs) if d == "SHORT"]))
        return -1, conf_vote, dirs, confs

    def _prepare_X_for_model(self, Xrow: pd.DataFrame, exp: list[str]) -> pd.DataFrame:
        """Připrav řádek pro model. Chybné features vyplní mediánem z dostupných dat."""
        X_use = Xrow.copy()
        missing = [c for c in exp if c not in X_use.columns]
        
        if missing:
            key = f"missing_feats_{hash(tuple(exp))}"
            if not hasattr(self, "_diag_once"):
                self._diag_once = {}
            if key not in self._diag_once:
                sample = ", ".join(missing[:5])
                more = "" if len(missing) <= 5 else f" (+{len(missing)-5} more)"
                self._append_log(f"[WARN] Chybí features ({len(missing)}/{len(exp)}): {sample}{more}")
                self._diag_once[key] = True
            
            # Doplň chybné features mediánem z existujícíích dat
            for c in missing:
                # Vezmi medián z jiného sloupce (např. 'close' pokud chybí indikátor)
                if "close" in X_use.columns:
                    X_use[c] = X_use["close"].median()
                else:
                    # Fallback: medián z prvního numerického sloupce
                    numeric_cols = X_use.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        X_use[c] = X_use[numeric_cols[0]].median()
                    else:
                        X_use[c] = 0.0
        
        X_use = X_use[exp]
        return X_use.astype(float)

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
            # Použij entry_threshold z user_settings (náklad z Tab 3), nebo fallback
            entry_thr = self.user_settings.get("entry_threshold", self.config.sensitivity)
            self.warm = LiveWarmupService(
                base_service=adapter,
                config=WarmupConfig(
                    threshold=float(entry_thr),
                    warmup_bars=500,
                    min_sim_trades=3,
                    start_sharpe=0.00,
                    max_dd=25.0,
                    diag_first_n=220,
                    force_live_after_warmup=True,
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
        
        # Udržuj VĚTŠÍ buffer (700 barů) pro rolling indicators - neřezat na max_bars_buffer!
        # Display se ořeže až při rendering
        max_live_buffer = max(700, self.config.max_bars_buffer + 400)  # +400 pro rolling warmup
        if len(self.live_df) > max_live_buffer:
            self.live_df = self.live_df.tail(max_live_buffer).reset_index(drop=True)

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
            # Stáhnout více barů pro rolling warmup (700), pak ořezat na max_bars_buffer (300)
            initial_download = max(700, int(self.config.max_bars_buffer) + 200)  # +200 pro rolling warmup
            df = tv.get_history(self.config.symbol, self.config.exchange, tf_label, limit=initial_download)
            if df is not None and not df.empty:
                df = df.copy()
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = df.dropna(subset=["time"]).sort_values("time")
                # Nevškrtnúť na max_bars_buffer hned - počkáme až po compute_features (kde se dropnou první řádky)

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
                # NIKDY NEŘEZAT live_df - potřebujeme plnou historii pro rolling indicators!
                # Ořezávání se řeší až v _rescore_all() output
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
                    try:
                        cls_vals = [int(c) for c in list(getattr(mdl, "classes_", []))]
                    except Exception:
                        cls_vals = []
                    if len(set(cls_vals)) > 2:
                        continue
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
            vv = str(v).upper()
            if vv == "SHORT":
                inv[k] = "LONG"
            elif vv == "LONG":
                inv[k] = "SHORT"
            else:
                inv[k] = vv
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
        """Vypočítej všechny indikátory a featury pro daný DataFrame."""
        df = df.copy()
        if "timestamp" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        df_features = compute_all_features(df)
        
        df_features["ma_fast"] = df_features["close"].rolling(9, min_periods=1).mean()
        df_features["ma_slow"] = df_features["close"].rolling(21, min_periods=1).mean()
        if "average" not in df_features.columns:
            df_features["average"] = (
                df_features["open"] + df_features["high"] + df_features["low"] + df_features["close"]
            ) / 4.0
        return df_features

    def _get_raw_indicators(self) -> pd.DataFrame | None:
        if self.live_df is None or self.live_df.empty:
            return None
        df = self.live_df.rename(columns={"timestamp": "date"})
        df = df[["date","open","high","low","close","volume"]].dropna(subset=["close"]).copy()
        if df.empty:
            return None
        
        # compute_all_features() vrátí DataFrame bez prvních ~26 řádků (MACD warmup)
        # a bez řádků s NaN v klíčových indicatorech
        ind = self._compute_indicators(df)
        
        if ind.empty:
            return None
        
        if "timestamp" in ind.columns:
            ind["timestamp"] = pd.to_datetime(ind["timestamp"], utc=True, errors="coerce")
            ind = ind.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        
        # ✅ Zbyly jen validní řádky - bez NaN v klíčových featurách
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

    def _sanitize_feature_matrix(self, feat: pd.DataFrame) -> pd.DataFrame:
        """Ponechá všechny dostupné featury, jen je převede na numerický tvar + imputuje NaN."""
        df = feat.copy()
        if 'average' not in df.columns and all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            df['average'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0

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
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna(subset=['close']).copy()
        if df.empty:
            return None
        
        # Vezmi posledních 100 barů (ne jen 1!) pro správný výpočet rolling indicators
        # Rolling windows (RSI, ATR, MACD) potřebují dostatek dat, aby se správně počítaly
        tail_bars = min(100, len(df))
        df_tail = df.iloc[-tail_bars:].copy()
        
        feat_df = self._compute_indicators(df_tail)
        last = feat_df.iloc[[-1]].copy()
        return self._sanitize_feature_matrix(last)

    def _build_features_for_all(self) -> pd.DataFrame | None:
        if self.live_df is None or self.live_df.empty:
            return None
        df = self.live_df.rename(columns={"timestamp": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"]).copy()
        if df.empty:
            return None
        feat = self._compute_indicators(df)
        if "date" in feat.columns:
            feat["date"] = pd.to_datetime(feat["date"], utc=True, errors="coerce")
            feat = feat.dropna(subset=["date"]).set_index("date").sort_index()
        elif "timestamp" in feat.columns:
            feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True, errors="coerce")
            feat = feat.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        elif not isinstance(feat.index, pd.DatetimeIndex):
            return None
        return self._sanitize_feature_matrix(feat)

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
        if s in ("-1", "short", "sell", "down"):
            return "SHORT"
        if s in ("0", "hold", "flat", "neutral", "none"):
            return "FLAT"
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

        feats = self._sanitize_feature_matrix(raw)
        # Praktikovat thresh z user_settings (z Tab 3), nebo fallback
        thr = self.user_settings.get("entry_threshold", self._curr_entry_thr) if self.user_settings else self._curr_entry_thr
        thr = float(thr) if isinstance(thr, (int, float)) else self._curr_entry_thr

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
            
            # Sleduj obchody i v MA-only režimu
            try:
                self._update_position_and_trades(raw)
            except Exception as e:
                self._append_log(f"[WARN] Sledování obchodů v MA-only režimu selhalo: {e}")
            return

        # Ensemble AND (volitelně MA ∧ L1_AND)
        use_ma_and = self.config.use_and_ensemble

        n_mapped = n_shown = n_long = n_short = 0
        n_none = 0
        n_l1_flat = 0
        for ts, d0 in zip(feats.index, l0_dir):
            idx = self._nearest_bar_index(ts)
            if idx is None:
                continue
            Xrow = feats.loc[[ts]]

            if use_ma_and:
                thr_model = 0.0  # nefiltruj směr uvnitř AND, prahy řeší hysterese
                label, conf_min, dirs, confs = self._predict_one_label_AND(Xrow, thr_model)
            else:
                label, conf_min, dirs, confs = self._predict_one_label_VOTE(Xrow)
            l1 = "LONG" if label == +1 else "SHORT" if label == -1 else "FLAT"
            if l1 == "FLAT":
                n_l1_flat += 1

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

            n_mapped += 1
            if final == "LONG":  n_long += 1; n_shown += 1
            elif final == "SHORT":
                n_short += 1
                n_shown += 1
            else:
                n_none += 1

        # Sleduj obchody po všech signálech
        try:
            self._update_position_and_trades(raw)
        except Exception as e:
            self._append_log(f"[WARN] Sledování obchodů selhalo: {e}")

        self._append_log(
            f"[RESCORE] bars={len(self._bars)} mapped={n_mapped} shown={n_shown} "
            f"(LONG={n_long}, SHORT={n_short}) thr={thr:.2f} models={len(self.models)} AND_MA={use_ma_and}"
        )
        self._append_log(
            f"[TAB4-DIAG] signal_dist mapped={n_mapped} LONG={n_long} SHORT={n_short} ACTIVE={n_shown} NONE={n_none} L1_FLAT={n_l1_flat}"
        )

        # Track predictions a ceny pro degradation diagnostics
        try:
            self._track_predictions_for_degradation(raw)
        except Exception as e:
            self._append_log(f"[WARN] Tracking degradace selhal: {e}")

    # ----------
    def _update_position_and_trades(self, raw: pd.DataFrame) -> None:
        """
        Aktualizuje pozici a obchody na základě signálů uložených v self._bars.
        Volá se z obou cest (MA-only i ensemble).
        Také kontroluje, zda je poslední bar.
        """
        if not self._bars:
            return
        
        last_bar = self._bars[-1]
        ts = pd.to_datetime(last_bar.get("time"), utc=True, errors="coerce")
        if pd.isna(ts) or ts not in raw.index:
            return
        
        final = last_bar.get("signal")
        if final is None:
            return
        
        if final == "LONG":
            if self._live_pos <= 0:  # vstup/otočka
                # Uzavři předchozí obchod, pokud existuje
                if self._open_trade is not None:
                    exit_price = float(raw.loc[ts, "close"])
                    pnl = exit_price - self._open_trade["entry_price"] if self._open_trade["direction"] == "LONG" else self._open_trade["entry_price"] - exit_price
                    self._add_trade_to_table(
                        self._open_trade["entry_time"], self._open_trade["direction"],
                        self._open_trade["entry_price"], str(ts)[:19], exit_price, pnl
                    )
                    self._trades.append({
                        "entry_time": self._open_trade["entry_time"],
                        "direction": self._open_trade["direction"],
                        "entry_price": self._open_trade["entry_price"],
                        "exit_time": str(ts)[:19],
                        "exit_price": exit_price,
                        "pnl": pnl
                    })
                # Otevři nový LONG
                self._live_pos = +1
                self._live_entry_px = float(raw.loc[ts, "close"])
                self._open_trade = {
                    "direction": "LONG",
                    "entry_time": str(ts)[:19],
                    "entry_price": self._live_entry_px
                }
        elif final == "SHORT":
            if self._live_pos >= 0:
                # Uzavři předchozí obchod, pokud existuje
                if self._open_trade is not None:
                    exit_price = float(raw.loc[ts, "close"])
                    pnl = exit_price - self._open_trade["entry_price"] if self._open_trade["direction"] == "LONG" else self._open_trade["entry_price"] - exit_price
                    self._add_trade_to_table(
                        self._open_trade["entry_time"], self._open_trade["direction"],
                        self._open_trade["entry_price"], str(ts)[:19], exit_price, pnl
                    )
                    self._trades.append({
                        "entry_time": self._open_trade["entry_time"],
                        "direction": self._open_trade["direction"],
                        "entry_price": self._open_trade["entry_price"],
                        "exit_time": str(ts)[:19],
                        "exit_price": exit_price,
                        "pnl": pnl
                    })
                # Otevři nový SHORT
                self._live_pos = -1
                self._live_entry_px = float(raw.loc[ts, "close"])
                self._open_trade = {
                    "direction": "SHORT",
                    "entry_time": str(ts)[:19],
                    "entry_price": self._live_entry_px
                }
        else:
            # FLAT – uzavři případnou živou pozici
            if self._live_pos != 0:
                if self._open_trade is not None:
                    exit_price = float(raw.loc[ts, "close"])
                    pnl = exit_price - self._open_trade["entry_price"] if self._open_trade["direction"] == "LONG" else self._open_trade["entry_price"] - exit_price
                    self._add_trade_to_table(
                        self._open_trade["entry_time"], self._open_trade["direction"],
                        self._open_trade["entry_price"], str(ts)[:19], exit_price, pnl
                    )
                    self._trades.append({
                        "entry_time": self._open_trade["entry_time"],
                        "direction": self._open_trade["direction"],
                        "entry_price": self._open_trade["entry_price"],
                        "exit_time": str(ts)[:19],
                        "exit_price": exit_price,
                        "pnl": pnl
                    })
                self._live_pos = 0
                self._live_entry_px = None
                self._open_trade = None

    # ========== Degradation Diagnostics METHODS ==========
    
    def _track_predictions_for_degradation(self, raw: pd.DataFrame) -> None:
        """
        Ukládá signály a ceny do bufferů pro sledování degradace modelu.
        Volá se po každém rescoring v _rescore_all().
        Trackuje jen NOVÉ bary (deduplicita pomocí timestampů).
        """
        if raw is None or raw.empty:
            return
        
        new_bars_tracked = 0
        
        # Extrahuj NOVÉ signály a ceny z self._bars (jen ty, které jsme ještě netrackovali)
        for bar in self._bars:
            ts = pd.to_datetime(bar.get("time"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            
            # Převeď timestamp na string pro set (hashable)
            ts_key = str(ts)
            
            # Pokud už jsme tento bar trackovali, přeskoč
            if ts_key in self._tracked_timestamps:
                continue
            
            signal = bar.get("signal")  # "LONG", "SHORT" or None
            close_price = float(bar.get("close", np.nan))
            
            if np.isnan(close_price):
                continue
            
            # Převeď signál na numerickou hodnotu: LONG=+1, SHORT=-1, None/FLAT=0
            pred_value = 1 if signal == "LONG" else (-1 if signal == "SHORT" else 0)
            
            # Přidej do bufferů (jen nové bary)
            self._prediction_buffer.append(pred_value)
            self._price_buffer.append(close_price)
            
            # V paper/backtest režimu bychom měli y_true
            # V live režimu ground truth není dostupný → append None
            self._y_true_buffer.append(None)  # TODO: pokud je dostupné ground truth
            
            # Označ tento timestamp jako trackovaný
            self._tracked_timestamps.add(ts_key)
            new_bars_tracked += 1
        
        # Trim bufferů na max size (2x window pro sliding window analýzu)
        max_buffer = self.degradation_window_size * 2
        if len(self._prediction_buffer) > max_buffer:
            self._prediction_buffer = self._prediction_buffer[-max_buffer:]
            self._price_buffer = self._price_buffer[-max_buffer:]
            self._y_true_buffer = self._y_true_buffer[-max_buffer:]
            
            # Cleanup tracked timestamps - odstraň staré, které už nejsou v bufferu
            # (Necháme jen posledních max_buffer timestampů)
            # Tohle je náročné implementovat správně, tak to zatím necháme
            # Worst case: set poroste, ale to není kritické
        
        if new_bars_tracked > 0:
            self._append_log(f"[DEGRADATION] Trackováno {new_bars_tracked} nových barů. Buffer: {len(self._prediction_buffer)}/{self.degradation_window_size}")
        
        # Spusť degradation check každých N barů
        check_interval = 100  # Kontroluj každých 100 barů
        if len(self._prediction_buffer) >= self.degradation_window_size and \
           len(self._prediction_buffer) - self._last_degradation_check >= check_interval:
            self._update_degradation_diagnostics()
            self._last_degradation_check = len(self._prediction_buffer)

    
    def _update_degradation_diagnostics(self) -> None:
        """
        Vypočítá live metriky na recent window a porovná s reference metrikami z metadata.
        Zobrazí diagnostiku degradace modelu.
        """
        # Musíme mít dostatek dat
        if len(self._prediction_buffer) < self.degradation_window_size:
            remaining = self.degradation_window_size - len(self._prediction_buffer)
            self.degradation_console.setPlainText(
                f"⏳ Sbírám data pro diagnostiku...\n"
                f"   Aktuálně: {len(self._prediction_buffer)} barů\n"
                f"   Potřeba: {self.degradation_window_size} barů\n"
                f"   Zbývá: {remaining} barů"
            )
            return
        
        if not self.reference_metrics:
            self.degradation_console.setPlainText("(Žádné referenční metriky k dispozici)")
            return
        
        try:
            # Vezmi poslední N barů pro recent window
            recent_preds = np.array(self._prediction_buffer[-self.degradation_window_size:])
            recent_prices = np.array(self._price_buffer[-self.degradation_window_size:])
            
            # Importuj calculate_metrics z utils
            from ibkr_trading_bot.utils.metrics import calculate_metrics
            
            # V live režimu nemáme y_true → počítáme jen trading metriky
            # Vytvoř dummy y_true (všechny 0) protože calculate_metrics to vyžaduje
            y_true_dummy = np.zeros(len(recent_preds))
            
            # Vytvoř DataFrame s cenami
            df_recent = pd.DataFrame({"close": recent_prices})
            
            # Vypočítej metriky na recent window
            recent_metrics = calculate_metrics(
                y_true=y_true_dummy,
                y_pred=recent_preds,
                df=df_recent,
                fee_per_trade=0.0,
                slippage_bps=0.0,
                rolling_window=50,
                annualize_sharpe=False
            )
            
            self.live_metrics_recent = recent_metrics
            
            # Porovnej s reference metrikami
            self._display_degradation_comparison()
            
        except Exception as e:
            self._append_log(f"[WARN] Výpočet live metrik selhal: {e}")
            self.degradation_console.setPlainText(f"Chyba při výpočtu metrik: {e}")
    
    def _display_degradation_comparison(self) -> None:
        """Zobrazí porovnání Reference vs Live metrik v diagnostické konzoli."""
        ref = self.reference_metrics
        live = self.live_metrics_recent
        
        if not ref or not live:
            return
        
        # Extrahuj klíčové metriky
        ref_sharpe = ref.get("sharpe_net") or ref.get("sharpe", 0.0)
        live_sharpe = live.get("sharpe_net") or live.get("sharpe", 0.0)
        
        ref_profit = ref.get("profit_net", 0.0)
        live_profit = live.get("profit_net", 0.0)
        
        ref_acc = ref.get("accuracy", 0.0) if ref.get("accuracy", 0.0) else 0.0
        live_acc = live.get("accuracy", 0.0) if live.get("accuracy", 0.0) else 0.0
        
        ref_f1 = ref.get("f1", 0.0) if ref.get("f1", 0.0) else 0.0
        live_f1 = live.get("f1", 0.0) if live.get("f1", 0.0) else 0.0
        
        # Vypočítej rozdíly
        diff_sharpe = float(live_sharpe) - float(ref_sharpe) if isinstance(live_sharpe, (int, float)) and isinstance(ref_sharpe, (int, float)) else 0.0
        diff_profit = float(live_profit) - float(ref_profit) if isinstance(live_profit, (int, float)) and isinstance(ref_profit, (int, float)) else 0.0
        diff_acc = float(live_acc) - float(ref_acc) if isinstance(live_acc, (int, float)) and isinstance(ref_acc, (int, float)) else 0.0
        diff_f1 = float(live_f1) - float(ref_f1) if isinstance(live_f1, (int, float)) and isinstance(ref_f1, (int, float)) else 0.0
        
        # Formátuj hodnoty pro zobrazení (podmínky nelze dát přímo do f-string specifieru)
        ref_sharpe_str = f"{ref_sharpe:7.4f}" if isinstance(ref_sharpe, (int, float)) else f"{str(ref_sharpe):>7}"
        live_sharpe_str = f"{live_sharpe:7.4f}" if isinstance(live_sharpe, (int, float)) else f"{str(live_sharpe):>7}"
        ref_profit_str = f"{ref_profit:7.2f}" if isinstance(ref_profit, (int, float)) else f"{str(ref_profit):>7}"
        live_profit_str = f"{live_profit:7.2f}" if isinstance(live_profit, (int, float)) else f"{str(live_profit):>7}"
        ref_acc_str = f"{ref_acc:7.4f}" if isinstance(ref_acc, (int, float)) else f"{str(ref_acc):>7}"
        live_acc_str = f"{live_acc:7.4f}" if isinstance(live_acc, (int, float)) else f"{str(live_acc):>7}"
        ref_f1_str = f"{ref_f1:7.4f}" if isinstance(ref_f1, (int, float)) else f"{str(ref_f1):>7}"
        live_f1_str = f"{live_f1:7.4f}" if isinstance(live_f1, (int, float)) else f"{str(live_f1):>7}"
        
        # Formátuj zobrazení
        lines = [
            "╔════════════════════════════════════════════════════════╗",
            "║      DIAGNOSTIKA DEGRADACE MODELU                      ║",
            "╠════════════════════════════════════════════════════════╣",
            f"║ Sharpe (Ref):     {ref_sharpe_str}                        ║",
            f"║ Sharpe (Live):    {live_sharpe_str}                        ║",
            f"║ Rozdíl:           {diff_sharpe:+7.4f}                        ║",
            "║ ─────────────────────────────────────────────────────  ║",
            f"║ Profit (Ref):     {ref_profit_str}                        ║",
            f"║ Profit (Live):    {live_profit_str}                        ║",
            f"║ Rozdíl:           {diff_profit:+7.2f}                        ║",
            "║ ─────────────────────────────────────────────────────  ║",
            f"║ Accuracy (Ref):   {ref_acc_str}                        ║",
            f"║ Accuracy (Live):  {live_acc_str}                        ║",
            "║ ─────────────────────────────────────────────────────  ║",
            f"║ F1 (Ref):         {ref_f1_str}                        ║",
            f"║ F1 (Live):        {live_f1_str}                        ║",
            "╠════════════════════════════════════════════════════════╣",
        ]
        
        # Diagnóza degradace - OPRAVENÁ LOGIKA
        # Nejprve zkontroluj, zda reference nejsou podezřelé (špatný training)
        
        # Reference jsou "podezřelé" pokud:
        # - F1 = 0 (žádné signály nebo všechny špatné)
        # - Accuracy = 1.0 AND F1 = 0 (přetrénování na neutralitě)
        # - Sharpe < -0.5 (velmi špatná reference)
        
        ref_is_suspicious = (
            (isinstance(ref_f1, (int, float)) and ref_f1 < 0.05) and
            (isinstance(ref_acc, (int, float)) and ref_acc >= 0.95)
        ) or (
            isinstance(ref_sharpe, (int, float)) and ref_sharpe < -0.5
        )
        
        if ref_is_suspicious:
            # Reference jsou špatné - diagnostika se nedá provádět
            lines.append("║ ⚠️  UPOZORNĚNÍ: Referenční metriky nejsou spolehlivé   ║")
            lines.append("║    Model měl špatný výkon v tréninku (F1≈0, Acc=100%)  ║")
            lines.append("║    Live metriky nelze interpretovat jako degradaci!    ║")
            lines.append("║    → Přetrénujte model s lepšíma daty                  ║")
        else:
            # Reference jsou OK - normální diagnóza
            # Logika: Porovnávej změny v Sharpe a Profitu
            
            # Změny v klíčových metrikách
            sharpe_improved = diff_sharpe > 0.1  # Zlepšení > 0.1
            sharpe_degraded = diff_sharpe < -0.1  # Zhoršení > 0.1
            
            profit_improved = diff_profit > 10  # Profit vzrostl o 10+
            profit_degraded = diff_profit < -10  # Profit klesl o 10+
            
            f1_improved = diff_f1 > 0.1
            f1_degraded = diff_f1 < -0.1
            
            # Diagnóza na základě trend
            if sharpe_degraded or profit_degraded or f1_degraded:
                if sharpe_degraded and profit_degraded and f1_degraded:
                    lines.append("║ ❌ DEGRADACE: Model zhoršil výkon v TŘECH metrikách  ║")
                    lines.append("║    → Zvažte přetrénování modelu                        ║")
                elif sharpe_degraded or profit_degraded:
                    lines.append("║ ⚠️  MÍRNÉ ZHORŠENÍ: Live výkon pod referenční úrovní   ║")
                    lines.append("║    → Sledujte další vývoj, zvažte retraining          ║")
                else:
                    lines.append("║ ⚠️  F1 POKLES: Model dává méně signálů než v tréninku  ║")
            elif sharpe_improved or profit_improved or f1_improved:
                lines.append("║ ✅ ZLEPŠENÍ: Live výkon je lepší než reference!         ║")
                lines.append("║    Model se chová lépe než v tréninku                   ║")
            else:
                lines.append("║ ✅ STABILNÍ: Live výkon je srovnatelný s referencí      ║")
        
        lines.append("╚════════════════════════════════════════════════════════╝")
        lines.append(f"📊 Recent window: {self.degradation_window_size} barů | Last check: {len(self._prediction_buffer)} barů total")
        
        self.degradation_console.setPlainText("\n".join(lines))
    
    # ========== END Degradation Diagnostics ==========

    def _append_log(self, text: str) -> None:
        self.console.appendPlainText(text)
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.console.setTextCursor(cursor)

    def _add_trade_to_table(self, entry_time: str, direction: str, entry_price: float, exit_time: str, exit_price: float, pnl: float) -> None:
        """Přidá obchod do tabulky."""
        row = self.tbl_trades.rowCount()
        self.tbl_trades.insertRow(row)
        self.tbl_trades.setItem(row, 0, QTableWidgetItem(f"{entry_time} → {exit_time}"))
        self.tbl_trades.setItem(row, 1, QTableWidgetItem(direction))
        self.tbl_trades.setItem(row, 2, QTableWidgetItem(f"{entry_price:.2f}"))
        self.tbl_trades.setItem(row, 3, QTableWidgetItem(f"{exit_price:.2f}"))
        self.tbl_trades.setItem(row, 4, QTableWidgetItem(f"{pnl:+.2f}"))
        # Scroll to bottom
        self.tbl_trades.scrollToBottom()

    def _play_alert(self) -> None:
        try:
            if self._se is not None:
                self._se.play()
            else:
                from PySide6.QtGui import QGuiApplication
                QGuiApplication.beep()
        except Exception:
            pass

    def _play_exit_alert(self) -> None:
        """Exit alert: dvojité pípnutí, aby byl výstup odlišitelný od vstupu/směru."""
        try:
            self._play_alert()
            QTimer.singleShot(180, self._play_alert)
        except Exception:
            pass

    def _maybe_alert_flip_on_last_bar(self) -> None:
        if not self._bars:
            return

        # Zvukové alerty chceme vždy v LIVE módu Tab 4 (nikoli ve WARM-UP fázi)
        warm_state = str(getattr(getattr(self, "warm", None), "state", "")).upper()
        is_live_runtime = (warm_state == "LIVE") or (self.warm is None and str(getattr(self.config, "mode", "")).lower() == "live")
        if not is_live_runtime:
            self._last_signal = None
            return

        last = self._bars[-1]
        sig = last.get("signal")
        if sig not in ("LONG", "SHORT"):
            sig = None
        ts = pd.to_datetime(last.get("time"), utc=True, errors="coerce")
        if pd.isna(ts):
            return
        ts_ns = int(ts.value)
        if self._last_alert_bar_ns == ts_ns:
            return

        prev = self._last_signal

        # 1) Každý LONG/SHORT signál na nové svíčce = pípnutí
        if sig in ("LONG", "SHORT"):
            self._append_log(f"[ALERT] Signal {sig} @ {ts}")
            self._play_alert()

        # 2) Výstup z pozice / zrušení směru = odlišné (dvojité) pípnutí
        if prev in ("LONG", "SHORT") and sig is None:
            self._append_log(f"[ALERT] Exit {prev} → FLAT @ {ts}")
            self._play_exit_alert()

        self._last_beep_time = pd.Timestamp.now(tz="UTC")
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
