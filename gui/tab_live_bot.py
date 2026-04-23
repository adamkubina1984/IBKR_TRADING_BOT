# ==============================================
# Záložka 5) Live trading bot – TradingView only
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
from collections import deque
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from PySide6.QtCore import QSettings, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ibkr_trading_bot.core.config.presets import PRESETS_BY_TF
from ibkr_trading_bot.core.services.signal_policy import apply_live_hysteresis, build_live_proposal
from ibkr_trading_bot.core.services.trade_executor import ClosedTrade, TradeExecutor
from ibkr_trading_bot.features.feature_engineering import compute_all_features

try:
    from PySide6.QtCore import QUrl
    from PySide6.QtMultimedia import QSoundEffect
except Exception:
    QSoundEffect = None
    QUrl = None

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from ibkr_trading_bot.core.services.model_service import build_sklearn_version_warning, read_sidecar_model_meta
from ibkr_trading_bot.gui.components.workers import TaskWorker

# Warm-up service
try:
    from ibkr_trading_bot.core.services.live.live_warmup_service import LiveWarmupService, WarmupConfig
except Exception as e:
    raise ImportError(f"Nelze importovat LiveWarmupService: {e}")


# TradingView klient (různé fallback importy)
try:
    from ibkr_trading_bot.core.datasource.tradingview_client import (
        TradingViewClient,
        load_saved_tv_credentials,
        save_tv_credentials,
    )
except ModuleNotFoundError:
    try:
        from ibkr_trading_bot.core.data_sources.tradingview_client import (
            TradingViewClient,
            load_saved_tv_credentials,
            save_tv_credentials,
        )
    except ModuleNotFoundError:
        try:
            from core.datasource.tradingview_client import (
                TradingViewClient,
                load_saved_tv_credentials,
                save_tv_credentials,
            )
        except ModuleNotFoundError:
            from core.data_sources.tradingview_client import (
                TradingViewClient,
                load_saved_tv_credentials,
                save_tv_credentials,
            )

# Logger
try:
    from ibkr_trading_bot.core.utils.logging_setup import get_logger
except Exception:
    def get_logger(name: str):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

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


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(default)

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
    short_threshold: float,
    long_threshold: float,
):
    """
    Určí směr z raw pravděpodobností konzistentně s Tab 3:
    - LONG pokud p_long >= long_threshold a současně p_long >= p_short
    - SHORT pokud p_short >= short_threshold a současně p_short > p_long
    - jinak FLAT
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

        t_short = float(short_threshold)
        t_long = float(long_threshold)
        if p_long >= t_long and p_long >= p_short:
            return "LONG", float(p_long)
        if p_short >= t_short and p_short > p_long:
            return "SHORT", float(p_short)

        return "FLAT", float(max(p_long, p_short, p_hold))
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
    display_bars: int = 144  # Pocet svicek zobrazenych v grafech
    use_ma_only: bool = False
    use_and_ensemble: bool = False  # default VOTE (AND vypnuto)
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

    entry_thr: float = 0.50
    exit_thr: float = 0.50
    rounds_enabled: bool = False


@dataclass
class LiveBootstrapPayload:
    bars: list[dict[str, Any]]
    live_df: pd.DataFrame
    label_maps: list[dict[int, str] | None]
    snapshot_bars: int


@dataclass
class DegradationPreloadPayload:
    predictions: list[int]
    prices: list[float]
    timestamps: list[Any]


def _empty_live_bootstrap_payload(model_count: int = 0) -> LiveBootstrapPayload:
    return LiveBootstrapPayload(
        bars=[],
        live_df=pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        label_maps=[None] * max(0, int(model_count)),
        snapshot_bars=0,
    )


def _sanitize_live_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "average" not in out.columns and all(c in out.columns for c in ["open", "high", "low", "close"]):
        out["average"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0
    for c in out.columns:
        key = str(c).strip().lower()
        if key in {"date", "time", "timestamp", "datetime"}:
            ts = pd.to_datetime(out[c], utc=True, errors="coerce")
            out[c] = pd.Series(
                np.where(ts.notna(), ts.astype("int64"), np.nan),
                index=out.index,
                dtype="float64",
            )
        elif not pd.api.types.is_bool_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    med = out.median(numeric_only=True)
    return out.fillna(med).fillna(0.0).astype("float32")


def _compute_snapshot_features(live_df: pd.DataFrame) -> pd.DataFrame | None:
    if live_df is None or live_df.empty:
        return None
    df = live_df.copy()
    if "timestamp" not in df.columns:
        time_col = next((col for col in ("time", "date") if col in df.columns), None)
        if time_col is None:
            return None
        df = df.rename(columns={time_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna(subset=["timestamp", "close"]).copy()
    if df.empty:
        return None
    feat = compute_all_features(df)
    feat["ma_fast"] = feat["close"].rolling(9, min_periods=1).mean()
    feat["ma_slow"] = feat["close"].rolling(21, min_periods=1).mean()
    if "average" not in feat.columns:
        feat["average"] = (feat["open"] + feat["high"] + feat["low"] + feat["close"]) / 4.0
    if "timestamp" in feat.columns:
        feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True, errors="coerce")
        feat = feat.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    elif not isinstance(feat.index, pd.DatetimeIndex):
        return None
    return _sanitize_live_feature_frame(feat)


def _prepare_X_for_model_static(Xrow: pd.DataFrame, exp: list[str] | None) -> pd.DataFrame:
    X_use = Xrow.copy()
    if "average" not in X_use.columns and all(c in X_use.columns for c in ["open", "high", "low", "close"]):
        X_use["average"] = (X_use["open"] + X_use["high"] + X_use["low"] + X_use["close"]) / 4.0
    exp_list = [str(c) for c in (exp or [])]
    if exp_list:
        missing = [c for c in exp_list if c not in X_use.columns]
        for c in missing:
            if "close" in X_use.columns:
                X_use[c] = X_use["close"].median()
            else:
                numeric_cols = X_use.select_dtypes(include=[np.number]).columns
                X_use[c] = X_use[numeric_cols[0]].median() if len(numeric_cols) > 0 else 0.0
        X_use = X_use[exp_list]
    return _sanitize_live_feature_frame(X_use).astype(float)


def _build_live_bootstrap_payload_from_history_df(
    df: pd.DataFrame | None,
    models: list[dict[str, Any]],
    *,
    max_bars_buffer: int | None = None,
) -> LiveBootstrapPayload:
    if df is None or df.empty:
        return _empty_live_bootstrap_payload(len(models))

    work = df.copy()
    time_col = next((col for col in ("time", "timestamp", "date") if col in work.columns), None)
    required_cols = {"open", "high", "low", "close"}
    if time_col is None or not required_cols.issubset(work.columns):
        return _empty_live_bootstrap_payload(len(models))

    if "volume" not in work.columns:
        work["volume"] = 0.0

    work = work.rename(columns={time_col: "time"}).copy()
    work["time"] = pd.to_datetime(work["time"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    if work.empty:
        return _empty_live_bootstrap_payload(len(models))

    if max_bars_buffer is not None:
        initial_download = max(700, int(max_bars_buffer) + 200)
        if len(work) > initial_download:
            work = work.tail(initial_download).reset_index(drop=True)

    bars: list[dict[str, Any]] = []
    for _, r in work.iterrows():
        bars.append(
            {
                "time": r["time"],
                "open": float(r.get("open", np.nan)),
                "high": float(r.get("high", np.nan)),
                "low": float(r.get("low", np.nan)),
                "close": float(r.get("close", np.nan)),
                "volume": float(r.get("volume", 0) or 0),
            }
        )

    live_df = pd.DataFrame(
        {
            "timestamp": work["time"].to_numpy(),
            "open": work["open"].astype(float).to_numpy(),
            "high": work["high"].astype(float).to_numpy(),
            "low": work["low"].astype(float).to_numpy(),
            "close": work["close"].astype(float).to_numpy(),
            "volume": work["volume"].astype(float).to_numpy(),
        }
    )

    label_maps: list[dict[int, str] | None] = [None] * len(models)
    feat_all = _compute_snapshot_features(live_df)
    raw_df = live_df.rename(columns={"timestamp": "date"})[["date", "open", "high", "low", "close", "volume"]].copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"], utc=True, errors="coerce")
    raw_df = raw_df.dropna(subset=["date"]).sort_values("date")

    if feat_all is not None and not feat_all.empty:
        for idx, m in enumerate(models):
            try:
                mdl = m.get("predictor")
                exp = m.get("exp_feats")
                if mdl is None:
                    continue
                cls_vals = [int(c) for c in list(getattr(mdl, "classes_", []))]
            except Exception:
                cls_vals = []
            try:
                if len(set(cls_vals)) > 2:
                    continue
                X_use = feat_all
                if exp:
                    X_use = _prepare_X_for_model_static(feat_all, exp)
                auto_map = _auto_detect_label_polarity(mdl, X_use, raw_df)
                if auto_map and set(auto_map.values()) == {"LONG", "SHORT"}:
                    label_maps[idx] = dict(auto_map)
            except Exception:
                continue

    return LiveBootstrapPayload(
        bars=bars,
        live_df=live_df,
        label_maps=label_maps,
        snapshot_bars=int(len(live_df)),
    )


def _task_build_live_bootstrap(
    *,
    models: list[dict[str, Any]],
    symbol: str,
    exchange: str,
    bar_size: str,
    max_bars_buffer: int,
    progress_cb=None,
) -> LiveBootstrapPayload:
    if callable(progress_cb):
        progress_cb("Live bootstrap: nacitam snapshot...")
    tv = TradingViewClient(username=os.getenv("TV_USERNAME"), password=os.getenv("TV_PASSWORD"))
    tf_label = (bar_size or "1 hour").replace("mins", "min")
    initial_download = max(700, int(max_bars_buffer) + 200)
    df = tv.get_history(symbol, exchange, tf_label, limit=initial_download)
    if callable(progress_cb) and df is not None and not df.empty:
        progress_cb("Live bootstrap: autodetekce polarity modelu...")
    return _build_live_bootstrap_payload_from_history_df(
        df,
        models,
        max_bars_buffer=int(max_bars_buffer),
    )


def _task_preload_degradation(
    *,
    models: list[dict[str, Any]],
    symbol: str,
    exchange: str,
    timeframe: str,
    degradation_window_size: int,
    progress_cb=None,
) -> DegradationPreloadPayload:
    if not models:
        return DegradationPreloadPayload(predictions=[], prices=[], timestamps=[])

    if callable(progress_cb):
        progress_cb("Degradation: nacitam historicka data...")
    bars_needed = int(degradation_window_size) + 200
    tv = TradingViewClient(username=os.getenv("TV_USERNAME"), password=os.getenv("TV_PASSWORD"))
    df = tv.get_history(symbol, exchange, timeframe, limit=bars_needed)
    if df is None or df.empty:
        return DegradationPreloadPayload(predictions=[], prices=[], timestamps=[])

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    df.index.name = "timestamp"

    df_feats = compute_all_features(df)
    if df_feats.empty:
        return DegradationPreloadPayload(predictions=[], prices=[], timestamps=[])

    df_recent = df_feats.tail(int(degradation_window_size)).copy().reset_index(drop=True)
    model_info = models[0]
    model = model_info["predictor"]
    exp_feats = model_info.get("exp_feats")
    X_prepared = df_recent
    if exp_feats:
        X_prepared = _prepare_X_for_model_static(df_recent, exp_feats)

    try:
        label_map = model_info.get("label_map") or _infer_label_map_from_classes(getattr(model, "classes_", None))
        X_pred = _align_X_for_model(model, X_prepared)
        proba_all = _predict_proba_safely(model, X_pred)
        t_short = float(model_info.get("t_short", 0.5))
        t_long = float(model_info.get("t_long", 0.5))
        classes = getattr(model, "classes_", None)
        if classes is not None:
            if any(isinstance(c, str) for c in classes):
                lut = {str(c).upper(): i for i, c in enumerate(classes)}
                idx_long = lut.get("LONG")
                idx_short = lut.get("SHORT")
            else:
                idx_long = next((i for i, c in enumerate(classes) if str(label_map.get(int(c), "")).upper() == "LONG"), None)
                idx_short = next((i for i, c in enumerate(classes) if str(label_map.get(int(c), "")).upper() == "SHORT"), None)
        else:
            idx_long = 1
            idx_short = 0

        predictions: list[int] = []
        for proba_row in proba_all:
            pL = float(proba_row[idx_long]) if idx_long is not None else 0.5
            pS = float(proba_row[idx_short]) if idx_short is not None else 0.5
            if pL >= t_long and pL >= pS:
                predictions.append(1)
            elif pS >= t_short and pS > pL:
                predictions.append(-1)
            else:
                predictions.append(0)
    except Exception:
        predictions = [0] * len(df_recent)

    prices = df_recent["close"].astype(float).tolist()
    timestamps = df_recent["time"].tolist()
    return DegradationPreloadPayload(predictions=predictions, prices=prices, timestamps=timestamps)

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
            wait_status_step = 6  # emit wait status every N stale polls
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
                        self.statusChanged.emit("Connected")
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

                if stale_count > 0 and (stale_count % wait_status_step) == 0:
                    wait_min = (stale_count * max(1, self._poll_interval_s())) / 60.0
                    self.statusChanged.emit(f"Waiting for closed bar ({wait_min:.1f}m)")

                if stale_count >= 30:
                    try:
                        self.statusChanged.emit("Reconnecting...")
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
        use_and = bool(getattr(self.w.config, "use_and_ensemble", False))

        classes = ["LONG", "SHORT"]

        # L0: MA
        l0 = self._ma_sig_from_features(features) or "FLAT"

        # MA-only režim -> vrať rovnou MA
        if bool(getattr(self.w.config, "use_ma_only", False)):
            probs = [1.0, 0.0] if l0 == "LONG" else [0.0, 1.0] if l0 == "SHORT" else [0.5, 0.5]
            return l0, probs, classes

        # Když není žádný model -> vrať MA
        if not self.w.models:
            probs = [1.0, 0.0] if l0 == "LONG" else [0.0, 1.0] if l0 == "SHORT" else [0.5, 0.5]
            return l0, probs, classes

        # L1: AND nebo VOTE podle nastavení (model-only = VOTE)
        if use_and:
            label, conf_min, dirs, confs = self.w._predict_one_label_AND(features, thr=0.0)
        else:
            label, conf_min, dirs, confs = self.w._predict_one_label_VOTE(features)
        l1 = "LONG" if label == +1 else "SHORT" if label == -1 else "FLAT"

        # L2: (volitelně) MA ∧ L1 + aplikace prahu z UI (thr_ui) – stejná politika jako v _rescore_all
        proposal = build_live_proposal(l0, l1, use_and)

        # Hystereze (vstup/výstup) – stejně jako v _rescore_all
        final = None
        try:
            # k featurám se v adaptér módu nedostaneme k celé historii – použij poslední bar z features
            close = float(features.get("close").iloc[-1])
            atr   = float(features.get("atr").iloc[-1]) if "atr" in features.columns else np.nan
        except Exception:
            close, atr = np.nan, np.nan

        final = apply_live_hysteresis(
            proposal,
            conf_min,
            self.w._live_pos,
            thr_ui,
            max(0.0, thr_ui - 0.05),
            block_entry=bool(self.w._live_pos == 0 and self.w._near_round_level(close, atr)),
        )

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
    log_message = Signal(str)

    def __init__(self, parent: QWidget | None = None, config: LiveConfig | None = None) -> None:
        super().__init__(parent)
        self.logger = get_logger("live_bot.gui")
        self.config = config or LiveConfig()
        self._ui_settings = QSettings("ibkr_trading_bot", "live_tab")
        self._load_ui_settings()
        self.model = None                          # používá se jen pro feature_names_in_
        self.models: list[dict[str, Any]] = []     # členové ensemble
        self.class_to_dir = {1: "LONG", 0: "SHORT"}  # lze přepsat z meta
        self._diag_once = {}
        self.label_map_from_meta = False
        self.model_expected_features: list[str] | None = None
        self.worker: TVWorker | None = None
        self._bootstrap_worker: TaskWorker | None = None
        self._degradation_worker: TaskWorker | None = None
        self._retired_threads: list[QThread] = []
        self.warm: LiveWarmupService | None = None

        self.live_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        self._bars: list[dict[str, Any]] = []
        self._bar_index: dict[int, int] = {}
        self._last_arrival_utc: pd.Timestamp | None = None
        self._pending_bar_payloads: list[dict[str, Any]] = []
        self._bar_refresh_scheduled = False
        self._bootstrap_request_id = 0
        self._degradation_request_id = 0
        self._log_queue = deque()

        self._build_ui()
        self._load_tv_credentials_into_ui()
        self._wire_basic_logic()
        self._last_alert_sig: str | None = None
        self._last_alert_bar_key: int | None = None
        self._se = None

        self._live_pos = 0       # -1 short, 0 flat, +1 long
        self._live_entry_px = None
        self._trade_executor = TradeExecutor()
        self._trading_enabled = False
        self._curr_entry_thr = self.config.entry_thr
        self._curr_exit_thr = self.config.exit_thr
        self._curr_t_short = None
        self._curr_t_long = None
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
        self._last_tv_status: str | None = None
        self.log_message.connect(self._enqueue_log_text)
        self._log_timer = QTimer(self)
        self._log_timer.setInterval(100)
        self._log_timer.timeout.connect(self._flush_log_queue)
        self._log_timer.start()
        self._set_session_running_ui(False)

    def _load_ui_settings(self) -> None:
        try:
            raw = self._ui_settings.value("display_bars", self.config.display_bars)
            bars = int(raw)
            self.config.display_bars = max(30, min(2000, bars))
        except Exception:
            self.config.display_bars = max(30, min(2000, int(self.config.display_bars)))

    def _save_ui_settings(self) -> None:
        try:
            self._ui_settings.setValue("display_bars", int(self.config.display_bars))
            self._ui_settings.sync()
        except Exception:
            pass

    def _load_tv_credentials_into_ui(self) -> None:
        try:
            username, password = load_saved_tv_credentials()
            self.ed_tv_username.setText(str(username or ""))
            self.ed_tv_password.setText(str(password or ""))
            self.lbl_tv_auth.setText("TV login: ulozen" if username and password else "TV login: neni ulozen")
        except Exception:
            self.lbl_tv_auth.setText("TV login: chyba nacteni")

    def _save_tv_credentials_from_ui(self) -> None:
        username = (self.ed_tv_username.text() or "").strip()
        password = self.ed_tv_password.text() or ""
        try:
            save_tv_credentials(username, password)
            if username and password:
                self.lbl_tv_auth.setText("TV login: ulozen")
                self._append_log("[TV] Login ulozen lokalne. Pokud uz bezi session, pouzij Start (RESET) a znovu Start (SET).")
            else:
                self.lbl_tv_auth.setText("TV login: neni ulozen")
                self._append_log("[TV] Ulozeny login byl vymazan.")
        except Exception as exc:
            self._append_log(f"[TV][ERROR] Nelze ulozit login: {exc}")

    def _apply_tf_presets(self):
        tf = self.cmb_interval.currentText()
        p = PRESETS_BY_TF.get(tf, PRESETS_BY_TF["1 hour"])

        # UI práh (hystereze) - z user_settings (načteno z Tab 3)
        s_entry = self.user_settings.get("entry_threshold", self.config.sensitivity)
        s_exit = self.user_settings.get("exit_threshold", self.config.exit_thr)
        self._curr_entry_thr = float(s_entry) if isinstance(s_entry, (int, float)) else self.config.sensitivity
        if isinstance(s_exit, (int, float)):
            self._curr_exit_thr = float(s_exit)
        else:
            self._curr_exit_thr = max(0.0, min(self._curr_entry_thr - 0.05, self._curr_entry_thr))

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
        self.ed_symbol = QLineEdit(self.config.symbol)
        self.ed_expiry = QLineEdit(self.config.exchange)
        self.cmb_interval = QComboBox(); self.cmb_interval.addItems(["5 min", "15 min", "30 min", "1 hour"])
        self.cmb_interval.setCurrentText(self.config.bar_size)
        self.spn_display_bars = QSpinBox()
        self.spn_display_bars.setRange(30, 2000)
        self.spn_display_bars.setSingleStep(12)
        self.spn_display_bars.setValue(max(30, int(getattr(self.config, "display_bars", 144))))
        self.spn_display_bars.setToolTip("Kolik poslednich uzavrenych svicek zobrazit v grafech.")
        self.btn_start = QPushButton("Start (SET)")
        self.btn_reset = QPushButton("Reset")
        self.btn_trade = QPushButton("Obchodovat: OFF")
        self.btn_trade.setCheckable(True)
        h.addWidget(QLabel("Symbol:"));     h.addWidget(self.ed_symbol)
        h.addWidget(QLabel("Exchange:"));   h.addWidget(self.ed_expiry)
        h.addWidget(QLabel("Timeframe:"));  h.addWidget(self.cmb_interval)
        h.addWidget(QLabel("Svíček:"));     h.addWidget(self.spn_display_bars)
        h.addWidget(self.btn_start)
        h.addWidget(self.btn_reset)
        h.addWidget(self.btn_trade)
        session_box.setLayout(h)

        tv_auth_box = QGroupBox("TradingView login")
        tv_auth_layout = QHBoxLayout()
        self.ed_tv_username = QLineEdit()
        self.ed_tv_username.setPlaceholderText("TV username")
        self.ed_tv_username.setFixedWidth(180)
        self.ed_tv_password = QLineEdit()
        self.ed_tv_password.setPlaceholderText("TV password")
        self.ed_tv_password.setEchoMode(QLineEdit.Password)
        self.ed_tv_password.setFixedWidth(180)
        self.btn_tv_save = QPushButton("Ulozit TV login")
        self.lbl_tv_auth = QLabel("TV login: neni ulozen")
        tv_auth_layout.addWidget(QLabel("Uzivatel:"))
        tv_auth_layout.addWidget(self.ed_tv_username)
        tv_auth_layout.addWidget(QLabel("Heslo:"))
        tv_auth_layout.addWidget(self.ed_tv_password)
        tv_auth_layout.addWidget(self.btn_tv_save)
        tv_auth_layout.addWidget(self.lbl_tv_auth)
        tv_auth_layout.addStretch(1)
        tv_auth_box.setLayout(tv_auth_layout)

        # Model
        model_box = QGroupBox("Model")
        g = QGridLayout()
        self.le_model_path = QLineEdit(DEFAULT_MODEL_DIR)
        self.btn_model = QPushButton("…")

        # Cesta k modelu
        g.addWidget(QLabel("Cesta:"), 0, 0); g.addWidget(self.le_model_path, 0, 1); g.addWidget(self.btn_model, 0, 2)

        self.lbl_decision_threshold = QLabel("T-short/T-long: – / –")
        self.lbl_entry_threshold = QLabel("Entry Threshold: –")
        self.lbl_exit_threshold = QLabel("Exit Threshold: –")
        self.lbl_ma_only = QLabel("MA-Only: –")
        self.lbl_and_ensemble = QLabel("AND Ensemble: –")
        self.lbl_ma_only.setVisible(False)
        self.lbl_and_ensemble.setVisible(False)
        model_box.setLayout(g)

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
        left.addWidget(tv_auth_box)
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
        self.btn_start.clicked.connect(self._on_toggle_start)
        self.btn_reset.clicked.connect(self._on_reset_tab)
        self.btn_trade.clicked.connect(self._on_toggle_trading)
        self.btn_tv_save.clicked.connect(self._save_tv_credentials_from_ui)
        self.spn_display_bars.valueChanged.connect(self._on_display_bars_changed)

        self.fresh_timer = QTimer(self); self.fresh_timer.setInterval(1000)
        self.fresh_timer.timeout.connect(self._update_clock)
        self.fresh_timer.start()

    def _set_session_running_ui(self, running: bool) -> None:
        self.btn_start.setText("Start (RESET)" if running else "Start (SET)")

    @Slot()
    def _on_toggle_start(self) -> None:
        if self.worker is not None or self._bootstrap_worker is not None:
            self._on_stop()
            return
        self._on_start()

    @Slot()
    def _on_reset_tab(self) -> None:
        self._append_log("[INFO] Resetuji Tab 5 do vychoziho stavu.")
        self._stop_worker(wait_ms=1500, allow_background=True)
        self._reset_runtime_state(full_reset=True)
        self._set_session_running_ui(False)

    @Slot(bool)
    def _on_toggle_trading(self, enabled: bool) -> None:
        self._trading_enabled = bool(enabled)
        self.btn_trade.setText("Obchodovat: ON" if self._trading_enabled else "Obchodovat: OFF")
        if not self._trading_enabled:
            self._trade_executor = TradeExecutor()
            self._sync_live_state_from_executor()
        self._append_log(
            "[TRADE] Obchodni exekuce signalu je zapnuta." if self._trading_enabled
            else "[TRADE] Obchodni exekuce signalu je vypnuta."
        )

    def _reset_runtime_state(self, *, full_reset: bool = False) -> None:
        self.warm = None
        self._pending_bar_payloads.clear()
        self._bar_refresh_scheduled = False
        self._bars.clear()
        self._bar_index.clear()
        self.live_df = self.live_df.iloc[0:0].copy()
        self._bootstrap_request_id += 1
        self._degradation_request_id += 1
        self._trade_executor = TradeExecutor()
        self._sync_live_state_from_executor()
        self._trades.clear()
        self.tbl_trades.setRowCount(0)
        self._prediction_buffer = []
        self._price_buffer = []
        self._y_true_buffer = []
        self._tracked_timestamps = set()
        self._last_degradation_check = 0
        self.live_metrics_recent = {}
        self._last_arrival_utc = None
        self._last_alert_sig = None
        self._last_alert_bar_key = None
        self._last_alert_bar_ns = None
        self._last_signal = None
        self.lbl_mode.setText("Mode: WARM-UP")
        self.lbl_ib_status.setText("TV: Disconnected")
        self.lbl_fresh.setText("Freshness: --")
        self.degradation_console.setPlainText("(Žádný model načten)")
        if full_reset:
            self.le_model_path.setText(DEFAULT_MODEL_DIR)
            self.models = []
            self.model = None
            self.model_expected_features = None
            self.reference_metrics = {}
            self.user_settings = {}
            self._curr_t_short = None
            self._curr_t_long = None
            self.btn_trade.blockSignals(True)
            self.btn_trade.setChecked(False)
            self.btn_trade.blockSignals(False)
            self._trading_enabled = False
            self.btn_trade.setText("Obchodovat: OFF")
        self._render_charts()

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

        def _safe_float(v):
            try:
                out = float(v)
                if np.isfinite(out):
                    return out
            except Exception:
                pass
            return None

        def _meta_ternary_thresholds(meta_dict: dict) -> tuple[float | None, float | None]:
            if not isinstance(meta_dict, dict):
                return None, None
            t_short = _safe_float(meta_dict.get("ternary_threshold_short"))
            t_long = _safe_float(meta_dict.get("ternary_threshold_long"))
            user = meta_dict.get("user_settings")
            if isinstance(user, dict):
                if t_short is None:
                    t_short = _safe_float(user.get("ternary_threshold_short_eval"))
                if t_long is None:
                    t_long = _safe_float(user.get("ternary_threshold_long_eval"))
            return t_short, t_long

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

                version_warning = build_sklearn_version_warning(meta, model_path=p)
                if version_warning:
                    self._append_log(f"[WARN] {version_warning}")

                # Tab 5 hard gate: pouze ternární modely s modelovými T-short/T-long
                if not hasattr(pred, "predict_proba"):
                    self._append_log(f"[ERROR] Model nepodporuje predict_proba: {os.path.basename(p)}")
                    return False
                cls_raw = getattr(pred, "classes_", None)
                classes_dbg = list(cls_raw) if cls_raw is not None else []
                if len(classes_dbg) != 3:
                    self._append_log(
                        f"[ERROR] Tab 5 vyžaduje ternární model (3 třídy). "
                        f"Model {os.path.basename(p)} má classes={classes_dbg}"
                    )
                    return False

                t_short, t_long = _meta_ternary_thresholds(meta)
                if not isinstance(t_short, (int, float)) or not isinstance(t_long, (int, float)):
                    self._append_log(
                        f"[ERROR] Chybí ternární prahy v metadatech: {os.path.basename(p)} "
                        "(ternary_threshold_short/long)."
                    )
                    return False
                self._append_log(
                    f"[META] Ternary thresholds model={os.path.basename(p)} "
                    f"T-short={float(t_short):.3f} T-long={float(t_long):.3f}"
                )

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
                    "t_short": float(t_short),
                    "t_long": float(t_long),
                    "metadata": meta,  # uložit metadata pro později
                })

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

                # Diagnostika směrového biasu už na holdoutu (důležité pro očekávání v LIVE)
                try:
                    hold = (meta.get("metrics_holdout") or {}) if isinstance(meta, dict) else {}
                    n_long_h = hold.get("num_trades_long")
                    n_short_h = hold.get("num_trades_short")
                    if isinstance(n_long_h, (int, float)) and isinstance(n_short_h, (int, float)):
                        self._append_log(f"[TAB4-DIAG] holdout_trades LONG={int(n_long_h)} SHORT={int(n_short_h)}")
                        if int(n_long_h) == 0 and int(n_short_h) > 0:
                            self._append_log("[TAB4-DIAG] ⚠️ Holdout bias: model negeneruje LONG obchody (jen SHORT).")
                        elif int(n_short_h) == 0 and int(n_long_h) > 0:
                            self._append_log("[TAB4-DIAG] ⚠️ Holdout bias: model negeneruje SHORT obchody (jen LONG).")
                except Exception:
                    pass
                
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

    @staticmethod
    def _extract_ternary_thresholds_from_metadata(metadata: dict) -> tuple[float | None, float | None]:
        def _sf(v):
            try:
                out = float(v)
                if np.isfinite(out):
                    return out
            except Exception:
                pass
            return None

        if not isinstance(metadata, dict):
            return None, None
        t_short = _sf(metadata.get("ternary_threshold_short"))
        t_long = _sf(metadata.get("ternary_threshold_long"))
        user = metadata.get("user_settings")
        if isinstance(user, dict):
            if t_short is None:
                t_short = _sf(user.get("ternary_threshold_short_eval"))
            if t_long is None:
                t_long = _sf(user.get("ternary_threshold_long_eval"))
        return t_short, t_long

    def _load_user_settings_from_first_model(self) -> None:
        """Načte user_settings z metadat prvního modelu a zobrazí je jako read-only info panel."""
        if not self.models or not self.models[0]:
            self._append_log("[SETTINGS] ❌ Žádné modely načteny")
            self._update_settings_display({}, {})
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
                self._update_settings_display({}, metadata)
                return
            
            # Zobraz nastavení v info panelu
            self._update_settings_display(user_settings, metadata)
            self._append_log("[SETTINGS] ✅ Nastavení modelu načtena z Tab 3")
            
        except Exception as e:
            self._append_log(f"[WARN] Nelze načít user_settings: {e}")
            self._update_settings_display({}, {})
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
    
    def _update_settings_display(self, user_settings: dict, metadata: dict | None = None) -> None:
        """Aktualizuje display panelu s nastavením modelu (read-only) a uloží do self.user_settings."""
        self.user_settings = user_settings  # uložit pro použití v predikci

        t_short, t_long = self._extract_ternary_thresholds_from_metadata(metadata or {})
        entry_threshold = user_settings.get("entry_threshold", "–")
        exit_threshold = user_settings.get("exit_threshold", "–")
        use_ma_only = _coerce_bool(user_settings.get("use_ma_only"), default=self.config.use_ma_only)
        use_and_ensemble = _coerce_bool(
            user_settings.get("use_and_ensemble"),
            default=self.config.use_and_ensemble,
        )

        t_short_disp = f"{float(t_short):.3f}" if isinstance(t_short, (int, float)) else "–"
        t_long_disp = f"{float(t_long):.3f}" if isinstance(t_long, (int, float)) else "–"
        self.lbl_decision_threshold.setText(f"T-short/T-long: {t_short_disp} / {t_long_disp}")
        self.lbl_entry_threshold.setText(f"Entry Threshold: {entry_threshold}")
        self.lbl_exit_threshold.setText(f"Exit Threshold: {exit_threshold}")
        self.lbl_ma_only.setText(f"MA-Only: {'✓ zapnuto' if use_ma_only else '✗ vypnuto'}")
        self.lbl_and_ensemble.setText(f"AND Ensemble: {'✓ AND' if use_and_ensemble else '✗ VOTE'}")

        # Synchronizuj runtime chování Tab 5 s nastavením načteným z Tab 3
        self.config.use_ma_only = use_ma_only
        self.config.use_and_ensemble = use_and_ensemble
        
        # Aplikuj entry/exit thresholdy na aktivní konfiguraci
        if isinstance(entry_threshold, (int, float)):
            self._curr_entry_thr = float(entry_threshold)
        if isinstance(exit_threshold, (int, float)):
            self._curr_exit_thr = float(exit_threshold)
        if isinstance(t_short, (int, float)):
            self._curr_t_short = float(t_short)
        if isinstance(t_long, (int, float)):
            self._curr_t_long = float(t_long)
        
        # Log po aktualizaci
        if user_settings or isinstance(t_short, (int, float)) or isinstance(t_long, (int, float)):
            self._append_log(
                f"[SETTINGS] T-short={t_short_disp}, T-long={t_long_disp}, "
                f"Entry={entry_threshold}, Exit={exit_threshold}, MA-Only={use_ma_only}, AND={use_and_ensemble}"
            )

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
            t_short = float(m.get("t_short", self._curr_t_short if isinstance(self._curr_t_short, (int, float)) else 0.5))
            t_long = float(m.get("t_long", self._curr_t_long if isinstance(self._curr_t_long, (int, float)) else 0.5))
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
            direction, conf = _pick_direction_from_raw_proba(
                classes_i, raw_proba, label_map,
                short_threshold=t_short,
                long_threshold=t_long,
            )

            if direction is None:
                direction = "FLAT"
                conf = 0.0

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
            t_short = float(m.get("t_short", self._curr_t_short if isinstance(self._curr_t_short, (int, float)) else 0.5))
            t_long = float(m.get("t_long", self._curr_t_long if isinstance(self._curr_t_long, (int, float)) else 0.5))
            if exp:
                X_use = self._prepare_X_for_model(Xrow, exp)
            try:
                label_map = m.get("label_map") or self.class_to_dir
                pL, pS, classes_i, raw_proba = _extract_long_short_proba(mdl, X_use, label_map=label_map)
            except Exception:
                pL, pS, classes_i, raw_proba = 0.5, 0.5, None, None

            direction, conf = _pick_direction_from_raw_proba(
                classes_i, raw_proba, label_map,
                short_threshold=t_short,
                long_threshold=t_long,
            )

            if direction is None:
                direction = "FLAT"
                conf = 0.0

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
        
        X_use = X_use[exp].copy()
        for c in X_use.columns:
            key = str(c).strip().lower()
            if key in {"date", "time", "timestamp", "datetime"}:
                ts = pd.to_datetime(X_use[c], utc=True, errors="coerce")
                X_use[c] = pd.Series(
                    np.where(ts.notna(), ts.astype("int64"), np.nan),
                    index=X_use.index,
                    dtype="float64",
                )
        return X_use.astype(float)

    @Slot()
    def _on_choose_model(self) -> None:
        default_dir = DEFAULT_MODEL_DIR if os.path.isdir(DEFAULT_MODEL_DIR) else os.getcwd()
        fnames, _ = QFileDialog.getOpenFileNames(self, "Vybrat modely", default_dir, "Pickle files (*.pkl);;All files (*)")
        if fnames:
            self.set_model_paths(fnames)

    # ---------- Ovládání ----------
    @Slot()
    def _on_start(self):
        if not self._load_models():
            return
        self._set_session_running_ui(True)
        self._apply_tf_presets()
        self._start_worker()
        self._append_log("[INFO] Start sezení…")
        self._append_log(f"[MODE] MA-only={self.config.use_ma_only} | AND={self.config.use_and_ensemble}")
        self._append_log(
            f"[INFO] Cekam na uzavreni nove svicky ({self.config.bar_size}). "
            "Mimo obchodni hodiny muze byt delsi pauza bez noveho baru."
        )

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
            self._seed_snapshot_from_warmup_history(adapter)
            self.lbl_mode.setText("Mode: LIVE" if self.warm.state == "LIVE" else "Mode: WARM-UP")
        except Exception as e:
            self._append_log(f"[WARN] Warm-up inicializace selhala: {e}")

    @Slot()
    def _on_stop(self) -> None:
        self._append_log("[INFO] Stop sezení.")
        self._stop_worker()
        self._set_session_running_ui(False)

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

    @Slot(int)
    def _on_display_bars_changed(self, val: int) -> None:
        bars = max(30, int(val))
        self.config.display_bars = bars
        self._save_ui_settings()
        self._append_log(f"[INFO] Zobrazeni svicek nastaveno na {bars}.")
        self._render_charts()

    # ---------- Worker reakce ----------
    @Slot(str)
    def _on_ib_status(self, status: str) -> None:
        status_text = str(status or "").strip() or "Unknown"
        self.lbl_ib_status.setText(f"TV: {status_text}")
        if status_text != self._last_tv_status:
            self._append_log(f"[TV] {status_text}")
            self._last_tv_status = status_text

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._append_log("[ERROR] " + message)

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
            key = str(c).strip().lower()
            if key in {"date", "time", "timestamp", "datetime"}:
                ts = pd.to_datetime(df[c], utc=True, errors="coerce")
                df[c] = pd.Series(
                    np.where(ts.notna(), ts.astype("int64"), np.nan),
                    index=df.index,
                    dtype="float64",
                )
            elif not pd.api.types.is_bool_dtype(df[c]):
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
            key = str(c).strip().lower()
            if key in {"date", "time", "timestamp", "datetime"}:
                ts = pd.to_datetime(df[c], utc=True, errors="coerce")
                df[c] = pd.Series(
                    np.where(ts.notna(), ts.astype("int64"), np.nan),
                    index=df.index,
                    dtype="float64",
                )
            elif not pd.api.types.is_bool_dtype(df[c]):
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
            proposal = build_live_proposal(d0, l1, use_ma_and)

            # --- Hysterese: tady teprve aplikuj prahy ---
            final = None
            curr_close = float(raw.loc[ts, "close"])
            curr_atr   = float(raw.loc[ts, "atr"])

            final = apply_live_hysteresis(
                proposal,
                conf_min,
                self._live_pos,
                self._curr_entry_thr,
                self._curr_exit_thr,
            )

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
            if self._trading_enabled:
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
        result = self._trade_executor.step(final, float(raw.loc[ts, "close"]), ts)
        if result.closed_trade is not None:
            self._store_closed_trade(result.closed_trade)
        self._sync_live_state_from_executor()

    def _sync_live_state_from_executor(self) -> None:
        state = self._trade_executor.state
        self._live_pos = int(state.position)
        self._live_entry_px = float(state.entry_price) if state.entry_price is not None else None
        if self._live_pos == 0 or state.entry_price is None:
            self._open_trade = None
            return

        direction = "LONG" if self._live_pos > 0 else "SHORT"
        entry_time = pd.to_datetime(state.entry_time, utc=True, errors="coerce")
        entry_time_text = str(entry_time)[:19] if pd.notna(entry_time) else str(state.entry_time or "")[:19]
        self._open_trade = {
            "direction": direction,
            "entry_time": entry_time_text,
            "entry_price": float(state.entry_price),
        }

    def _store_closed_trade(self, trade: ClosedTrade) -> None:
        entry_time = pd.to_datetime(trade.entry_time, utc=True, errors="coerce")
        exit_time = pd.to_datetime(trade.exit_time, utc=True, errors="coerce")
        entry_time_text = str(entry_time)[:19] if pd.notna(entry_time) else str(trade.entry_time or "")[:19]
        exit_time_text = str(exit_time)[:19] if pd.notna(exit_time) else str(trade.exit_time or "")[:19]
        direction = "LONG" if int(trade.side) > 0 else "SHORT"
        pnl = float(trade.pnl)
        self._add_trade_to_table(
            entry_time_text,
            direction,
            float(trade.entry_price),
            exit_time_text,
            float(trade.exit_price),
            pnl,
        )
        self._trades.append({
            "entry_time": entry_time_text,
            "direction": direction,
            "entry_price": float(trade.entry_price),
            "exit_time": exit_time_text,
            "exit_price": float(trade.exit_price),
            "pnl": pnl,
        })

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

        # Zvukové alerty chceme vždy v LIVE módu Tab 5 (nikoli ve WARM-UP fázi)
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
        wait_suffix = ""
        if self.worker is not None:
            try:
                poll_s = max(1, int(self.worker._poll_interval_s()))
            except Exception:
                poll_s = 30
            if age_s > max(poll_s * 2, 30):
                wait_suffix = " (cekam na novy uzavreny bar)"
        self.lbl_fresh.setText(f"Freshness: {mins}m {secs}s{wait_suffix}")
        self.lbl_fresh.setStyleSheet("color: #119911;" if age_s <= ok_threshold else "color: #cc0000;")

    def _render_charts(self) -> None:
        if not self._bars:
            self.ax_price.cla(); self.ax_macd.cla(); self.canvas.draw_idle(); return
        display_n = max(30, int(getattr(self.config, "display_bars", 144)))
        df = pd.DataFrame(self._bars).reset_index(drop=True)
        if len(df) > display_n:
            df = df.tail(display_n).reset_index(drop=True)

        # MACD 12-26-9
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig  = macd.ewm(span=9, adjust=False).mean()
        hist = macd - sig

        ax1, ax2 = self.ax_price, self.ax_macd
        ax1.cla(); ax2.cla()
        x = np.arange(len(df))
        bullish_color = '#1f9d55'
        bearish_color = '#d64545'
        candle_ranges = (df['high'] - df['low']).astype(float)
        median_range = float(np.nanmedian(candle_ranges.to_numpy(dtype=float))) if len(candle_ranges) else 0.0
        if not np.isfinite(median_range) or median_range <= 0:
            median_range = float(np.nanmax(candle_ranges.to_numpy(dtype=float))) if len(candle_ranges) else 0.0
        if not np.isfinite(median_range) or median_range <= 0:
            median_range = 1.0
        min_body_height = median_range * 0.06
        body_width = 0.56

        # svíčky
        for i, row in df.iterrows():
            o, h, l, c = map(float, (row['open'], row['high'], row['low'], row['close']))
            color = bullish_color if c >= o else bearish_color
            ax1.vlines(i, l, h, linewidth=1.15, color=color, zorder=2)
            body_bottom = min(o, c)
            body_height = abs(c - o)
            if body_height < min_body_height:
                body_height = min_body_height
                body_bottom = ((o + c) / 2.0) - (body_height / 2.0)
            ax1.add_patch(
                Rectangle(
                    (i - body_width / 2.0, body_bottom),
                    body_width,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=3,
                )
            )

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
                ax1.scatter(long_x, long_y, marker='^', s=90, color=bullish_color, zorder=5)
            if short_x:
                ax1.scatter(short_x, short_y, marker='v', s=90, color=bearish_color, zorder=5)

        axis_pad = max(min_body_height, median_range * 0.18)
        y_min = float(df['low'].min()) - axis_pad
        y_max = float(df['high'].max()) + axis_pad
        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            ax1.set_ylim(y_min, y_max)
        ax1.set_xlim(-0.75, len(df) - 0.25)

        ax1.set_ylabel('Price')
        ax2.plot(x, macd.values, label='MACD')
        ax2.plot(x, sig.values,  label='Signal')
        ax2.bar(x, hist.values, width=0.8, alpha=0.3)
        ax2.set_ylabel('MACD')
        ax2.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.2); ax2.grid(True, alpha=0.2)
        self.canvas.draw_idle()


# Kompatibilitní aliasy
    def _enqueue_log_text(self, text: str) -> None:
        self._log_queue.append(str(text))

    def _flush_log_queue(self) -> None:
        if not hasattr(self, "console"):
            return
        while self._log_queue:
            self.console.appendPlainText(self._log_queue.popleft())
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.console.setTextCursor(cursor)

    def _append_log(self, text: str) -> None:
        try:
            self.log_message.emit(str(text))
        except Exception:
            pass

    def _track_retired_thread(self, thread: QThread) -> None:
        if thread in self._retired_threads:
            return
        self._retired_threads.append(thread)

        def _cleanup_retired_thread() -> None:
            try:
                self._retired_threads.remove(thread)
            except ValueError:
                pass
            try:
                thread.deleteLater()
            except Exception:
                pass

        thread.finished.connect(_cleanup_retired_thread)

    def _stop_thread_attr(self, attr_name: str, *, wait_ms: int = 1000, allow_background: bool = True) -> bool:
        thread = getattr(self, attr_name, None)
        if thread is None:
            return True
        setattr(self, attr_name, None)

        for signal_name in ("statusChanged", "error", "barClosed", "progress_text", "result", "finished"):
            try:
                getattr(thread, signal_name).disconnect()
            except Exception:
                pass

        try:
            if hasattr(thread, "stop"):
                thread.stop()
        except Exception:
            pass

        is_running = False
        try:
            is_running = bool(thread.isRunning())
        except Exception:
            is_running = False

        if is_running and not thread.wait(wait_ms):
            if allow_background:
                self._track_retired_thread(thread)
                return False
            setattr(self, attr_name, thread)
            return False

        try:
            thread.deleteLater()
        except Exception:
            pass
        return True

    def _stop_background_workers(self, *, wait_ms: int = 1000, allow_background: bool = True) -> bool:
        ok = True
        for attr_name in ("_bootstrap_worker", "_degradation_worker"):
            ok = self._stop_thread_attr(attr_name, wait_ms=wait_ms, allow_background=allow_background) and ok
        return ok

    def _stop_worker(self, *, wait_ms: int = 1000, allow_background: bool = True) -> bool:
        ok = self._stop_background_workers(wait_ms=wait_ms, allow_background=allow_background)
        self._pending_bar_payloads.clear()
        self._bar_refresh_scheduled = False
        ok = self._stop_thread_attr("worker", wait_ms=wait_ms, allow_background=allow_background) and ok
        return ok

    def _launch_stream_worker(self) -> None:
        if self.worker is not None:
            self._stop_thread_attr("worker", wait_ms=500, allow_background=True)
        self.worker = TVWorker(self.config, parent=self)
        self.worker.statusChanged.connect(self._on_ib_status)
        self.worker.error.connect(self._on_error)
        self.worker.barClosed.connect(self._on_bar_closed)
        self.worker.start()

    def _start_worker(self) -> None:
        if self.worker is not None:
            self._stop_worker(wait_ms=500, allow_background=True)
        self._stop_background_workers(wait_ms=500, allow_background=True)
        self._pending_bar_payloads.clear()
        self._bar_refresh_scheduled = False

        self.config.mode = "live"
        self.config.symbol = (self.ed_symbol.text() or "GOLD").strip()
        self.config.exchange = (self.ed_expiry.text() or "TVC").strip()
        self.config.bar_size = self.cmb_interval.currentText()

        self._bootstrap_request_id += 1
        req_id = self._bootstrap_request_id
        self._append_log("[INFO] Nacitam pocatecni snapshot...")
        worker = TaskWorker(
            _task_build_live_bootstrap,
            models=self.models,
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            bar_size=self.config.bar_size,
            max_bars_buffer=int(self.config.max_bars_buffer),
        )
        self._bootstrap_worker = worker
        worker.progress_text.connect(self._append_log)
        worker.result.connect(lambda payload, rid=req_id: self._on_bootstrap_result(rid, payload))
        worker.error.connect(lambda msg, rid=req_id: self._on_bootstrap_error(rid, msg))
        worker.finished.connect(lambda rid=req_id: self._on_bootstrap_finished(rid))
        worker.start()

    def shutdown(self) -> bool:
        ok = self._stop_worker(wait_ms=3000, allow_background=False)
        self.warm = None
        try:
            self._log_timer.stop()
        except Exception:
            pass
        self._save_ui_settings()
        return ok

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.shutdown():
            QMessageBox.warning(
                self,
                "Live Trading Bot",
                "Live worker se jeste nepodarilo bezpecne ukoncit. Pockej chvili a zkus zavreni znovu.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _on_bootstrap_result(self, req_id: int, payload: LiveBootstrapPayload) -> None:
        if req_id != self._bootstrap_request_id or payload is None:
            return
        used_local_snapshot = False
        if int(payload.snapshot_bars or 0) < 2:
            existing_snapshot_bars = int(len(self.live_df.index)) if isinstance(self.live_df, pd.DataFrame) else 0
            if existing_snapshot_bars >= 2:
                self._append_log(
                    f"[INFO] Pocatecni snapshot z TradingView vratil jen {int(payload.snapshot_bars or 0)} baru. "
                    "Ponechavam cerstvou historii z warm-upu."
                )
                used_local_snapshot = True
            elif payload.snapshot_bars > 0:
                self._append_log(
                    f"[WARN] Pocatecni snapshot vratil jen {payload.snapshot_bars} bar. Zkousim lokalni CSV fallback."
                )
                used_local_snapshot = self._try_seed_local_snapshot()
            else:
                self._append_log("[WARN] Pocatecni snapshot prazdny. Zkousim lokalni CSV fallback.")
                used_local_snapshot = self._try_seed_local_snapshot()

        if not used_local_snapshot:
            self._apply_bootstrap_payload(payload, source_label="pocatecni snapshot")

        self._launch_stream_worker()

    def _on_bootstrap_error(self, req_id: int, message: str) -> None:
        if req_id != self._bootstrap_request_id:
            return
        self._append_log(f"[WARN] Pocatecni snapshot selhal: {message}")
        existing_snapshot_bars = int(len(self.live_df.index)) if isinstance(self.live_df, pd.DataFrame) else 0
        if existing_snapshot_bars >= 2:
            self._append_log("[INFO] Ponechavam cerstvou historii z warm-upu a preskakuji lokalni CSV fallback.")
        else:
            self._try_seed_local_snapshot()
        self._launch_stream_worker()

    def _on_bootstrap_finished(self, req_id: int) -> None:
        if req_id == self._bootstrap_request_id:
            self._bootstrap_worker = None

    def _seed_snapshot_from_warmup_history(self, adapter: _WarmAdapter | None) -> bool:
        hist_df = getattr(adapter, "_hist_df", None)
        if not isinstance(hist_df, pd.DataFrame) or hist_df.empty:
            self._append_log("[WARN] Warm-up historie z TradingView je prazdna.")
            return False

        payload = _build_live_bootstrap_payload_from_history_df(
            hist_df,
            self.models,
            max_bars_buffer=int(self.config.max_bars_buffer),
        )
        if int(payload.snapshot_bars or 0) < 2:
            self._append_log(
                f"[WARN] Warm-up historie z TradingView vratila jen {int(payload.snapshot_bars or 0)} bar."
            )
            return False

        existing_snapshot_bars = int(len(self.live_df.index)) if isinstance(self.live_df, pd.DataFrame) else 0
        if existing_snapshot_bars >= payload.snapshot_bars:
            return False

        self._apply_bootstrap_payload(payload, source_label="warm-up snapshot z TradingView")
        return True

    def _preload_historical_data_for_degradation(self) -> None:
        if not self.models:
            self._append_log("[DEGRADATION] Preskakuji preload - zadne modely nacteny")
            self.degradation_console.setPlainText("(Nactete model pro spusteni diagnostiky)")
            return

        if not self.reference_metrics:
            self._append_log("[DEGRADATION] Preskakuji preload - zadne referencni metriky")
            self.degradation_console.setPlainText("(Model neobsahuje referencni metriky)")
            return

        existing = getattr(self, "_degradation_worker", None)
        if existing is not None:
            self._stop_thread_attr("_degradation_worker", wait_ms=500, allow_background=True)

        self._degradation_request_id += 1
        req_id = self._degradation_request_id
        self.degradation_console.setPlainText("Nacitam historicka data z TradingView...\nMuze trvat nekolik sekund...")

        worker = TaskWorker(
            _task_preload_degradation,
            models=self.models,
            symbol=(self.ed_symbol.text() or "GOLD").strip(),
            exchange=(self.ed_expiry.text() or "TVC").strip(),
            timeframe=self.cmb_interval.currentText().replace("mins", "min"),
            degradation_window_size=int(self.degradation_window_size),
        )
        self._degradation_worker = worker
        worker.progress_text.connect(lambda text: self._append_log(f"[DEGRADATION] {text}"))
        worker.result.connect(lambda payload, rid=req_id: self._on_degradation_preload_result(rid, payload))
        worker.error.connect(lambda msg, rid=req_id: self._on_degradation_preload_error(rid, msg))
        worker.finished.connect(lambda rid=req_id: self._on_degradation_preload_finished(rid))
        worker.start()

    def _on_degradation_preload_result(self, req_id: int, payload: DegradationPreloadPayload) -> None:
        if req_id != self._degradation_request_id or payload is None:
            return
        if not payload.predictions:
            self.degradation_console.setPlainText(
                f"Diagnostika bude dostupna po nacteni {self.degradation_window_size} live baru."
            )
            self._append_log("[DEGRADATION] Historicky preload nevratil zadna data.")
            return

        self._prediction_buffer = list(payload.predictions)
        self._price_buffer = list(payload.prices)
        self._y_true_buffer = [None] * len(payload.predictions)
        self._tracked_timestamps = {str(ts) for ts in payload.timestamps}
        self._append_log(f"[DEGRADATION] Nacteno {len(payload.predictions)} historickych baru")

        if len(self._prediction_buffer) >= self.degradation_window_size:
            self._update_degradation_diagnostics()
            self._last_degradation_check = len(self._prediction_buffer)
        else:
            self.degradation_console.setPlainText(
                f"Nacteno {len(self._prediction_buffer)} baru (potreba {self.degradation_window_size})"
            )

    def _on_degradation_preload_error(self, req_id: int, message: str) -> None:
        if req_id != self._degradation_request_id:
            return
        self._append_log(f"[ERROR] Preload historickych dat selhal: {message}")
        self.degradation_console.setPlainText(
            f"Preload historickych dat selhal: {message}\n\n"
            f"Diagnostika bude dostupna po nacteni {self.degradation_window_size} live baru."
        )

    def _on_degradation_preload_finished(self, req_id: int) -> None:
        if req_id == self._degradation_request_id:
            self._degradation_worker = None

    @Slot(dict)
    def _on_bar_closed(self, bar: dict) -> None:
        ts_raw = bar.get("time")
        close = float(bar.get("close", 0.0))
        self._append_log(f"[BAR] {ts_raw} close={close}")

        ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        if pd.isna(ts):
            return
        payload = {
            "time": ts,
            "open": float(bar.get("open", np.nan)),
            "high": float(bar.get("high", np.nan)),
            "low": float(bar.get("low", np.nan)),
            "close": close,
            "volume": float(bar.get("volume", 0) or 0),
        }

        key = int(ts.value)
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
            "low": payload["low"],
            "close": payload["close"],
            "volume": payload["volume"],
        }
        self.live_df = pd.concat([self.live_df, pd.DataFrame([row])], ignore_index=True)
        self.live_df.dropna(subset=["timestamp"], inplace=True)
        self.live_df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
        self.live_df.sort_values("timestamp", inplace=True)
        max_live_buffer = max(700, self.config.max_bars_buffer + 400)
        if len(self.live_df) > max_live_buffer:
            self.live_df = self.live_df.tail(max_live_buffer).reset_index(drop=True)

        self._pending_bar_payloads.append(payload)
        self._last_arrival_utc = pd.Timestamp.now(tz="UTC")
        self._schedule_bar_refresh()

    def _schedule_bar_refresh(self) -> None:
        if self._bar_refresh_scheduled:
            return
        self._bar_refresh_scheduled = True
        QTimer.singleShot(0, self._process_pending_bar_updates)

    def _maybe_send_flip_email_on_last_bar(self) -> None:
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

    def _process_pending_bar_updates(self) -> None:
        self._bar_refresh_scheduled = False
        pending = self._pending_bar_payloads
        self._pending_bar_payloads = []
        if not pending:
            return

        try:
            if self.warm is not None:
                for payload in pending:
                    self.warm.on_new_bar(payload)
                self.lbl_mode.setText("Mode: LIVE" if self.warm.state == "LIVE" else "Mode: WARM-UP")
            self._rescore_all()
            self._maybe_alert_flip_on_last_bar()
            self._maybe_send_flip_email_on_last_bar()
        except Exception as e:
            self._append_log(f"[WARN] Re-score selhal: {e}")

        self._update_freshness()
        self._render_charts()
        if self._pending_bar_payloads:
            self._schedule_bar_refresh()

    def _apply_bootstrap_payload(self, payload: LiveBootstrapPayload, *, source_label: str) -> None:
        self._bars = list(payload.bars or [])
        self._bar_index = {
            int(pd.to_datetime(bar["time"]).value): idx
            for idx, bar in enumerate(self._bars)
            if pd.notna(pd.to_datetime(bar.get("time"), utc=True, errors="coerce"))
        }
        if isinstance(payload.live_df, pd.DataFrame):
            self.live_df = payload.live_df.copy()
        else:
            self.live_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        changed = 0
        for idx, label_map in enumerate(payload.label_maps or []):
            if idx >= len(self.models) or not label_map:
                continue
            self.models[idx]["label_map"] = dict(label_map)
            changed += 1

        if payload.snapshot_bars > 0:
            self._append_log(f"[INFO] Nacten {source_label}: {payload.snapshot_bars} baru.")
            if payload.bars:
                try:
                    first_ts = pd.to_datetime(payload.bars[0].get("time"), utc=True, errors="coerce")
                    last_ts = pd.to_datetime(payload.bars[-1].get("time"), utc=True, errors="coerce")
                    self._append_log(f"[INFO] Snapshot rozsah: {first_ts} -> {last_ts}")
                except Exception:
                    pass
            self._last_arrival_utc = pd.Timestamp.now(tz="UTC")
            try:
                self._rescore_all()
            except Exception as e:
                self._append_log(f"[WARN] Re-score snapshotu selhal: {e}")
            self._update_freshness()
            self._render_charts()
        else:
            self._append_log(f"[WARN] {source_label.capitalize()} prazdny.")

        self._append_log(f"[AUTO] Per-model mapy trid nastaveny pro {changed}/{len(self.models)} modelu.")

    def _local_snapshot_interval_tokens(self) -> tuple[str, ...]:
        tf = (self.cmb_interval.currentText() or getattr(self.config, "bar_size", "1 hour") or "1 hour").strip().lower()
        mapping = {
            "5 min": ("5m", "5min", "5_min"),
            "15 min": ("15m", "15min", "15_min"),
            "30 min": ("30m", "30min", "30_min"),
            "1 hour": ("1hour", "1_hour", "60m", "60min", "1h"),
        }
        return mapping.get(tf, tuple(part for part in tf.replace(" ", "").split(";") if part))

    def _local_snapshot_symbol_tokens(self) -> tuple[str, ...]:
        raw_tokens = {
            str(self.ed_symbol.text() or "").strip().lower(),
            str(getattr(self.config, "symbol", "") or "").strip().lower(),
            str(self.ed_expiry.text() or "").strip().lower(),
            str(getattr(self.config, "exchange", "") or "").strip().lower(),
        }
        raw_tokens.discard("")
        if raw_tokens & {"gold", "gc"}:
            raw_tokens.update({"gold", "gc", "comex", "tvc"})
        return tuple(sorted(raw_tokens))

    def _find_local_snapshot_csv(self) -> Path | None:
        raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
        if not raw_dir.exists():
            return None

        files = [p for p in raw_dir.glob("*.csv") if p.is_file()]
        if not files:
            return None

        interval_tokens = self._local_snapshot_interval_tokens()
        symbol_tokens = self._local_snapshot_symbol_tokens()

        def _score(path: Path) -> tuple[int, int, int]:
            stem = path.stem.lower()
            interval_score = 1 if any(tok and tok in stem for tok in interval_tokens) else 0
            symbol_score = 1 if any(tok and tok in stem for tok in symbol_tokens) else 0
            try:
                mtime_ns = int(path.stat().st_mtime_ns)
            except OSError:
                mtime_ns = 0
            return (interval_score + symbol_score, interval_score, mtime_ns)

        files.sort(key=_score, reverse=True)
        return files[0] if files else None

    def _load_local_snapshot_payload(self, csv_path: Path) -> LiveBootstrapPayload | None:
        df = pd.read_csv(
            csv_path,
            encoding="utf-8",
            engine="python",
            usecols=lambda c: c in {"date", "timestamp", "time", "open", "high", "low", "close", "volume"},
        )
        if df is None or df.empty:
            return None

        time_col = next((col for col in ("timestamp", "time", "date") if col in df.columns), None)
        required_cols = {"open", "high", "low", "close"}
        if time_col is None or not required_cols.issubset(df.columns):
            return None

        if "volume" not in df.columns:
            df["volume"] = 0.0
        payload = _build_live_bootstrap_payload_from_history_df(
            df.rename(columns={time_col: "time"}),
            self.models,
            max_bars_buffer=int(self.config.max_bars_buffer),
        )
        if int(payload.snapshot_bars or 0) < 1:
            return None
        return payload

    def _try_seed_local_snapshot(self) -> bool:
        csv_path = self._find_local_snapshot_csv()
        if csv_path is None:
            self._append_log("[WARN] Lokalni CSV fallback nebyl nalezen.")
            return False

        try:
            payload = self._load_local_snapshot_payload(csv_path)
        except Exception as e:
            self._append_log(f"[WARN] Lokalni CSV fallback selhal ({csv_path.name}): {e}")
            return False

        if payload is None or int(payload.snapshot_bars or 0) < 2:
            self._append_log(f"[WARN] Lokalni CSV fallback nema dost baru: {csv_path.name}")
            return False

        self._apply_bootstrap_payload(payload, source_label=f"lokalni snapshot {csv_path.name}")
        return True

    def _apply_market_context_from_model_paths(self, paths: list[str]) -> None:
        for raw_path in paths:
            model_path = str(raw_path or "").strip()
            if not model_path:
                continue
            try:
                meta = read_sidecar_model_meta(model_path)
            except Exception:
                continue
            context = live_market_context_from_model_meta(meta)
            if not isinstance(context, dict):
                continue

            symbol = str(context.get("symbol") or "").strip()
            exchange = str(context.get("exchange") or "").strip()
            bar_size = str(context.get("bar_size") or "").strip()
            if symbol and self.ed_symbol.text().strip() != symbol:
                self.ed_symbol.setText(symbol)
            if exchange and self.ed_expiry.text().strip() != exchange:
                self.ed_expiry.setText(exchange)
            if bar_size:
                idx = self.cmb_interval.findText(bar_size)
                if idx >= 0 and self.cmb_interval.currentIndex() != idx:
                    self.cmb_interval.setCurrentIndex(idx)

            self.config.symbol = symbol or self.config.symbol
            self.config.exchange = exchange or self.config.exchange
            self.config.bar_size = bar_size or self.config.bar_size
            self._append_log(
                f"[MODEL] TradingView session synchronizovana z metadata: "
                f"{self.config.symbol} / {self.config.exchange} / {self.config.bar_size}"
            )
            return

    def set_model_paths(self, paths: str | list[str]) -> None:
        if isinstance(paths, str):
            items = [paths]
        else:
            items = [str(path).strip() for path in (paths or []) if str(path).strip()]
        text = ";".join(items)
        if text and self.le_model_path.text() != text:
            self.le_model_path.setText(text)
        if items:
            self._apply_market_context_from_model_paths(items)

def live_interval_label_from_model_timeframe(timeframe: str | None) -> str | None:
    raw = str(timeframe or "").strip().lower().replace(" ", "")
    mapping = {
        "5m": "5 min",
        "5min": "5 min",
        "15m": "15 min",
        "15min": "15 min",
        "30m": "30 min",
        "30min": "30 min",
        "1h": "1 hour",
        "1hour": "1 hour",
        "60m": "1 hour",
        "60min": "1 hour",
    }
    return mapping.get(raw)

def live_tradingview_market_from_model(symbol: str | None, exchange: str | None) -> tuple[str | None, str | None]:
    raw_symbol = str(symbol or "").strip().upper() or None
    raw_exchange = str(exchange or "").strip().upper() or None
    tv_mapping = {
        ("GC", "COMEX"): ("GOLD", "TVC"),
    }
    return tv_mapping.get((raw_symbol, raw_exchange), (raw_symbol, raw_exchange))

def live_market_context_from_model_meta(meta: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(meta, dict):
        return None
    symbol, exchange = live_tradingview_market_from_model(
        meta.get("instrument"),
        meta.get("exchange"),
    )
    bar_size = live_interval_label_from_model_timeframe(meta.get("timeframe"))
    if not symbol or not exchange or not bar_size:
        return None
    return {
        "symbol": symbol,
        "exchange": exchange,
        "bar_size": bar_size,
    }

class LiveTradingBotTab(LiveBotWidget):
    pass

class LiveBotTab(LiveBotWidget):
    pass
