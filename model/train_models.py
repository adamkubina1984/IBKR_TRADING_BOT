# ibkr_trading_bot/model/train_models.py
from __future__ import annotations

import hashlib
import json as jsonlib
import logging
import pathlib
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from ibkr_trading_bot.core.services.model_service import runtime_python_version, runtime_sklearn_version
from ibkr_trading_bot.model.feature_stability import (
    compute_feature_stability_score,
    evaluate_feature_stability_filter,
)

# --- volitelné knihovny
try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except Exception:
    optuna = None  # type: ignore[assignment]
    HAS_OPTUNA = False


LOGGER = logging.getLogger(__name__)
_VALID_TRAINING_MODES = {"quick", "standard", "strict"}
_CANONICAL_TRAINING_MODE_ALIASES = {
    "explore": "quick",
    "refine": "standard",
    "refresh": "strict",
}

# --- purged walk-forward split
from ibkr_trading_bot.model.tscv import PurgedWalkForwardSplit

# --- metriky / scorer
try:
    from ibkr_trading_bot.utils.metrics import calculate_metrics, pnl_scorer  # type: ignore
    HAS_CALC_METRICS = True
except ImportError:
    def pnl_scorer(estimator, X_val, y_val, df_val=None, fee=0.0, slippage=0.0):
        pred = _call_with_feature_name_warning_suppressed(estimator.predict, X_val)
        return float((pred == y_val).mean())
    HAS_CALC_METRICS = False

# ------------------- Pomocné -------------------
def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _project_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "ibkr_trading_bot").is_dir():
            return p / "ibkr_trading_bot"
    return here.parent

def _model_dir() -> pathlib.Path:
    root = _project_root()
    out = root / "model_outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out

def _normalize_legacy_training_mode(mode: Any) -> str:
    normalized = str(mode or "standard").strip().lower()
    normalized = _CANONICAL_TRAINING_MODE_ALIASES.get(normalized, normalized)
    if normalized not in _VALID_TRAINING_MODES:
        raise ValueError(
            "Unsupported training_mode. Expected one of quick/standard/strict "
            "or canonical aliases explore/refine/refresh."
        )
    return normalized


def _chain_shortlist_path(name_prefix: str | None, estimator_name: str) -> pathlib.Path:
    est_short = str(estimator_name or "model").lower()
    stem = str(name_prefix or "default").strip()
    safe_stem = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in stem)
    return _model_dir() / f"{safe_stem}_{est_short}_chain_shortlist.json"


def _normalize_candidate_criterion(v: str | None) -> str:
    x = str(v or "").strip().lower()
    aliases = {
        "balanced": "balanced",
        "balance": "balanced",
        "profit_first": "profit_first",
        "profit": "profit_first",
        "robustness_first": "robustness_first",
        "robust": "robustness_first",
        "recall_balance": "recall_balance",
        "recall": "recall_balance",
    }
    return aliases.get(x, "balanced")


def _normalize_search_backend(value: str | None) -> str:
    txt = str(value or "").strip().lower()
    return txt if txt in {"grid", "optuna"} else "grid"


def _normalize_estimator_family(value: str | None) -> str:
    txt = str(value or "").strip().lower()
    if txt in {"hgbt", "histgb", "histgradientboosting"}:
        return "hgbt"
    if txt in {"lgb", "lightgbm"}:
        return "lgb"
    if txt in {"xgb", "xgboost"}:
        return "xgb"
    if txt in {"rf", "random_forest", "randomforest"}:
        return "rf"
    if txt in {"et", "extratrees", "extra_trees"}:
        return "et"
    if txt in {"svm", "svc"}:
        return "svm"
    return txt or "hgbt"


def _optuna_supported_estimator(value: str | None) -> bool:
    return _normalize_estimator_family(value) in {"hgbt", "lgb"}


def _normalize_optuna_trials(value: Any) -> int | None:
    try:
        iv = int(value)
        return iv if iv > 0 else None
    except Exception:
        return None


def _normalize_optuna_timeout_seconds(value: Any) -> float | None:
    try:
        fv = float(value)
        return float(fv) if np.isfinite(fv) and fv > 0.0 else None
    except Exception:
        return None


def _resolve_search_backend(
    value: str | None,
    *,
    estimator_name: str | None = None,
) -> tuple[str, str, str | None]:
    requested = _normalize_search_backend(value)
    if requested == "optuna" and not HAS_OPTUNA:
        LOGGER.warning(
            "search_backend='optuna' requested but Optuna is not available; falling back to 'grid'."
        )
        return requested, "grid", "optuna_not_available"
    if requested == "optuna" and not _optuna_supported_estimator(estimator_name):
        LOGGER.warning(
            "search_backend='optuna' requested for unsupported estimator '%s'; falling back to 'grid'.",
            estimator_name,
        )
        return requested, "grid", "optuna_estimator_not_supported"
    return requested, requested, None


def _coerce_grid_choices(grid: dict[str, Any] | None) -> dict[str, list[Any]]:
    choices: dict[str, list[Any]] = {}
    if not isinstance(grid, dict):
        return choices
    for key, raw_values in grid.items():
        if isinstance(raw_values, (str, bytes)):
            values = [raw_values]
        else:
            try:
                values = list(raw_values)
            except Exception:
                values = [raw_values]
        values = [v for v in values if v is not None]
        if values:
            choices[str(key)] = values
    return choices


def _build_chain_signature(
    *,
    name_prefix: str | None,
    estimator_name: str,
    meta_extra: dict[str, Any],
    holdout_mode: str,
    holdout_pct: float | None,
    holdout_min_bars: int,
    holdout_max_bars: int | None,
    holdout_bars: int,
    label_lookahead_bars: int,
    is_ternary: bool,
) -> tuple[str, dict[str, Any]]:
    sig = {
        "name_prefix": str(name_prefix or ""),
        "estimator_name": str(estimator_name or ""),
        "instrument": str(meta_extra.get("instrument", "")),
        "exchange": str(meta_extra.get("exchange", "")),
        "timeframe": str(meta_extra.get("timeframe", "")),
        "n_total_bars": int(meta_extra.get("n_total_bars", 0) or 0),
        "label_horizon_bars": int(meta_extra.get("label_horizon_bars", 0) or 0),
        "label_take_profit_bps": float(meta_extra.get("label_take_profit_bps", 0.0) or 0.0),
        "label_stop_loss_bps": float(meta_extra.get("label_stop_loss_bps", 0.0) or 0.0),
        "label_same_bar_policy": str(meta_extra.get("label_same_bar_policy", "")),
        "label_lookahead_bars": int(max(0, label_lookahead_bars)),
        "is_ternary": bool(is_ternary),
        "holdout_mode": str(holdout_mode or ""),
        "holdout_pct": (float(holdout_pct) if holdout_pct is not None else None),
        "holdout_min_bars": int(holdout_min_bars),
        "holdout_max_bars": (int(holdout_max_bars) if holdout_max_bars is not None else None),
        "holdout_bars_applied": int(holdout_bars),
    }
    blob = jsonlib.dumps(sig, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    return digest, sig


def _select_threshold_calibration_split(
    df_train: pd.DataFrame,
    *,
    is_ternary: bool,
    threshold_calibration_enabled: bool,
    threshold_calibration_pct: float,
    threshold_calibration_min_bars: int,
    threshold_calibration_max_bars: int,
    threshold_calibration_train_min_guard: int,
    embargo: int,
    label_lookahead_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any], int]:
    n_train_eff = int(len(df_train))
    label_lookahead = int(max(0, label_lookahead_bars))
    effective_embargo = int(max(int(embargo), label_lookahead))

    selection: dict[str, Any] = {
        "enabled": False,
        "mode": "disabled",
        "requested_pct": None,
        "requested_bars": 0,
        "min_bars": 0,
        "max_bars": None,
        "applied_bars": 0,
        "train_core_bars": int(n_train_eff),
        "train_full_bars": int(n_train_eff),
        "train_min_guard": 0,
        "gap_bars": 0,
        "embargo_bars": int(effective_embargo),
        "label_lookahead_bars": int(label_lookahead),
        "embargo_respects_label_lookahead": bool(effective_embargo >= label_lookahead),
        "no_overlap": True,
    }

    df_train_core = df_train.copy()
    df_threshold_calib: pd.DataFrame | None = None
    if not is_ternary or not bool(threshold_calibration_enabled):
        return df_train_core, df_threshold_calib, selection, effective_embargo

    calib_pct = float(np.clip(threshold_calibration_pct, 0.0, 0.90))
    calib_min_bars = max(0, int(threshold_calibration_min_bars))
    calib_max_bars = max(0, int(threshold_calibration_max_bars))
    calib_train_min_guard = max(100, int(threshold_calibration_train_min_guard))

    requested_calib_bars = int(round(float(n_train_eff) * calib_pct))
    n_calib = int(max(requested_calib_bars, calib_min_bars))
    if calib_max_bars > 0:
        n_calib = int(min(n_calib, calib_max_bars))

    max_calib_allowed = max(0, int(n_train_eff - calib_train_min_guard - effective_embargo))
    n_calib = int(min(max(0, n_calib), max_calib_allowed))

    selection = {
        "enabled": True,
        "mode": "tail_pct",
        "requested_pct": float(calib_pct),
        "requested_bars": int(requested_calib_bars),
        "min_bars": int(calib_min_bars),
        "max_bars": int(calib_max_bars) if calib_max_bars > 0 else None,
        "applied_bars": int(n_calib),
        "train_core_bars": int(n_train_eff),
        "train_full_bars": int(n_train_eff),
        "train_min_guard": int(calib_train_min_guard),
        "gap_bars": 0,
        "embargo_bars": int(effective_embargo),
        "label_lookahead_bars": int(label_lookahead),
        "embargo_respects_label_lookahead": bool(effective_embargo >= label_lookahead),
        "no_overlap": True,
    }

    if n_calib <= 0:
        selection["reason"] = "not_enough_train_bars_for_calibration_split"
        return df_train_core, df_threshold_calib, selection, effective_embargo

    calib_start = int(n_train_eff - n_calib)
    core_end = int(max(0, calib_start - effective_embargo))
    df_train_core = df_train.iloc[:core_end].reset_index(drop=True)
    df_threshold_calib = df_train.iloc[calib_start:].reset_index(drop=True)

    gap_bars = int(max(0, calib_start - core_end))
    selection["train_core_bars"] = int(len(df_train_core))
    selection["applied_bars"] = int(len(df_threshold_calib))
    selection["gap_bars"] = int(gap_bars)
    selection["no_overlap"] = bool(gap_bars >= effective_embargo)
    return df_train_core, df_threshold_calib, selection, effective_embargo


def _params_key(params: dict[str, Any]) -> str:
    try:
        return jsonlib.dumps(dict(params or {}), sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        return str(params)


def _candidate_priority_score(row: dict[str, Any], criterion: str) -> float:
    crit = _normalize_candidate_criterion(criterion)

    def _fv(key: str, default: float = float("nan")) -> float:
        try:
            v = float(row.get(key, default))
            return float(v) if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    cv_score = _fv("cv_score", -1e9)
    f1m = _fv("f1_macro_3", 0.0)
    profit = _fv("profit_net", 0.0)
    sharpe = _fv("sharpe", 0.0)
    pf = _fv("pf", 1.0)
    rec_short = _fv("rec_short", float("nan"))
    rec_long = _fv("rec_long", float("nan"))
    n_dir = _fv("n_dir_pred_mean", 0.0)
    n_short = _fv("n_short_pred_mean", 0.0)
    n_long = _fv("n_long_pred_mean", 0.0)

    profit_c = float(np.clip(profit / 250.0, -2.0, 2.0))
    sharpe_c = float(np.clip(sharpe / 2.0, -1.5, 1.5))
    pf_c = float(np.clip(pf - 1.0, -1.0, 2.0))
    trades_c = _trade_count_preference_score(n_dir)

    rec_min = 0.0
    rec_bal = 0.5
    if np.isfinite(rec_short) and np.isfinite(rec_long):
        rec_min = float(np.clip(min(rec_short, rec_long), 0.0, 1.0))
        rec_bal = float(np.clip(1.0 - abs(rec_short - rec_long), 0.0, 1.0))
    elif np.isfinite(rec_short):
        rec_min = float(np.clip(rec_short, 0.0, 1.0))
        rec_bal = rec_min
    elif np.isfinite(rec_long):
        rec_min = float(np.clip(rec_long, 0.0, 1.0))
        rec_bal = rec_min

    if crit == "profit_first":
        score = (
            (0.45 * profit_c)
            + (0.20 * pf_c)
            + (0.15 * cv_score)
            + (0.10 * sharpe_c)
            + (0.10 * trades_c)
        )
    elif crit == "robustness_first":
        score = (
            (0.35 * sharpe_c)
            + (0.25 * cv_score)
            + (0.20 * f1m)
            + (0.10 * rec_bal)
            + (0.10 * trades_c)
        )
    elif crit == "recall_balance":
        score = (
            (0.25 * cv_score)
            + (0.20 * f1m)
            + (0.25 * rec_min)
            + (0.20 * rec_bal)
            + (0.10 * trades_c)
        )
    else:
        score = (
            (0.40 * cv_score)
            + (0.20 * f1m)
            + (0.20 * profit_c)
            + (0.10 * sharpe_c)
            + (0.10 * rec_bal)
        )
    if n_short <= 0.0 or n_long <= 0.0:
        score -= 0.50
    elif n_dir > 0.0:
        min_side_share = float(np.clip(min(n_short, n_long) / max(n_dir, 1e-9), 0.0, 0.5))
        side_share_floor = 0.20
        if min_side_share < side_share_floor:
            score -= 0.35 * ((side_share_floor - min_side_share) / side_share_floor)
    return float(score)


def _trade_count_preference_score(
    n_trades: float,
    *,
    hard_min: float = 60.0,
    preferred_low: float = 150.0,
    preferred_high: float = 300.0,
    soft_max: float = 450.0,
) -> float:
    try:
        value = float(n_trades)
    except Exception:
        return 0.0
    if not np.isfinite(value) or value < hard_min:
        return 0.0
    if value <= preferred_low:
        span = max(preferred_low - hard_min, 1.0)
        return float(np.clip((value - hard_min) / span, 0.0, 1.0))
    if value <= preferred_high:
        return 1.0
    if value <= soft_max:
        span = max(soft_max - preferred_high, 1.0)
        return float(np.clip(1.0 - ((value - preferred_high) / span), 0.0, 1.0))
    return 0.0


def _fallback_candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    def _finite_side_recall(key: str) -> float:
        try:
            value = float(row.get(key, np.nan))
        except Exception:
            return -1.0
        return float(value) if np.isfinite(value) else -1.0

    rec_short = _finite_side_recall("rec_short")
    rec_long = _finite_side_recall("rec_long")
    available = [value for value in (rec_short, rec_long) if value >= 0.0]
    if not available:
        rec_min = -1.0
        rec_mean = -1.0
        rec_bal = -1.0
    elif len(available) == 1:
        rec_min = available[0]
        rec_mean = available[0]
        rec_bal = available[0]
    else:
        rec_min = min(available)
        rec_mean = float(sum(available) / len(available))
        rec_bal = float(np.clip(1.0 - abs(rec_short - rec_long), 0.0, 1.0))
    try:
        cheap_score = float(row.get("cheap_score", -1e18))
    except Exception:
        cheap_score = -1e18
    return float(rec_min), float(rec_bal), float(rec_mean), float(cheap_score)


def _rank_candidates_for_chain(
    rows: list[dict[str, Any]],
    criterion: str,
) -> list[dict[str, Any]]:
    crit = _normalize_candidate_criterion(criterion)
    ranked: list[dict[str, Any]] = []
    for r in list(rows or []):
        rr = dict(r)
        rr["criterion_score"] = _candidate_priority_score(rr, crit)
        ranked.append(rr)
    ranked.sort(
        key=lambda x: (
            float(x.get("criterion_score", -1e18)),
            float(x.get("cv_score", -1e18)),
            -float(x.get("cv_std", 1e18)),
        ),
        reverse=True,
    )
    return ranked

# --- bezpečnější výběr featur (bez leaků + odfiltrování kvazi-konstant)
SAFE_EXCLUDE_PATTERNS = (
    "target", "label", "class", "y_",
    "future", "fwd", "lead", "leak",
    "signal", "proba", "pred", "score",
    "pnl", "ret_", "return_fwd", "trade_",
    "event_", "barrier", "tb_", "tripbar", "horizon"
)

def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Preferujeme engineered featury; zároveň vylučujeme potenciální leaky a
    featury s velmi nízkou variabilitou. Když nic nezůstane, spadneme na
    všechny numerické mimo timestamp/target.
    """
    ignore_core = {"timestamp", "target"}
    numeric_cols = [
        c for c in df.columns
        if c not in ignore_core and pd.api.types.is_numeric_dtype(df[c])
    ]
    filtered = []
    for c in numeric_cols:
        cname = c.lower()
        if any(tok in cname for tok in SAFE_EXCLUDE_PATTERNS):
            continue
        filtered.append(c)
    try:
        if filtered:
            nunique = df[filtered].nunique(dropna=True)
            filtered = [c for c in filtered if int(nunique.get(c, 0)) > 5]
    except Exception:
        pass
    if not filtered:
        filtered = numeric_cols
    return filtered

def _build_estimator(name: str) -> tuple[object, dict[str, list]]:
    name = (name or "hgbt").lower()
    if name in ("hgbt","histgb","histgradientboosting"):
        est = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.04,
            l2_regularization=0.5,
            max_iter=250,
            random_state=42,
        )
        # Conservative grid for better holdout robustness on intraday data.
        grid = {
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.06],
            "max_iter": [200, 300],
            "l2_regularization": [0.1, 0.5, 1.0],
        }
        return est, grid
    if name in ("rf","random_forest","randomforest"):
        est = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced")
        grid = {"n_estimators":[300,500,800],"max_depth":[None,8,16],"min_samples_leaf":[1,2,4]}
        return est, grid
    if name in ("et","extratrees","extra_trees"):
        est = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced")
        grid = {"n_estimators":[400,600,800],"max_depth":[None,8,16],"min_samples_leaf":[1,2,4]}
        return est, grid
    if name in ("svm","svc"):
        est = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"))])
        grid = {"clf__C":[0.5,1.0,2.0], "clf__gamma":["scale",0.1,0.01]}
        return est, grid
    if name in ("xgb","xgboost") and HAS_XGB:
        est = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=42, n_jobs=-1, scale_pos_weight=2.5)
        grid = {"n_estimators":[400,700], "max_depth":[4,6,8], "learning_rate":[0.03,0.06,0.1], "subsample":[0.8,1.0], "colsample_bytree":[0.8,1.0]}
        return est, grid
    if name in ("lgb","lightgbm") and HAS_LGB:
        est = lgb.LGBMClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=100,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1,
        )
        # Compact conservative grid for faster training and better holdout robustness.
        grid = {
            "n_estimators": [300, 500],
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.05],
            "num_leaves": [15, 31],
            "min_child_samples": [80, 120],
            "reg_lambda": [0.5, 2.0],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8],
        }
        return est, grid
    return _build_estimator("hgbt")

def _ensure_pipeline(estimator) -> object:
    if isinstance(estimator, Pipeline):
        return estimator
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("clf", estimator)])

def _namespaced_param_grid(estimator, grid: dict | None) -> dict | None:
    if grid is None:
        return None
    if not isinstance(estimator, Pipeline):
        return grid
    step = estimator.steps[-1][0]
    return {(k if "__" in k else f"{step}__{k}"): v for k, v in grid.items()}


def _feature_names_for_estimator(estimator) -> list[str] | None:
    try:
        names = getattr(estimator, "feature_names_in_", None)
        if names is not None:
            return [str(c) for c in list(names)]
    except Exception:
        pass
    try:
        if isinstance(estimator, Pipeline):
            last = estimator.steps[-1][1]
            names = getattr(last, "feature_names_in_", None)
            if names is not None:
                return [str(c) for c in list(names)]
    except Exception:
        pass
    try:
        names = getattr(estimator, "feature_name_", None)
        if names is not None:
            names_list = [str(c) for c in list(names) if str(c)]
            if names_list:
                return names_list
    except Exception:
        pass
    try:
        booster = getattr(estimator, "booster_", None)
        if booster is not None and hasattr(booster, "feature_name"):
            names = booster.feature_name()
            names_list = [str(c) for c in list(names) if str(c)]
            if names_list:
                return names_list
    except Exception:
        pass
    try:
        if isinstance(estimator, Pipeline):
            last = estimator.steps[-1][1]
            names = getattr(last, "feature_name_", None)
            if names is not None:
                names_list = [str(c) for c in list(names) if str(c)]
                if names_list:
                    return names_list
    except Exception:
        pass
    try:
        if isinstance(estimator, Pipeline):
            last = estimator.steps[-1][1]
            booster = getattr(last, "booster_", None)
            if booster is not None and hasattr(booster, "feature_name"):
                names = booster.feature_name()
                names_list = [str(c) for c in list(names) if str(c)]
                if names_list:
                    return names_list
    except Exception:
        pass
    return None


def _align_X_for_estimator(estimator, X):
    """Ensure prediction input has stable DataFrame columns matching fitted feature names."""
    try:
        names = _feature_names_for_estimator(estimator)
        if names:
            if isinstance(X, pd.DataFrame):
                Xdf = X.copy()
                if all(col in Xdf.columns for col in names):
                    return Xdf.reindex(columns=names, fill_value=0.0)
                if len(Xdf.columns) == len(names):
                    Xdf.columns = names
                    return Xdf
                for col in names:
                    if col not in Xdf.columns:
                        Xdf[col] = 0.0
                return Xdf.reindex(columns=names, fill_value=0.0)

            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim == 2 and arr.shape[1] == len(names):
                return pd.DataFrame(arr, columns=names)

            Xdf = pd.DataFrame(arr)
            for col in names:
                if col not in Xdf.columns:
                    Xdf[col] = 0.0
            return Xdf.reindex(columns=names, fill_value=0.0)
    except Exception:
        pass
    return X


def _call_with_feature_name_warning_suppressed(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but .* was fitted with feature names",
            category=UserWarning,
        )
        return func(*args, **kwargs)

def _predict_proba(estimator, X: pd.DataFrame, class_idx: int = 1) -> np.ndarray | None:
    """
    Return predicted probability for class at given index (1 for binary class 1, or for multiclass).
    For ternary, use class_idx=2 for LONG class.
    """
    try:
        X_use = _align_X_for_estimator(estimator, X)
        if hasattr(estimator, "predict_proba"):
            proba = _call_with_feature_name_warning_suppressed(estimator.predict_proba, X_use)
            if isinstance(proba, np.ndarray) and proba.ndim == 2:
                # if more than 2 classes, return proba for class_idx; else return class 1
                return proba[:, min(class_idx, proba.shape[1] - 1)]
            return np.asarray(proba).ravel()
        if isinstance(estimator, Pipeline):
            last = estimator.steps[-1][1]
            if hasattr(last, "predict_proba"):
                proba = _call_with_feature_name_warning_suppressed(estimator.predict_proba, X_use)
                if isinstance(proba, np.ndarray) and proba.ndim == 2:
                    return proba[:, min(class_idx, proba.shape[1] - 1)]
                return np.asarray(proba).ravel()
    except Exception:
        pass
    return None


def _unwrap_final_estimator(estimator):
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


def _extract_feature_importance_values(estimator, n_features: int) -> np.ndarray | None:
    est = _unwrap_final_estimator(estimator)
    try:
        if hasattr(est, "feature_importances_"):
            values = np.asarray(getattr(est, "feature_importances_"), dtype=float).ravel()
            if values.size == int(n_features):
                return values
        predictors = getattr(est, "_predictors", None)
        if predictors is not None:
            gains = np.zeros(int(n_features), dtype=float)
            for stage in list(predictors):
                for tree in list(stage or []):
                    nodes = getattr(tree, "nodes", None)
                    if nodes is None:
                        continue
                    feature_idx = np.asarray(nodes["feature_idx"], dtype=int)
                    split_gain = np.asarray(nodes["gain"], dtype=float)
                    is_leaf = np.asarray(nodes["is_leaf"], dtype=bool)
                    valid = (
                        ~is_leaf
                        & np.isfinite(split_gain)
                        & (split_gain > 0.0)
                        & (feature_idx >= 0)
                        & (feature_idx < int(n_features))
                    )
                    if not np.any(valid):
                        continue
                    np.add.at(gains, feature_idx[valid], split_gain[valid])
            if np.any(gains > 0.0):
                return gains
        coef = getattr(est, "coef_", None)
        if coef is not None:
            coef_arr = np.asarray(coef, dtype=float)
            if coef_arr.ndim <= 1:
                values = np.abs(coef_arr).ravel()
            else:
                values = np.mean(np.abs(coef_arr), axis=0).ravel()
            if values.size == int(n_features):
                return values
    except Exception:
        pass
    return None


def _normalize_feature_importance_values(values: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    norm = float(np.sum(np.abs(arr)))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return arr / norm


def _extract_normalized_feature_importance_map(
    estimator,
    feature_names: list[str],
    X: pd.DataFrame | None = None,
    y: np.ndarray | None = None,
) -> dict[str, float] | None:
    values = _extract_feature_importance_values(estimator, len(feature_names))
    if values is None:
        return {}
    normalized = _normalize_feature_importance_values(values)
    if normalized is None:
        return None
    return {
        str(feature_names[i]): float(normalized[i])
        for i in range(min(len(feature_names), normalized.size))
    }


def _aggregate_feature_stability(
    fold_feature_importances: list[dict[str, float]],
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    valid_folds = [
        fi for fi in fold_feature_importances
        if isinstance(fi, dict) and len(fi) > 0
    ]
    if not valid_folds or not feature_names:
        return {}

    matrix = np.asarray(
        [
            [float(fi.get(name, 0.0)) for name in feature_names]
            for fi in valid_folds
        ],
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(feature_names):
        return {}

    mean_vals = np.mean(matrix, axis=0)
    std_vals = np.std(matrix, axis=0)
    min_vals = np.min(matrix, axis=0)
    max_vals = np.max(matrix, axis=0)
    folds_present_vals = np.asarray(
        [sum(1 for fi in valid_folds if name in fi) for name in feature_names],
        dtype=int,
    )
    order = np.argsort(mean_vals)[::-1]

    return {
        str(feature_names[idx]): {
            "mean": float(mean_vals[idx]),
            "std": float(std_vals[idx]),
            "min": float(min_vals[idx]),
            "max": float(max_vals[idx]),
            "folds_present": int(folds_present_vals[idx]),
        }
        for idx in order
    }


def _collect_fold_feature_importances_for_params(
    *,
    cv,
    base_estimator,
    params: dict[str, Any],
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    feature_names: list[str],
    balance_classes: bool,
    class_balance_power: float,
    class_balance_max_ratio: float,
) -> list[dict[str, float]]:
    fold_feature_importances: list[dict[str, float]] = []
    for tr_idx, _ in cv.split(X_all):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
        est = _fit_with_params(base_estimator, params)
        sw_tr = (
            _balanced_sample_weight(
                y_tr,
                power=class_balance_power,
                max_ratio=class_balance_max_ratio,
            )
            if balance_classes
            else None
        )
        _fit_estimator(est, X_tr, y_tr, sample_weight=sw_tr)
        fold_imp = _extract_normalized_feature_importance_map(est, feature_names, X_tr, y_tr)
        if fold_imp is None:
            continue
        fold_feature_importances.append(dict(fold_imp))
    return fold_feature_importances


def _ternary_predict_mapped(
    prob_short: np.ndarray,
    prob_long: np.ndarray,
    thr_short: float,
    thr_long: float,
) -> np.ndarray:
    """
    Build mapped ternary labels: 0=SHORT, 1=HOLD, 2=LONG.
    If both SHORT/LONG thresholds are hit, decide by larger margin above threshold.
    """
    ps = np.asarray(prob_short, dtype=float)
    pl = np.asarray(prob_long, dtype=float)
    n = min(ps.size, pl.size)
    if n <= 0:
        return np.asarray([], dtype=int)
    ps = ps[:n]
    pl = pl[:n]
    ts = float(thr_short)
    tl = float(thr_long)

    y = np.ones(n, dtype=int)
    hit_s = ps >= ts
    hit_l = pl >= tl

    y[hit_s & ~hit_l] = 0
    y[hit_l & ~hit_s] = 2

    both = hit_s & hit_l
    if both.any():
        s_margin = ps[both] - ts
        l_margin = pl[both] - tl
        choose_short = s_margin > l_margin
        ties = s_margin == l_margin
        if np.any(ties):
            ps_b = ps[both]
            pl_b = pl[both]
            choose_short[ties] = ps_b[ties] > pl_b[ties]
        y[both] = np.where(choose_short, 0, 2).astype(int)
    return y


def _predict_labels_for_metrics(
    estimator,
    X: pd.DataFrame,
    *,
    decision_threshold: float = 0.5,
    ternary_threshold_short: float | None = None,
    ternary_threshold_long: float | None = None,
    is_ternary: bool = False,
) -> tuple[np.ndarray, int, np.ndarray | None]:
    """
    Build labels for metrics from estimator outputs.

    Returns:
      - y_pred labels
      - n_signals (binary: count(label==1), ternary: count(label!=1/HOLD))
      - proba matrix/vector if available, else None
    """
    proba = None
    try:
        X_use = _align_X_for_estimator(estimator, X)
        if hasattr(estimator, "predict_proba"):
            proba = _call_with_feature_name_warning_suppressed(estimator.predict_proba, X_use)
    except Exception:
        proba = None
        X_use = X

    thr = float(decision_threshold)
    thr_short = float(ternary_threshold_short) if ternary_threshold_short is not None else thr
    thr_long = float(ternary_threshold_long) if ternary_threshold_long is not None else thr

    if isinstance(proba, np.ndarray) and proba.ndim == 2:
        if is_ternary and proba.shape[1] >= 3:
            prob_short = proba[:, 0]
            prob_long = proba[:, 2]
            # mapped ternary labels expected by sklearn model: 0=SHORT, 1=HOLD, 2=LONG
            y_pred = _ternary_predict_mapped(prob_short, prob_long, thr_short, thr_long)
            n_signals = int((y_pred != 1).sum())
            return y_pred, n_signals, proba

        if proba.shape[1] >= 2:
            p1 = proba[:, 1]
            y_pred = (p1 >= thr).astype(int)
            n_signals = int((y_pred == 1).sum())
            return y_pred, n_signals, p1

    y_pred = np.asarray(_call_with_feature_name_warning_suppressed(estimator.predict, X_use)).astype(int)
    if is_ternary:
        n_signals = int((y_pred != 1).sum())
    else:
        n_signals = int((y_pred == 1).sum())
    return y_pred, n_signals, None


def _mapped_ternary_to_signed(arr: np.ndarray) -> np.ndarray:
    """Convert mapped ternary labels 0/1/2 to signed -1/0/1."""
    a = np.asarray(arr).astype(int)
    return np.where(a == 0, -1, np.where(a == 1, 0, 1)).astype(int)

def _fit_with_params(base_estimator, params: dict) -> object:
    est = clone(base_estimator)
    try:
        est.set_params(**params)
    except Exception:
        pass
    return est


def _balanced_sample_weight(
    y: np.ndarray,
    *,
    power: float = 1.0,
    max_ratio: float | None = None,
) -> np.ndarray | None:
    """Tempered inverse-frequency class weights as sample_weight."""
    try:
        yv = np.asarray(y).astype(int)
        classes, counts = np.unique(yv, return_counts=True)
        if classes.size <= 1:
            return None
        n = float(len(yv))
        k = float(len(classes))
        base_map = {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts) if cnt > 0}
        if not base_map:
            return None

        p = float(np.clip(power, 0.0, 1.0))
        w_map = {
            int(c): float((w ** p) if (np.isfinite(w) and w > 0.0) else 1.0)
            for c, w in base_map.items()
        }
        if max_ratio is not None and np.isfinite(float(max_ratio)) and float(max_ratio) > 1.0:
            w_min = max(1e-12, float(min(w_map.values())))
            cap = float(w_min * float(max_ratio))
            w_map = {int(c): float(min(w, cap)) for c, w in w_map.items()}

        w = np.asarray([w_map.get(int(v), 1.0) for v in yv], dtype=float)
        w_mean = float(np.mean(w))
        if np.isfinite(w_mean) and w_mean > 0.0:
            w = w / w_mean
        return w
    except Exception:
        return None


def _fit_estimator(estimator, X, y, sample_weight: np.ndarray | None = None):
    """Fit estimator with optional sample_weight (supports Pipeline namespacing)."""
    X_use = _align_X_for_estimator(estimator, X)
    if sample_weight is None:
        estimator.fit(X_use, y)
        return estimator
    try:
        estimator.fit(X_use, y, sample_weight=sample_weight)
        return estimator
    except TypeError:
        pass
    except Exception:
        pass
    try:
        if isinstance(estimator, Pipeline):
            step = estimator.steps[-1][0]
            estimator.fit(X_use, y, **{f"{step}__sample_weight": sample_weight})
            return estimator
    except Exception:
        pass
    estimator.fit(X_use, y)
    return estimator


def _target_distribution(y: np.ndarray) -> dict[str, int]:
    arr = np.asarray(y).astype(int)
    if arr.size == 0:
        return {}
    vals, cnts = np.unique(arr, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, cnts)}


def _directional_f1_from_metrics(metrics: dict[str, Any]) -> float:
    """Mean F1 over directional classes (-1 SHORT, +1 LONG) when available."""
    try:
        pc = metrics.get("per_class_3")
        if isinstance(pc, dict):
            vals: list[float] = []
            for k in ("-1", "1"):
                row = pc.get(k) if isinstance(pc.get(k), dict) else None
                if row is None:
                    continue
                fv = float(row.get("f1", np.nan))
                if np.isfinite(fv):
                    vals.append(float(fv))
            if len(vals) == 2:
                return float(np.mean(vals))
    except Exception:
        pass
    try:
        fb = float(metrics.get("f1_binary", np.nan))
        if np.isfinite(fb):
            return float(fb)
    except Exception:
        pass
    return float("nan")


def _prediction_side_balance_summary(y_pred: np.ndarray | None) -> dict[str, float | int]:
    arr = np.asarray(y_pred if y_pred is not None else [], dtype=int).reshape(-1)
    if arr.size <= 0:
        return {
            "n_short": 0,
            "n_long": 0,
            "n_hold": 0,
            "n_directional": 0,
            "short_share": float("nan"),
            "long_share": float("nan"),
        }

    n_short = int(np.sum(arr < 0))
    n_long = int(np.sum(arr > 0))
    n_hold = int(np.sum(arr == 0))
    n_directional = int(n_short + n_long)
    short_share = (float(n_short) / float(n_directional)) if n_directional > 0 else float("nan")
    long_share = (float(n_long) / float(n_directional)) if n_directional > 0 else float("nan")
    return {
        "n_short": int(n_short),
        "n_long": int(n_long),
        "n_hold": int(n_hold),
        "n_directional": int(n_directional),
        "short_share": float(short_share),
        "long_share": float(long_share),
    }


def _build_holdout_chunk_diagnostics(
    y_true: np.ndarray | None,
    y_pred: np.ndarray | None,
    df_hold: pd.DataFrame | None,
    *,
    fee_per_trade: float,
    slippage_bps: float,
    annualize_sharpe: bool,
    max_chunks: int = 3,
) -> list[dict[str, Any]]:
    arr_true = np.asarray(y_true if y_true is not None else [], dtype=int).reshape(-1)
    arr_pred = np.asarray(y_pred if y_pred is not None else [], dtype=int).reshape(-1)
    if df_hold is None or len(df_hold) <= 0 or arr_true.size <= 0 or arr_pred.size <= 0:
        return []

    n_rows = int(min(len(df_hold), arr_true.size, arr_pred.size))
    if n_rows <= 0:
        return []

    frame = df_hold.iloc[:n_rows].reset_index(drop=True).copy()
    arr_true = arr_true[:n_rows]
    arr_pred = arr_pred[:n_rows]
    n_chunks = int(max(1, min(int(max_chunks), n_rows)))
    row_slices = np.array_split(np.arange(n_rows, dtype=int), n_chunks)
    ts_col = next((col for col in ("timestamp", "time", "date") if col in frame.columns), None)

    def _ts_value(idx: int) -> str | None:
        if ts_col is None:
            return None
        try:
            ts = pd.to_datetime(frame.iloc[int(idx)][ts_col], utc=True, errors="coerce")
            return None if pd.isna(ts) else str(ts.isoformat())
        except Exception:
            return None

    def _safe_float(value: Any) -> float | None:
        try:
            fv = float(value)
            return float(fv) if np.isfinite(fv) else None
        except Exception:
            return None

    chunks: list[dict[str, Any]] = []
    for idx, rows in enumerate(row_slices, start=1):
        rows_arr = np.asarray(rows, dtype=int)
        if rows_arr.size <= 0:
            continue
        start = int(rows_arr[0])
        stop = int(rows_arr[-1]) + 1
        y_true_chunk = arr_true[start:stop]
        y_pred_chunk = arr_pred[start:stop]
        df_chunk = frame.iloc[start:stop].reset_index(drop=True)
        metrics_chunk: dict[str, Any] = {}
        if HAS_CALC_METRICS and len(df_chunk) > 0:
            try:
                metrics_chunk = calculate_metrics(
                    y_true=y_true_chunk,
                    y_pred=y_pred_chunk,
                    df=df_chunk,
                    fee_per_trade=fee_per_trade,
                    slippage_bps=slippage_bps,
                    annualize_sharpe=annualize_sharpe,
                )
            except Exception:
                metrics_chunk = {}
        if not metrics_chunk:
            metrics_chunk = {
                "accuracy": float((y_true_chunk == y_pred_chunk).mean()) if y_true_chunk.size > 0 else None,
                "num_trades": int(np.sum(y_pred_chunk != 0)),
                "num_trades_short": int(np.sum(y_pred_chunk < 0)),
                "num_trades_long": int(np.sum(y_pred_chunk > 0)),
            }
        chunks.append(
            {
                "chunk_index": int(idx),
                "n_rows": int(len(df_chunk)),
                "start_row": int(start),
                "end_row": int(stop - 1),
                "start_timestamp": _ts_value(start),
                "end_timestamp": _ts_value(stop - 1),
                "prediction_balance": _prediction_side_balance_summary(y_pred_chunk),
                "profit_net": _safe_float(metrics_chunk.get("profit_net")),
                "sharpe": _safe_float(metrics_chunk.get("sharpe", metrics_chunk.get("sharpe_ann"))),
                "accuracy": _safe_float(metrics_chunk.get("accuracy")),
                "num_trades": int(metrics_chunk.get("num_trades", 0) or 0),
                "num_trades_short": int(metrics_chunk.get("num_trades_short", 0) or 0),
                "num_trades_long": int(metrics_chunk.get("num_trades_long", 0) or 0),
                "directional_f1": _safe_float(_directional_f1_from_metrics(metrics_chunk)),
            }
        )
    return chunks


def _quality_gate_vs_baseline_ternary(
    holdout_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    min_f1_lift: float = 0.01,
    min_trades: int = 8,
    y_pred: np.ndarray | None = None,
    min_side_prediction_share: float = 0.0,
    min_side_prediction_count: int = 0,
) -> tuple[bool, list[str]]:
    """Gate model quality against trivial all-HOLD baseline on holdout."""
    reasons: list[str] = []
    dir_f1 = _directional_f1_from_metrics(holdout_metrics)
    dir_f1_base = _directional_f1_from_metrics(baseline_metrics)
    f1_macro = float(holdout_metrics.get("f1_macro_3", holdout_metrics.get("f1", np.nan)))
    profit = float(holdout_metrics.get("profit_net", np.nan))
    profit_base = float(baseline_metrics.get("profit_net", 0.0))
    n_trades = int(holdout_metrics.get("num_trades", 0) or 0)
    n_short = int(holdout_metrics.get("num_trades_short", 0) or 0)
    n_long = int(holdout_metrics.get("num_trades_long", 0) or 0)

    if np.isfinite(dir_f1) and np.isfinite(dir_f1_base):
        if dir_f1 < (dir_f1_base + float(min_f1_lift)):
            reasons.append(
                f"directional_f1_not_better_than_baseline({dir_f1:.4f}<{dir_f1_base + float(min_f1_lift):.4f})"
            )
    elif np.isfinite(f1_macro) and f1_macro < 0.10:
        reasons.append(f"f1_macro_3_too_low({f1_macro:.4f}<0.1000)")
    if np.isfinite(profit) and np.isfinite(profit_base) and profit < profit_base:
        reasons.append(f"profit_net_below_baseline({profit:.2f}<{profit_base:.2f})")
    if n_trades < int(min_trades):
        reasons.append(f"too_few_trades({n_trades}<{int(min_trades)})")
    if n_short == 0:
        reasons.append("no_short_trades")
    if n_long == 0:
        reasons.append("no_long_trades")

    pred_balance = _prediction_side_balance_summary(y_pred)
    pred_short = int(pred_balance.get("n_short", 0) or 0)
    pred_long = int(pred_balance.get("n_long", 0) or 0)
    pred_directional = int(pred_balance.get("n_directional", 0) or 0)
    pred_short_share = float(pred_balance.get("short_share", np.nan))
    pred_long_share = float(pred_balance.get("long_share", np.nan))
    min_pred_share = float(max(0.0, min_side_prediction_share))
    min_pred_count = int(max(0, min_side_prediction_count))

    if pred_directional <= 0 and (min_pred_share > 0.0 or min_pred_count > 0):
        reasons.append("no_directional_predictions")
    else:
        if pred_short == 0:
            reasons.append("no_short_predictions")
        if pred_long == 0:
            reasons.append("no_long_predictions")
        if min_pred_count > 0:
            if pred_short < min_pred_count:
                reasons.append(f"short_predictions_too_few({pred_short}<{min_pred_count})")
            if pred_long < min_pred_count:
                reasons.append(f"long_predictions_too_few({pred_long}<{min_pred_count})")
        if min_pred_share > 0.0:
            if np.isfinite(pred_short_share) and pred_short_share < min_pred_share:
                reasons.append(f"short_prediction_share_too_low({pred_short_share:.4f}<{min_pred_share:.4f})")
            if np.isfinite(pred_long_share) and pred_long_share < min_pred_share:
                reasons.append(f"long_prediction_share_too_low({pred_long_share:.4f}<{min_pred_share:.4f})")
    return (len(reasons) == 0), reasons

def _choose_threshold_from_oof(y_true: np.ndarray, oof_proba: np.ndarray, df_oof: pd.DataFrame | None,
                               fee_per_trade: float, slippage_bps: float) -> float:
    if (not HAS_CALC_METRICS) or (oof_proba is None) or (df_oof is None):
        return 0.5
    best_thr, best_score = 0.5, -1e18
    for thr in np.linspace(0.3, 0.7, 41):
        y_pred = (oof_proba >= thr).astype(int)
        try:
            m = calculate_metrics(
                y_true=y_true, y_pred=y_pred, df=df_oof,
                fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                annualize_sharpe=True
            )
            score = float(m.get("profit_net", np.nan))
        except Exception:
            score = np.nan
        if not np.isnan(score) and score > best_score:
            best_score, best_thr = score, float(thr)
    return float(best_thr)


def _ternary_composite_score(
    m: dict[str, Any],
    *,
    n_short: int,
    n_long: int,
) -> float:
    """Score tuned for ternary trading: quality + profitability + directional balance."""
    f1m = float(m.get("f1_macro_3", np.nan))
    if not np.isfinite(f1m):
        f1m = float(m.get("f1", 0.0))
    dir_f1 = _directional_f1_from_metrics(m)
    if not np.isfinite(dir_f1):
        dir_f1 = f1m
    profit = float(m.get("profit_net", np.nan))
    if not np.isfinite(profit):
        profit = float(m.get("profit_gross", 0.0))
    sharpe = float(m.get("sharpe_ann", np.nan))
    if not np.isfinite(sharpe):
        sharpe = float(m.get("sharpe", np.nan))
    if not np.isfinite(sharpe):
        sharpe = 0.0
    avg_trade = float(m.get("avg_pnl_trade", np.nan))
    if not np.isfinite(avg_trade):
        avg_trade = float(m.get("mean_pnl_trade", np.nan))
    if not np.isfinite(avg_trade):
        avg_trade = 0.0
    pf = float(m.get("pf", np.nan))
    if not np.isfinite(pf):
        pf = float(m.get("profit_factor", np.nan))

    # Keep components bounded so one noisy metric cannot dominate.
    profit_component = float(np.clip(profit / 250.0, -2.0, 2.0))
    sharpe_component = float(np.clip(sharpe / 2.0, -1.5, 1.5))
    avg_trade_component = float(np.clip(avg_trade / 20.0, -1.0, 1.0))

    n_dir = int(max(0, n_short) + max(0, n_long))
    if n_dir <= 0:
        balance = 0.0
    else:
        balance = float(min(max(0, n_short), max(0, n_long)) / max(1, n_dir))

    # Economic quality has high weight to better align CV selection with holdout gate.
    score = (
        (0.28 * dir_f1)
        + (0.12 * f1m)
        + (0.34 * profit_component)
        + (0.16 * sharpe_component)
        + (0.05 * avg_trade_component)
        + (0.05 * balance)
    )

    if profit < 0.0:
        score -= 0.20 + min(0.50, abs(profit) / 500.0)
    if sharpe < 0.0:
        score -= 0.10 + min(0.30, abs(sharpe) / 10.0)
    if avg_trade < 0.0:
        score -= 0.05 + min(0.20, abs(avg_trade) / 20.0)
    if np.isfinite(pf) and pf < 1.0:
        score -= min(0.20, 0.20 * (1.0 - max(0.0, pf)))

    try:
        pc = m.get("per_class_3")
        if isinstance(pc, dict):
            rec_short = float((pc.get("-1") or {}).get("recall", np.nan))
            rec_long = float((pc.get("1") or {}).get("recall", np.nan))
            if np.isfinite(rec_short) and rec_short < 0.01:
                score -= 0.25
            elif np.isfinite(rec_short) and rec_short < 0.03:
                score -= 0.10
            if np.isfinite(rec_long) and rec_long < 0.01:
                score -= 0.25
            elif np.isfinite(rec_long) and rec_long < 0.03:
                score -= 0.10
    except Exception:
        pass
    if n_short == 0 or n_long == 0:
        score -= 0.35
    if n_dir < 12:
        score -= 0.20
    if n_short < 3 or n_long < 3:
        score -= 0.20
    if n_dir > 0:
        dominance = float(max(max(0, n_short), max(0, n_long)) / n_dir)
        if dominance > 0.90:
            score -= 0.20 + (dominance - 0.90) * 0.50
    return float(score)


def _choose_thresholds_from_oof_ternary(
    y_true_mapped: np.ndarray,
    oof_short: np.ndarray,
    oof_long: np.ndarray,
    df_oof: pd.DataFrame | None,
    fee_per_trade: float,
    slippage_bps: float,
    min_signals: int = 20,
    min_side_signals: int = 5,
    max_side_dominance: float = 0.95,
    target_short_share: float | None = None,
    target_dir_rate: float | None = None,
    short_share_tolerance: float = 0.25,
    dir_rate_tolerance: float = 0.10,
    balance_penalty_weight: float = 0.35,
    min_side_recall_target: float = 0.01,
    shortlist_max_candidates: int = 140,
) -> tuple[float, float]:
    """
    Tune separate SHORT/LONG thresholds for mapped ternary labels (0/1/2).
    Two-stage optimization:
      1) adaptive candidate preselection from quantiles + cheap constraints/recall score
      2) expensive metric evaluation only on shortlist
    """
    if (not HAS_CALC_METRICS) or df_oof is None:
        ts_fb, tl_fb = _fallback_ternary_thresholds_from_oof(
            y_true_mapped=y_true_mapped,
            oof_short=oof_short,
            oof_long=oof_long,
            min_side_signals=int(max(4, min_side_signals)),
            min_total_signals=int(max(8, min_signals)),
        )
        return float(ts_fb), float(tl_fb)
    best = (0.5, 0.5)
    best_score = -1e18
    best_balanced = (0.5, 0.5)
    best_balanced_score = -1e18
    best_relaxed = (0.5, 0.5)
    best_relaxed_score = -1e18
    cheap_recall_reward = 1.35
    cheap_recall_penalty = 1.50
    shortlist_recall_penalty = 0.95
    balanced_recall_penalty = 0.60
    min_side_floor = max(4, min(int(min_side_signals), 30))
    max_side_dominance_relaxed = min(0.90, float(max_side_dominance) + 0.10)
    short_share_tolerance_relaxed = float(short_share_tolerance) + 0.08
    dir_rate_tolerance_relaxed = float(dir_rate_tolerance) + 0.08
    target_dir_rate = (
        float(target_dir_rate) if target_dir_rate is not None and np.isfinite(target_dir_rate) else None
    )
    y_true_eval = _mapped_ternary_to_signed(y_true_mapped)
    n_obs = max(1, len(y_true_mapped))

    def _safe_quantile(values: np.ndarray, q: float, lo: float, hi: float) -> float:
        try:
            vv = np.asarray(values, dtype=float)
            vv = vv[np.isfinite(vv)]
            if vv.size <= 0:
                return float(np.clip(0.5, lo, hi))
            qv = float(np.quantile(vv, float(np.clip(q, 0.0, 1.0))))
            return float(np.clip(qv, lo, hi))
        except Exception:
            return float(np.clip(0.5, lo, hi))

    # Build adaptive candidate grids around expected directional activation rates.
    obs_dir_rate = float(np.mean(np.asarray(y_true_mapped) != 1))
    dir_target_used = (
        float(np.clip(target_dir_rate, 0.03, 0.45))
        if target_dir_rate is not None and np.isfinite(target_dir_rate)
        else float(np.clip(obs_dir_rate, 0.03, 0.45))
    )
    short_share_used = (
        float(np.clip(target_short_share, 0.20, 0.80))
        if target_short_share is not None and np.isfinite(target_short_share)
        else 0.50
    )
    p_short_target = float(np.clip(dir_target_used * short_share_used, 0.01, 0.30))
    p_long_target = float(np.clip(dir_target_used * (1.0 - short_share_used), 0.01, 0.30))
    q_short_center = float(np.clip(1.0 - p_short_target, 0.60, 0.995))
    q_long_center = float(np.clip(1.0 - p_long_target, 0.60, 0.995))
    q_offsets = np.asarray([-0.22, -0.16, -0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10, 0.16, 0.22], dtype=float)
    q_short = np.clip(q_short_center + q_offsets, 0.35, 0.995)
    q_long = np.clip(q_long_center + q_offsets, 0.35, 0.995)
    grid_short_vals = [_safe_quantile(oof_short, float(q), 0.03, 0.92) for q in q_short]
    grid_long_vals = [_safe_quantile(oof_long, float(q), 0.03, 0.92) for q in q_long]
    # Add a few anchors for robustness.
    grid_short_vals.extend([0.05, 0.08, 0.12, 0.18, 0.24, 0.32, 0.44, 0.56, 0.68, 0.78, 0.88])
    grid_long_vals.extend([0.05, 0.08, 0.12, 0.18, 0.24, 0.32, 0.44, 0.56, 0.68, 0.78, 0.88])
    grid_short = np.asarray(sorted({float(np.clip(v, 0.03, 0.92)) for v in grid_short_vals}), dtype=float)
    grid_long = np.asarray(sorted({float(np.clip(v, 0.03, 0.92)) for v in grid_long_vals}), dtype=float)

    # Stage 1: cheap constrained preselection.
    candidate_rows: list[dict[str, Any]] = []
    recall_floor_relaxed = float(max(0.001, min_side_recall_target * 0.50))
    for thr_s in grid_short:
        for thr_l in grid_long:
            y_pred_mapped = _ternary_predict_mapped(oof_short, oof_long, float(thr_s), float(thr_l))
            n_short = int((y_pred_mapped == 0).sum())
            n_long = int((y_pred_mapped == 2).sum())
            n_total = n_short + n_long
            if n_total < int(max(6, min_signals // 2)):
                continue
            dir_rate = float(n_total / max(1, n_obs))
            dominance = float(max(n_short, n_long) / max(1, n_total))
            short_share = float(n_short / max(1, n_total))
            share_dev = (
                abs(short_share - float(target_short_share))
                if target_short_share is not None and np.isfinite(target_short_share)
                else 0.0
            )
            dir_dev = (
                abs(dir_rate - float(target_dir_rate))
                if target_dir_rate is not None and np.isfinite(target_dir_rate)
                else 0.0
            )
            mask_short = (np.asarray(y_true_mapped, dtype=int) == 0)
            mask_long = (np.asarray(y_true_mapped, dtype=int) == 2)
            rec_short = float(np.mean(y_pred_mapped[mask_short] == 0)) if np.any(mask_short) else float("nan")
            rec_long = float(np.mean(y_pred_mapped[mask_long] == 2)) if np.any(mask_long) else float("nan")

            cheap = 0.0
            if np.isfinite(rec_short):
                cheap += cheap_recall_reward * float(rec_short)
                cheap -= cheap_recall_penalty * max(0.0, float(min_side_recall_target) - float(rec_short))
            if np.isfinite(rec_long):
                cheap += cheap_recall_reward * float(rec_long)
                cheap -= cheap_recall_penalty * max(0.0, float(min_side_recall_target) - float(rec_long))
            cheap += 0.15 * min(2.0, float(n_total) / max(1.0, float(min_signals)))
            cheap -= 1.8 * max(0.0, float(dominance) - float(max_side_dominance_relaxed))
            if share_dev > 0.0:
                cheap -= float(balance_penalty_weight) * float(share_dev)
            if dir_dev > 0.0:
                cheap -= float(balance_penalty_weight) * 0.75 * float(dir_dev)
            if n_short < max(2, int(min_side_floor // 2)):
                cheap -= 0.6
            if n_long < max(2, int(min_side_floor // 2)):
                cheap -= 0.3
            if not np.isfinite(rec_short) or rec_short < recall_floor_relaxed:
                cheap -= 0.35

            candidate_rows.append(
                {
                    "thr_s": float(thr_s),
                    "thr_l": float(thr_l),
                    "y_pred": y_pred_mapped,
                    "n_short": n_short,
                    "n_long": n_long,
                    "n_total": n_total,
                    "dominance": float(dominance),
                    "share_dev": float(share_dev),
                    "dir_dev": float(dir_dev),
                    "rec_short": float(rec_short) if np.isfinite(rec_short) else float("nan"),
                    "rec_long": float(rec_long) if np.isfinite(rec_long) else float("nan"),
                    "cheap_score": float(cheap),
                }
            )

    if not candidate_rows:
        ts_fb, tl_fb = _fallback_ternary_thresholds_from_oof(
            y_true_mapped=y_true_mapped,
            oof_short=oof_short,
            oof_long=oof_long,
            min_side_signals=int(max(4, min_side_signals)),
            min_total_signals=int(max(8, min_signals)),
        )
        return float(ts_fb), float(tl_fb)

    candidate_rows.sort(key=lambda r: float(r.get("cheap_score", -1e18)), reverse=True)
    strict_rows = [
        r
        for r in candidate_rows
        if (
            int(r.get("n_total", 0)) >= int(min_signals)
            and np.isfinite(float(r.get("rec_short", np.nan)))
            and float(r.get("rec_short", np.nan)) >= float(min_side_recall_target)
            and np.isfinite(float(r.get("rec_long", np.nan)))
            and float(r.get("rec_long", np.nan)) >= float(min_side_recall_target)
        )
    ]
    hard_recall_pool_used = bool(len(strict_rows) > 0)
    pool_rows = strict_rows if hard_recall_pool_used else candidate_rows
    top_n = int(max(40, min(int(shortlist_max_candidates), len(pool_rows))))
    shortlist = pool_rows[:top_n]

    # Stage 2: expensive evaluation on shortlist only.
    for row in shortlist:
        thr_s = float(row["thr_s"])
        thr_l = float(row["thr_l"])
        y_pred_mapped = np.asarray(row["y_pred"], dtype=int)
        n_short = int(row["n_short"])
        n_long = int(row["n_long"])
        n_total = int(row["n_total"])
        if n_total < int(min_signals):
            continue
        dominance = float(row["dominance"])
        share_dev = float(row["share_dev"])
        dir_dev = float(row["dir_dev"])
        rec_short = float(row["rec_short"]) if np.isfinite(float(row["rec_short"])) else float("nan")
        rec_long = float(row["rec_long"]) if np.isfinite(float(row["rec_long"])) else float("nan")

        y_pred_eval = _mapped_ternary_to_signed(y_pred_mapped)
        try:
            m = calculate_metrics(
                y_true=y_true_eval,
                y_pred=y_pred_eval,
                df=df_oof,
                fee_per_trade=fee_per_trade,
                slippage_bps=slippage_bps,
                annualize_sharpe=True,
            )
            score = _ternary_composite_score(m, n_short=n_short, n_long=n_long)
            if share_dev > 0.0:
                score -= float(balance_penalty_weight) * float(share_dev)
            if dir_dev > 0.0:
                score -= float(balance_penalty_weight) * 0.75 * float(dir_dev)
            if np.isfinite(rec_short):
                score -= shortlist_recall_penalty * max(0.0, float(min_side_recall_target) - float(rec_short))
            if np.isfinite(rec_long):
                score -= shortlist_recall_penalty * max(0.0, float(min_side_recall_target) - float(rec_long))
        except Exception:
            score = -1e18

        if score > best_relaxed_score:
            best_relaxed_score = float(score)
            best_relaxed = (float(thr_s), float(thr_l))

        if n_short >= min_side_floor and n_long >= min_side_floor:
            relaxed_ok = dominance <= max_side_dominance_relaxed
            if target_short_share is not None and np.isfinite(target_short_share):
                relaxed_ok = relaxed_ok and (share_dev <= short_share_tolerance_relaxed)
            if target_dir_rate is not None and np.isfinite(target_dir_rate):
                relaxed_ok = relaxed_ok and (dir_dev <= dir_rate_tolerance_relaxed)
            if np.isfinite(rec_short):
                relaxed_ok = relaxed_ok and (rec_short >= recall_floor_relaxed)
            if np.isfinite(rec_long):
                relaxed_ok = relaxed_ok and (rec_long >= recall_floor_relaxed)
            if relaxed_ok:
                bal_score = float(score)
                bal_score -= max(0.0, dominance - float(max_side_dominance)) * 2.0
                if target_short_share is not None and np.isfinite(target_short_share):
                    bal_score -= max(0.0, share_dev - float(short_share_tolerance)) * 2.0
                if target_dir_rate is not None and np.isfinite(target_dir_rate):
                    bal_score -= max(0.0, dir_dev - float(dir_rate_tolerance)) * 2.0
                if np.isfinite(rec_short):
                    bal_score -= balanced_recall_penalty * max(0.0, float(min_side_recall_target) - float(rec_short))
                if np.isfinite(rec_long):
                    bal_score -= balanced_recall_penalty * max(0.0, float(min_side_recall_target) - float(rec_long))
                if bal_score > best_balanced_score:
                    best_balanced_score = float(bal_score)
                    best_balanced = (float(thr_s), float(thr_l))

        if (
            n_short >= int(min_side_signals)
            and n_long >= int(min_side_signals)
            and dominance <= float(max_side_dominance)
            and (share_dev <= float(short_share_tolerance))
            and (dir_dev <= float(dir_rate_tolerance))
            and (not np.isfinite(rec_short) or rec_short >= float(min_side_recall_target))
            and (not np.isfinite(rec_long) or rec_long >= float(min_side_recall_target))
            and score > best_score
        ):
            best_score = float(score)
            best = (float(thr_s), float(thr_l))
    if best_score > -1e17:
        return best
    if hard_recall_pool_used:
        if best_balanced_score > -1e17:
            return best_balanced
        if best_relaxed_score > -1e17:
            return best_relaxed
        if strict_rows:
            return float(strict_rows[0]["thr_s"]), float(strict_rows[0]["thr_l"])
        ts_fb, tl_fb = _fallback_ternary_thresholds_from_oof(
            y_true_mapped=y_true_mapped,
            oof_short=oof_short,
            oof_long=oof_long,
            min_side_signals=int(max(4, min_side_signals)),
            min_total_signals=int(max(8, min_signals)),
        )
        return float(ts_fb), float(tl_fb)
    if best_balanced_score > -1e17:
        return best_balanced
    if best_relaxed_score > -1e17:
        return best_relaxed
    # Last fallback (when metric eval fails): prefer the most balanced side-recall candidate.
    try:
        side_row = max(candidate_rows, key=_fallback_candidate_sort_key)
        return float(side_row["thr_s"]), float(side_row["thr_l"])
    except Exception:
        ts_fb, tl_fb = _fallback_ternary_thresholds_from_oof(
            y_true_mapped=y_true_mapped,
            oof_short=oof_short,
            oof_long=oof_long,
            min_side_signals=int(max(4, min_side_signals)),
            min_total_signals=int(max(8, min_signals)),
        )
        return float(ts_fb), float(tl_fb)


def _fallback_ternary_thresholds_from_oof(
    y_true_mapped: np.ndarray,
    oof_short: np.ndarray,
    oof_long: np.ndarray,
    *,
    min_side_signals: int = 4,
    min_total_signals: int = 12,
) -> tuple[float, float]:
    """Data-driven fallback for ternary thresholds when score optimization is inconclusive."""
    try:
        y_arr = np.asarray(y_true_mapped, dtype=int)
        ps = np.asarray(oof_short, dtype=float)
        pl = np.asarray(oof_long, dtype=float)
        valid = np.isfinite(ps) & np.isfinite(pl)
        if not np.any(valid):
            return 0.5, 0.5
        y_arr = y_arr[valid]
        ps = ps[valid]
        pl = pl[valid]
        if y_arr.size <= 0:
            return 0.5, 0.5

        y_dir = y_arr[y_arr != 1]
        target_short_share = float((y_dir == 0).mean()) if y_dir.size > 0 else 0.5
        target_short_share = float(np.clip(target_short_share, 0.30, 0.70))
        target_dir_rate = float(np.mean(y_arr != 1))
        floor_rate = float(max(0.05, float(min_total_signals) / max(1.0, float(y_arr.size))))
        target_dir_rate = float(np.clip(target_dir_rate, floor_rate, 0.45))

        p_short = float(np.clip(target_dir_rate * target_short_share, 0.01, 0.30))
        p_long = float(np.clip(target_dir_rate * (1.0 - target_short_share), 0.01, 0.30))
        ts = float(np.clip(np.quantile(ps, 1.0 - p_short), 0.03, 0.92))
        tl = float(np.clip(np.quantile(pl, 1.0 - p_long), 0.03, 0.92))

        # Soften until we get minimally useful directional coverage.
        min_side = int(max(2, min_side_signals))
        min_total = int(max(2 * min_side, min_total_signals))
        for _ in range(35):
            yp = _ternary_predict_mapped(ps, pl, float(ts), float(tl))
            n_short = int((yp == 0).sum())
            n_long = int((yp == 2).sum())
            n_dir = int(n_short + n_long)
            changed = False
            if n_short < min_side and ts > 0.03:
                ts = float(max(0.03, ts - 0.01))
                changed = True
            if n_long < min_side and tl > 0.03:
                tl = float(max(0.03, tl - 0.01))
                changed = True
            if n_dir < min_total:
                if ts > 0.03:
                    ts = float(max(0.03, ts - 0.006))
                    changed = True
                if tl > 0.03:
                    tl = float(max(0.03, tl - 0.006))
                    changed = True
            if (not changed) or (n_short >= min_side and n_long >= min_side and n_dir >= min_total):
                break
        return float(ts), float(tl)
    except Exception:
        return 0.5, 0.5


def _fallback_shared_threshold_from_oof(
    y_true_mapped: np.ndarray,
    oof_short: np.ndarray,
    oof_long: np.ndarray,
    *,
    min_signals: int = 20,
) -> float:
    """Data-driven fallback for shared threshold used by asymmetry/shared guards."""
    try:
        y_arr = np.asarray(y_true_mapped, dtype=int)
        ps = np.asarray(oof_short, dtype=float)
        pl = np.asarray(oof_long, dtype=float)
        valid = np.isfinite(ps) & np.isfinite(pl)
        if not np.any(valid):
            return 0.5
        y_arr = y_arr[valid]
        score = np.maximum(ps[valid], pl[valid])
        if score.size <= 0:
            return 0.5

        base_rate = float(np.mean(y_arr != 1))
        floor_rate = float(max(0.05, float(min_signals) / max(1.0, float(score.size))))
        p = float(np.clip(base_rate, floor_rate, 0.45))
        thr = float(np.clip(np.quantile(score, 1.0 - p), 0.03, 0.92))
        for _ in range(25):
            yp = _ternary_predict_mapped(ps[valid], pl[valid], float(thr), float(thr))
            n_dir = int((yp != 1).sum())
            if n_dir >= int(min_signals):
                break
            thr = float(max(0.03, thr - 0.01))
        return float(thr)
    except Exception:
        return 0.5


def _ternary_signal_stats_from_oof(
    oof_short: np.ndarray,
    oof_long: np.ndarray,
    thr_short: float,
    thr_long: float,
) -> dict[str, float]:
    y_pred = _ternary_predict_mapped(oof_short, oof_long, float(thr_short), float(thr_long))
    n_short = int((y_pred == 0).sum())
    n_long = int((y_pred == 2).sum())
    n_dir = int(n_short + n_long)
    n_total = int(y_pred.size)
    if n_dir <= 0:
        # No directional signals must not look "balanced" in downstream guards.
        dominance = 1.0
        short_share = 0.5
    else:
        dominance = float(max(n_short, n_long) / n_dir)
        short_share = float(n_short / n_dir)
    dir_rate = float(n_dir / max(1, n_total))
    return {
        "n_short": float(n_short),
        "n_long": float(n_long),
        "n_dir": float(n_dir),
        "n_total": float(n_total),
        "dominance": float(dominance),
        "short_share": float(short_share),
        "dir_rate": float(dir_rate),
    }


def _rebalance_ternary_thresholds_on_oof(
    oof_short: np.ndarray,
    oof_long: np.ndarray,
    thr_short: float,
    thr_long: float,
    *,
    min_side_signals: int,
    max_side_dominance: float,
    target_short_share: float | None = None,
    target_dir_rate: float | None = None,
    dir_rate_tolerance: float = 0.10,
    min_threshold: float = 0.03,
    max_iters: int = 30,
) -> tuple[float, float, dict[str, float]]:
    """Iteratively rebalance thresholds to avoid one-sided or overactive directional signaling."""
    ts = float(np.clip(thr_short, min_threshold, 0.98))
    tl = float(np.clip(thr_long, min_threshold, 0.98))
    tgt_short = (
        float(target_short_share)
        if target_short_share is not None and np.isfinite(target_short_share)
        else None
    )
    tgt_dir = (
        float(target_dir_rate)
        if target_dir_rate is not None and np.isfinite(target_dir_rate)
        else None
    )
    tol_dir = float(max(0.02, dir_rate_tolerance))
    floor_side = int(max(3, min_side_signals))
    stats = _ternary_signal_stats_from_oof(oof_short, oof_long, ts, tl)

    for _ in range(int(max_iters)):
        n_short = int(stats.get("n_short", 0.0))
        n_long = int(stats.get("n_long", 0.0))
        n_dir = int(stats.get("n_dir", 0.0))
        dominance = float(stats.get("dominance", 1.0))
        short_share = float(stats.get("short_share", 0.5))
        dir_rate = float(stats.get("dir_rate", 0.0))

        changed = False
        if tgt_dir is not None:
            upper = min(0.95, float(tgt_dir + tol_dir))
            lower = max(0.01, float(tgt_dir - tol_dir))
            if dir_rate > upper:
                ts += 0.02
                tl += 0.02
                changed = True
            elif dir_rate < lower and n_dir > 0:
                ts -= 0.01
                tl -= 0.01
                changed = True

        if n_dir > 0 and dominance > float(max_side_dominance):
            if n_short >= n_long:
                ts += 0.02
                tl -= 0.01
            else:
                tl += 0.02
                ts -= 0.01
            changed = True

        if tgt_short is not None and n_dir > 0:
            delta = float(short_share - tgt_short)
            if delta > 0.04:
                ts += 0.02
                tl -= 0.01
                changed = True
            elif delta < -0.04:
                tl += 0.02
                ts -= 0.01
                changed = True

        if n_short < floor_side:
            ts -= 0.01
            changed = True
        if n_long < floor_side:
            tl -= 0.01
            changed = True

        ts = float(np.clip(ts, min_threshold, 0.98))
        tl = float(np.clip(tl, min_threshold, 0.98))
        if not changed:
            break
        stats = _ternary_signal_stats_from_oof(oof_short, oof_long, ts, tl)

    return ts, tl, stats


def _finalize_threshold_tuning_outcome(
    threshold_tuning: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(threshold_tuning, dict) or not threshold_tuning:
        return threshold_tuning if isinstance(threshold_tuning, dict) else {}

    primary_mode = str(
        threshold_tuning.get("selected_mode_base")
        or threshold_tuning.get("selected_mode")
        or ""
    ).strip()
    final_mode = str(threshold_tuning.get("selected_mode") or primary_mode or "").strip()
    adjustments_applied: list[str] = []

    rebalance_adjustment = threshold_tuning.get("rebalance_adjustment")
    if isinstance(rebalance_adjustment, dict):
        if bool(rebalance_adjustment.get("reverted")):
            adjustments_applied.append("rebalance_reverted")
        else:
            before_thresholds = rebalance_adjustment.get("before_thresholds")
            after_thresholds = rebalance_adjustment.get("after_thresholds")
            if isinstance(before_thresholds, dict) and isinstance(after_thresholds, dict):
                try:
                    before_short = float(before_thresholds.get("short"))
                    before_long = float(before_thresholds.get("long"))
                    after_short = float(after_thresholds.get("short"))
                    after_long = float(after_thresholds.get("long"))
                    if (
                        abs(before_short - after_short) > 1e-12
                        or abs(before_long - after_long) > 1e-12
                    ):
                        adjustments_applied.append("rebalance")
                except Exception:
                    pass

    if "threshold_cap" in threshold_tuning:
        adjustments_applied.append("threshold_cap")

    threshold_gap_guard = threshold_tuning.get("threshold_gap_guard")
    if isinstance(threshold_gap_guard, dict) and bool(threshold_gap_guard.get("accepted")):
        adjustments_applied.append("threshold_gap_guard")

    threshold_asymmetry_guard = threshold_tuning.get("threshold_asymmetry_guard")
    if isinstance(threshold_asymmetry_guard, dict) and bool(threshold_asymmetry_guard.get("accepted")):
        adjustments_applied.append("threshold_asymmetry_guard")

    short_recall_enforce = threshold_tuning.get("short_recall_enforce")
    if isinstance(short_recall_enforce, dict) and bool(short_recall_enforce.get("applied")):
        adjustments_applied.append("short_recall_enforce")

    if isinstance(threshold_tuning.get("data_driven_fallback"), dict):
        adjustments_applied.append("data_driven_fallback")

    if final_mode == "shared_threshold_fallback":
        adjustments_applied.append("shared_threshold_fallback")
    elif final_mode == "shared_threshold_asymmetry_guard":
        adjustments_applied.append("shared_threshold_asymmetry_guard")
    elif final_mode == "revert_pre_guard_balanced":
        adjustments_applied.append("revert_pre_guard_balanced")

    fallback_reason = threshold_tuning.get("fallback_reason")
    if fallback_reason is None:
        if final_mode in {
            "shared_threshold_fallback",
            "shared_threshold_asymmetry_guard",
            "data_driven_fallback",
            "short_recall_enforce",
            "revert_pre_guard_balanced",
        }:
            fallback_reason = final_mode
        elif final_mode.startswith("rebalance_reverted("):
            fallback_reason = (
                rebalance_adjustment.get("revert_reason")
                if isinstance(rebalance_adjustment, dict)
                else None
            ) or "rebalance_reverted"

    deduped_adjustments: list[str] = []
    seen_adjustments: set[str] = set()
    for adjustment in adjustments_applied:
        adj = str(adjustment or "").strip()
        if adj and adj not in seen_adjustments:
            seen_adjustments.add(adj)
            deduped_adjustments.append(adj)

    threshold_tuning["primary_search_mode"] = primary_mode or None
    threshold_tuning["final_selected_mode"] = final_mode or None
    threshold_tuning["fallback_reason"] = (
        str(fallback_reason) if fallback_reason is not None else None
    )
    threshold_tuning["adjustments_applied"] = deduped_adjustments
    return threshold_tuning


def _ternary_threshold_guardrail_config(estimator_name: str | None) -> dict[str, Any]:
    est_name_lc = str(estimator_name or "").lower()
    is_lgb_family = est_name_lc in {"lgb", "lightgbm"}
    config = {
        "estimator": est_name_lc,
        "is_lgb_family": bool(is_lgb_family),
        "short_bounds": [0.03, 0.75],
        "long_bounds": [0.03, 0.75],
        "max_gap": 0.45,
        "max_dir_rate": 0.85,
    }
    if is_lgb_family:
        config.update(
            {
                "short_bounds": [0.05, 0.68],
                "long_bounds": [0.05, 0.72],
                "max_gap": 0.30,
            }
        )
    return config


def _side_recalls_for_ternary_thresholds(
    y_true_mapped: np.ndarray,
    oof_short: np.ndarray,
    oof_long: np.ndarray,
    thr_short: float,
    thr_long: float,
) -> tuple[float, float]:
    y_arr = np.asarray(y_true_mapped, dtype=int)
    yp = _ternary_predict_mapped(oof_short, oof_long, float(thr_short), float(thr_long))
    mask_short = y_arr == 0
    mask_long = y_arr == 2
    rec_short = float(np.mean(yp[mask_short] == 0)) if np.any(mask_short) else float("nan")
    rec_long = float(np.mean(yp[mask_long] == 2)) if np.any(mask_long) else float("nan")
    return rec_short, rec_long


def _guardrail_reasons_for_ternary_thresholds(
    stats: dict[str, float],
    *,
    thr_short: float,
    thr_long: float,
    rec_short: float,
    rec_long: float,
    min_side_floor: int,
    max_side_dominance: float,
    max_gap: float,
    max_dir_rate: float,
    min_side_recall: float,
) -> list[str]:
    reasons: list[str] = []
    n_short = int(stats.get("n_short", 0.0))
    n_long = int(stats.get("n_long", 0.0))
    dominance = float(stats.get("dominance", 1.0))
    dir_rate = float(stats.get("dir_rate", 0.0))
    if n_short < int(min_side_floor):
        reasons.append("n_short_below_min_side_signals")
    if n_long < int(min_side_floor):
        reasons.append("n_long_below_min_side_signals")
    if dominance > max(0.90, float(max_side_dominance) + 0.12):
        reasons.append("dominance_above_limit")
    if dir_rate > float(max_dir_rate):
        reasons.append("dir_rate_above_limit")
    if abs(float(thr_short) - float(thr_long)) > float(max_gap):
        reasons.append("threshold_gap_above_limit")
    if np.isfinite(rec_short) and float(rec_short) < float(min_side_recall):
        reasons.append("short_recall_below_min")
    if np.isfinite(rec_long) and float(rec_long) < float(min_side_recall):
        reasons.append("long_recall_below_min")
    return reasons


def _apply_single_fallback_ternary_thresholds(
    *,
    y_true_mapped: np.ndarray,
    oof_short: np.ndarray,
    oof_long: np.ndarray,
    thr_short: float,
    thr_long: float,
    estimator_name: str | None,
    min_side_floor: int,
    max_side_dominance: float,
    min_side_recall: float,
) -> tuple[float, float, dict[str, Any]]:
    config = _ternary_threshold_guardrail_config(estimator_name)
    short_bounds = list(config.get("short_bounds") or [0.03, 0.75])
    long_bounds = list(config.get("long_bounds") or [0.03, 0.75])
    max_gap = float(config.get("max_gap", 0.45))
    max_dir_rate = float(config.get("max_dir_rate", 0.85))

    ts_primary = float(np.clip(float(thr_short), float(short_bounds[0]), float(short_bounds[1])))
    tl_primary = float(np.clip(float(thr_long), float(long_bounds[0]), float(long_bounds[1])))
    metadata: dict[str, Any] = {
        "threshold_guardrails": {
            **config,
            "min_side_floor": int(min_side_floor),
            "max_side_dominance": float(max_side_dominance),
            "min_side_recall": float(min_side_recall),
        }
    }
    if abs(ts_primary - float(thr_short)) > 1e-12 or abs(tl_primary - float(thr_long)) > 1e-12:
        metadata["threshold_cap"] = {
            "short_bounds": [float(short_bounds[0]), float(short_bounds[1])],
            "long_bounds": [float(long_bounds[0]), float(long_bounds[1])],
            "before": {"short": float(thr_short), "long": float(thr_long)},
            "after": {"short": float(ts_primary), "long": float(tl_primary)},
        }

    primary_stats = _ternary_signal_stats_from_oof(
        oof_short,
        oof_long,
        float(ts_primary),
        float(tl_primary),
    )
    rec_short_primary, rec_long_primary = _side_recalls_for_ternary_thresholds(
        y_true_mapped,
        oof_short,
        oof_long,
        float(ts_primary),
        float(tl_primary),
    )
    guardrail_reasons = _guardrail_reasons_for_ternary_thresholds(
        primary_stats,
        thr_short=float(ts_primary),
        thr_long=float(tl_primary),
        rec_short=float(rec_short_primary),
        rec_long=float(rec_long_primary),
        min_side_floor=int(min_side_floor),
        max_side_dominance=float(max_side_dominance),
        max_gap=float(max_gap),
        max_dir_rate=float(max_dir_rate),
        min_side_recall=float(min_side_recall),
    )
    metadata["guardrail_reasons"] = list(guardrail_reasons)

    if not guardrail_reasons:
        metadata["oof_selected_final"] = primary_stats
        metadata["oof_side_recall_final"] = {
            "short": float(rec_short_primary) if np.isfinite(rec_short_primary) else None,
            "long": float(rec_long_primary) if np.isfinite(rec_long_primary) else None,
            "min_required": float(min_side_recall),
        }
        return float(ts_primary), float(tl_primary), metadata

    min_dir_rescue = int(max(12, int(2 * max(1, int(min_side_floor)))))
    ts_fb, tl_fb = _fallback_ternary_thresholds_from_oof(
        y_true_mapped=y_true_mapped,
        oof_short=oof_short,
        oof_long=oof_long,
        min_side_signals=int(max(2, max(1, int(min_side_floor // 2)))),
        min_total_signals=int(min_dir_rescue),
    )
    ts_fb = float(np.clip(float(ts_fb), float(short_bounds[0]), float(short_bounds[1])))
    tl_fb = float(np.clip(float(tl_fb), float(long_bounds[0]), float(long_bounds[1])))
    fb_stats = _ternary_signal_stats_from_oof(
        oof_short,
        oof_long,
        float(ts_fb),
        float(tl_fb),
    )
    rec_short_fb, rec_long_fb = _side_recalls_for_ternary_thresholds(
        y_true_mapped,
        oof_short,
        oof_long,
        float(ts_fb),
        float(tl_fb),
    )
    metadata["selected_mode"] = "data_driven_fallback"
    metadata["fallback_reason"] = "primary_thresholds_failed_guardrails"
    metadata["data_driven_fallback"] = {
        "before": {"short": float(ts_primary), "long": float(tl_primary)},
        "after": {"short": float(ts_fb), "long": float(tl_fb)},
        "stats_before": primary_stats,
        "stats_after": fb_stats,
        "recall_before": {
            "short": float(rec_short_primary) if np.isfinite(rec_short_primary) else None,
            "long": float(rec_long_primary) if np.isfinite(rec_long_primary) else None,
        },
        "recall_after": {
            "short": float(rec_short_fb) if np.isfinite(rec_short_fb) else None,
            "long": float(rec_long_fb) if np.isfinite(rec_long_fb) else None,
        },
        "guardrail_reasons": list(guardrail_reasons),
        "min_dir_required": int(min_dir_rescue),
    }
    metadata["oof_selected_final"] = fb_stats
    metadata["oof_side_recall_final"] = {
        "short": float(rec_short_fb) if np.isfinite(rec_short_fb) else None,
        "long": float(rec_long_fb) if np.isfinite(rec_long_fb) else None,
        "min_required": float(min_side_recall),
    }
    return float(ts_fb), float(tl_fb), metadata


def _choose_threshold_from_oof_ternary(
    y_true_mapped: np.ndarray,
    oof_short: np.ndarray,
    oof_long: np.ndarray,
    df_oof: pd.DataFrame | None,
    fee_per_trade: float,
    slippage_bps: float,
    min_signals: int = 20,
) -> float:
    """Tune shared LONG/SHORT threshold for mapped ternary labels (0/1/2)."""
    if (not HAS_CALC_METRICS) or df_oof is None:
        return float(
            _fallback_shared_threshold_from_oof(
                y_true_mapped=y_true_mapped,
                oof_short=oof_short,
                oof_long=oof_long,
                min_signals=int(max(8, min_signals)),
            )
        )
    best_thr, best_score = 0.5, -1e18
    y_true_eval = _mapped_ternary_to_signed(y_true_mapped)
    for thr in np.linspace(0.3, 0.7, 41):
        y_pred_mapped = _ternary_predict_mapped(oof_short, oof_long, float(thr), float(thr))
        if int((y_pred_mapped != 1).sum()) < int(min_signals):
            continue
        y_pred_eval = _mapped_ternary_to_signed(y_pred_mapped)
        try:
            m = calculate_metrics(
                y_true=y_true_eval, y_pred=y_pred_eval, df=df_oof,
                fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                annualize_sharpe=True
            )
            score = float(m.get("profit_net", np.nan))
            if not np.isfinite(score):
                score = float(m.get("f1_macro_3", np.nan))
        except Exception:
            score = np.nan
        if np.isfinite(score) and score > best_score:
            best_score, best_thr = float(score), float(thr)
    if best_score <= -1e17:
        return float(
            _fallback_shared_threshold_from_oof(
                y_true_mapped=y_true_mapped,
                oof_short=oof_short,
                oof_long=oof_long,
                min_signals=int(max(8, min_signals)),
            )
        )
    return float(best_thr)

# -------------- Monte Carlo --------------
def _mc_block_bootstrap_indices(n: int, block_len: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    b = max(1, int(block_len))
    k = (n + b - 1) // b
    starts = np.random.randint(0, n, size=k)
    idx = []
    for s in starts:
        block = [(s + t) % n for t in range(b)]
        idx.extend(block)
        if len(idx) >= n:
            break
    return np.asarray(idx[:n], dtype=int)

def _mc_eval_holdout_adaptive(
    estimator,
    df_hold: pd.DataFrame,
    features: list[str],
    base_threshold: float = 0.5,
    ternary_threshold_short: float | None = None,
    ternary_threshold_long: float | None = None,
    iters: int = 200,
    block_len: int = 100,
    fee_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    min_trades: int = 20,
    trial_thresholds: tuple[float, ...] = (0.5, 0.48, 0.46, 0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26, 0.25),
) -> dict[str, Any]:
    if (not HAS_CALC_METRICS) or df_hold is None or len(df_hold) < 10:
        return {}

    try:
        Xh = df_hold[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_true = df_hold["target"].astype(int).to_numpy()
    except Exception:
        return {}

    proba_all = None
    proba_short_all = None
    proba_long_all = None
    is_ternary_proba = False
    try:
        Xh_use = _align_X_for_estimator(estimator, Xh)
        if hasattr(estimator, "predict_proba"):
            pr = _call_with_feature_name_warning_suppressed(estimator.predict_proba, Xh_use)
            if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 3:
                proba_short_all = pr[:, 0]
                proba_long_all = pr[:, 2]
                is_ternary_proba = True
            else:
                proba_all = pr[:, 1] if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 2 else np.asarray(pr).ravel()
        elif hasattr(estimator, "decision_function"):
            z = np.asarray(estimator.decision_function(Xh_use)).ravel()
            proba_all = 1.0 / (1.0 + np.exp(-z))
    except Exception:
        proba_all = None

    def _q(a, p):
        a = [x for x in a if x is not None and np.isfinite(x)]
        return float(np.nanpercentile(a, p)) if len(a) else None

    sharpe_vals, dd_vals, profit_vals = [], [], []
    valid_sharpes = []

    for _ in range(int(max(1, iters))):
        idx = _mc_block_bootstrap_indices(len(y_true), int(max(1, block_len)))
        yt = y_true[idx]
        dfb = df_hold.iloc[idx]

        if is_ternary_proba and proba_short_all is not None and proba_long_all is not None:
            ps = proba_short_all[idx]
            pl = proba_long_all[idx]
            thr_s_base = float(ternary_threshold_short) if ternary_threshold_short is not None else float(base_threshold)
            thr_l_base = float(ternary_threshold_long) if ternary_threshold_long is not None else float(base_threshold)
            thr_s_used = float(np.clip(thr_s_base, 0.03, 0.99))
            thr_l_used = float(np.clip(thr_l_base, 0.03, 0.99))
            yp = _ternary_predict_mapped(ps, pl, float(thr_s_used), float(thr_l_used))
            if int((yp != 1).sum()) < min_trades:
                used = False
                for scale in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
                    thr_s_used = float(np.clip(thr_s_base * float(scale), 0.03, 0.99))
                    thr_l_used = float(np.clip(thr_l_base * float(scale), 0.03, 0.99))
                    yp = _ternary_predict_mapped(ps, pl, float(thr_s_used), float(thr_l_used))
                    if int((yp != 1).sum()) >= min_trades:
                        used = True
                        break
                if not used:
                    thr_s_used = float(np.clip(thr_s_base * 0.65, 0.03, 0.99))
                    thr_l_used = float(np.clip(thr_l_base * 0.65, 0.03, 0.99))
                    yp = _ternary_predict_mapped(ps, pl, float(thr_s_used), float(thr_l_used))
        elif proba_all is not None:
            pb = proba_all[idx]
            thr_used = float(base_threshold)
            yp = (pb >= thr_used).astype(int)
            if yp.sum() < min_trades:
                used = False
                for thr in trial_thresholds:
                    thr_used = float(thr)
                    yp = (pb >= thr_used).astype(int)
                    if yp.sum() >= min_trades:
                        used = True
                        break
                if not used:
                    thr_used = float(trial_thresholds[-1])
                    yp = (pb >= thr_used).astype(int)
        else:
            X_pred = _align_X_for_estimator(estimator, dfb[features])
            yp = _call_with_feature_name_warning_suppressed(estimator.predict, X_pred)

        try:
            yt_eval = _mapped_ternary_to_signed(yt) if is_ternary_proba else yt
            yp_eval = _mapped_ternary_to_signed(yp) if is_ternary_proba else yp
            m = calculate_metrics(
                y_true=yt_eval, y_pred=yp_eval, df=dfb,
                fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                annualize_sharpe=True
            )
            sv = m.get("sharpe", np.nan)
            if sv is not None and np.isfinite(sv):
                valid_sharpes.append(float(sv))
            sharpe_vals.append(float(sv) if np.isfinite(sv) else np.nan)
            dd_vals.append(float(m.get("max_drawdown", np.nan)))
            profit_vals.append(float(m.get("profit_net", np.nan)))
        except Exception:
            sharpe_vals.append(np.nan)
            dd_vals.append(np.nan)
            profit_vals.append(np.nan)

    res = {
        "iters": int(iters),
        "block_len": int(block_len),
        "min_trades": int(min_trades),
        "thr_scan": list(trial_thresholds),
        "sharpe": {"p10": _q(sharpe_vals, 10) if len(valid_sharpes) else None,
                   "p50": _q(sharpe_vals, 50) if len(valid_sharpes) else None,
                   "p90": _q(sharpe_vals, 90) if len(valid_sharpes) else None},
        "max_drawdown": {"p90": _q(dd_vals, 90)},
        "profit_net": {"p10": _q(profit_vals, 10), "p50": _q(profit_vals, 50), "p90": _q(profit_vals, 90)},
    }
    if res["sharpe"]["p50"] is None:
        res["note"] = "no_trades_or_zero_variance"
    return res

# ------------------- Hlavní trénink -------------------
def train_and_evaluate_model(
    df: pd.DataFrame,
    estimator_name: str = "hgbt",
    param_grid: dict | None = None,
    n_splits: int = 5,
    embargo: int = 10,
    fee_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    calibrate: bool = False,
    on_progress=None,
    feature_stability_threshold: float | None = None,
    **kwargs,
) -> dict:
    """
    Trénink na posledním okně (bez window sweepu) + čistý holdout + (volitelně) MC evaluace holdoutu.
    """
    if "param_grid" in kwargs and kwargs["param_grid"] is not None:
        param_grid = kwargs.pop("param_grid")
    if param_grid is None:
        param_grid = kwargs.pop("grid", None) or kwargs.pop("params_grid", None)

    holdout_bars: int = int(kwargs.pop("holdout_bars", 250))
    holdout_pct_kw = kwargs.pop("holdout_pct", None)
    holdout_min_bars_kw = kwargs.pop("holdout_min_bars", None)
    holdout_max_bars_kw = kwargs.pop("holdout_max_bars", None)
    name_prefix: str | None = kwargs.pop("name_prefix", None)
    meta_extra: dict[str, Any] = kwargs.pop("meta_extra", {})

    top_k_features: int | None = kwargs.pop("top_k_features", None)
    ranking_folds: int = int(kwargs.pop("ranking_folds", 3))
    label_lookahead_bars: int = int(kwargs.pop("label_lookahead_bars", 0))
    balance_classes: bool = bool(kwargs.pop("balance_classes", True))
    class_balance_power_kw = kwargs.pop("class_balance_power", None)
    class_balance_max_ratio_kw = kwargs.pop("class_balance_max_ratio", None)
    quality_gate_enabled: bool = bool(kwargs.pop("quality_gate_enabled", True))
    quality_min_f1_lift: float = float(kwargs.pop("quality_min_f1_lift", 0.01))
    quality_min_trades: int = int(kwargs.pop("quality_min_trades", 8))
    quality_gate_hard_reject: bool = bool(kwargs.pop("quality_gate_hard_reject", False))
    quality_min_directional_f1: float = float(kwargs.pop("quality_min_directional_f1", 0.03))
    quality_min_side_recall: float = float(kwargs.pop("quality_min_side_recall", 0.01))
    quality_min_side_prediction_share: float = float(kwargs.pop("quality_min_side_prediction_share", 0.0))
    quality_min_side_prediction_count: int = int(kwargs.pop("quality_min_side_prediction_count", 0))
    quality_require_mc_nonnegative: bool = bool(kwargs.pop("quality_require_mc_nonnegative", True))
    quality_min_mc_sharpe_p50: float = float(kwargs.pop("quality_min_mc_sharpe_p50", -0.02))
    quality_min_profit_net: float = float(kwargs.pop("quality_min_profit_net", 0.0))
    quality_min_holdout_sharpe: float = float(kwargs.pop("quality_min_holdout_sharpe", 0.0))
    max_param_candidates_kw = kwargs.pop("max_param_candidates", None)
    param_sample_seed_kw = kwargs.pop("param_sample_seed", 42)
    search_backend_kw = kwargs.pop("search_backend", "grid")
    optuna_trials_kw = kwargs.pop("optuna_trials", None)
    optuna_timeout_seconds_kw = kwargs.pop("optuna_timeout_seconds", None)
    training_mode_kw = kwargs.pop("training_mode", "standard")
    candidate_chain_enabled_kw = kwargs.pop("candidate_chain_enabled", True)
    candidate_selection_criterion_kw = kwargs.pop("candidate_selection_criterion", "balanced")
    candidate_top_n_kw = kwargs.pop("candidate_top_n", 5)
    candidate_fresh_ratio_kw = kwargs.pop("candidate_fresh_ratio", 0.30)
    threshold_calibration_enabled_kw = kwargs.pop("threshold_calibration_enabled", True)
    threshold_calibration_pct_kw = kwargs.pop("threshold_calibration_pct", 0.08)
    threshold_calibration_min_bars_kw = kwargs.pop("threshold_calibration_min_bars", 500)
    threshold_calibration_max_bars_kw = kwargs.pop("threshold_calibration_max_bars", 4000)
    threshold_calibration_train_min_guard_kw = kwargs.pop("threshold_calibration_train_min_guard", 2000)

    mc_enabled: bool = bool(kwargs.pop("mc_enabled", True))
    mc_iters: int = int(kwargs.pop("mc_iters", 200))
    mc_block_len: int = int(kwargs.pop("mc_block_len", 100))

    annualize_sharpe: bool = bool(kwargs.pop("annualize_sharpe", True))

    training_mode = str(training_mode_kw or "standard").strip().lower()
    training_mode = _normalize_legacy_training_mode(training_mode_kw)
    candidate_chain_enabled = bool(candidate_chain_enabled_kw)
    candidate_selection_criterion = _normalize_candidate_criterion(candidate_selection_criterion_kw)
    try:
        candidate_top_n = int(candidate_top_n_kw)
    except Exception:
        candidate_top_n = 5
    candidate_top_n = int(np.clip(candidate_top_n, 1, 25))
    try:
        candidate_fresh_ratio = float(candidate_fresh_ratio_kw)
    except Exception:
        candidate_fresh_ratio = 0.30
    candidate_fresh_ratio = float(np.clip(candidate_fresh_ratio, 0.05, 0.80))

    if feature_stability_threshold is not None:
        try:
            threshold_val = float(feature_stability_threshold)
            if np.isfinite(threshold_val):
                feature_stability_threshold = float(np.clip(threshold_val, 0.0, 1.0))
            else:
                feature_stability_threshold = None
        except Exception:
            LOGGER.warning(
                "Ignoring invalid feature_stability_threshold=%r",
                feature_stability_threshold,
            )
            feature_stability_threshold = None

    _ = kwargs

    if "target" not in df.columns:
        raise ValueError("DataFrame musí obsahovat 'target'.")
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame musí obsahovat 'timestamp'.")
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Detect ternary vs binary target
    unique_targets = sorted(df["target"].unique())
    is_ternary = set(unique_targets) == {-1, 0, 1} or (set(unique_targets).issubset({-1, 0, 1}) and len(unique_targets) == 3)
    if is_ternary:
        LOGGER.info("Ternary target detected: %s", unique_targets)
        # For ternary, convert to 3-class: map -1->0, 0->1, 1->2 for sklearn
        y_map = {-1: 0, 0: 1, 1: 2}
        df["target"] = df["target"].map(y_map).astype(int)
    else:
        LOGGER.info("Binary target detected: %s", sorted(df["target"].unique()))
        df["target"] = df["target"].astype(int)

    if class_balance_power_kw is None:
        class_balance_power = 0.60 if is_ternary else 1.00
    else:
        class_balance_power = float(class_balance_power_kw)
    class_balance_power = float(np.clip(class_balance_power, 0.0, 1.0))

    if class_balance_max_ratio_kw is None:
        class_balance_max_ratio = 4.0 if is_ternary else 10.0
    else:
        class_balance_max_ratio = float(class_balance_max_ratio_kw)
    if (not np.isfinite(class_balance_max_ratio)) or class_balance_max_ratio < 1.0:
        class_balance_max_ratio = 10.0

    # split
    n_total = len(df)
    max_hold_allowed = max(n_total - 50, 0)

    holdout_pct: float | None = None
    try:
        if holdout_pct_kw is not None:
            hv = float(holdout_pct_kw)
            if np.isfinite(hv) and hv > 0.0:
                holdout_pct = float(np.clip(hv, 0.0, 0.95))
    except Exception:
        holdout_pct = None

    holdout_mode = "pct" if holdout_pct is not None else "bars"

    if holdout_min_bars_kw is None:
        holdout_min_bars = 250 if holdout_mode == "pct" else 0
    else:
        try:
            holdout_min_bars = max(0, int(holdout_min_bars_kw))
        except Exception:
            holdout_min_bars = 0

    if holdout_max_bars_kw is None:
        holdout_max_bars: int | None = None
    else:
        try:
            hv = int(holdout_max_bars_kw)
            holdout_max_bars = max(0, hv)
        except Exception:
            holdout_max_bars = None

    requested_holdout_bars = int(max(0, holdout_bars))
    if holdout_mode == "pct" and holdout_pct is not None:
        requested_holdout_bars = int(round(float(n_total) * float(holdout_pct)))

    n_hold = int(max(requested_holdout_bars, int(holdout_min_bars)))
    if holdout_max_bars is not None:
        n_hold = int(min(n_hold, int(holdout_max_bars)))
    n_hold = int(min(max(0, n_hold), int(max_hold_allowed)))

    holdout_selection = {
        "mode": holdout_mode,
        "requested_bars": int(requested_holdout_bars),
        "requested_pct": (float(holdout_pct) if holdout_pct is not None else None),
        "min_bars": int(holdout_min_bars),
        "max_bars": (int(holdout_max_bars) if holdout_max_bars is not None else None),
        "applied_bars": int(n_hold),
        "max_allowed_bars": int(max_hold_allowed),
        "min_train_bars_guard": 50,
    }

    if n_hold > 0:
        df_train = df.iloc[: n_total - n_hold].reset_index(drop=True)
        df_hold  = df.iloc[n_total - n_hold :].reset_index(drop=True)
    else:
        df_train = df
        df_hold = None
    df_train_pre_guard = df_train.copy()

    # Leak guard for forward-looking labels (e.g. shift(-1), triple-barrier horizon).
    if n_hold > 0 and label_lookahead_bars > 0 and len(df_train) > (50 + label_lookahead_bars):
        df_train = df_train.iloc[: len(df_train) - int(label_lookahead_bars)].reset_index(drop=True)

    df_train_core, df_threshold_calib, threshold_calibration_selection, effective_embargo = _select_threshold_calibration_split(
        df_train,
        is_ternary=is_ternary,
        threshold_calibration_enabled=bool(threshold_calibration_enabled_kw),
        threshold_calibration_pct=float(threshold_calibration_pct_kw),
        threshold_calibration_min_bars=int(threshold_calibration_min_bars_kw),
        threshold_calibration_max_bars=int(threshold_calibration_max_bars_kw),
        threshold_calibration_train_min_guard=int(threshold_calibration_train_min_guard_kw),
        embargo=int(embargo),
        label_lookahead_bars=int(max(0, label_lookahead_bars)),
    )

    def _dist_from_frame(frame: pd.DataFrame | None) -> dict[str, int]:
        if frame is None or len(frame) == 0:
            return {}
        yv = frame["target"].astype(int).to_numpy()
        if is_ternary:
            yv = _mapped_ternary_to_signed(yv)
        return _target_distribution(yv)

    class_dist_all = _dist_from_frame(df)
    class_dist_train_pre_guard = _dist_from_frame(df_train_pre_guard)
    class_dist_train = _dist_from_frame(df_train)
    class_dist_train_core = _dist_from_frame(df_train_core)
    class_dist_threshold_calib = _dist_from_frame(df_threshold_calib)
    class_dist_holdout = _dist_from_frame(df_hold)

    n_train_pre_guard = int(len(df_train_pre_guard))
    n_train_effective = int(len(df_train))
    n_train_core = int(len(df_train_core))
    n_threshold_calib = int(len(df_threshold_calib) if df_threshold_calib is not None else 0)
    n_holdout_final = int(len(df_hold) if df_hold is not None else 0)

    meta_extra_safe = dict(meta_extra or {})
    for _k in (
        "n_total_bars",
        "n_train_bars",
        "n_train_bars_pre_guard",
        "n_train_core_bars",
        "n_threshold_calibration_bars",
        "n_holdout_bars",
        "holdout_selection",
        "threshold_calibration_selection",
        "class_distribution",
        "effective_embargo",
    ):
        meta_extra_safe.pop(_k, None)

    # featury
    feats_all = _select_feature_columns(df_train_core)
    if not feats_all:
        raise ValueError("Nenalezeny numerické featury.")
    feats = feats_all[:]
    if top_k_features and top_k_features > 0 and len(feats) > top_k_features:
        try:
            cv_rank = PurgedWalkForwardSplit(n_splits=max(2, ranking_folds), embargo=effective_embargo)
            imp_acc = np.zeros(len(feats))
            for tr_idx, _ in cv_rank.split(df_train_core[feats]):
                Xtr = df_train_core.iloc[tr_idx][feats].replace([np.inf, -np.inf], np.nan)
                ytr = df_train_core.iloc[tr_idx]["target"].astype(int).to_numpy()
                est_rank = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
                sw_rank = (
                    _balanced_sample_weight(
                        ytr,
                        power=class_balance_power,
                        max_ratio=class_balance_max_ratio,
                    )
                    if balance_classes
                    else None
                )
                _fit_estimator(est_rank, Xtr, ytr, sample_weight=sw_rank)
                fi = getattr(est_rank, "feature_importances_", None)
                if fi is not None and len(fi) == len(feats):
                    imp_acc += np.asarray(fi, dtype=float)
            order = np.argsort(imp_acc)[::-1]
            feats = [feats[i] for i in order[:top_k_features]]
        except Exception:
            pass
    X_all = df_train_core[feats].replace([np.inf, -np.inf], np.nan)
    y_all = df_train_core["target"].astype(int).to_numpy()
    feature_names_all = [str(col) for col in list(X_all.columns)]

    base_estimator, default_grid = _build_estimator(estimator_name)
    base_estimator = _ensure_pipeline(base_estimator)
    raw_grid = param_grid if isinstance(param_grid, dict) and len(param_grid) > 0 else default_grid
    grid_base = _namespaced_param_grid(base_estimator, raw_grid)
    all_param_sets_full = list(ParameterGrid(grid_base)) if grid_base else [dict()]

    max_param_candidates: int | None = None
    try:
        if max_param_candidates_kw is not None:
            mv = int(max_param_candidates_kw)
            if mv > 0:
                max_param_candidates = int(mv)
    except Exception:
        max_param_candidates = None

    try:
        param_sample_seed = int(param_sample_seed_kw)
    except Exception:
        param_sample_seed = 42

    optuna_trials_requested = _normalize_optuna_trials(optuna_trials_kw)
    optuna_timeout_seconds = _normalize_optuna_timeout_seconds(optuna_timeout_seconds_kw)

    search_backend_requested, search_backend_used, search_backend_fallback_reason = _resolve_search_backend(
        search_backend_kw,
        estimator_name=estimator_name,
    )

    all_param_sets = all_param_sets_full
    sampled_candidates = False
    candidate_source_by_key: dict[str, str] = {_params_key(p): "grid" for p in all_param_sets_full}

    chain_source_mode = {"standard": "quick", "strict": "standard"}.get(training_mode)
    chain_signature_hash, chain_signature_payload = _build_chain_signature(
        name_prefix=name_prefix,
        estimator_name=estimator_name,
        meta_extra=meta_extra_safe,
        holdout_mode=holdout_mode,
        holdout_pct=holdout_pct,
        holdout_min_bars=int(holdout_min_bars),
        holdout_max_bars=holdout_max_bars,
        holdout_bars=int(n_hold),
        label_lookahead_bars=int(max(0, label_lookahead_bars)),
        is_ternary=bool(is_ternary),
    )
    chain_path = _chain_shortlist_path(name_prefix, estimator_name)
    chain_info: dict[str, Any] = {
        "enabled": bool(candidate_chain_enabled),
        "mode": str(training_mode),
        "source_mode": chain_source_mode,
        "criterion": str(candidate_selection_criterion),
        "source_criterion": None,
        "invalid_reason": None,
        "reuse_decision": (
            "disabled"
            if not bool(candidate_chain_enabled)
            else ("source_mode_unavailable" if chain_source_mode is None else "fresh_sampling")
        ),
        "reranked_with_current_criterion": False,
        "top_n": int(candidate_top_n),
        "fresh_ratio": float(candidate_fresh_ratio),
        "path": str(chain_path),
        "signature_hash": str(chain_signature_hash),
        "signature_match": False,
        "used": False,
        "carry_count": 0,
        "fresh_count": 0,
    }

    carry_params: list[dict[str, Any]] = []
    if (
        search_backend_used == "grid"
        and bool(candidate_chain_enabled)
        and chain_source_mode is not None
    ):
        try:
            if chain_path.exists():
                chain_raw = jsonlib.loads(chain_path.read_text(encoding="utf-8"))
                if isinstance(chain_raw, dict):
                    sig_saved = str(chain_raw.get("signature_hash", "") or "")
                    chain_info["signature_match"] = bool(sig_saved == chain_signature_hash)
                    modes = chain_raw.get("modes") if isinstance(chain_raw.get("modes"), dict) else {}
                    src_entry = modes.get(chain_source_mode) if isinstance(modes, dict) else None
                    if not chain_info["signature_match"]:
                        chain_info["invalid_reason"] = "signature_mismatch"
                    elif not isinstance(src_entry, dict):
                        chain_info["invalid_reason"] = "source_mode_missing"
                    else:
                        all_keys = {_params_key(p) for p in all_param_sets_full}
                        src_criterion_norm = _normalize_candidate_criterion(src_entry.get("criterion"))
                        chain_info["source_criterion"] = str(src_criterion_norm)
                        if src_criterion_norm != candidate_selection_criterion:
                            # Criterion changes must invalidate carry reuse; otherwise the
                            # next phase silently mixes a new decision rule with stale evidence.
                            chain_info["invalid_reason"] = "criterion_mismatch"
                        else:
                            raw_cands = src_entry.get("candidates")
                            carry_rows: list[dict[str, Any]] = []
                            if isinstance(raw_cands, list):
                                for row in raw_cands:
                                    if not isinstance(row, dict):
                                        continue
                                    p = row.get("params")
                                    if not isinstance(p, dict):
                                        continue
                                    pk = _params_key(p)
                                    if pk in all_keys:
                                        row_copy = dict(row)
                                        row_copy["params"] = dict(p)
                                        carry_rows.append(row_copy)
                            if carry_rows:
                                ranked_carry_rows = _rank_candidates_for_chain(
                                    carry_rows, candidate_selection_criterion
                                )
                                carry_params = [
                                    dict(rr.get("params"))
                                    for rr in ranked_carry_rows
                                    if isinstance(rr.get("params"), dict)
                                ]
                                chain_info["reuse_decision"] = "carry_plus_fresh"
                            else:
                                chain_info["invalid_reason"] = "no_candidate_overlap"
                            chain_info["source_candidates"] = int(len(carry_params))
            else:
                chain_info["invalid_reason"] = "source_shortlist_missing"
        except Exception as e_chain_load:
            chain_info["load_error"] = str(e_chain_load)

    if search_backend_used != "grid":
        chain_info["disabled_for_backend"] = str(search_backend_used)
    elif carry_params:
        # Keep unique order from stored ranking.
        seen_carry: set[str] = set()
        carry_unique: list[dict[str, Any]] = []
        for p in carry_params:
            pk = _params_key(p)
            if pk in seen_carry:
                continue
            seen_carry.add(pk)
            carry_unique.append(p)
        carry_params = carry_unique[: int(max(1, candidate_top_n))]
        carry_keys = {_params_key(p) for p in carry_params}
        for ck in carry_keys:
            candidate_source_by_key[ck] = f"carry:{chain_source_mode}"

        remaining = [p for p in all_param_sets_full if _params_key(p) not in carry_keys]
        if max_param_candidates is not None:
            total_budget = max(int(max_param_candidates), len(carry_params))
        else:
            est_total = int(round(float(len(carry_params)) / max(1e-6, 1.0 - float(candidate_fresh_ratio))))
            total_budget = max(len(carry_params) + 4, est_total)
            total_budget = min(total_budget, len(all_param_sets_full))
        fresh_target = int(round(float(total_budget) * float(candidate_fresh_ratio)))
        fresh_target = min(fresh_target, max(0, int(total_budget) - len(carry_params)))
        if len(remaining) > 0 and fresh_target <= 0:
            fresh_target = 1
        fresh_target = int(min(max(0, fresh_target), len(remaining)))

        fresh_params: list[dict[str, Any]] = []
        if fresh_target > 0:
            rng_chain = np.random.default_rng(int(param_sample_seed) + 97)
            pick = rng_chain.choice(len(remaining), size=int(fresh_target), replace=False)
            pick_idx = sorted(int(i) for i in np.asarray(pick, dtype=int).tolist())
            fresh_params = [remaining[i] for i in pick_idx]
            for fp in fresh_params:
                candidate_source_by_key[_params_key(fp)] = "fresh"

        all_param_sets = carry_params + fresh_params
        sampled_candidates = len(all_param_sets) < len(all_param_sets_full)
        chain_info["used"] = True
        chain_info["carry_count"] = int(len(carry_params))
        chain_info["fresh_count"] = int(len(fresh_params))
    else:
        if max_param_candidates is not None and len(all_param_sets_full) > int(max_param_candidates):
            rng = np.random.default_rng(int(param_sample_seed))
            pick = rng.choice(len(all_param_sets_full), size=int(max_param_candidates), replace=False)
            pick_idx = sorted(int(i) for i in np.asarray(pick, dtype=int).tolist())
            all_param_sets = [all_param_sets_full[i] for i in pick_idx]
            sampled_candidates = True
            for p in all_param_sets:
                candidate_source_by_key[_params_key(p)] = "sampled"

    search_plan = {
        "search_backend_requested": str(search_backend_requested),
        "search_backend_used": str(search_backend_used),
        "search_backend_fallback_reason": search_backend_fallback_reason,
        "optuna_available": bool(HAS_OPTUNA),
        "optuna_trials_requested": (int(optuna_trials_requested) if optuna_trials_requested is not None else None),
        "optuna_timeout_seconds": (float(optuna_timeout_seconds) if optuna_timeout_seconds is not None else None),
        "optuna_completed_trials": 0,
        "optuna_pruned_trials": 0,
        "optuna_best_score": None,
        "optuna_best_params": None,
        "grid_total_candidates": int(len(all_param_sets_full)),
        "grid_used_candidates": int(len(all_param_sets)),
        "sampled_candidates": bool(sampled_candidates),
        "max_param_candidates": (int(max_param_candidates) if max_param_candidates is not None else None),
        "param_sample_seed": int(param_sample_seed),
        "candidate_chain": chain_info,
    }

    optuna_param_space = _coerce_grid_choices(grid_base) if search_backend_used == "optuna" else {}
    optuna_trials_effective = optuna_trials_requested
    if search_backend_used == "optuna" and optuna_trials_effective is None:
        optuna_trials_effective = max(1, len(all_param_sets_full))
    search_plan["optuna_trials_effective"] = (
        int(optuna_trials_effective) if optuna_trials_effective is not None else None
    )

    cv = PurgedWalkForwardSplit(n_splits=n_splits, embargo=effective_embargo)
    step_idx, total = 0, max(
        1,
        int(optuna_trials_effective)
        if search_backend_used == "optuna" and optuna_trials_effective is not None
        else len(all_param_sets),
    )
    best_score, best_params, best_estimator, best_oof = -1e18, None, None, None
    candidate_records: list[dict[str, Any]] = []

    def _emit(onp, idx, total, params, mean, std):
        if not onp:
            return
        try:
            onp(int(idx), int(total), dict(params), float(mean), float(std))
        except TypeError:
            onp(f"[CV {idx}/{total}] score={mean:.4f} std={std:.4f} params={params}")

    def _evaluate_candidate(
        params: dict[str, Any],
        *,
        trial=None,
    ) -> tuple[float, float, dict[str, Any], np.ndarray]:
        fold_scores, fold_sizes = [], []
        metric_wsum = 0.0
        metric_acc: dict[str, float] = {}
        n_short_sum = 0.0
        n_long_sum = 0.0
        n_dir_sum = 0.0
        fold_count = 0
        if is_ternary:
            tmp_oof = np.full(shape=(len(X_all), 2), fill_value=np.nan, dtype=float)  # [:,0]=SHORT, [:,1]=LONG
        else:
            tmp_oof = np.full(shape=(len(X_all),), fill_value=np.nan, dtype=float)

        for tr_idx, te_idx in cv.split(X_all):
            fold_count += 1
            X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
            X_te, y_te = X_all.iloc[te_idx], y_all[te_idx]
            df_te = df_train_core.iloc[te_idx]
            est = _fit_with_params(base_estimator, params)
            m_fold: dict[str, Any] | None = None
            n_short = 0
            n_long = 0
            sw_tr = (
                _balanced_sample_weight(
                    y_tr,
                    power=class_balance_power,
                    max_ratio=class_balance_max_ratio,
                )
                if balance_classes
                else None
            )
            _fit_estimator(est, X_tr, y_tr, sample_weight=sw_tr)
            if is_ternary and HAS_CALC_METRICS:
                try:
                    X_te_use = _align_X_for_estimator(est, X_te)
                    if hasattr(est, "predict_proba"):
                        pr = _call_with_feature_name_warning_suppressed(est.predict_proba, X_te_use)
                        if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 3:
                            y_pred_fold = _ternary_predict_mapped(pr[:, 0], pr[:, 2], 0.5, 0.5)
                        else:
                            y_pred_fold = np.asarray(_call_with_feature_name_warning_suppressed(est.predict, X_te_use)).astype(int)
                    else:
                        y_pred_fold = np.asarray(_call_with_feature_name_warning_suppressed(est.predict, X_te_use)).astype(int)
                    n_short = int((y_pred_fold == 0).sum())
                    n_long = int((y_pred_fold == 2).sum())
                    y_te_eval = _mapped_ternary_to_signed(y_te)
                    y_pred_eval = _mapped_ternary_to_signed(y_pred_fold)
                    m_fold = calculate_metrics(
                        y_true=y_te_eval, y_pred=y_pred_eval, df=df_te,
                        fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                        annualize_sharpe=True
                    )
                    score = _ternary_composite_score(m_fold, n_short=n_short, n_long=n_long)
                except Exception:
                    pred = _call_with_feature_name_warning_suppressed(est.predict, _align_X_for_estimator(est, X_te))
                    score = float((pred == y_te).mean())
            else:
                try:
                    score = pnl_scorer(est, X_te, y_te, df_te, fee=fee_per_trade, slippage=slippage_bps)
                except Exception:
                    pred = _call_with_feature_name_warning_suppressed(est.predict, _align_X_for_estimator(est, X_te))
                    score = float((pred == y_te).mean())
            fold_scores.append(float(score))
            fold_sizes.append(len(te_idx))
            if trial is not None:
                trial.report(float(np.average(fold_scores, weights=fold_sizes)), step=int(fold_count))
                if trial.should_prune():
                    trial.set_user_attr("pruned_after_fold", int(fold_count))
                    raise optuna.TrialPruned()
            if m_fold is not None:
                w = float(len(te_idx))
                metric_wsum += w

                def _acc_metric(metric_key: str, metric_val: Any):
                    try:
                        fv = float(metric_val)
                        if np.isfinite(fv):
                            metric_acc[metric_key] = metric_acc.get(metric_key, 0.0) + (w * fv)
                    except Exception:
                        pass

                _acc_metric("f1_macro_3", m_fold.get("f1_macro_3", np.nan))
                _acc_metric("profit_net", m_fold.get("profit_net", np.nan))
                shv = m_fold.get("sharpe_ann", np.nan)
                try:
                    if not np.isfinite(float(shv)):
                        shv = m_fold.get("sharpe", np.nan)
                except Exception:
                    shv = m_fold.get("sharpe", np.nan)
                _acc_metric("sharpe", shv)
                _acc_metric("pf", m_fold.get("pf", m_fold.get("profit_factor", np.nan)))
                _acc_metric("num_trades", m_fold.get("num_trades", m_fold.get("trades", np.nan)))
                try:
                    pc = m_fold.get("per_class_3") if isinstance(m_fold, dict) else None
                    rec_s = float((pc or {}).get("-1", {}).get("recall", np.nan))
                    rec_l = float((pc or {}).get("1", {}).get("recall", np.nan))
                    _acc_metric("rec_short", rec_s)
                    _acc_metric("rec_long", rec_l)
                except Exception:
                    pass
            n_short_sum += float(max(0, n_short))
            n_long_sum += float(max(0, n_long))
            n_dir_sum += float(max(0, n_short + n_long))
            if is_ternary:
                try:
                    X_te_use = _align_X_for_estimator(est, X_te)
                    pr = _call_with_feature_name_warning_suppressed(est.predict_proba, X_te_use) if hasattr(est, "predict_proba") else None
                    if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 3:
                        tmp_oof[te_idx, 0] = pr[:, 0]
                        tmp_oof[te_idx, 1] = pr[:, 2]
                except Exception:
                    pass
            else:
                proba = _predict_proba(est, X_te)
                if proba is not None:
                    tmp_oof[te_idx] = proba

        mean_score = float(np.average(fold_scores, weights=fold_sizes)) if fold_scores else -1e18
        std_score  = float(np.std(fold_scores)) if fold_scores else float("nan")
        row_rec: dict[str, Any] = {
            "params": dict(params),
            "cv_score": float(mean_score),
            "cv_std": float(std_score),
            "source": (
                "optuna"
                if search_backend_used == "optuna"
                else str(candidate_source_by_key.get(_params_key(params), "grid"))
            ),
            "folds": int(len(fold_scores)),
            "n_short_pred_mean": (float(n_short_sum / max(1, fold_count))),
            "n_long_pred_mean": (float(n_long_sum / max(1, fold_count))),
            "n_dir_pred_mean": (float(n_dir_sum / max(1, fold_count))),
        }
        if metric_wsum > 0.0:
            for mk, mv in metric_acc.items():
                row_rec[mk] = float(mv / metric_wsum)
        return mean_score, std_score, row_rec, tmp_oof.copy()

    def _run_candidate(params: dict[str, Any], *, trial=None) -> float:
        nonlocal step_idx, best_score, best_params, best_estimator, best_oof
        step_idx += 1
        mean_score, std_score, row_rec, tmp_oof = _evaluate_candidate(params, trial=trial)
        _emit(on_progress, step_idx, total, params, mean_score, std_score)
        candidate_records.append(row_rec)
        if mean_score > best_score:
            best_score, best_params = mean_score, params
            best_estimator = _fit_with_params(base_estimator, params)
            best_oof = tmp_oof
        return float(mean_score)

    if search_backend_used == "optuna":
        assert optuna is not None
        sampler = optuna.samplers.TPESampler(seed=int(param_sample_seed))
        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(1, min(5, total)), n_warmup_steps=1)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def _objective(trial) -> float:
            params = {
                str(param_name): trial.suggest_categorical(str(param_name), list(choices))
                for param_name, choices in optuna_param_space.items()
            }
            score = _run_candidate(params, trial=trial)
            trial.set_user_attr("params", dict(params))
            return score

        study.optimize(
            _objective,
            n_trials=int(optuna_trials_effective or total),
            timeout=float(optuna_timeout_seconds) if optuna_timeout_seconds is not None else None,
            gc_after_trial=True,
            show_progress_bar=False,
        )
        trial_states = optuna.trial.TrialState
        completed_trials = [t for t in study.trials if t.state == trial_states.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == trial_states.PRUNED]
        search_plan["optuna_completed_trials"] = int(len(completed_trials))
        search_plan["optuna_pruned_trials"] = int(len(pruned_trials))
        search_plan["optuna_timeout_reached"] = bool(
            optuna_timeout_seconds is not None and len(study.trials) < int(optuna_trials_effective or total)
        )
        if best_params is None:
            fallback_params = dict(all_param_sets_full[0]) if all_param_sets_full else {}
            _run_candidate(fallback_params)
            search_plan["optuna_no_completed_trials"] = True
        else:
            search_plan["optuna_no_completed_trials"] = False
        search_plan["optuna_best_score"] = float(best_score) if best_params is not None else None
        search_plan["optuna_best_params"] = dict(best_params or {})
    else:
        for params in all_param_sets:
            _run_candidate(dict(params))

    best_fold_feature_importances: list[dict[str, float]] = []
    if best_params is not None:
        try:
            best_fold_feature_importances = _collect_fold_feature_importances_for_params(
                cv=cv,
                base_estimator=base_estimator,
                params=dict(best_params),
                X_all=X_all,
                y_all=y_all,
                feature_names=feature_names_all,
                balance_classes=bool(balance_classes),
                class_balance_power=float(class_balance_power),
                class_balance_max_ratio=float(class_balance_max_ratio),
            )
        except Exception:
            best_fold_feature_importances = []

    try:
        feature_stability = _aggregate_feature_stability(
            best_fold_feature_importances,
            feature_names_all,
        )
    except Exception:
        feature_stability = {}
    raw_trained_features = list(feature_names_all)
    feature_stability_score_raw = compute_feature_stability_score(feature_stability)
    feature_stability_score = {
        str(feature_name): float(feature_stability_score_raw.get(feature_name, 0.0))
        for feature_name in raw_trained_features
    }
    trained_features = list(raw_trained_features)
    features_removed_by_stability: list[str] = []
    features_kept_by_stability: list[str] = list(raw_trained_features)
    feature_stability_filter_applied = False
    feature_stability_filter_fallback_reason: str | None = None

    if best_estimator is None:
        best_estimator = base_estimator
    sw_all = (
        _balanced_sample_weight(
            y_all,
            power=class_balance_power,
            max_ratio=class_balance_max_ratio,
        )
        if balance_classes
        else None
    )
    _fit_estimator(best_estimator, X_all, y_all, sample_weight=sw_all)

    calibrated_estimator = best_estimator
    if calibrate:
        try:
            calibrated_estimator = CalibratedClassifierCV(best_estimator, method="isotonic", cv=3)
            try:
                calibrated_estimator.fit(X_all, y_all, sample_weight=sw_all)
            except TypeError:
                calibrated_estimator.fit(X_all, y_all)
        except Exception:
            calibrated_estimator = best_estimator

    decision_threshold = 0.5
    ternary_threshold_short = 0.5
    ternary_threshold_long = 0.5
    threshold_tuning: dict[str, Any] = {}
    if is_ternary and best_oof is not None and HAS_CALC_METRICS:
        try:
            valid = np.isfinite(best_oof).all(axis=1)
            if valid.any():
                idx = valid.nonzero()[0]
                df_oof = df_train_core.iloc[idx].reset_index(drop=True)
                y_oof = df_oof["target"].astype(int).to_numpy()
                oof_short_full = np.asarray(best_oof[valid, 0], dtype=float)
                oof_long_full = np.asarray(best_oof[valid, 1], dtype=float)
                threshold_source = "oof_core"
                if df_threshold_calib is not None and len(df_threshold_calib) >= 120:
                    try:
                        X_cal = df_threshold_calib[feats].replace([np.inf, -np.inf], np.nan)
                        X_cal_use = _align_X_for_estimator(best_estimator, X_cal)
                        pr_cal = _call_with_feature_name_warning_suppressed(best_estimator.predict_proba, X_cal_use) if hasattr(best_estimator, "predict_proba") else None
                        if isinstance(pr_cal, np.ndarray) and pr_cal.ndim == 2 and pr_cal.shape[1] >= 3:
                            df_oof = df_threshold_calib.reset_index(drop=True)
                            y_oof = df_oof["target"].astype(int).to_numpy()
                            oof_short_full = np.asarray(pr_cal[:, 0], dtype=float)
                            oof_long_full = np.asarray(pr_cal[:, 2], dtype=float)
                            threshold_source = "calibration_split"
                    except Exception:
                        pass
                tune_window = min(len(y_oof), max(1200, min(4000, len(y_oof) // 6)))
                y_oof_tune = y_oof[-int(tune_window):]
                oof_short_tune = oof_short_full[-int(tune_window):]
                oof_long_tune = oof_long_full[-int(tune_window):]
                df_oof_tune = df_oof.iloc[-int(tune_window):].reset_index(drop=True)
                y_oof_dir = y_oof_tune[y_oof_tune != 1]
                n_recent = min(len(y_oof_tune), max(500, min(1500, len(y_oof_tune) // 3)))
                y_recent = y_oof_tune[-int(n_recent):]
                y_recent_dir = y_recent[y_recent != 1]
                target_short_share = None
                if y_recent_dir.size > 0:
                    target_short_share = float((y_recent_dir == 0).mean())
                elif y_oof_dir.size > 0:
                    target_short_share = float((y_oof_dir == 0).mean())
                if target_short_share is not None:
                    target_short_share = float(np.clip(target_short_share, 0.40, 0.60))
                target_dir_rate = None
                if y_recent.size > 0:
                    target_dir_rate = float(np.mean(y_recent != 1))
                elif y_oof_tune.size > 0:
                    target_dir_rate = float(np.mean(y_oof_tune != 1))
                if target_dir_rate is not None:
                    target_dir_rate = float(np.clip(target_dir_rate, 0.05, 0.30))
                dir_n = int(max(1, y_oof_dir.size))
                # Keep side minimum realistic: enough for stability, but never excessive.
                min_side = int(np.clip(round(dir_n * 0.01), 6, 40))
                max_dom = 0.72
                share_tol = 0.10
                dir_tol = 0.08
                threshold_tuning = {
                    "source": str(threshold_source),
                    "source_bars": int(len(y_oof)),
                    "target_short_share": target_short_share,
                    "target_dir_rate": target_dir_rate,
                    "short_share_tolerance": float(share_tol),
                    "dir_rate_tolerance": float(dir_tol),
                    "max_side_dominance": float(max_dom),
                    "min_side_signals": int(min_side),
                    "tune_window_bars": int(tune_window),
                    "recent_window_bars": int(n_recent),
                }
                ternary_threshold_short, ternary_threshold_long = _choose_thresholds_from_oof_ternary(
                    y_true_mapped=y_oof_tune, oof_short=oof_short_tune, oof_long=oof_long_tune,
                    df_oof=df_oof_tune, fee_per_trade=fee_per_trade, slippage_bps=slippage_bps, min_signals=20,
                    min_side_signals=min_side,
                    max_side_dominance=max_dom,
                    target_short_share=target_short_share,
                    target_dir_rate=target_dir_rate,
                    short_share_tolerance=share_tol,
                    dir_rate_tolerance=dir_tol,
                    balance_penalty_weight=0.90,
                    min_side_recall_target=float(quality_min_side_recall),
                    shortlist_max_candidates=140,
                )
                oof_stats = _ternary_signal_stats_from_oof(
                    oof_short_tune,
                    oof_long_tune,
                    float(ternary_threshold_short),
                    float(ternary_threshold_long),
                )
                threshold_tuning["oof_selected"] = oof_stats
                threshold_tuning["selected_mode_base"] = "score_grid"
                # If selected thresholds remain too one-sided on OOF, override by balanced quantiles.
                needs_override = (
                    int(oof_stats["n_short"]) < int(min_side)
                    or int(oof_stats["n_long"]) < int(min_side)
                    or float(oof_stats["dominance"]) > float(max_dom)
                    or (
                        target_dir_rate is not None
                        and abs(float(oof_stats.get("dir_rate", 0.0)) - float(target_dir_rate)) > float(dir_tol)
                    )
                )
                if needs_override:
                    non_hold_rate = (
                        float(target_dir_rate)
                        if target_dir_rate is not None and np.isfinite(target_dir_rate)
                        else float(np.mean(y_oof_tune != 1))
                    )
                    non_hold_rate = float(np.clip(non_hold_rate, 0.05, 0.30))
                    short_share_target = float(target_short_share) if target_short_share is not None else 0.5
                    p_short = float(np.clip(non_hold_rate * short_share_target, 0.01, 0.25))
                    p_long = float(np.clip(non_hold_rate * (1.0 - short_share_target), 0.01, 0.25))
                    q_short = float(np.quantile(oof_short_tune, 1.0 - p_short))
                    q_long = float(np.quantile(oof_long_tune, 1.0 - p_long))
                    ternary_threshold_short = float(np.clip(q_short, 0.03, 0.98))
                    ternary_threshold_long = float(np.clip(q_long, 0.03, 0.98))
                    threshold_tuning["selected_mode"] = "quantile_override"
                    threshold_tuning["selected_mode_base"] = "quantile_override"
                    threshold_tuning["quantile_override"] = {
                        "non_hold_rate": float(non_hold_rate),
                        "p_short": float(p_short),
                        "p_long": float(p_long),
                    }
                    threshold_tuning["oof_selected_after_override"] = _ternary_signal_stats_from_oof(
                        oof_short_tune,
                        oof_long_tune,
                        float(ternary_threshold_short),
                        float(ternary_threshold_long),
                    )
                else:
                    threshold_tuning["selected_mode"] = "score_grid"
                    threshold_tuning["selected_mode_base"] = "score_grid"

                thr_before_rebalance = {
                    "short": float(ternary_threshold_short),
                    "long": float(ternary_threshold_long),
                }
                stats_before_rebalance = _ternary_signal_stats_from_oof(
                    oof_short_tune,
                    oof_long_tune,
                    float(ternary_threshold_short),
                    float(ternary_threshold_long),
                )
                ternary_threshold_short, ternary_threshold_long, oof_final = _rebalance_ternary_thresholds_on_oof(
                    oof_short_tune,
                    oof_long_tune,
                    float(ternary_threshold_short),
                    float(ternary_threshold_long),
                    min_side_signals=min_side,
                    max_side_dominance=max_dom,
                    target_short_share=target_short_share,
                    target_dir_rate=target_dir_rate,
                    dir_rate_tolerance=dir_tol,
                    min_threshold=0.03,
                    max_iters=35,
                )
                threshold_tuning["oof_selected_after_rebalance"] = oof_final
                threshold_tuning["rebalance_adjustment"] = {
                    "before_thresholds": thr_before_rebalance,
                    "after_thresholds": {
                        "short": float(ternary_threshold_short),
                        "long": float(ternary_threshold_long),
                    },
                    "before_stats": stats_before_rebalance,
                    "after_stats": oof_final,
                }
                try:
                    before_dir = int(stats_before_rebalance.get("n_dir", 0.0))
                    after_dir = int((oof_final or {}).get("n_dir", 0.0))
                    before_short = int(stats_before_rebalance.get("n_short", 0.0))
                    before_long = int(stats_before_rebalance.get("n_long", 0.0))
                    after_short = int((oof_final or {}).get("n_short", 0.0))
                    after_long = int((oof_final or {}).get("n_long", 0.0))

                    rebalance_degraded = (
                        (before_dir > 0 and after_dir == 0)
                        or (before_dir >= 8 and after_dir <= max(1, int(round(0.25 * float(before_dir)))))
                        or (
                            before_short >= 4
                            and before_long >= 4
                            and after_short <= max(0, int(round(0.20 * float(before_short))))
                            and after_long <= max(0, int(round(0.20 * float(before_long))))
                        )
                    )
                    if rebalance_degraded:
                        ternary_threshold_short = float(thr_before_rebalance["short"])
                        ternary_threshold_long = float(thr_before_rebalance["long"])
                        oof_final = dict(stats_before_rebalance)
                        threshold_tuning["rebalance_adjustment"]["reverted"] = True
                        threshold_tuning["rebalance_adjustment"]["revert_reason"] = (
                            f"coverage_drop(before_dir={before_dir}, after_dir={after_dir})"
                        )
                        threshold_tuning["selected_mode"] = (
                            f"rebalance_reverted({threshold_tuning.get('selected_mode_base', threshold_tuning.get('selected_mode', 'n/a'))})"
                        )
                except Exception:
                    pass
                if (
                    abs(float(thr_before_rebalance["short"]) - float(ternary_threshold_short)) > 1e-12
                    or abs(float(thr_before_rebalance["long"]) - float(ternary_threshold_long)) > 1e-12
                ):
                    if str(threshold_tuning.get("selected_mode", "")).startswith("rebalance_reverted("):
                        pass
                    else:
                        threshold_tuning["selected_mode"] = (
                            f"rebalance({threshold_tuning.get('selected_mode_base', threshold_tuning.get('selected_mode', 'n/a'))})"
                        )
                min_side_floor = max(4, min(int(min_side), 20))
                ternary_threshold_short, ternary_threshold_long, post_search_tuning = _apply_single_fallback_ternary_thresholds(
                    y_true_mapped=y_oof_tune,
                    oof_short=oof_short_tune,
                    oof_long=oof_long_tune,
                    thr_short=float(ternary_threshold_short),
                    thr_long=float(ternary_threshold_long),
                    estimator_name=estimator_name,
                    min_side_floor=int(min_side_floor),
                    max_side_dominance=float(max_dom),
                    min_side_recall=float(quality_min_side_recall),
                )
                threshold_tuning.update(post_search_tuning)
                threshold_tuning = _finalize_threshold_tuning_outcome(threshold_tuning)
                decision_threshold = float((ternary_threshold_short + ternary_threshold_long) / 2.0)
        except Exception:
            pass
    elif (not is_ternary) and best_oof is not None and HAS_CALC_METRICS:
        valid = np.isfinite(best_oof)
        if valid.any():
            df_oof = df_train_core.iloc[valid.nonzero()[0]]
            y_oof = df_oof["target"].astype(int).to_numpy()
            decision_threshold = _choose_threshold_from_oof(
                y_true=y_oof, oof_proba=best_oof[valid], df_oof=df_oof,
                fee_per_trade=fee_per_trade, slippage_bps=slippage_bps
            )

    if feature_stability_threshold is not None:
        filter_result = evaluate_feature_stability_filter(
            feature_stability,
            raw_trained_features,
            float(feature_stability_threshold),
            logger=LOGGER,
        )
        trained_features = list(filter_result.kept_features)
        features_removed_by_stability = list(filter_result.removed_features)
        features_kept_by_stability = list(filter_result.kept_features)
        feature_stability_filter_applied = bool(filter_result.filter_applied)
        feature_stability_filter_fallback_reason = filter_result.fallback_reason

    # Refit final model on full pre-holdout train once thresholds are tuned on train_core/calibration.
    if (
        feature_stability_filter_applied
        or (is_ternary and df_threshold_calib is not None and len(df_threshold_calib) > 0)
    ):
        try:
            refit_df = (
                df_train
                if (is_ternary and df_threshold_calib is not None and len(df_threshold_calib) > 0)
                else df_train_core
            )
            X_refit = refit_df[trained_features].replace([np.inf, -np.inf], np.nan)
            y_refit = refit_df["target"].astype(int).to_numpy()
            sw_refit = (
                _balanced_sample_weight(
                    y_refit,
                    power=class_balance_power,
                    max_ratio=class_balance_max_ratio,
                )
                if balance_classes
                else None
            )
            est_refit = _fit_with_params(base_estimator, best_params or {})
            _fit_estimator(est_refit, X_refit, y_refit, sample_weight=sw_refit)
            best_estimator = est_refit
            calibrated_estimator = best_estimator
            if calibrate:
                try:
                    cal_refit = CalibratedClassifierCV(best_estimator, method="isotonic", cv=3)
                    try:
                        cal_refit.fit(X_refit, y_refit, sample_weight=sw_refit)
                    except TypeError:
                        cal_refit.fit(X_refit, y_refit)
                    calibrated_estimator = cal_refit
                except Exception:
                    calibrated_estimator = best_estimator
            threshold_tuning["refit_on_train_plus_calib"] = bool(
                is_ternary and df_threshold_calib is not None and len(df_threshold_calib) > 0
            )
            threshold_tuning["refit_train_bars"] = int(len(refit_df))
            threshold_tuning["refit_core_bars"] = int(len(df_train_core))
            threshold_tuning["refit_calibration_bars"] = int(
                len(df_threshold_calib) if df_threshold_calib is not None else 0
            )
            threshold_tuning["refit_feature_count"] = int(len(trained_features))
            threshold_tuning["refit_feature_stability_filtered"] = bool(
                feature_stability_filter_applied
            )
        except Exception as e_refit:
            if feature_stability_filter_applied:
                trained_features = list(raw_trained_features)
                features_removed_by_stability = []
                features_kept_by_stability = list(raw_trained_features)
                feature_stability_filter_applied = False
                if feature_stability_filter_fallback_reason is None:
                    feature_stability_filter_fallback_reason = (
                        "stability_filter_refit_failed_reverted_to_original_features"
                    )
            threshold_tuning["refit_on_train_plus_calib"] = False
            threshold_tuning["refit_error"] = str(e_refit)

    holdout_metrics, train_metrics, mc_summary = {}, {}, {}
    baseline_holdout_metrics: dict[str, Any] = {}
    holdout_chunk_diagnostics: list[dict[str, Any]] = []
    quality_gate: dict[str, Any] = {
        "enabled": bool(quality_gate_enabled),
        "hard_reject": bool(quality_gate_hard_reject),
        "evaluated": False,
        "passed": None,
        "reasons": [],
        "min_f1_lift": float(quality_min_f1_lift),
        "min_trades": int(quality_min_trades),
        "min_directional_f1": float(quality_min_directional_f1),
        "min_side_recall": float(quality_min_side_recall),
        "min_side_prediction_share": float(quality_min_side_prediction_share),
        "min_side_prediction_count": int(quality_min_side_prediction_count),
        "require_mc_nonnegative": bool(quality_require_mc_nonnegative),
        "min_mc_sharpe_p50": float(quality_min_mc_sharpe_p50),
        "min_profit_net": float(quality_min_profit_net),
        "min_holdout_sharpe": float(quality_min_holdout_sharpe),
        "baseline_metrics": {},
        "holdout_prediction_balance": {},
        "holdout_chunks": [],
    }
    n_signals_holdout: int | None = None
    n_signals_train: int | None = None
    base_threshold_mc: float = float(decision_threshold)

    # --- TRAIN metriky ---
    if is_ternary and best_oof is not None:
        valid = np.isfinite(best_oof).all(axis=1)
        if valid.any():
            try:
                df_train_valid = df_train_core.iloc[valid.nonzero()[0]]
                y_train = df_train_valid["target"].astype(int).to_numpy()
                ps = best_oof[valid, 0]
                pl = best_oof[valid, 1]
                y_train_pred = _ternary_predict_mapped(
                    ps,
                    pl,
                    float(ternary_threshold_short),
                    float(ternary_threshold_long),
                )
                n_signals_train = int((y_train_pred != 1).sum())
                if HAS_CALC_METRICS:
                    y_train_eval = _mapped_ternary_to_signed(y_train)
                    y_train_pred_eval = _mapped_ternary_to_signed(y_train_pred)
                    train_metrics = calculate_metrics(
                        y_true=y_train_eval,
                        y_pred=y_train_pred_eval,
                        df=df_train_valid,
                        fee_per_trade=fee_per_trade,
                        slippage_bps=slippage_bps,
                        annualize_sharpe=annualize_sharpe,
                    )
                else:
                    train_metrics = {"accuracy": float((y_train_pred == y_train).mean())}
            except Exception:
                pass
        else:
            # Fallback only if OOF is unavailable.
            try:
                X_train_eval = df_train_core[trained_features].replace([np.inf, -np.inf], np.nan)
                y_train = df_train_core["target"].astype(int).to_numpy()
                y_train_pred, n_signals_train, _ = _predict_labels_for_metrics(
                    calibrated_estimator,
                    X_train_eval,
                    decision_threshold=decision_threshold,
                    ternary_threshold_short=ternary_threshold_short,
                    ternary_threshold_long=ternary_threshold_long,
                    is_ternary=True,
                )
                if HAS_CALC_METRICS:
                    y_train_eval = _mapped_ternary_to_signed(y_train)
                    y_train_pred_eval = _mapped_ternary_to_signed(y_train_pred)
                    train_metrics = calculate_metrics(
                        y_true=y_train_eval,
                        y_pred=y_train_pred_eval,
                        df=df_train_core,
                        fee_per_trade=fee_per_trade,
                        slippage_bps=slippage_bps,
                        annualize_sharpe=annualize_sharpe,
                    )
                else:
                    train_metrics = {"accuracy": float((y_train_pred == y_train).mean())}
            except Exception:
                pass
    elif best_oof is not None:
        valid = np.isfinite(best_oof)
        if valid.any():
            try:
                df_train_valid = df_train_core.iloc[valid.nonzero()[0]]
                y_train = df_train_valid["target"].astype(int).to_numpy()
                y_train_pred = (best_oof[valid] >= float(decision_threshold)).astype(int)
                n_signals_train = int((best_oof[valid] >= float(decision_threshold)).sum())
                
                if HAS_CALC_METRICS:
                    train_metrics = calculate_metrics(
                        y_true=y_train, y_pred=y_train_pred, df=df_train_valid,
                        fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                        annualize_sharpe=annualize_sharpe
                    )
                else:
                    train_metrics = {"accuracy": float((y_train_pred == y_train).mean())}
            except Exception:
                pass

    if df_hold is not None and len(df_hold) >= 10:
        used_feats = list(trained_features)
        Xh = df_hold[used_feats].replace([np.inf, -np.inf], np.nan)
        yh = df_hold["target"].astype(int).to_numpy()
        proba = None
        try:
            if HAS_CALC_METRICS:
                ypred, n_signals_holdout, proba = _predict_labels_for_metrics(
                    calibrated_estimator,
                    Xh,
                    decision_threshold=decision_threshold,
                    ternary_threshold_short=ternary_threshold_short,
                    ternary_threshold_long=ternary_threshold_long,
                    is_ternary=is_ternary,
                )

                yh_eval = _mapped_ternary_to_signed(yh) if is_ternary else yh
                ypred_eval = _mapped_ternary_to_signed(ypred) if is_ternary else ypred
                holdout_metrics = calculate_metrics(
                    y_true=yh_eval, y_pred=ypred_eval, df=df_hold,
                    fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                    annualize_sharpe=annualize_sharpe
                )
                if is_ternary:
                    ybase_eval = np.zeros_like(yh_eval, dtype=int)  # all HOLD in signed label space
                    baseline_holdout_metrics = calculate_metrics(
                        y_true=yh_eval, y_pred=ybase_eval, df=df_hold,
                        fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                        annualize_sharpe=annualize_sharpe,
                    )
                    holdout_chunk_diagnostics = _build_holdout_chunk_diagnostics(
                        yh_eval,
                        ypred_eval,
                        df_hold,
                        fee_per_trade=fee_per_trade,
                        slippage_bps=slippage_bps,
                        annualize_sharpe=annualize_sharpe,
                    )
                    quality_gate["holdout_chunks"] = list(holdout_chunk_diagnostics)
                    if quality_gate_enabled:
                        quality_gate["holdout_prediction_balance"] = _prediction_side_balance_summary(ypred_eval)
                        gate_ok, gate_reasons = _quality_gate_vs_baseline_ternary(
                            holdout_metrics=holdout_metrics,
                            baseline_metrics=baseline_holdout_metrics,
                            min_f1_lift=quality_min_f1_lift,
                            min_trades=quality_min_trades,
                            y_pred=ypred_eval,
                            min_side_prediction_share=quality_min_side_prediction_share,
                            min_side_prediction_count=quality_min_side_prediction_count,
                        )
                        quality_gate["evaluated"] = True
                        quality_gate["passed"] = bool(gate_ok)
                        quality_gate["reasons"] = list(gate_reasons)
                        quality_gate["baseline_metrics"] = baseline_holdout_metrics

                mc_target_trades = 100
                if (not is_ternary) and proba is not None and len(proba) > 0 and n_signals_holdout is not None:
                    if n_signals_holdout < max(20, 0.2 * mc_target_trades):
                        frac = max(1, mc_target_trades) / max(1, len(proba))
                        q = 1.0 - min(0.95, max(0.01, float(frac)))
                        base_threshold_mc = max(float(np.quantile(proba, q)), 0.25)

                if mc_enabled:
                    mc_summary = _mc_eval_holdout_adaptive(
                        estimator=calibrated_estimator,
                        df_hold=df_hold,
                        features=used_feats,
                        base_threshold=float(base_threshold_mc),
                        ternary_threshold_short=(float(ternary_threshold_short) if is_ternary else None),
                        ternary_threshold_long=(float(ternary_threshold_long) if is_ternary else None),
                        iters=mc_iters,
                        block_len=mc_block_len,
                        fee_per_trade=fee_per_trade,
                        slippage_bps=slippage_bps,
                        min_trades=20,
                    )
            else:
                Xh_pred = _align_X_for_estimator(calibrated_estimator, Xh)
                acc = float((
                    _call_with_feature_name_warning_suppressed(calibrated_estimator.predict, Xh_pred) == yh
                ).mean())
                holdout_metrics = {"accuracy": acc}
        except Exception:
            pass

    if quality_gate_enabled:
        if is_ternary:
            if not bool(quality_gate.get("evaluated")):
                quality_gate["passed"] = False
                quality_gate["reasons"] = ["not_evaluated_no_valid_holdout_metrics"]
            else:
                extra_reasons: list[str] = []
                dir_f1 = _directional_f1_from_metrics(holdout_metrics)
                if np.isfinite(dir_f1) and dir_f1 < float(quality_min_directional_f1):
                    extra_reasons.append(
                        f"directional_f1_too_low({dir_f1:.4f}<{float(quality_min_directional_f1):.4f})"
                    )
                try:
                    pc = holdout_metrics.get("per_class_3") if isinstance(holdout_metrics, dict) else None
                    rec_short = float((pc or {}).get("-1", {}).get("recall", np.nan))
                    rec_long = float((pc or {}).get("1", {}).get("recall", np.nan))
                    if np.isfinite(rec_short) and rec_short < float(quality_min_side_recall):
                        extra_reasons.append(
                            f"short_recall_too_low({rec_short:.4f}<{float(quality_min_side_recall):.4f})"
                        )
                    if np.isfinite(rec_long) and rec_long < float(quality_min_side_recall):
                        extra_reasons.append(
                            f"long_recall_too_low({rec_long:.4f}<{float(quality_min_side_recall):.4f})"
                        )
                except Exception:
                    pass
                try:
                    profit_net = float(holdout_metrics.get("profit_net", np.nan))
                    min_profit = float(quality_min_profit_net)
                    if np.isfinite(profit_net) and profit_net < min_profit:
                        extra_reasons.append(f"profit_net_too_low({profit_net:.2f}<{min_profit:.2f})")
                except Exception:
                    pass
                try:
                    holdout_sharpe = float(holdout_metrics.get("sharpe", np.nan))
                    min_holdout_sharpe = float(quality_min_holdout_sharpe)
                    if np.isfinite(holdout_sharpe) and holdout_sharpe < min_holdout_sharpe:
                        extra_reasons.append(
                            f"holdout_sharpe_too_low({holdout_sharpe:.4f}<{min_holdout_sharpe:.4f})"
                        )
                except Exception:
                    pass
                if bool(quality_require_mc_nonnegative):
                    try:
                        mc_p50 = float((mc_summary or {}).get("sharpe", {}).get("p50", np.nan))
                        min_mc = float(quality_min_mc_sharpe_p50)
                        if np.isfinite(mc_p50) and mc_p50 < min_mc:
                            extra_reasons.append(f"mc_sharpe_p50_too_low({mc_p50:.4f}<{min_mc:.4f})")
                    except Exception:
                        pass
                if extra_reasons:
                    prev = list(quality_gate.get("reasons") or [])
                    merged = prev + [r for r in extra_reasons if r not in prev]
                    quality_gate["reasons"] = merged
                    quality_gate["passed"] = False
        else:
            quality_gate["reasons"] = ["not_applicable_binary_target"]

    # Persist cross-mode candidate shortlist (uses CV/OOF only; no holdout leakage in ranking).
    try:
        ranked_for_chain = _rank_candidates_for_chain(candidate_records, candidate_selection_criterion)
        shortlist_keep_n = int(np.clip(max(int(candidate_top_n), 12), 3, 40))
        shortlist_rows: list[dict[str, Any]] = []
        def _num_or_none(v: Any) -> float | None:
            try:
                fv = float(v)
                return float(fv) if np.isfinite(fv) else None
            except Exception:
                return None
        for rr in ranked_for_chain[:shortlist_keep_n]:
            row = {
                "params": dict(rr.get("params") or {}),
                "cv_score": float(rr.get("cv_score", np.nan)),
                "cv_std": float(rr.get("cv_std", np.nan)),
                "criterion_score": float(rr.get("criterion_score", np.nan)),
                "source": str(rr.get("source", "grid")),
                "f1_macro_3": _num_or_none(rr.get("f1_macro_3")),
                "profit_net": _num_or_none(rr.get("profit_net")),
                "sharpe": _num_or_none(rr.get("sharpe")),
                "pf": _num_or_none(rr.get("pf")),
                "rec_short": _num_or_none(rr.get("rec_short")),
                "rec_long": _num_or_none(rr.get("rec_long")),
                "n_dir_pred_mean": float(rr.get("n_dir_pred_mean", 0.0)),
                "n_short_pred_mean": float(rr.get("n_short_pred_mean", 0.0)),
                "n_long_pred_mean": float(rr.get("n_long_pred_mean", 0.0)),
            }
            shortlist_rows.append(row)

        payload_chain: dict[str, Any] = {}
        if chain_path.exists():
            try:
                prev_payload = jsonlib.loads(chain_path.read_text(encoding="utf-8"))
                if isinstance(prev_payload, dict):
                    payload_chain = prev_payload
            except Exception:
                payload_chain = {}
        modes_payload = payload_chain.get("modes") if isinstance(payload_chain.get("modes"), dict) else {}
        entry_now = {
            "updated_at": _now_str(),
            "mode": str(training_mode),
            "criterion": str(candidate_selection_criterion),
            "top_n": int(candidate_top_n),
            "fresh_ratio": float(candidate_fresh_ratio),
            "candidate_count_stored": int(len(shortlist_rows)),
            "candidate_count_seen": int(len(candidate_records)),
            "candidates": shortlist_rows,
        }
        modes_payload[str(training_mode)] = entry_now
        payload_chain["updated_at"] = _now_str()
        payload_chain["version"] = "1.0"
        payload_chain["name_prefix"] = str(name_prefix or "")
        payload_chain["estimator_name"] = str(estimator_name or "")
        payload_chain["signature_hash"] = str(chain_signature_hash)
        payload_chain["signature"] = chain_signature_payload
        payload_chain["modes"] = modes_payload
        chain_path.write_text(jsonlib.dumps(payload_chain, ensure_ascii=False, indent=2), encoding="utf-8")
        chain_info["shortlist_saved"] = True
        chain_info["shortlist_saved_count"] = int(len(shortlist_rows))
    except Exception as e_chain_save:
        chain_info["shortlist_saved"] = False
        chain_info["save_error"] = str(e_chain_save)
    if quality_gate_enabled and quality_gate_hard_reject and is_ternary:
        if bool(quality_gate.get("passed")) is not True:
            reasons = quality_gate.get("reasons") or ["quality_gate_failed"]
            out_dir = _model_dir()
            rej_ts = _now_str()
            est_short = (estimator_name or "model").lower()
            rej_name = f"{est_short}_{rej_ts}_rejected_meta.json" if not name_prefix else f"{name_prefix}_{est_short}_{rej_ts}_rejected_meta.json"
            rej_path = out_dir / rej_name
            rej_meta = {
                "created_at": rej_ts,
                "status": "rejected_by_quality_gate",
                "estimator_name": estimator_name,
                "search_backend": str(search_backend_used),
                "optuna_trials": (int(optuna_trials_effective) if search_backend_used == "optuna" and optuna_trials_effective is not None else None),
                "optuna_best_score": (float(best_score) if search_backend_used == "optuna" and best_params is not None else None),
                "optuna_best_params": (dict(best_params or {}) if search_backend_used == "optuna" else None),
                "best_params": best_params or {},
                "decision_threshold": float(decision_threshold),
                "ternary_threshold_short": float(ternary_threshold_short),
                "ternary_threshold_long": float(ternary_threshold_long),
                "threshold_tuning": threshold_tuning,
                "n_total_bars": int(n_total),
                "n_train_bars": int(n_train_effective),
                "n_train_bars_pre_guard": int(n_train_pre_guard),
                "n_train_core_bars": int(n_train_core),
                "n_threshold_calibration_bars": int(n_threshold_calib),
                "n_holdout_bars": int(n_holdout_final),
                "holdout_selection": holdout_selection,
                "threshold_calibration_selection": threshold_calibration_selection,
                "label_lookahead_bars": int(max(0, label_lookahead_bars)),
                "effective_embargo": int(effective_embargo),
                "balance_classes": bool(balance_classes),
                "class_balance_power": float(class_balance_power),
                "class_balance_max_ratio": float(class_balance_max_ratio),
                "class_distribution": {
                    "all": class_dist_all,
                    "train_pre_guard": class_dist_train_pre_guard,
                    "train": class_dist_train,
                    "train_core": class_dist_train_core,
                    "threshold_calibration": class_dist_threshold_calib,
                    "holdout": class_dist_holdout,
                },
                "metrics_train": {**train_metrics, **({"n_signals_train": n_signals_train} if n_signals_train is not None else {})},
                "metrics_holdout": {**holdout_metrics, **({"n_signals_holdout": n_signals_holdout} if n_signals_holdout is not None else {})},
                "mc_summary": mc_summary,
                "quality_gate": quality_gate,
                "search_plan": search_plan,
                **meta_extra_safe,
            }
            try:
                rej_path.write_text(jsonlib.dumps(rej_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            raise RuntimeError(
                "QUALITY_GATE_REJECT: holdout quality gate failed: "
                + "; ".join(str(r) for r in reasons)
                + f" | diag_meta={str(rej_path)}"
            )

    # --- uložení
    out_dir = _model_dir()
    ts = _now_str()
    est_short = (estimator_name or "model").lower()
    fname = f"{est_short}_{ts}.pkl" if not name_prefix else f"{name_prefix}_{est_short}_{ts}.pkl"
    fpath = out_dir / fname

    payload = {
        "model": calibrated_estimator,
        "features": list(trained_features),
        "feature_stability": dict(feature_stability),
        "feature_stability_threshold": (
            float(feature_stability_threshold)
            if feature_stability_threshold is not None
            else None
        ),
        "feature_stability_filter_requested": bool(feature_stability_threshold is not None),
        "feature_stability_score": dict(feature_stability_score),
        "features_removed_by_stability": list(features_removed_by_stability),
        "features_kept_by_stability": list(features_kept_by_stability),
        "feature_stability_filter_applied": bool(feature_stability_filter_applied),
        "feature_stability_filter_fallback_reason": (
            str(feature_stability_filter_fallback_reason)
            if feature_stability_filter_fallback_reason is not None
            else None
        ),
        "estimator_name": estimator_name,
        "search_backend": str(search_backend_used),
        "optuna_trials": (int(optuna_trials_effective) if search_backend_used == "optuna" and optuna_trials_effective is not None else None),
        "optuna_best_score": (float(best_score) if search_backend_used == "optuna" and best_params is not None else None),
        "optuna_best_params": (dict(best_params or {}) if search_backend_used == "optuna" else None),
        "best_params": best_params or {},
        "cv_results_full": [],
        "decision_threshold": float(decision_threshold),
        "ternary_threshold_short": float(ternary_threshold_short),
        "ternary_threshold_long": float(ternary_threshold_long),
        "created_at": ts,
        "version": "1.8_mc_ann_qgate_hard",
        "sklearn_version": runtime_sklearn_version(),
        "python_version": runtime_python_version(),
        "fee_per_trade": float(fee_per_trade),
        "slippage_bps": float(slippage_bps),
        "n_total_bars": int(n_total),
        "n_train_bars": int(n_train_effective),
        "n_train_bars_pre_guard": int(n_train_pre_guard),
        "n_train_core_bars": int(n_train_core),
        "n_threshold_calibration_bars": int(n_threshold_calib),
        "n_holdout_bars": int(n_holdout_final),
        "holdout_selection": holdout_selection,
        "threshold_calibration_selection": threshold_calibration_selection,
        "label_lookahead_bars": int(max(0, label_lookahead_bars)),
        "effective_embargo": int(effective_embargo),
        "balance_classes": bool(balance_classes),
        "class_balance_power": float(class_balance_power),
        "class_balance_max_ratio": float(class_balance_max_ratio),
        "quality_gate": quality_gate,
        "threshold_tuning": threshold_tuning,
        "search_plan": search_plan,
        "class_distribution": {
            "all": class_dist_all,
            "train_pre_guard": class_dist_train_pre_guard,
            "train": class_dist_train,
            "train_core": class_dist_train_core,
            "threshold_calibration": class_dist_threshold_calib,
            "holdout": class_dist_holdout,
        },
        "annualize_sharpe": bool(annualize_sharpe),
        "metrics_train": {**train_metrics, **({"n_signals_train": n_signals_train} if n_signals_train is not None else {})},
        "metrics_holdout": {**holdout_metrics, **({"n_signals_holdout": n_signals_holdout} if n_signals_holdout is not None else {})},
        **meta_extra_safe,
    }
    import joblib
    joblib.dump(payload, fpath)

    meta = {
        "created_at": ts,
        "created_at_iso": datetime.now().isoformat(),
        "estimator_name": estimator_name,
        "search_backend": str(search_backend_used),
        "optuna_trials": (int(optuna_trials_effective) if search_backend_used == "optuna" and optuna_trials_effective is not None else None),
        "optuna_best_score": (float(best_score) if search_backend_used == "optuna" and best_params is not None else None),
        "optuna_best_params": (dict(best_params or {}) if search_backend_used == "optuna" else None),
        "sklearn_version": runtime_sklearn_version(),
        "python_version": runtime_python_version(),
        "best_params": best_params or {},
        "decision_threshold": float(decision_threshold),
        "ternary_threshold_short": float(ternary_threshold_short),
        "ternary_threshold_long": float(ternary_threshold_long),
        "trained_features": list(trained_features),
        "n_features": len(trained_features),
        "n_total_bars": int(n_total),
        "n_train_bars": int(n_train_effective),
        "n_train_bars_pre_guard": int(n_train_pre_guard),
        "n_train_core_bars": int(n_train_core),
        "n_threshold_calibration_bars": int(n_threshold_calib),
        "n_holdout_bars": int(n_holdout_final),
        "holdout_selection": holdout_selection,
        "threshold_calibration_selection": threshold_calibration_selection,
        "label_lookahead_bars": int(max(0, label_lookahead_bars)),
        "effective_embargo": int(effective_embargo),
        "balance_classes": bool(balance_classes),
        "class_balance_power": float(class_balance_power),
        "class_balance_max_ratio": float(class_balance_max_ratio),
        "quality_gate": quality_gate,
        "threshold_tuning": threshold_tuning,
        "search_plan": search_plan,
        "class_distribution": {
            "all": class_dist_all,
            "train_pre_guard": class_dist_train_pre_guard,
            "train": class_dist_train,
            "train_core": class_dist_train_core,
            "threshold_calibration": class_dist_threshold_calib,
            "holdout": class_dist_holdout,
        },
        "class_to_dir": ({0: "SHORT", 1: "HOLD", 2: "LONG"} if is_ternary else {0: "SHORT", 1: "LONG"}),
        "classes": None,
        "annualize_sharpe": bool(annualize_sharpe),
        **meta_extra_safe,
        "metrics_train": {**train_metrics, **({"n_signals_train": n_signals_train} if n_signals_train is not None else {})},
        "metrics_holdout": {**holdout_metrics, **({"n_signals_holdout": n_signals_holdout} if n_signals_holdout is not None else {})},
        "metrics": {**holdout_metrics, **({"n_signals_holdout": n_signals_holdout} if n_signals_holdout is not None else {})},  # backward compat
        "mc": mc_summary,
        "feature_importance": {},  # bude naplněno níže
    }
    
    meta["feature_stability"] = dict(feature_stability)
    meta["feature_stability_threshold"] = (
        float(feature_stability_threshold)
        if feature_stability_threshold is not None
        else None
    )
    meta["feature_stability_filter_requested"] = bool(feature_stability_threshold is not None)
    meta["feature_stability_score"] = dict(feature_stability_score)
    meta["features_removed_by_stability"] = list(features_removed_by_stability)
    meta["features_kept_by_stability"] = list(features_kept_by_stability)
    meta["feature_stability_filter_applied"] = bool(feature_stability_filter_applied)
    meta["feature_stability_filter_fallback_reason"] = (
        str(feature_stability_filter_fallback_reason)
        if feature_stability_filter_fallback_reason is not None
        else None
    )

    # Feature importance
    try:
        X_feature_importance = df_train[trained_features].replace([np.inf, -np.inf], np.nan)
        y_feature_importance = df_train["target"].astype(int).to_numpy()
        imp_map = _extract_normalized_feature_importance_map(
            calibrated_estimator,
            list(trained_features),
            X_feature_importance,
            y_feature_importance,
        )
        if imp_map:
            sorted_imp = sorted(imp_map.items(), key=lambda item: item[1], reverse=True)
            meta["feature_importance"] = {
                str(feature_name): float(score)
                for feature_name, score in sorted_imp[:20]
            }
    except Exception:
        meta["feature_importance"] = {}
    try:
        est_for_cls = calibrated_estimator
        if isinstance(calibrated_estimator, Pipeline):
            est_for_cls = calibrated_estimator.steps[-1][1]
        if hasattr(est_for_cls, "classes_"):
            meta["classes"] = [int(c) for c in list(getattr(est_for_cls, "classes_"))]
    except Exception:
        meta["classes"] = None

    meta_path = fpath.with_name(f"{fpath.stem}_meta.json")

    def _json_default_for_meta(value: Any):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    try:
        meta_path.write_text(
            jsonlib.dumps(meta, ensure_ascii=False, indent=2, default=_json_default_for_meta),
            encoding="utf-8",
        )
    except Exception:
        pass

    def _emit_done(onp, idx, total, params, mean, std):
        if not onp:
            return
        try:
            onp(int(idx), int(total), dict(params), float(mean), float(std))
        except TypeError:
            onp(f"[DONE] {params}")
    _emit_done(on_progress, total, total, {"saved_model": str(fpath)}, float(best_score), 0.0)

    return {
        "output_path": str(fpath),
        "best_score": float(best_score),
        "best_params": best_params or {},
        "search_backend": str(search_backend_used),
        "optuna_trials": (int(optuna_trials_effective) if search_backend_used == "optuna" and optuna_trials_effective is not None else None),
        "optuna_best_score": (float(best_score) if search_backend_used == "optuna" and best_params is not None else None),
        "optuna_best_params": (dict(best_params or {}) if search_backend_used == "optuna" else None),
        "n_features": len(X_all.columns),
        "decision_threshold": float(decision_threshold),
        "cv_records_len": int(len(candidate_records)),
        "n_total_bars": int(n_total),
        "n_train_bars": int(n_train_effective),
        "n_train_bars_pre_guard": int(n_train_pre_guard),
        "n_train_core_bars": int(n_train_core),
        "n_threshold_calibration_bars": int(n_threshold_calib),
        "n_holdout_bars": int(n_holdout_final),
        "holdout_selection": holdout_selection,
        "threshold_calibration_selection": threshold_calibration_selection,
        "effective_embargo": int(effective_embargo),
        "quality_gate": quality_gate,
        "threshold_tuning": threshold_tuning,
        "search_plan": search_plan,
    }


# Backwards compatibility: some tests / CLI expect `train_simple_model`
# to be importable from `ibkr_trading_bot.model.train_models`.
from ibkr_trading_bot.model.data_split import train_simple_model as train_simple_model  # type: ignore
