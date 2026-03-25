from __future__ import annotations

import json as jsonlib
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ibkr_trading_bot.core.services.dataset_service import DatasetService
from ibkr_trading_bot.model.train_models import _model_dir, train_and_evaluate_model

ALLOWED_CANDIDATE_CRITERIA = {
    "balanced",
    "profit_first",
    "robustness_first",
    "recall_balance",
}


def normalize_training_mode(mode: str | None) -> str:
    txt = str(mode or "").strip().lower()
    return txt if txt in {"quick", "standard", "strict"} else "standard"


def normalize_candidate_criterion(value: str | None, *, default: str = "balanced") -> str:
    txt = str(value or "").strip().lower()
    if txt in ALLOWED_CANDIDATE_CRITERIA:
        return txt
    fallback = str(default or "balanced").strip().lower()
    return fallback if fallback in ALLOWED_CANDIDATE_CRITERIA else "balanced"


def training_profile_for_mode(mode: str | None) -> dict[str, Any]:
    normalized = normalize_training_mode(mode)
    if normalized == "quick":
        return {
            "n_splits": 3,
            "top_k_features": 8,
            "max_param_candidates": 48,
            "param_sample_seed": 42,
            "search_backend": "grid",
            "optuna_trials": 12,
            "optuna_timeout_seconds": 120,
            "mc_enabled": False,
            "mc_iters": 80,
            "quality_gate_enabled": False,
            "quality_gate_hard_reject": False,
            "quality_min_trades": 6,
            "quality_min_side_recall": 0.005,
            "quality_require_mc_nonnegative": False,
            "quality_min_mc_sharpe_p50": -0.05,
            "quality_min_profit_net": -50.0,
            "quality_min_holdout_sharpe": -0.05,
            "candidate_chain_enabled": True,
            "candidate_selection_criterion": "balanced",
            "candidate_top_n": 5,
            "candidate_fresh_ratio": 0.30,
        }
    if normalized == "strict":
        return {
            "n_splits": 6,
            "top_k_features": 14,
            "max_param_candidates": 160,
            "param_sample_seed": 42,
            "search_backend": "grid",
            "optuna_trials": 48,
            "optuna_timeout_seconds": 900,
            "mc_enabled": True,
            "mc_iters": 300,
            "quality_gate_enabled": True,
            "quality_gate_hard_reject": True,
            "quality_min_trades": 8,
            "quality_min_side_recall": 0.01,
            "quality_require_mc_nonnegative": True,
            "quality_min_mc_sharpe_p50": 0.0,
            "quality_min_profit_net": 120.0,
            "quality_min_holdout_sharpe": 0.01,
            "candidate_chain_enabled": True,
            "candidate_selection_criterion": "balanced",
            "candidate_top_n": 5,
            "candidate_fresh_ratio": 0.30,
        }
    return {
        "n_splits": 5,
        "top_k_features": 12,
        "max_param_candidates": 96,
        "param_sample_seed": 42,
        "search_backend": "grid",
        "optuna_trials": 24,
        "optuna_timeout_seconds": 300,
        "mc_enabled": True,
        "mc_iters": 300,
        "quality_gate_enabled": True,
        "quality_gate_hard_reject": True,
        "quality_min_trades": 8,
        "quality_min_side_recall": 0.01,
        "quality_require_mc_nonnegative": True,
        "quality_min_mc_sharpe_p50": -0.03,
        "quality_min_profit_net": 60.0,
        "quality_min_holdout_sharpe": 0.005,
        "candidate_chain_enabled": True,
        "candidate_selection_criterion": "balanced",
        "candidate_top_n": 5,
        "candidate_fresh_ratio": 0.30,
    }


def compute_holdout_bars(n_total: int, pct: float, min_bars: int, max_bars: int) -> int:
    n = int(max(0, n_total))
    n_hold = int(round(float(n) * float(np.clip(pct, 0.0, 0.95))))
    n_hold = max(int(min_bars), n_hold)
    n_hold = min(int(max_bars), n_hold)
    n_hold = min(max(0, n_hold), max(n - 50, 0))
    return int(n_hold)


def name_and_meta_from_csv(path: str, n_total: int, n_train: int, n_hold: int) -> tuple[str, dict[str, Any]]:
    base = os.path.basename(path)
    instrument, exchange, timeframe = ("UNKNOWN", "UNK", "UNK")

    match = re.match(r"tv_([^_]+)_([^_]+)_([^_]+)_.+\.csv$", base)
    if match:
        instrument, exchange, timeframe = match.group(1), match.group(2), match.group(3)
    else:
        match = re.match(r"([A-Z0-9]+)_([0-9]+m|[0-9]+h|[0-9]+d)_(.+\.csv)$", base)
        if match:
            instrument = match.group(1)
            timeframe = match.group(2)
            exchange = "COMEX"

    name_prefix = f"{instrument}_{exchange}_{timeframe}_{int(n_total)}bars"
    meta_extra = {
        "instrument": instrument,
        "exchange": exchange,
        "timeframe": timeframe,
        "n_total_bars": int(n_total),
        "n_train_bars": int(n_train),
        "n_holdout_bars": int(n_hold),
    }
    return name_prefix, meta_extra


def mc_block_len_for_timeframe(timeframe: str | None) -> int:
    tf = str(timeframe or "").strip().lower()
    if tf in {"5min", "5m"}:
        return 40
    if tf in {"15min", "15m"}:
        return 80
    if tf in {"30min", "30m"}:
        return 120
    if tf in {"1hour", "1h"}:
        return 150
    return 100


def dataset_snapshot_signature_from_parts(
    instrument: str | None,
    exchange: str | None,
    timeframe: str | None,
    n_total_bars: int | str | None,
) -> tuple[str, str, str, int] | None:
    try:
        n_total = int(n_total_bars or 0)
    except Exception:
        n_total = 0
    inst = str(instrument or "").strip().upper()
    exch = str(exchange or "").strip().upper()
    tf = str(timeframe or "").strip().lower()
    if not inst or not exch or not tf or n_total <= 0:
        return None
    return (inst, exch, tf, int(n_total))


def dataset_snapshot_signature_from_csv(csv_path: str | Path, n_total_bars: int) -> tuple[str, str, str, int] | None:
    _, meta = name_and_meta_from_csv(str(csv_path), int(n_total_bars), 0, 0)
    return dataset_snapshot_signature_from_parts(
        meta.get("instrument"),
        meta.get("exchange"),
        meta.get("timeframe"),
        meta.get("n_total_bars"),
    )


def dataset_snapshot_signature_from_meta(meta: dict[str, Any] | None) -> tuple[str, str, str, int] | None:
    if not isinstance(meta, dict):
        return None
    return dataset_snapshot_signature_from_parts(
        meta.get("instrument"),
        meta.get("exchange"),
        meta.get("timeframe"),
        meta.get("n_total_bars"),
    )


def candidate_selection_criterion_from_meta(meta: dict[str, Any] | None, *, default: str = "balanced") -> str:
    if not isinstance(meta, dict):
        return normalize_candidate_criterion(default)

    candidates = [
        meta.get("candidate_selection_criterion"),
        (meta.get("training_profile") or {}).get("candidate_selection_criterion"),
        ((meta.get("search_plan") or {}).get("candidate_chain") or {}).get("criterion"),
        (meta.get("search_plan") or {}).get("criterion"),
    ]
    for value in candidates:
        normalized = normalize_candidate_criterion(value, default=default)
        if normalized in ALLOWED_CANDIDATE_CRITERIA:
            return normalized
    return normalize_candidate_criterion(default)


def _result_metrics_from_meta(meta_obj: dict[str, Any]) -> dict[str, Any]:
    metrics = (meta_obj.get("metrics_holdout") or meta_obj.get("metrics") or {}) if isinstance(meta_obj, dict) else {}
    return metrics if isinstance(metrics, dict) else {}


def run_training_job(
    *,
    csv_path: str,
    holdout_pct: float,
    holdout_min_bars: int,
    holdout_max_bars: int,
    phase: str,
    estimator_name: str,
    criterion: str,
    horizon: int,
    tp_bps: float,
    sl_bps: float,
    candidate_top_n: int,
    candidate_fresh_ratio: float,
    training_profile: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    phase_norm = normalize_training_mode(phase)
    estimator_norm = str(estimator_name or "").strip().lower()
    criterion_norm = normalize_candidate_criterion(criterion)
    horizon_value = int(horizon)
    tp_value = float(tp_bps)
    sl_value = float(sl_bps)
    profile = dict(training_profile or training_profile_for_mode(phase_norm))
    profile["training_mode"] = phase_norm
    profile["candidate_chain_enabled"] = bool(profile.get("candidate_chain_enabled", True))
    profile["candidate_selection_criterion"] = criterion_norm
    profile["candidate_top_n"] = int(max(1, candidate_top_n))
    profile["candidate_fresh_ratio"] = float(np.clip(candidate_fresh_ratio, 0.05, 0.80))

    svc = DatasetService()
    df = svc.prepare_from_csv(
        csv_path,
        labeling="triple_barrier",
        target_mode="ternary",
        horizon=horizon_value,
        take_profit_bps=tp_value,
        stop_loss_bps=sl_value,
        same_bar_policy="neutral",
    ).sort_values("timestamp").reset_index(drop=True)

    n_total = int(len(df))
    n_hold = compute_holdout_bars(
        n_total=n_total,
        pct=float(holdout_pct),
        min_bars=int(holdout_min_bars),
        max_bars=int(holdout_max_bars),
    )
    n_train = int(max(0, n_total - n_hold))
    name_prefix, meta_extra = name_and_meta_from_csv(str(csv_path), n_total, n_train, n_hold)

    mc_block = int(profile.get("mc_block_len", mc_block_len_for_timeframe(meta_extra.get("timeframe"))))
    profile["mc_block_len"] = mc_block

    meta_extra["mc_block_len"] = mc_block
    meta_extra["label_horizon_bars"] = int(horizon_value)
    meta_extra["label_take_profit_bps"] = float(tp_value)
    meta_extra["label_stop_loss_bps"] = float(sl_value)
    meta_extra["label_same_bar_policy"] = "neutral"
    meta_extra["label_lookahead_bars"] = int(horizon_value)
    meta_extra["holdout_mode"] = "pct"
    meta_extra["holdout_pct"] = float(holdout_pct)
    meta_extra["holdout_min_bars"] = int(holdout_min_bars)
    meta_extra["holdout_max_bars"] = int(holdout_max_bars)
    meta_extra["training_mode"] = phase_norm
    meta_extra["training_profile"] = dict(profile)
    if isinstance(extra_meta, dict) and extra_meta:
        meta_extra.update(extra_meta)

    started_ts = pd.Timestamp.now(tz="UTC").timestamp()
    model_path = ""
    meta_path = ""
    meta_obj: dict[str, Any] = {}
    status = "ok"
    err_msg = ""

    try:
        train_and_evaluate_model(
            df,
            estimator_name=estimator_norm,
            param_grid=None,
            on_progress=None,
            n_splits=int(profile.get("n_splits", 5)),
            holdout_bars=int(n_hold),
            holdout_pct=float(holdout_pct),
            holdout_min_bars=int(holdout_min_bars),
            holdout_max_bars=int(holdout_max_bars),
            name_prefix=name_prefix,
            meta_extra=meta_extra,
            mc_enabled=bool(profile.get("mc_enabled", True)),
            mc_iters=int(profile.get("mc_iters", 200)),
            mc_block_len=int(profile.get("mc_block_len", 100)),
            annualize_sharpe=True,
            top_k_features=int(profile.get("top_k_features", 12)),
            label_lookahead_bars=int(horizon_value),
            quality_gate_enabled=bool(profile.get("quality_gate_enabled", True)),
            quality_gate_hard_reject=bool(profile.get("quality_gate_hard_reject", True)),
            quality_min_trades=int(profile.get("quality_min_trades", 8)),
            quality_min_side_recall=float(profile.get("quality_min_side_recall", 0.01)),
            quality_require_mc_nonnegative=bool(profile.get("quality_require_mc_nonnegative", True)),
            quality_min_mc_sharpe_p50=float(profile.get("quality_min_mc_sharpe_p50", -0.02)),
            quality_min_profit_net=float(profile.get("quality_min_profit_net", 0.0)),
            quality_min_holdout_sharpe=float(profile.get("quality_min_holdout_sharpe", 0.0)),
            max_param_candidates=profile.get("max_param_candidates"),
            param_sample_seed=int(profile.get("param_sample_seed", 42)),
            search_backend=profile.get("search_backend", "grid"),
            optuna_trials=profile.get("optuna_trials"),
            optuna_timeout_seconds=profile.get("optuna_timeout_seconds"),
            training_mode=phase_norm,
            candidate_chain_enabled=bool(profile.get("candidate_chain_enabled", True)),
            candidate_selection_criterion=criterion_norm,
            candidate_top_n=int(profile.get("candidate_top_n", 5)),
            candidate_fresh_ratio=float(profile.get("candidate_fresh_ratio", 0.30)),
        )
    except Exception as exc:
        status = "rejected" if "QUALITY_GATE_REJECT" in str(exc) else "error"
        err_msg = str(exc)
        match = re.search(r"\|\s*diag_meta=(.+)$", err_msg)
        if match:
            meta_candidate = Path(match.group(1).strip().strip('"').strip("'"))
            if meta_candidate.exists():
                meta_path = meta_candidate.as_posix()
                try:
                    meta_obj = jsonlib.loads(meta_candidate.read_text(encoding="utf-8"))
                except Exception:
                    meta_obj = {}

    if not meta_obj:
        out_dir = Path(_model_dir())
        pattern = f"{name_prefix}_{estimator_norm}_*.pkl"
        files = [path for path in out_dir.glob(pattern) if path.stat().st_mtime >= (started_ts - 3.0)]
        if files:
            latest = sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)[0]
            model_path = latest.as_posix()
            meta_candidate = latest.with_name(latest.stem + "_meta.json")
            if meta_candidate.exists():
                meta_path = meta_candidate.as_posix()
                try:
                    meta_obj = jsonlib.loads(meta_candidate.read_text(encoding="utf-8"))
                except Exception:
                    meta_obj = {}

    metrics = _result_metrics_from_meta(meta_obj)
    quality_gate = (meta_obj.get("quality_gate") or {}) if isinstance(meta_obj, dict) else {}
    qg_reasons = list(quality_gate.get("reasons") or []) if isinstance(quality_gate, dict) else []

    return {
        "phase": phase_norm,
        "model": estimator_norm,
        "criterion": criterion_norm,
        "horizon": int(horizon_value),
        "tp_bps": float(tp_value),
        "sl_bps": float(sl_value),
        "status": status,
        "error": err_msg,
        "model_path": model_path,
        "meta_path": meta_path,
        "meta_obj": meta_obj,
        "search_plan": ((meta_obj.get("search_plan") or {}) if isinstance(meta_obj, dict) else {}),
        "search_backend_requested": (((meta_obj.get("search_plan") or {}).get("search_backend_requested")) if isinstance(meta_obj, dict) else None),
        "search_backend_used": (((meta_obj.get("search_plan") or {}).get("search_backend_used")) if isinstance(meta_obj, dict) else None),
        "search_backend_fallback_reason": (((meta_obj.get("search_plan") or {}).get("search_backend_fallback_reason")) if isinstance(meta_obj, dict) else None),
        "profit_net": metrics.get("profit_net"),
        "sharpe": metrics.get("sharpe"),
        "pf": metrics.get("pf"),
        "trades": metrics.get("trades", metrics.get("num_trades")),
        "num_trades_short": metrics.get("num_trades_short"),
        "num_trades_long": metrics.get("num_trades_long"),
        "qg_reasons": qg_reasons,
        "created_at": str(pd.Timestamp.now(tz="UTC").isoformat()),
    }


class ModelTrainingService:
    def __init__(self, model_repo, results_repo, logger):
        self.models = model_repo
        self.results = results_repo
        self.log = logger

    def rolling_retrain(self, df, features_cfg_path: str, **kwargs):
        self.log.info("rolling_retrain placeholder: zachovavame puvodni trenovaci tok")
        return None, {}

    def train_and_evaluate_model(self, *args, **kwargs):
        return train_and_evaluate_model(*args, **kwargs)
