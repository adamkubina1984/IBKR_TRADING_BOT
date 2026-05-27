from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ibkr_trading_bot.core.services.model_eval_helpers import (
    _build_holdout_metrics_payload,
    _model_threshold_snapshot,
    normalize_path,
    safe_float,
    utc_now_iso,
)
from ibkr_trading_bot.core.services.model_service import (
    feature_contract_from_meta,
    infer_label_mode_from_meta,
)
from ibkr_trading_bot.core.services.signal_policy import (
    DEFAULT_EXIT_POLICY,
    resolve_exit_policy_setting,
)


TAB5_HOLDOUT_RANKING_KEY = "tab5_holdout_ranking"
TAB5_HOLDOUT_RANKING_BY_POLICY_KEY = "tab5_holdout_ranking_by_policy"
TAB5_HOLDOUT_RANKING_SCHEMA_VERSION = 3


def build_tab5_holdout_ranking_payload(
    *,
    data_path: str | Path,
    fee_per_trade: float,
    exit_policy: str = DEFAULT_EXIT_POLICY,
    metadata: dict[str, Any] | None = None,
    base_entry_threshold: float | None = None,
    base_exit_threshold: float | None = None,
    base_metrics: dict[str, Any] | None = None,
    optimized_entry_threshold: float | None = None,
    optimized_exit_threshold: float | None = None,
    optimized_metrics: dict[str, Any] | None = None,
    entry_threshold: float | None,
    exit_threshold: float | None,
    metrics: dict[str, Any] | None = None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    normalized_path = normalize_path(data_path)
    csv_stat = Path(normalized_path).stat()
    meta = metadata if isinstance(metadata, dict) else {}
    if optimized_entry_threshold is None:
        optimized_entry_threshold = entry_threshold
    if optimized_exit_threshold is None:
        optimized_exit_threshold = exit_threshold
    if optimized_metrics is None:
        optimized_metrics = metrics

    base_payload = _build_holdout_metrics_payload(
        entry_threshold=base_entry_threshold,
        exit_threshold=base_exit_threshold,
        metrics=base_metrics,
    )
    optimized_payload = _build_holdout_metrics_payload(
        entry_threshold=optimized_entry_threshold,
        exit_threshold=optimized_exit_threshold,
        metrics=optimized_metrics,
    )
    selected_payload = optimized_payload or base_payload or {}
    threshold_snapshot = _model_threshold_snapshot(meta)

    out: dict[str, Any] = {
        "schema_version": TAB5_HOLDOUT_RANKING_SCHEMA_VERSION,
        "status": str(status),
        "exit_policy": resolve_exit_policy_setting(exit_policy, default=DEFAULT_EXIT_POLICY),
        "csv_path": normalized_path,
        "csv_size": int(csv_stat.st_size),
        "csv_mtime_ns": int(csv_stat.st_mtime_ns),
        "fee_per_trade": float(fee_per_trade),
        "scope": "holdout",
        "threshold_source": threshold_snapshot.get("source"),
        "label_mode": infer_label_mode_from_meta(meta),
        "feature_contract": feature_contract_from_meta(meta),
        "model_thresholds": threshold_snapshot,
        "entry_threshold": selected_payload.get("entry_threshold"),
        "exit_threshold": selected_payload.get("exit_threshold"),
        "profit_h": selected_payload.get("profit_h"),
        "max_dd_h": selected_payload.get("max_dd_h"),
        "trades_h": selected_payload.get("trades_h"),
        "evaluated_at": utc_now_iso(),
    }
    if base_payload is not None:
        out["holdout_base"] = base_payload
        out["base_entry_threshold"] = base_payload.get("entry_threshold")
        out["base_exit_threshold"] = base_payload.get("exit_threshold")
        out["base_profit_h"] = base_payload.get("profit_h")
        out["base_max_dd_h"] = base_payload.get("max_dd_h")
        out["base_trades_h"] = base_payload.get("trades_h")
    if optimized_payload is not None:
        optimized_payload["source"] = "holdout_auto_threshold_search"
        out["holdout_optimized"] = optimized_payload
    if error:
        out["error"] = str(error)
    return out


def set_tab5_holdout_ranking(
    meta: dict[str, Any] | None,
    ranking: dict[str, Any] | None,
    *,
    exit_policy: str = DEFAULT_EXIT_POLICY,
) -> dict[str, Any] | None:
    if not isinstance(meta, dict) or not isinstance(ranking, dict):
        return None
    policy_name = resolve_exit_policy_setting(
        exit_policy if exit_policy is not None else ranking,
        default=DEFAULT_EXIT_POLICY,
    )
    payload = dict(ranking)
    payload["exit_policy"] = policy_name
    by_policy = meta.get(TAB5_HOLDOUT_RANKING_BY_POLICY_KEY)
    if not isinstance(by_policy, dict):
        by_policy = {}
    by_policy = dict(by_policy)
    by_policy[policy_name] = payload
    meta[TAB5_HOLDOUT_RANKING_BY_POLICY_KEY] = by_policy
    meta[TAB5_HOLDOUT_RANKING_KEY] = payload
    return payload


def get_tab5_holdout_ranking(
    meta: dict[str, Any] | None,
    *,
    exit_policy: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    requested_policy = None
    if exit_policy is not None:
        requested_policy = resolve_exit_policy_setting(exit_policy, default=DEFAULT_EXIT_POLICY)
        by_policy = meta.get(TAB5_HOLDOUT_RANKING_BY_POLICY_KEY)
        if isinstance(by_policy, dict):
            ranking = by_policy.get(requested_policy)
            if isinstance(ranking, dict):
                return ranking
    ranking = meta.get(TAB5_HOLDOUT_RANKING_KEY)
    if not isinstance(ranking, dict):
        return None
    if requested_policy is None:
        return ranking
    ranking_policy = ranking.get("exit_policy")
    if ranking_policy is None:
        return None
    return ranking if resolve_exit_policy_setting(ranking_policy, default=DEFAULT_EXIT_POLICY) == requested_policy else None


def is_tab5_holdout_ranking_stale(
    meta: dict[str, Any] | None,
    *,
    data_path: str | Path,
    fee_per_trade: float,
    exit_policy: str = DEFAULT_EXIT_POLICY,
    model_path: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> bool:
    requested_policy = resolve_exit_policy_setting(exit_policy, default=DEFAULT_EXIT_POLICY)
    ranking = get_tab5_holdout_ranking(meta, exit_policy=requested_policy)
    if not isinstance(ranking, dict):
        return True
    ranking_policy = resolve_exit_policy_setting(ranking.get("exit_policy"), default=DEFAULT_EXIT_POLICY)
    if ranking_policy != requested_policy:
        return True
    if int(ranking.get("schema_version", 0) or 0) != TAB5_HOLDOUT_RANKING_SCHEMA_VERSION:
        return True
    normalized_path = normalize_path(data_path)
    csv_stat = Path(normalized_path).stat()
    if ranking.get("scope") != "holdout":
        return True
    if str(ranking.get("csv_path") or "") != normalized_path:
        return True
    if int(ranking.get("csv_size", -1) or -1) != int(csv_stat.st_size):
        return True
    if int(ranking.get("csv_mtime_ns", -1) or -1) != int(csv_stat.st_mtime_ns):
        return True
    ranking_fee = safe_float(ranking.get("fee_per_trade"))
    if ranking_fee is None or not np.isclose(float(ranking_fee), float(fee_per_trade), atol=1e-12):
        return True
    if ranking.get("feature_contract") != feature_contract_from_meta(meta):
        return True
    if str(ranking.get("label_mode") or "") != infer_label_mode_from_meta(meta):
        return True
    if ranking.get("model_thresholds") != _model_threshold_snapshot(meta):
        return True
    if meta_path is not None and model_path is not None:
        try:
            if Path(meta_path).exists() and Path(meta_path).stat().st_mtime_ns < Path(model_path).stat().st_mtime_ns:
                return True
        except OSError:
            return True
    return False


def ranking_status_from_error_message(message: str) -> str:
    msg = str(message or "").lower()
    unsupported_markers = [
        "ternarni model",
        "ternarni prahy",
        "predict_proba",
        "3 tridy",
        "threshold_short",
        "threshold_long",
    ]
    return "unsupported" if any(marker in msg for marker in unsupported_markers) else "error"