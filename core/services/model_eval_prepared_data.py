from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ibkr_trading_bot.core.services.dataset_service import DatasetService
from ibkr_trading_bot.core.services.model_eval_helpers import normalize_path, safe_float
from ibkr_trading_bot.core.services.model_service import infer_label_mode_from_meta


@dataclass
class PreparedEvaluationData:
    data_path: str
    X_full: pd.DataFrame | np.ndarray
    y_true_full: np.ndarray | None
    df_for_metrics_full: pd.DataFrame


def evaluation_dataset_contract(metadata: dict[str, Any] | None) -> dict[str, Any]:
    meta = metadata if isinstance(metadata, dict) else {}
    label_mode = infer_label_mode_from_meta(meta)
    target_mode = "ternary" if label_mode.startswith("ternary") else "binary"

    horizon = 12
    try:
        horizon = int(meta.get("label_horizon_bars") or horizon)
    except Exception:
        pass
    horizon = int(max(1, horizon))

    tp_bps = safe_float(meta.get("label_take_profit_bps"))
    if tp_bps is None:
        tp_bps = 60.0

    sl_bps = safe_float(meta.get("label_stop_loss_bps"))
    if sl_bps is None:
        sl_bps = 40.0

    same_bar_policy = str(meta.get("label_same_bar_policy") or "neutral").strip().lower()
    if same_bar_policy not in {"neutral", "tp", "sl", "close"}:
        same_bar_policy = "neutral"

    has_triple_barrier_meta = (
        safe_float(meta.get("label_horizon_bars")) is not None
        and safe_float(meta.get("label_take_profit_bps")) is not None
        and safe_float(meta.get("label_stop_loss_bps")) is not None
    )

    return {
        "label_mode": label_mode,
        "target_mode": target_mode,
        "labeling": "triple_barrier" if has_triple_barrier_meta else "prepared",
        "horizon": horizon,
        "take_profit_bps": float(tp_bps),
        "stop_loss_bps": float(sl_bps),
        "same_bar_policy": same_bar_policy,
    }


def prepared_evaluation_cache_key(data_path: str | Path, metadata: dict[str, Any] | None) -> tuple[Any, ...]:
    contract = evaluation_dataset_contract(metadata)
    return (
        normalize_path(data_path),
        contract["label_mode"],
        contract["labeling"],
        contract["target_mode"],
        int(contract["horizon"]),
        float(contract["take_profit_bps"]),
        float(contract["stop_loss_bps"]),
        contract["same_bar_policy"],
    )


def load_prepared_evaluation_data(
    data_path: str | Path,
    metadata: dict[str, Any] | None = None,
    progress_cb=None,
) -> PreparedEvaluationData:
    normalized_path = normalize_path(data_path)
    contract = evaluation_dataset_contract(metadata)
    if callable(progress_cb):
        progress_cb("Vyhodnoceni: nacitam CSV...")
    if callable(progress_cb):
        progress_cb("Vyhodnoceni: pripravuji dataset...")
    prepared_df = DatasetService().prepare_from_csv(
        normalized_path,
        labeling=contract["labeling"],
        target_mode=contract["target_mode"],
        horizon=int(contract["horizon"]),
        take_profit_bps=float(contract["take_profit_bps"]),
        stop_loss_bps=float(contract["stop_loss_bps"]),
        same_bar_policy=contract["same_bar_policy"],
    )
    if "target" not in prepared_df.columns:
        raise ValueError("Po priprave datasetu chybi cilova promenna 'target'.")

    X = prepared_df.drop(columns=["target", "timestamp"], errors="ignore")
    y_true = np.asarray(prepared_df["target"])
    return PreparedEvaluationData(
        data_path=normalized_path,
        X_full=X,
        y_true_full=(np.asarray(y_true) if y_true is not None else None),
        df_for_metrics_full=prepared_df,
    )