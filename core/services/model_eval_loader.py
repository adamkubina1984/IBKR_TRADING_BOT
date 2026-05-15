from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ibkr_trading_bot.core.services.model_eval_helpers import normalize_path
from ibkr_trading_bot.core.services.model_service import (
    build_sklearn_version_warning,
    merge_model_metadata,
    read_sidecar_model_meta,
)


@dataclass
class LoadedPredictor:
    predictor: Any
    metadata: dict[str, Any]
    model_path: str
    version_warning: str | None = None


def extract_predictor_from_object(obj: Any) -> tuple[Any, dict[str, Any] | None]:
    if hasattr(obj, "predict"):
        return obj, None
    if isinstance(obj, dict):
        for key in ["model", "estimator", "pipeline", "clf", "best_estimator_", "sk_model", "predictor"]:
            value = obj.get(key)
            if hasattr(value, "predict"):
                return value, obj
        for value in obj.values():
            if hasattr(value, "predict"):
                return value, obj
        raise ValueError("Ve slovniku neni zadny objekt s `.predict`.")
    if isinstance(obj, (tuple, list)):
        for value in obj:
            if hasattr(value, "predict"):
                return value, None
        raise ValueError("V tuple/listu neni zadna polozka s `.predict`.")
    raise ValueError(f"Neocekavany typ ulozeneho modelu: {type(obj).__name__}.")


def load_predictor_with_merged_meta(model_path: str | Path) -> LoadedPredictor:
    normalized_path = normalize_path(model_path)
    obj = joblib.load(normalized_path)
    predictor, embedded_meta = extract_predictor_from_object(obj)
    metadata = merge_model_metadata(
        embedded_meta if isinstance(embedded_meta, dict) else {},
        read_sidecar_model_meta(normalized_path),
    )
    version_warning = build_sklearn_version_warning(metadata, model_path=normalized_path)
    return LoadedPredictor(
        predictor=predictor,
        metadata=metadata,
        model_path=normalized_path,
        version_warning=version_warning,
    )