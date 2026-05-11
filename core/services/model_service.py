from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import date, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

@dataclass
class LoadedModel:
    model: Any
    meta: dict
    path: Path
    sha1: str
    size: int
    version_warning: str | None = None


def runtime_python_version() -> str:
    return sys.version.split()[0]


def runtime_sklearn_version() -> str | None:
    try:
        import sklearn

        return str(sklearn.__version__)
    except Exception:
        return None


def build_sklearn_version_warning(meta: dict[str, Any] | None, *, model_path: str | Path | None = None) -> str | None:
    if not isinstance(meta, dict):
        return None

    stored = str(meta.get("sklearn_version") or "").strip()
    current = str(runtime_sklearn_version() or "").strip()
    if not stored or not current or stored == current:
        return None

    model_label = Path(model_path).name if model_path else "model"
    return (
        f"scikit-learn mismatch for {model_label}: "
        f"model={stored}, runtime={current}. Compatibility is not guaranteed."
    )


def model_sidecar_meta_path(model_path: str | Path) -> Path:
    p = Path(model_path).expanduser().resolve()
    return p.with_name(p.stem + "_meta.json")


def read_sidecar_model_meta(model_path: str | Path) -> dict[str, Any]:
    meta_path = model_sidecar_meta_path(model_path)
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Cannot read metadata %s: %s", meta_path, e)
        return {}
    return data if isinstance(data, dict) else {}


def merge_model_metadata(embedded: dict[str, Any] | None, sidecar: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(embedded, dict):
        merged.update(embedded)
    if isinstance(sidecar, dict):
        merged.update(sidecar)
    return merged


def _jsonify_meta_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonify_meta_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_meta_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonify_meta_value(item) for item in value.tolist()]
    return value


def synthesize_sidecar_model_meta(model: Any) -> dict[str, Any]:
    payload = model if isinstance(model, dict) else {}
    predictor = payload.get("model") if isinstance(payload, dict) and "model" in payload else model
    meta_keys = [
        "created_at",
        "created_at_iso",
        "estimator_name",
        "search_backend",
        "optuna_trials",
        "optuna_best_score",
        "optuna_best_params",
        "sklearn_version",
        "python_version",
        "best_params",
        "decision_threshold",
        "ternary_threshold_short",
        "ternary_threshold_long",
        "n_total_bars",
        "n_train_bars",
        "n_train_bars_pre_guard",
        "n_train_core_bars",
        "n_threshold_calibration_bars",
        "n_holdout_bars",
        "holdout_selection",
        "threshold_calibration_selection",
        "label_lookahead_bars",
        "label_horizon_bars",
        "label_take_profit_bps",
        "label_stop_loss_bps",
        "label_same_bar_policy",
        "effective_embargo",
        "balance_classes",
        "class_balance_power",
        "class_balance_max_ratio",
        "quality_gate",
        "threshold_tuning",
        "search_plan",
        "class_distribution",
        "class_to_dir",
        "annualize_sharpe",
        "metrics_train",
        "metrics_holdout",
        "metrics",
        "mc",
        "mc_summary",
        "feature_importance",
        "feature_stability",
        "feature_stability_threshold",
        "feature_stability_score",
        "features_removed_by_stability",
        "features_kept_by_stability",
        "feature_stability_filter_applied",
        "feature_stability_filter_fallback_reason",
        "instrument",
        "exchange",
        "timeframe",
        "training_mode",
        "training_profile",
        "fee_per_trade",
        "slippage_bps",
        "version",
    ]
    meta: dict[str, Any] = {}
    for key in meta_keys:
        if key in payload:
            meta[key] = _jsonify_meta_value(payload.get(key))

    features = payload.get("trained_features")
    if not isinstance(features, list):
        features = payload.get("features")
    if isinstance(features, list):
        meta["trained_features"] = [str(item) for item in features]

    classes = payload.get("model_classes")
    if not isinstance(classes, list):
        classes = payload.get("classes")
    if not isinstance(classes, list) and hasattr(predictor, "classes_"):
        try:
            classes = list(getattr(predictor, "classes_"))
        except Exception:
            classes = None
    if isinstance(classes, list):
        meta["model_classes"] = [_jsonify_meta_value(item) for item in classes]
        meta.setdefault("classes", [_jsonify_meta_value(item) for item in classes])

    if not meta.get("trained_features") and hasattr(predictor, "feature_names_in_"):
        try:
            meta["trained_features"] = [str(item) for item in list(getattr(predictor, "feature_names_in_"))]
        except Exception:
            pass

    if "metrics" not in meta and isinstance(meta.get("metrics_holdout"), dict):
        meta["metrics"] = dict(meta["metrics_holdout"])

    return meta


def ensure_model_sidecar_meta(model_path: str | Path, *, model: Any | None = None) -> tuple[Path | None, dict[str, Any]]:
    p = Path(model_path).expanduser().resolve()
    meta_path = model_sidecar_meta_path(p)
    meta = read_sidecar_model_meta(p)
    if meta:
        return meta_path, meta

    payload = model
    if payload is None:
        try:
            payload = joblib.load(p)
        except Exception as e:
            log.warning("Cannot load model payload %s to synthesize metadata: %s", p, e)
            return None, {}

    meta = synthesize_sidecar_model_meta(payload)
    if not meta:
        return None, {}

    try:
        write_sidecar_model_meta(p, meta)
    except Exception as e:
        log.warning("Cannot write recovered metadata %s: %s", meta_path, e)
        return None, meta

    return meta_path, meta


def write_sidecar_model_meta(model_path: str | Path, meta: dict[str, Any]) -> Path:
    meta_path = model_sidecar_meta_path(model_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path

def _sha1_file(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def load_model_with_meta(model_path: str | Path) -> LoadedModel:
    """
    Load model (joblib/pkl) with sidecar metadata if present.
    Metadata fields (optional): trained_features: List[str], model_classes: List[str]
    """
    p = Path(model_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model path not found: {p}")
    model = joblib.load(p)
    sha1 = _sha1_file(p)
    size = p.stat().st_size

    meta_path, meta = ensure_model_sidecar_meta(p, model=model)
    if not meta and p.is_file():
        fallback_meta = p.parent / "model_meta.json"
        if fallback_meta.exists():
            try:
                loaded = json.loads(fallback_meta.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Cannot read metadata %s: %s", fallback_meta, e)
            else:
                if isinstance(loaded, dict):
                    meta = loaded
                    meta_path = fallback_meta

    predictor = model.get("model") if isinstance(model, dict) and "model" in model else model

    # normalize class keys across metadata producers
    if not meta.get("model_classes"):
        if isinstance(meta.get("classes"), list):
            meta["model_classes"] = list(meta.get("classes"))
        elif hasattr(predictor, "classes_"):
            meta["model_classes"] = list(getattr(predictor, "classes_"))

    if not meta.get("trained_features"):
        if isinstance(meta.get("features"), list):
            meta["trained_features"] = list(meta.get("features"))
        elif hasattr(predictor, "feature_names_in_"):
            meta["trained_features"] = list(getattr(predictor, "feature_names_in_"))

    version_warning = build_sklearn_version_warning(meta, model_path=p)
    if version_warning:
        log.warning(version_warning)

    if meta_path is not None:
        meta.setdefault("meta_path", str(meta_path))

    log.info(
        "Model loaded | path=%s | size=%.1f kB | sha1=%s | classes=%s | n_features=%s",
        str(p), size / 1024.0, sha1, meta.get("model_classes"), len(meta.get("trained_features", [])),
    )

    return LoadedModel(model=model, meta=meta, path=p, sha1=sha1, size=size, version_warning=version_warning)

def save_model_with_meta(
    model,
    X_train_df: pd.DataFrame,
    out_pkl_path: str | Path,
    *,
    model_name: str,
    metrics: dict | None = None,
    class_to_dir: dict[int | str, str] | None = None,
) -> tuple[str, str]:
    """
    Uloží model (.pkl) + metadata (.json) vedle sebe.
    - trained_features: jména featur v přesném pořadí
    - model_classes: z modelu (classes_) nebo z class_to_dir mapy
    - metrics: např. {"sharpe": 0.83}

    Vrací: (cesta_k_pkl, cesta_k_meta_json)
    """
    p = Path(out_pkl_path).with_suffix(".pkl")
    p.parent.mkdir(parents=True, exist_ok=True)

    # 1) dump modelu
    joblib.dump(model, p)

    # 2) featury – nejdřív z DF (spolehlivé), teprve pak z modelu
    feats = None
    if isinstance(X_train_df, pd.DataFrame):
        feats = [str(c) for c in list(X_train_df.columns)]
    elif hasattr(model, "feature_names_in_") and getattr(model, "feature_names_in_", None) is not None:
        feats = [str(c) for c in list(model.feature_names_in_)]
    else:
        feats = []

    # 3) třídy: priorita classes_ ze scikit, jinak z class_to_dir
    classes = None
    if hasattr(model, "classes_"):
        classes = [str(c) for c in list(model.classes_)]
    elif class_to_dir:
        # např. {0:"SHORT", 1:"LONG"} -> ["SHORT","LONG"] podle indexu
        try:
            keys = sorted(class_to_dir.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        except Exception:
            keys = class_to_dir.keys()
        classes = [str(class_to_dir[k]).upper() for k in keys]

    meta = {
        "model_name": model_name,
        "created_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "trained_features": feats or [],
        "model_classes": classes or [],
        "metrics": metrics or {},
        "sklearn_version": runtime_sklearn_version(),
        "python_version": runtime_python_version(),
        "schema_version": 1,
    }
    p_meta = p.with_name(p.stem + "_meta.json")
    p_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p), str(p_meta)
