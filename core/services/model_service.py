from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

log = logging.getLogger(__name__)

@dataclass
class LoadedModel:
    model: Any
    meta: dict
    path: Path
    sha1: str
    size: int

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

    # metadata file search: <stem>_meta.json or sibling model_meta.json
    meta = {}
    candidates = [p.with_name(p.stem + "_meta.json")]
    if p.is_file():
        candidates.append(p.parent / "model_meta.json")
    for cand in candidates:
        if cand.exists():
            try:
                meta = json.loads(cand.read_text(encoding="utf-8"))
                break
            except Exception as e:
                log.warning("Cannot read metadata %s: %s", cand, e)

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

    log.info(
        "Model loaded | path=%s | size=%.1f kB | sha1=%s | classes=%s | n_features=%s",
        str(p), size / 1024.0, sha1, meta.get("model_classes"), len(meta.get("trained_features", [])),
    )

    return LoadedModel(model=model, meta=meta, path=p, sha1=sha1, size=size)

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
        "schema_version": 1,
    }
    p_meta = p.with_name(p.stem + "_meta.json")
    p_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p), str(p_meta)

