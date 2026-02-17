from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def save_snapshot(df: pd.DataFrame, out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Snapshot saved: %s (%d rows, %d cols)", p, len(df), len(df.columns))
    return p

def proba_histogram(proba: np.ndarray, bins: int = 10) -> dict:
    hist, edges = np.histogram(proba, bins=bins, range=(0.0, 1.0))
    return {"edges": edges.tolist(), "hist": hist.tolist()}
