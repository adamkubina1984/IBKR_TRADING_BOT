from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def extract_X_y_eval(prepared: Any) -> tuple[pd.DataFrame | np.ndarray, np.ndarray | None]:
    if isinstance(prepared, (tuple, list)):
        if len(prepared) >= 2:
            return prepared[0], prepared[1]
        return prepared[0], None
    if isinstance(prepared, dict):
        X = _first_present(prepared, "X", "features", "data", "df")
        y = _first_present(prepared, "y", "target", "y_true")
        if X is None:
            raise ValueError("V dictu chybi klic 'X'/'features'/'data'.")
        return X, y
    if isinstance(prepared, pd.DataFrame):
        X, y = prepared, None
        for candidate in ["target", "y", "label"]:
            if candidate in prepared.columns:
                y = prepared[candidate].values
                X = prepared.drop(columns=[candidate])
                break
        return X, y
    if isinstance(prepared, np.ndarray):
        return prepared, None
    raise ValueError("Neocekavany navratovy typ z prepare_dataset_with_targets(df).")