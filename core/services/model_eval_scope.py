from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def tail_rows_eval(obj: Any, n_rows: int):
    if obj is None:
        return None
    n = int(max(0, n_rows))
    if isinstance(obj, pd.DataFrame):
        return obj.tail(n).reset_index(drop=True)
    if isinstance(obj, pd.Series):
        return obj.tail(n).reset_index(drop=True)
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr
    return arr[-n:] if n < arr.shape[0] else arr


def infer_holdout_bars_from_metadata(meta: dict[str, Any], n_rows: int) -> int | None:
    n = int(max(0, n_rows))
    if n <= 0:
        return None
    try:
        n_hold = int(meta.get("n_holdout_bars", 0))
        if n_hold > 0:
            return int(min(n, n_hold))
    except Exception:
        pass
    hold_sel = meta.get("holdout_selection") if isinstance(meta, dict) else None
    if isinstance(hold_sel, dict):
        try:
            applied = int(hold_sel.get("applied_bars", 0))
            if applied > 0:
                return int(min(n, applied))
        except Exception:
            pass
        try:
            pct = float(hold_sel.get("requested_pct"))
            if np.isfinite(pct) and pct > 0.0:
                calc = int(round(float(n) * float(np.clip(pct, 0.0, 0.95))))
                if calc > 0:
                    return int(min(n, calc))
        except Exception:
            pass
    return None


def apply_eval_scope(
    X: pd.DataFrame | np.ndarray,
    y_true: np.ndarray | None,
    df_for_metrics: pd.DataFrame | None,
    scope_mode: str,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray | None, pd.DataFrame | None, dict[str, Any]]:
    lengths: list[int] = []
    for obj in (X, y_true, df_for_metrics):
        if obj is None:
            continue
        try:
            lengths.append(int(len(obj)))
        except Exception:
            pass
    if not lengths:
        raise ValueError("Nelze urcit delku datasetu pro evaluaci.")

    n_base = int(max(0, min(lengths)))
    if n_base <= 0:
        raise ValueError("Dataset pro evaluaci je prazdny.")

    X_aligned = tail_rows_eval(X, n_base)
    y_aligned = tail_rows_eval(y_true, n_base) if y_true is not None else None
    df_aligned = tail_rows_eval(df_for_metrics, n_base) if df_for_metrics is not None else None

    mode = scope_mode if scope_mode in {"holdout", "full"} else "holdout"
    if mode == "holdout":
        n_hold = infer_holdout_bars_from_metadata(metadata or {}, n_base)
        if n_hold is not None and n_hold > 0:
            n_eval = int(min(n_base, n_hold))
            X_eval = tail_rows_eval(X_aligned, n_eval)
            y_eval = tail_rows_eval(y_aligned, n_eval) if y_aligned is not None else None
            df_eval = tail_rows_eval(df_aligned, n_eval) if df_aligned is not None else None
        else:
            n_eval = n_base
            X_eval, y_eval, df_eval = X_aligned, y_aligned, df_aligned
    else:
        n_eval = n_base
        X_eval, y_eval, df_eval = X_aligned, y_aligned, df_aligned

    return X_eval, y_eval, df_eval, {
        "mode": mode,
        "applied_rows": int(n_eval),
        "total_rows": int(n_base),
    }