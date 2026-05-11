from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

LabelMode = Literal[
    "auto",
    "binary_01",
    "binary_signed",
    "ternary_signed",
    "ternary_mapped",
]


TERNARY_SIGNED_VALUES = {-1, 0, 1}
TERNARY_MAPPED_VALUES = {0, 1, 2}
BINARY_SIGNED_VALUES = {-1, 1}
BINARY_01_VALUES = {0, 1}


def _finite_label_values(*arrays) -> set[int]:
    values: set[int] = set()
    for arr in arrays:
        if arr is None:
            continue
        series = pd.Series(arr)
        series = series[series.notna()]
        if series.empty:
            continue
        try:
            values.update(int(x) for x in np.asarray(series, dtype=int).tolist())
        except Exception:
            continue
    return values


def infer_label_mode(*arrays, preferred: LabelMode = "auto") -> LabelMode:
    if preferred != "auto":
        return preferred

    values = _finite_label_values(*arrays)
    if not values:
        return "binary_01"
    if values.issubset(TERNARY_MAPPED_VALUES) and 2 in values:
        return "ternary_mapped"
    if values.issubset(TERNARY_SIGNED_VALUES) and ((-1 in values and 0 in values) or (0 in values and 1 in values)):
        return "ternary_signed"
    if values.issubset(BINARY_SIGNED_VALUES):
        return "binary_signed"
    if values.issubset(BINARY_01_VALUES):
        return "binary_01"
    if values.issubset(TERNARY_SIGNED_VALUES):
        return "ternary_signed"
    return "binary_signed" if any(v < 0 for v in values) else "binary_01"


def is_ternary_label_mode(label_mode: LabelMode) -> bool:
    return label_mode in {"ternary_signed", "ternary_mapped"}


def mapped_ternary_to_signed(arr) -> np.ndarray:
    a = np.asarray(arr).astype(int)
    return np.where(a == 0, -1, np.where(a == 1, 0, 1)).astype(int)


def signed_ternary_to_mapped(arr) -> np.ndarray:
    a = np.asarray(arr).astype(int)
    return np.where(a < 0, 0, np.where(a > 0, 2, 1)).astype(int)


def normalize_labels_for_mode(arr, label_mode: LabelMode) -> np.ndarray:
    a = np.asarray(arr).astype(int)
    if label_mode == "ternary_mapped":
        return mapped_ternary_to_signed(a)
    if label_mode == "ternary_signed":
        return np.where(a > 0, 1, np.where(a < 0, -1, 0)).astype(int)
    if label_mode == "binary_signed":
        return np.where(a > 0, 1, np.where(a < 0, -1, 0)).astype(int)
    if label_mode == "binary_01":
        return (a > 0).astype(int)
    resolved = infer_label_mode(a)
    return normalize_labels_for_mode(a, resolved)


def normalize_target_for_mode(values, target_mode: Literal["binary", "ternary"]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if target_mode == "ternary":
        return np.where(arr > 0.0, 1, np.where(arr < 0.0, -1, 0)).astype(int)
    return (arr > 0.0).astype(int)


def ternary_predict_mapped(
    prob_short: np.ndarray,
    prob_long: np.ndarray,
    thr_short: float,
    thr_long: float,
) -> np.ndarray:
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


def ternary_predict_signed(
    prob_short: np.ndarray,
    prob_long: np.ndarray,
    thr_short: float,
    thr_long: float,
) -> np.ndarray:
    return mapped_ternary_to_signed(ternary_predict_mapped(prob_short, prob_long, thr_short, thr_long))