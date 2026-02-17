# presets.py
from __future__ import annotations

PRESETS_BY_TF = {
    "5 min": {
        "triple_barrier": {"K": 24, "tp_atr": 1.2, "sl_atr": 0.8},
        "hysteresis": {"entry_thr": 0.62, "exit_thr": 0.47},
        "trailing": {"activate_atr": 1.2, "step_atr": 0.6},
        "zigzag_atr": 0.6,
        "rounds": {"grid": [1, 5], "tol_atr": 0.15},
    },
    "15 min": {
        "triple_barrier": {"K": 20, "tp_atr": 1.5, "sl_atr": 1.0},
        "hysteresis": {"entry_thr": 0.60, "exit_thr": 0.45},
        "trailing": {"activate_atr": 1.6, "step_atr": 0.8},
        "zigzag_atr": 0.8,
        "rounds": {"grid": [2.5, 10], "tol_atr": 0.20},
    },
    "30 min": {
        "triple_barrier": {"K": 16, "tp_atr": 1.8, "sl_atr": 1.2},
        "hysteresis": {"entry_thr": 0.60, "exit_thr": 0.45},
        "trailing": {"activate_atr": 2.0, "step_atr": 1.0},
        "zigzag_atr": 1.0,
        "rounds": {"grid": [5, 25], "tol_atr": 0.25},
    },
    "1 hour": {
        "triple_barrier": {"K": 12, "tp_atr": 2.2, "sl_atr": 1.4},
        "hysteresis": {"entry_thr": 0.58, "exit_thr": 0.43},
        "trailing": {"activate_atr": 2.5, "step_atr": 1.2},
        "zigzag_atr": 1.2,
        "rounds": {"grid": [10, 50], "tol_atr": 0.30},
    },
}
