from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_EXIT_POLICY = "hold_until_opposite"


def _normalize_direction_label(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    aliases = {
        "LONG": "LONG",
        "BUY": "LONG",
        "UP": "LONG",
        "SHORT": "SHORT",
        "SELL": "SHORT",
        "DOWN": "SHORT",
        "HOLD": "HOLD",
        "FLAT": "HOLD",
        "NONE": "HOLD",
        "NEUTRAL": "HOLD",
    }
    return aliases.get(text)


def infer_label_map_from_classes(classes: Any, base_map: dict[Any, Any] | None = None) -> dict[int, str]:
    inferred: dict[int, str] = {}
    numeric_classes: list[int] = []

    if classes is not None:
        for cls in list(classes):
            try:
                numeric_classes.append(int(cls))
            except Exception:
                continue

    uniq = sorted(set(numeric_classes))
    if uniq:
        values = set(uniq)
        if values == {-1, 1}:
            inferred = {-1: "SHORT", 1: "LONG"}
        elif values == {0, 1}:
            inferred = {0: "SHORT", 1: "LONG"}
        elif values == {-1, 0, 1}:
            inferred = {-1: "SHORT", 0: "HOLD", 1: "LONG"}
        elif values == {0, 1, 2}:
            inferred = {0: "SHORT", 1: "HOLD", 2: "LONG"}
        else:
            if len(uniq) >= 2:
                inferred[uniq[0]] = "SHORT"
                inferred[uniq[-1]] = "LONG"
            for value in uniq[1:-1]:
                inferred[value] = "HOLD"

    if not inferred:
        inferred = {0: "SHORT", 1: "LONG"}

    if isinstance(base_map, dict):
        for key, value in base_map.items():
            try:
                inferred[int(key)] = str(value).upper()
            except Exception:
                continue

    return inferred


def extract_directional_probabilities(
    raw_proba: Any,
    classes: Any,
    *,
    label_map: dict[Any, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba = np.asarray(raw_proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)

    if proba.ndim != 2 or proba.shape[1] <= 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty

    prob_short = np.zeros(proba.shape[0], dtype=float)
    prob_hold = np.zeros(proba.shape[0], dtype=float)
    prob_long = np.zeros(proba.shape[0], dtype=float)

    classes_list = list(classes) if classes is not None else []
    if len(classes_list) != proba.shape[1]:
        prob_short = np.asarray(proba[:, 0], dtype=float)
        if proba.shape[1] >= 3:
            prob_hold = np.asarray(proba[:, 1], dtype=float)
            prob_long = np.asarray(proba[:, 2], dtype=float)
        elif proba.shape[1] >= 2:
            prob_long = np.asarray(proba[:, 1], dtype=float)
        return prob_short, prob_hold, prob_long

    resolved_map = infer_label_map_from_classes(classes_list, base_map=label_map)
    for idx, cls in enumerate(classes_list):
        direction = None
        try:
            direction = _normalize_direction_label(resolved_map.get(int(cls), ""))
        except Exception:
            direction = None
        if direction is None:
            direction = _normalize_direction_label(resolved_map.get(cls, ""))
        if direction is None:
            direction = _normalize_direction_label(cls)

        if direction == "SHORT":
            prob_short += proba[:, idx]
        elif direction == "LONG":
            prob_long += proba[:, idx]
        elif direction == "HOLD":
            prob_hold += proba[:, idx]

    return prob_short, prob_hold, prob_long


def pick_ternary_direction_from_raw_proba(
    raw_proba: Any,
    classes: Any,
    *,
    short_threshold: float,
    long_threshold: float,
    label_map: dict[Any, Any] | None = None,
) -> tuple[str | None, float]:
    prob_short, prob_hold, prob_long = extract_directional_probabilities(
        raw_proba,
        classes,
        label_map=label_map,
    )
    if prob_short.size <= 0 or prob_long.size <= 0:
        return None, 0.0

    signal = ternary_proba_to_signal(
        prob_short,
        prob_long,
        short_threshold,
        long_threshold,
    )
    signal_value = int(signal[0]) if signal.size else 0
    if signal_value > 0:
        return "LONG", float(prob_long[0])
    if signal_value < 0:
        return "SHORT", float(prob_short[0])
    return "FLAT", float(max(prob_long[0], prob_short[0], prob_hold[0] if prob_hold.size else 0.0))


def normalize_exit_policy_name(value: Any, *, default: str = DEFAULT_EXIT_POLICY) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": default,
        "default": DEFAULT_EXIT_POLICY,
        "legacy": "legacy_flat_exit",
        "legacy_flat": "legacy_flat_exit",
        "legacy_flat_exit": "legacy_flat_exit",
        "flat_on_neutral": "legacy_flat_exit",
        "flat_on_weak_signal": "legacy_flat_exit",
        "hold_until_opposite": "hold_until_opposite",
        "hold-opposite": "hold_until_opposite",
        "hold_opposite": "hold_until_opposite",
        "reverse_on_opposite": "hold_until_opposite",
    }
    return aliases.get(text, default)


def resolve_exit_policy_setting(source: Any, *, default: str = DEFAULT_EXIT_POLICY) -> str:
    if isinstance(source, dict):
        nested = source.get("user_settings")
        if isinstance(nested, dict):
            nested_policy = nested.get("exit_policy") or nested.get("live_exit_policy")
            if nested_policy is not None:
                return normalize_exit_policy_name(nested_policy, default=default)
        direct_policy = source.get("exit_policy") or source.get("live_exit_policy")
        if direct_policy is not None:
            return normalize_exit_policy_name(direct_policy, default=default)
    return normalize_exit_policy_name(source, default=default)


@dataclass(frozen=True)
class LivePolicyDecision:
    proposal: str | None
    final_signal: str | None
    reason: str
    close_reason: str | None = None


def ternary_proba_to_signal(prob_short, prob_long, thr_short: float, thr_long: float) -> np.ndarray:
    ps = np.asarray(prob_short, dtype=float)
    pl = np.asarray(prob_long, dtype=float)
    n = min(ps.size, pl.size)
    if n <= 0:
        return np.asarray([], dtype=int)
    ps = ps[:n]
    pl = pl[:n]
    out = np.zeros(n, dtype=int)
    out[pl >= float(thr_long)] = 1
    out[ps >= float(thr_short)] = -1
    both = (pl >= float(thr_long)) & (ps >= float(thr_short))
    if both.any():
        out[both] = np.where(pl[both] >= ps[both], 1, -1).astype(int)
    return out


def normalize_signal_array(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=object)
    out = np.zeros(a.shape, dtype=float)
    num_mask = np.array([isinstance(x, (int, float, np.number)) for x in a], dtype=bool)
    out[num_mask] = np.sign(a[num_mask].astype(float))
    txt = np.char.lower(a.astype(str))
    out[(txt == "long") | (txt == "buy") | (txt == "up") | (txt == "1") | (txt == "+1")] = 1.0
    out[(txt == "short") | (txt == "sell") | (txt == "down") | (txt == "-1")] = -1.0
    return out


def apply_confidence_entry_threshold(raw_pred: Any, confidence: Any, threshold: float) -> np.ndarray:
    arr = np.asarray(raw_pred).copy()
    conf = np.asarray(confidence).reshape(-1)
    mask_low = conf < float(threshold)
    try:
        arr[mask_low] = 0
    except Exception:
        tmp = np.array(arr, dtype=object)
        tmp[mask_low] = 0
        arr = tmp
    return arr


def apply_exit_confidence_threshold(y_pred: Any, confidence: Any, exit_thr: float) -> np.ndarray:
    arr = np.asarray(y_pred).copy()
    conf = np.asarray(confidence).reshape(-1)
    mask_low = conf < float(exit_thr)
    open_pos = np.abs(normalize_signal_array(arr)) > 0.5
    arr[mask_low & open_pos] = 0
    return arr


def apply_stateful_hold_until_opposite(
    raw_pred: Any,
    confidence: Any,
    entry_threshold: float,
    exit_threshold: float,
) -> np.ndarray:
    raw = normalize_signal_array(raw_pred).astype(int, copy=False)
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    n = min(raw.size, conf.size)
    if n <= 0:
        return np.asarray([], dtype=int)

    out = np.zeros(n, dtype=int)
    position = 0
    entry_thr = float(entry_threshold)
    exit_thr = float(exit_threshold)

    for idx in range(n):
        signal = int(raw[idx])
        conf_now = float(conf[idx])

        if position == 0:
            if signal != 0 and conf_now >= entry_thr:
                position = signal
            out[idx] = position
            continue

        if signal == -position and conf_now >= exit_thr:
            position = signal

        out[idx] = position

    return out


def apply_entry_exit_thresholds(
    raw_pred: Any,
    confidence: Any,
    entry_threshold: float,
    exit_threshold: float,
    *,
    exit_policy: str = DEFAULT_EXIT_POLICY,
) -> np.ndarray:
    policy = normalize_exit_policy_name(exit_policy)
    if policy == "legacy_flat_exit":
        out = apply_confidence_entry_threshold(raw_pred, confidence, entry_threshold)
        out = normalize_signal_array(out)
        if float(exit_threshold) > 0.0:
            out = apply_exit_confidence_threshold(out, confidence, exit_threshold)
            out = normalize_signal_array(out)
        return np.asarray(out)

    return apply_stateful_hold_until_opposite(raw_pred, confidence, entry_threshold, exit_threshold)


def build_live_proposal(ma_direction: str, model_direction: str, use_ma_alignment: bool) -> str | None:
    l0 = str(ma_direction or "FLAT").upper()
    l1 = str(model_direction or "FLAT").upper()
    if use_ma_alignment:
        if l0 == "FLAT":
            return l1 if l1 in {"LONG", "SHORT"} else None
        return l1 if l1 == l0 else None
    return l1 if l1 in {"LONG", "SHORT"} else None


def evaluate_live_policy(
    ma_direction: str,
    model_direction: str,
    use_ma_alignment: bool,
    conf_min: float,
    live_position: int,
    entry_threshold: float,
    exit_threshold: float,
    *,
    block_entry: bool = False,
    exit_policy: str = DEFAULT_EXIT_POLICY,
) -> LivePolicyDecision:
    proposal = build_live_proposal(ma_direction, model_direction, use_ma_alignment)
    current_dir = "LONG" if int(live_position) > 0 else "SHORT" if int(live_position) < 0 else None
    confidence = float(conf_min)
    policy = normalize_exit_policy_name(exit_policy)

    if int(live_position) == 0:
        if proposal not in {"LONG", "SHORT"}:
            return LivePolicyDecision(proposal=proposal, final_signal=None, reason="flat_no_entry_signal")
        if block_entry:
            return LivePolicyDecision(proposal=proposal, final_signal=None, reason="entry_blocked")
        if confidence >= float(entry_threshold):
            return LivePolicyDecision(proposal=proposal, final_signal=proposal, reason="entry_confirmed")
        return LivePolicyDecision(proposal=proposal, final_signal=None, reason="entry_low_confidence")

    if policy == "legacy_flat_exit":
        if proposal == current_dir and confidence >= float(exit_threshold):
            return LivePolicyDecision(proposal=proposal, final_signal=current_dir, reason="hold_confirmed")
        if proposal == current_dir:
            return LivePolicyDecision(
                proposal=proposal,
                final_signal=None,
                reason="exit_low_confidence",
                close_reason="low_confidence",
            )
        if proposal in {"LONG", "SHORT"} and proposal != current_dir:
            return LivePolicyDecision(
                proposal=proposal,
                final_signal=None,
                reason="exit_opposite_signal",
                close_reason="opposite_signal",
            )
        return LivePolicyDecision(
            proposal=proposal,
            final_signal=None,
            reason="exit_no_signal",
            close_reason="signal_missing",
        )

    if proposal in {"LONG", "SHORT"} and proposal != current_dir and confidence >= float(exit_threshold):
        return LivePolicyDecision(
            proposal=proposal,
            final_signal=proposal,
            reason="flip_confirmed",
            close_reason="opposite_signal",
        )
    if proposal in {"LONG", "SHORT"} and proposal != current_dir:
        return LivePolicyDecision(proposal=proposal, final_signal=current_dir, reason="hold_opposite_unconfirmed")
    if proposal == current_dir:
        return LivePolicyDecision(proposal=proposal, final_signal=current_dir, reason="hold_same_signal")
    return LivePolicyDecision(proposal=proposal, final_signal=current_dir, reason="hold_no_signal")


def apply_live_hysteresis(
    proposal: str | None,
    conf_min: float,
    live_position: int,
    entry_threshold: float,
    exit_threshold: float,
    *,
    block_entry: bool = False,
    exit_policy: str = DEFAULT_EXIT_POLICY,
) -> str | None:
    decision = evaluate_live_policy(
        ma_direction="FLAT",
        model_direction=str(proposal or "FLAT"),
        use_ma_alignment=False,
        conf_min=conf_min,
        live_position=live_position,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        block_entry=block_entry,
        exit_policy=exit_policy,
    )
    return decision.final_signal