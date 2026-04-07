from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FeatureStabilityFilterResult:
    kept_features: list[str]
    removed_features: list[str]
    scores: dict[str, float]
    filter_applied: bool
    fallback_reason: str | None


def _coerce_float(value: Any, *, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _score_single_feature(stats: Mapping[str, Any] | None) -> float:
    if not isinstance(stats, Mapping):
        return 0.0

    mean_val = _coerce_float(stats.get("mean"))
    std_val = _coerce_float(stats.get("std"))
    folds_present = _coerce_int(stats.get("folds_present"), default=0)

    if not math.isfinite(mean_val) or mean_val <= 0.0:
        return 0.0
    if not math.isfinite(std_val) or std_val < 0.0:
        return 0.0
    if folds_present < 2:
        return 0.0

    score = 1.0 - (std_val / max(mean_val, 1e-8))
    return float(min(1.0, max(0.0, score)))


def compute_feature_stability_score(
    feature_stability: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, float]:
    """
    Compute deterministic per-feature stability scores.

    Formula:
        score = clip(1.0 - (std / max(mean, 1e-8)), 0.0, 1.0)

    This favors features whose average importance stays large relative to their
    fold-to-fold variation. The trade-off is intentionally conservative:
    features with tiny or non-positive mean importance, non-finite dispersion,
    or fewer than two contributing folds are forced to 0.0 so unreliable
    stability estimates never get promoted by the auto-filter.
    """

    if not isinstance(feature_stability, Mapping):
        return {}

    return {
        str(feature_name): float(_score_single_feature(stats))
        for feature_name, stats in feature_stability.items()
    }


def evaluate_feature_stability_filter(
    feature_stability: Mapping[str, Mapping[str, Any]] | None,
    trained_features: Sequence[str] | None,
    threshold: float,
    *,
    logger: logging.Logger | None = None,
) -> FeatureStabilityFilterResult:
    feature_names = [str(feature_name) for feature_name in (trained_features or [])]
    if not feature_names:
        return FeatureStabilityFilterResult([], [], {}, False, None)

    threshold_value = _coerce_float(threshold, default=0.0)
    if not math.isfinite(threshold_value):
        threshold_value = 0.0
    threshold_value = float(min(1.0, max(0.0, threshold_value)))

    stability_map = feature_stability if isinstance(feature_stability, Mapping) else {}
    score_map_raw = compute_feature_stability_score(stability_map)
    score_map = {
        feature_name: float(score_map_raw.get(feature_name, 0.0))
        for feature_name in feature_names
    }

    kept_features: list[str] = []
    removed_features: list[str] = []

    for feature_name in feature_names:
        score = float(score_map.get(feature_name, 0.0))
        stats = stability_map.get(feature_name) if isinstance(stability_map, Mapping) else None
        mean_val = _coerce_float((stats or {}).get("mean"))
        std_val = _coerce_float((stats or {}).get("std"))
        folds_present = _coerce_int((stats or {}).get("folds_present"), default=0)

        if score < threshold_value:
            removed_features.append(feature_name)
            if logger is not None:
                logger.info(
                    "Removing unstable feature '%s' (score=%.4f, threshold=%.4f, mean=%s, std=%s, folds_present=%s)",
                    feature_name,
                    score,
                    threshold_value,
                    mean_val,
                    std_val,
                    folds_present,
                )
            continue
        kept_features.append(feature_name)

    if removed_features and not kept_features:
        fallback_reason = (
            f"all_features_below_stability_threshold(threshold={threshold_value:.4f})"
        )
        if logger is not None:
            logger.warning(
                "Feature stability filter removed all %d trained features at threshold %.4f; keeping original feature set.",
                len(feature_names),
                threshold_value,
            )
        return FeatureStabilityFilterResult(
            kept_features=list(feature_names),
            removed_features=[],
            scores=score_map,
            filter_applied=False,
            fallback_reason=fallback_reason,
        )

    return FeatureStabilityFilterResult(
        kept_features=kept_features,
        removed_features=removed_features,
        scores=score_map,
        filter_applied=bool(removed_features),
        fallback_reason=None,
    )


def filter_unstable_features(
    feature_stability: Mapping[str, Mapping[str, Any]] | None,
    trained_features: Sequence[str] | None,
    threshold: float,
    *,
    logger: logging.Logger | None = None,
) -> list[str]:
    return evaluate_feature_stability_filter(
        feature_stability,
        trained_features,
        threshold,
        logger=logger,
    ).kept_features
