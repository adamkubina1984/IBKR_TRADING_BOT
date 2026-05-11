import numpy as np
import pandas as pd
import pytest

from ibkr_trading_bot.model.train_models import (
    _align_X_for_estimator,
    _fallback_candidate_sort_key,
    _feature_names_for_estimator,
    _trade_count_preference_score,
)


class _DummyLgbLikeEstimator:
    def __init__(self):
        self.feature_name_ = ["feat_a", "feat_b", "feat_c"]


def test_feature_names_for_estimator_supports_lightgbm_feature_name_attr():
    est = _DummyLgbLikeEstimator()

    names = _feature_names_for_estimator(est)

    assert names == ["feat_a", "feat_b", "feat_c"]


def test_align_x_for_estimator_maps_ndarray_by_position_when_names_known():
    est = _DummyLgbLikeEstimator()
    x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=float)

    aligned = _align_X_for_estimator(est, x)

    assert isinstance(aligned, pd.DataFrame)
    assert list(aligned.columns) == ["feat_a", "feat_b", "feat_c"]
    np.testing.assert_allclose(aligned.to_numpy(dtype=float), x)


def test_fallback_candidate_sort_key_prefers_balanced_side_recall():
    short_biased = {"rec_short": 0.90, "rec_long": 0.05, "cheap_score": 1.50}
    balanced = {"rec_short": 0.48, "rec_long": 0.46, "cheap_score": 1.10}

    selected = max([short_biased, balanced], key=_fallback_candidate_sort_key)

    assert selected is balanced


def test_trade_count_preference_score_prefers_midrange_activity():
    assert _trade_count_preference_score(40) == 0.0
    assert _trade_count_preference_score(200) == pytest.approx(1.0)
    assert _trade_count_preference_score(380) < _trade_count_preference_score(200)
    assert _trade_count_preference_score(520) == 0.0
