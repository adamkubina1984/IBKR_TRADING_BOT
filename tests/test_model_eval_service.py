import joblib
import numpy as np
import pandas as pd
import pytest

from ibkr_trading_bot.core.services import model_eval_service as model_eval_runtime
from ibkr_trading_bot.core.services.model_service import write_sidecar_model_meta


class DummyEvalPredictor:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def test_model_eval_service_declares_expected_facade_exports():
    expected_exports = {
        "AutoThresholdPayload",
        "DatasetService",
        "EvaluationPayload",
        "EvaluationService",
        "LoadedPredictor",
        "PreparedEvaluationData",
        "apply_eval_scope",
        "build_tab5_holdout_ranking_payload",
        "extract_X_y_eval",
        "extract_predictor_from_object",
        "get_tab5_holdout_base_payload",
        "get_tab5_holdout_optimized_payload",
        "get_tab5_holdout_ranking",
        "is_tab5_holdout_ranking_stale",
        "load_predictor_with_merged_meta",
        "prepared_evaluation_cache_key",
        "recalculate_metrics_from_predictions",
        "run_auto_threshold_search_from_context",
        "run_model_evaluation",
        "set_tab5_holdout_ranking",
    }

    assert expected_exports.issubset(set(model_eval_runtime.__all__))
    for name in expected_exports:
        assert hasattr(model_eval_runtime, name)


def test_load_prepared_evaluation_data_uses_metadata_driven_dataset_service(monkeypatch, tmp_path):
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
    calls = []

    def fake_prepare_from_csv(self, path, **kwargs):
        calls.append((path, kwargs))
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-03-18T10:00:00Z", "2026-03-18T10:05:00Z"], utc=True),
                "close": [100.0, 101.0],
                "feat_a": [1.0, 2.0],
                "feat_b": [3.0, 4.0],
                "target": [-1, 1],
            }
        )

    monkeypatch.setattr(
        "ibkr_trading_bot.core.services.model_eval_service.DatasetService.prepare_from_csv",
        fake_prepare_from_csv,
    )

    prepared = model_eval_runtime.load_prepared_evaluation_data(
        csv_path,
        metadata={
            "label_mode": "ternary_mapped",
            "label_horizon_bars": 8,
            "label_take_profit_bps": 50.0,
            "label_stop_loss_bps": 40.0,
            "label_same_bar_policy": "sl",
        },
    )

    assert len(calls) == 1
    assert calls[0][0] == str(csv_path.resolve())
    assert calls[0][1]["labeling"] == "triple_barrier"
    assert calls[0][1]["target_mode"] == "ternary"
    assert calls[0][1]["horizon"] == 8
    assert calls[0][1]["take_profit_bps"] == pytest.approx(50.0)
    assert calls[0][1]["stop_loss_bps"] == pytest.approx(40.0)
    assert calls[0][1]["same_bar_policy"] == "sl"
    assert list(prepared.X_full.columns) == ["close", "feat_a", "feat_b"]
    assert prepared.y_true_full.tolist() == [-1, 1]


def test_prepared_evaluation_cache_key_tracks_metadata_contract(tmp_path):
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("timestamp,close\n2026-03-18T10:00:00Z,100\n", encoding="utf-8")

    ternary_key = model_eval_runtime.prepared_evaluation_cache_key(
        csv_path,
        {
            "label_mode": "ternary_mapped",
            "label_horizon_bars": 8,
            "label_take_profit_bps": 50.0,
            "label_stop_loss_bps": 40.0,
            "label_same_bar_policy": "sl",
        },
    )
    prepared_key = model_eval_runtime.prepared_evaluation_cache_key(
        csv_path,
        {
            "label_mode": "ternary_mapped",
        },
    )

    assert ternary_key[0] == str(csv_path.resolve())
    assert ternary_key[1:] == ("ternary_mapped", "triple_barrier", "ternary", 8, 50.0, 40.0, "sl")
    assert prepared_key[1:] == ("ternary_mapped", "prepared", "ternary", 12, 60.0, 40.0, "neutral")


def test_extract_predictor_from_object_reads_predictor_from_embedded_dict():
    predictor = DummyEvalPredictor()
    payload = {"model": predictor, "embedded_only": "embedded"}

    extracted, embedded_meta = model_eval_runtime.extract_predictor_from_object(payload)

    assert extracted is predictor
    assert embedded_meta is payload


def test_load_predictor_with_merged_meta_merges_embedded_and_sidecar_metadata(tmp_path):
    model_path = tmp_path / "dummy.pkl"
    joblib.dump(
        {
            "model": DummyEvalPredictor(),
            "embedded_only": "embedded",
            "trained_features": ["feat_a", "feat_b"],
        },
        model_path,
    )
    write_sidecar_model_meta(
        model_path,
        {
            "sidecar_only": "sidecar",
            "ternary_threshold_short": 0.45,
            "ternary_threshold_long": 0.55,
        },
    )

    loaded = model_eval_runtime.load_predictor_with_merged_meta(model_path)

    assert isinstance(loaded, model_eval_runtime.LoadedPredictor)
    assert hasattr(loaded.predictor, "predict")
    assert loaded.model_path == str(model_path.resolve())
    assert loaded.metadata["embedded_only"] == "embedded"
    assert loaded.metadata["sidecar_only"] == "sidecar"
    assert loaded.metadata["feature_contract"]["count"] == 2


def test_run_model_evaluation_uses_service_recalc_and_prepared_data(monkeypatch):
    class DummyProbModel:
        classes_ = np.array([0, 1, 2])

        def predict(self, X):
            return np.array([0, 2], dtype=int)

        def predict_proba(self, X):
            return np.array(
                [
                    [0.70, 0.20, 0.10],
                    [0.10, 0.20, 0.70],
                ],
                dtype=float,
            )

    captured: dict[str, Any] = {}

    def _fake_recalc_metrics(**kwargs):
        captured.update(kwargs)
        return np.array([-1, 1], dtype=int), {"profit_net": 4.0, "max_dd": -1.0, "trades": 2}

    monkeypatch.setattr(model_eval_runtime, "recalculate_metrics_from_predictions", _fake_recalc_metrics)

    prepared = model_eval_runtime.PreparedEvaluationData(
        data_path="dummy.csv",
        X_full=pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]}),
        y_true_full=np.array([-1, 1], dtype=int),
        df_for_metrics_full=pd.DataFrame(
            {"close": [100.0, 101.0], "feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]}
        ),
    )

    payload = model_eval_runtime.run_model_evaluation(
        model=DummyProbModel(),
        metadata={
            "trained_features": ["feat_a", "feat_b"],
            "ternary_threshold_short": 0.45,
            "ternary_threshold_long": 0.55,
            "class_to_dir": {0: "SHORT", 1: "HOLD", 2: "LONG"},
        },
        prepared_data=prepared,
        scope_mode="full",
        fee_per_trade=0.0,
        entry_threshold=0.6,
        exit_threshold=0.7,
    )

    assert payload.threshold_source == "model"
    assert payload.thr_short == pytest.approx(0.45)
    assert payload.thr_long == pytest.approx(0.55)
    assert payload.y_pred_raw.tolist() == [-1, 1]
    assert payload.y_pred_used.tolist() == [-1, 1]
    assert payload.results["profit_net"] == pytest.approx(4.0)
    assert captured["entry_threshold"] == pytest.approx(0.6)
    assert captured["exit_threshold"] == pytest.approx(0.7)
    assert captured["df_current"]["close"].tolist() == [100.0, 101.0]


def test_recalculate_metrics_from_predictions_rejects_empty_metric_result(monkeypatch):
    monkeypatch.setattr(
        "ibkr_trading_bot.core.services.model_eval_service.EvaluationService.calculate_metrics",
        lambda self, **kwargs: {},
    )

    with pytest.raises(ValueError, match="prazdny vysledek"):
        model_eval_runtime.recalculate_metrics_from_predictions(
            y_pred_raw=np.array([1, -1], dtype=int),
            confidence_arr=np.array([0.9, 0.8], dtype=float),
            y_true_current=np.array([1, -1], dtype=int),
            df_current=pd.DataFrame({"close": [100.0, 101.0]}),
            fee_per_trade=0.0,
            entry_threshold=0.3,
            exit_threshold=0.4,
        )


def test_extract_x_y_eval_reads_dict_features_and_y_true():
    X, y = model_eval_runtime.extract_X_y_eval(
        {
            "features": pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0]}),
            "y_true": np.array([1]),
        }
    )

    assert list(X.columns) == ["feat_a", "feat_b"]
    assert y.tolist() == [1]


def test_extract_x_y_eval_splits_target_column_from_dataframe():
    X, y = model_eval_runtime.extract_X_y_eval(
        pd.DataFrame({"feat_a": [1.0], "target": [-1]})
    )

    assert list(X.columns) == ["feat_a"]
    assert y.tolist() == [-1]


def test_coerce_features_for_model_eval_rejects_missing_trained_features():
    X = pd.DataFrame({"feat_a": [1.0], "other": [2.0]})

    with pytest.raises(ValueError, match="feat_b"):
        model_eval_runtime.coerce_features_for_model_eval(
            X,
            model=object(),
            metadata={"trained_features": ["feat_a", "feat_b"]},
        )


def test_align_x_for_model_eval_reorders_and_fills_missing_columns():
    class DummyModel:
        feature_names_in_ = np.array(["feat_b", "feat_a", "feat_c"])

    aligned = model_eval_runtime.align_X_for_model_eval(
        DummyModel(),
        pd.DataFrame({"feat_a": [1.5], "feat_b": [2.5]}),
    )

    assert list(aligned.columns) == ["feat_b", "feat_a", "feat_c"]
    assert aligned.iloc[0].tolist() == pytest.approx([2.5, 1.5, 0.0])


def test_resolve_ternary_thresholds_eval_falls_back_to_user_settings():
    thr_short, thr_long, threshold_source = model_eval_runtime.resolve_ternary_thresholds_eval(
        {
            "user_settings": {
                "ternary_threshold_short_eval": "0.42",
                "ternary_threshold_long_eval": 0.58,
            }
        }
    )

    assert thr_short == pytest.approx(0.42)
    assert thr_long == pytest.approx(0.58)
    assert threshold_source == "model"


def test_safe_close_series_eval_accepts_alternative_close_column_name():
    close_series = model_eval_runtime.safe_close_series_eval(
        pd.DataFrame({"Adj Close": [100, "101.5", None]})
    )

    assert close_series.tolist()[0:2] == pytest.approx([100.0, 101.5])
    assert pd.isna(close_series.iloc[2])


def test_get_tab5_holdout_base_payload_reads_legacy_flat_fields_with_metric_fallback():
    payload = model_eval_runtime.get_tab5_holdout_base_payload(
        {
            "base_entry_threshold": "0.34",
            "base_exit_threshold": 0.41,
            "base_profit_h": None,
            "base_max_dd_h": -5.0,
            "base_trades_h": None,
        },
        fallback_metrics={
            "profit_net": 12.5,
            "trades": 9,
        },
    )

    assert payload["entry_threshold"] == pytest.approx(0.34)
    assert payload["exit_threshold"] == pytest.approx(0.41)
    assert payload["profit_h"] == pytest.approx(12.5)
    assert payload["max_dd_h"] == pytest.approx(-5.0)
    assert payload["trades_h"] == pytest.approx(9.0)
    assert payload["metrics"]["profit_net"] == pytest.approx(12.5)


def test_apply_eval_scope_respects_holdout_metadata_and_alignment():
    X = pd.DataFrame({"feat": [10, 11, 12, 13, 14]})
    y = np.array([-1, 0, 1, 0, 1])
    df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})

    X_eval, y_eval, df_eval, scope = model_eval_runtime.apply_eval_scope(
        X,
        y,
        df,
        "holdout",
        {"holdout_selection": {"applied_bars": 2}},
    )

    assert scope == {"mode": "holdout", "applied_rows": 2, "total_rows": 5}
    assert X_eval["feat"].tolist() == [13, 14]
    assert y_eval.tolist() == [0, 1]
    assert df_eval["close"].tolist() == [103, 104]