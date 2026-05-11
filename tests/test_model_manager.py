from pathlib import Path

import joblib
import numpy as np

from ibkr_trading_bot.gui.tab_model_manager import ModelRecord, _record_rank_key
from ibkr_trading_bot.gui.tab_model_manager import discover_models


class DummyManagerPredictor:
    def __init__(self):
        self.classes_ = np.array([0, 1, 2])
        self.feature_names_in_ = np.array(["f_signal", "f_volume"], dtype=object)


def test_record_rank_key_prefers_num_trades_over_trade_detail_list(tmp_path):
    record = ModelRecord(
        model_path=tmp_path / "model.pkl",
        meta_path=None,
        sha1="abc",
        created="2026-04-20T10:00:00",
        metrics={
            "profit_net": 12.0,
            "sharpe": 1.5,
            "num_trades": 17,
            "trades": [{"pnl": 1.0}, {"pnl": -0.5}],
        },
        features_n=3,
        classes=["0", "1", "2"],
        top_feature="f_signal",
        file_size=100,
        file_mtime_ns=1,
    )

    profit, sharpe, trades, _ = _record_rank_key(record)

    assert profit == 12.0
    assert sharpe == 1.5
    assert trades == 17.0


def test_discover_models_recovers_missing_sidecar_from_embedded_payload(tmp_path):
    model_path = tmp_path / "refresh_model.pkl"
    joblib.dump(
        {
            "model": DummyManagerPredictor(),
            "created_at": "20260421_131050",
            "features": ["f_signal", "f_volume"],
            "metrics_holdout": {"profit_net": 33.0, "sharpe": 1.2},
            "ternary_threshold_short": 0.35,
            "ternary_threshold_long": 0.65,
        },
        model_path,
    )

    records = discover_models(tmp_path)
    meta_path = model_path.with_name(model_path.stem + "_meta.json")

    assert len(records) == 1
    assert meta_path.exists()
    assert records[0].meta_path == meta_path
    assert records[0].features_n == 2
    assert records[0].classes == ["0", "1", "2"]
    assert records[0].metrics["profit_net"] == 33.0