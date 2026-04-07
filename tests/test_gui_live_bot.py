import numpy as np
import pandas as pd
from PySide6.QtWidgets import QWidget

from ._gui_test_helpers import DummyTernaryPredictor, StubCanvas


def test_project_loggers_do_not_propagate_to_root():
    from ibkr_trading_bot.core.utils.logging_setup import get_logger as get_core_logger
    from ibkr_trading_bot.utils.logger import get_logger as get_app_logger

    assert get_app_logger("IBKR_BOT.test").propagate is False
    assert get_core_logger("live_bot.test").propagate is False


def test_live_bootstrap_keeps_warmup_snapshot_when_tv_snapshot_empty(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        tab._bootstrap_request_id = 1
        tab.live_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-03-18T10:00:00Z", "2026-03-18T10:05:00Z"], utc=True),
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 12.0],
            }
        )
        state = {"local_called": False, "launch_called": False}
        monkeypatch.setattr(tab, "_try_seed_local_snapshot", lambda: state.__setitem__("local_called", True) or False)
        monkeypatch.setattr(tab, "_launch_stream_worker", lambda: state.__setitem__("launch_called", True))

        payload = tab_live_bot_module.LiveBootstrapPayload(
            bars=[],
            live_df=pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            label_maps=[],
            snapshot_bars=0,
        )
        tab._on_bootstrap_result(1, payload)

        assert state["launch_called"] is True
        assert state["local_called"] is False
        assert len(tab.live_df) == 2
    finally:
        tab.close()


def test_live_bootstrap_payload_survives_auto_detect_failure(monkeypatch):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(
        tab_live_bot_module,
        "_auto_detect_label_polarity",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        tab_live_bot_module,
        "_compute_snapshot_features",
        lambda live_df: pd.DataFrame(
            {
                "feat_a": [0.1] * len(live_df),
                "feat_b": [0.2] * len(live_df),
            }
        ),
    )

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-03-18T10:00:00Z",
                    "2026-03-18T10:05:00Z",
                    "2026-03-18T10:10:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    models = [{"predictor": DummyTernaryPredictor(), "exp_feats": ["feat_a", "feat_b"]}]

    payload = tab_live_bot_module._build_live_bootstrap_payload_from_history_df(df, models, max_bars_buffer=300)

    assert payload.snapshot_bars == 3
    assert len(payload.bars) == 3
    assert list(payload.live_df["close"]) == [1.2, 2.2, 3.2]


def test_live_compute_snapshot_features_accepts_timestamp_column():
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    timestamps = pd.date_range("2026-03-18 09:00:00+00:00", periods=80, freq="5min", tz="UTC")
    base = np.linspace(100.0, 120.0, num=len(timestamps))
    live_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.2,
            "volume": np.full(len(timestamps), 10.0),
        }
    )

    feat = tab_live_bot_module._compute_snapshot_features(live_df)

    assert feat is not None
    assert isinstance(feat, pd.DataFrame)
    assert "ma_fast" in feat.columns
    assert "ma_slow" in feat.columns


def test_live_market_context_from_model_meta_maps_gc_comex_5m():
    from ibkr_trading_bot.gui.tab_live_bot import (
        live_interval_label_from_model_timeframe,
        live_market_context_from_model_meta,
        live_tradingview_market_from_model,
    )

    assert live_interval_label_from_model_timeframe("5m") == "5 min"
    assert live_interval_label_from_model_timeframe("1h") == "1 hour"
    assert live_tradingview_market_from_model("GC", "COMEX") == ("GOLD", "TVC")
    assert (
        live_market_context_from_model_meta(
            {
                "instrument": "GC",
                "exchange": "COMEX",
                "timeframe": "5m",
            }
        )
        == {
            "symbol": "GOLD",
            "exchange": "TVC",
            "bar_size": "5 min",
        }
    )