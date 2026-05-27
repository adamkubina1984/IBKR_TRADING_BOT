import numpy as np
import pandas as pd
import pytest
import time
from matplotlib.collections import PathCollection
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QGroupBox, QWidget

from ._gui_test_helpers import DummyTernaryPredictor, StubCanvas


class _OfflinePaperBrokerClient:
    def __init__(self, message: str = "IBKR TWS paper is offline") -> None:
        self.message = message

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self) -> None:
        raise RuntimeError(self.message)

    def disconnect(self) -> None:
        return None


class _OnlinePaperBrokerClient:
    def __init__(self, account: str = "DU123456", connected_client_id: int = 2) -> None:
        self.account = account
        self.connected_client_id = connected_client_id
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_account_state(self, account: str | None = None):
        resolved = account or self.account
        return type("AccountState", (), {"account": resolved})()


@pytest.fixture(autouse=True)
def _patch_default_live_controller(monkeypatch, tmp_path):
    from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    def _build_controller(self):
        return LiveServiceController(
            strategy_id="tab5-live-test",
            instrument=(self.config.symbol or "GOLD").strip(),
            exchange=(self.config.exchange or "TVC").strip(),
            timeframe=(self.config.bar_size or "5 min").strip(),
            entry_threshold=float(self._curr_entry_thr),
            exit_threshold=float(self._curr_exit_thr),
            use_ma_alignment=bool(self.config.use_and_ensemble),
            freshness_timeout_sec=max(60, int(self.config.max_fresh_age_min) * 60),
            session_root=tmp_path,
        )

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_build_live_service_controller", _build_controller)


def test_tab5_pick_direction_respects_reordered_classes():
    from ibkr_trading_bot.gui.tab_live_bot import _pick_direction_from_raw_proba

    direction, confidence = _pick_direction_from_raw_proba(
        classes_i=[2, 1, 0],
        raw_proba=[0.80, 0.10, 0.10],
        label_map={0: "SHORT", 1: "HOLD", 2: "LONG"},
        short_threshold=0.50,
        long_threshold=0.50,
    )

    assert direction == "LONG"
    assert confidence == 0.80


def test_task_worker_finished_emits_after_thread_exit(qapp):
    from ibkr_trading_bot.gui.components.workers import TaskWorker

    worker = TaskWorker(lambda: "ok")
    observed = {"result": None, "finished": False, "is_finished": None}

    worker.result.connect(lambda value: observed.__setitem__("result", value))

    def _on_finished():
        observed["finished"] = True
        observed["is_finished"] = worker.isFinished()

    worker.finished.connect(_on_finished)
    worker.start()

    deadline = time.monotonic() + 2.0
    while not observed["finished"] and time.monotonic() < deadline:
        QCoreApplication.processEvents()

    assert observed["result"] == "ok"
    assert observed["finished"] is True
    assert observed["is_finished"] is True
    assert worker.wait(1000) is True


def test_project_loggers_do_not_propagate_to_root():
    from ibkr_trading_bot.core.utils.logging_setup import get_logger as get_core_logger
    from ibkr_trading_bot.utils.logger import get_logger as get_app_logger

    assert get_app_logger("IBKR_BOT.test").propagate is False
    assert get_core_logger("live_bot.test").propagate is False


def test_live_tab_defers_startup_check_until_activated(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    calls: list[bool] = []
    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_sync_live_state_from_controller", lambda self: None)
    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_refresh_trades_from_controller", lambda self: None)

    def _fake_refresh(self, *, run_startup_check: bool = False) -> None:
        calls.append(bool(run_startup_check))

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_refresh_controller_status", _fake_refresh)

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        assert calls == [False]

        tab.on_tab_activated()
        qapp.processEvents()

        assert calls == [False, True]
    finally:
        tab.close()


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


def test_live_bootstrap_payload_drops_forming_trailing_bar(monkeypatch):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

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

    payload = tab_live_bot_module._build_live_bootstrap_payload_from_history_df(
        df,
        models,
        bar_size="5 min",
        max_bars_buffer=300,
        reference_time_utc=pd.Timestamp("2026-03-18T10:12:00Z"),
    )

    assert payload.snapshot_bars == 2
    assert len(payload.bars) == 2
    assert list(payload.live_df["close"]) == [1.2, 2.2]


def test_live_tv_worker_uses_closed_bar_filter_before_emit(monkeypatch):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    raw_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-03-18T10:05:00Z", "2026-03-18T10:10:00Z"], utc=True),
            "open": [2.0, 3.0],
            "high": [2.5, 3.5],
            "low": [1.5, 2.5],
            "close": [2.2, 3.2],
            "volume": [11.0, 12.0],
        }
    )
    state = {"filter_calls": 0, "emitted": []}

    def _fake_trim(df, *, bar_size, reference_time_utc=None):
        state["filter_calls"] += 1
        assert bar_size == "5 min"
        return df.iloc[[0]].copy()

    class _StubTv:
        def __init__(self, worker):
            self.worker = worker

        def get_history(self, symbol, exchange, tf_label, limit=2):
            self.worker._stop = True
            return raw_df.copy()

    monkeypatch.setattr(tab_live_bot_module, "_trim_forming_history_tail", _fake_trim)

    worker = tab_live_bot_module.TVWorker(tab_live_bot_module.LiveConfig(bar_size="5 min"))
    worker.tv = _StubTv(worker)
    worker._last_ns = int(pd.Timestamp("2026-03-18T10:05:00Z").value)
    monkeypatch.setattr(worker, "_poll_interval_s", lambda: 0)
    worker.barClosed.connect(lambda payload: state["emitted"].append(payload))

    worker.run()

    assert state["filter_calls"] == 1
    assert state["emitted"] == []


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


def test_live_settings_display_applies_runtime_flags(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        tab._update_settings_display(
            {
                "entry_threshold": 0.55,
                "exit_threshold": 0.45,
                "exit_policy": "legacy_flat",
                "use_ma_only": True,
                "use_and_ensemble": True,
            },
            {
                "ternary_threshold_short": 0.40,
                "ternary_threshold_long": 0.60,
            },
        )

        assert tab.config.use_ma_only is True
        assert tab.config.use_and_ensemble is True
        assert tab._curr_entry_thr == 0.55
        assert tab._curr_exit_thr == 0.45
        assert tab._curr_exit_policy == "flat_on_weak_signal"
        assert tab._curr_t_short == 0.40
        assert tab._curr_t_long == 0.60
        assert tab.user_settings["exit_policy"] == "flat_on_weak_signal"
        assert "flat_on_weak_signal" in tab.lbl_exit_policy.text()
    finally:
        tab.close()


def test_live_rescore_forwards_active_exit_policy(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        ts1 = pd.Timestamp("2026-03-18T10:00:00Z")
        tab._bars = [
            {"time": ts1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        ]
        tab.models = [{"predictor": object(), "label_map": {0: "SHORT", 1: "LONG"}, "t_short": 0.5, "t_long": 0.5}]
        tab.config.use_and_ensemble = False
        tab._update_settings_display({"exit_policy": "legacy_flat"}, {})

        raw = pd.DataFrame(
            {
                "close": [100.0],
                "atr": [1.0],
                "ma_fast": [10.0],
                "ma_slow": [10.0],
            },
            index=pd.to_datetime([ts1], utc=True),
        )
        calls: list[dict[str, object]] = []

        monkeypatch.setattr(tab, "_get_raw_indicators", lambda: raw)
        monkeypatch.setattr(tab, "_sanitize_feature_matrix", lambda feat: feat)
        monkeypatch.setattr(tab, "_predict_one_label_VOTE", lambda Xrow: (-1, 0.8, ["SHORT"], [0.8]))
        monkeypatch.setattr(tab, "_update_position_and_trades", lambda raw_df: None)
        monkeypatch.setattr(tab, "_track_predictions_for_degradation", lambda raw_df: None)
        monkeypatch.setattr(
            tab_live_bot_module,
            "apply_live_hysteresis",
            lambda *args, **kwargs: (calls.append(kwargs), "SHORT")[1],
        )

        tab._rescore_all()

        assert len(calls) == 1
        assert calls[0]["exit_policy"] == "flat_on_weak_signal"
    finally:
        tab.close()


def test_live_trade_updates_use_shared_executor(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        ts1 = pd.Timestamp("2026-03-18T10:00:00Z")
        ts2 = pd.Timestamp("2026-03-18T10:05:00Z")
        raw = pd.DataFrame(
            {"close": [100.0, 101.5]},
            index=pd.to_datetime([ts1, ts2], utc=True),
        )

        tab._bars = [{"time": ts1, "signal": "LONG"}]
        tab._update_position_and_trades(raw)
        assert tab._live_pos == 1
        assert tab._open_trade is not None

        tab._bars = [{"time": ts2, "signal": "FLAT"}]
        tab._update_position_and_trades(raw)
        assert tab._live_pos == 0
        assert tab._open_trade is None
        assert len(tab._trades) == 1
        assert tab._trades[0]["pnl"] == 1.5
    finally:
        tab.close()


def test_live_chart_renders_doji_with_visible_body(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        ts1 = pd.Timestamp("2026-03-18T10:00:00Z")
        ts2 = pd.Timestamp("2026-03-18T10:05:00Z")
        tab._bars = [
            {"time": ts1, "open": 100.0, "high": 101.0, "low": 99.4, "close": 100.8, "signal": "LONG"},
            {"time": ts2, "open": 100.2, "high": 100.6, "low": 99.9, "close": 100.2, "signal": None},
        ]

        tab._render_charts()

        assert len(tab.ax_price.patches) == 2
        heights = [float(patch.get_height()) for patch in tab.ax_price.patches]
        assert all(height > 0 for height in heights)
    finally:
        tab.close()


def test_live_chart_keeps_existing_signal_marker_when_rescore_fails_on_bar_update(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        ts = pd.Timestamp("2026-03-18T10:00:00Z")
        tab._bars = [
            {
                "time": ts,
                "open": 100.0,
                "high": 101.0,
                "low": 99.4,
                "close": 100.8,
                "volume": 10.0,
                "signal": "LONG",
                "chart_signal": "LONG",
            }
        ]
        tab._bar_index = {int(ts.value): 0}
        tab.live_df = pd.DataFrame(
            {
                "timestamp": [ts],
                "open": [100.0],
                "high": [101.0],
                "low": [99.4],
                "close": [100.8],
                "volume": [10.0],
            }
        )

        def _raise_rescore() -> None:
            raise RuntimeError("rescore failed")

        monkeypatch.setattr(tab, "_rescore_all", _raise_rescore)

        tab._render_charts()
        before_offsets = sum(
            len(collection.get_offsets())
            for collection in tab.ax_price.collections
            if isinstance(collection, PathCollection)
        )

        tab._on_bar_closed(
            {
                "time": ts,
                "open": 100.1,
                "high": 101.2,
                "low": 99.3,
                "close": 100.9,
                "volume": 11.0,
            }
        )
        qapp.processEvents()

        after_offsets = sum(
            len(collection.get_offsets())
            for collection in tab.ax_price.collections
            if isinstance(collection, PathCollection)
        )

        assert before_offsets == 1
        assert after_offsets == 1
        assert tab._bars[0].get("signal") == "LONG"
        assert tab._bars[0].get("chart_signal") == "LONG"
    finally:
        tab.close()


def test_live_chart_renders_active_signal_marker_on_each_bar_in_position(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        ts1 = pd.Timestamp("2026-03-18T10:00:00Z")
        ts2 = pd.Timestamp("2026-03-18T10:05:00Z")
        ts3 = pd.Timestamp("2026-03-18T10:10:00Z")
        tab._bars = [
            {"time": ts1, "open": 100.0, "high": 101.0, "low": 99.2, "close": 100.6, "signal": "SHORT", "chart_signal": "SHORT"},
            {"time": ts2, "open": 100.4, "high": 100.9, "low": 99.1, "close": 99.8, "signal": "SHORT", "chart_signal": None},
            {"time": ts3, "open": 99.8, "high": 100.1, "low": 98.9, "close": 99.4, "signal": "SHORT", "chart_signal": None},
        ]

        tab._render_charts()

        marker_offsets = sum(
            len(collection.get_offsets())
            for collection in tab.ax_price.collections
            if isinstance(collection, PathCollection)
        )

        assert marker_offsets == 3
    finally:
        tab.close()


def test_live_chart_keeps_markers_on_all_visible_bars_of_held_position(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        tab.config.display_bars = 30
        base_ts = pd.Timestamp("2026-03-18T10:00:00Z")
        bars = []
        for idx in range(31):
            bars.append(
                {
                    "time": base_ts + pd.Timedelta(minutes=5 * idx),
                    "open": 100.0 - idx * 0.1,
                    "high": 101.0 - idx * 0.1,
                    "low": 99.0 - idx * 0.1,
                    "close": 99.8 - idx * 0.1,
                    "signal": "SHORT",
                    "chart_signal": "SHORT" if idx == 0 else None,
                }
            )
        tab._bars = bars

        tab._render_charts()

        marker_offsets = sum(
            len(collection.get_offsets())
            for collection in tab.ax_price.collections
            if isinstance(collection, PathCollection)
        )

        assert marker_offsets == 30
    finally:
        tab.close()


def test_live_rescore_marks_chart_signal_on_proposal_direction_changes(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        ts1 = pd.Timestamp("2026-03-18T10:00:00Z")
        ts2 = pd.Timestamp("2026-03-18T10:05:00Z")
        ts3 = pd.Timestamp("2026-03-18T10:10:00Z")
        ts4 = pd.Timestamp("2026-03-18T10:15:00Z")
        tab._bars = [
            {"time": ts1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"time": ts2, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"time": ts3, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"time": ts4, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        ]
        tab.models = [{"predictor": object(), "label_map": {0: "SHORT", 1: "LONG"}, "t_short": 0.5, "t_long": 0.5}]
        tab.config.use_and_ensemble = False
        raw = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0, 100.0],
                "atr": [1.0, 1.0, 1.0, 1.0],
                "ma_fast": [10.0, 10.0, 10.0, 10.0],
                "ma_slow": [10.0, 10.0, 10.0, 10.0],
            },
            index=pd.to_datetime([ts1, ts2, ts3, ts4], utc=True),
        )
        outputs = iter([
            (-1, 0.8, ["SHORT"], [0.8]),
            (-1, 0.8, ["SHORT"], [0.8]),
            (0, 0.0, ["FLAT"], [0.0]),
            (-1, 0.8, ["SHORT"], [0.8]),
        ])

        monkeypatch.setattr(tab, "_get_raw_indicators", lambda: raw)
        monkeypatch.setattr(tab, "_sanitize_feature_matrix", lambda feat: feat)
        monkeypatch.setattr(tab, "_predict_one_label_VOTE", lambda Xrow: next(outputs))
        monkeypatch.setattr(tab, "_update_position_and_trades", lambda raw_df: None)
        monkeypatch.setattr(tab, "_track_predictions_for_degradation", lambda raw_df: None)
        tab._update_settings_display({"exit_policy": "hold_until_opposite"}, {})

        tab._rescore_all()

        assert [bar.get("signal") for bar in tab._bars] == ["SHORT", "SHORT", "SHORT", "SHORT"]
        assert [bar.get("chart_signal") for bar in tab._bars] == ["SHORT", None, None, "SHORT"]
    finally:
        tab.close()


def test_live_bot_layout_avoids_model_wrapper_and_clarifies_ibkr_account(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        titles = {box.title() for box in tab.findChildren(QGroupBox)}

        assert "Model" not in titles
        assert "Nastaveni z Tab 3 (read-only)" not in titles
        assert tab.ed_ib_account.placeholderText() == "DU123456 (volitelne)"
        assert "DU" in tab.ed_ib_account.toolTip()
        assert tab.degradation_console.minimumHeight() >= 220
    finally:
        tab.close()


def test_live_prepare_x_for_model_static_rejects_missing_features():
    from ibkr_trading_bot.gui.tab_live_bot import _prepare_X_for_model_static

    with pytest.raises(ValueError, match="feat_b"):
        _prepare_X_for_model_static(
            pd.DataFrame({"feat_a": [1.0], "close": [100.0]}),
            ["feat_a", "feat_b"],
        )


def test_live_load_models_rejects_mixed_feature_contracts(monkeypatch, qapp, tmp_path):
    import json
    import joblib

    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    def _write_model(path, features):
        joblib.dump(DummyTernaryPredictor(), path)
        meta = {
            "created_at": "2026-03-06T12:00:00",
            "created_at_iso": "2026-03-06T12:00:00",
            "trained_features": list(features),
            "classes": [0, 1, 2],
            "class_to_dir": {0: "SHORT", 1: "HOLD", 2: "LONG"},
            "label_mode": "ternary_mapped",
            "ternary_threshold_short": 0.40,
            "ternary_threshold_long": 0.60,
            "metrics_holdout": {"profit_net": 1.0},
        }
        path.with_name(path.stem + "_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return path

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    model_a = _write_model(tmp_path / "model_a.pkl", ["feat_a", "feat_b"])
    model_b = _write_model(tmp_path / "model_b.pkl", ["feat_a", "feat_c"])

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        tab.le_model_path.setText(f"{model_a};{model_b}")

        assert tab._load_models() is False
        assert tab.model_expected_features is None
        qapp.processEvents()
        tab._flush_log_queue()
        assert "Feature kontrakt modelu model_b.pkl nesedi s model_a.pkl." in tab.console.toPlainText()
    finally:
        tab.close()


def test_live_load_models_rejects_mixed_exit_policy(monkeypatch, qapp, tmp_path):
    import json
    import joblib

    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    def _write_model(path, exit_policy):
        joblib.dump(DummyTernaryPredictor(), path)
        meta = {
            "created_at": "2026-03-06T12:00:00",
            "created_at_iso": "2026-03-06T12:00:00",
            "trained_features": ["feat_a", "feat_b"],
            "classes": [0, 1, 2],
            "class_to_dir": {0: "SHORT", 1: "HOLD", 2: "LONG"},
            "label_mode": "ternary_mapped",
            "ternary_threshold_short": 0.40,
            "ternary_threshold_long": 0.60,
            "metrics_holdout": {"profit_net": 1.0},
            "user_settings": {"exit_policy": exit_policy},
        }
        path.with_name(path.stem + "_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return path

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    model_a = _write_model(tmp_path / "model_a.pkl", "legacy_flat")
    model_b = _write_model(tmp_path / "model_b.pkl", "hold_until_opposite")

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        tab.le_model_path.setText(f"{model_a};{model_b}")

        assert tab._load_models() is False
        qapp.processEvents()
        tab._flush_log_queue()
        assert "Exit policy modelu model_b.pkl nesedi s model_a.pkl:" in tab.console.toPlainText()
    finally:
        tab.close()


def test_live_rescore_replays_history_without_smearing_current_short_position(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        ts1 = pd.Timestamp("2026-03-18T10:00:00Z")
        ts2 = pd.Timestamp("2026-03-18T10:05:00Z")
        ts3 = pd.Timestamp("2026-03-18T10:10:00Z")
        tab._bars = [
            {"time": ts1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"time": ts2, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"time": ts3, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        ]
        tab.models = [{"predictor": object(), "label_map": {0: "SHORT", 1: "LONG"}, "t_short": 0.5, "t_long": 0.5}]
        tab.config.use_and_ensemble = False
        tab._live_pos = -1
        raw = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0],
                "atr": [1.0, 1.0, 1.0],
                "ma_fast": [10.0, 10.0, 10.0],
                "ma_slow": [10.0, 10.0, 10.0],
            },
            index=pd.to_datetime([ts1, ts2, ts3], utc=True),
        )

        monkeypatch.setattr(tab, "_get_raw_indicators", lambda: raw)
        monkeypatch.setattr(tab, "_sanitize_feature_matrix", lambda feat: feat)
        monkeypatch.setattr(tab, "_predict_one_label_VOTE", lambda Xrow: (0, 0.0, ["FLAT"], [0.0]))
        monkeypatch.setattr(tab, "_update_position_and_trades", lambda raw_df: None)
        monkeypatch.setattr(tab, "_track_predictions_for_degradation", lambda raw_df: None)

        tab._rescore_all()

        assert [bar.get("signal") for bar in tab._bars] == [None, None, None]
    finally:
        tab.close()


def test_live_log_flush_preserves_manual_scroll_position(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    class _FakeScrollBar:
        def __init__(self, value: int, maximum: int):
            self._value = value
            self._maximum = maximum

        def value(self):
            return self._value

        def maximum(self):
            return self._maximum

        def setValue(self, value):
            self._value = value

    class _FakeConsole:
        def __init__(self):
            self.lines = []
            self._scrollbar = _FakeScrollBar(value=4, maximum=20)

        def verticalScrollBar(self):
            return self._scrollbar

        def appendPlainText(self, text):
            self.lines.append(text)
            self._scrollbar._maximum += 1

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        fake_console = _FakeConsole()
        tab.console = fake_console
        tab._log_queue.extend(["line-1", "line-2"])

        tab._flush_log_queue()

        assert fake_console.lines == ["line-1", "line-2"]
        assert fake_console.verticalScrollBar().value() == 4
    finally:
        tab.close()


def test_live_bot_keeps_closed_bars_sorted_when_older_bar_arrives_late(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        monkeypatch.setattr(tab, "_rescore_all", lambda: None)

        tab._on_bar_closed(
            {
                "time": "2026-03-18T10:05:00Z",
                "open": 100.2,
                "high": 100.6,
                "low": 99.9,
                "close": 100.2,
                "volume": 11,
            }
        )
        tab._on_bar_closed(
            {
                "time": "2026-03-18T10:00:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.4,
                "close": 100.8,
                "volume": 10,
            }
        )

        times = [pd.to_datetime(bar["time"], utc=True) for bar in tab._bars]
        tab._process_pending_bar_updates()

        assert times == sorted(times)
        assert len(tab.ax_price.patches) == 2
    finally:
        tab.close()


def test_live_shutdown_stops_stream_and_background_workers(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    class DummyWorker:
        def __init__(self):
            self.stop_calls = 0
            self.finished = type("_Signal", (), {"connect": lambda self, cb: None, "disconnect": lambda self: None})()

        def stop(self):
            self.stop_calls += 1

        def isRunning(self):
            return False

        def wait(self, ms):
            return True

        def deleteLater(self):
            return None

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    stream_worker = DummyWorker()
    bootstrap_worker = DummyWorker()
    warmup_worker = DummyWorker()
    degradation_worker = DummyWorker()
    tab.worker = stream_worker
    tab._bootstrap_worker = bootstrap_worker
    tab._warmup_worker = warmup_worker
    tab._degradation_worker = degradation_worker

    assert tab.shutdown() is True

    assert stream_worker.stop_calls == 1
    assert bootstrap_worker.stop_calls == 1
    assert warmup_worker.stop_calls == 1
    assert degradation_worker.stop_calls == 1
    assert tab.worker is None
    assert tab._bootstrap_worker is None
    assert tab._warmup_worker is None
    assert tab._degradation_worker is None
    assert tab._log_timer.isActive() is False


def test_live_shutdown_returns_false_when_stream_worker_does_not_stop(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    class HangingWorker:
        def __init__(self):
            self.stop_calls = 0
            self.finished = type("_Signal", (), {"connect": lambda self, cb: None, "disconnect": lambda self: None})()

        def stop(self):
            self.stop_calls += 1

        def isRunning(self):
            return True

        def wait(self, ms):
            return False

        def deleteLater(self):
            return None

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        hanging = HangingWorker()
        tab.worker = hanging

        assert tab.shutdown() is False
        assert hanging.stop_calls == 1
        assert tab.worker is hanging
    finally:
        tab.worker = None
        tab.close()


def test_live_warm_adapter_uses_ma_only_mode(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        tab.config.use_ma_only = True
        tab.models = [{"predictor": DummyTernaryPredictor()}]
        adapter = tab_live_bot_module._WarmAdapter(tab)

        features = pd.DataFrame(
            {
                "ma_fast": [12.0],
                "ma_slow": [10.0],
                "close": [100.0],
                "atr": [1.0],
            }
        )

        label, probs, classes = adapter.predict(features)

        assert label == "LONG"
        assert probs == [1.0, 0.0]
        assert classes == ["LONG", "SHORT"]
    finally:
        tab.close()


def test_live_warm_adapter_builds_history_incrementally_without_duplicates(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    class _StubTv:
        def get_history(self, symbol, exchange, tf_label, limit):
            return pd.DataFrame(
                {
                    "time": pd.to_datetime(
                        [
                            "2026-03-18T10:00:00Z",
                            "2026-03-18T10:05:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                    "volume": [10.0, 11.0],
                }
            )

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    monkeypatch.setattr(tab_live_bot_module, "TradingViewClient", lambda **kwargs: _StubTv())

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        monkeypatch.setattr(tab, "_compute_indicators", lambda df: df.assign(ma_fast=df["close"], ma_slow=df["close"]))
        monkeypatch.setattr(tab, "_sanitize_feature_matrix", lambda df: df.copy())
        adapter = tab_live_bot_module._WarmAdapter(tab)

        bars = adapter.fetch_history("GOLD", "TVC", "5 min", n_bars=10)

        assert len(bars) == 2
        assert adapter._hist_df.empty is True

        adapter.featurize_until(bars[0])
        adapter.featurize_until(bars[1])

        dates = list(adapter._hist_df["date"])
        assert len(dates) == 2
        assert dates == sorted(dates)
        assert len({int(pd.Timestamp(value).value) for value in dates}) == 2
    finally:
        tab.close()


def test_live_warm_adapter_featurize_until_falls_back_when_indicators_are_empty(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        monkeypatch.setattr(tab, "_compute_indicators", lambda df: pd.DataFrame())
        monkeypatch.setattr(tab, "_sanitize_feature_matrix", lambda df: df.copy())
        adapter = tab_live_bot_module._WarmAdapter(tab)

        features = adapter.featurize_until(
            {
                "time": pd.Timestamp("2026-03-18T10:00:00Z"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
            }
        )

        assert len(features.index) == 1
        assert float(features["close"].iloc[-1]) == 100.5
        assert float(features["ma_fast"].iloc[-1]) == 100.5
        assert float(features["ma_slow"].iloc[-1]) == 100.5
    finally:
        tab.close()


def test_live_prepare_x_for_model_accepts_model_label_kwarg(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        prepared = tab._prepare_X_for_model(
            pd.DataFrame({"close": [100.5]}),
            ["close"],
            model_label="demo_model.pkl",
        )

        assert list(prepared.columns) == ["close"]
        assert float(prepared["close"].iloc[-1]) == 100.5
    finally:
        tab.close()


def test_live_bot_restores_last_model_paths(monkeypatch, qapp, tmp_path):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    class _DummySettings:
        _store: dict[str, str] = {}

        def __init__(self, *args, **kwargs):
            pass

        def value(self, key, default=None):
            return self._store.get(str(key), default)

        def setValue(self, key, value):
            self._store[str(key)] = str(value)

        def sync(self):
            return None

    _DummySettings._store = {}
    monkeypatch.setattr(tab_live_bot_module, "QSettings", _DummySettings)
    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)

    model_file = tmp_path / "model_a.pkl"
    model_file.write_bytes(b"dummy")

    first = tab_live_bot_module.LiveBotWidget()
    try:
        first.set_model_paths([str(model_file)])
        first._save_ui_settings()
    finally:
        first.close()

    second = tab_live_bot_module.LiveBotWidget()
    try:
        assert str(model_file) in second.le_model_path.text()
    finally:
        second.close()


def test_live_bot_restores_ibkr_paper_settings(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    class _DummySettings:
        _store: dict[str, object] = {}

        def __init__(self, *args, **kwargs):
            pass

        def value(self, key, default=None):
            return self._store.get(str(key), default)

        def setValue(self, key, value):
            self._store[str(key)] = value

        def sync(self):
            return None

    _DummySettings._store = {}
    monkeypatch.setattr(tab_live_bot_module, "QSettings", _DummySettings)
    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)

    first = tab_live_bot_module.LiveBotWidget()
    try:
        first.chk_ib_startup_check.setChecked(False)
        first.ed_ib_host.setText("192.168.0.10")
        first.spn_ib_port.setValue(4002)
        first.spn_ib_client_id.setValue(77)
        first.ed_ib_account.setText("DU765432")
        first.chk_ib_readonly.setChecked(False)
        first._save_ui_settings()
    finally:
        first.close()

    second = tab_live_bot_module.LiveBotWidget()
    try:
        assert second.chk_ib_startup_check.isChecked() is False
        assert second.ed_ib_host.text() == "192.168.0.10"
        assert second.spn_ib_port.value() == 4002
        assert second.spn_ib_client_id.value() == 77
        assert second.ed_ib_account.text() == "DU765432"
        assert second.chk_ib_readonly.isChecked() is False
    finally:
        second.close()


def test_live_widget_shows_blocked_status_when_paper_broker_is_offline(monkeypatch, qapp, tmp_path):
    from ibkr_trading_bot.core.services.live import TwsConnectionConfig
    from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)

    def _build_controller(self):
        return LiveServiceController(
            strategy_id="tab5-live-test",
            instrument=(self.config.symbol or "GOLD").strip(),
            exchange=(self.config.exchange or "TVC").strip(),
            timeframe=(self.config.bar_size or "5 min").strip(),
            entry_threshold=float(self._curr_entry_thr),
            exit_threshold=float(self._curr_exit_thr),
            use_ma_alignment=bool(self.config.use_and_ensemble),
            freshness_timeout_sec=max(60, int(self.config.max_fresh_age_min) * 60),
            session_root=tmp_path,
            broker_connection=TwsConnectionConfig(account="DU123456"),
            broker_client=_OfflinePaperBrokerClient(),
        )

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_build_live_service_controller", _build_controller)

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        assert tab.lbl_broker_status.text() == "Broker: OFFLINE"
        assert tab.lbl_mode.text() == "Mode: BLOCKED (BROKER_OFFLINE)"
        assert tab.btn_trade.isEnabled() is False
        assert "127.0.0.1:7497" in tab.lbl_broker_status.toolTip()
    finally:
        tab.close()


def test_live_widget_start_is_blocked_without_paper_broker(monkeypatch, qapp, tmp_path):
    from ibkr_trading_bot.core.services.live import TwsConnectionConfig
    from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)

    def _build_controller(self):
        return LiveServiceController(
            strategy_id="tab5-live-test",
            instrument=(self.config.symbol or "GOLD").strip(),
            exchange=(self.config.exchange or "TVC").strip(),
            timeframe=(self.config.bar_size or "5 min").strip(),
            entry_threshold=float(self._curr_entry_thr),
            exit_threshold=float(self._curr_exit_thr),
            use_ma_alignment=bool(self.config.use_and_ensemble),
            freshness_timeout_sec=max(60, int(self.config.max_fresh_age_min) * 60),
            session_root=tmp_path,
            broker_connection=TwsConnectionConfig(account="DU123456"),
            broker_client=_OfflinePaperBrokerClient("Connection refused"),
        )

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_build_live_service_controller", _build_controller)

    warnings: list[str] = []
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        monkeypatch.setattr(tab, "_load_models", lambda: True)
        monkeypatch.setattr(tab, "_apply_tf_presets", lambda: None)
        monkeypatch.setattr(tab, "_start_worker", lambda: (_ for _ in ()).throw(AssertionError("worker must not start")))
        monkeypatch.setattr(
            tab_live_bot_module.QMessageBox,
            "warning",
            lambda _parent, _title, text: warnings.append(str(text)),
        )

        tab._on_start()

        assert warnings
        assert "Paper broker není dostupný" in warnings[-1]
        assert "Connection refused" in warnings[-1]
        assert tab.lbl_mode.text() == "Mode: BLOCKED (BROKER_OFFLINE)"
    finally:
        tab.close()


def test_live_widget_start_suggests_new_client_id_when_current_one_is_in_use(monkeypatch, qapp, tmp_path):
    from ibkr_trading_bot.core.services.live import TwsConnectionConfig
    from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)

    def _build_controller(self):
        return LiveServiceController(
            strategy_id="tab5-live-test",
            instrument=(self.config.symbol or "GOLD").strip(),
            exchange=(self.config.exchange or "TVC").strip(),
            timeframe=(self.config.bar_size or "5 min").strip(),
            entry_threshold=float(self._curr_entry_thr),
            exit_threshold=float(self._curr_exit_thr),
            use_ma_alignment=bool(self.config.use_and_ensemble),
            freshness_timeout_sec=max(60, int(self.config.max_fresh_age_min) * 60),
            session_root=tmp_path,
            broker_connection=TwsConnectionConfig(account="DU123456", client_id=1),
            broker_client=_OfflinePaperBrokerClient(
                "IBKR API rejected client ID 1: already in use. Choose a different client ID."
            ),
        )

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_build_live_service_controller", _build_controller)

    warnings: list[str] = []
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        monkeypatch.setattr(tab, "_load_models", lambda: True)
        monkeypatch.setattr(tab, "_apply_tf_presets", lambda: None)
        monkeypatch.setattr(tab, "_start_worker", lambda: (_ for _ in ()).throw(AssertionError("worker must not start")))
        monkeypatch.setattr(
            tab_live_bot_module.QMessageBox,
            "warning",
            lambda _parent, _title, text: warnings.append(str(text)),
        )

        tab._on_start()

        assert warnings
        assert "already in use" in warnings[-1]
        assert "Client ID" in warnings[-1]
        assert "automaticky zkouší několik vyšších Client ID" in warnings[-1]
        assert tab.lbl_mode.text() == "Mode: BLOCKED (BROKER_OFFLINE)"
    finally:
        tab.close()


def test_live_widget_shows_configured_and_connected_client_id_when_fallback_used(monkeypatch, qapp, tmp_path):
    from ibkr_trading_bot.core.services.live import TwsConnectionConfig
    from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)

    def _build_controller(self):
        return LiveServiceController(
            strategy_id="tab5-live-test",
            instrument=(self.config.symbol or "GOLD").strip(),
            exchange=(self.config.exchange or "TVC").strip(),
            timeframe=(self.config.bar_size or "5 min").strip(),
            entry_threshold=float(self._curr_entry_thr),
            exit_threshold=float(self._curr_exit_thr),
            use_ma_alignment=bool(self.config.use_and_ensemble),
            freshness_timeout_sec=max(60, int(self.config.max_fresh_age_min) * 60),
            session_root=tmp_path,
            broker_connection=TwsConnectionConfig(account="DU123456", client_id=1),
            broker_client=_OnlinePaperBrokerClient(account="DU123456", connected_client_id=2),
        )

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_build_live_service_controller", _build_controller)

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        tab.on_tab_activated()
        qapp.processEvents()
        tooltip = tab.lbl_broker_status.toolTip()
        assert tab.lbl_broker_status.text() == "Broker: PAPER OK (DU123456)"
        assert "clientId=2" in tooltip
        assert "configured clientId=1" in tooltip
        assert "auto-fallback active" in tooltip
    finally:
        tab.close()


def test_live_widget_restores_service_backed_mode_and_trades(monkeypatch, qapp, tmp_path):
    from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)

    def _build_controller(self):
        return LiveServiceController(
            strategy_id="tab5-live-test",
            instrument=(self.config.symbol or "GOLD").strip(),
            exchange=(self.config.exchange or "TVC").strip(),
            timeframe=(self.config.bar_size or "5 min").strip(),
            entry_threshold=float(self._curr_entry_thr),
            exit_threshold=float(self._curr_exit_thr),
            exit_policy="hold_until_opposite",
            use_ma_alignment=bool(self.config.use_and_ensemble),
            freshness_timeout_sec=max(60, int(self.config.max_fresh_age_min) * 60),
            session_root=tmp_path,
        )

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_build_live_service_controller", _build_controller)

    first = tab_live_bot_module.LiveBotWidget()
    try:
        first._on_toggle_trading(True)
        first.live_controller.process_closed_bar(
            "2026-04-29T12:30:00Z",
            100.0,
            signal="LONG",
        )
        first.live_controller.process_closed_bar(
            "2026-04-29T12:35:00Z",
            101.0,
            signal="SHORT",
        )
        first._sync_live_state_from_controller()
        first._refresh_trades_from_controller()

        assert first.lbl_mode.text() == "Mode: LIVE"
        assert first.tbl_trades.rowCount() == 1
    finally:
        first.close()

    second = tab_live_bot_module.LiveBotWidget()
    try:
        assert second.lbl_mode.text() == "Mode: LIVE"
        assert second.tbl_trades.rowCount() == 1
        assert len(second._trades) == 1
        assert second._trades[0]["direction"] == "LONG"
        assert second._live_pos == -1
    finally:
        second.close()


def test_live_widget_shows_blocked_release_gate_status(monkeypatch, qapp, tmp_path):
    from ibkr_trading_bot.core.services.live_release_gate import LiveReleaseGateInputs
    from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)

    def _build_controller(self):
        return LiveServiceController(
            strategy_id="tab5-live-test",
            instrument=(self.config.symbol or "GOLD").strip(),
            exchange=(self.config.exchange or "TVC").strip(),
            timeframe=(self.config.bar_size or "5 min").strip(),
            entry_threshold=float(self._curr_entry_thr),
            exit_threshold=float(self._curr_exit_thr),
            use_ma_alignment=bool(self.config.use_and_ensemble),
            freshness_timeout_sec=max(60, int(self.config.max_fresh_age_min) * 60),
            session_root=tmp_path,
            release_gate_inputs=LiveReleaseGateInputs(
                automated_tests_passed=False,
                paper_soak_completed=False,
                paper_soak_days=0,
                audit_trail_complete=False,
            ),
        )

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_build_live_service_controller", _build_controller)

    warnings: list[str] = []
    monkeypatch.setattr(
        tab_live_bot_module.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(str(text)),
    )

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        assert "BLOCKED" in tab.lbl_release_gate.text()
        assert "PAPER_SOAK_NOT_COMPLETED" in tab.lbl_release_gate.toolTip()
        assert "AUTOMATED_TESTS_FAILED" in tab.lbl_release_gate.toolTip()
        assert tab.cmb_execution_mode.currentData() == "PAPER"
        real_item = tab.cmb_execution_mode.model().item(1)
        assert real_item is not None
        assert real_item.isEnabled() is False

        tab.cmb_execution_mode.setCurrentIndex(1)
        tab._on_execution_mode_changed(1)

        assert tab.live_controller.execution_mode == "PAPER"
        assert tab.cmb_execution_mode.currentData() == "PAPER"
        assert warnings
        assert "PAPER_SOAK_NOT_COMPLETED" in warnings[-1]
    finally:
        tab.close()


def test_live_widget_shows_ready_release_gate_warnings(monkeypatch, qapp, tmp_path):
    from ibkr_trading_bot.core.services.live_release_gate import LiveReleaseGateInputs
    from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)

    def _build_controller(self):
        return LiveServiceController(
            strategy_id="tab5-live-test",
            instrument=(self.config.symbol or "GOLD").strip(),
            exchange=(self.config.exchange or "TVC").strip(),
            timeframe=(self.config.bar_size or "5 min").strip(),
            entry_threshold=float(self._curr_entry_thr),
            exit_threshold=float(self._curr_exit_thr),
            use_ma_alignment=bool(self.config.use_and_ensemble),
            freshness_timeout_sec=max(60, int(self.config.max_fresh_age_min) * 60),
            session_root=tmp_path,
            release_gate_inputs=LiveReleaseGateInputs(
                automated_tests_passed=True,
                paper_soak_completed=True,
                paper_soak_days=3,
                audit_trail_complete=True,
            ),
        )

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_build_live_service_controller", _build_controller)

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        assert "READY" in tab.lbl_release_gate.text()
        assert "PAPER_SOAK_AT_MINIMUM_DURATION" in tab.lbl_release_gate.toolTip()
        real_item = tab.cmb_execution_mode.model().item(1)
        assert real_item is not None
        assert real_item.isEnabled() is True

        tab.cmb_execution_mode.setCurrentIndex(1)
        tab._on_execution_mode_changed(1)

        assert tab.live_controller.execution_mode == "REAL"
        assert tab.cmb_execution_mode.currentData() == "REAL"
    finally:
        tab.close()