from __future__ import annotations

from ibkr_trading_bot.core.services.live import ExecutionJournal, RuntimeStateStore
from ibkr_trading_bot.core.services.live_trading_execution_service import LiveTradingExecutionConfig, LiveTradingExecutionService


def _make_service(tmp_path, **overrides) -> LiveTradingExecutionService:
    config = LiveTradingExecutionConfig(
        strategy_id="persistence-test",
        instrument="GC",
        timeframe="5m",
        warning_alpha=1.0,
        **overrides,
    )
    return LiveTradingExecutionService(
        config,
        journal=ExecutionJournal(tmp_path / "execution.journal.jsonl"),
        state_store=RuntimeStateStore(tmp_path / "runtime.state.json"),
    )


def test_warning_state_and_metrics_restore_across_restart(tmp_path):
    service = _make_service(tmp_path, rolling_window_bars=2, min_bars_for_health=2)
    service.arm_trading(baseline_profit_per_bar=0.6, actor="tester")
    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)
    warning = service.process_closed_bar("2026-04-29T12:35:00Z", 100.8, model_direction="SHORT", confidence=1.0)

    assert warning.status.mode == "WARNING"

    restored = _make_service(tmp_path, rolling_window_bars=2, min_bars_for_health=2)
    assert restored.status.mode == "WARNING"
    assert restored.status.last_warning_reason == "EWMA_PROFIT_PER_BAR_DEGRADED"
    assert restored.status.live_profit_per_bar > 0.0


def test_safe_stop_reason_restores_across_restart(tmp_path):
    service = _make_service(tmp_path, min_bars_for_health=2)
    service.arm_trading(baseline_profit_per_bar=0.4, actor="tester")
    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)
    stopped = service.process_closed_bar("2026-04-29T12:35:00Z", 99.0, model_direction="SHORT", confidence=1.0)

    assert stopped.status.mode == "SAFE_STOP"

    restored = _make_service(tmp_path, min_bars_for_health=2)
    assert restored.status.mode == "SAFE_STOP"
    assert restored.status.last_safe_stop_reason == "LIVE_PROFIT_PER_BAR_BELOW_ZERO"