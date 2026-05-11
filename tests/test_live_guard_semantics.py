from __future__ import annotations

from ibkr_trading_bot.core.services.live import ExecutionJournal, RuntimeStateStore
from ibkr_trading_bot.core.services.live_trading_execution_service import LiveTradingExecutionConfig, LiveTradingExecutionService


def _make_service(tmp_path, **overrides) -> LiveTradingExecutionService:
    config = LiveTradingExecutionConfig(
        strategy_id="guard-semantics",
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


def test_safe_stop_blocks_new_entries_until_manual_rearm(tmp_path):
    service = _make_service(tmp_path, min_bars_for_health=2)
    service.arm_trading(baseline_profit_per_bar=0.4, actor="tester")

    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)
    stopped = service.process_closed_bar("2026-04-29T12:35:00Z", 99.0, model_direction="SHORT", confidence=1.0)

    assert stopped.status.mode == "SAFE_STOP"

    blocked = service.process_closed_bar("2026-04-29T12:40:00Z", 101.0, model_direction="LONG", confidence=1.0)
    assert blocked.order_intents == ()
    assert blocked.status.runtime_state.position.side == "SHORT"
    assert blocked.status.mode == "SAFE_STOP"

    rearmed_status = service.arm_trading(actor="tester")
    assert rearmed_status.mode == "LIVE"
    assert rearmed_status.last_safe_stop_reason is None

    rearmed = service.process_closed_bar("2026-04-29T12:45:00Z", 101.0, model_direction="LONG", confidence=1.0)
    assert [intent.action for intent in rearmed.order_intents] == ["FLIP_TO_LONG"]


def test_warning_mode_restores_after_restart(tmp_path):
    service = _make_service(tmp_path, rolling_window_bars=2, min_bars_for_health=2)
    service.arm_trading(baseline_profit_per_bar=0.6, actor="tester")
    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)
    warning = service.process_closed_bar("2026-04-29T12:35:00Z", 100.8, model_direction="SHORT", confidence=1.0)

    assert warning.status.mode == "WARNING"

    restored = _make_service(tmp_path, rolling_window_bars=2, min_bars_for_health=2)
    assert restored.status.mode == "WARNING"
    assert restored.status.last_warning_reason == "EWMA_PROFIT_PER_BAR_DEGRADED"