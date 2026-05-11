import pandas as pd

from ibkr_trading_bot.core.services.trade_executor import TradeExecutor, replay_signals_over_market_data


def test_trade_executor_handles_entry_exit_and_flip():
    executor = TradeExecutor()

    step1 = executor.step("LONG", 101.0, "t1")
    step2 = executor.step("LONG", 102.0, "t2")
    step3 = executor.step("SHORT", 100.0, "t3")
    step4 = executor.step("FLAT", 99.0, "t4")

    assert step1.action == "ENTRY_LONG"
    assert step2.action == "HOLD_LONG"
    assert step3.action == "FLIP_TO_SHORT"
    assert step3.closed_trade is not None
    assert step3.closed_trade.pnl == -1.0
    assert step3.closed_trade.exit_reason == "flip_position"
    assert step4.action == "EXIT_SHORT"
    assert step4.closed_trade is not None
    assert step4.closed_trade.pnl == 1.0
    assert step4.closed_trade.exit_reason == "exit_to_flat"


def test_trade_executor_preserves_explicit_close_reason():
    executor = TradeExecutor()

    executor.step("LONG", 100.0, "t1")
    step = executor.step("FLAT", 99.0, "t2", step_reason="exit_low_confidence", close_reason="low_confidence")

    assert step.reason == "exit_low_confidence"
    assert step.closed_trade is not None
    assert step.closed_trade.exit_reason == "low_confidence"


def test_replay_signals_over_market_data_forces_last_close():
    prices = [100.0, 101.0, 103.0]
    timestamps = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    replay = replay_signals_over_market_data([0, 1, 1], prices, timestamps=timestamps)

    assert replay["trade_pnls"] == [2.0]
    assert replay["trade_sides"] == [1]
    assert replay["closed_count_curve"][-1] == 1
    assert replay["equity_curve"][-1] == 2.0