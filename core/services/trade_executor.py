from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TradeState:
    position: int = 0
    entry_price: float | None = None
    entry_time: Any = None


@dataclass(frozen=True)
class ClosedTrade:
    side: int
    entry_price: float
    exit_price: float
    entry_time: Any = None
    exit_time: Any = None
    pnl: float = 0.0
    exit_reason: str = ""


@dataclass(frozen=True)
class TradeStepResult:
    state: TradeState
    action: str
    closed_trade: ClosedTrade | None = None
    reason: str = ""


def normalize_trade_signal(value: Any) -> int:
    if isinstance(value, (np.generic,)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"long", "buy", "up", "+1", "1"}:
            return 1
        if text in {"short", "sell", "down", "-1"}:
            return -1
        if text in {"hold", "flat", "neutral", "none", "", "0"}:
            return 0
    try:
        numeric = float(value)
    except Exception:
        return 0
    if numeric > 0:
        return 1
    if numeric < 0:
        return -1
    return 0


class TradeExecutor:
    def __init__(self, initial_state: TradeState | None = None) -> None:
        self.state = initial_state or TradeState()

    def step(
        self,
        signal: Any,
        price: float,
        timestamp: Any = None,
        *,
        step_reason: str | None = None,
        close_reason: str | None = None,
    ) -> TradeStepResult:
        target = normalize_trade_signal(signal)
        px = float(price)
        current = self.state.position

        if current == 0:
            if target == 1:
                self.state = TradeState(position=1, entry_price=px, entry_time=timestamp)
                return TradeStepResult(self.state, "ENTRY_LONG", reason=step_reason or "flat_to_long")
            if target == -1:
                self.state = TradeState(position=-1, entry_price=px, entry_time=timestamp)
                return TradeStepResult(self.state, "ENTRY_SHORT", reason=step_reason or "flat_to_short")
            return TradeStepResult(self.state, "FLAT", reason=step_reason or "stay_flat")

        if target == current:
            action = "HOLD_LONG" if current > 0 else "HOLD_SHORT"
            return TradeStepResult(self.state, action, reason=step_reason or "hold_position")

        exit_trade_reason = close_reason or ("exit_to_flat" if target == 0 else "flip_position")
        closed_trade = self._close_trade(px, timestamp, exit_trade_reason)
        if target == 0:
            self.state = TradeState()
            action = "EXIT_LONG" if current > 0 else "EXIT_SHORT"
            return TradeStepResult(self.state, action, closed_trade=closed_trade, reason=step_reason or "exit_to_flat")

        self.state = TradeState(position=target, entry_price=px, entry_time=timestamp)
        action = "FLIP_TO_LONG" if target > 0 else "FLIP_TO_SHORT"
        return TradeStepResult(self.state, action, closed_trade=closed_trade, reason=step_reason or "flip_position")

    def force_close(self, price: float, timestamp: Any = None, *, reason: str = "end_of_series") -> TradeStepResult:
        if self.state.position == 0:
            return TradeStepResult(self.state, "FLAT", reason="already_flat")
        px = float(price)
        current = self.state.position
        closed_trade = self._close_trade(px, timestamp, reason)
        self.state = TradeState()
        action = "EXIT_LONG" if current > 0 else "EXIT_SHORT"
        return TradeStepResult(self.state, action, closed_trade=closed_trade, reason=reason)

    def _close_trade(self, price: float, timestamp: Any, reason: str) -> ClosedTrade:
        entry = float(self.state.entry_price) if self.state.entry_price is not None else float(price)
        side = int(self.state.position)
        pnl = float(price - entry) if side > 0 else float(entry - price)
        return ClosedTrade(
            side=side,
            entry_price=entry,
            exit_price=float(price),
            entry_time=self.state.entry_time,
            exit_time=timestamp,
            pnl=pnl,
            exit_reason=reason,
        )


def replay_signals_over_market_data(
    signals: list[Any] | np.ndarray,
    prices: list[float] | np.ndarray,
    timestamps: list[Any] | np.ndarray | None = None,
    *,
    force_close: bool = True,
) -> dict[str, Any]:
    sig_arr = np.asarray(signals, dtype=object)
    px_arr = np.asarray(prices, dtype=float)
    n = min(sig_arr.size, px_arr.size)
    if n <= 0:
        return {
            "closed_trades": [],
            "trade_pnls": [],
            "trade_sides": [],
            "trade_exit_indices": [],
            "actions": [],
            "equity_curve": [],
            "closed_count_curve": [],
            "final_state": TradeState(),
        }

    ts_seq = list(timestamps[:n]) if timestamps is not None else [None] * n
    executor = TradeExecutor()
    closed_trades: list[ClosedTrade] = []
    actions: list[str] = []
    trade_exit_indices: list[int] = []
    equity_curve: list[float] = []
    closed_count_curve: list[int] = []
    realized = 0.0

    for idx in range(n):
        result = executor.step(sig_arr[idx], float(px_arr[idx]), ts_seq[idx])
        actions.append(result.action)
        if result.closed_trade is not None:
            closed_trades.append(result.closed_trade)
            trade_exit_indices.append(idx)
            realized += float(result.closed_trade.pnl)
        unrealized = 0.0
        if executor.state.position != 0 and executor.state.entry_price is not None:
            if executor.state.position > 0:
                unrealized = float(px_arr[idx] - executor.state.entry_price)
            else:
                unrealized = float(executor.state.entry_price - px_arr[idx])
        equity_curve.append(float(realized + unrealized))
        closed_count_curve.append(len(closed_trades))

    if force_close and executor.state.position != 0:
        result = executor.force_close(float(px_arr[n - 1]), ts_seq[n - 1], reason="end_of_series")
        if result.closed_trade is not None:
            closed_trades.append(result.closed_trade)
            trade_exit_indices.append(n - 1)
            realized += float(result.closed_trade.pnl)
            actions[-1] = result.action
            equity_curve[-1] = float(realized)
            closed_count_curve[-1] = len(closed_trades)

    return {
        "closed_trades": closed_trades,
        "trade_pnls": [float(t.pnl) for t in closed_trades],
        "trade_sides": [int(t.side) for t in closed_trades],
        "trade_exit_indices": trade_exit_indices,
        "actions": actions,
        "equity_curve": equity_curve,
        "closed_count_curve": closed_count_curve,
        "final_state": executor.state,
    }