from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from .live import (
    BaselineState,
    BrokerExecution,
    BrokerOrder,
    BrokerPosition,
    ExecutionJournal,
    PositionReconciler,
    PositionState,
    ProtectiveOrderRequest,
    ProtectiveOrdersManager,
    ProtectiveOrdersPlan,
    ReconciliationReport,
    RuntimeState,
    RuntimeStateStore,
)
from .signal_policy import DEFAULT_EXIT_POLICY, LivePolicyDecision, evaluate_live_policy, resolve_exit_policy_setting
from .trade_executor import TradeExecutor, TradeState, TradeStepResult

ServiceMode = Literal["OBSERVE", "LIVE", "WARNING", "SAFE_STOP", "EMERGENCY_STOP"]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return _utcnow_iso()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _normalize_signal_text(value: str | None) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"LONG", "SHORT"}:
        return text
    return None


def _position_to_trade_state(position: PositionState) -> TradeState:
    if position.side == "FLAT":
        return TradeState()
    return TradeState(
        position=1 if position.side == "LONG" else -1,
        entry_price=position.avg_price,
        entry_time=position.opened_at,
    )


def _trade_state_to_position(state: TradeState, quantity: float) -> PositionState:
    if state.position == 0:
        return PositionState.flat()
    return PositionState(
        side="LONG" if state.position > 0 else "SHORT",
        quantity=quantity,
        avg_price=state.entry_price,
        opened_at=state.entry_time,
    )


@dataclass(frozen=True)
class LiveTradingExecutionConfig:
    strategy_id: str
    instrument: str
    timeframe: str
    order_quantity: float = 1.0
    rolling_window_bars: int = 600
    min_bars_for_health: int = 20
    warning_alpha: float = 0.15
    warning_enter_ratio: float = 0.75
    warning_exit_ratio: float = 0.90
    safe_stop_ratio: float = 0.50
    freshness_timeout_sec: int = 600
    entry_threshold: float = 0.50
    exit_threshold: float = 0.50
    use_ma_alignment: bool = False
    exit_policy: str = DEFAULT_EXIT_POLICY

    def __post_init__(self) -> None:
        if float(self.order_quantity) <= 0.0:
            raise ValueError("order_quantity must be positive.")
        if int(self.rolling_window_bars) < 1:
            raise ValueError("rolling_window_bars must be >= 1.")
        if int(self.min_bars_for_health) < 1:
            raise ValueError("min_bars_for_health must be >= 1.")
        if not 0.0 < float(self.warning_alpha) <= 1.0:
            raise ValueError("warning_alpha must be in (0, 1].")
        if not 0.0 < float(self.safe_stop_ratio) <= 1.0:
            raise ValueError("safe_stop_ratio must be in (0, 1].")
        if not 0.0 < float(self.warning_enter_ratio) <= float(self.warning_exit_ratio) <= 1.0:
            raise ValueError("warning ratios must satisfy 0 < enter <= exit <= 1.")
        object.__setattr__(
            self,
            "exit_policy",
            resolve_exit_policy_setting(self.exit_policy, default=DEFAULT_EXIT_POLICY),
        )


@dataclass(frozen=True)
class ServiceOrderIntent:
    action: str
    target_side: str | None
    quantity: float
    price: float
    timestamp: str
    reason: str = ""


@dataclass(frozen=True)
class ServiceStatus:
    mode: ServiceMode
    runtime_state: RuntimeState
    live_profit_per_bar: float
    ewma_profit_per_bar: float | None
    baseline_profit_per_bar: float | None
    last_safe_stop_reason: str | None
    last_warning_reason: str | None


@dataclass(frozen=True)
class ProcessBarResult:
    status: ServiceStatus
    policy_decision: LivePolicyDecision
    trade_result: TradeStepResult | None
    order_intents: tuple[ServiceOrderIntent, ...] = ()
    protective_plan: ProtectiveOrdersPlan = field(default_factory=ProtectiveOrdersPlan)
    reconciliation_report: ReconciliationReport | None = None


class LiveTradingExecutionService:
    def __init__(
        self,
        config: LiveTradingExecutionConfig,
        *,
        journal: ExecutionJournal | None = None,
        state_store: RuntimeStateStore | None = None,
        reconciler: PositionReconciler | None = None,
        protective_orders_manager: ProtectiveOrdersManager | None = None,
    ) -> None:
        self.config = config
        self.journal = journal
        self.state_store = state_store
        self.reconciler = reconciler or PositionReconciler()
        self.protective_orders_manager = protective_orders_manager or ProtectiveOrdersManager()

        self.state = self._restore_runtime_state()
        self._mode: ServiceMode = self.state.extra.get("service_mode", "LIVE" if self.state.armed else "OBSERVE")
        self._last_safe_stop_reason: str | None = self.state.extra.get("safe_stop_reason")
        self._last_warning_reason: str | None = self.state.extra.get("warning_reason")
        self._rolling_bar_pnls: deque[float] = deque(
            (float(item) for item in self.state.extra.get("rolling_bar_pnls") or ()),
            maxlen=self.config.rolling_window_bars,
        )
        last_mark = self.state.extra.get("last_mark_price")
        self._last_mark_price: float | None = None if last_mark is None else float(last_mark)
        ewma = self.state.extra.get("ewma_profit_per_bar")
        self._ewma_profit_per_bar: float | None = None if ewma is None else float(ewma)
        self._trade_executor = TradeExecutor(_position_to_trade_state(self.state.position))

    @property
    def status(self) -> ServiceStatus:
        return ServiceStatus(
            mode=self._mode,
            runtime_state=self.state,
            live_profit_per_bar=self._current_profit_per_bar(),
            ewma_profit_per_bar=self._ewma_profit_per_bar,
            baseline_profit_per_bar=(self.state.baseline.profit_per_bar if self.state.baseline is not None else None),
            last_safe_stop_reason=self._last_safe_stop_reason,
            last_warning_reason=self._last_warning_reason,
        )

    def arm_trading(
        self,
        *,
        baseline_profit_per_bar: float | None = None,
        actor: str | None = None,
        reason: str | None = None,
        captured_at: Any | None = None,
    ) -> ServiceStatus:
        timestamp = _normalize_timestamp(captured_at)
        if not self.state.armed:
            self._append_event(
                "armed",
                {"actor": actor, "reason": reason or "arm_trading"},
                occurred_at=timestamp,
            )
            self.state = replace(self.state, armed=True)
        if baseline_profit_per_bar is not None and self.state.baseline is None:
            baseline = BaselineState(
                profit_per_bar=float(baseline_profit_per_bar),
                window_bars=self.config.rolling_window_bars,
                captured_at=timestamp,
                source="phase4_service",
            )
            self._append_event("baseline_captured", baseline.to_dict(), occurred_at=timestamp)
            self.state = replace(self.state, baseline=baseline)
        if self._mode != "EMERGENCY_STOP":
            # Při každém manuálním re-armu resetuj health window, aby staré záporné
            # MtM hodnoty z předchozí session nezapříčinily okamžitý SAFE_STOP.
            # min_bars_for_health pak dá systému čas nasbírat nová data.
            self._rolling_bar_pnls.clear()
            self._ewma_profit_per_bar = None
            self._last_mark_price = None
            self._mode = "LIVE"
            self._last_warning_reason = None
            self._last_safe_stop_reason = None
        self._persist_state()
        return self.status

    def disarm_trading(self, *, actor: str | None = None, reason: str | None = None) -> ServiceStatus:
        if self.state.armed:
            self._append_event(
                "disarmed",
                {"actor": actor, "reason": reason or "controlled_disarm"},
            )
            self.state = replace(self.state, armed=False)
        if self._mode not in {"SAFE_STOP", "EMERGENCY_STOP"}:
            self._mode = "OBSERVE"
            self._last_warning_reason = None
        self._persist_state()
        return self.status

    def emergency_stop(self, reason: str, *, actor: str | None = None) -> ServiceStatus:
        if self.state.armed:
            self._append_event(
                "disarmed",
                {"actor": actor, "reason": reason},
            )
        self._append_event(
            "operator_action",
            {"action": "EMERGENCY_STOP", "actor": actor, "reason": reason},
        )
        self.state = replace(self.state, armed=False)
        self._mode = "EMERGENCY_STOP"
        self._last_safe_stop_reason = reason
        self._last_warning_reason = None
        self._persist_state()
        return self.status

    def check_freshness(self, *, now: Any | None = None) -> ServiceStatus:
        if self.state.last_processed_closed_bar_at is None:
            return self.status
        current = datetime.fromisoformat(_normalize_timestamp(now).replace("Z", "+00:00"))
        last_bar = datetime.fromisoformat(self.state.last_processed_closed_bar_at.replace("Z", "+00:00"))
        delta = (current - last_bar).total_seconds()
        if delta > self.config.freshness_timeout_sec:
            self._enter_safe_stop("STALE_DATA")
        return self.status

    def reconcile_broker_state(
        self,
        *,
        broker_positions: Iterable[BrokerPosition],
        open_orders: Iterable[BrokerOrder],
        fills: Iterable[BrokerExecution],
    ) -> ReconciliationReport:
        report = self.reconciler.reconcile(self.state, broker_positions, open_orders, fills)
        if report.status == "SAFE_STOP":
            reason = report.safe_stop_reasons[0] if report.safe_stop_reasons else "BROKER_RECONCILIATION_FAILED"
            self._enter_safe_stop(reason)
        return report

    def process_closed_bar(
        self,
        bar_timestamp: Any,
        price: float,
        *,
        model_direction: str | None,
        confidence: float,
        ma_direction: str = "FLAT",
        broker_positions: Iterable[BrokerPosition] = (),
        open_orders: Iterable[BrokerOrder] = (),
        fills: Iterable[BrokerExecution] = (),
        protective_request: ProtectiveOrderRequest | None = None,
        baseline_profit_per_bar: float | None = None,
    ) -> ProcessBarResult:
        timestamp = _normalize_timestamp(bar_timestamp)
        price_value = float(price)
        self._capture_baseline_if_missing(baseline_profit_per_bar, timestamp)

        reconciliation_report = None
        if list(broker_positions) or list(open_orders) or list(fills):
            reconciliation_report = self.reconcile_broker_state(
                broker_positions=broker_positions,
                open_orders=open_orders,
                fills=fills,
            )

        self._append_event(
            "heartbeat",
            {"bar_timestamp": timestamp, "closed_bar_timestamp": timestamp},
            occurred_at=timestamp,
        )
        self.state = replace(
            self.state,
            strategy_id=self.config.strategy_id,
            instrument=self.config.instrument,
            timeframe=self.config.timeframe,
            last_seen_bar_at=timestamp,
            last_processed_closed_bar_at=timestamp,
        )

        policy_decision = evaluate_live_policy(
            ma_direction=ma_direction,
            model_direction=str(model_direction or "FLAT"),
            use_ma_alignment=self.config.use_ma_alignment,
            conf_min=float(confidence),
            live_position=self._trade_executor.state.position,
            entry_threshold=self.config.entry_threshold,
            exit_threshold=self.config.exit_threshold,
            block_entry=not self.state.armed or self._mode in {"SAFE_STOP", "EMERGENCY_STOP"},
            exit_policy=self.config.exit_policy,
        )

        if self._mode in {"SAFE_STOP", "EMERGENCY_STOP"}:
            self._append_event(
                "decision",
                {
                    "decision": _normalize_signal_text(policy_decision.final_signal),
                    "proposal": _normalize_signal_text(policy_decision.proposal),
                    "reason": policy_decision.reason,
                    "closed_bar_timestamp": timestamp,
                },
                occurred_at=timestamp,
            )
            self._persist_state()
            return ProcessBarResult(
                status=self.status,
                policy_decision=policy_decision,
                trade_result=None,
                reconciliation_report=reconciliation_report,
            )

        pre_step_position = int(self._trade_executor.state.position)
        bar_mark_to_market_pnl = 0.0
        if self._last_mark_price is not None and pre_step_position != 0:
            bar_mark_to_market_pnl = float((price_value - float(self._last_mark_price)) * float(pre_step_position))

        target_signal, trade_reason = self._resolve_target_signal(policy_decision)
        trade_result = self._trade_executor.step(
            target_signal,
            price_value,
            timestamp,
            step_reason=trade_reason,
            close_reason=(
                "controlled_disarm" if (not self.state.armed and self._trade_executor.state.position != 0) else policy_decision.close_reason
            ),
        )
        order_intents = self._build_order_intents(trade_result, price_value, timestamp)
        resulting_position = self._position_from_executor()

        if trade_result.closed_trade is not None:
            self._append_event(
                "trade_closed",
                {
                    "entry_time": trade_result.closed_trade.entry_time,
                    "exit_time": trade_result.closed_trade.exit_time,
                    "entry_price": float(trade_result.closed_trade.entry_price),
                    "exit_price": float(trade_result.closed_trade.exit_price),
                    "pnl": float(trade_result.closed_trade.pnl),
                    "side": int(trade_result.closed_trade.side),
                    "exit_reason": trade_result.closed_trade.exit_reason,
                },
                occurred_at=timestamp,
            )

        self._append_event(
            "decision",
            {
                "decision": _normalize_signal_text(target_signal),
                "proposal": _normalize_signal_text(policy_decision.proposal),
                "reason": trade_reason,
                "closed_bar_timestamp": timestamp,
            },
            occurred_at=timestamp,
        )
        self.state = replace(
            self.state,
            strategy_id=self.config.strategy_id,
            instrument=self.config.instrument,
            timeframe=self.config.timeframe,
            last_decision=_normalize_signal_text(target_signal),
            position=resulting_position,
        )

        protective_plan = ProtectiveOrdersPlan()
        if protective_request is not None and resulting_position.side != "FLAT":
            protective_plan = self.protective_orders_manager.plan(
                resulting_position,
                open_orders,
                protective_request,
            )

        self._rolling_bar_pnls.append(float(bar_mark_to_market_pnl))
        self._last_mark_price = float(price_value)
        self._update_ewma()
        self._evaluate_health()
        self._persist_state()

        return ProcessBarResult(
            status=self.status,
            policy_decision=policy_decision,
            trade_result=trade_result,
            order_intents=order_intents,
            protective_plan=protective_plan,
            reconciliation_report=reconciliation_report,
        )

    def _restore_runtime_state(self) -> RuntimeState:
        if self.state_store is not None:
            restored = self.state_store.restore(self.journal)
        elif self.journal is not None:
            restored = RuntimeState()
        else:
            restored = RuntimeState()
        extra = dict(restored.extra)
        extra["exit_policy"] = self.config.exit_policy
        return replace(
            restored,
            strategy_id=restored.strategy_id or self.config.strategy_id,
            instrument=restored.instrument or self.config.instrument,
            timeframe=restored.timeframe or self.config.timeframe,
            extra=extra,
        )

    def _capture_baseline_if_missing(self, baseline_profit_per_bar: float | None, timestamp: str) -> None:
        if baseline_profit_per_bar is None or self.state.baseline is not None:
            return
        baseline = BaselineState(
            profit_per_bar=float(baseline_profit_per_bar),
            window_bars=self.config.rolling_window_bars,
            captured_at=timestamp,
            source="phase4_service",
        )
        self._append_event("baseline_captured", baseline.to_dict(), occurred_at=timestamp)
        self.state = replace(self.state, baseline=baseline)

    def _resolve_target_signal(self, decision: LivePolicyDecision) -> tuple[str | None, str]:
        current_direction = "LONG" if self._trade_executor.state.position > 0 else "SHORT" if self._trade_executor.state.position < 0 else None
        target_signal = _normalize_signal_text(decision.final_signal)
        if self.state.armed:
            return target_signal, decision.reason
        if current_direction is None:
            return None, "observe_mode_block_entry"
        return None, "controlled_disarm_exit"

    def _position_from_executor(self) -> PositionState:
        quantity = self.state.position.quantity if self.state.position.side != "FLAT" else self.config.order_quantity
        return _trade_state_to_position(self._trade_executor.state, quantity)

    def _build_order_intents(
        self,
        trade_result: TradeStepResult,
        price: float,
        timestamp: str,
    ) -> tuple[ServiceOrderIntent, ...]:
        if trade_result.action in {"FLAT", "HOLD_LONG", "HOLD_SHORT"}:
            return ()
        side_map = {
            "ENTRY_LONG": "LONG",
            "ENTRY_SHORT": "SHORT",
            "EXIT_LONG": "FLAT",
            "EXIT_SHORT": "FLAT",
            "FLIP_TO_LONG": "LONG",
            "FLIP_TO_SHORT": "SHORT",
        }
        return (
            ServiceOrderIntent(
                action=trade_result.action,
                target_side=side_map.get(trade_result.action),
                quantity=self.config.order_quantity,
                price=price,
                timestamp=timestamp,
                reason=trade_result.reason,
            ),
        )

    def _current_profit_per_bar(self) -> float:
        if not self._rolling_bar_pnls:
            return 0.0
        return float(sum(self._rolling_bar_pnls) / len(self._rolling_bar_pnls))

    def _update_ewma(self) -> None:
        current = self._current_profit_per_bar()
        if self._ewma_profit_per_bar is None:
            self._ewma_profit_per_bar = current
            return
        alpha = float(self.config.warning_alpha)
        self._ewma_profit_per_bar = float(alpha * current + (1.0 - alpha) * self._ewma_profit_per_bar)

    def _evaluate_health(self) -> None:
        if self._mode == "EMERGENCY_STOP":
            return
        if len(self._rolling_bar_pnls) < self.config.min_bars_for_health:
            self._mode = "LIVE" if self.state.armed else "OBSERVE"
            self._last_warning_reason = None
            return
        baseline = self.state.baseline.profit_per_bar if self.state.baseline is not None else None
        live_profit_per_bar = self._current_profit_per_bar()
        if live_profit_per_bar < 0.0:
            self._enter_safe_stop("LIVE_PROFIT_PER_BAR_BELOW_ZERO")
            return
        if baseline is not None and baseline > 0.0 and live_profit_per_bar < baseline * self.config.safe_stop_ratio:
            self._enter_safe_stop("LIVE_PROFIT_PER_BAR_BELOW_BASELINE_FLOOR")
            return

        if not self.state.armed:
            self._mode = "OBSERVE"
            self._last_warning_reason = None
            return

        ewma = self._ewma_profit_per_bar if self._ewma_profit_per_bar is not None else live_profit_per_bar
        if baseline is None or baseline <= 0.0:
            self._mode = "LIVE"
            self._last_warning_reason = None
            return

        warning_floor = baseline * self.config.warning_enter_ratio
        warning_clear = baseline * self.config.warning_exit_ratio
        if self._mode == "WARNING":
            if ewma < warning_clear:
                self._last_warning_reason = "EWMA_PROFIT_PER_BAR_DEGRADED"
                return
            self._mode = "LIVE"
            self._last_warning_reason = None
            return

        if ewma < warning_floor:
            self._mode = "WARNING"
            self._last_warning_reason = "EWMA_PROFIT_PER_BAR_DEGRADED"
            return

        self._mode = "LIVE"
        self._last_warning_reason = None

    def _enter_safe_stop(self, reason: str) -> None:
        self._append_event(
            "operator_action",
            {"action": "SAFE_STOP", "reason": reason},
        )
        self.state = replace(self.state, armed=False)
        self._mode = "SAFE_STOP"
        self._last_safe_stop_reason = reason
        self._last_warning_reason = None
        self._persist_state()

    def _append_event(self, event_type: str, payload: dict[str, Any], *, occurred_at: str | None = None) -> None:
        if self.journal is None:
            return
        enriched_payload = dict(payload)
        enriched_payload.setdefault("exit_policy", self.config.exit_policy)
        self.journal.append_event(
            event_type,
            payload=enriched_payload,
            occurred_at=occurred_at or _utcnow_iso(),
            instrument=self.config.instrument,
            strategy_id=self.config.strategy_id,
        )

    def _persist_state(self) -> None:
        extra = dict(self.state.extra)
        extra.update(
            {
                "service_mode": self._mode,
                "safe_stop_reason": self._last_safe_stop_reason,
                "warning_reason": self._last_warning_reason,
                "rolling_bar_pnls": list(self._rolling_bar_pnls),
                "ewma_profit_per_bar": self._ewma_profit_per_bar,
                "last_mark_price": self._last_mark_price,
                "exit_policy": self.config.exit_policy,
            }
        )
        persisted = replace(
            self.state,
            strategy_id=self.config.strategy_id,
            instrument=self.config.instrument,
            timeframe=self.config.timeframe,
            extra=extra,
        )
        if self.state_store is not None:
            try:
                persisted = self.state_store.save(persisted)
            except OSError:
                pass
        self.state = persisted