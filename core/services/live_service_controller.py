from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Callable, Literal

from .live import ExecutionJournal, PaperSafeTwsClient, RuntimeStateStore, TwsConnectionConfig
from .live_release_gate import LiveReleaseGateInputs, LiveReleaseGateResult, evaluate_live_release_gate
from .live_trading_execution_service import LiveTradingExecutionConfig, LiveTradingExecutionService, ProcessBarResult, ServiceStatus
from .signal_policy import DEFAULT_EXIT_POLICY, resolve_exit_policy_setting

ExecutionMode = Literal["PAPER", "REAL"]


@dataclass(frozen=True)
class BrokerSessionStatus:
    enabled: bool
    connected: bool
    host: str | None = None
    port: int | None = None
    client_id: int | None = None
    account: str | None = None
    error: str | None = None

    @property
    def summary(self) -> str:
        if not self.enabled:
            return "startup check disabled"
        if self.connected:
            account_text = f" account={self.account}" if self.account else ""
            return f"connected{account_text}"
        return f"offline: {self.error or 'BROKER_OFFLINE'}"


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("_") or "default"


def _default_release_gate_inputs() -> LiveReleaseGateInputs:
    return LiveReleaseGateInputs(
        automated_tests_passed=False,
        paper_soak_completed=False,
        paper_soak_days=0,
        audit_trail_complete=False,
    )


class LiveServiceController:
    def __init__(
        self,
        *,
        strategy_id: str,
        instrument: str,
        exchange: str,
        timeframe: str,
        entry_threshold: float,
        exit_threshold: float,
        exit_policy: str = DEFAULT_EXIT_POLICY,
        use_ma_alignment: bool,
        freshness_timeout_sec: int,
        session_root: str | Path | None = None,
        release_gate_inputs: LiveReleaseGateInputs | None = None,
        broker_connection: TwsConnectionConfig | None = None,
        broker_client: PaperSafeTwsClient | None = None,
        ib_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.strategy_id = str(strategy_id)
        self.instrument = str(instrument)
        self.exchange = str(exchange)
        self.timeframe = str(timeframe)
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.exit_policy = resolve_exit_policy_setting(exit_policy, default=DEFAULT_EXIT_POLICY)
        self.use_ma_alignment = bool(use_ma_alignment)
        self.freshness_timeout_sec = int(freshness_timeout_sec)
        self._release_gate_inputs = release_gate_inputs or _default_release_gate_inputs()
        self._execution_mode: ExecutionMode = "PAPER"
        self._broker_connection = broker_connection
        self._broker_client = broker_client
        self._owns_broker_client = broker_client is None and broker_connection is not None
        self._ib_factory = ib_factory
        self.session_root = Path(session_root or (Path(__file__).resolve().parents[2] / ".live_service")).expanduser().resolve()
        self.session_dir = self.session_root / self._session_name()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._broker_status = BrokerSessionStatus(
            enabled=broker_connection is not None,
            connected=False,
            host=(broker_connection.host if broker_connection is not None else None),
            port=(int(broker_connection.port) if broker_connection is not None else None),
            client_id=(int(broker_connection.client_id) if broker_connection is not None else None),
            account=(broker_connection.account if broker_connection is not None else None),
            error=("Broker startup check not yet run." if broker_connection is not None else None),
        )
        self._service = self._build_service()

    @property
    def status(self) -> ServiceStatus:
        return self._service.status

    @property
    def release_gate_result(self) -> LiveReleaseGateResult:
        return evaluate_live_release_gate(self._release_gate_inputs)

    @property
    def execution_mode(self) -> ExecutionMode:
        return self._execution_mode

    @property
    def broker_status(self) -> BrokerSessionStatus:
        return self._broker_status

    def refresh_paper_broker_session(self) -> BrokerSessionStatus:
        if self._broker_connection is None:
            self._broker_status = BrokerSessionStatus(enabled=False, connected=False)
            return self._broker_status

        client = self._broker_client
        if client is None:
            client = PaperSafeTwsClient(self._broker_connection, ib_factory=self._ib_factory)
            self._broker_client = client
            self._owns_broker_client = True

        account = self._broker_connection.account
        try:
            if not bool(getattr(client, "is_connected", False)):
                client.connect()
            get_state = getattr(client, "get_account_state", None)
            if callable(get_state):
                try:
                    state = get_state(account)
                except TypeError:
                    state = get_state()
                account = account or getattr(state, "account", None)
            self._broker_status = BrokerSessionStatus(
                enabled=True,
                connected=True,
                host=self._broker_connection.host,
                port=int(self._broker_connection.port),
                client_id=(getattr(client, "connected_client_id", None) or int(self._broker_connection.client_id)),
                account=account,
                error=None,
            )
        except Exception as exc:
            self._broker_status = BrokerSessionStatus(
                enabled=True,
                connected=False,
                host=self._broker_connection.host,
                port=int(self._broker_connection.port),
                client_id=int(self._broker_connection.client_id),
                account=self._broker_connection.account,
                error=(str(exc).strip() or exc.__class__.__name__),
            )
        return self._broker_status

    def assert_paper_broker_ready(self) -> BrokerSessionStatus:
        status = self.refresh_paper_broker_session()
        if status.connected:
            return status
        raise RuntimeError(f"Paper broker startup check failed: {status.error or 'BROKER_OFFLINE'}")

    def assert_real_money_ready(self) -> LiveReleaseGateResult:
        result = self.release_gate_result
        if result.allowed:
            return result
        blocker_text = ", ".join(result.blockers) or "UNKNOWN_GATE_BLOCKER"
        raise RuntimeError(f"Real-money promotion blocked by release gate: {blocker_text}")

    def set_execution_mode(self, mode: str) -> ExecutionMode:
        normalized = str(mode or "").strip().upper()
        if normalized in {"PAPER", "PAPER_ONLY", "PAPER ONLY", "PAPER-ONLY"}:
            self._execution_mode = "PAPER"
            return self._execution_mode
        if normalized == "REAL":
            self.assert_real_money_ready()
            self._execution_mode = "REAL"
            return self._execution_mode
        raise ValueError(f"Unsupported execution mode: {mode}")

    def start_session(self) -> ServiceStatus:
        self.refresh_paper_broker_session()
        return self._service.status

    def stop_session(self) -> ServiceStatus:
        if self._owns_broker_client and self._broker_client is not None and self._broker_connection is not None:
            disconnect = getattr(self._broker_client, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass
            self._broker_status = BrokerSessionStatus(
                enabled=True,
                connected=False,
                host=self._broker_connection.host,
                port=int(self._broker_connection.port),
                client_id=int(self._broker_connection.client_id),
                account=self._broker_connection.account,
                error="Disconnected",
            )
        if self._service.state_store is not None:
            try:
                self._service.state = self._service.state_store.save(self._service.state)
            except OSError:
                pass
        return self._service.status

    def arm_trading(
        self,
        *,
        baseline_profit_per_bar: float | None = None,
        actor: str | None = None,
        reason: str | None = None,
    ) -> ServiceStatus:
        return self._service.arm_trading(
            baseline_profit_per_bar=baseline_profit_per_bar,
            actor=actor,
            reason=reason,
        )

    def disarm_trading(self, *, actor: str | None = None, reason: str | None = None) -> ServiceStatus:
        return self._service.disarm_trading(actor=actor, reason=reason)

    def emergency_stop(self, reason: str, *, actor: str | None = None) -> ServiceStatus:
        return self._service.emergency_stop(reason, actor=actor)

    def check_freshness(self, *, now: Any | None = None) -> ServiceStatus:
        return self._service.check_freshness(now=now)

    def process_closed_bar(
        self,
        bar_timestamp: Any,
        price: float,
        *,
        signal: str | None,
        baseline_profit_per_bar: float | None = None,
    ) -> ProcessBarResult:
        return self._service.process_closed_bar(
            bar_timestamp,
            price,
            model_direction=signal,
            confidence=1.0,
            baseline_profit_per_bar=baseline_profit_per_bar,
        )

    def list_closed_trades(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self._service.journal is None:
            return rows
        for event in self._service.journal.read_events():
            if event.event_type != "trade_closed":
                continue
            payload = dict(event.payload)
            side = int(payload.get("side", 0) or 0)
            rows.append(
                {
                    "entry_time": str(payload.get("entry_time") or "")[:19],
                    "direction": "LONG" if side > 0 else "SHORT",
                    "entry_price": float(payload.get("entry_price", 0.0) or 0.0),
                    "exit_time": str(payload.get("exit_time") or "")[:19],
                    "exit_price": float(payload.get("exit_price", 0.0) or 0.0),
                    "pnl": float(payload.get("pnl", 0.0) or 0.0),
                }
            )
        return rows

    def reset_session(self, *, clear_persistence: bool = True) -> ServiceStatus:
        if clear_persistence:
            for name in ("execution.journal.jsonl", "runtime.state.json"):
                path = self.session_dir / name
                if path.exists():
                    path.unlink()
        self._service = self._build_service()
        return self._service.status

    def _build_service(self) -> LiveTradingExecutionService:
        journal = ExecutionJournal(self.session_dir / "execution.journal.jsonl")
        state_store = RuntimeStateStore(self.session_dir / "runtime.state.json")
        config = LiveTradingExecutionConfig(
            strategy_id=self.strategy_id,
            instrument=self.instrument,
            timeframe=self.timeframe,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            exit_policy=self.exit_policy,
            use_ma_alignment=self.use_ma_alignment,
            freshness_timeout_sec=self.freshness_timeout_sec,
        )
        return LiveTradingExecutionService(config, journal=journal, state_store=state_store)

    def _session_name(self) -> str:
        return "__".join(
            [
                _slugify(self.strategy_id),
                _slugify(self.instrument),
                _slugify(self.exchange),
                _slugify(self.timeframe),
            ]
        )