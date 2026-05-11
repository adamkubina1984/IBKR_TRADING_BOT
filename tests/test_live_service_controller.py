from __future__ import annotations

import pytest

from ibkr_trading_bot.core.services.live import TwsConnectionConfig
from ibkr_trading_bot.core.services.live_release_gate import LiveReleaseGateInputs
from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController


class _ReadyPaperBrokerClient:
    def __init__(self, account: str = "DU123456", client_id: int = 1) -> None:
        self.account = account
        self.connected_client_id = client_id
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self) -> None:
        self.connect_calls += 1
        self.connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def get_account_state(self, account: str | None = None):
        resolved_account = account or self.account
        return type("AccountState", (), {"account": resolved_account})()


class _FailingPaperBrokerClient:
    def __init__(self, message: str = "Connection refused") -> None:
        self.message = message
        self.connect_calls = 0

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self) -> None:
        self.connect_calls += 1
        raise RuntimeError(self.message)

    def disconnect(self) -> None:
        return None


def _build_controller(tmp_path, **kwargs) -> LiveServiceController:
    return LiveServiceController(
        strategy_id="tab5-live",
        instrument="GOLD",
        exchange="TVC",
        timeframe="5 min",
        entry_threshold=0.55,
        exit_threshold=0.50,
        use_ma_alignment=False,
        freshness_timeout_sec=300,
        session_root=tmp_path,
        **kwargs,
    )


def test_live_service_controller_blocks_real_money_by_default(tmp_path):
    controller = _build_controller(tmp_path)

    result = controller.release_gate_result

    assert result.allowed is False
    assert "AUTOMATED_TESTS_FAILED" in result.blockers
    assert "PAPER_SOAK_NOT_COMPLETED" in result.blockers
    with pytest.raises(RuntimeError, match="AUTOMATED_TESTS_FAILED"):
        controller.assert_real_money_ready()


def test_live_service_controller_allows_real_money_after_clean_signoff(tmp_path):
    controller = _build_controller(
        tmp_path,
        release_gate_inputs=LiveReleaseGateInputs(
            automated_tests_passed=True,
            paper_soak_completed=True,
            paper_soak_days=4,
            audit_trail_complete=True,
        ),
    )

    result = controller.assert_real_money_ready()

    assert result.allowed is True
    assert result.blockers == ()


def test_live_service_controller_exposes_release_gate_warnings(tmp_path):
    controller = _build_controller(
        tmp_path,
        release_gate_inputs=LiveReleaseGateInputs(
            automated_tests_passed=True,
            paper_soak_completed=True,
            paper_soak_days=3,
            audit_trail_complete=True,
        ),
    )

    result = controller.release_gate_result

    assert result.allowed is True
    assert result.warnings == ("PAPER_SOAK_AT_MINIMUM_DURATION",)


def test_live_service_controller_keeps_paper_mode_when_real_money_is_blocked(tmp_path):
    controller = _build_controller(tmp_path)

    assert controller.execution_mode == "PAPER"

    with pytest.raises(RuntimeError, match="PAPER_SOAK_NOT_COMPLETED"):
        controller.set_execution_mode("REAL")

    assert controller.execution_mode == "PAPER"


def test_live_service_controller_switches_to_real_mode_when_gate_is_ready(tmp_path):
    controller = _build_controller(
        tmp_path,
        release_gate_inputs=LiveReleaseGateInputs(
            automated_tests_passed=True,
            paper_soak_completed=True,
            paper_soak_days=4,
            audit_trail_complete=True,
        ),
    )

    assert controller.set_execution_mode("REAL") == "REAL"
    assert controller.execution_mode == "REAL"
    assert controller.set_execution_mode("PAPER") == "PAPER"
    assert controller.execution_mode == "PAPER"


def test_live_service_controller_reports_connected_paper_broker_status(tmp_path):
    controller = _build_controller(
        tmp_path,
        broker_connection=TwsConnectionConfig(account="DU123456"),
        broker_client=_ReadyPaperBrokerClient(account="DU123456", client_id=2),
    )

    controller.start_session()

    assert controller.broker_status.enabled is True
    assert controller.broker_status.connected is True
    assert controller.broker_status.client_id == 2
    assert controller.broker_status.account == "DU123456"
    assert controller.assert_paper_broker_ready().connected is True


def test_live_service_controller_reports_offline_paper_broker_status(tmp_path):
    controller = _build_controller(
        tmp_path,
        broker_connection=TwsConnectionConfig(account="DU123456"),
        broker_client=_FailingPaperBrokerClient("IBKR TWS paper is offline"),
    )

    controller.start_session()

    assert controller.broker_status.enabled is True
    assert controller.broker_status.connected is False
    assert "offline" in controller.broker_status.summary
    assert "IBKR TWS paper is offline" in str(controller.broker_status.error)
    with pytest.raises(RuntimeError, match="IBKR TWS paper is offline"):
        controller.assert_paper_broker_ready()