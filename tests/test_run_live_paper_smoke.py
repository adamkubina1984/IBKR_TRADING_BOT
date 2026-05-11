from __future__ import annotations

import importlib.util
from pathlib import Path

from ibkr_trading_bot.core.services.live import BrokerAccountState, TwsConnectionConfig
from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_live_paper_smoke.py"
    spec = importlib.util.spec_from_file_location("run_live_paper_smoke", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_live_paper_smoke_main_reports_blocked_real_mode_and_closed_trade(tmp_path, capsys):
    module = _load_script_module()

    exit_code = module.main(["--session-root", str(tmp_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "SMOKE_EXECUTION_MODE=PAPER" in output
    assert "SMOKE_GATE_ALLOWED=0" in output
    assert "SMOKE_REAL_BLOCKED=1" in output
    assert "AUTOMATED_TESTS_FAILED" in output
    assert "SMOKE_ACTIONS=ENTRY_LONG,FLIP_TO_SHORT" in output
    assert "SMOKE_CLOSED_TRADES=1" in output
    assert "SMOKE_FIRST_TRADE_PNL=1.0" in output
    assert "SMOKE_FINAL_POSITION=SHORT" in output
    assert "SMOKE_RESTORE_POSITION=SHORT" in output
    assert "SMOKE_RESTORE_CLOSED_TRADES=1" in output
    assert "SMOKE_RESTORE_LAST_PROCESSED_BAR=2026-04-29T12:35:00+00:00" in output
    assert "SMOKE_OK" in output


def test_run_live_paper_smoke_main_reports_connected_broker_when_enabled(tmp_path, monkeypatch, capsys):
    module = _load_script_module()

    class _ReadyPaperBrokerClient:
        def __init__(self):
            self.is_connected = False
            self.connected_client_id = 2

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def get_account_state(self, account=None):
            return BrokerAccountState(account=account or "DU123456", values=[])

    def _build_controller(args):
        return LiveServiceController(
            strategy_id=args.strategy_id,
            instrument=args.instrument,
            exchange=args.exchange,
            timeframe=args.timeframe,
            entry_threshold=args.entry_threshold,
            exit_threshold=args.exit_threshold,
            use_ma_alignment=False,
            freshness_timeout_sec=args.freshness_timeout_sec,
            session_root=Path(args.session_root),
            broker_connection=TwsConnectionConfig(
                host=args.ib_host,
                port=args.ib_port,
                client_id=args.ib_client_id,
                account=args.ib_account,
                readonly=args.ib_readonly,
            ),
            broker_client=_ReadyPaperBrokerClient(),
        )

    monkeypatch.setattr(module, "build_controller", _build_controller)

    exit_code = module.main(
        [
            "--session-root",
            str(tmp_path),
            "--paper-broker-check",
            "--ib-port",
            "7497",
            "--ib-client-id",
            "1",
            "--ib-account",
            "DU123456",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "SMOKE_BROKER_CHECK_ENABLED=1" in output
    assert "SMOKE_BROKER_PORT=7497" in output
    assert "SMOKE_BROKER_CLIENT_ID=1" in output
    assert "SMOKE_BROKER_ACCOUNT=DU123456" in output
    assert "SMOKE_BROKER_CONNECTED=1" in output
    assert "SMOKE_BROKER_CONNECTED_CLIENT_ID=2" in output
    assert "SMOKE_BROKER_SUMMARY=connected account=DU123456" in output
    assert "SMOKE_OK" in output