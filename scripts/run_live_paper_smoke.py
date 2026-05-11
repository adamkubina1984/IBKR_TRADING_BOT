from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


# Ensure package root import works from scripts/
script_dir = os.path.dirname(os.path.abspath(__file__))
pkg_parent = os.path.dirname(script_dir)
repo_root = os.path.dirname(pkg_parent)
sys.path.insert(0, repo_root)

from ibkr_trading_bot.core.services.live import TwsConnectionConfig
from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0")
    return str(raw or "").strip().lower() not in {"0", "false", "no", "off"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a deterministic paper-mode smoke test for the live service controller.",
    )
    parser.add_argument("--session-root", default=".smoke_live_service", help="Directory used for journal/state artifacts.")
    parser.add_argument("--strategy-id", default="paper-smoke", help="Strategy identifier for the smoke session.")
    parser.add_argument("--instrument", default="GOLD", help="Instrument label stored in journal/state.")
    parser.add_argument("--exchange", default="TVC", help="Exchange label stored in session metadata.")
    parser.add_argument("--timeframe", default="5 min", help="Timeframe label stored in journal/state.")
    parser.add_argument("--entry-threshold", type=float, default=0.55, help="Policy entry threshold for the smoke run.")
    parser.add_argument("--exit-threshold", type=float, default=0.50, help="Policy exit threshold for the smoke run.")
    parser.add_argument("--freshness-timeout-sec", type=int, default=300, help="Freshness timeout passed to the controller.")
    parser.add_argument(
        "--baseline-profit-per-bar",
        type=float,
        default=0.8,
        help="Baseline used when arming the deterministic smoke run.",
    )
    parser.add_argument(
        "--paper-broker-check",
        action="store_true",
        help="Require a real paper IBKR startup check before the deterministic smoke replay.",
    )
    parser.add_argument("--ib-host", default=os.getenv("IBKR_TWS_HOST", "127.0.0.1"), help="IBKR paper host.")
    parser.add_argument("--ib-port", type=int, default=int(os.getenv("IBKR_TWS_PORT", "7497")), help="IBKR paper port.")
    parser.add_argument("--ib-client-id", type=int, default=int(os.getenv("IBKR_TWS_CLIENT_ID", "1")), help="IBKR client ID used for the paper startup check.")
    parser.add_argument("--ib-account", default=os.getenv("IBKR_PAPER_ACCOUNT"), help="Optional DU account to require for the paper startup check.")
    parser.add_argument(
        "--ib-readonly",
        dest="ib_readonly",
        action="store_true",
        default=_env_bool("IBKR_TWS_READONLY", True),
        help="Use readonly mode for the broker startup check.",
    )
    parser.add_argument(
        "--ib-readwrite",
        dest="ib_readonly",
        action="store_false",
        help="Disable readonly mode for the broker startup check.",
    )
    return parser


def _build_broker_connection(args: argparse.Namespace) -> TwsConnectionConfig | None:
    if not bool(getattr(args, "paper_broker_check", False)):
        return None
    return TwsConnectionConfig(
        host=str(args.ib_host or "127.0.0.1").strip() or "127.0.0.1",
        port=int(args.ib_port),
        client_id=int(args.ib_client_id),
        account=(str(args.ib_account or "").strip() or None),
        readonly=bool(args.ib_readonly),
    )


def build_controller(args: argparse.Namespace) -> LiveServiceController:
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
        broker_connection=_build_broker_connection(args),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controller = build_controller(args)
    controller.reset_session(clear_persistence=True)

    print(f"SMOKE_SESSION_DIR={controller.session_dir.as_posix()}")
    print(f"SMOKE_EXECUTION_MODE={controller.execution_mode}")
    print(f"SMOKE_BROKER_CHECK_ENABLED={int(bool(args.paper_broker_check))}")

    if args.paper_broker_check:
        print(f"SMOKE_BROKER_HOST={args.ib_host}")
        print(f"SMOKE_BROKER_PORT={int(args.ib_port)}")
        print(f"SMOKE_BROKER_CLIENT_ID={int(args.ib_client_id)}")
        print(f"SMOKE_BROKER_ACCOUNT={str(args.ib_account or 'AUTO')}")
        print(f"SMOKE_BROKER_READONLY={int(bool(args.ib_readonly))}")
        try:
            broker_status = controller.assert_paper_broker_ready()
        except RuntimeError as exc:
            print("SMOKE_BROKER_CONNECTED=0")
            print(f"SMOKE_BROKER_ERROR={str(exc)}")
            controller.stop_session()
            return 10
        print("SMOKE_BROKER_CONNECTED=1")
        print(f"SMOKE_BROKER_CONNECTED_CLIENT_ID={int(getattr(broker_status, 'client_id', None) or args.ib_client_id)}")
        print(f"SMOKE_BROKER_SUMMARY={broker_status.summary}")

    gate = controller.release_gate_result
    print(f"SMOKE_GATE_ALLOWED={int(gate.allowed)}")
    print(f"SMOKE_GATE_BLOCKERS={','.join(gate.blockers) if gate.blockers else 'NONE'}")
    print(f"SMOKE_GATE_WARNINGS={','.join(gate.warnings) if gate.warnings else 'NONE'}")

    try:
        controller.set_execution_mode("REAL")
    except RuntimeError as exc:
        print(f"SMOKE_REAL_BLOCKED=1")
        print(f"SMOKE_REAL_BLOCK_REASON={str(exc)}")
    else:
        print("SMOKE_REAL_BLOCKED=0")
        print("ERROR: REAL mode unexpectedly unlocked during paper smoke")
        return 3

    controller.set_execution_mode("PAPER")
    controller.start_session()
    armed = controller.arm_trading(
        baseline_profit_per_bar=args.baseline_profit_per_bar,
        actor="paper_smoke",
        reason="deterministic_cli_smoke",
    )
    print(f"SMOKE_ARMED_MODE={armed.mode}")
    print(f"SMOKE_BASELINE={armed.baseline_profit_per_bar}")

    first = controller.process_closed_bar(
        "2026-04-29T12:30:00Z",
        100.0,
        signal="LONG",
    )
    second = controller.process_closed_bar(
        "2026-04-29T12:35:00Z",
        101.0,
        signal="SHORT",
    )
    trades = controller.list_closed_trades()
    controller.stop_session()

    first_action = first.trade_result.action if first.trade_result is not None else "NONE"
    second_action = second.trade_result.action if second.trade_result is not None else "NONE"
    print(f"SMOKE_ACTIONS={first_action},{second_action}")
    print(f"SMOKE_FINAL_MODE={controller.status.mode}")
    print(f"SMOKE_FINAL_POSITION={controller.status.runtime_state.position.side}")
    print(f"SMOKE_CLOSED_TRADES={len(trades)}")
    if trades:
        print(f"SMOKE_FIRST_TRADE_PNL={trades[0]['pnl']}")
        print(f"SMOKE_FIRST_TRADE_DIRECTION={trades[0]['direction']}")

    if len(trades) != 1:
        print("ERROR: expected exactly one closed trade in deterministic smoke run")
        return 4
    if second_action != "FLIP_TO_SHORT":
        print(f"ERROR: expected second action FLIP_TO_SHORT, got {second_action}")
        return 5
    if controller.execution_mode != "PAPER":
        print(f"ERROR: expected controller to remain in PAPER mode, got {controller.execution_mode}")
        return 6

    restored_controller = build_controller(args)
    restored_trades = restored_controller.list_closed_trades()
    restored_status = restored_controller.status
    print(f"SMOKE_RESTORE_MODE={restored_status.mode}")
    print(f"SMOKE_RESTORE_POSITION={restored_status.runtime_state.position.side}")
    print(f"SMOKE_RESTORE_CLOSED_TRADES={len(restored_trades)}")
    print(
        "SMOKE_RESTORE_LAST_PROCESSED_BAR="
        f"{restored_status.runtime_state.last_processed_closed_bar_at or 'NONE'}"
    )

    if len(restored_trades) != 1:
        print(f"ERROR: expected one restored closed trade, got {len(restored_trades)}")
        return 7
    if restored_status.runtime_state.position.side != "SHORT":
        print(
            "ERROR: expected restored runtime position SHORT, "
            f"got {restored_status.runtime_state.position.side}"
        )
        return 8
    if restored_status.runtime_state.last_processed_closed_bar_at != "2026-04-29T12:35:00+00:00":
        print(
            "ERROR: expected restored last_processed_closed_bar_at 2026-04-29T12:35:00+00:00, "
            f"got {restored_status.runtime_state.last_processed_closed_bar_at}"
        )
        return 9

    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("SMOKE_FAILED")
        traceback.print_exc()
        raise