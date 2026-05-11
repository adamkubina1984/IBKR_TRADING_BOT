from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ibkr_trading_bot.core.services.live import PaperSafeTwsClient, TwsConnectionConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-safe IBKR connectivity check.")
    parser.add_argument("--host", default=os.getenv("IBKR_TWS_HOST", "127.0.0.1"), help="IBKR host.")
    parser.add_argument("--port", type=int, default=int(os.getenv("IBKR_TWS_PORT", "7497")), help="IBKR paper port (7497 TWS or 4002 Gateway).")
    parser.add_argument("--client-id", type=int, default=int(os.getenv("IBKR_TWS_CLIENT_ID", "1")), help="Client ID used for the check.")
    parser.add_argument("--account", default=os.getenv("IBKR_PAPER_ACCOUNT"), help="Optional DU paper account to require.")
    parser.add_argument("--readonly", action="store_true", default=os.getenv("IBKR_TWS_READONLY", "1").lower() not in ("0", "false", "no"), help="Connect in readonly mode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"PAPER_CHECK_HOST={args.host}")
    print(f"PAPER_CHECK_PORT={args.port}")
    print(f"PAPER_CHECK_CLIENT_ID={args.client_id}")
    print(f"PAPER_CHECK_ACCOUNT={args.account or 'AUTO'}")
    print(f"PAPER_CHECK_READONLY={int(bool(args.readonly))}")

    try:
        config = TwsConnectionConfig(
            host=str(args.host or "127.0.0.1").strip() or "127.0.0.1",
            port=int(args.port),
            client_id=int(args.client_id),
            account=(str(args.account or "").strip() or None),
            readonly=bool(args.readonly),
        )
        client = PaperSafeTwsClient(config)
        try:
            client.connect()
            resolved_client_id = getattr(client, "connected_client_id", None) or int(config.client_id)
            managed_accounts = client._managed_accounts(client.ib)
            state = client.get_account_state(config.account)
        finally:
            client.disconnect()
    except Exception as exc:
        print("PAPER_CHECK_CONNECTED=0")
        error_text = str(exc).strip() or exc.__class__.__name__
        print(f"PAPER_CHECK_ERROR={error_text}")
        return 2

    print("PAPER_CHECK_CONNECTED=1")
    print(f"PAPER_CHECK_CONNECTED_CLIENT_ID={resolved_client_id}")
    print(f"PAPER_CHECK_MANAGED_ACCOUNTS={','.join(managed_accounts) if managed_accounts else 'NONE'}")
    print(f"PAPER_CHECK_ACCOUNT_SUMMARY_ACCOUNT={state.account or 'NONE'}")
    print(f"PAPER_CHECK_SUMMARY_KEYS={','.join(item.key for item in state.values[:8]) if state.values else 'NONE'}")
    print("PAPER_CHECK_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("PAPER_CHECK_FAILED")
        traceback.print_exc()
        raise
