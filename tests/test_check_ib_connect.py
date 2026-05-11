from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_ib_connect.py"
    spec = importlib.util.spec_from_file_location("check_ib_connect", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_ib_connect_main_reports_connected_paper_session(monkeypatch, capsys):
    module = _load_script_module()

    class _FakeValue:
        def __init__(self, key: str) -> None:
            self.key = key
            self.value = "1"
            self.currency = "USD"
            self.account = "DU123456"

    class _FakeClient:
        def __init__(self, config):
            self.config = config
            self.connected = False
            self.connected_client_id = 4
            self.ib = object()

        def connect(self):
            self.connected = True

        def disconnect(self):
            self.connected = False

        def _managed_accounts(self, _ib):
            return ["DU123456"]

        def get_account_state(self, _account=None):
            return type("AccountState", (), {"account": "DU123456", "values": (_FakeValue("NetLiquidation"), _FakeValue("BuyingPower"))})()

    monkeypatch.setattr(module, "PaperSafeTwsClient", _FakeClient)

    exit_code = module.main(["--port", "4002", "--account", "DU123456"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "PAPER_CHECK_PORT=4002" in output
    assert "PAPER_CHECK_ACCOUNT=DU123456" in output
    assert "PAPER_CHECK_CONNECTED=1" in output
    assert "PAPER_CHECK_CONNECTED_CLIENT_ID=4" in output
    assert "PAPER_CHECK_MANAGED_ACCOUNTS=DU123456" in output
    assert "PAPER_CHECK_SUMMARY_KEYS=NetLiquidation,BuyingPower" in output
    assert "PAPER_CHECK_OK" in output


def test_check_ib_connect_main_reports_error_when_connection_fails(monkeypatch, capsys):
    module = _load_script_module()

    class _FailingClient:
        def __init__(self, config):
            self.config = config

        def connect(self):
            raise RuntimeError("Connection refused")

        def disconnect(self):
            return None

    monkeypatch.setattr(module, "PaperSafeTwsClient", _FailingClient)

    exit_code = module.main(["--port", "4002"])
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "PAPER_CHECK_CONNECTED=0" in output
    assert "PAPER_CHECK_ERROR=Connection refused" in output