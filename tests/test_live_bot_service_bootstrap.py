from __future__ import annotations

from ibkr_trading_bot.core.services.live import FuturesContractSpec, TwsConnectionConfig
from ibkr_trading_bot.core.services.live_bot_service import LiveBotService


class _QualifiedContract:
    localSymbol = "GCZ6"
    symbol = "GC"


class _ExternalPaperClientStub:
    def __init__(self) -> None:
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.qualified_specs: list[FuturesContractSpec] = []
        self.ib = object()
        self.contract = _QualifiedContract()

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False

    def qualify_contract(self, spec: FuturesContractSpec):
        self.qualified_specs.append(spec)
        return self.contract


class _OwnedFakeIB:
    def __init__(self) -> None:
        self.connected = False
        self.disconnect_calls = 0

    def connect(self, host: str, port: int, *, clientId: int, readonly: bool) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def managedAccounts(self) -> list[str]:
        return ["DU123456"]

    def qualifyContracts(self, contract):
        return [contract]


def test_live_bot_service_bootstrap_uses_external_paper_client_without_rewiring_gui():
    client = _ExternalPaperClientStub()
    service = LiveBotService.from_paper_tws(
        TwsConnectionConfig(account="DU123456"),
        FuturesContractSpec(symbol="GC", expiry="202612"),
        "5m",
        broker_client=client,
    )

    assert client.connect_calls == 1
    assert client.qualified_specs[0].symbol == "GC"
    assert service.ib is client.ib
    assert service.contract is client.contract
    assert service.symbol == "GCZ6"
    assert service.bar_size == "5 mins"
    assert service.duration == "2 D"
    assert service.paper_tws_client is client

    service.close()

    assert client.disconnect_calls == 0


def test_live_bot_service_close_disconnects_owned_paper_client():
    fake_ib = _OwnedFakeIB()
    service = LiveBotService.from_paper_tws(
        TwsConnectionConfig(account="DU123456"),
        FuturesContractSpec(symbol="GC", expiry="202612"),
        "5m",
        ib_factory=lambda: fake_ib,
    )

    assert service.paper_tws_client is not None
    assert fake_ib.connected is True

    service.close()

    assert fake_ib.disconnect_calls == 1