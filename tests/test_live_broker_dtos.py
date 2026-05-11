from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ibkr_trading_bot.core.services.live import (
    BrokerAccountState,
    BrokerBar,
    BrokerExecution,
    BrokerOrder,
    BrokerPosition,
    FuturesContractSpec,
    HistoricalBarsRequest,
    PaperSafeTwsClient,
    PaperTradingGuardError,
    TwsConnectionConfig,
)


class _FakeBar:
    date = datetime(2026, 4, 29, 12, 30, tzinfo=timezone.utc)
    open = 100.0
    high = 102.0
    low = 99.5
    close = 101.5
    volume = 42
    average = 101.0
    barCount = 3


class _FakeIB:
    def __init__(self, accounts: list[str]) -> None:
        self.accounts = accounts
        self.connected = False
        self.connect_calls: list[dict[str, object]] = []
        self.placed_orders: list[tuple[object, object]] = []
        self.cancelled_orders: list[object] = []

    def connect(self, host: str, port: int, *, clientId: int, readonly: bool) -> None:
        self.connect_calls.append(
            {
                "host": host,
                "port": port,
                "clientId": clientId,
                "readonly": readonly,
            }
        )
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def managedAccounts(self) -> list[str]:
        return list(self.accounts)

    def qualifyContracts(self, contract):
        return [contract]

    def reqHistoricalData(self, contract, **kwargs):
        return [_FakeBar()]

    def positions(self):
        contract = type("Contract", (), {"symbol": "GC", "exchange": "COMEX", "currency": "USD", "localSymbol": "GCZ6"})()
        return [("DU123456", contract, 2.0, 101.25)]

    def openTrades(self):
        contract = type("Contract", (), {"symbol": "GC", "localSymbol": "GCZ6"})()
        order = type(
            "Order",
            (),
            {
                "orderId": 11,
                "permId": 22,
                "parentId": 0,
                "account": "DU123456",
                "action": "BUY",
                "orderType": "LMT",
                "totalQuantity": 2,
                "lmtPrice": 100.75,
                "auxPrice": None,
                "tif": "DAY",
                "outsideRth": False,
            },
        )()
        order_status = type("OrderStatus", (), {"filled": 1.0, "remaining": 1.0, "status": "Submitted"})()
        return [type("Trade", (), {"contract": contract, "order": order, "orderStatus": order_status})()]

    def fills(self):
        contract = type("Contract", (), {"symbol": "GC", "localSymbol": "GCZ6", "currency": "USD"})()
        execution = type(
            "Execution",
            (),
            {
                "execId": "fill-1",
                "orderId": 11,
                "permId": 22,
                "acctNumber": "DU123456",
                "side": "BOT",
                "shares": 1,
                "price": 100.8,
                "time": "2026-04-29T12:31:00Z",
            },
        )()
        commission = type("CommissionReport", (), {"commission": 1.25, "realizedPNL": 4.5, "currency": "USD"})()
        return [type("Fill", (), {"contract": contract, "execution": execution, "commissionReport": commission})()]

    def accountSummary(self, account=None):
        account_code = account or self.accounts[0]
        value = type("AccountValue", (), {"tag": "NetLiquidation", "value": "100000", "currency": "USD", "account": account_code})()
        return [value]

    def placeOrder(self, contract, order):
        self.placed_orders.append((contract, order))
        order_status = type("OrderStatus", (), {"filled": 0.0, "remaining": 1.0, "status": "Submitted"})()
        return type("Trade", (), {"contract": contract, "order": order, "orderStatus": order_status})()

    def cancelOrder(self, order):
        self.cancelled_orders.append(order)
        return True


class _FakeEvent:
    def __init__(self) -> None:
        self.handlers: list[object] = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers = [item for item in self.handlers if item is not handler]
        return self

    def emit(self, req_id, error_code, error_string, contract=None):
        for handler in list(self.handlers):
            handler(req_id, error_code, error_string, contract)


class _ClientIdInUseIB(_FakeIB):
    def __init__(self) -> None:
        super().__init__(["DU123456"])
        self.errorEvent = _FakeEvent()

    def connect(self, host: str, port: int, *, clientId: int, readonly: bool) -> None:
        super().connect(host, port, clientId=clientId, readonly=readonly)
        self.errorEvent.emit(
            -1,
            326,
            "Unable to connect as the client id is already in use. Retry with a unique client id.",
        )
        raise TimeoutError()


class _RetryOnBusyIB(_FakeIB):
    def __init__(self, accounts: list[str], attempts: list[int]) -> None:
        super().__init__(accounts)
        self.errorEvent = _FakeEvent()
        self._attempts = attempts

    def connect(self, host: str, port: int, *, clientId: int, readonly: bool) -> None:
        self._attempts.append(clientId)
        if clientId == 1:
            self.connect_calls.append(
                {
                    "host": host,
                    "port": port,
                    "clientId": clientId,
                    "readonly": readonly,
                }
            )
            self.errorEvent.emit(
                -1,
                326,
                "Unable to connect as the client id is already in use. Retry with a unique client id.",
            )
            raise TimeoutError()
        super().connect(host, port, clientId=clientId, readonly=readonly)


def test_tws_connection_config_enforces_paper_tws_guardrails():
    with pytest.raises(ValueError):
        TwsConnectionConfig(port=7496)

    with pytest.raises(ValueError):
        TwsConnectionConfig(port=4001)

    with pytest.raises(ValueError):
        TwsConnectionConfig(account="U123456")

    config = TwsConnectionConfig()
    assert config.port == 7497
    assert config.readonly is True

    gateway_config = TwsConnectionConfig(port=4002)
    assert gateway_config.port == 4002


def test_broker_dtos_round_trip_and_bar_conversion():
    contract = FuturesContractSpec(symbol="gc", expiry="202612")
    bars_request = HistoricalBarsRequest(duration="2 D", bar_size="5 mins", keep_up_to_date=True)

    restored_contract = FuturesContractSpec.from_dict(contract.to_dict())
    restored_request = HistoricalBarsRequest.from_dict(bars_request.to_dict())
    restored_bar = BrokerBar.from_dict(BrokerBar.from_ib_bar(_FakeBar()).to_dict())
    restored_position = BrokerPosition.from_dict(
        BrokerPosition(account="DU123456", symbol="GC", quantity=2, side="LONG", avg_cost=101.25).to_dict()
    )
    restored_order = BrokerOrder.from_dict(
        BrokerOrder(order_id="11", status="Submitted", total_quantity=2, remaining_quantity=1).to_dict()
    )
    restored_execution = BrokerExecution.from_dict(
        BrokerExecution(execution_id="fill-1", quantity=1, side="BOT", price=100.8).to_dict()
    )
    restored_account = BrokerAccountState.from_dict(
        BrokerAccountState.from_ib_values(_FakeIB(["DU123456"]).accountSummary("DU123456")).to_dict()
    )

    assert restored_contract == contract
    assert restored_request == bars_request
    assert restored_bar.close == 101.5
    assert restored_bar.timestamp.startswith("2026-04-29T12:30:00")
    assert restored_position.side == "LONG"
    assert restored_order.status == "SUBMITTED"
    assert restored_execution.side == "BOT"
    assert restored_account.values[0].key == "NetLiquidation"


def test_paper_safe_tws_client_rejects_non_paper_accounts_and_maps_historical_bars_and_reads_state():
    rejecting_client = PaperSafeTwsClient(TwsConnectionConfig(), ib_factory=lambda: _FakeIB(["U123456"]))
    with pytest.raises(PaperTradingGuardError):
        rejecting_client.connect()

    fake_ib = _FakeIB(["DU123456"])
    paper_client = PaperSafeTwsClient(TwsConnectionConfig(account="DU123456"), ib_factory=lambda: fake_ib)
    paper_client.connect()

    bars = paper_client.request_historical_bars(
        FuturesContractSpec(symbol="GC", expiry="202612"),
        HistoricalBarsRequest(),
    )

    assert paper_client.is_connected is True
    assert fake_ib.connect_calls[0]["readonly"] is True
    assert len(bars) == 1
    assert bars[0].volume == 42.0
    assert paper_client.get_positions()[0].symbol == "GC"
    assert paper_client.get_open_orders()[0].status == "SUBMITTED"
    assert paper_client.get_fills()[0].execution_id == "fill-1"
    assert paper_client.get_account_state().values[0].key == "NetLiquidation"

    with pytest.raises(PaperTradingGuardError):
        paper_client.place_order(FuturesContractSpec(symbol="GC", expiry="202612"), object())

    writable_ib = _FakeIB(["DU123456"])
    writable_client = PaperSafeTwsClient(
        TwsConnectionConfig(account="DU123456", readonly=False),
        ib_factory=lambda: writable_ib,
    )
    writable_client.connect()
    order = type(
        "Order",
        (),
        {
            "orderId": 14,
            "permId": 33,
            "parentId": 0,
            "account": "DU123456",
            "action": "BUY",
            "orderType": "MKT",
            "totalQuantity": 1,
            "lmtPrice": None,
            "auxPrice": None,
            "tif": "DAY",
            "outsideRth": False,
        },
    )()
    placed = writable_client.place_order(FuturesContractSpec(symbol="GC", expiry="202612"), order)

    assert placed.order_id == "14"
    assert writable_ib.placed_orders

    assert writable_client.cancel_order(order) is True
    assert writable_ib.cancelled_orders == [order]


def test_paper_safe_tws_client_surfaces_client_id_in_use_error():
    fake_ib = _ClientIdInUseIB()
    client = PaperSafeTwsClient(TwsConnectionConfig(client_id=2, client_id_retry_span=0), ib_factory=lambda: fake_ib)

    with pytest.raises(RuntimeError, match="client ID 2: already in use"):
        client.connect()

    assert fake_ib.connected is False
    assert fake_ib.errorEvent.handlers == []


def test_paper_safe_tws_client_retries_next_client_id_when_requested_one_is_busy():
    attempts: list[int] = []
    client = PaperSafeTwsClient(
        TwsConnectionConfig(client_id=1, client_id_retry_span=2, account="DU123456"),
        ib_factory=lambda: _RetryOnBusyIB(["DU123456"], attempts),
    )

    client.connect()

    assert client.is_connected is True
    assert client.connected_client_id == 2
    assert attempts == [1, 2]

    client.disconnect()
    assert client.connected_client_id is None