from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

PaperMode = Literal["paper"]
ContractMode = Literal["FUT", "CONT"]
PositionSide = Literal["FLAT", "LONG", "SHORT"]


def _normalize_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        for parser in (
            lambda candidate: datetime.fromisoformat(candidate.replace("Z", "+00:00")),
            lambda candidate: datetime.strptime(candidate, "%Y%m%d %H:%M:%S"),
            lambda candidate: datetime.strptime(candidate, "%Y%m%d"),
        ):
            try:
                parsed = parser(text)
                break
            except ValueError:
                continue
        else:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_position_side(value: Any, *, quantity: float = 0.0) -> PositionSide:
    raw = str(value or "").strip().upper()
    if raw in {"BOT", "BUY", "LONG", "B"}:
        return "LONG"
    if raw in {"SLD", "SELL", "SHORT", "S"}:
        return "SHORT"
    if raw == "FLAT":
        return "FLAT"
    if quantity > 0.0:
        return "LONG"
    if quantity < 0.0:
        return "SHORT"
    return "FLAT"


def _normalize_order_action(value: Any) -> str | None:
    action = _normalize_text(value)
    return action.upper() if action else None


def _normalize_order_status(value: Any) -> str | None:
    status = _normalize_text(value)
    return status.upper() if status else None


@dataclass(frozen=True)
class TwsConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    client_id_retry_span: int = 3
    account: str | None = None
    readonly: bool = True
    mode: PaperMode = "paper"
    allow_custom_tws_paper_port: bool = False

    def __post_init__(self) -> None:
        if self.mode != "paper":
            raise ValueError("Only paper trading mode is supported in Phase 1.")
        if self.port in {4001, 7496}:
            raise ValueError("Live IBKR ports are blocked by the paper-only guardrail.")
        if self.port not in {4002, 7497} and not self.allow_custom_tws_paper_port:
            raise ValueError(
                "Paper-safe IBKR connections use port 7497 (TWS) or 4002 (IB Gateway) unless allow_custom_tws_paper_port is enabled."
            )
        if int(self.client_id_retry_span) < 0:
            raise ValueError("client_id_retry_span must be >= 0.")
        if self.account and not self.account.upper().startswith("DU"):
            raise ValueError("Paper accounts must use the IBKR DU prefix.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "client_id_retry_span": self.client_id_retry_span,
            "account": self.account,
            "readonly": self.readonly,
            "mode": self.mode,
            "allow_custom_tws_paper_port": self.allow_custom_tws_paper_port,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TwsConnectionConfig":
        return cls(
            host=str(payload.get("host", "127.0.0.1")),
            port=int(payload.get("port", 7497)),
            client_id=int(payload.get("client_id", 1)),
            client_id_retry_span=int(payload.get("client_id_retry_span", 3)),
            account=payload.get("account"),
            readonly=bool(payload.get("readonly", True)),
            mode=str(payload.get("mode", "paper")),
            allow_custom_tws_paper_port=bool(payload.get("allow_custom_tws_paper_port", False)),
        )


@dataclass(frozen=True)
class FuturesContractSpec:
    symbol: str
    expiry: str | None = None
    exchange: str = "COMEX"
    currency: str = "USD"
    contract_mode: ContractMode = "FUT"
    include_expired: bool = True
    multiplier: str | None = None
    local_symbol: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "exchange", self.exchange.strip().upper())
        object.__setattr__(self, "currency", self.currency.strip().upper())
        object.__setattr__(self, "contract_mode", self.contract_mode.strip().upper())
        if self.contract_mode not in {"FUT", "CONT"}:
            raise ValueError("contract_mode must be either FUT or CONT.")
        if self.contract_mode == "FUT" and not str(self.expiry or "").strip():
            raise ValueError("expiry is required for FUT contracts.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "expiry": self.expiry,
            "exchange": self.exchange,
            "currency": self.currency,
            "contract_mode": self.contract_mode,
            "include_expired": self.include_expired,
            "multiplier": self.multiplier,
            "local_symbol": self.local_symbol,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FuturesContractSpec":
        return cls(
            symbol=str(payload["symbol"]),
            expiry=payload.get("expiry"),
            exchange=str(payload.get("exchange", "COMEX")),
            currency=str(payload.get("currency", "USD")),
            contract_mode=str(payload.get("contract_mode", "FUT")),
            include_expired=bool(payload.get("include_expired", True)),
            multiplier=payload.get("multiplier"),
            local_symbol=payload.get("local_symbol"),
        )


@dataclass(frozen=True)
class HistoricalBarsRequest:
    duration: str = "2 D"
    bar_size: str = "5 mins"
    what_to_show: str = "TRADES"
    use_rth: bool = False
    keep_up_to_date: bool = False
    end_datetime: str = ""
    format_date: int = 1

    def __post_init__(self) -> None:
        if not self.duration.strip():
            raise ValueError("duration must not be empty.")
        if not self.bar_size.strip():
            raise ValueError("bar_size must not be empty.")
        if not self.what_to_show.strip():
            raise ValueError("what_to_show must not be empty.")

    def to_ib_kwargs(self) -> dict[str, Any]:
        return {
            "endDateTime": self.end_datetime,
            "durationStr": self.duration,
            "barSizeSetting": self.bar_size,
            "whatToShow": self.what_to_show,
            "useRTH": int(self.use_rth),
            "formatDate": self.format_date,
            "keepUpToDate": self.keep_up_to_date,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "bar_size": self.bar_size,
            "what_to_show": self.what_to_show,
            "use_rth": self.use_rth,
            "keep_up_to_date": self.keep_up_to_date,
            "end_datetime": self.end_datetime,
            "format_date": self.format_date,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HistoricalBarsRequest":
        return cls(
            duration=str(payload.get("duration", "2 D")),
            bar_size=str(payload.get("bar_size", "5 mins")),
            what_to_show=str(payload.get("what_to_show", "TRADES")),
            use_rth=bool(payload.get("use_rth", False)),
            keep_up_to_date=bool(payload.get("keep_up_to_date", False)),
            end_datetime=str(payload.get("end_datetime", "")),
            format_date=int(payload.get("format_date", 1)),
        )


@dataclass(frozen=True)
class BrokerBar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    average: float | None = None
    bar_count: int | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_timestamp(self.timestamp)
        object.__setattr__(self, "timestamp", normalized or str(self.timestamp))
        object.__setattr__(self, "open", float(self.open))
        object.__setattr__(self, "high", float(self.high))
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "close", float(self.close))
        object.__setattr__(self, "volume", float(self.volume))
        if self.average is not None:
            object.__setattr__(self, "average", float(self.average))
        if self.bar_count is not None:
            object.__setattr__(self, "bar_count", int(self.bar_count))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "average": self.average,
            "bar_count": self.bar_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BrokerBar":
        return cls(
            timestamp=str(payload["timestamp"]),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload.get("volume", 0.0)),
            average=payload.get("average"),
            bar_count=payload.get("bar_count"),
        )

    @classmethod
    def from_ib_bar(cls, bar: Any) -> "BrokerBar":
        return cls(
            timestamp=_normalize_timestamp(getattr(bar, "date", None)) or str(getattr(bar, "date", "")),
            open=float(getattr(bar, "open", 0.0)),
            high=float(getattr(bar, "high", 0.0)),
            low=float(getattr(bar, "low", 0.0)),
            close=float(getattr(bar, "close", 0.0)),
            volume=float(getattr(bar, "volume", 0.0) or 0.0),
            average=getattr(bar, "average", None),
            bar_count=getattr(bar, "barCount", None),
        )


@dataclass(frozen=True)
class BrokerPosition:
    account: str | None
    symbol: str
    exchange: str | None = None
    currency: str | None = None
    local_symbol: str | None = None
    quantity: float = 0.0
    side: PositionSide = "FLAT"
    avg_cost: float | None = None
    contract_id: int | None = None

    def __post_init__(self) -> None:
        quantity = float(self.quantity)
        object.__setattr__(self, "account", _normalize_text(self.account))
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "exchange", _normalize_text(self.exchange))
        object.__setattr__(self, "currency", _normalize_text(self.currency))
        object.__setattr__(self, "local_symbol", _normalize_text(self.local_symbol))
        object.__setattr__(self, "quantity", abs(quantity))
        object.__setattr__(self, "side", _normalize_position_side(self.side, quantity=quantity))
        if self.avg_cost is not None:
            object.__setattr__(self, "avg_cost", float(self.avg_cost))
        if self.contract_id is not None:
            object.__setattr__(self, "contract_id", int(self.contract_id))
        if self.side == "FLAT":
            object.__setattr__(self, "quantity", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "currency": self.currency,
            "local_symbol": self.local_symbol,
            "quantity": self.quantity,
            "side": self.side,
            "avg_cost": self.avg_cost,
            "contract_id": self.contract_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BrokerPosition":
        return cls(
            account=payload.get("account"),
            symbol=str(payload.get("symbol", "")),
            exchange=payload.get("exchange"),
            currency=payload.get("currency"),
            local_symbol=payload.get("local_symbol"),
            quantity=float(payload.get("quantity", 0.0)),
            side=str(payload.get("side", "FLAT")),
            avg_cost=payload.get("avg_cost"),
            contract_id=payload.get("contract_id"),
        )

    @classmethod
    def from_ib_position(
        cls,
        account: Any,
        contract: Any,
        quantity: Any,
        avg_cost: Any,
    ) -> "BrokerPosition":
        quantity_value = float(quantity or 0.0)
        return cls(
            account=account,
            symbol=str(getattr(contract, "symbol", "")),
            exchange=getattr(contract, "exchange", None),
            currency=getattr(contract, "currency", None),
            local_symbol=getattr(contract, "localSymbol", None),
            quantity=abs(quantity_value),
            side=_normalize_position_side(None, quantity=quantity_value),
            avg_cost=avg_cost,
            contract_id=getattr(contract, "conId", None),
        )


@dataclass(frozen=True)
class BrokerOrder:
    order_id: str
    perm_id: int | None = None
    parent_order_id: str | None = None
    account: str | None = None
    symbol: str | None = None
    action: str | None = None
    order_type: str | None = None
    total_quantity: float = 0.0
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    status: str | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str | None = None
    outside_rth: bool = False

    def __post_init__(self) -> None:
        if not str(self.order_id).strip():
            raise ValueError("order_id must not be empty.")
        object.__setattr__(self, "order_id", str(self.order_id).strip())
        object.__setattr__(self, "parent_order_id", _normalize_text(self.parent_order_id))
        object.__setattr__(self, "account", _normalize_text(self.account))
        object.__setattr__(self, "symbol", _normalize_text(self.symbol))
        object.__setattr__(self, "action", _normalize_order_action(self.action))
        object.__setattr__(self, "order_type", _normalize_text(self.order_type))
        object.__setattr__(self, "status", _normalize_order_status(self.status))
        object.__setattr__(self, "total_quantity", float(self.total_quantity))
        object.__setattr__(self, "filled_quantity", float(self.filled_quantity))
        object.__setattr__(self, "remaining_quantity", float(self.remaining_quantity))
        if self.limit_price is not None:
            object.__setattr__(self, "limit_price", float(self.limit_price))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", float(self.stop_price))
        object.__setattr__(self, "tif", _normalize_text(self.tif))
        if self.perm_id is not None:
            object.__setattr__(self, "perm_id", int(self.perm_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "perm_id": self.perm_id,
            "parent_order_id": self.parent_order_id,
            "account": self.account,
            "symbol": self.symbol,
            "action": self.action,
            "order_type": self.order_type,
            "total_quantity": self.total_quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "status": self.status,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "tif": self.tif,
            "outside_rth": self.outside_rth,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BrokerOrder":
        return cls(
            order_id=str(payload["order_id"]),
            perm_id=payload.get("perm_id"),
            parent_order_id=payload.get("parent_order_id"),
            account=payload.get("account"),
            symbol=payload.get("symbol"),
            action=payload.get("action"),
            order_type=payload.get("order_type"),
            total_quantity=float(payload.get("total_quantity", 0.0)),
            filled_quantity=float(payload.get("filled_quantity", 0.0)),
            remaining_quantity=float(payload.get("remaining_quantity", 0.0)),
            status=payload.get("status"),
            limit_price=payload.get("limit_price"),
            stop_price=payload.get("stop_price"),
            tif=payload.get("tif"),
            outside_rth=bool(payload.get("outside_rth", False)),
        )

    @classmethod
    def from_ib_trade(cls, trade: Any) -> "BrokerOrder":
        contract = getattr(trade, "contract", None)
        order = getattr(trade, "order", None)
        order_status = getattr(trade, "orderStatus", None)
        order_id = getattr(order, "orderId", None)
        return cls(
            order_id=str(order_id if order_id is not None else getattr(order_status, "orderId", "")),
            perm_id=getattr(order, "permId", None),
            parent_order_id=_normalize_text(getattr(order, "parentId", None)),
            account=getattr(order, "account", None),
            symbol=getattr(contract, "localSymbol", None) or getattr(contract, "symbol", None),
            action=getattr(order, "action", None),
            order_type=getattr(order, "orderType", None),
            total_quantity=float(getattr(order, "totalQuantity", 0.0) or 0.0),
            filled_quantity=float(getattr(order_status, "filled", 0.0) or 0.0),
            remaining_quantity=float(getattr(order_status, "remaining", 0.0) or 0.0),
            status=getattr(order_status, "status", None),
            limit_price=getattr(order, "lmtPrice", None),
            stop_price=getattr(order, "auxPrice", None),
            tif=getattr(order, "tif", None),
            outside_rth=bool(getattr(order, "outsideRth", False)),
        )


@dataclass(frozen=True)
class BrokerExecution:
    execution_id: str
    order_id: str | None = None
    perm_id: int | None = None
    account: str | None = None
    symbol: str | None = None
    side: str | None = None
    quantity: float = 0.0
    price: float | None = None
    occurred_at: str | None = None
    commission: float | None = None
    realized_pnl: float | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        if not str(self.execution_id).strip():
            raise ValueError("execution_id must not be empty.")
        object.__setattr__(self, "execution_id", str(self.execution_id).strip())
        object.__setattr__(self, "order_id", _normalize_text(self.order_id))
        object.__setattr__(self, "account", _normalize_text(self.account))
        object.__setattr__(self, "symbol", _normalize_text(self.symbol))
        object.__setattr__(self, "side", _normalize_order_action(self.side))
        object.__setattr__(self, "quantity", float(self.quantity))
        object.__setattr__(self, "occurred_at", _normalize_timestamp(self.occurred_at))
        object.__setattr__(self, "currency", _normalize_text(self.currency))
        if self.price is not None:
            object.__setattr__(self, "price", float(self.price))
        if self.commission is not None:
            object.__setattr__(self, "commission", float(self.commission))
        if self.realized_pnl is not None:
            object.__setattr__(self, "realized_pnl", float(self.realized_pnl))
        if self.perm_id is not None:
            object.__setattr__(self, "perm_id", int(self.perm_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "order_id": self.order_id,
            "perm_id": self.perm_id,
            "account": self.account,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "occurred_at": self.occurred_at,
            "commission": self.commission,
            "realized_pnl": self.realized_pnl,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BrokerExecution":
        return cls(
            execution_id=str(payload["execution_id"]),
            order_id=payload.get("order_id"),
            perm_id=payload.get("perm_id"),
            account=payload.get("account"),
            symbol=payload.get("symbol"),
            side=payload.get("side"),
            quantity=float(payload.get("quantity", 0.0)),
            price=payload.get("price"),
            occurred_at=payload.get("occurred_at"),
            commission=payload.get("commission"),
            realized_pnl=payload.get("realized_pnl"),
            currency=payload.get("currency"),
        )

    @classmethod
    def from_ib_fill(cls, fill: Any) -> "BrokerExecution":
        contract = getattr(fill, "contract", None)
        execution = getattr(fill, "execution", None)
        commission_report = getattr(fill, "commissionReport", None)
        return cls(
            execution_id=str(getattr(execution, "execId", "")),
            order_id=_normalize_text(getattr(execution, "orderId", None)),
            perm_id=getattr(execution, "permId", None),
            account=getattr(execution, "acctNumber", None),
            symbol=getattr(contract, "localSymbol", None) or getattr(contract, "symbol", None),
            side=getattr(execution, "side", None),
            quantity=float(getattr(execution, "shares", 0.0) or 0.0),
            price=getattr(execution, "price", None),
            occurred_at=getattr(execution, "time", None),
            commission=getattr(commission_report, "commission", None),
            realized_pnl=getattr(commission_report, "realizedPNL", None),
            currency=getattr(commission_report, "currency", None) or getattr(contract, "currency", None),
        )


@dataclass(frozen=True)
class BrokerAccountValue:
    key: str
    value: str
    currency: str | None = None
    account: str | None = None

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        if not key:
            raise ValueError("key must not be empty.")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "value", str(self.value).strip())
        object.__setattr__(self, "currency", _normalize_text(self.currency))
        object.__setattr__(self, "account", _normalize_text(self.account))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "currency": self.currency,
            "account": self.account,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BrokerAccountValue":
        return cls(
            key=str(payload["key"]),
            value=str(payload.get("value", "")),
            currency=payload.get("currency"),
            account=payload.get("account"),
        )

    @classmethod
    def from_ib_value(cls, value: Any) -> "BrokerAccountValue":
        return cls(
            key=str(getattr(value, "tag", getattr(value, "key", ""))),
            value=str(getattr(value, "value", "")),
            currency=getattr(value, "currency", None),
            account=getattr(value, "account", None),
        )


@dataclass(frozen=True)
class BrokerAccountState:
    account: str | None = None
    values: tuple[BrokerAccountValue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "account", _normalize_text(self.account))
        object.__setattr__(self, "values", tuple(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "values": [item.to_dict() for item in self.values],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BrokerAccountState":
        return cls(
            account=payload.get("account"),
            values=tuple(BrokerAccountValue.from_dict(item) for item in payload.get("values") or ()),
        )

    @classmethod
    def from_ib_values(cls, values: list[Any]) -> "BrokerAccountState":
        restored = tuple(BrokerAccountValue.from_ib_value(item) for item in values or [])
        account = None
        for item in restored:
            if item.account:
                account = item.account
                break
        return cls(account=account, values=restored)