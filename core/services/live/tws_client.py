from __future__ import annotations

from typing import Any, Callable

from .broker_dtos import (
    BrokerAccountState,
    BrokerBar,
    BrokerExecution,
    BrokerOrder,
    BrokerPosition,
    FuturesContractSpec,
    HistoricalBarsRequest,
    TwsConnectionConfig,
)


def _default_ib_factory() -> Any:
    from ib_insync import IB

    return IB()


def _import_contract_types() -> tuple[Any, Any]:
    from ib_insync import ContFuture, Future

    return ContFuture, Future


class PaperTradingGuardError(RuntimeError):
    pass


class PaperSafeTwsClient:
    def __init__(
        self,
        config: TwsConnectionConfig,
        *,
        ib_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self._ib_factory = ib_factory or _default_ib_factory
        self._ib: Any | None = None
        self._connected_client_id: int | None = None

    @property
    def is_connected(self) -> bool:
        checker = getattr(self._ib, "isConnected", None)
        return bool(self._ib is not None and callable(checker) and checker())

    @property
    def ib(self) -> Any:
        if not self.is_connected:
            raise RuntimeError("The TWS client is not connected.")
        return self._ib

    @property
    def connected_client_id(self) -> int | None:
        return self._connected_client_id if self.is_connected else None

    def connect(self) -> None:
        if self.is_connected:
            return
        candidate_ids = self._candidate_client_ids()
        rejected_client_ids: list[int] = []
        last_exc: Exception | None = None

        for candidate_client_id in candidate_ids:
            ib = self._ib_factory()
            error_messages, detach_error_handler = self._attach_error_collector(ib)
            try:
                ib.connect(
                    self.config.host,
                    self.config.port,
                    clientId=candidate_client_id,
                    readonly=self.config.readonly,
                )
                self._assert_paper_session(ib)
            except Exception as exc:
                disconnect = getattr(ib, "disconnect", None)
                if callable(disconnect):
                    disconnect()
                normalized = self._normalize_connect_error(
                    exc,
                    error_messages,
                    attempted_client_id=candidate_client_id,
                )
                last_exc = normalized if isinstance(normalized, Exception) else exc
                if self._is_client_id_in_use_error(normalized):
                    rejected_client_ids.append(candidate_client_id)
                    if candidate_client_id != candidate_ids[-1]:
                        continue
                    if len(rejected_client_ids) > 1:
                        tried = ", ".join(str(item) for item in rejected_client_ids)
                        raise RuntimeError(
                            f"IBKR API rejected client IDs {tried}: already in use. Choose a different base client ID."
                        ) from exc
                if normalized is exc:
                    raise
                raise normalized from exc
            finally:
                detach_error_handler()
            self._ib = ib
            self._connected_client_id = candidate_client_id
            return

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Paper broker connect attempt finished without a result.")

    def disconnect(self) -> None:
        if self._ib is None:
            return
        disconnect = getattr(self._ib, "disconnect", None)
        if callable(disconnect):
            disconnect()
        self._ib = None
        self._connected_client_id = None

    def qualify_contract(self, spec: FuturesContractSpec) -> Any:
        contract = self._build_contract(spec)
        qualified = self.ib.qualifyContracts(contract)
        if qualified:
            return qualified[0]
        return contract

    def request_historical_bars(
        self,
        spec: FuturesContractSpec,
        request: HistoricalBarsRequest,
    ) -> list[BrokerBar]:
        contract = self.qualify_contract(spec)
        bars = self.ib.reqHistoricalData(contract, **request.to_ib_kwargs())
        return [BrokerBar.from_ib_bar(bar) for bar in list(bars or [])]

    def get_positions(self) -> list[BrokerPosition]:
        getter = getattr(self.ib, "positions", None)
        entries = getter() if callable(getter) else []
        return [self._map_position_entry(entry) for entry in list(entries or [])]

    def get_open_orders(self) -> list[BrokerOrder]:
        getter = getattr(self.ib, "openTrades", None)
        trades = getter() if callable(getter) else []
        return [BrokerOrder.from_ib_trade(trade) for trade in list(trades or [])]

    def get_fills(self) -> list[BrokerExecution]:
        getter = getattr(self.ib, "fills", None)
        fills = getter() if callable(getter) else []
        return [BrokerExecution.from_ib_fill(fill) for fill in list(fills or [])]

    def get_account_state(self, account: str | None = None) -> BrokerAccountState:
        summary = self._account_summary(account or self.config.account)
        return BrokerAccountState.from_ib_values(summary)

    def place_order(self, spec: FuturesContractSpec, order: Any) -> BrokerOrder:
        self._assert_submit_enabled()
        contract = self.qualify_contract(spec)
        trade = self.ib.placeOrder(contract, order)
        return BrokerOrder.from_ib_trade(trade)

    def cancel_order(self, order: Any) -> Any:
        self._assert_submit_enabled()
        return self.ib.cancelOrder(order)

    def _assert_paper_session(self, ib: Any) -> None:
        accounts = self._managed_accounts(ib)
        configured = self.config.account.upper() if self.config.account else None
        if configured and not configured.startswith("DU"):
            raise PaperTradingGuardError("Configured account is not a paper DU account.")
        non_paper = [account for account in accounts if not account.upper().startswith("DU")]
        if non_paper:
            joined = ", ".join(sorted(non_paper))
            raise PaperTradingGuardError(f"Paper-only guardrail rejected non-paper account(s): {joined}")
        if configured and accounts and configured not in {account.upper() for account in accounts}:
            raise PaperTradingGuardError(f"Configured paper account {configured} is not present in the TWS session.")

    def _assert_submit_enabled(self) -> None:
        if self.config.readonly:
            raise PaperTradingGuardError("Order submission requires readonly=False on the paper session config.")

    def _account_summary(self, account: str | None) -> list[Any]:
        getter = getattr(self.ib, "accountSummary", None)
        if callable(getter):
            try:
                if account:
                    return list(getter(account=account) or [])
            except TypeError:
                if account:
                    return list(getter(account) or [])
            return list(getter() or [])
        req_getter = getattr(self.ib, "reqAccountSummary", None)
        if callable(req_getter):
            if account:
                return list(req_getter(account) or [])
            return list(req_getter() or [])
        return []

    @staticmethod
    def _managed_accounts(ib: Any) -> list[str]:
        getter = getattr(ib, "managedAccounts", None)
        if getter is None:
            return []
        values = getter() if callable(getter) else getter
        if isinstance(values, str):
            return [part.strip() for part in values.split(",") if part.strip()]
        return [str(part).strip() for part in list(values or []) if str(part).strip()]

    @staticmethod
    def _map_position_entry(entry: Any) -> BrokerPosition:
        if hasattr(entry, "account") and hasattr(entry, "contract"):
            return BrokerPosition.from_ib_position(
                getattr(entry, "account", None),
                getattr(entry, "contract", None),
                getattr(entry, "position", 0.0),
                getattr(entry, "avgCost", None),
            )
        account, contract, quantity, avg_cost = entry
        return BrokerPosition.from_ib_position(account, contract, quantity, avg_cost)

    @staticmethod
    def _build_contract(spec: FuturesContractSpec) -> Any:
        cont_future, future = _import_contract_types()
        if spec.contract_mode == "CONT":
            return cont_future(spec.symbol, spec.exchange)

        contract = future(spec.symbol, spec.expiry, spec.exchange, currency=spec.currency)
        contract.includeExpired = spec.include_expired
        if spec.multiplier:
            contract.multiplier = spec.multiplier
        if spec.local_symbol:
            contract.localSymbol = spec.local_symbol
        return contract

    def _candidate_client_ids(self) -> list[int]:
        base_client_id = int(self.config.client_id)
        retry_span = max(0, int(getattr(self.config, "client_id_retry_span", 0) or 0))
        return [base_client_id + offset for offset in range(retry_span + 1)]

    @staticmethod
    def _attach_error_collector(ib: Any) -> tuple[list[tuple[int | None, str]], Callable[[], None]]:
        error_messages: list[tuple[int | None, str]] = []
        error_event = getattr(ib, "errorEvent", None)
        if error_event is None:
            return error_messages, (lambda: None)

        def _handle_error(req_id: Any, error_code: Any, error_string: Any, contract: Any = None) -> None:
            del req_id, contract
            code: int | None
            try:
                code = int(error_code)
            except (TypeError, ValueError):
                code = None
            error_messages.append((code, str(error_string or "").strip()))

        subscribed = False
        add_handler = getattr(error_event, "__iadd__", None)
        try:
            if callable(add_handler):
                add_handler(_handle_error)
                subscribed = True
            else:
                append = getattr(error_event, "append", None)
                if callable(append):
                    append(_handle_error)
                    subscribed = True
        except Exception:
            append = getattr(error_event, "append", None)
            if callable(append):
                append(_handle_error)
                subscribed = True

        def _detach() -> None:
            if not subscribed:
                return
            try:
                remove_handler = getattr(error_event, "__isub__", None)
                if callable(remove_handler):
                    remove_handler(_handle_error)
                    return
            except Exception:
                pass
            try:
                remove = getattr(error_event, "remove", None)
                if callable(remove):
                    remove(_handle_error)
            except Exception:
                pass

        return error_messages, _detach

    @staticmethod
    def _is_client_id_in_use_message(text: str) -> bool:
        lowered = str(text or "").lower()
        return "already in use" in lowered and "client id" in lowered

    def _is_client_id_in_use_error(self, exc: Exception) -> bool:
        return self._is_client_id_in_use_message(str(exc).strip())

    def _normalize_connect_error(
        self,
        exc: Exception,
        error_messages: list[tuple[int | None, str]],
        *,
        attempted_client_id: int,
    ) -> Exception:
        normalized_messages = [text for _code, text in error_messages if text]
        unique_messages = list(dict.fromkeys(normalized_messages))
        for code, text in error_messages:
            if code == 326 or self._is_client_id_in_use_message(text):
                return RuntimeError(
                    f"IBKR API rejected client ID {attempted_client_id}: already in use. Choose a different client ID."
                )

        if unique_messages and not str(exc).strip():
            return RuntimeError("; ".join(unique_messages))
        return exc