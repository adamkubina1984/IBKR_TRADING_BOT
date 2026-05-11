# tests/test_metrics.py
# ---------------------------------------------------
# Popis:
# - Ověření výpočtů v utils.metrics.calculate_metrics na syntetických datech.
# - Testujeme: základní PnL, max drawdown (trade-level), LONG/SHORT breakdown,
#   stabilitu signálu a VaR/CVaR (NET).
#
# Spuštění: pytest -q

import numpy as np
import pandas as pd
import importlib

from ibkr_trading_bot.utils.model_io import calculate_metrics as legacy_calculate_metrics
from ibkr_trading_bot.utils.metrics import calculate_metrics


def _df_from_prices(prices):
    ts = pd.date_range("2024-01-01", periods=len(prices), freq="5min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": prices,
        "high": np.maximum(prices, np.array(prices) + 0.2),
        "low":  np.minimum(prices, np.array(prices) - 0.2),
        "close": prices,
        "volume": 1000,
    })
    return df

def test_basic_long_pnl_one_trade():
    # 1 long obchod: vstup na 2. baru, výstup na 4. baru
    prices = [100, 101, 102, 101, 103]
    df = _df_from_prices(prices)

    # signály: 0,1,1,0,0  -> jeden long trade s PnL = exit(101) - entry(101) = 0
    # Pozor: poslední bar uzavírá nedokončené pozice; tady se zavře na 4. baru.
    y_true = [0, 1, 1, 0, 0]  # nepotřebné, ale předáme
    y_pred = [0, 1, 1, 0, 0]

    out = calculate_metrics(y_true, y_pred, df=df, fee_per_trade=0.0, slippage_bps=0.0)
    assert "trade_pnls_gross" in out
    assert out["num_trades"] == 1
    assert abs(out["profit_gross"] - 0.0) < 1e-9

def test_long_short_breakdown_and_net_costs():
    # Dva obchody: long (zisk +2), short (ztráta -1)
    prices = [100, 101, 102, 101, 100, 99]
    df = _df_from_prices(prices)

    # signály: 0,1,1,0,-1,-1
    # Long: entry 101 -> exit 101 => 0 (uzavření při přechodu na 0)
    # Ale díky poslednímu baru dojde k uzavření shortu: entry 100 -> exit 99 => +1 (pro short)
    # Abychom měli jistotu dvou obchodů s jasnými hodnotami, přepneme sekvenci:
    y_pred = [0, 1, 1, 0, -1, 0]  # Long: 101->101 = 0; Short: 100->100 = 0
    # Uděláme to ziskové/neztrátové přes fee: nastavíme fee_per_trade, ať je NET < GROSS
    out = calculate_metrics([0]*len(y_pred), y_pred, df=df, fee_per_trade=0.5, slippage_bps=0.0)

    assert out["num_trades"] == 2
    assert "profit_gross" in out and "profit_net" in out
    assert out["profit_net"] <= out["profit_gross"]
    # Breakdown počty
    assert out["num_trades_long"] + out["num_trades_short"] == out["num_trades"]

def test_trade_level_max_drawdown_present():
    prices = [100, 102, 101, 105, 103]
    df = _df_from_prices(prices)
    # jeden long trade přes celé období -> gross PnL = 103 - 102 = +1 (záleží na přesném vstupu/exit)
    y_pred = [0, 1, 1, 1, 0]
    out = calculate_metrics([0]*len(y_pred), y_pred, df=df, fee_per_trade=0.0, slippage_bps=0.0)
    assert "max_drawdown_trade_gross" in out
    # DD by neměl být záporný (reportujeme kladné číslo velikosti propadu)
    assert out["max_drawdown_trade_gross"] >= 0.0

def test_signal_stability():
    # střídání signálu → nízká stabilita
    prices = [100, 101, 100, 101, 100, 101]
    df = _df_from_prices(prices)
    y_pred = [1, -1, 1, -1, 1, -1]
    out = calculate_metrics([0]*len(y_pred), y_pred, df=df)
    assert "signal_stability" in out
    assert out["signal_stability"] < 0.5  # hodně přepíná

def test_var_cvar_available_on_net_trades():
    prices = [100, 99, 101, 98, 102, 97, 103]
    df = _df_from_prices(prices)
    # střídání long/flat → pár obchodů
    y_pred = [0, 1, 1, 0, 1, 0, 0]
    out = calculate_metrics([0]*len(y_pred), y_pred, df=df, fee_per_trade=0.1, slippage_bps=0.0)
    # var_95 a cvar_95 se plní, pokud vznikly net obchody
    assert "var_95" in out and "cvar_95" in out


def test_binary_labels_are_not_ternary_remapped():
    prices = [100, 101, 102, 103]
    df = _df_from_prices(prices)

    y_true = [0, 1, 1, 1]
    y_pred = [1, 1, 1, 1]

    out = calculate_metrics(y_true, y_pred, df=df, fee_per_trade=0.0, slippage_bps=0.0)

    assert out["signals"] == len(y_pred)
    assert out["num_trades"] >= 1
    assert out["num_trades_long"] >= 1


def test_metrics_expose_equity_curves_from_shared_executor():
    prices = [100, 101, 102, 101]
    df = _df_from_prices(prices)
    y_pred = [0, 1, 1, 0]

    out = calculate_metrics([0] * len(y_pred), y_pred, df=df, fee_per_trade=0.1, slippage_bps=0.0)

    assert "equity_curve" in out
    assert "equity_curve_net" in out
    assert len(out["equity_curve"]) == len(prices)
    assert len(out["equity_curve_net"]) == len(prices)
    assert out["equity_curve_net"][-1] == out["profit_net"]


def test_metrics_expose_detailed_trade_records():
    prices = [100, 101, 103, 102]
    df = _df_from_prices(prices)
    y_pred = [0, 1, 1, 0]

    out = calculate_metrics([0] * len(y_pred), y_pred, df=df, fee_per_trade=0.25, slippage_bps=0.0)

    assert "trades" in out
    assert len(out["trades"]) == 1
    trade = out["trades"][0]
    assert trade["direction"] == "LONG"
    assert trade["entry_price"] == 101.0
    assert trade["exit_price"] == 102.0
    assert trade["pnl"] == 1.0
    assert trade["pnl_net"] == 0.75


def test_metrics_support_trading_only_mode_without_ground_truth():
    prices = [100, 101, 103, 102]
    df = _df_from_prices(prices)
    y_pred = [0, 1, 1, 0]

    out = calculate_metrics(None, y_pred, df=df, fee_per_trade=0.25, slippage_bps=0.0)

    assert out["classification_available"] is False
    assert "accuracy" not in out
    assert out["profit_net"] == 0.75
    assert "winrate_net" in out
    assert "profit_factor_net" in out


def test_metrics_explicit_ternary_mapped_mode_handles_missing_class_two():
    prices = [100, 101, 100, 99]
    df = _df_from_prices(prices)

    out = calculate_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 1, 0],
        df=df,
        label_mode="ternary_mapped",
    )

    assert out["classification_mode"] == "ternary"
    assert out["label_mode"] == "ternary_mapped"
    assert "f1_macro_3" in out


def test_legacy_model_io_metrics_delegates_to_canonical_metrics():
    prices = [100, 101, 100, 99]
    df = _df_from_prices(prices)

    out = legacy_calculate_metrics(
        y_true=[-1, 0, 1, 0],
        y_pred=[-1, 0, 1, 0],
        df=df,
        label_mode="ternary_signed",
    )

    assert out["classification_mode"] == "ternary"
    assert out["label_mode"] == "ternary_signed"


def test_legacy_model_io_import_smoke_and_sidecar_roundtrip(tmp_path):
    module = importlib.import_module("ibkr_trading_bot.utils.model_io")
    model_path = tmp_path / "model.pkl"

    module.save_model_meta(model_path, {"label_mode": "ternary_signed", "profit_net": 1.25})
    loaded = module.load_model_meta(model_path)

    assert loaded["label_mode"] == "ternary_signed"
    assert loaded["profit_net"] == 1.25
