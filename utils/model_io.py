# utils/metrics.py
# ---------------------------------------------------
# Popis:
# - calculate_metrics(y_true, y_pred, df) vrací kombinaci klasifikačních i trading metrik.
# - Pokud jsou dostupná cenová data (close/Close), dopočítá trade-level PnL, equity curve
#   a rizikové metriky (VaR, CVaR, drawdown). Jinak vrátí jen klasifikační metriky.
# - y_pred je interpretováno jako signál: >0 = long, <0 = short, 0 = flat.
# - Vše v UTF-8, bez povinné závislosti na sklearn (fallback výpočty).

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Volitelně použijeme sklearn pro přesné metriky, ale není povinné.
try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# ------------------------- Pomocné funkce -------------------------

def _to_numpy_binary(a) -> np.ndarray:
    """Převede y v {0,1} nebo {-1,1,0} na binární {0,1} pro klasifikaci LONG vs. ostatní.
    Pokud není jasné, použije prahování >0 jako 1, jinak 0.
    """
    arr = np.asarray(a)
    # Pokud už je 0/1, ponecháme
    uniq = np.unique(arr[~pd.isna(arr)])
    if set(uniq.tolist()).issubset({0, 1}):
        return arr.astype(int)
    # Pokud obsahuje -1/0/1, tak 1 je >0
    return (arr > 0).astype(int)


def _safe_close_series(df: pd.DataFrame) -> pd.Series | None:
    """Vrátí sloupec s close cenou (pokusí se najít běžné názvy)."""
    candidates = ["close", "Close", "CLOSE", "adj_close", "Adj Close"]
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return None


def _build_positions_from_signals(y_pred: np.ndarray) -> np.ndarray:
    """Z binárních/tri-state signálů vytvoří pozice: 1 = long, -1 = short, 0 = flat.
    Používá jednoduché pravidlo: val > 0 → 1, val < 0 → -1, jinak 0.
    """
    pos = np.zeros_like(y_pred, dtype=int)
    pos[y_pred > 0] = 1
    pos[y_pred < 0] = -1
    return pos


def _trades_from_positions(positions: np.ndarray, prices: np.ndarray) -> list[float]:
    """Sestaví obchody při změně pozice. Vstup/exit na 'close'.
    Příklad: 0,0,1,1,1,-1,-1,0 → vzniknou 2 obchody (long a short).
    PnL pro long: exit_price - entry_price; pro short: entry_price - exit_price.
    """
    trades: list[float] = []
    if len(positions) == 0 or len(positions) != len(prices):
        return trades

    prev_pos = 0
    entry_price = None

    for i in range(len(positions)):
        p = int(positions[i])

        # vstup
        if prev_pos == 0 and p != 0:
            entry_price = float(prices[i])

        # změna pozice nebo výstup na 0 → uzavři obchod
        if prev_pos != 0 and (p != prev_pos):
            exit_price = float(prices[i])
            if prev_pos == 1:  # long
                trades.append(exit_price - float(entry_price))
            elif prev_pos == -1:  # short
                trades.append(float(entry_price) - exit_price)
            entry_price = float(prices[i]) if p != 0 else None

        prev_pos = p

    # pokud držíme na konci, ukončíme na poslední ceně
    if prev_pos != 0 and entry_price is not None:
        exit_price = float(prices[-1])
        if prev_pos == 1:
            trades.append(exit_price - float(entry_price))
        else:
            trades.append(float(entry_price) - exit_price)

    return trades


def _equity_curve_from_trades(trade_pnls: list[float]) -> np.ndarray:
    """Kumulativní PnL po obchodech."""
    if not trade_pnls:
        return np.array([], dtype=float)
    return np.cumsum(np.asarray(trade_pnls, dtype=float))


def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    """Max drawdown z kumulativní PnL křivky."""
    if equity.size == 0:
        return 0.0
    peak = -np.inf
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _sharpe_from_trades(trade_pnls: list[float]) -> float:
    """Jednoduchý Sharpe z trade-level PnL (bez annualizace)."""
    if not trade_pnls:
        return 0.0
    arr = np.asarray(trade_pnls, dtype=float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(mu / sd)


def _profit_factor(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    gains = np.sum([x for x in trade_pnls if x > 0])
    losses = -np.sum([x for x in trade_pnls if x < 0])
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _winrate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    arr = np.asarray(trade_pnls, dtype=float)
    return float(np.mean(arr > 0))


def _signal_stability(y_pred: np.ndarray) -> float:
    """Míra nepřepínání: 1 - (počet změn signálu / max(1, N-1))."""
    if y_pred is None:
        return 0.0
    arr = np.asarray(y_pred).astype(float)
    if arr.size <= 1:
        return 1.0
    switches = np.sum(np.sign(arr[1:]) != np.sign(arr[:-1]))
    return float(1.0 - switches / max(1, arr.size - 1))


def _var_cvar(arr: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """VaR a CVaR na hladině (např. 0.95). Vrací (VaR, CVaR) – v našem značení 5% kvantil = riziko ztráty."""
    if arr.size == 0:
        return 0.0, 0.0
    q = np.nanpercentile(arr, 100 * (1 - alpha))  # 5. percentil pro alpha=0.95
    tail = arr[arr <= q]
    cvar = np.nanmean(tail) if tail.size > 0 else q
    return float(q), float(cvar)


# ------------------------- Veřejná funkce -------------------------

def calculate_metrics(y_true, y_pred, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    Spočítá klasifikační a trading metriky.
    Parametry:
        y_true: Array-like, skutečné hodnoty (binární 0/1 nebo -1/0/1).
        y_pred: Array-like, predikované signály (interpretace >0 long, <0 short, 0 flat).
        df:     DataFrame s historickými daty; pro trading metriky se očekává sloupec 'close'/'Close'.

    Vrací:
        dict s klíči:
            - f1, precision, recall
            - profit, sharpe_ratio, max_drawdown, winrate, profit_factor, signal_stability
            - trade_pnls (List[float]), equity_curve (List[float])
            - num_trades, avg_profit_per_trade, median_profit_per_trade, var_95, cvar_95
        Některé klíče mohou chybět, pokud nejsou dostupná data.
    """
    out: dict[str, Any] = {}

    # --- Klasifikační metriky (LONG vs. ostatní) ---
    try:
        y_true_bin = _to_numpy_binary(y_true)
        y_pred_bin = _to_numpy_binary(y_pred)

        if _HAVE_SK:
            out["f1"] = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
            out["precision"] = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
            out["recall"] = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        else:
            # Fallback bez sklearn
            tp = float(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
            fp = float(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
            fn = float(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            out["f1"] = float(f1)
            out["precision"] = float(precision)
            out["recall"] = float(recall)
    except Exception:
        # Pokud cokoliv selže, necháme klasifikační metriky prázdné
        pass

    # --- Trading metriky (vyžadují ceny) ---
    trade_pnls: list[float] = []
    equity_curve: np.ndarray = np.array([], dtype=float)

    close = None
    if df is not None and isinstance(df, pd.DataFrame):
        close = _safe_close_series(df)

    try:
        if close is not None:
            positions = _build_positions_from_signals(np.asarray(y_pred))
            prices = close.to_numpy(dtype=float)

            # Seznam PnL na obchod
            trade_pnls = _trades_from_positions(positions, prices)
            equity_curve = _equity_curve_from_trades(trade_pnls)

            # Trading metriky
            total_profit = float(np.sum(trade_pnls)) if trade_pnls else 0.0
            sharpe = _sharpe_from_trades(trade_pnls)
            max_dd = _max_drawdown_from_equity(equity_curve)
            pf = _profit_factor(trade_pnls)
            wr = _winrate(trade_pnls)
            stability = _signal_stability(np.asarray(y_pred))

            out.update({
                "profit": total_profit,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "profit_factor": pf,
                "winrate": wr,
                "signal_stability": stability,
                "trade_pnls": trade_pnls,
                "equity_curve": equity_curve.tolist(),
            })

            # Rozšířené metriky z distribuce obchodů
            if trade_pnls:
                arr = np.asarray(trade_pnls, dtype=float)
                out["num_trades"] = int(arr.size)
                out["avg_profit_per_trade"] = float(np.nanmean(arr))
                out["median_profit_per_trade"] = float(np.nanmedian(arr))
                var95, cvar95 = _var_cvar(arr, alpha=0.95)
                out["var_95"] = var95
                out["cvar_95"] = cvar95
        else:
            # Bez cen nedává smysl PnL – ponecháme jen klasifikační metriky
            out["signal_stability"] = _signal_stability(np.asarray(y_pred))
    except Exception:
        # Tichý fallback – aspoň klasifikační metriky a stabilita signálu
        out.setdefault("signal_stability", _signal_stability(np.asarray(y_pred)))

    return out

# --- META sidecar pro model (vedle .pkl) ---
import json
from pathlib import Path


def _meta_path(model_path: str | Path) -> Path:
    p = Path(model_path)
    return p.with_suffix(p.suffix + ".meta.json")

def save_model_meta(model_path: str | Path, meta: dict) -> None:
    mp = _meta_path(model_path)
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def load_model_meta(model_path: str | Path) -> dict:
    mp = _meta_path(model_path)
    if mp.exists():
        return json.loads(mp.read_text(encoding="utf-8"))
    return {}
