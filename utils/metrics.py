# ibkr_trading_bot/utils/metrics.py
# Kombinované metriky pro binární i 3-třídní (SHORT/HOLD/LONG) vyhodnocení + "trading" pohled.
# - calculate_metrics(y_true, y_pred, df, fee_per_trade, slippage_bps, annualize_sharpe=False, bars_per_year=None, **kwargs)
# - pnl_scorer(estimator, X_val, y_val, df_val, fee, slippage)

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# volitelně sklearn
try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


# ---------------- Pomocné ----------------
_PRICE_CANDS = ("close", "Close", "price", "Price", "average", "Average", "close_price")

def _pick_price_series(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None:
        return None
    for c in _PRICE_CANDS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s.astype(float)
    return None

def _positions_from_predictions(y_pred: np.ndarray) -> np.ndarray:
    """
    Mapování predikcí na pozici:
      >0 => LONG (1),  0 => FLAT (0),  <0 => SHORT (-1)
    Binární 0/1 také fungují (1 => LONG).
    """
    pos = np.zeros_like(y_pred, dtype=int)
    pos[y_pred > 0] = 1
    pos[y_pred < 0] = -1
    return pos

def _equity_from_positions(pos: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, list[float], list[int], list[int]]:
    """
    Hrubá equity křivka při plnění na close.
    Per-bar PnL = pos[t-1] * (px[t] - px[t-1]).
    Vrací: equity_gross, list PnL obchodů, indexy barů uzavření obchodů.
    """
    n = len(px)
    if n == 0:
        return np.zeros(0), [], []
    pos = np.asarray(pos, dtype=int)
    px = np.asarray(px, dtype=float)

    # Align lengths: sometimes predictions (pos) have one less element than
    # price series (px) due to upstream shifting; handle gracefully.
    if pos.size == px.size - 1:
        pos = np.concatenate(([0], pos))
    elif pos.size == px.size + 1:
        pos = pos[:-1]
    elif pos.size != px.size:
        # Truncate to the minimal common length (best-effort fallback).
        m = min(pos.size, px.size)
        pos = pos[:m]
        px = px[:m]

    dpx = np.diff(px, prepend=px[0])
    ret_bar = pos[:-1] * dpx[1:]  # výnos od t-1 do t
    ret = np.zeros(n)
    if ret_bar.size:
        ret[1:] = ret_bar

    # extrahuj obchody podle změny pozice
    trades: list[float] = []
    trade_idx: list[int] = []
    trade_sides: list[int] = []
    cur = 0
    entry = None
    for t in range(n):
        p = pos[t]
        if cur == 0 and p != 0:
            cur = p
            entry = px[t]
        elif cur != 0 and p != cur:
            exit_px = px[t]
            pnl = (exit_px - entry) if cur == 1 else (entry - exit_px)
            trades.append(float(pnl))
            trade_idx.append(t)
            trade_sides.append(int(cur))
            cur = p
            entry = px[t] if p != 0 else None
    if cur != 0 and entry is not None:
        exit_px = px[-1]
        pnl = (exit_px - entry) if cur == 1 else (entry - exit_px)
        trades.append(float(pnl))
        trade_idx.append(n - 1)
        trade_sides.append(int(cur))

    equity = np.cumsum(ret)
    return equity, trades, trade_idx, trade_sides

def _max_drawdown(equity: np.ndarray) -> float:
    peak = -1e18
    mdd = 0.0
    for x in equity:
        peak = max(peak, x)
        mdd = min(mdd, x - peak)
    return float(mdd)

def _sharpe(ret: np.ndarray) -> float:
    if len(ret) == 0:
        return float("nan")
    mu = float(np.nanmean(ret))
    sd = float(np.nanstd(ret))
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    return mu / sd


def _signal_stability_from_signals(signals: np.ndarray) -> float:
    signals = np.asarray(signals)
    if signals.size == 0:
        return 1.0
    changes = np.diff(signals)
    reversals = int(np.sum(np.abs(changes) == 2))
    total_changes = int(np.sum(changes != 0))
    return 1.0 - (reversals / total_changes) if total_changes > 0 else 1.0

def _profit_factor(trades: list[float]) -> float:
    if not trades:
        return float("nan")
    g = sum(x for x in trades if x > 0)
    loss = -sum(x for x in trades if x < 0)
    if loss == 0:
        return float("inf") if g > 0 else float("nan")
    return float(g / loss)

def _var(arr: list[float], p: float) -> float:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.nanpercentile(a, 100 * (1 - p)))

def _cvar(arr: list[float], p: float) -> float:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return float("nan")
    thr = np.nanpercentile(a, 100 * (1 - p))
    tail = a[a <= thr]
    if tail.size == 0:
        return float("nan")
    return float(np.nanmean(tail))

def _infer_bars_per_year(df: pd.DataFrame | None) -> int | None:
    """Odhad počtu barů za rok z median delta(timestamp)."""
    try:
        if df is None or "timestamp" not in df.columns:
            return None
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        ts = ts[ts.notna()]
        if len(ts) < 3:
            return None
        dt = ts.diff().dt.total_seconds().median()
        if not np.isfinite(dt) or dt <= 0:
            return None
        seconds_per_year = 365.25 * 24 * 3600
        bpy = round(seconds_per_year / dt)
        return int(bpy) if bpy > 0 else None
    except Exception:
        return None


# ---------------- Hlavní API ----------------
def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    df: pd.DataFrame | None = None,
    fee_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    rolling_window: int = 200,
    *,
    annualize_sharpe: bool = False,
    bars_per_year: int | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Vrací dict s klasifikačními metrikami (binární, ternární a 3-třídní) + (pokud jsou ceny) obchodní metriky.
    Konvence: 
      - Binární: y_pred > 0 = LONG, < 0 = SHORT, 0 = FLAT. Pro 0/1: 1 = LONG.
      - Ternární (mapovaná): 0=-1 (SHORT), 1=0 (FLAT), 2=1 (LONG); internal conversion to -1/0/1
      - 3-class: labels -1, 0, 1 (SHORT, FLAT, LONG)

    Parametry navíc:
      - annualize_sharpe: pokud True, dopočítá 'sharpe_ann' a 'sharpe_net_ann'
      - bars_per_year: volitelně explicitní počet barů za rok; pokud není, pokusíme se odhadnout z df['timestamp']
      - **kwargs: ignorováno (kvůli zpětné kompatibilitě)
    """
    yt_raw = pd.Series(y_true).astype(int)
    yp_raw = pd.Series(y_pred).astype(int)
    yt = np.asarray(yt_raw.to_numpy())
    yp = np.asarray(yp_raw.to_numpy())

    # Remap ternary (0,1,2) to (-1,0,1) only when class 2 is present.
    # NOTE:
    # - Binary 0/1 labels in this project represent SHORT/LONG and must NOT be remapped,
    #   otherwise class 1 would become FLAT (0) and trading metrics collapse.
    uniq_yt = set(np.unique(yt).tolist())
    uniq_yp = set(np.unique(yp).tolist())
    if (2 in uniq_yt) or (2 in uniq_yp):
        # Likely mapped ternary: 0->-1, 1->0, 2->1
        yt_remap = np.array([-1 if c == 0 else (0 if c == 1 else 1) for c in yt])
        yp_remap = np.array([-1 if c == 0 else (0 if c == 1 else 1) for c in yp])
        yt = yt_remap
        yp = yp_remap

    # ---------- BINÁRNÍ pohled (mapujeme >0 -> 1, FLAT/SHORT -> 0)
    yt_bin = (yt > 0).astype(int)
    yp_bin = (yp > 0).astype(int)
    if _HAVE_SK:
        acc = float(accuracy_score(yt_bin, yp_bin))
        f1 = float(f1_score(yt_bin, yp_bin, zero_division=0))
        prec = float(precision_score(yt_bin, yp_bin, zero_division=0))
        rec = float(recall_score(yt_bin, yp_bin, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(yt_bin, yp_bin, labels=[0, 1]).ravel()
    else:
        tp = int(((yt_bin == 1) & (yp_bin == 1)).sum())
        tn = int(((yt_bin == 0) & (yp_bin == 0)).sum())
        fp = int(((yt_bin == 0) & (yp_bin == 1)).sum())
        fn = int(((yt_bin == 1) & (yp_bin == 0)).sum())
        acc = float((tp + tn) / max(1, tp + tn + fp + fn))
        prec = float(tp / max(1, tp + fp))
        rec = float(tp / max(1, tp + fn))
        f1 = float(2 * prec * rec / max(1e-12, (prec + rec)))

    out: dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "signals": int((yp_bin == 1).sum()),
    }

    # ---------- 3-TŘÍDNÍ pohled (SHORT=-1, HOLD=0, LONG=1)
    labels3 = [-1, 0, 1]
    if _HAVE_SK:
        cm3 = confusion_matrix(yt, yp, labels=labels3)
        prec3, rec3, f13, supp3 = precision_recall_fscore_support(
            yt, yp, labels=labels3, zero_division=0
        )
        f1_micro_3 = float(f1_score(yt, yp, labels=labels3, average="micro", zero_division=0))
        f1_macro_3 = float(f1_score(yt, yp, labels=labels3, average="macro", zero_division=0))
        f1_weighted_3 = float(f1_score(yt, yp, labels=labels3, average="weighted", zero_division=0))
    else:
        idx_map = {v: i for i, v in enumerate(labels3)}
        cm3 = np.zeros((3, 3), dtype=int)
        for a, b in zip(yt, yp):
            ia = idx_map.get(int(a), None)
            ib = idx_map.get(int(b), None)
            if ia is not None and ib is not None:
                cm3[ia, ib] += 1
        prec3, rec3, f13, supp3 = [], [], [], []
        for i in range(3):
            tp_i = cm3[i, i]
            fp_i = cm3[:, i].sum() - tp_i
            fn_i = cm3[i, :].sum() - tp_i
            supp_i = cm3[i, :].sum()
            p_i = float(tp_i / max(1, tp_i + fp_i))
            r_i = float(tp_i / max(1, tp_i + fn_i))
            f_i = float(2 * p_i * r_i / max(1e-12, (p_i + r_i)))
            prec3.append(p_i)
            rec3.append(r_i)
            f13.append(f_i)
            supp3.append(int(supp_i))
        f1_macro_3 = float(np.nanmean(f13)) if len(f13) else 0.0
        f1_micro_3 = float(np.trace(cm3) / max(1, cm3.sum()))
        weights = np.asarray(supp3, dtype=float)
        w = weights / max(1, weights.sum())
        f1_weighted_3 = float(np.nansum(np.asarray(f13) * w)) if w.size else f1_macro_3

    out["confusion3"] = {
        "labels": labels3,
        "matrix": [[int(x) for x in row] for row in cm3.tolist()],
    }
    out["per_class_3"] = {
        str(lbl): {
            "precision": float(prec3[i]),
            "recall": float(rec3[i]),
            "f1": float(f13[i]),
            "support": int(supp3[i]),
        }
        for i, lbl in enumerate(labels3)
    }
    out["f1_micro_3"] = f1_micro_3
    out["f1_macro_3"]  = f1_macro_3
    out["f1_weighted_3"] = f1_weighted_3

    # ---------- trading metriky (pokud máme ceny)
    px = _pick_price_series(df)
    sharpe_ann = None
    sharpe_net_ann = None
    if px is not None:
        # Use full 3-state predictions for positions (SHORT/HOLD/LONG)
        pos = _positions_from_predictions(yp)
        equity_gross, trades_gross, trade_idx, trade_sides = _equity_from_positions(pos, px.values)

        n_trades = len(trades_gross)
        # approximate per-trade slippage as proportional to mean price
        per_trade_slip = float(slippage_bps) / 10_000.0 * float(np.nanmean(px.values))
        per_trade_fee = float(fee_per_trade)
        trade_pnls_net = [float(p - per_trade_fee - per_trade_slip) for p in trades_gross]

        slip_cost = per_trade_slip * n_trades
        fee_cost = per_trade_fee * n_trades
        equity_net = equity_gross - (slip_cost + fee_cost)

        ret_gross = np.diff(equity_gross, prepend=equity_gross[0])
        ret_net = np.diff(equity_net, prepend=equity_net[0])

        sharpe_val = _sharpe(ret_gross)
        sharpe_net_val = _sharpe(ret_net)

        if annualize_sharpe:
            bpy = bars_per_year or _infer_bars_per_year(df) or None
            if bpy:
                root = float(np.sqrt(bpy))
                sharpe_ann = sharpe_val * root if np.isfinite(sharpe_val) else None
                sharpe_net_ann = sharpe_net_val * root if np.isfinite(sharpe_net_val) else None

        num_trades_long = int(sum(1 for s in trade_sides if s == 1))
        num_trades_short = int(sum(1 for s in trade_sides if s == -1))

        out.update({
            "profit_gross": float(equity_gross[-1]) if equity_gross.size else 0.0,
            "profit_net": float(equity_net[-1]) if equity_net.size else 0.0,
            "trade_pnls_gross": list(trades_gross),
            "trade_pnls_net": list(trade_pnls_net),
            "sharpe": sharpe_val,
            "sharpe_net": sharpe_net_val,
            "sharpe_ann": sharpe_ann,
            "sharpe_net_ann": sharpe_net_ann,
            "max_drawdown": _max_drawdown(equity_gross),
            "max_drawdown_net": _max_drawdown(equity_net),
            "max_drawdown_trade_gross": abs(_max_drawdown(equity_gross)),
            "max_drawdown_trade_net": abs(_max_drawdown(equity_net)),
            "num_trades": int(n_trades),
            "num_trades_long": num_trades_long,
            "num_trades_short": num_trades_short,
            "winrate": float(np.mean([t > 0 for t in trades_gross])) if n_trades else float("nan"),
            "profit_factor": _profit_factor(trades_gross),
            "avg_pnl_trade": float(np.nanmean(trades_gross)) if n_trades else float("nan"),
            "median_pnl_trade": float(np.nanmedian(trades_gross)) if n_trades else float("nan"),
            "var_95": _var(trades_gross, 0.95),
            "cvar_95": _cvar(trades_gross, 0.95),
        })

        if n_trades and rolling_window > 1:
            ser = pd.Series(trades_gross, dtype=float)
            out["rolling_pnl_mean"] = float(ser.rolling(min_periods=1, window=rolling_window).mean().iloc[-1])

        # signal stability (1.0 = very stable, 0.0 = highly unstable)
        try:
            out["signal_stability"] = float(_signal_stability_from_signals(yp))
        except Exception:
            out["signal_stability"] = 1.0

    # ---------- aliasy pro kompatibilitu (pokud někde UI očekává starší názvy)
    out.setdefault("sharpe_ratio", out.get("sharpe"))
    out.setdefault("max_dd", out.get("max_drawdown"))
    out.setdefault("trades", out.get("num_trades"))
    out.setdefault("pf", out.get("profit_factor"))

    return out


def pnl_scorer(
    estimator,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    df_val: pd.DataFrame | None = None,
    fee: float = 0.0,
    slippage: float = 0.0,
) -> float:
    """
    Skóre pro Grid/CV: preferujeme profit_net, pokud máme ceny; jinak F1/accuracy.
    """
    try:
        if hasattr(estimator, "predict_proba"):
            pr = estimator.predict_proba(X_val)
            if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 3:
                prob_short = pr[:, 0]
                prob_long = pr[:, 2]
                # mapped ternary labels: 0=SHORT, 1=HOLD, 2=LONG
                y_pred = np.where(prob_long >= 0.5, 2, np.where(prob_short >= 0.5, 0, 1)).astype(int)
            else:
                p1 = pr[:, 1] if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 2 else np.asarray(pr).ravel()
                y_pred = (p1 >= 0.5).astype(int)
        elif hasattr(estimator, "decision_function"):
            z = np.asarray(estimator.decision_function(X_val)).ravel()
            y_pred = (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int)
        else:
            y_pred = estimator.predict(X_val)

        m = calculate_metrics(
            y_true=np.asarray(y_val).astype(int),
            y_pred=y_pred,
            df=df_val,
            fee_per_trade=fee,
            slippage_bps=slippage
        )
        if "profit_net" in m and np.isfinite(m["profit_net"]):
            return float(m["profit_net"])
        if "f1" in m and np.isfinite(m["f1"]):
            return float(m["f1"])
        return float(m.get("accuracy", 0.0))
    except Exception:
        try:
            pred = estimator.predict(X_val)
            return float((pred == np.asarray(y_val).astype(int)).mean())
        except Exception:
            return 0.0
