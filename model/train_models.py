# ibkr_trading_bot/model/train_models.py
from __future__ import annotations

import json as jsonlib
import pathlib
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

# --- volitelné knihovny
try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# --- purged walk-forward split
try:
    from ibkr_trading_bot.model.tscv import PurgedWalkForwardSplit
except Exception:
    class PurgedWalkForwardSplit:  # type: ignore
        def __init__(self, n_splits=5, embargo=0):
            self.n_splits = n_splits
            self.embargo = embargo
        def split(self, X):
            n = len(X)
            fold = n // (self.n_splits + 1)
            for k in range(self.n_splits):
                train_end = fold * (k + 1)
                test_start = min(train_end + self.embargo, n - 1)
                test_end = min(test_start + fold, n)
                tr = np.arange(0, train_end)
                te = np.arange(test_start, test_end)
                yield tr, te

# --- metriky / scorer
try:
    from ibkr_trading_bot.utils.metrics import calculate_metrics, pnl_scorer  # type: ignore
    HAS_CALC_METRICS = True
except Exception:
    try:
        from ibkr_trading_bot.utils.metrics import pnl_scorer  # type: ignore
        HAS_CALC_METRICS = False
    except Exception:
        def pnl_scorer(estimator, X_val, y_val, df_val=None, fee=0.0, slippage=0.0):
            pred = estimator.predict(X_val)
            return float((pred == y_val).mean())
        HAS_CALC_METRICS = False

# ------------------- Pomocné -------------------
def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _project_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "ibkr_trading_bot").is_dir():
            return p / "ibkr_trading_bot"
    return here.parent

def _model_dir() -> pathlib.Path:
    root = _project_root()
    out = root / "model_outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out

# --- bezpečnější výběr featur (bez leaků + odfiltrování kvazi-konstant)
SAFE_EXCLUDE_PATTERNS = (
    "target", "label", "class", "y_",
    "future", "fwd", "lead", "leak",
    "signal", "proba", "pred", "score",
    "pnl", "ret_", "return_fwd", "trade_",
    "event_", "barrier", "tb_", "tripbar", "horizon"
)

def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Preferujeme engineered featury; zároveň vylučujeme potenciální leaky a
    featury s velmi nízkou variabilitou. Když nic nezůstane, spadneme na
    všechny numerické mimo timestamp/target.
    """
    ignore_core = {"timestamp", "target"}
    numeric_cols = [
        c for c in df.columns
        if c not in ignore_core and pd.api.types.is_numeric_dtype(df[c])
    ]
    filtered = []
    for c in numeric_cols:
        cname = c.lower()
        if any(tok in cname for tok in SAFE_EXCLUDE_PATTERNS):
            continue
        filtered.append(c)
    try:
        if filtered:
            nunique = df[filtered].nunique(dropna=True)
            filtered = [c for c in filtered if int(nunique.get(c, 0)) > 5]
    except Exception:
        pass
    if not filtered:
        filtered = numeric_cols
    return filtered

def _build_estimator(name: str) -> tuple[object, dict[str, list]]:
    name = (name or "hgbt").lower()
    if name in ("hgbt","histgb","histgradientboosting"):
        est = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06, l2_regularization=0.0, max_iter=300, random_state=42)
        grid = {"max_depth":[4,6,8],"learning_rate":[0.03,0.06,0.1],"max_iter":[200,300,500],"l2_regularization":[0.0,0.1]}
        return est, grid
    if name in ("rf","random_forest","randomforest"):
        est = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
        grid = {"n_estimators":[300,500,800],"max_depth":[None,8,16],"min_samples_leaf":[1,2,4]}
        return est, grid
    if name in ("et","extratrees","extra_trees"):
        est = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
        grid = {"n_estimators":[400,600,800],"max_depth":[None,8,16],"min_samples_leaf":[1,2,4]}
        return est, grid
    if name in ("svm","svc"):
        est = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=42))])
        grid = {"clf__C":[0.5,1.0,2.0], "clf__gamma":["scale",0.1,0.01]}
        return est, grid
    if name in ("xgb","xgboost") and HAS_XGB:
        est = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=42, n_jobs=-1)
        grid = {"n_estimators":[400,700], "max_depth":[4,6,8], "learning_rate":[0.03,0.06,0.1], "subsample":[0.8,1.0], "colsample_bytree":[0.8,1.0]}
        return est, grid
    return _build_estimator("hgbt")

def _ensure_pipeline(estimator) -> object:
    if isinstance(estimator, Pipeline):
        return estimator
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("clf", estimator)])

def _namespaced_param_grid(estimator, grid: dict | None) -> dict | None:
    if grid is None:
        return None
    if not isinstance(estimator, Pipeline):
        return grid
    step = estimator.steps[-1][0]
    return {(k if "__" in k else f"{step}__{k}"): v for k, v in grid.items()}

def _predict_proba(estimator, X: pd.DataFrame) -> np.ndarray | None:
    try:
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X)[:, 1]
        if isinstance(estimator, Pipeline):
            last = estimator.steps[-1][1]
            if hasattr(last, "predict_proba"):
                return estimator.predict_proba(X)[:, 1]
    except Exception:
        pass
    return None

def _fit_with_params(base_estimator, params: dict) -> object:
    est = clone(base_estimator)
    try:
        est.set_params(**params)
    except Exception:
        pass
    return est

def _choose_threshold_from_oof(y_true: np.ndarray, oof_proba: np.ndarray, df_oof: pd.DataFrame | None,
                               fee_per_trade: float, slippage_bps: float) -> float:
    if (not HAS_CALC_METRICS) or (oof_proba is None) or (df_oof is None):
        return 0.5
    best_thr, best_score = 0.5, -1e18
    for thr in np.linspace(0.3, 0.7, 41):
        y_pred = (oof_proba >= thr).astype(int)
        try:
            m = calculate_metrics(
                y_true=y_true, y_pred=y_pred, df=df_oof,
                fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                annualize_sharpe=True
            )
            score = float(m.get("profit_net", np.nan))
        except Exception:
            score = np.nan
        if not np.isnan(score) and score > best_score:
            best_score, best_thr = score, float(thr)
    return float(best_thr)

# -------------- Monte Carlo --------------
def _mc_block_bootstrap_indices(n: int, block_len: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    b = max(1, int(block_len))
    k = (n + b - 1) // b
    starts = np.random.randint(0, n, size=k)
    idx = []
    for s in starts:
        block = [(s + t) % n for t in range(b)]
        idx.extend(block)
        if len(idx) >= n:
            break
    return np.asarray(idx[:n], dtype=int)

def _mc_eval_holdout_adaptive(
    estimator,
    df_hold: pd.DataFrame,
    features: list[str],
    base_threshold: float = 0.5,
    iters: int = 200,
    block_len: int = 100,
    fee_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    min_trades: int = 20,
    trial_thresholds: tuple[float, ...] = (0.5, 0.48, 0.46, 0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28, 0.26, 0.25),
) -> dict[str, Any]:
    if (not HAS_CALC_METRICS) or df_hold is None or len(df_hold) < 10:
        return {}

    try:
        Xh = df_hold[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_true = df_hold["target"].astype(int).to_numpy()
    except Exception:
        return {}

    proba_all = None
    try:
        if hasattr(estimator, "predict_proba"):
            pr = estimator.predict_proba(Xh)
            proba_all = pr[:, 1] if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 2 else np.asarray(pr).ravel()
        elif hasattr(estimator, "decision_function"):
            z = np.asarray(estimator.decision_function(Xh)).ravel()
            proba_all = 1.0 / (1.0 + np.exp(-z))
    except Exception:
        proba_all = None

    def _q(a, p):
        a = [x for x in a if x is not None and np.isfinite(x)]
        return float(np.nanpercentile(a, p)) if len(a) else None

    sharpe_vals, dd_vals, profit_vals = [], [], []
    valid_sharpes = []

    for _ in range(int(max(1, iters))):
        idx = _mc_block_bootstrap_indices(len(y_true), int(max(1, block_len)))
        yt = y_true[idx]
        dfb = df_hold.iloc[idx]

        if proba_all is not None:
            pb = proba_all[idx]
            thr_used = float(base_threshold)
            yp = (pb >= thr_used).astype(int)
            if yp.sum() < min_trades:
                used = False
                for thr in trial_thresholds:
                    thr_used = float(thr)
                    yp = (pb >= thr_used).astype(int)
                    if yp.sum() >= min_trades:
                        used = True
                        break
                if not used:
                    thr_used = float(trial_thresholds[-1])
                    yp = (pb >= thr_used).astype(int)
        else:
            yp = estimator.predict(dfb[features])

        try:
            m = calculate_metrics(
                y_true=yt, y_pred=yp, df=dfb,
                fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                annualize_sharpe=True
            )
            sv = m.get("sharpe", np.nan)
            if sv is not None and np.isfinite(sv):
                valid_sharpes.append(float(sv))
            sharpe_vals.append(float(sv) if np.isfinite(sv) else np.nan)
            dd_vals.append(float(m.get("max_drawdown", np.nan)))
            profit_vals.append(float(m.get("profit_net", np.nan)))
        except Exception:
            sharpe_vals.append(np.nan)
            dd_vals.append(np.nan)
            profit_vals.append(np.nan)

    res = {
        "iters": int(iters),
        "block_len": int(block_len),
        "min_trades": int(min_trades),
        "thr_scan": list(trial_thresholds),
        "sharpe": {"p10": _q(sharpe_vals, 10) if len(valid_sharpes) else None,
                   "p50": _q(sharpe_vals, 50) if len(valid_sharpes) else None,
                   "p90": _q(sharpe_vals, 90) if len(valid_sharpes) else None},
        "max_drawdown": {"p90": _q(dd_vals, 90)},
        "profit_net": {"p10": _q(profit_vals, 10), "p50": _q(profit_vals, 50), "p90": _q(profit_vals, 90)},
    }
    if res["sharpe"]["p50"] is None:
        res["note"] = "no_trades_or_zero_variance"
    return res

# ------------------- Hlavní trénink -------------------
def train_and_evaluate_model(
    df: pd.DataFrame,
    estimator_name: str = "hgbt",
    param_grid: dict | None = None,
    n_splits: int = 5,
    embargo: int = 10,
    fee_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    calibrate: bool = False,
    on_progress=None,
    **kwargs,
) -> dict:
    """
    Trénink na posledním okně (bez window sweepu) + čistý holdout + (volitelně) MC evaluace holdoutu.
    """
    if "param_grid" in kwargs and kwargs["param_grid"] is not None:
        param_grid = kwargs.pop("param_grid")
    if param_grid is None:
        param_grid = kwargs.pop("grid", None) or kwargs.pop("params_grid", None)

    holdout_bars: int = int(kwargs.pop("holdout_bars", 250))
    name_prefix: str | None = kwargs.pop("name_prefix", None)
    meta_extra: dict[str, Any] = kwargs.pop("meta_extra", {})

    top_k_features: int | None = kwargs.pop("top_k_features", None)
    ranking_folds: int = int(kwargs.pop("ranking_folds", 3))

    mc_enabled: bool = bool(kwargs.pop("mc_enabled", True))
    mc_iters: int = int(kwargs.pop("mc_iters", 200))
    mc_block_len: int = int(kwargs.pop("mc_block_len", 100))

    annualize_sharpe: bool = bool(kwargs.pop("annualize_sharpe", True))

    _ = kwargs

    if "target" not in df.columns:
        raise ValueError("DataFrame musí obsahovat 'target'.")
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame musí obsahovat 'timestamp'.")
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # split
    n_total = len(df)
    n_hold  = min(max(int(holdout_bars), 0), max(n_total - 50, 0))
    if n_hold > 0:
        df_train = df.iloc[: n_total - n_hold].reset_index(drop=True)
        df_hold  = df.iloc[n_total - n_hold :].reset_index(drop=True)
    else:
        df_train = df
        df_hold = None

    # featury
    feats_all = _select_feature_columns(df_train)
    if not feats_all:
        raise ValueError("Nenalezeny numerické featury.")
    feats = feats_all[:]
    if top_k_features and top_k_features > 0 and len(feats) > top_k_features:
        try:
            cv_rank = PurgedWalkForwardSplit(n_splits=max(2, ranking_folds), embargo=embargo)
            imp_acc = np.zeros(len(feats))
            for tr_idx, _ in cv_rank.split(df_train[feats]):
                Xtr = df_train.iloc[tr_idx][feats].replace([np.inf, -np.inf], np.nan)
                ytr = df_train.iloc[tr_idx]["target"].astype(int).to_numpy()
                est_rank = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
                est_rank.fit(Xtr, ytr)
                fi = getattr(est_rank, "feature_importances_", None)
                if fi is not None and len(fi) == len(feats):
                    imp_acc += np.asarray(fi, dtype=float)
            order = np.argsort(imp_acc)[::-1]
            feats = [feats[i] for i in order[:top_k_features]]
        except Exception:
            pass

    X_all = df_train[feats].replace([np.inf, -np.inf], np.nan)
    y_all = df_train["target"].astype(int).to_numpy()

    base_estimator, default_grid = _build_estimator(estimator_name)
    base_estimator = _ensure_pipeline(base_estimator)
    raw_grid = param_grid if isinstance(param_grid, dict) and len(param_grid) > 0 else default_grid
    grid_base = _namespaced_param_grid(base_estimator, raw_grid)
    all_param_sets = list(ParameterGrid(grid_base)) if grid_base else [dict()]

    cv = PurgedWalkForwardSplit(n_splits=n_splits, embargo=embargo)
    step_idx, total = 0, max(1, len(all_param_sets))
    best_score, best_params, best_estimator, best_oof = -1e18, None, None, None

    def _emit(onp, idx, total, params, mean, std):
        if not onp:
            return
        try:
            onp(int(idx), int(total), dict(params), float(mean), float(std))
        except TypeError:
            onp(f"[CV {idx}/{total}] score={mean:.4f} std={std:.4f} params={params}")

    for params in all_param_sets:
        step_idx += 1
        fold_scores, fold_sizes = [], []
        tmp_oof = np.full(shape=(len(X_all),), fill_value=np.nan, dtype=float)

        for tr_idx, te_idx in cv.split(X_all):
            X_tr, y_tr = X_all.iloc[tr_idx], y_all[tr_idx]
            X_te, y_te = X_all.iloc[te_idx], y_all[te_idx]
            df_te = df_train.iloc[te_idx]
            est = _fit_with_params(base_estimator, params)
            est.fit(X_tr, y_tr)
            try:
                score = pnl_scorer(est, X_te, y_te, df_te, fee=fee_per_trade, slippage=slippage_bps)
            except Exception:
                pred = est.predict(X_te)
                score = float((pred == y_te).mean())
            fold_scores.append(float(score))
            fold_sizes.append(len(te_idx))
            proba = _predict_proba(est, X_te)
            if proba is not None:
                tmp_oof[te_idx] = proba

        mean_score = float(np.average(fold_scores, weights=fold_sizes)) if fold_scores else -1e18
        std_score  = float(np.std(fold_scores)) if fold_scores else float("nan")
        _emit(on_progress, step_idx, total, params, mean_score, std_score)
        if mean_score > best_score:
            best_score, best_params = mean_score, params
            best_estimator = _fit_with_params(base_estimator, params)
            best_oof = tmp_oof.copy()

    if best_estimator is None:
        best_estimator = base_estimator
    best_estimator.fit(X_all, y_all)

    calibrated_estimator = best_estimator
    if calibrate:
        try:
            calibrated_estimator = CalibratedClassifierCV(best_estimator, method="isotonic", cv=3)
            calibrated_estimator.fit(X_all, y_all)
        except Exception:
            calibrated_estimator = best_estimator

    decision_threshold = 0.5
    if best_oof is not None and HAS_CALC_METRICS:
        valid = np.isfinite(best_oof)
        if valid.any():
            df_oof = df_train.iloc[valid.nonzero()[0]]
            y_oof = df_oof["target"].astype(int).to_numpy()
            decision_threshold = _choose_threshold_from_oof(
                y_true=y_oof, oof_proba=best_oof[valid], df_oof=df_oof,
                fee_per_trade=fee_per_trade, slippage_bps=slippage_bps
            )

    holdout_metrics, mc_summary = {}, {}
    n_signals_holdout: int | None = None
    base_threshold_mc: float = float(decision_threshold)

    if df_hold is not None and len(df_hold) >= 10:
        used_feats = list(X_all.columns)
        Xh = df_hold[used_feats].replace([np.inf, -np.inf], np.nan)
        yh = df_hold["target"].astype(int).to_numpy()
        proba = None
        try:
            if HAS_CALC_METRICS:
                proba = _predict_proba(calibrated_estimator, Xh)
                if proba is not None:
                    ypred = (proba >= float(decision_threshold)).astype(int)
                    n_signals_holdout = int((proba >= float(decision_threshold)).sum())
                else:
                    ypred = calibrated_estimator.predict(Xh)
                    n_signals_holdout = int((ypred == 1).sum())

                holdout_metrics = calculate_metrics(
                    y_true=yh, y_pred=ypred, df=df_hold,
                    fee_per_trade=fee_per_trade, slippage_bps=slippage_bps,
                    annualize_sharpe=annualize_sharpe
                )

                mc_target_trades = 100
                if proba is not None and len(proba) > 0 and n_signals_holdout is not None:
                    if n_signals_holdout < max(20, 0.2 * mc_target_trades):
                        frac = max(1, mc_target_trades) / max(1, len(proba))
                        q = 1.0 - min(0.95, max(0.01, float(frac)))
                        base_threshold_mc = max(float(np.quantile(proba, q)), 0.25)

                if mc_enabled:
                    mc_summary = _mc_eval_holdout_adaptive(
                        estimator=calibrated_estimator,
                        df_hold=df_hold,
                        features=used_feats,
                        base_threshold=float(base_threshold_mc),
                        iters=mc_iters,
                        block_len=mc_block_len,
                        fee_per_trade=fee_per_trade,
                        slippage_bps=slippage_bps,
                        min_trades=20,
                    )
            else:
                acc = float((calibrated_estimator.predict(Xh) == yh).mean())
                holdout_metrics = {"accuracy": acc}
        except Exception:
            pass

    # --- uložení
    out_dir = _model_dir()
    ts = _now_str()
    est_short = (estimator_name or "model").lower()
    fname = f"{est_short}_{ts}.pkl" if not name_prefix else f"{name_prefix}_{est_short}_{ts}.pkl"
    fpath = out_dir / fname

    payload = {
        "model": calibrated_estimator,
        "features": list(X_all.columns),
        "estimator_name": estimator_name,
        "best_params": best_params or {},
        "cv_results_full": [],
        "decision_threshold": float(decision_threshold),
        "created_at": ts,
        "version": "1.6_mc_ann",
        "fee_per_trade": float(fee_per_trade),
        "slippage_bps": float(slippage_bps),
        "n_total_bars": int(n_total),
        "n_train_bars": len(df_train),
        "n_holdout_bars": int(len(df_hold) if df_hold is not None else 0),
        "annualize_sharpe": bool(annualize_sharpe),
        **(meta_extra or {}),
    }
    import joblib
    joblib.dump(payload, fpath)

    meta = {
        "created_at": ts,
        "created_at_iso": datetime.now().isoformat(),
        "estimator_name": estimator_name,
        "best_params": best_params or {},
        "decision_threshold": float(decision_threshold),
        "trained_features": list(X_all.columns),
        "n_features": len(X_all.columns),
        "n_total_bars": int(n_total),
        "n_train_bars": len(df_train),
        "n_holdout_bars": int(len(df_hold) if df_hold is not None else 0),
        "class_to_dir": {0: "SHORT", 1: "LONG"},
        "classes": None,
        "annualize_sharpe": bool(annualize_sharpe),
        **(meta_extra or {}),
        "metrics": {**holdout_metrics, **({"n_signals_holdout": n_signals_holdout} if n_signals_holdout is not None else {})},
        "mc": mc_summary,
    }
    try:
        est_for_cls = calibrated_estimator
        if isinstance(calibrated_estimator, Pipeline):
            est_for_cls = calibrated_estimator.steps[-1][1]
        if hasattr(est_for_cls, "classes_"):
            meta["classes"] = [int(c) for c in list(getattr(est_for_cls, "classes_"))]
    except Exception:
        meta["classes"] = None

    meta_path = fpath.with_name(f"{fpath.stem}_meta.json")
    try:
        meta_path.write_text(jsonlib.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    def _emit_done(onp, idx, total, params, mean, std):
        if not onp:
            return
        try:
            onp(int(idx), int(total), dict(params), float(mean), float(std))
        except TypeError:
            onp(f"[DONE] {params}")
    _emit_done(on_progress, total, total, {"saved_model": str(fpath)}, float(best_score), 0.0)

    return {
        "output_path": str(fpath),
        "best_score": float(best_score),
        "best_params": best_params or {},
        "n_features": len(X_all.columns),
        "decision_threshold": float(decision_threshold),
        "cv_records_len": 0,
        "n_total_bars": int(n_total),
        "n_train_bars": len(df_train),
        "n_holdout_bars": int(len(df_hold) if df_hold is not None else 0),
    }


# Backwards compatibility: some tests / CLI expect `train_simple_model`
# to be importable from `ibkr_trading_bot.model.train_models`.
try:
    from ibkr_trading_bot.model.data_split import train_simple_model as train_simple_model  # type: ignore
except Exception:
    # If import fails, leave it to callers to import from the canonical module.
    train_simple_model = None
