from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ibkr_trading_bot.core.services.auto_threshold_search import run_auto_threshold_search
from ibkr_trading_bot.core.services.evaluation_service import EvaluationService
from ibkr_trading_bot.core.services.model_service import (
    build_sklearn_version_warning,
    merge_model_metadata,
    read_sidecar_model_meta,
)
from ibkr_trading_bot.features.feature_engineering import prepare_dataset_with_targets


TAB5_HOLDOUT_RANKING_KEY = "tab5_holdout_ranking"


@dataclass
class PreparedEvaluationData:
    data_path: str
    X_full: pd.DataFrame | np.ndarray
    y_true_full: np.ndarray | None
    df_for_metrics_full: pd.DataFrame


@dataclass
class EvaluationPayload:
    X_current: pd.DataFrame | np.ndarray
    y_true_current: np.ndarray
    df_current: pd.DataFrame
    close_series: pd.Series | None
    confidence_arr: np.ndarray
    y_pred_raw: np.ndarray
    y_pred_used: np.ndarray
    results: dict[str, Any]
    scope_info: dict[str, Any]
    threshold_source: str
    thr_short: float
    thr_long: float
    entry_threshold: float
    exit_threshold: float


@dataclass
class AutoThresholdPayload:
    best_entry: float
    best_exit: float
    best_score: float
    best_metrics: dict[str, Any] | None


@dataclass
class LoadedPredictor:
    predictor: Any
    metadata: dict[str, Any]
    model_path: str
    version_warning: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if np.isfinite(out):
            return float(out)
    except Exception:
        pass
    return None


def finite_or_none(value: Any) -> float | None:
    out = safe_float(value)
    return out if out is not None else None


def pick_metric(metrics: dict[str, Any] | None, *keys: str) -> Any:
    source = metrics if isinstance(metrics, dict) else {}
    for key in keys:
        if key in source:
            value = source.get(key)
            if value is not None:
                return value
    return None


def extract_predictor_from_object(obj: Any) -> tuple[Any, dict[str, Any] | None]:
    if hasattr(obj, "predict"):
        return obj, None
    if isinstance(obj, dict):
        for key in ["model", "estimator", "pipeline", "clf", "best_estimator_", "sk_model", "predictor"]:
            value = obj.get(key)
            if hasattr(value, "predict"):
                return value, obj
        for value in obj.values():
            if hasattr(value, "predict"):
                return value, obj
        raise ValueError("Ve slovniku neni zadny objekt s `.predict`.")
    if isinstance(obj, (tuple, list)):
        for value in obj:
            if hasattr(value, "predict"):
                return value, None
        raise ValueError("V tuple/listu neni zadna polozka s `.predict`.")
    raise ValueError(f"Neocekavany typ ulozeneho modelu: {type(obj).__name__}.")


def load_predictor_with_merged_meta(model_path: str | Path) -> LoadedPredictor:
    normalized_path = normalize_path(model_path)
    obj = joblib.load(normalized_path)
    predictor, embedded_meta = extract_predictor_from_object(obj)
    metadata = merge_model_metadata(embedded_meta if isinstance(embedded_meta, dict) else {}, read_sidecar_model_meta(normalized_path))
    version_warning = build_sklearn_version_warning(metadata, model_path=normalized_path)
    return LoadedPredictor(
        predictor=predictor,
        metadata=metadata,
        model_path=normalized_path,
        version_warning=version_warning,
    )


def extract_X_y_eval(prepared: Any) -> tuple[pd.DataFrame | np.ndarray, np.ndarray | None]:
    if isinstance(prepared, (tuple, list)):
        if len(prepared) >= 2:
            return prepared[0], prepared[1]
        return prepared[0], None
    if isinstance(prepared, dict):
        X = prepared.get("X") or prepared.get("features") or prepared.get("data") or prepared.get("df")
        y = prepared.get("y") or prepared.get("target") or prepared.get("y_true")
        if X is None:
            raise ValueError("V dictu chybi klic 'X'/'features'/'data'.")
        return X, y
    if isinstance(prepared, pd.DataFrame):
        X, y = prepared, None
        for candidate in ["target", "y", "label"]:
            if candidate in prepared.columns:
                y = prepared[candidate].values
                X = prepared.drop(columns=[candidate])
                break
        return X, y
    if isinstance(prepared, np.ndarray):
        return prepared, None
    raise ValueError("Neocekavany navratovy typ z prepare_dataset_with_targets(df).")


def load_prepared_evaluation_data(data_path: str | Path, progress_cb=None) -> PreparedEvaluationData:
    normalized_path = normalize_path(data_path)
    if callable(progress_cb):
        progress_cb("Vyhodnoceni: nacitam CSV...")
    df = pd.read_csv(normalized_path, encoding="utf-8", engine="python")
    if callable(progress_cb):
        progress_cb("Vyhodnoceni: pripravuji dataset...")
    prepared = prepare_dataset_with_targets(df)
    X, y_true = extract_X_y_eval(prepared)
    df_for_metrics = prepared if isinstance(prepared, pd.DataFrame) else df
    return PreparedEvaluationData(
        data_path=normalized_path,
        X_full=X,
        y_true_full=(np.asarray(y_true) if y_true is not None else None),
        df_for_metrics_full=df_for_metrics,
    )


def feature_names_for_model_eval(model: Any) -> list[str] | None:
    try:
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            return [str(x) for x in list(names)]
    except Exception:
        pass
    try:
        names = getattr(model, "feature_name_", None)
        if names is not None:
            out = [str(x) for x in list(names) if str(x)]
            if out:
                return out
    except Exception:
        pass
    try:
        booster = getattr(model, "booster_", None)
        if booster is not None and hasattr(booster, "feature_name"):
            names = booster.feature_name()
            out = [str(x) for x in list(names) if str(x)]
            if out:
                return out
    except Exception:
        pass
    try:
        steps = getattr(model, "steps", None)
        if steps:
            last = steps[-1][1]
            names = getattr(last, "feature_names_in_", None)
            if names is not None:
                return [str(x) for x in list(names)]
            names = getattr(last, "feature_name_", None)
            if names is not None:
                out = [str(x) for x in list(names) if str(x)]
                if out:
                    return out
            booster = getattr(last, "booster_", None)
            if booster is not None and hasattr(booster, "feature_name"):
                names = booster.feature_name()
                out = [str(x) for x in list(names) if str(x)]
                if out:
                    return out
    except Exception:
        pass
    return None


def tail_rows_eval(obj: Any, n_rows: int):
    if obj is None:
        return None
    n = int(max(0, n_rows))
    if isinstance(obj, pd.DataFrame):
        return obj.tail(n).reset_index(drop=True)
    if isinstance(obj, pd.Series):
        return obj.tail(n).reset_index(drop=True)
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr
    return arr[-n:] if n < arr.shape[0] else arr


def infer_holdout_bars_from_metadata(meta: dict[str, Any], n_rows: int) -> int | None:
    n = int(max(0, n_rows))
    if n <= 0:
        return None
    try:
        n_hold = int(meta.get("n_holdout_bars", 0))
        if n_hold > 0:
            return int(min(n, n_hold))
    except Exception:
        pass
    hold_sel = meta.get("holdout_selection") if isinstance(meta, dict) else None
    if isinstance(hold_sel, dict):
        try:
            applied = int(hold_sel.get("applied_bars", 0))
            if applied > 0:
                return int(min(n, applied))
        except Exception:
            pass
        try:
            pct = float(hold_sel.get("requested_pct"))
            if np.isfinite(pct) and pct > 0.0:
                calc = int(round(float(n) * float(np.clip(pct, 0.0, 0.95))))
                if calc > 0:
                    return int(min(n, calc))
        except Exception:
            pass
    return None


def apply_eval_scope(
    X: pd.DataFrame | np.ndarray,
    y_true: np.ndarray | None,
    df_for_metrics: pd.DataFrame | None,
    scope_mode: str,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray | None, pd.DataFrame | None, dict[str, Any]]:
    lengths: list[int] = []
    for obj in (X, y_true, df_for_metrics):
        if obj is None:
            continue
        try:
            lengths.append(int(len(obj)))
        except Exception:
            pass
    if not lengths:
        raise ValueError("Nelze urcit delku datasetu pro evaluaci.")

    n_base = int(max(0, min(lengths)))
    if n_base <= 0:
        raise ValueError("Dataset pro evaluaci je prazdny.")

    X_aligned = tail_rows_eval(X, n_base)
    y_aligned = tail_rows_eval(y_true, n_base) if y_true is not None else None
    df_aligned = tail_rows_eval(df_for_metrics, n_base) if df_for_metrics is not None else None

    mode = scope_mode if scope_mode in {"holdout", "full"} else "holdout"
    if mode == "holdout":
        n_hold = infer_holdout_bars_from_metadata(metadata or {}, n_base)
        if n_hold is not None and n_hold > 0:
            n_eval = int(min(n_base, n_hold))
            X_eval = tail_rows_eval(X_aligned, n_eval)
            y_eval = tail_rows_eval(y_aligned, n_eval) if y_aligned is not None else None
            df_eval = tail_rows_eval(df_aligned, n_eval) if df_aligned is not None else None
        else:
            n_eval = n_base
            X_eval, y_eval, df_eval = X_aligned, y_aligned, df_aligned
    else:
        n_eval = n_base
        X_eval, y_eval, df_eval = X_aligned, y_aligned, df_aligned

    return X_eval, y_eval, df_eval, {
        "mode": mode,
        "applied_rows": int(n_eval),
        "total_rows": int(n_base),
    }


def coerce_features_for_model_eval(X: pd.DataFrame | np.ndarray, model: Any, metadata: dict[str, Any]):
    if not isinstance(X, pd.DataFrame):
        return X

    dfX = X.copy()
    for col in dfX.columns:
        if pd.api.types.is_datetime64_any_dtype(dfX[col]):
            dfX[col] = dfX[col].astype("int64") // 10**6
        elif dfX[col].dtype == "object":
            try:
                parsed = pd.to_datetime(dfX[col], errors="raise")
                dfX[col] = parsed.astype("int64") // 10**6
            except Exception:
                pass

    for column in list(dfX.columns):
        if (not pd.api.types.is_bool_dtype(dfX[column])) and (not pd.api.types.is_numeric_dtype(dfX[column])):
            dfX.drop(columns=[column], inplace=True, errors="ignore")

    expected = None
    if isinstance(metadata, dict):
        expected = metadata.get("expected_features") or metadata.get("features")
    if isinstance(expected, (list, tuple)) and all(isinstance(item, str) for item in expected):
        for item in expected:
            if item not in dfX.columns:
                dfX[item] = 0.0
        dfX = dfX[list(expected)]
        med = dfX.median(numeric_only=True)
        dfX = dfX.fillna(med).fillna(0.0)
        for column in dfX.columns:
            if not pd.api.types.is_bool_dtype(dfX[column]):
                dfX[column] = dfX[column].astype("float32", copy=False)
        return dfX

    names = feature_names_for_model_eval(model)
    if names is not None:
        for item in names:
            if item not in dfX.columns:
                dfX[item] = 0.0
        dfX = dfX[names]

    med = dfX.median(numeric_only=True)
    dfX = dfX.fillna(med).fillna(0.0)
    for column in dfX.columns:
        if not pd.api.types.is_bool_dtype(dfX[column]):
            dfX[column] = dfX[column].astype("float32", copy=False)
    return dfX


def align_X_for_model_eval(model: Any, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        Xdf = X.copy()
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        Xdf = pd.DataFrame(arr)
    names = feature_names_for_model_eval(model)
    if names:
        if all(name in Xdf.columns for name in names):
            Xdf = Xdf.reindex(columns=names, fill_value=0.0)
        elif len(Xdf.columns) == len(names):
            Xdf.columns = list(names)
        else:
            for name in names:
                if name not in Xdf.columns:
                    Xdf[name] = 0.0
            Xdf = Xdf.reindex(columns=names, fill_value=0.0)
    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med).fillna(0.0)
    for column in Xdf.columns:
        if not pd.api.types.is_bool_dtype(Xdf[column]):
            Xdf[column] = Xdf[column].astype("float32", copy=False)
    return Xdf


def resolve_ternary_thresholds_eval(metadata: dict[str, Any]) -> tuple[float, float, str]:
    meta = metadata if isinstance(metadata, dict) else {}
    tshort = safe_float(meta.get("ternary_threshold_short"))
    tlong = safe_float(meta.get("ternary_threshold_long"))
    user = meta.get("user_settings")
    if isinstance(user, dict):
        if tshort is None:
            tshort = safe_float(user.get("ternary_threshold_short_eval"))
        if tlong is None:
            tlong = safe_float(user.get("ternary_threshold_long_eval"))
    if not isinstance(tshort, (int, float)) or not isinstance(tlong, (int, float)):
        raise ValueError(
            "Model neobsahuje platne ternarni prahy (ternary_threshold_short/long). "
            "Nahraj model natrenovany v nove pipeline."
        )
    return float(tshort), float(tlong), "model"


def apply_confidence_threshold_eval(raw_pred, confidence, threshold):
    arr = np.asarray(raw_pred).copy()
    conf = np.asarray(confidence).reshape(-1)
    thr = float(threshold)
    mask_low = conf < thr
    try:
        arr[mask_low] = 0
    except Exception:
        tmp = np.array(arr, dtype=object)
        tmp[mask_low] = 0
        arr = tmp
    return arr


def apply_exit_threshold_eval(y_pred: np.ndarray, confidence: np.ndarray, exit_thr: float) -> np.ndarray:
    arr = np.asarray(y_pred).copy()
    conf = np.asarray(confidence).reshape(-1)
    mask_low = conf < float(exit_thr)
    open_pos = np.abs(arr) > 0.5
    arr[mask_low & open_pos] = 0
    return arr


def normalize_pred_eval(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=object)
    out = np.zeros(a.shape, dtype=float)
    num_mask = np.array([isinstance(x, (int, float, np.number)) for x in a], dtype=bool)
    out[num_mask] = np.sign(a[num_mask].astype(float))
    txt = np.char.lower(a.astype(str))
    out[(txt == "long") | (txt == "buy") | (txt == "up") | (txt == "1") | (txt == "+1")] = 1.0
    out[(txt == "short") | (txt == "sell") | (txt == "down") | (txt == "-1")] = -1.0
    return out


def safe_close_series_eval(df: pd.DataFrame | None):
    if not isinstance(df, pd.DataFrame):
        return None
    for column in ["close", "Close", "CLOSE", "adj_close", "Adj Close"]:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return None


def run_model_evaluation(
    *,
    model: Any,
    metadata: dict[str, Any] | None,
    data_path: str | Path | None = None,
    prepared_data: PreparedEvaluationData | None = None,
    scope_mode: str,
    fee_per_trade: float,
    entry_threshold: float,
    exit_threshold: float,
    progress_cb=None,
) -> EvaluationPayload:
    if model is None:
        raise ValueError("Nejprve vyber model (.pkl).")
    if not hasattr(model, "predict"):
        raise AttributeError("Nacteny objekt nema metodu `.predict`.")

    prepared = prepared_data if prepared_data is not None else load_prepared_evaluation_data(data_path, progress_cb=progress_cb)
    X, y_true, df_for_metrics, scope_info = apply_eval_scope(
        prepared.X_full,
        prepared.y_true_full,
        prepared.df_for_metrics_full,
        scope_mode,
        metadata or {},
    )
    X = coerce_features_for_model_eval(X, model, metadata or {})
    if y_true is None:
        raise ValueError("Po priprave datasetu chybi cilova promenna (target/y).")

    if callable(progress_cb):
        progress_cb("Vyhodnoceni: pocitam predikce...")
    proba = None
    X_pred = align_X_for_model_eval(model, X)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_pred)
        except Exception:
            proba = None

    thr_short, thr_long, threshold_source = resolve_ternary_thresholds_eval(metadata or {})
    if proba is None or proba.ndim != 2 or int(proba.shape[1]) != 3:
        raise ValueError("Tab 3 vyzaduje ternarni model s predict_proba (3 tridy: short/neutral/long).")

    prob_long = proba[:, 2]
    prob_short = proba[:, 0]
    y_pred_raw = np.where(prob_long >= thr_long, 1, np.where(prob_short >= thr_short, -1, 0))
    confidence_arr = np.max(proba, axis=1)
    y_pred_used, results = recalculate_metrics_from_predictions(
        y_pred_raw=np.asarray(y_pred_raw),
        confidence_arr=np.asarray(confidence_arr),
        y_true_current=np.asarray(y_true),
        df_current=df_for_metrics,
        fee_per_trade=float(fee_per_trade),
        entry_threshold=float(entry_threshold),
        exit_threshold=float(exit_threshold),
        progress_cb=progress_cb,
    )

    return EvaluationPayload(
        X_current=X,
        y_true_current=np.asarray(y_true),
        df_current=df_for_metrics,
        close_series=safe_close_series_eval(df_for_metrics),
        confidence_arr=np.asarray(confidence_arr),
        y_pred_raw=np.asarray(y_pred_raw),
        y_pred_used=np.asarray(y_pred_used),
        results=results,
        scope_info=scope_info,
        threshold_source=threshold_source,
        thr_short=float(thr_short),
        thr_long=float(thr_long),
        entry_threshold=float(entry_threshold),
        exit_threshold=float(exit_threshold),
    )


def recalculate_metrics_from_predictions(
    *,
    y_pred_raw,
    confidence_arr,
    y_true_current,
    df_current,
    fee_per_trade: float,
    entry_threshold: float,
    exit_threshold: float,
    progress_cb=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if callable(progress_cb):
        progress_cb("Prepocitavam metriky...")
    y_pred_used = apply_confidence_threshold_eval(y_pred_raw, confidence_arr, entry_threshold)
    y_pred_used = normalize_pred_eval(y_pred_used)
    if float(exit_threshold) > 0.0:
        y_pred_used = apply_exit_threshold_eval(y_pred_used, confidence_arr, exit_threshold)
        y_pred_used = normalize_pred_eval(y_pred_used)

    results = EvaluationService(None, None, None).calculate_metrics(
        y_true=np.asarray(y_true_current),
        y_pred=y_pred_used,
        df=df_current,
        fee_per_trade=float(fee_per_trade),
        slippage_bps=0.0,
        rolling_window=200,
        annualize_sharpe=False,
    )
    if not isinstance(results, dict) or not results:
        raise ValueError("Vypocet metrik vratil prazdny vysledek.")
    return np.asarray(y_pred_used), results


def run_auto_threshold_search_from_context(
    *,
    y_pred_raw,
    confidence_arr,
    y_true_current,
    df_current,
    fee_per_trade: float,
    current_entry: float,
    current_exit: float,
    progress_cb=None,
    should_run=None,
) -> AutoThresholdPayload:
    def evaluate_pair(entry_thr: float, exit_thr: float) -> tuple[float, dict[str, Any]]:
        y = apply_confidence_threshold_eval(y_pred_raw, confidence_arr, float(entry_thr))
        y = normalize_pred_eval(y)
        if float(exit_thr) > 0.0:
            y = apply_exit_threshold_eval(y, confidence_arr, float(exit_thr))
            y = normalize_pred_eval(y)
        metrics = EvaluationService(None, None, None).calculate_metrics(
            y_true=np.asarray(y_true_current),
            y_pred=y,
            df=df_current,
            fee_per_trade=float(fee_per_trade),
            slippage_bps=0.0,
            rolling_window=200,
            annualize_sharpe=False,
        )
        profit = pick_metric(metrics, "profit_net", "profit_gross", "profit")
        score = safe_float(profit)
        return (score if score is not None else float("-inf")), metrics

    def pick_metric_for_search(metrics: dict[str, Any] | None, metric_name: str):
        if metric_name == "max_dd":
            return pick_metric(metrics, "max_dd", "max_drawdown_net", "max_drawdown")
        if metric_name == "trades":
            return pick_metric(metrics, "trades", "num_trades")
        raise KeyError(metric_name)

    result = run_auto_threshold_search(
        current_entry=float(current_entry),
        current_exit=float(current_exit),
        evaluate_pair=evaluate_pair,
        pick_metric=pick_metric_for_search,
        progress_cb=progress_cb,
        should_run=should_run,
    )
    return AutoThresholdPayload(
        best_entry=float(result.best_entry),
        best_exit=float(result.best_exit),
        best_score=float(result.best_score),
        best_metrics=result.best_metrics,
    )


def build_tab5_holdout_ranking_payload(
    *,
    data_path: str | Path,
    fee_per_trade: float,
    entry_threshold: float | None,
    exit_threshold: float | None,
    metrics: dict[str, Any] | None = None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    normalized_path = normalize_path(data_path)
    csv_stat = Path(normalized_path).stat()
    out: dict[str, Any] = {
        "status": str(status),
        "csv_path": normalized_path,
        "csv_size": int(csv_stat.st_size),
        "csv_mtime_ns": int(csv_stat.st_mtime_ns),
        "fee_per_trade": float(fee_per_trade),
        "scope": "holdout",
        "entry_threshold": finite_or_none(entry_threshold),
        "exit_threshold": finite_or_none(exit_threshold),
        "profit_h": finite_or_none(pick_metric(metrics, "profit_net", "profit_gross", "profit")),
        "max_dd_h": finite_or_none(pick_metric(metrics, "max_dd", "max_drawdown_net", "max_drawdown")),
        "trades_h": finite_or_none(pick_metric(metrics, "trades", "num_trades")),
        "evaluated_at": utc_now_iso(),
    }
    if error:
        out["error"] = str(error)
    return out


def get_tab5_holdout_ranking(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(meta, dict):
        return None
    ranking = meta.get(TAB5_HOLDOUT_RANKING_KEY)
    return ranking if isinstance(ranking, dict) else None


def is_tab5_holdout_ranking_stale(
    meta: dict[str, Any] | None,
    *,
    data_path: str | Path,
    fee_per_trade: float,
    model_path: str | Path | None = None,
    meta_path: str | Path | None = None,
) -> bool:
    ranking = get_tab5_holdout_ranking(meta)
    if not isinstance(ranking, dict):
        return True
    normalized_path = normalize_path(data_path)
    csv_stat = Path(normalized_path).stat()
    if ranking.get("scope") != "holdout":
        return True
    if str(ranking.get("csv_path") or "") != normalized_path:
        return True
    if int(ranking.get("csv_size", -1) or -1) != int(csv_stat.st_size):
        return True
    if int(ranking.get("csv_mtime_ns", -1) or -1) != int(csv_stat.st_mtime_ns):
        return True
    ranking_fee = safe_float(ranking.get("fee_per_trade"))
    if ranking_fee is None or not np.isclose(float(ranking_fee), float(fee_per_trade), atol=1e-12):
        return True
    if meta_path is not None and model_path is not None:
        try:
            if Path(meta_path).exists() and Path(meta_path).stat().st_mtime_ns < Path(model_path).stat().st_mtime_ns:
                return True
        except OSError:
            return True
    return False


def ranking_status_from_error_message(message: str) -> str:
    msg = str(message or "").lower()
    unsupported_markers = [
        "ternarni model",
        "ternarni prahy",
        "predict_proba",
        "3 tridy",
        "threshold_short",
        "threshold_long",
    ]
    return "unsupported" if any(marker in msg for marker in unsupported_markers) else "error"
