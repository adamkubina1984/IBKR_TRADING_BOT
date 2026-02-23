# model/evaluate_models.py

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from ibkr_trading_bot.features.feature_engineering import prepare_dataset_with_targets
from ibkr_trading_bot.utils.io_helpers import load_dataframe
from ibkr_trading_bot.utils.metrics import calculate_metrics


def _unwrap_model_bundle(bundle_or_model):
    if isinstance(bundle_or_model, dict) and "model" in bundle_or_model:
        return bundle_or_model["model"], bundle_or_model.get("features")
    return bundle_or_model, None


def _select_feature_matrix(dataset: pd.DataFrame, model, bundle_feats: list[str] | None) -> pd.DataFrame:
    if bundle_feats:
        missing = [f for f in bundle_feats if f not in dataset.columns]
        if missing:
            _ensure_fallback_features(dataset, missing)
            missing = [f for f in bundle_feats if f not in dataset.columns]
            if missing:
                raise ValueError(f"V datech chybí featury, které model vyžaduje: {missing}")
        return dataset[bundle_feats].fillna(0.0)

    if hasattr(model, "feature_names_in_") and getattr(model, "feature_names_in_", None) is not None:
        names = [str(c) for c in list(model.feature_names_in_)]
        missing = [f for f in names if f not in dataset.columns]
        if missing:
            raise ValueError(f"V datech chybí featury požadované modelem: {missing}")
        return dataset[names].fillna(0.0)

    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and hasattr(booster, "feature_names"):
            names = list(booster.feature_names)
            missing = [f for f in names if f not in dataset.columns]
            if missing:
                raise ValueError(f"V datech chybí featury požadované modelem: {missing}")
            return dataset[names].fillna(0.0)
    except Exception:
        pass

    blacklist = {"timestamp", "open", "high", "low", "close", "volume", "target", "y", "signal"}
    num_cols = [c for c in dataset.columns if c not in blacklist and pd.api.types.is_numeric_dtype(dataset[c])]
    return dataset[num_cols].fillna(0.0)


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = set(np.unique(y_true_arr).tolist()) | set(np.unique(y_pred_arr).tolist())
    is_multiclass = len(labels) > 2
    if is_multiclass:
        f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        precision = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        recall = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    else:
        f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
        precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
        recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    accuracy = float((y_true_arr == y_pred_arr).mean())
    return float(f1), float(precision), float(recall), accuracy


def _prepare_dataset_for_eval(df_raw: pd.DataFrame) -> pd.DataFrame:
    try:
        return prepare_dataset_with_targets(df_raw)
    except Exception:
        dataset = df_raw.copy()
        if "target" not in dataset.columns:
            if "close" not in dataset.columns:
                raise
            dataset["target"] = (dataset["close"].shift(-1) > dataset["close"]).astype(int)
            dataset = dataset.dropna(subset=["target"]).copy()
        return dataset


def profit_factor(signals: list[int], prices: list[float]) -> float:
    profits = []
    for i in range(1, len(signals)):
        if signals[i-1] != 0:
            pnl = (prices[i] - prices[i-1]) * signals[i-1]
            profits.append(pnl)
    profits = np.array(profits)
    gross_profit = profits[profits > 0].sum()
    gross_loss = -profits[profits < 0].sum()
    return gross_profit / gross_loss if gross_loss > 0 else np.inf

def _ensure_fallback_features(df: pd.DataFrame, required: list[str]) -> None:
    """
    Pokud dataset postrádá některé z fallback featur použitých při tréninku,
    dopočítá je in-place (vyžaduje sloupce: open, high, low, close).
    """
    need = set(required)
    fb = {"_ret1", "_hl_range", "_oc_change"}
    to_build = list(need & fb - set(df.columns))
    if not to_build:
        return
    missing_ohlc = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing_ohlc:
        # Nemáme z čeho dopočítat – necháme původní logiku vyhodit chybu výše.
        return
    # Výpočet fallback featur stejně jako v train_models._select_feature_columns
    if "_ret1" in to_build:
        df["_ret1"] = df["close"].pct_change().fillna(0.0)
    if "_hl_range" in to_build:
        df["_hl_range"] = (df["high"] - df["low"]).fillna(0.0)
    if "_oc_change" in to_build:
        df["_oc_change"] = (df["close"] - df["open"]).fillna(0.0)


def signal_stability(signals: list[int]) -> float:
    signals = np.array(signals)
    changes = np.diff(signals)
    reversals = np.sum(np.abs(changes) == 2)
    total_changes = np.sum(changes != 0)
    return 1 - (reversals / total_changes) if total_changes > 0 else 1.0

def evaluate_model(model_path: str, data_path: str) -> dict:
    bundle_or_model = joblib.load(model_path)
    model, bundle_feats = _unwrap_model_bundle(bundle_or_model)
    df = load_dataframe(data_path)
    dataset = _prepare_dataset_for_eval(df)

    X = _select_feature_matrix(dataset, model, bundle_feats)

    y_true = dataset["target"].astype(int)
    y_pred = model.predict(X)

    f1, precision, recall, _ = _classification_metrics(y_true, y_pred)

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        df=dataset,
        fee_per_trade=0.0,
        slippage_bps=0.0,
        rolling_window=200,
        annualize_sharpe=False
    )

    profit    = metrics.get("profit_net", metrics.get("profit_gross", metrics.get("profit", 0.0)))
    sharpe    = metrics.get("sharpe_trade_net", metrics.get("sharpe_trade_gross", metrics.get("sharpe_ratio", 0.0)))
    drawdown  = metrics.get("max_drawdown", metrics.get("max_drawdown_trade_net", metrics.get("max_drawdown_trade_gross", 0.0)))
    win_rate  = metrics.get("winrate_net", metrics.get("winrate", 0.0))
    pf        = metrics.get("profit_factor_net", metrics.get("profit_factor", 0.0))
    stability = metrics.get("signal_stability", 0.0)


    return {
        "model_path": model_path,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "profit": float(profit),
        "sharpe_ratio": float(sharpe),
        "drawdown": float(drawdown),
        "winrate": float(win_rate),
        "profit_factor": float(pf),
        "signal_stability": float(stability)
    }

# === Wrapper pro CLI: načte features, vyhodnotí model a zapíše 1 řádek do CSV ===
def evaluate_model_once(features_csv: str, model_path: str, results_out: str) -> str:
    """
    Kompatibilní s příkazem: python -m ibkr_trading_bot.main evaluate --model ... [--features ...] [--results-out ...]
    - Podporuje joblib bundle {"model": ..., "features": [...]} i přímo estimator (XGB/LGBM/RF).
    - Pokud jsou v bundle uvedeny názvy featur, použije je. Jinak vybere numerické sloupce (bez OHLC/target).
    - Doplňuje metriky (včetně obchodních), zapíše jeden řádek do results_out (append/nový).
    """
    import os


    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model neexistuje: {model_path}")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Soubor s featurami neexistuje: {features_csv}")

    # 1) načteme featury a připravíme dataset s targetem přes tvoji funkci
    df_raw = load_dataframe(features_csv)
    dataset = _prepare_dataset_for_eval(df_raw)

    # 2) vybereme X/y
    if "target" not in dataset.columns:
        raise ValueError("Ve vstupním datasetu chybí sloupec 'target' po prepare_dataset_with_targets().")
    y_true = dataset["target"].astype(int)
    # numerické featury bez zjevných sloupců
    blacklist = {"timestamp", "open", "high", "low", "close", "volume", "target", "y", "signal"}
    num_cols = [c for c in dataset.columns if c not in blacklist and pd.api.types.is_numeric_dtype(dataset[c])]

    # 3) načtení modelu (bundle i čistý estimator)
    bundle_or_model = joblib.load(model_path)
    model, model_feats = _unwrap_model_bundle(bundle_or_model)

    # 4) robustní výběr featur
    X = _select_feature_matrix(dataset, model, model_feats)

    # 5) predikce (vynecháme poslední řádek kvůli posunu targetu v prepare_dataset_with_targets, pokud je)
    # Pozn.: Pokud prepare_dataset_with_targets NEdělá shift(-1), není třeba ořezávat.
    # Využijeme konzistenci s tvou evaluate_model: zachováme celou délku a metriky se spočítají na X.shape[0].
    y_pred = getattr(model, "predict", None)
    if y_pred is None:
        raise TypeError("Načtený objekt modelu nemá metodu .predict()")
    y_hat = model.predict(X)

    # 6) metriky
    f1, precision, recall, accuracy = _classification_metrics(y_true, y_hat)

    # 7) obchodní metriky – použijeme jednotné calculate_metrics
    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_hat,
        df=dataset,
        fee_per_trade=0.0,
        slippage_bps=0.0,
        rolling_window=200,
        annualize_sharpe=False
    )

    profit    = metrics.get("profit_net", metrics.get("profit_gross", metrics.get("profit", 0.0)))
    sharpe    = metrics.get("sharpe_trade_net", metrics.get("sharpe_trade_gross", metrics.get("sharpe_ratio", 0.0)))
    drawdown  = metrics.get("max_drawdown", metrics.get("max_drawdown_trade_net", metrics.get("max_drawdown_trade_gross", 0.0)))
    win_rate  = metrics.get("winrate_net", metrics.get("winrate", 0.0))
    pf        = metrics.get("profit_factor_net", metrics.get("profit_factor", 0.0))
    stability = metrics.get("signal_stability", 0.0)
    num_trades = metrics.get("num_trades", 0)

    # 8) uložení results.csv (append/nový)
    out_dir = os.path.dirname(results_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    row = pd.DataFrame([{
        "model_path": model_path,
        "profit": float(profit),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "num_trades": int(num_trades),
        "sharpe_ratio": float(sharpe),
        "drawdown": float(drawdown),
        "winrate": float(win_rate),
        "profit_factor": float(pf),
        "signal_stability": float(stability),
    }])

    if os.path.exists(results_out):
        row.to_csv(results_out, mode="a", header=False, index=False)
    else:
        row.to_csv(results_out, index=False)

    return results_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-json", type=str)
    parser.add_argument("--output-csv", type=str)
    args = parser.parse_args()

    results = evaluate_model(args.model_path, args.data_path)

    print("=== Výsledky vyhodnocení modelu ===")
    for key, value in results.items():
        print(f"{key}: {value}")

    if args.output_json:
        json_dir = os.path.dirname(args.output_json)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Výsledky uloženy do {args.output_json}")

    if args.output_csv:
        df_out = pd.DataFrame([results])
        df_out.to_csv(args.output_csv, index=False)
        print(f"✅ Výsledky uloženy do {args.output_csv}")
