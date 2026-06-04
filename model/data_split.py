# model/data_split.py
# 🟦 Modul pro walk-forward validaci a export datasetů

import os
import logging
from pathlib import Path
import json

import pandas as pd


LOGGER = logging.getLogger(__name__)


def walk_forward_split(df, window_size, test_size, step_size, expanding=False):
    """
    Rozdělí data na postupné trénovací a testovací sady (walk-forward).

    :param df: vstupní DataFrame s featurami a cílovou proměnnou
    :param window_size: délka trénovacího okna
    :param test_size: délka validačního okna
    :param step_size: krok posunu okna
    :param expanding: True = expanding window, False = rolling window
    :return: seznam dvojic (train_df, test_df)
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if test_size <= 0:
        raise ValueError("test_size must be > 0")
    if step_size <= 0:
        raise ValueError("step_size must be > 0")

    splits = []
    start = 0
    while start + window_size + test_size <= len(df):
        train_end = start + window_size
        test_end = train_end + test_size

        train_df = df.iloc[:train_end] if expanding else df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        splits.append((train_df.copy(), test_df.copy()))
        start += step_size
    return splits



def export_datasets(pairs, output_dir, prefix="fold", format="csv"):
    """
    Uloží jednotlivé dvojice (train, test) datasetů do souborů.

    :param pairs: seznam dvojic (train_df, test_df)
    :param output_dir: cílová složka pro export
    :param prefix: název souboru (prefix_foldX_train/test)
    :param format: csv nebo pkl
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, (train_df, test_df) in enumerate(pairs):
        if format == "csv":
            train_df.to_csv(os.path.join(output_dir, f"{prefix}_{i}_train.csv"), index=False)
            test_df.to_csv(os.path.join(output_dir, f"{prefix}_{i}_test.csv"), index=False)
        elif format == "pkl":
            train_df.to_pickle(os.path.join(output_dir, f"{prefix}_{i}_train.pkl"))
            test_df.to_pickle(os.path.join(output_dir, f"{prefix}_{i}_test.pkl"))
        else:
            raise ValueError("Nepodporovaný formát exportu")

# ibkr_trading_bot/model/train_models.py
#
# Popisky:
# - Robustní trénování s GridSearchCV pro XGB/LGBM/RF s ohledem na nevyvážená data.
# - Sanitizace featur (konstantní/duplicitní), širší gridy, a optimalizace prahu rozhodnutí.
# - Základní (záchranné) časové featury, pokud dataset neobsahuje nic jiného.
# - Ukládáme i 'decision_threshold' pro jednotné rozhodování v dalších částech projektu.

import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from ibkr_trading_bot.core.services.model_service import runtime_python_version, runtime_sklearn_version
from ibkr_trading_bot.features.augmentations import add_noise, mix_dataframes, roll_shift
from ibkr_trading_bot.features.feature_engineering import prepare_dataset_with_targets
from ibkr_trading_bot.utils.io_helpers import load_dataframe


# --- Funkce pro generování syntetických dat ---
def generate_synthetic_data(df: pd.DataFrame, n_samples: int = 10, noise_level: float = 0.01) -> pd.DataFrame:
    """
    Vygeneruje n_samples syntetických variant datasetu pomocí:
    - přidání šumu
    - náhodného posunu (rolling)
    - kombinace dvou datasetů (mixování)
    Výstupem je dataframe připravený pro trénování (prepare_dataset_with_targets).
    """
    synthetic = []
    for _ in range(n_samples):
        noisy = add_noise(df, noise_level=noise_level)
        shifted = roll_shift(noisy)
        synthetic.append(shifted)

    # Mixování sousedních vzorků (zvýší diverzitu)
    for i in range(len(synthetic) - 1):
        mixed = mix_dataframes(synthetic[i], synthetic[i + 1], alpha=0.5)
        synthetic.append(mixed)

    result = pd.concat(synthetic).dropna()
    return prepare_dataset_with_targets(result)


def train_and_evaluate_model(X, y, model_name: str, param_grid: dict, window: str = None):
    """
    Trénuje model s grid search a ukládá nejlepší model + výsledky.

    Args:
        X (pd.DataFrame): Vstupní featury
        y (pd.Series): Cílová proměnná (binární 0/1)
        model_name (str): 'xgb' | 'lgbm' | 'rf'
        param_grid (dict): Grid parametrů pro daný model
        window (str, optional): Označení okna pro rolling retrain

    Returns:
        best_model: Natrénovaný model s nejlepšími parametry
        best_score: F1 skóre z cross-validace nejlepší konfigurace
    """
    # --- Sanitizace a kontrola cíle + info o nevyváženosti ---
    y_tmp = pd.Series(y).astype(int)
    if y_tmp.nunique() < 2:
        raise ValueError(
            "Stratifikovaná CV vyžaduje alespoň 2 třídy v 'y'. "
            "Zkontroluj přípravu targetu nebo zvol jiné období/parametry."
        )
    min_class = y_tmp.value_counts().min() if not y_tmp.empty else 0
    n_splits = 3 if min_class >= 3 else 2  # fallback na 2, když je málo vzorků
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # scale_pos_weight pro XGBoost (vyvážení tříd)
    pos = int((y_tmp == 1).sum())
    neg = int((y_tmp == 0).sum())
    spw = (neg / max(pos, 1)) if (pos > 0 and neg > 0) else 1.0
    LOGGER.info("Classes in y: neg=%s, pos=%s | scale_pos_weight=%.3f", neg, pos, spw)

    # --- Výběr modelu s robustními defaulty ---
    if model_name == "xgb":
        model = XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "lgbm":
        model = LGBMClassifier(
            objective="binary",
            num_leaves=31,
            min_data_in_leaf=5,       # menší listy pomůžou u malých oken
            learning_rate=0.1,
            n_estimators=200,
            class_weight="balanced",  # když je target nevyvážený
            force_col_wise=True,      # stabilnější/rychlejší na menších tabulkách
            verbosity=-1,
            n_jobs=-1,
            random_state=42,
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    else:
        raise ValueError(f"Neznámý model: {model_name}")

    # --- Fallback param_grid, pokud přijde prázdný (např. z GUI) ---
    if not param_grid:
        if model_name == "lgbm":
            param_grid = {
                "num_leaves": [15, 31, 63, 127],
                "min_data_in_leaf": [5, 10, 20, 40],
                "n_estimators": [200, 400, 700],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [-1, 6, 10],
            }
        elif model_name == "xgb":
            param_grid = {
                "n_estimators": [200, 400, 700],
                "max_depth": [3, 5, 8],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
            }
        elif model_name == "rf":
            param_grid = {
                "n_estimators": [300, 600, 1000],
                "max_depth": [None, 10, 16, 24],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2", None],
            }

    # --- Sanitizace X a odhoz nepoužitelných featur ---
    X = pd.DataFrame(X).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    # 1) drop konstantních sloupců
    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        LOGGER.warning(
            "Removing %s constant features: %s%s",
            len(const_cols),
            const_cols[:10],
            "..." if len(const_cols) > 10 else "",
        )
        X = X.drop(columns=const_cols)

    # 2) drop duplicitních sloupců (best-effort)
    try:
        unique_cols = X.T.drop_duplicates().T.columns
        if len(unique_cols) < X.shape[1]:
            removed = [c for c in X.columns if c not in unique_cols]
            LOGGER.warning(
                "Removing duplicate features: %s%s",
                removed[:10],
                "..." if len(removed) > 10 else "",
            )
            X = X.loc[:, unique_cols]
    except Exception:
        pass

    if X.shape[1] == 0:
        raise ValueError("Po očištění nezbyly žádné featury. Zkontroluj feature engineering / konfiguraci.")

    # --- Grid Search ---
    grid = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        error_score="raise",
    )
    grid.fit(X, y_tmp)


    best_model = grid.best_estimator_
    best_score = grid.best_score_

    # --- Optimalizace prahu rozhodnutí na základě F1 ---
    try:
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(X)[:, 1]
        elif hasattr(best_model, "decision_function"):
            import numpy as np
            raw = best_model.decision_function(X)
            raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            proba = raw
        else:
            proba = None

        best_threshold = 0.5
        best_f1 = -1.0
        if proba is not None:
            import numpy as np
            for t in np.linspace(0.05, 0.95, 37):
                tp = ((proba >= t) & (y_tmp == 1)).sum()
                fp = ((proba >= t) & (y_tmp == 0)).sum()
                fn = ((proba <  t) & (y_tmp == 1)).sum()
                precision = tp / max(tp + fp, 1)
                recall    = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(t)
        else:
            best_threshold = 0.5
            best_f1 = best_score

        LOGGER.info("Optimized decision threshold: %.3f | F1@threshold~=%.4f", best_threshold, best_f1)
    except Exception as e:
        LOGGER.warning("Decision threshold optimization failed: %s", e)
        best_threshold = 0.5

    # --- Uložení modelu ---
    suffix = f"_window{window}" if window else ""
    output_path = f"model_outputs/{model_name}{suffix}.pkl"
    os.makedirs("model_outputs", exist_ok=True)
    payload = {
        "model": best_model,
        "features": list(X.columns),
        "best_params": grid.best_params_,
        "decision_threshold": best_threshold,  # uložíme prah
    }
    joblib.dump(payload, output_path)

    LOGGER.info("Model %s saved to %s | F1 (CV): %.4f", model_name, output_path, best_score)

    return best_model, best_score


def _select_feature_columns(df: pd.DataFrame) -> list:
    """
    Vybere numerické featury pro trénink.
    1) Primárně všechny numerické sloupce mimo timestamp/target/y/signal a mimo syrové OHLC/volume.
    2) Pokud nic nezbyde, vytvoří minimální „časové“ featury:
       - _ret1/_ret3/_ret5: 1/3/5-kroková návratnost
       - _hl_range: high-low rozpětí
       - _oc_change: close-open změna (absolutní)
       - _vol_10: rolling volatilita (std z návratností)
    """
    hard_blacklist = {"timestamp", "target", "y", "signal"}
    ohlc = {"open", "high", "low", "close", "volume"}

    candidates = [c for c in df.columns if c not in hard_blacklist and pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in candidates if c not in ohlc]

    if not feat_cols:
        if all(c in df.columns for c in ("close", "open", "high", "low")):
            df["_ret1"] = df["close"].pct_change().fillna(0.0)
            df["_ret3"] = df["close"].pct_change(3).fillna(0.0)
            df["_ret5"] = df["close"].pct_change(5).fillna(0.0)
            df["_hl_range"] = (df["high"] - df["low"]).fillna(0.0)
            df["_oc_change"] = (df["close"] - df["open"]).fillna(0.0)
            df["_vol_10"] = df["close"].pct_change().rolling(10).std().fillna(0.0)
            feat_cols = ["_ret1", "_ret3", "_ret5", "_hl_range", "_oc_change", "_vol_10"]
        else:
            if not candidates:
                raise ValueError("Nenalezeny žádné numerické sloupce pro trénink.")
            feat_cols = candidates

    try:
        target_col = "target"
        if target_col in df.columns and feat_cols:
            corr = df[feat_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            leaky = corr[corr >= 0.999].index.tolist()
            if leaky:
                LOGGER.warning(
                    "Removing potentially leaky features (|corr|>=0.999 with target): %s%s",
                    leaky[:10],
                    "..." if len(leaky) > 10 else "",
                )
                feat_cols = [c for c in feat_cols if c not in leaky]
                if not feat_cols:
                    raise ValueError("Po anti-leak filtru nezbyly žádné featury. Zkontroluj feature engineering.")
    except Exception:
        # Na selhání korelace nereagujeme tvrdě; ponecháme původní feat_cols
        pass

    return feat_cols


def train_simple_model(features_csv: str, model_out: str) -> str:
    """
    Jednoduchý „plug-and-play“ trénink volaný z CLI:
      - načte featury z CSV,
      - připraví target (přes prepare_dataset_with_targets),
      - natrénuje rychlý RF,
      - uloží model + seznam featur do joblib/pkl.
    Vrací cestu k uloženému modelu.
    """
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Soubor s featurami neexistuje: {features_csv}")

    raw = pd.read_csv(features_csv)
    try:
        dataset = prepare_dataset_with_targets(raw)
    except Exception:
        dataset = raw.copy()
        if "target" not in dataset.columns:
            if "close" not in dataset.columns:
                raise
            dataset["target"] = (dataset["close"].shift(-1) > dataset["close"]).astype(int)
            dataset = dataset.dropna(subset=["target"]).copy()

    if "target" not in dataset.columns:
        raise ValueError("Ve vstupním datasetu chybí sloupec 'target' po prepare_dataset_with_targets().")
    y = dataset["target"].astype(int)
    X_cols = _select_feature_columns(dataset)
    X = dataset[X_cols].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    # Odhoď konstantní featury i tady (pro jistotu)
    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=1)
    model.fit(X, y)

    out_dir = os.path.dirname(model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {"model": model, "features": list(X.columns), "decision_threshold": 0.5}
    joblib.dump(payload, model_out)

    model_path = Path(model_out)
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    meta = {
        "created_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "created_at_iso": pd.Timestamp.utcnow().isoformat(),
        "trained_features": list(X.columns),
        "model_classes": [str(c) for c in list(getattr(model, "classes_", []))],
        "decision_threshold": 0.5,
        "metrics": {},
        "sklearn_version": runtime_sklearn_version(),
        "python_version": runtime_python_version(),
        "schema_version": 1,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Model saved: %s | n_features=%s", model_out, X.shape[1])
    return model_out


# 🟩 Volitelné spuštění walk-forward splitu přes CLI
if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--split-data", action="store_true", help="Provede walk-forward split a export")
    parser.add_argument("--train-model", type=str, help="Spustí trénování modelu (xgb, lgbm, rf)")
    args = parser.parse_args()

    if args.split_data:
        from ibkr_trading_bot.model.data_split import export_datasets, walk_forward_split

        here = os.path.dirname(os.path.abspath(__file__))             # .../ibkr_trading_bot/model
        cfg_path = os.path.normpath(os.path.join(here, "..", "config", "default_config.yaml"))
        with open(cfg_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        split_cfg = config.get("data_split", {})

        window_size = split_cfg.get("window_size", 500)
        test_size = split_cfg.get("test_size", 100)
        step_size = split_cfg.get("step_size", 50)
        expanding = split_cfg.get("expanding", False)
        export_format = split_cfg.get("export_format", "csv")
        output_dir = split_cfg.get("output_dir", "data/control/")

        df = load_dataframe("data/processed/features.csv")
        splits = walk_forward_split(df, window_size, test_size, step_size, expanding)
        export_datasets(splits, output_dir=output_dir, format=export_format)

        LOGGER.info("Walk-forward split completed. Exported %s sets to: %s", len(splits), output_dir)

    if args.train_model:
        df = load_dataframe("data/processed/features.csv")
        dataset = prepare_dataset_with_targets(df)

        X_cols = _select_feature_columns(dataset)
        X = dataset[X_cols].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        y = dataset["target"].astype(int)

        if args.train_model == "xgb":
            param_grid = {
                "n_estimators": [200, 400, 700],
                "max_depth": [3, 5, 8],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
            }
        elif args.train_model == "lgbm":
            param_grid = {
                "n_estimators": [200, 400, 700],
                "num_leaves": [31, 63, 127],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [-1, 6, 10],
                "min_data_in_leaf": [5, 10, 20, 40],
            }
        elif args.train_model == "rf":
            param_grid = {
                "n_estimators": [300, 600, 1000],
                "max_depth": [None, 10, 16, 24],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2", None],
            }
        else:
            raise ValueError(f"Model '{args.train_model}' není podporován.")

        train_and_evaluate_model(X, y, args.train_model, param_grid)


def split_by_calendar_days(
    df,
    train_days: int,
    test_days: int,
    timestamp_col: str = "timestamp",
    target_col: str = "target",
    feature_blacklist=None,
):
    """
    Deterministický split podle kalendářních dnů (od konce datasetu).
    - test = posledních `test_days` dní,
    - train = dnů bezprostředně před testem v počtu `train_days`.
    Vrací: X_train, y_train, X_test, y_test (sanitizované, numerické).
    """
    import pandas as pd

    if feature_blacklist is None:
        feature_blacklist = ["timestamp", "signal", "y"]
    if target_col not in feature_blacklist:
        feature_blacklist = list(set(feature_blacklist + [target_col]))

    if timestamp_col not in df.columns:
        raise ValueError(f"Ve vstupním DF chybí sloupec '{timestamp_col}'.")
    if target_col not in df.columns:
        raise ValueError(f"Ve vstupním DF chybí sloupec '{target_col}'.")

    dfx = df.copy()
    dfx[timestamp_col] = pd.to_datetime(dfx[timestamp_col])
    dfx = dfx.sort_values(timestamp_col).reset_index(drop=True)

    # seznam unikátních kalendářních dnů v pořadí
    dfx["_date"] = dfx[timestamp_col].dt.date
    unique_days = dfx["_date"].drop_duplicates().tolist()

    need = train_days + test_days
    if len(unique_days) < need:
        raise ValueError(f"Nedostatek dnů v datasetu: k dispozici {len(unique_days)}, potřeba {need}.")

    test_days_list = unique_days[-test_days:]
    train_days_list = unique_days[-(test_days + train_days):-test_days]

    df_train = dfx[dfx["_date"].isin(train_days_list)].copy()
    df_test  = dfx[dfx["_date"].isin(test_days_list)].copy()

    LOGGER.info("Day-based split selected")
    LOGGER.info("Train [%s]: %s -> %s", len(train_days_list), train_days_list[0], train_days_list[-1])
    LOGGER.info("Test [%s]: %s -> %s", len(test_days_list), test_days_list[0], test_days_list[-1])
    LOGGER.info("Row counts: train=%s, test=%s", len(df_train), len(df_test))

    # výběr numerických featur mimo blacklist (a preferenčně mimo syrové OHLC)
    hard_blacklist = set(feature_blacklist + ["_date"])
    ohlc = {"open", "high", "low", "close", "volume"}

    candidates = [c for c in df_train.columns
                  if c not in hard_blacklist and pd.api.types.is_numeric_dtype(df_train[c])]
    feat_cols = [c for c in candidates if c not in ohlc] or candidates
    if not feat_cols:
        raise ValueError("Nenalezeny použitelné featury po aplikaci blacklistu.")

    def sanitize(m: pd.DataFrame) -> pd.DataFrame:
        m = m[feat_cols].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        const_cols = [c for c in m.columns if m[c].nunique(dropna=False) <= 1]
        if const_cols:
            m = m.drop(columns=const_cols)
        return m

    X_train = sanitize(df_train)
    X_test  = sanitize(df_test)
    y_train = df_train[target_col].astype(int)
    y_test  = df_test[target_col].astype(int)

    if X_train.shape[1] == 0 or X_test.shape[1] == 0:
        raise ValueError("Po očištění nezbyly žádné featury pro train/test.")

    return X_train, y_train, X_test, y_test
