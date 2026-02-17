# scripts/check_model_pkl.py
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# --- volitelně vezmeme TradingView pro živá data ---
def fetch_tv(symbol, exchange, tf_label, limit):
    try:
        # drž se stejných import fallbacků jako v GUI podle toho, kde to spouštíš
        from ibkr_trading_bot.core.datasource.tradingview_client import TradingViewClient
    except Exception:
        try:
            from core.datasource.tradingview_client import TradingViewClient
        except Exception:
            return None
    tv = TradingViewClient(username=os.getenv("TV_USERNAME"), password=os.getenv("TV_PASSWORD"))
    df = tv.get_history(symbol, exchange, tf_label, limit=limit)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    df.rename(columns={"time":"date"}, inplace=True)
    return df[["date","open","high","low","close","volume"]]

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(9, min_periods=1).mean()
    df["ma_slow"] = df["close"].rolling(21, min_periods=1).mean()
    h_l  = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"]  - df["close"].shift(1)).abs()
    tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()
    df["average"] = (df["open"] + df["high"] + df["low"] + df["close"])/4.0
    return df

def align_to_model(X: pd.DataFrame, model, meta: dict|None):
    base_cols = ["close","ma_fast","ma_slow","atr","average"]
    use_cols = None
    if isinstance(meta, dict):
        mf = meta.get("expected_features") or meta.get("features")
        if isinstance(mf, (list, tuple)) and mf:
            use_cols = [str(c) for c in mf]
    if use_cols is None and hasattr(model, "feature_names_in_") and getattr(model, "feature_names_in_", None) is not None:
        use_cols = [str(c) for c in list(model.feature_names_in_)]
    if use_cols is None:
        use_cols = base_cols

    X = X.copy()
    for c in use_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[use_cols].apply(pd.to_numeric, errors="coerce")
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0).astype("float32")
    return X, use_cols

def load_model(pkl_path):
    obj = joblib.load(pkl_path)
    meta = {}
    model = obj
    if isinstance(obj, dict):
        model = obj.get("predictor") or obj.get("model") or obj.get("estimator")
        meta = obj.get("metadata") or obj.get("meta") or {}
        if model is None:
            raise TypeError("Dict neobsahuje 'predictor'/'model'/'estimator'.")
    return model, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--symbol", default="GC1!")
    ap.add_argument("--exchange", default="COMEX")
    ap.add_argument("--tf", default="15 min")
    ap.add_argument("--bars", type=int, default=200)
    ap.add_argument("--csv", help="volitelně test na CSV se sloupci: date,open,high,low,close,volume")
    args = ap.parse_args()

    model, meta = load_model(args.pkl)
    print("== MODEL INFO ==")
    print("type:", type(model).__name__)
    print("feature_names_in_:", getattr(model, "feature_names_in_", None))
    print("n_features_in_   :", getattr(model, "n_features_in_", None))
    print("expected_features:", (meta or {}).get("expected_features"))
    print("class_to_dir     :", (meta or {}).get("class_to_dir"))

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["date"])
    else:
        df = fetch_tv(args.symbol, args.exchange, args.tf.replace("mins","min"), args.bars)
        if df is None:
            raise SystemExit("Nepodařilo se načíst data (TradingView ani CSV).")

    feats = compute_indicators(df)
    X, used_cols = align_to_model(feats, model, meta)
    print("\n== FEAT INFO ==")
    print("used_cols:", used_cols)
    print("X.shape  :", X.shape)

    # Predikce
    y = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            pass

    unique = np.unique(np.asarray(y).ravel())
    print("\n== PRED ==")
    print("unique classes in window:", unique)
    if proba is not None:
        conf = np.asarray(proba, dtype=float)
        maxc = conf.max(axis=1) if conf.ndim == 2 else conf
        print("mean(conf):", float(np.mean(maxc)))
        print("min/max(conf):", float(np.min(maxc)), float(np.max(maxc)))

    # náhled posledních 5
    tail = min(5, len(X))
    print("\nlast predictions:", np.asarray(y).ravel()[-tail:])
    if proba is not None:
        print("last conf max   :", (np.asarray(proba)[-tail:]).max(axis=1) if np.asarray(proba).ndim==2 else np.asarray(proba)[-tail:])

if __name__ == "__main__":
    main()

#C:\Users\adamk\Můj disk\Trader\ibkr_trading_bot\model_outputs
#python tests/test_model_pkl.py --pkl "C:\Users\adamk\Můj disk\Trader\ibkr_trading_bot\model_outputs\rf_20251003_132426.pkl" --symbol GC1! --exchange COMEX --tf "15 min" --bars 200
