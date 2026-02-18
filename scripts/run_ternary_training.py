"""Runner: trénink modelu s ternárním targetem (-1, 0, 1).

Usage:
  python scripts/run_ternary_training.py --input data/processed/features_with_labels.csv --model hgbt
"""
import sys
import os

# Ensure we can import ibkr_trading_bot from any location
script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
pkg_parent = os.path.dirname(script_dir)                # ibkr_trading_bot/
repo_root = os.path.dirname(pkg_parent)                 # repo root (Trader/)
sys.path.insert(0, repo_root)

import argparse
from datetime import datetime
import pandas as pd

from ibkr_trading_bot.model.train_models import train_and_evaluate_model


def main(input_path: str, estimator_name: str = "hgbt", holdout_bars: int = 500):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"[INFO] Loading features+labels from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        if "date" in df.columns:
            df.rename(columns={"date": "timestamp"}, inplace=True)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "target"])
    
    print(f"[INFO] Loaded {len(df)} rows")
    print(f"[INFO] Target value counts:\n{df['target'].value_counts()}")
    print(f"[INFO] Unique targets: {sorted(df['target'].unique())}")
    
    # Train
    print(f"\n[INFO] Starting training with {estimator_name} estimator...")
    result = train_and_evaluate_model(
        df=df,
        estimator_name=estimator_name,
        param_grid=None,  # Use default grid
        n_splits=5,
        embargo=10,
        fee_per_trade=0.0,
        slippage_bps=0.0,
        calibrate=False,
        holdout_bars=holdout_bars,
        mc_enabled=True,
        annualize_sharpe=True,
    )
    
    print(f"\n[✓] Training complete!")
    print(f"  Model saved to: {result['output_path']}")
    print(f"  Best score: {result['best_score']:.6f}")
    print(f"  N features: {result['n_features']}")
    print(f"  Decision threshold: {result['decision_threshold']:.3f}")
    print(f"  N train bars: {result['n_train_bars']}")
    print(f"  N holdout bars: {result['n_holdout_bars']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/features_with_labels.csv", help="CSV with features+targets")
    p.add_argument("--model", default="hgbt", help="Estimator: hgbt, rf, et, xgb, svm")
    p.add_argument("--holdout", type=int, default=500, help="Holdout bars for final test")
    args = p.parse_args()

    main(args.input, estimator_name=args.model, holdout_bars=args.holdout)
