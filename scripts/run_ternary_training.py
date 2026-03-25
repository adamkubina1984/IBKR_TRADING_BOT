"""Runner: trening modelu s ternarnim targetem (-1, 0, 1).

Usage:
  python scripts/run_ternary_training.py --input data/processed/features_with_labels.csv --model hgbt
"""

import argparse
import os
import sys

import pandas as pd

# Ensure we can import ibkr_trading_bot from any location
script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
pkg_parent = os.path.dirname(script_dir)  # ibkr_trading_bot/
repo_root = os.path.dirname(pkg_parent)  # repo root (Trader/)
sys.path.insert(0, repo_root)

from ibkr_trading_bot.model.train_models import train_and_evaluate_model  # noqa: E402


def main(
    input_path: str,
    estimator_name: str = "hgbt",
    holdout_bars: int = 500,
    label_lookahead_bars: int = 12,
    search_backend: str = "grid",
    optuna_trials: int | None = None,
    optuna_timeout_seconds: int | None = None,
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[INFO] Loading features+labels from: {input_path}")
    df = pd.read_csv(input_path)

    if "timestamp" not in df.columns and "date" in df.columns:
        df.rename(columns={"date": "timestamp"}, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "target"])

    print(f"[INFO] Loaded {len(df)} rows")
    print(f"[INFO] Target value counts:\n{df['target'].value_counts()}")
    print(f"[INFO] Unique targets: {sorted(df['target'].unique())}")
    print(
        "[INFO] Search config: "
        f"backend={search_backend} "
        f"optuna_trials={optuna_trials} "
        f"optuna_timeout={optuna_timeout_seconds}"
    )

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
        label_lookahead_bars=label_lookahead_bars,
        mc_enabled=True,
        annualize_sharpe=True,
        search_backend=search_backend,
        optuna_trials=optuna_trials,
        optuna_timeout_seconds=optuna_timeout_seconds,
    )

    search_plan = result.get("search_plan") or {}
    if isinstance(search_plan, dict):
        print(
            "[INFO] Search plan: "
            f"requested={search_plan.get('search_backend_requested')} "
            f"used={search_plan.get('search_backend_used')} "
            f"fallback={search_plan.get('search_backend_fallback_reason')} "
            f"optuna_done={search_plan.get('optuna_completed_trials')} "
            f"optuna_pruned={search_plan.get('optuna_pruned_trials')}"
        )

    print("\n[OK] Training complete!")
    print(f"  Model saved to: {result['output_path']}")
    print(f"  Best score: {result['best_score']:.6f}")
    print(f"  N features: {result['n_features']}")
    print(f"  Decision threshold: {result['decision_threshold']:.3f}")
    print(f"  N train bars: {result['n_train_bars']}")
    print(f"  N holdout bars: {result['n_holdout_bars']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/features_with_labels.csv", help="CSV with features+targets")
    p.add_argument("--model", default="hgbt", help="Estimator: hgbt, rf, et, xgb, lgb, svm")
    p.add_argument("--holdout", type=int, default=500, help="Holdout bars for final test")
    p.add_argument("--label-lookahead", type=int, default=12, help="Forward label lookahead in bars (anti-leak guard)")
    p.add_argument("--search-backend", default="grid", choices=["grid", "optuna"], help="Hyperparameter search backend")
    p.add_argument("--optuna-trials", type=int, default=None, help="Optuna trial budget")
    p.add_argument("--optuna-timeout", type=int, default=None, help="Optuna timeout in seconds")
    args = p.parse_args()

    main(
        args.input,
        estimator_name=args.model,
        holdout_bars=args.holdout,
        label_lookahead_bars=args.label_lookahead,
        search_backend=args.search_backend,
        optuna_trials=args.optuna_trials,
        optuna_timeout_seconds=args.optuna_timeout,
    )
