import json
import os
import sys
from pathlib import Path
import traceback

import pandas as pd

# Ensure package root import works from scripts/
script_dir = os.path.dirname(os.path.abspath(__file__))
pkg_parent = os.path.dirname(script_dir)
repo_root = os.path.dirname(pkg_parent)
sys.path.insert(0, repo_root)

from ibkr_trading_bot.model.train_models import train_and_evaluate_model


def main() -> int:
    csv_path = Path("data/processed/features_with_labels.csv")
    if not csv_path.exists():
        print(f"ERROR: missing input {csv_path}")
        return 2

    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

    df = df.tail(1600).copy()

    print(f"SMOKE_INPUT_ROWS={len(df)}")
    print(f"SMOKE_TARGET_UNIQ={sorted(df['target'].dropna().astype(int).unique().tolist()) if 'target' in df.columns else 'MISSING'}")

    res = train_and_evaluate_model(
        df=df,
        estimator_name="hgbt",
        param_grid={
            "clf__max_depth": [4],
            "clf__learning_rate": [0.03],
            "clf__max_iter": [120],
            "clf__l2_regularization": [0.1],
        },
        n_splits=3,
        embargo=5,
        holdout_bars=120,
        fee_per_trade=0.0,
        slippage_bps=0.0,
        mc_enabled=False,
        annualize_sharpe=True,
    )

    output_path = Path(res["output_path"])
    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hold = meta.get("metrics_holdout") or {}

    print(f"OUTPUT_PATH={output_path.as_posix()}")
    print(f"META_PATH={meta_path.as_posix()}")
    print(f"CLASSES={meta.get('classes')}")
    print(f"CLASS_TO_DIR={meta.get('class_to_dir')}")
    print(f"HOLDOUT_NUM_TRADES={hold.get('num_trades')}")
    print(f"HOLDOUT_NUM_TRADES_SHORT={hold.get('num_trades_short')}")
    print(f"HOLDOUT_NUM_TRADES_LONG={hold.get('num_trades_long')}")
    print(f"HOLDOUT_PER_CLASS_3_KEYS={list((hold.get('per_class_3') or {}).keys())}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("SMOKE_FAILED")
        traceback.print_exc()
        raise
