"""Runner: počítá featury a triple-barrier labely a ukládá CSV pro trénink.

Usage:
  python scripts/run_triple_barrier_labeling.py --input data/raw/GOLD_5m_...csv --output data/processed/features_with_labels.csv
"""
from datetime import datetime
import argparse
import os
from pathlib import Path

import pandas as pd

import sys
import os
# Ensure package importable when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ibkr_trading_bot.features.feature_engineering import compute_all_features
from ibkr_trading_bot.labels.triple_barrier import make_triple_barrier_labels_ternary


def main(input_path: str, output_path: str | None = None, horizon: int = 12, tp: float = 60.0, sl: float = 40.0):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    # normalize OHLC and index handling is done inside compute_all_features via timestamp UTC
    df = df.rename(columns={"date": "timestamp"}) if "date" in df.columns else df

    # compute features (expects df indexed by timestamp)
    df_features = compute_all_features(df)

    # compute ternary labels
    labels = make_triple_barrier_labels_ternary(df_features, horizon=horizon, take_profit_bps=tp, stop_loss_bps=sl, price_col="close")

    df_out = df_features.reset_index()
    df_out["target"] = labels.values

    if output_path is None:
        out_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"features_with_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(output_path, index=False)
    print(f"✅ Saved features+labels to: {output_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV with raw OHLC data")
    p.add_argument("--output", default=None, help="Where to save features+labels")
    p.add_argument("--horizon", type=int, default=12, help="Horizon in bars for triple barrier")
    p.add_argument("--tp", type=float, default=60.0, help="Take profit in bps")
    p.add_argument("--sl", type=float, default=40.0, help="Stop loss in bps")
    args = p.parse_args()

    main(args.input, args.output, horizon=args.horizon, tp=args.tp, sl=args.sl)
