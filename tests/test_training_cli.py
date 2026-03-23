# tests/test_training_cli.py
import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import sklearn

from ibkr_trading_bot.data.generate_synthetic import generate_synthetic_data
from ibkr_trading_bot.features.feature_engineering import compute_all_features
from ibkr_trading_bot.model.evaluate_models import evaluate_model_once
from ibkr_trading_bot.model.train_models import train_simple_model


def test_train_and_evaluate_smoke():
    tmp_path = Path(".codex_test_tmp") / f"training_cli_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    # 1) syntetická data
    df = generate_synthetic_data(n_samples=500, noise_level=0.05)
    # 2) featury
    feats = compute_all_features(df)
    features_csv = tmp_path / "features.csv"
    feats.to_csv(features_csv, index=False)

    # 3) train
    model_out = tmp_path / "model.joblib"
    path_model = train_simple_model(features_csv=str(features_csv), model_out=str(model_out))
    assert os.path.exists(path_model)
    meta_path = model_out.with_name(model_out.stem + "_meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["sklearn_version"] == sklearn.__version__
    assert meta["python_version"] == sys.version.split()[0]

    # 4) evaluate
    results_csv = tmp_path / "results.csv"
    path_results = evaluate_model_once(features_csv=str(features_csv), model_path=str(model_out), results_out=str(results_csv))
    assert os.path.exists(path_results)

    # 5) results sanity
    res = pd.read_csv(results_csv)
    assert not res.empty
    for col in ["model_path", "profit", "f1", "accuracy", "num_trades"]:
        assert col in res.columns
