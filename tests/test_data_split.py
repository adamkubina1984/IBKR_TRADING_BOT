# tests/test_data_split.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
import pandas as pd
import pytest

from model.data_split import export_datasets, walk_forward_split


@pytest.fixture
def test_df():
    """Vytvoří jednoduchý testovací DataFrame se 1000 řádky."""
    n = 1000
    df = pd.DataFrame({
        "open": np.random.uniform(1800, 1900, n),
        "high": np.random.uniform(1900, 2000, n),
        "low": np.random.uniform(1800, 1900, n),
        "close": np.random.uniform(1850, 1950, n),
        "volume": np.random.randint(100, 1000, n),
    })
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()

def test_walk_forward_split_basic(test_df):
    """Ověří základní funkci walk-forward splitu."""
    window_size = 300
    test_size = 100
    step_size = 200
    expanding = False

    splits = walk_forward_split(test_df, window_size, test_size, step_size, expanding)

    # Očekávaný počet splitů (zhruba)
    expected_count = (len(test_df) - window_size - test_size) // step_size + 1
    assert len(splits) == expected_count

    for train_df, test_df in splits:
        assert len(train_df) == window_size
        assert len(test_df) == test_size

def test_walk_forward_split_expanding(test_df):
    """Ověří expanding mód - délka trénovacích dat musí růst."""
    window_size = 200
    test_size = 50
    step_size = 100
    expanding = True

    splits = walk_forward_split(test_df, window_size, test_size, step_size, expanding)

    previous_train_len = 0
    for train_df, test_df in splits:
        assert len(test_df) == test_size
        assert len(train_df) >= previous_train_len
        previous_train_len = len(train_df)

def test_export_datasets(tmp_path, test_df):
    """Ověří, že exportované soubory odpovídají původním datům."""
    window_size = 100
    test_size = 20
    step_size = 50
    splits = walk_forward_split(test_df, window_size, test_size, step_size)

    export_dir = tmp_path / "exported"
    export_datasets(splits, output_dir=str(export_dir), prefix="test", format="csv")

    # Zkontroluj, že všechny soubory existují a obsah se shoduje
    for i, (train_df, test_df) in enumerate(splits):
        train_path = export_dir / f"test_{i}_train.csv"
        test_path = export_dir / f"test_{i}_test.csv"

        assert train_path.exists()
        assert test_path.exists()

        train_loaded = pd.read_csv(train_path)
        test_loaded = pd.read_csv(test_path)

        train_loaded = pd.read_csv(train_path)
        test_loaded = pd.read_csv(test_path)

        pd.testing.assert_frame_equal(
            train_df.reset_index(drop=True),
            train_loaded,
            check_dtype=False
        )
        pd.testing.assert_frame_equal(
            test_df.reset_index(drop=True),
            test_loaded,
            check_dtype=False
        )

