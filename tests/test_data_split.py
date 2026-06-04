# tests/test_data_split.py

import numpy as np
import pandas as pd
import pytest

from ibkr_trading_bot.model.data_split import export_datasets, walk_forward_split


@pytest.fixture
def test_df():
    """Vytvoří jednoduchý testovací DataFrame se 1000 řádky."""
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "open": rng.uniform(1800, 1900, n),
        "high": rng.uniform(1900, 2000, n),
        "low": rng.uniform(1800, 1900, n),
        "close": rng.uniform(1850, 1950, n),
        "volume": rng.integers(100, 1000, n),
    })
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()


@pytest.mark.parametrize(
    ("window_size", "test_size", "step_size", "message"),
    [
        (0, 20, 10, "window_size must be > 0"),
        (-1, 20, 10, "window_size must be > 0"),
        (100, 0, 10, "test_size must be > 0"),
        (100, -1, 10, "test_size must be > 0"),
        (100, 20, 0, "step_size must be > 0"),
        (100, 20, -5, "step_size must be > 0"),
    ],
)
def test_walk_forward_split_rejects_invalid_sizes(test_df, window_size, test_size, step_size, message):
    with pytest.raises(ValueError, match=message):
        walk_forward_split(test_df, window_size, test_size, step_size)


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
        assert train_df.index.max() < test_df.index.min()
        assert test_df.index.min() == train_df.index.max() + 1

    for (train_df, test_df), (next_train_df, next_test_df) in zip(splits, splits[1:]):
        assert next_train_df.index.min() == train_df.index.min() + step_size
        assert next_train_df.index.max() == train_df.index.max() + step_size
        assert next_test_df.index.min() == test_df.index.min() + step_size

def test_walk_forward_split_expanding(test_df):
    """Ověří expanding mód - délka trénovacích dat musí růst."""
    window_size = 200
    test_size = 50
    step_size = 100
    expanding = True

    splits = walk_forward_split(test_df, window_size, test_size, step_size, expanding)

    previous_train_len = 0
    for split_index, (train_df, test_df) in enumerate(splits):
        assert len(test_df) == test_size
        assert len(train_df) >= previous_train_len
        assert train_df.index.min() == 0
        assert train_df.index.max() == test_df.index.min() - 1
        assert test_df.index.min() == window_size + split_index * step_size
        previous_train_len = len(train_df)


def test_walk_forward_split_returns_no_splits_for_empty_input(test_df):
    splits = walk_forward_split(test_df.iloc[:0].copy(), window_size=100, test_size=20, step_size=50)

    assert splits == []


def test_walk_forward_split_returns_no_splits_for_insufficient_data(test_df):
    splits = walk_forward_split(test_df.iloc[:110].copy(), window_size=100, test_size=20, step_size=50)

    assert splits == []

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


def test_export_datasets_pkl_roundtrip(tmp_path, test_df):
    splits = walk_forward_split(test_df, window_size=100, test_size=20, step_size=50)

    export_dir = tmp_path / "exported_pkl"
    export_datasets(splits, output_dir=str(export_dir), prefix="test", format="pkl")

    for i, (train_df, test_df) in enumerate(splits):
        train_path = export_dir / f"test_{i}_train.pkl"
        test_path = export_dir / f"test_{i}_test.pkl"

        assert train_path.exists()
        assert test_path.exists()

        train_loaded = pd.read_pickle(train_path)
        test_loaded = pd.read_pickle(test_path)

        pd.testing.assert_frame_equal(train_df, train_loaded)
        pd.testing.assert_frame_equal(test_df, test_loaded)

