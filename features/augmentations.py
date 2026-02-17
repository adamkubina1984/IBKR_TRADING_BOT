# features/augmentations.py

import numpy as np
import pandas as pd


def add_noise(df: pd.DataFrame, noise_level: float = 0.01) -> pd.DataFrame:
    """Přidá náhodný šum do číselných sloupců."""
    noisy_df = df.copy()
    for col in df.select_dtypes(include='number').columns:
        noisy_df[col] += np.random.normal(0, noise_level, size=len(df))
    return noisy_df

def roll_shift(df: pd.DataFrame, max_shift: int = 5) -> pd.DataFrame:
    """Posune časovou řadu o náhodný počet kroků v rozsahu [-max_shift, max_shift]."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return df.shift(shift).dropna()

def mix_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Spojí dvě datové řady lineární kombinací podle váhy alpha."""
    return df1.multiply(alpha) + df2.multiply(1 - alpha)
