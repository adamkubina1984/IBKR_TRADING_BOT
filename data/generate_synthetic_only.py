# data/generate_synthetic.py
"""
Generátor syntetických dat pro trénování.
Použití přes CLI v main.py (viz příkaz --generate-synthetic).
"""

from features.augmentations import add_noise, mix_dataframes, roll_shift
from features.feature_engineering import prepare_dataset_with_targets
from utils.io_helpers import load_data, save_data

# Vstupní data
df = load_data("data/processed/original_dataset.csv")

# Generování variant
synthetic_data = []
for i in range(10):
    noisy = add_noise(df, noise_level=0.02)
    shifted = roll_shift(noisy, max_shift=3)
    synthetic_data.append(shifted)

# Kombinace (mixování)
for i in range(0, len(synthetic_data) - 1):
    mixed = mix_dataframes(synthetic_data[i], synthetic_data[i+1], alpha=0.5)
    synthetic_data.append(mixed)

# Sloučení, úprava targetů a uložení
all_data = pd.concat(synthetic_data).dropna()
prepared = prepare_dataset_with_targets(all_data)
save_data(prepared, "data/synthetic/synthetic_augmented_dataset.csv")
