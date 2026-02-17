# generate_test_data.py

import os

import numpy as np
import pandas as pd

# 📅 Vytvoření 1000 časových kroků
n = 1000
df = pd.DataFrame({
    "open": np.random.uniform(1800, 1900, n),
    "high": np.random.uniform(1900, 2000, n),
    "low": np.random.uniform(1800, 1900, n),
    "close": np.random.uniform(1850, 1950, n),
    "volume": np.random.randint(100, 1000, n)
})

# 🟨 Přidání jednoduchého cílového sloupce pro testovací účely
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
df = df.dropna()

# 💾 Uložení do správné složky
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/processed_data.csv", index=False)

print("✅ Testovací data byla vytvořena v data/processed/processed_data.csv")
