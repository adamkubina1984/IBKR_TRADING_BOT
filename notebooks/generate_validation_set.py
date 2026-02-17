import os

import pandas as pd

# Načti zpracovaný dataset
source_path = "data/processed/processed_data.csv"
output_path = "data/control/val_set_01.csv"

if not os.path.exists(source_path):
    raise FileNotFoundError(f"Soubor {source_path} neexistuje. Spusť nejprve pipeline pro přípravu dat.")

# Načti posledních 100 řádků jako validační sadu
df = pd.read_csv(source_path)
val_df = df.tail(100)

# Ulož do složky control
os.makedirs("data/control", exist_ok=True)
val_df.to_csv(output_path, index=False)

print(f"✅ Uloženo do {output_path} ({len(val_df)} řádků)")
