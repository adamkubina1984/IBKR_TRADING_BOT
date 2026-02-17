from ibkr_trading_bot.features.feature_engineering import prepare_dataset_with_targets
from ibkr_trading_bot.utils.io_helpers import load_dataframe

# Krok 1: načti historická data
df_raw = load_dataframe("data/raw/ohlc_data.csv")  # změň název souboru podle potřeby

# Krok 2: vytvoř featury a cílové proměnné
df_prepared = prepare_dataset_with_targets(df_raw)

# Krok 3: ulož jako validační CSV
df_prepared.to_csv("data/processed/validation.csv", index=False)

print("✅ Uloženo do data/processed/validation.csv")
