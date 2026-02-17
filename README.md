\## Quickstart



```bash

\# 1) Vytvoření a aktivace virtuálního prostředí

python -m venv .venv

\# Windows:

. .venv/Scripts/activate

\# macOS/Linux:

\# source .venv/bin/activate



\# 2) Instalace závislostí

pip install -r requirements.txt



\# 3) Generování featur z historických dat (CSV se sloupci: timestamp, open/high/low/close/volume)

python -m ibkr\_trading\_bot.main generate --input data/raw/ohlc\_data.csv



\# 4) Rychlý trénink modelu a uložení

python -m ibkr\_trading\_bot.main train --features data/processed/features.csv --model-out model\_outputs/model.pkl



\# 5) Vyhodnocení uloženého modelu na datech

python -m ibkr\_trading\_bot.main evaluate --model model\_outputs/model.pkl --features data/processed/features.csv



\# 6) Spuštění desktop GUI

python -m ibkr\_trading\_bot.main gui



