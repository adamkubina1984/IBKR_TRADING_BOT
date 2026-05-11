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

\# Po změně pinu scikit-learn obnov virtuální prostředí nebo proveď čistý reinstall závislostí.



\# 3) Generování featur z historických dat (CSV se sloupci: timestamp, open/high/low/close/volume)

python -m ibkr\_trading\_bot.main generate --input data/raw/ohlc\_data.csv



\# 4) Rychlý trénink modelu a uložení

python -m ibkr\_trading\_bot.main train --features data/processed/features.csv --model-out model\_outputs/model.pkl



\# 5) Vyhodnocení uloženého modelu na datech

python -m ibkr\_trading\_bot.main evaluate --model model\_outputs/model.pkl --features data/processed/features.csv



\# 6) Spuštění desktop GUI

python -m ibkr\_trading\_bot.main gui

```

## Metrics And Labels

Projekt od dubna 2026 používá jednotnou label sémantiku napříč trainingem, CLI evaluací, GUI evaluací i live diagnostikou.

- Binární režim může interně pracovat s `0/1` nebo `-1/1`, ale před centrálním vyhodnocením se sémantika normalizuje explicitně.
- Ternární obchodní režim používá jako kanonický význam `-1 = short`, `0 = flat`, `1 = long`.
- Pokud estimator potřebuje mapované třídy pro sklearn, používá se pouze interní mapování `0/1/2 = short/flat/long`; při evaluaci se vrací zpět do signed režimu.
- Centrální metriky jsou pouze v `ibkr_trading_bot.utils.metrics.calculate_metrics`.
- Legacy modul `utils/model_io.py` už neobsahuje vlastní výpočty, je to jen kompatibilitní shim.

## Evaluation Rules

- Trading-only diagnostika nepoužívá umělé `y_true`; pokud ground truth není dostupné, reportují se jen trading metriky.
- GUI a reporty preferují net trading metriky, zejména `profit_net`, `winrate_net`, `profit_factor_net` a `sharpe_ratio_net`.
- Side recall pro ranking bias vychází jen ze signed tříd `-1` a `1`; flat třída se do bias skóre nezapočítává jako náhrada za short nebo long.
- CLI evaluace 3-class modelů vrací do vyhodnocení signed predikce `-1/0/1`, aby se nemíchaly mapped labely s dataset targety.

## Developer Notes

- Pro nové call-site vždy používej `label_mode` nebo `infer_label_mode(...)` z `ibkr_trading_bot.utils.labeling`.
- Pro ternární thresholding používej sdílené helpery `ternary_predict_mapped(...)` a `ternary_predict_signed(...)`.
- Pokud přidáš nový report nebo export, nepřepočítávej metriky lokálně; ber je z centrálního výsledku `calculate_metrics(...)`.

## Live Paper Smoke

Pro rychlé ověření Phase 6 gate + persistence flow bez GUI spusť:

```bash
python scripts/run_live_paper_smoke.py --session-root .smoke_live_service
```

Úspěšný běh musí obsahovat `SMOKE_OK`, současně `SMOKE_REAL_BLOCKED=1` a restore výstup `SMOKE_RESTORE_CLOSED_TRADES=1`.

Ve VS Code je stejný krok dostupný jako task `Live: Paper Smoke`.


