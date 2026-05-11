# Refaktor – fáze 1 (bezpečné rozsekání, zachování chování)

Tato fáze přidává `app_context`, `core/*` (services, repositories, utils, models) a `config/settings.py`,
aniž by měnila existující GUI workflow. Všechny **původní funkce a logika** zůstávají funkční.
Postupně lze přesouvat kód do služeb 1:1 (beze změny signatur).

## Co je hotové
- `Timeframe` (1m/5m/15m/30m/1h) a mapování na IBKR + resample pravidla.
- `DataDownloadService` – volá existující `utils/download_ibkr_data.download_data`, umí fallback 1m→resample.
- Repo vrstva pro ukládání dat/modelů/výsledků.
- `AppContext` – sjednocené předávání závislostí.

## Co zůstává nezměněno
- GUI záložky nadále fungují jako dříve. Není nutný zásah do tvé současné logiky.
- Live trading a trénink jsou připravené k přesunu do služeb, ale logika zatím běží původně.

## Další krok (fáze 2)
- Přidat ComboBox **Timeframe** do všech záložek a předávat `Timeframe` do služeb.
- Přesunout kreslení svíček a grafů do `core/utils/plotting.py` (kód z tabů 1:1).
- Přesunout live logiku do `LiveBotService` a v tabu ponechat jen worker tick + signály.


## Fáze 2 – provedené změny (minimálně invazivní)
- Odstraněna duplicitní varianta souboru `gui/tab_live_bot - kopie.py`.
- Přidán `gui/timeframe.py` s jednotnými volbami timeframe. GUI taby mohou použít `TIMEFRAME_OPTIONS`.
- Poznámka: Další přesun logiky (plotting, evaluation, live) do služeb je připraven v `core/`, zachování chování zůstává prioritou.

## Fáze 3 – sjednocení labelů a metrik
- Přidán sdílený modul `utils/labeling.py` jako jediný převodník mezi `binary_01`, `binary_signed`, `ternary_signed` a `ternary_mapped`.
- `utils/metrics.py` je jediný kanonický zdroj klasifikačních a trading metrik.
- `utils/model_io.py` už není druhá implementace metrik; zůstal jen jako kompatibilitní shim.
- Training, CLI evaluace, GUI evaluace i eval service nyní předávají explicitní nebo odvozený `label_mode`, takže se nemíchají mapped labely `0/1/2` se signed labely `-1/0/1`.
- Live diagnostika zůstává trading-only, pokud není k dispozici ground truth.
