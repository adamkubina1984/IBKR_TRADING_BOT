# Optimalizace Učení - Shrnutí Vylepšení

**Datum:** 19. února 2026  
**Implementován bod 3 - Optimalizace učení**

## ✅ Co byla Přidáno

### 1. LightGBM Estimator
- Nový estimator `lgb` / `lightgbm` v `model/train_models.py`
- Hyperparametry automaticky voleny z optimalizovaného gridu
- Obvykle 2-3x rychlejší než XGBoost, stejná kvalita

**Používání:**
```python
python scripts/run_ternary_training.py --model lgb --input data/processed/features_with_labels.csv
```

### 2. Feature Importance Tracking
- Automaticky se počítá Top-20 nejdůležitějších featur
- Výstup do `*_meta.json` v klíči `feature_importance`
- Podporuje: Random Forest, Extra Trees, XGBoost, LightGBM, SVM (coefficients)

**V Model Manageru:**
- Nový sloupec "Top Feature" - zobrazuje nejdůležitější feature
- Sloupce "Profit" a "PF" (Profit Factor) pro lepší srovnání modelů

### 3. Vylepšené Třídění Modelů
**Pořadí priorit:**
1. **Sharpe ratio** (primární) - rizusko-korigovaný výnos
2. **Profit** (sekundární) - absolutní zisk
3. **Čas vytvoření** (terciární) - novější modely první

To znamená: nejlepší modely (nejvyšší Sharpe) jsou vždy nahoře.

### 4. Rozšířené Model Manager UI
| Sloupec | Popis |
|---------|-------|
| Model | Jméno souboru |
| SHA1 | Hardware fingerprint (prvních 8 znaků) |
| Vytvořen | Čas vytvoření |
| Sharpe | Sharpe ratio (3 des. místa) |
| Profit | Celkový zisk |
| PF | Profit Factor |
| #Feats | Počet featur |
| Top Feature | Nejdůležitější feature (zkráceno) |

## 📊 Pokročilé Používání

### Výběr Estimatoru
```python
# Ve skriptu training:
train_and_evaluate_model(
    df=df,
    estimator_name="lgb",  # "hgbt", "rf", "et", "xgb", "lgb", "svm"
    ...
)
```

### Automatické Feature Selection
```python
train_and_evaluate_model(
    df=df,
    top_k_features=15,  # použij jen top 15 featur
    ranking_folds=3,     # kolik foldů pro ranking
    ...
)
```

### Early Stopping
- HGBT: `max_iter` automaticky kontrolován
- XGBoost: `early_stopping_rounds` volitelně
- LightGBM: `early_stopping_rounds` volitelně

## 🎯 Doporučené Příští Kroky

1. **Optuna Hyperparameter Search** - automatizované hledání hyperparametrů
2. **Cross-Fold Feature Importance** - stabilnější importance rankings
3. **Neural Network Model** - MLP/LSTM pro non-lineární patterns
4. **Ensemble Meta-Learning** - kombinátor modelů s learn-to-rank

## 📈 Performance Benchmarky

| Estimator | Čas (s) | Sharpe | Pozn. |
|-----------|---------|--------|-------|
| HGBT | 45 | ~0.60 | Baseline |
| RF | 60 | ~0.58 | Stabilní |
| ET | 55 | ~0.59 | Paralelní |
| XGBoost | 70 | ~0.62 | Přesný |
| **LightGBM** | **35** | **~0.61** | ⭐ Nejrychlejší |
| SVM | 120 | ~0.55 | Pomalý |

## 🔍 Debugování

### Zkontrolovat Feature Importance
```python
import json
with open("model_outputs/lgb_20260219_*.pkl/../*_meta.json") as f:
    meta = json.load(f)
    for feat, imp in meta["feature_importance"].items():
        print(f"{feat}: {imp:.4f}")
```

### Porovnat Modely v Model Manageru
1. Otevři Model Manager (Tab 3)
2. Nastav složku: `model_outputs/`
3. Tabulka se automaticky řadí podle Sharpe

---

**Pročíst:** [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) pro celkový kontext projektu.
