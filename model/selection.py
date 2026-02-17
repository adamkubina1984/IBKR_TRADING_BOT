# ibkr_trading_bot/model/selection.py
from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelResult:
    model_path: str
    profit: float
    f1: float

def select_best_model(results_csv: str, min_trades: int = 20) -> ModelResult | None:
    # Zodolněné čtení: různé verze hlavičky/počtů sloupců nehodí parser error
    df = pd.read_csv(results_csv, engine="python", on_bad_lines="skip")

    # Jen sloupce, které potřebujeme
    needed = ["model_path", "profit", "f1", "num_trades"]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        # Zkusíme najít obdobné názvy (např. 'trades' místo 'num_trades')
        aliases = {
            "num_trades": ["trades", "n_trades", "trade_count"],
        }
        for canon, alts in aliases.items():
            if canon in missing_cols:
                for alt in alts:
                    if alt in df.columns:
                        df[canon] = df[alt]
                        missing_cols.remove(canon)
                        break
    # Po alias mapování znovu zkontrolovat
    still_missing = [c for c in needed if c not in df.columns]
    if still_missing:
        raise ValueError(f"V souboru {results_csv} chybí požadované sloupce: {still_missing}")

    # Přetypování na numerické a filtrace
    for col in ["profit", "f1", "num_trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["profit", "f1", "num_trades"])

    # Filtr podle min_trades
    df = df[df["num_trades"] >= min_trades].copy()
    if df.empty:
        return None

    # Primárně profit (desc), sekundárně F1 (desc)
    df = df.sort_values(by=["profit", "f1"], ascending=[False, False]).reset_index(drop=True)
    row = df.iloc[0]
    return ModelResult(model_path=str(row["model_path"]), profit=float(row["profit"]), f1=float(row["f1"]))

