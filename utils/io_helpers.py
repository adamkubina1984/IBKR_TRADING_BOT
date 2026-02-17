"""
Pomocné funkce pro načítání a ukládání dat.
"""

import os

import joblib
import pandas as pd


def load_csv_data(path: str) -> pd.DataFrame:
    """
    Načtení CSV souboru do DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Soubor {path} neexistuje.")
    return pd.read_csv(path)

def save_parquet_data(df: pd.DataFrame, path: str) -> None:
    """
    Uložení DataFrame do Parquet formátu.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    df.to_parquet(path, index=True)

def save_data(df: pd.DataFrame, path: str, format: str = "csv") -> None:
    """
    Uloží DataFrame do CSV nebo Pickle.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    format = (format or "csv").lower()
    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "pkl":
        df.to_pickle(path)
    else:
        raise ValueError("Nepodporovaný formát: použij 'csv' nebo 'pkl'")

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Načte DataFrame ze souboru CSV nebo Pickle.

    :param path: Cesta k souboru .csv nebo .pkl
    :return: DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Soubor neexistuje: {path}")

    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    elif lower.endswith(".pkl"):
        return pd.read_pickle(path)
    else:
        raise ValueError("Podporované formáty jsou pouze .csv a .pkl")

def save_model(model, path: str) -> None:
    """
    Uloží ML model do souboru pomocí joblib.

    :param model: Trénovaný model (např. RandomForest, XGBoost)
    :param path: Cesta k souboru (např. 'model_outputs/model.pkl')
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    joblib.dump(model, path)

def load_latest_data(folder: str = "data/raw", filename: str = "ohlc_data.csv") -> pd.DataFrame:
    """
    Načte nejnovější uložená data z dané složky. Pokud zadaný soubor neexistuje,
    pokusí se vybrat nejnovější CSV v `folder`.

    :param folder: Složka s daty (default 'data/raw')
    :param filename: Preferovaný soubor (default 'ohlc_data.csv')
    :return: DataFrame s indexem 'datetime' (UTC)
    """
    full_path = os.path.join(folder, filename)

    # Fallback logika: pokud preferovaný soubor neexistuje, vyber nejnovější *.csv ve složce
    if not os.path.exists(full_path):
        candidates = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
        if not candidates:
            raise FileNotFoundError(
                f"Soubor {full_path} neexistuje a nebyl nalezen žádný jiný CSV v {folder}."
            )
        # vezmeme „nejnovější“ podle názvu; pokud chceš podle mtime, seřaď podle os.path.getmtime
        candidates.sort(reverse=True)
        full_path = os.path.join(folder, candidates[0])

    df = pd.read_csv(full_path)

    # Autodetekce časového sloupce
    dt_col = None
    for cand in ("datetime", "timestamp", "date"):
        if cand in df.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise ValueError(
            f"CSV {full_path} musí obsahovat sloupec 'datetime' nebo 'timestamp' nebo 'date'."
        )

    # Normalizace času → 'datetime' (UTC) + index
    df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df.set_index("datetime", inplace=True)

    return df

def ensure_dir_exists(path: str) -> None:
    """
    Zajistí, že cílový adresář existuje.
    Pokud neexistuje, vytvoří ho včetně všech nadřazených složek.
    """
    os.makedirs(path, exist_ok=True)
