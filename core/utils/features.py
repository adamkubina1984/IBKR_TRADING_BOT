# ibkr_trading_bot/core/utils/features.py
import pandas as pd

# Kanonický název a pořadí pro OHLCV(+average)
CANONICAL_ORDER = ["open", "high", "low", "close", "volume", "average"]

# Přípustné aliasy -> kanonický název
ALIASES = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "avg": "average",
    "typical": "average",
    "tp": "average",
    "median": "average",
}

def _normalize_name(col: str) -> str:
    c = str(col).strip().lower()
    return ALIASES.get(c, c)

def canonicalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Přejmenuje známé aliasy a doplní chybějící kanonické sloupce jako NaN.
    Vrátí df s přesným pořadím CANONICAL_ORDER, pokud sloupce existují.
    Ostatní nemění (zůstávají za nimi)."""
    if df is None or df.empty:
        return df

    # Přejmenuj aliasy
    rename_map = {}
    for col in df.columns:
        norm = _normalize_name(col)
        if norm != col:
            rename_map[col] = norm
    if rename_map:
        df = df.rename(columns=rename_map)

    # Doplnění chybějících kanonických sloupců
    for name in CANONICAL_ORDER:
        if name not in df.columns:
            df[name] = pd.NA

    # Primární pořadí
    ordered = [c for c in CANONICAL_ORDER if c in df.columns]
    # Zbytek (feature engineering, indikátory…)
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]

def align_to_expected(df: pd.DataFrame, expected: list[str] | None) -> pd.DataFrame:
    """Zarovná df přesně do pořadí 'expected'.
    Chybějící doplní NaN, přebývající ponechá (za expected)."""
    df = canonicalize_ohlcv(df)
    if not expected:
        return df  # Bez metadat se spokojíme s kanonizací

    for name in expected:
        if name not in df.columns:
            df[name] = pd.NA

    # přesné pořadí expected + zbytek
    rest = [c for c in df.columns if c not in expected]
    return df[expected + rest]
