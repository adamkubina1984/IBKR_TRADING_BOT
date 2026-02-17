# features/feature_engineering.py
"""
Hlavní skript pro výpočet všech featur z historických dat.
Načítá raw data, aplikuje technické indikátory a ukládá featury do processed složky.
"""
import os

import numpy as np
import pandas as pd

from ibkr_trading_bot.utils.logger import logger

# Popisek: Validace a standardizace OHLCV
# ZMĚNA: doplnění helperů a fallback indikátorů

OHLCV_ALIASES = {
    "open": ["open", "Open", "OPEN"],
    "high": ["high", "High", "HIGH"],
    "low":  ["low", "Low", "LOW"],
    "close":["close","Close","CLOSE","Adj Close","adj_close"],
    "volume":["volume","Volume","VOL"]
}

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Popisek: standardizace názvů + konverze na numerické typy
    # ZMĚNA: přidány aliasy pro timestamp + coercion číselných sloupců
    out = df.copy()

    # vč. timestamp aliasů
    ts_aliases = ["timestamp", "Timestamp", "time", "Time", "datetime", "Date", "Datetime"]
    for c in ts_aliases:
        if c in out.columns:
            out.rename(columns={c: "timestamp"}, inplace=True)
            break

    for std, cands in OHLCV_ALIASES.items():
        for c in cands:
            if c in out.columns:
                out.rename(columns={c: std}, inplace=True)
                break

    req = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"Chybí sloupce: {missing}")

    # typy
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # drop zjevně špatných řádků
    out = out.dropna(subset=req)
    return out


# === compute_all_features: hlavní výpočet featur podle features_config.yaml ===

# Bezpečný import indikátorů – preferuj balíčkové importy, ale buď tolerantní
try:
    from ibkr_trading_bot.features.indicators import (
        calculate_atr,
        calculate_macd,
        calculate_rsi,
        calculate_stochastic,
        calculate_williams_r,
    )
except Exception:
    # Pokud by někdo spouštěl mimo balíček, zkusíme lokální importy
    try:
        from features.indicators import (
            calculate_atr,
            calculate_macd,
            calculate_rsi,
            calculate_stochastic,
            calculate_williams_r,
        )
    except Exception:
        # Jako poslední pojistka si jednoduché verze spočteme sami
        calculate_rsi = None
        calculate_macd = None
        calculate_atr = None
        calculate_williams_r = None
        calculate_stochastic = None

def _safe_feature_config():
    """Načti features_config.yaml relativně k tomuto souboru."""
    import yaml
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.normpath(os.path.join(here, "..", "config", "features_config.yaml"))
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Konfigurace featur nebyla nalezena: {cfg_path} "
            f"(očekáváno v ibkr_trading_bot/config/features_config.yaml)"
        )
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_ohlc_columns(df: pd.DataFrame):
    req = ["open", "high", "low", "close", "volume"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Vstupní DataFrame postrádá sloupce: {miss}")

def _fallback_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    down = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def _fallback_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def _fallback_atr(df: pd.DataFrame, window=14):
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

def _fallback_williams_r(df: pd.DataFrame, window=14):
    hh = df["high"].rolling(window=window, min_periods=window).max()
    ll = df["low"].rolling(window=window, min_periods=window).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)

def _fallback_stoch(df: pd.DataFrame, k_period=14, d_period=3):
    ll = df["low"].rolling(window=k_period, min_periods=k_period).min()
    hh = df["high"].rolling(window=k_period, min_periods=k_period).max()
    percent_k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()
    return percent_k, percent_d

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vypočítá featury dle features_config.yaml.
    Vstup: df s indexem = timestamp (UTC) a sloupci open/high/low/close/volume.
    Výstup: df_features s původními OHLC + featurami, bez NA dropů u „potřebných“ sloupců.
    """
    logger.info("Výpočet featur – start")

    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").set_index("timestamp")

    if df.index.name != "timestamp":
        raise ValueError("Očekávám index = 'timestamp' (UTC).")
    if getattr(df.index, "tz", None) is None:
        df = df.tz_localize("UTC")

    _ensure_ohlc_columns(df)

    cfg = _safe_feature_config()

    rsi_cfg   = cfg.get("rsi", {})
    macd_cfg  = cfg.get("macd", {})
    atr_cfg   = cfg.get("atr", {})
    wr_cfg    = cfg.get("williams_r", {})
    stoch_cfg = cfg.get("stochastic", {})
    roll_cfg  = cfg.get("rolling", {})
    br_cfg    = cfg.get("breakouts", {})

    out = df.copy()

    # RSI
    win = int(rsi_cfg.get("window", 14))
    if callable(calculate_rsi):
        out["rsi"] = calculate_rsi(out, period=win)  # DataFrame + period
    else:
        out["rsi"] = _fallback_rsi(out["close"], window=win)

    # MACD
    fast = int(macd_cfg.get("fast", 12))
    slow = int(macd_cfg.get("slow", 26))
    signal = int(macd_cfg.get("signal", 9))
    if callable(calculate_macd):
        macd_res = calculate_macd(out, fast=fast, slow=slow, signal=signal)  # může být DF nebo tuple
        if isinstance(macd_res, pd.DataFrame):
            if "macd" in macd_res:    out["macd_macd"]   = macd_res["macd"]
            if "signal" in macd_res:  out["macd_signal"] = macd_res["signal"]
            if "hist" in macd_res:    out["macd_hist"]   = macd_res["hist"]
        elif isinstance(macd_res, tuple) and len(macd_res) == 3:
            macd, signal_line, hist = macd_res
            out["macd_macd"] = macd
            out["macd_signal"] = signal_line
            out["macd_hist"] = hist
        else:
            # Fallback, kdyby knihovna vrátila nečekaný tvar
            macd, signal_line, hist = _fallback_macd(out["close"], fast, slow, signal)
            out["macd_macd"] = macd
            out["macd_signal"] = signal_line
            out["macd_hist"] = hist
    else:
        macd, signal_line, hist = _fallback_macd(out["close"], fast, slow, signal)
        out["macd_macd"] = macd
        out["macd_signal"] = signal_line
        out["macd_hist"] = hist

    # ATR
    win = int(atr_cfg.get("window", 14))
    if callable(calculate_atr):
        out["atr"] = calculate_atr(out, period=win)  # DataFrame + period
    else:
        out["atr"] = _fallback_atr(out, window=win)

    # Williams %R
    win = int(wr_cfg.get("window", 14))
    if callable(calculate_williams_r):
        out["williams_r"] = calculate_williams_r(out, period=win)  # DataFrame + period
    else:
        out["williams_r"] = _fallback_williams_r(out, window=win)

    # Stochastic
    k = int(stoch_cfg.get("k_period", 14))
    d = int(stoch_cfg.get("d_period", 3))
    if callable(calculate_stochastic):
        st = calculate_stochastic(out, k_period=k, d_period=d)  # DataFrame in, vrací DF se stoch_k, stoch_d
        for col in ["stoch_k", "stoch_d"]:
            if col in st:
                out[col] = st[col]
    else:
        k_series, d_series = _fallback_stoch(out, k_period=k, d_period=d)
        out["stoch_k"] = k_series
        out["stoch_d"] = d_series


    # === Rolling statistiky ===
    if roll_cfg.get("enabled", False):
        wins = roll_cfg.get("windows", [5, 10, 20])
        for w in wins:
            w = int(w)
            out[f"roll_mean_close_{w}"] = out["close"].rolling(window=w, min_periods=w).mean()
            out[f"roll_std_close_{w}"]  = out["close"].rolling(window=w, min_periods=w).std()

    # === Breakouts (Donchian) ===
    if br_cfg.get("enabled", False):
        w = int(br_cfg.get("window", 20))
        highest = out["high"].rolling(window=w, min_periods=w).max()
        lowest  = out["low"].rolling(window=w, min_periods=w).min()
        out[f"donchian_high_{w}"] = highest
        out[f"donchian_low_{w}"]  = lowest
        out[f"breakout_up_{w}"]   = (out["close"] > highest.shift(1)).astype(int)
        out[f"breakout_dn_{w}"]   = (out["close"] < lowest.shift(1)).astype(int)

    # Drop řádky, kde chybí klíčové featury (po začátečních oknech)
    must_have = ["rsi", "atr", "macd_signal", "macd_macd", "macd_hist"]
    subset = [c for c in must_have if c in out.columns]
    if subset:
        out = out.dropna(subset=subset, how="any").copy()
    logger.info("Výpočet featur – hotovo")
    n_rows, n_cols = out.shape
    logger.info(f"Features shape: {n_rows} řádků × {n_cols} sloupců")
    return out


def main(input_path: str, output_path: str | None = None) -> None:
    """
    Vstup:
      - input_path: cesta k CSV s OHLC daty (může být i syntetické: data/synthetic/synthetic_dataset.csv)
      - output_path: volitelné; kam uložit CSV s featurami
    Výstup:
      - CSV s featurami (default: data/processed/features.csv)
    """

    # 1) načtení OHLC
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Vstupní soubor '{input_path}' neexistuje.")
    df = pd.read_csv(input_path)
    df = _normalize_ohlcv(df).dropna().reset_index(drop=True)

    # standardizace sloupců a timestampu
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Vstupní CSV postrádá sloupce: {missing}")

    # parsování času + set index
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Nepodařilo se parse-ovat některé timestampy ve vstupním CSV.")
    df = df.sort_values("timestamp").set_index("timestamp")

    # 2) výpočet featur podle features_config.yaml
    #    (předpokládám, že v souboru už máš logiku, tady jen ukázkový drát)
    #    např.:
    # from .indicators import calculate_rsi, calculate_macd, ...
    # df_features = compute_all_features(df)  # tvá existující funkce
    df_features = compute_all_features(df)  # POZOR: pokud se jmenuje jinak, nahraď za reálnou

    # 3) uložení (relativně k balíčku, aby to fungovalo odkudkoli)
    from pathlib import Path

    if output_path is None:
        # BASE_DIR = .../ibkr_trading_bot
        BASE_DIR = Path(__file__).resolve().parents[1]
        output_dir = BASE_DIR / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "features.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = df_features.reset_index()
    df_out.to_csv(output_path.as_posix(), index=False)
    print(f"✅ Featury uloženy do: {output_path}")


def prepare_dataset_with_targets(df, to_signed: bool = True):
    """
    Dummy funkce – přidá cílovou proměnnou.
    to_signed=True → převede na {-1, 1}, jinak ponechá bool.
    """
    df = df.copy()
    df["target"] = df["close"].shift(-1) > df["close"]  # True/False
    df = df.dropna()
    if to_signed:
        df["target"] = df["target"].map({True: 1, False: -1}).astype(int)
    return df



if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Cesta k CSV s OHLC (musí obsahovat timestamp, open, high, low, close, volume)")
    p.add_argument("--output", default=None, help="Cesta pro uložení features.csv (volitelné)")
    args = p.parse_args()
    main(args.input, args.output)
