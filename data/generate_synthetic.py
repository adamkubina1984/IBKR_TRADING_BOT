# ibkr_trading_bot/data/generate_synthetic.py
"""
Generátor syntetických OHLCV dat pro rychlé testování pipeline.
- Výstup: DataFrame se sloupci ['timestamp','open','high','low','close','volume']
- Logika: jednoduchý GBM (geometric Brownian motion) s „lidsky“ vypadajícími high/low knoty.
- Časová osa: 5min svíčky (lze změnit parametrem).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    n_samples: int = 5_000         # počet svíček
    noise_level: float = 0.05      # relativní „volatilita“ (čím větší, tím živější průběh)
    bar_minutes: int = 5           # perioda svíčky (min)
    seed: int | None = 42       # seed pro reprodukovatelnost


def _calibrate_from_base(df_base: pd.DataFrame | None) -> tuple[float, float]:
    """
    Základní kalibrace startovní ceny a sigma z dodaných dat.
    Pokud df_base není, vrací (price0=100, sigma=0.005).
    """
    if df_base is None or df_base.empty or "close" not in df_base.columns:
        return 100.0, 0.005

    price0 = float(df_base["close"].dropna().iloc[-1])
    ret = df_base["close"].pct_change().dropna()
    sigma = float(ret.std()) if not ret.empty else 0.005
    sigma = float(np.clip(sigma, 0.0005, 0.02))
    return price0, sigma


def _generate_price_path(n: int, price0: float, mu: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Geometric Brownian-like: log-return ~ N(mu, sigma)."""
    rets = rng.normal(loc=mu, scale=sigma, size=n)
    prices = np.empty(n, dtype=float)
    prices[0] = max(0.01, price0 * np.exp(rets[0]))
    for i in range(1, n):
        prices[i] = max(0.01, prices[i - 1] * np.exp(rets[i]))
    return prices


def _intraday_profile(ts: pd.Series, vol: pd.Series, bar_minutes: int) -> pd.Series:
    """Vypočte profil volume podle času v rámci dne (minute-of-day). Normalizováno na průměr 1.0."""
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    prof = vol.groupby(minute_of_day).mean()
    prof = prof / prof.mean() if prof.mean() > 0 else prof

    # plný vektor 0..1439 min
    full = pd.Series(1.0, index=pd.RangeIndex(0, 24 * 60), dtype=float)
    full.loc[prof.index] = prof.values

    # seskupení do bucketů po bar_minutes
    groups = full.index // bar_minutes
    step_prof = full.groupby(groups).mean()
    step_prof.index = pd.RangeIndex(0, 24 * 60 // bar_minutes)
    return step_prof


def _wick_body_stats(df: pd.DataFrame) -> float:
    """
    Odhad „typické“ velikosti knotů relativně k tělu svíčky.
    Vrací mediánový multiplikátor (bezrozměrný).
    """
    body = (df["close"] - df["open"]).abs().replace(0, np.nan)
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0)
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0)
    total_wick = upper_wick + lower_wick
    ratio = (total_wick / body).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 0.8  # rozumný default
    return float(np.clip(ratio.median(), 0.2, 3.0))


def _calibrate_from_base_enhanced(df_base: pd.DataFrame | None, bar_minutes: int) -> dict:
    """
    Z df_base (OHLCV) spočítá:
      - price0: poslední close
      - mu, sigma: drift a volatilita log-návratností per-bar
      - wick_mult: typická velikost knotů (relativně k tělu)
      - vol_scale: typická velikost volume
      - vol_profile: intradenní profil volume (Series délky ~ 1440/bar_minutes)
    """
    if df_base is None or df_base.empty:
        return {
            "price0": 100.0,
            "mu": 0.0,
            "sigma": 0.005,
            "wick_mult": 0.8,
            "vol_scale": 1000.0,
            "vol_profile": pd.Series(1.0, index=pd.RangeIndex(0, 24 * 60 // bar_minutes)),
        }

    df = df_base.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    # Re-sample na požadovaný bar (pro jistotu), OHLCV
    df = df.resample(f"{bar_minutes}min", origin="start_day").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    # Log-returny
    logp = np.log(df["close"])
    r = logp.diff().dropna()
    mu = float(r.mean())
    sigma = float(np.clip(float(r.std()), 1e-4, 0.05))

    price0 = float(df["close"].iloc[-1])
    wick_mult = _wick_body_stats(df)
    vol_scale = float(max(100.0, df["volume"].median()))
    vol_profile = _intraday_profile(df.index.to_series(), df["volume"], bar_minutes)

    return {
        "price0": price0,
        "mu": mu,
        "sigma": sigma,
        "wick_mult": wick_mult,
        "vol_scale": vol_scale,
        "vol_profile": vol_profile,
    }


def generate_synthetic_data(
    df_base: pd.DataFrame | None = None,
    n_samples: int = 5_000,
    noise_level: float = 0.05,
    bar_minutes: int = 5,
    seed: int | None = 42,
    calibrate: bool = True,
) -> pd.DataFrame:
    """
    Vygeneruje syntetická OHLCV data.
    - Pokud je k dispozici df_base s 'close', využije poslední cenu a přibližnou sigma.
    - Parametr noise_level škáluje lokální výkyvy (větší = volatilnější high/low/volume).

    Args:
        df_base: volitelně historická data s 'close' pro kalibraci
        n_samples: počet svíček
        noise_level: 0.01–0.10 je rozumný interval
        bar_minutes: délka svíčky v minutách (default 5)
        seed: seed RNG

    Returns:
        pd.DataFrame: ['timestamp','open','high','low','close','volume']
    """
    rng = np.random.default_rng(seed)

    # 1) kalibrace
    if calibrate:
        calib = _calibrate_from_base_enhanced(df_base, bar_minutes)
    else:
        price0, sigma_base = _calibrate_from_base(df_base)  # jednoduchá varianta
        calib = {
            "price0": price0,
            "mu": 0.0,
            "sigma": sigma_base,
            "wick_mult": 0.8,
            "vol_scale": 1000.0,
            "vol_profile": pd.Series(1.0, index=pd.RangeIndex(0, 24 * 60 // bar_minutes)),
        }

    # noise_level => jemné zvětšení sigma a wicků
    sigma = float(np.clip(calib["sigma"] * (1.0 + noise_level), 1e-4, 0.08))
    mu = float(calib["mu"])
    price0 = float(calib["price0"])

    # 2) časová osa
    end = pd.Timestamp.utcnow().floor(f"{bar_minutes}min")
    idx = pd.date_range(end=end, periods=n_samples, freq=f"{bar_minutes}min", tz="UTC")

    # 3) close + open
    close = _generate_price_path(n_samples, price0, mu, sigma, rng)
    open_ = np.empty_like(close)
    open_[0] = max(0.01, close[0] * float(1.0 + rng.normal(0.0, sigma / 2.0)))
    open_[1:] = close[:-1]

    # 4) high/low – knot podle kalibrace
    body = np.abs(close - open_)
    wick_base = calib["wick_mult"]
    wick_noise = (1.0 + rng.normal(0.0, 0.25, size=n_samples))  # rozptyl v čase
    wick = ((open_ + close) / 2.0) * wick_base * (0.5 + noise_level) * np.abs(wick_noise)

    high = np.maximum(open_, close) + wick + body * rng.uniform(0.1, 0.6, size=n_samples)
    low = np.minimum(open_, close) - wick - body * rng.uniform(0.1, 0.6, size=n_samples)

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    close = np.maximum(close, 0.01)
    open_ = np.maximum(open_, 0.01)

    # 5) volume – intradenní profil × škála × noise
    prof = calib["vol_profile"]
    minute_of_day = (idx.hour * 60 + idx.minute)
    bucket = (minute_of_day // bar_minutes).values
    prof_vals = prof.reindex(pd.RangeIndex(0, len(prof)), fill_value=1.0).to_numpy()
    prof_used = prof_vals[np.clip(bucket, 0, len(prof_vals) - 1)]

    vol_base = calib["vol_scale"] * (1.0 + noise_level) * prof_used
    volume = np.exp(rng.normal(np.log(np.maximum(vol_base, 1.0)), 0.35)).astype(np.float64)
    volume = volume.astype(np.int64)

    # 6) sestavení DataFrame
    df = pd.DataFrame({
        "timestamp": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    return df


# Volitelný rychlý test přes:  python -m ibkr_trading_bot.data.generate_synthetic
if __name__ == "__main__":
    out = generate_synthetic_data(n_samples=20, noise_level=0.05, bar_minutes=5, seed=1)
    print(out.head())
