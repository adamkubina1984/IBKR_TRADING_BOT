# ibkr_trading_bot/utils/data_quality.py
import pandas as pd


def last_bar_info(df: pd.DataFrame, time_col: str = "timestamp"):
    if time_col not in df.columns:
        for c in ("datetime", "date", "time", "Date", "Datetime"):
            if c in df.columns:
                time_col = c
                break
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return None, None
    last_ts = ts.iloc[-1]
    now_utc = pd.Timestamp.utcnow()
    age_min = (now_utc - last_ts).total_seconds() / 60.0
    return last_ts, age_min

def is_fresh(df: pd.DataFrame, max_age_minutes: int = 5, time_col: str = "timestamp") -> tuple[bool,str]:
    last_ts, age = last_bar_info(df, time_col=time_col)
    if last_ts is None or age is None:
        return False, "Chybí použitelný časový sloupec nebo nelze spočítat stáří."
    if age > max_age_minutes:
        return False, f"Poslední bar {last_ts} UTC (~{age:.1f} min) je starší než {max_age_minutes} min."
    return True, f"OK • poslední bar {last_ts} UTC (~{age:.1f} min)"
