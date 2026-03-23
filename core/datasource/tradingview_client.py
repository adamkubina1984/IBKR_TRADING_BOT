import logging
import os

import pandas as pd
from dotenv import load_dotenv
from PySide6.QtCore import QSettings
from tvDatafeed import Interval, TvDatafeed

log = logging.getLogger(__name__)

TV_SETTINGS_ORG = "ibkr_trading_bot"
TV_SETTINGS_APP = "tv_auth"
TV_SETTINGS_KEY_USERNAME = "tv_username"
TV_SETTINGS_KEY_PASSWORD = "tv_password"


def _pick_interval(*cands):
    for name in cands:
        if hasattr(Interval, name):
            return getattr(Interval, name)
    raise AttributeError(f"Interval not found in {cands}")


TF_MAP = {
    "5 min": _pick_interval("in_5_minute", "in_5_min"),
    "15 min": _pick_interval("in_15_minute", "in_15_min"),
    "30 min": _pick_interval("in_30_minute", "in_30_min"),
    "1 hour": _pick_interval("in_1_hour", "in_60_minute"),
}


def _settings_store() -> QSettings:
    return QSettings(TV_SETTINGS_ORG, TV_SETTINGS_APP)


def load_saved_tv_credentials() -> tuple[str | None, str | None]:
    try:
        settings = _settings_store()
        username = str(settings.value(TV_SETTINGS_KEY_USERNAME, "") or "").strip() or None
        password = str(settings.value(TV_SETTINGS_KEY_PASSWORD, "") or "")
        return username, (password or None)
    except Exception:
        return None, None


def save_tv_credentials(username: str | None, password: str | None) -> None:
    user = str(username or "").strip()
    pwd = str(password or "")
    settings = _settings_store()

    if user:
        settings.setValue(TV_SETTINGS_KEY_USERNAME, user)
        os.environ["TV_USERNAME"] = user
    else:
        settings.remove(TV_SETTINGS_KEY_USERNAME)
        os.environ.pop("TV_USERNAME", None)

    if pwd:
        settings.setValue(TV_SETTINGS_KEY_PASSWORD, pwd)
        os.environ["TV_PASSWORD"] = pwd
    else:
        settings.remove(TV_SETTINGS_KEY_PASSWORD)
        os.environ.pop("TV_PASSWORD", None)

    settings.sync()


def resolve_tv_credentials(
    username: str | None = None,
    password: str | None = None,
) -> tuple[str | None, str | None, str]:
    explicit_user = str(username or "").strip() or None
    explicit_pwd = str(password or "") or None
    if explicit_user and explicit_pwd:
        return explicit_user, explicit_pwd, "input"

    env_user = str(os.getenv("TV_USERNAME") or "").strip() or None
    env_pwd = str(os.getenv("TV_PASSWORD") or "") or None
    if env_user and env_pwd:
        return env_user, env_pwd, "env"

    saved_user, saved_pwd = load_saved_tv_credentials()
    if saved_user and saved_pwd:
        return saved_user, saved_pwd, "saved"

    return None, None, "none"


class TradingViewClient:
    def __init__(self, username: str | None = None, password: str | None = None):
        load_dotenv(r"C:\Users\adamk\Můj disk\Trader\.env")
        load_dotenv()

        user, pwd, source = resolve_tv_credentials(username=username, password=password)

        try:
            if user and pwd:
                log.info(f"[TV] Using LOGIN ({source}).")
                self._tv = TvDatafeed(user, pwd)
            else:
                log.warning("[TV] NO-LOGIN mode (no credentials). Data may be limited.")
                self._tv = TvDatafeed()
        except Exception as e:
            log.error(f"[TV] Login failed ({e}). Falling back to NO-LOGIN.")
            self._tv = TvDatafeed()

    def get_history(self, symbol: str, exchange: str, timeframe_label: str, limit: int = 60) -> pd.DataFrame:
        interval = TF_MAP[timeframe_label]
        df = self._tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=int(limit))
        if df is None or df.empty:
            log.error("[TV] No data returned (check symbol/exchange or login).")
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        df = df.reset_index().rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return df[["time", "open", "high", "low", "close", "volume"]]
