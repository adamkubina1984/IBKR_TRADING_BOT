try:
    from ibkr_trading_bot.core.models.bars import Timeframe
    TIMEFRAME_OPTIONS = [
        Timeframe.M1.value,
        Timeframe.M5.value,
        Timeframe.M15.value,
        Timeframe.M30.value,
        Timeframe.H1.value,
    ]
    DEFAULT_TIMEFRAME = Timeframe.M1.value
except Exception:
    # Fallback na pevné hodnoty
    TIMEFRAME_OPTIONS = ["1 min", "5 mins", "15 mins", "30 mins", "1 hour"]
    DEFAULT_TIMEFRAME = "30 mins"
