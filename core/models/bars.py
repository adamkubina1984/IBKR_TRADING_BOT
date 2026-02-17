from enum import Enum


class Timeframe(str, Enum):
    M1 = "1 min"
    M5 = "5 mins"
    M15 = "15 mins"
    M30 = "30 mins"
    H1 = "1 hour"

IBKR_BAR_SIZE = {
    Timeframe.M1:  "1 min",
    Timeframe.M5:  "5 mins",
    Timeframe.M15: "15 mins",
    Timeframe.M30: "30 mins",
    Timeframe.H1:  "1 hour",
}

RESAMPLE_RULE = {
    Timeframe.M1:  "1T",
    Timeframe.M5:  "5T",
    Timeframe.M15: "15T",
    Timeframe.M30: "30T",
    Timeframe.H1:  "1H",
}
