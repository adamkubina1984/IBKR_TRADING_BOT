import pandas as pd

from ibkr_trading_bot.core.utils.plotting import prepare_for_chart


def test_prepare_for_chart_accepts_timezone_aware_datetime_series():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-03-16T10:00:00Z", "2026-03-16T10:05:00Z"],
                utc=True,
            ),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, 12.0],
        }
    )

    out = prepare_for_chart(df)

    assert list(out.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None
    assert len(out) == 2
