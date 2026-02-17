from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

class FeatureMismatchError(RuntimeError):
    pass

def validate_feature_frame(live_df: pd.DataFrame, trained_cols: list[str], *, strict_dtype: bool = False) -> None:
    """
    Validate exact match of feature names and order against training schema.
    Raises FeatureMismatchError on mismatch.
    """
    live_cols = list(live_df.columns)
    if list(trained_cols) != live_cols:
        missing = [c for c in trained_cols if c not in live_cols]
        extra = [c for c in live_cols if c not in trained_cols]
        msg = (
            f"Feature mismatch!\nExpected={trained_cols}\nLive={live_cols}\nMissing={missing}\nExtra={extra}"
        )
        log.error(msg)
        raise FeatureMismatchError(msg)
    if strict_dtype:
        # optional future hook
        pass
