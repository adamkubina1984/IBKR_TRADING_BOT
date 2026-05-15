from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def feature_names_for_model_eval(model: Any) -> list[str] | None:
    try:
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            return [str(x) for x in list(names)]
    except Exception:
        pass
    try:
        names = getattr(model, "feature_name_", None)
        if names is not None:
            out = [str(x) for x in list(names) if str(x)]
            if out:
                return out
    except Exception:
        pass
    try:
        booster = getattr(model, "booster_", None)
        if booster is not None and hasattr(booster, "feature_name"):
            names = booster.feature_name()
            out = [str(x) for x in list(names) if str(x)]
            if out:
                return out
    except Exception:
        pass
    try:
        steps = getattr(model, "steps", None)
        if steps:
            last = steps[-1][1]
            names = getattr(last, "feature_names_in_", None)
            if names is not None:
                return [str(x) for x in list(names)]
            names = getattr(last, "feature_name_", None)
            if names is not None:
                out = [str(x) for x in list(names) if str(x)]
                if out:
                    return out
            booster = getattr(last, "booster_", None)
            if booster is not None and hasattr(booster, "feature_name"):
                names = booster.feature_name()
                out = [str(x) for x in list(names) if str(x)]
                if out:
                    return out
    except Exception:
        pass
    return None


def coerce_features_for_model_eval(X: pd.DataFrame | np.ndarray, model: Any, metadata: dict[str, Any]):
    if not isinstance(X, pd.DataFrame):
        return X

    dfX = X.copy()
    for col in dfX.columns:
        if pd.api.types.is_datetime64_any_dtype(dfX[col]):
            dfX[col] = dfX[col].astype("int64") // 10**6
        elif dfX[col].dtype == "object":
            try:
                parsed = pd.to_datetime(dfX[col], errors="raise")
                dfX[col] = parsed.astype("int64") // 10**6
            except Exception:
                pass

    for column in list(dfX.columns):
        if (not pd.api.types.is_bool_dtype(dfX[column])) and (not pd.api.types.is_numeric_dtype(dfX[column])):
            dfX.drop(columns=[column], inplace=True, errors="ignore")

    expected = None
    if isinstance(metadata, dict):
        expected = metadata.get("trained_features") or metadata.get("expected_features") or metadata.get("features")
    if isinstance(expected, (list, tuple)) and all(isinstance(item, str) for item in expected):
        missing = [item for item in expected if item not in dfX.columns]
        if missing:
            sample = ", ".join(missing[:10])
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(
                f"Dataset pro evaluaci neobsahuje ocekavane featury modelu: {sample}{suffix}"
            )
        dfX = dfX[list(expected)]
        med = dfX.median(numeric_only=True)
        dfX = dfX.fillna(med).fillna(0.0)
        for column in dfX.columns:
            if not pd.api.types.is_bool_dtype(dfX[column]):
                dfX[column] = dfX[column].astype("float32", copy=False)
        return dfX

    names = feature_names_for_model_eval(model)
    if names is not None:
        for item in names:
            if item not in dfX.columns:
                dfX[item] = 0.0
        dfX = dfX[names]

    med = dfX.median(numeric_only=True)
    dfX = dfX.fillna(med).fillna(0.0)
    for column in dfX.columns:
        if not pd.api.types.is_bool_dtype(dfX[column]):
            dfX[column] = dfX[column].astype("float32", copy=False)
    return dfX


def align_X_for_model_eval(model: Any, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        Xdf = X.copy()
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        Xdf = pd.DataFrame(arr)
    names = feature_names_for_model_eval(model)
    if names:
        if all(name in Xdf.columns for name in names):
            Xdf = Xdf.reindex(columns=names, fill_value=0.0)
        elif len(Xdf.columns) == len(names):
            Xdf.columns = list(names)
        else:
            for name in names:
                if name not in Xdf.columns:
                    Xdf[name] = 0.0
            Xdf = Xdf.reindex(columns=names, fill_value=0.0)
    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med).fillna(0.0)
    for column in Xdf.columns:
        if not pd.api.types.is_bool_dtype(Xdf[column]):
            Xdf[column] = Xdf[column].astype("float32", copy=False)
    return Xdf