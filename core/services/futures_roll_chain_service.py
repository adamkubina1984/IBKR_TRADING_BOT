from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ibkr_trading_bot.config.settings import paths
from ibkr_trading_bot.features.feature_engineering import prepare_dataset_with_targets


RAW_DIR = Path(paths.data_raw())
PROCESSED_DIR = Path(paths.data_processed())
ROLL_OVERLAP_DAYS = 45
RAW_UPDATE_OVERLAP_BARS = 200


def dataset_sidecar_meta_path(csv_path: str | Path) -> Path:
    path = Path(csv_path).expanduser().resolve()
    return path.with_name(path.stem + "_meta.json")


def read_dataset_sidecar_meta(csv_path: str | Path) -> dict[str, Any]:
    meta_path = dataset_sidecar_meta_path(csv_path)
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_dataset_sidecar_meta(csv_path: str | Path, meta: dict[str, Any]) -> Path:
    meta_path = dataset_sidecar_meta_path(csv_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def validate_training_dataset_metadata(csv_path: str | Path) -> dict[str, Any]:
    meta = read_dataset_sidecar_meta(csv_path)
    if not meta:
        return {}
    if str(meta.get("dataset_kind") or "").strip().lower() == "gc_roll_chain":
        if meta.get("canonical") is not True:
            raise ValueError("Vybrany GC roll-chain dataset neni oznacen jako canonical.")
        if meta.get("quality_gate_passed") is not True:
            reasons = ", ".join(str(item) for item in (meta.get("quality_gate_reasons") or [])) or "unknown"
            raise ValueError(f"Vybrany GC roll-chain dataset neprosel quality gate: {reasons}")
    return meta


def parse_expiry_list(value: str | None) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[\s,;]+", str(value or "").strip()):
        expiry = raw.strip()
        if not expiry or not re.fullmatch(r"\d{6}", expiry):
            continue
        if expiry not in seen:
            seen.add(expiry)
            items.append(expiry)
    return items


def effective_download_end_for_expiry(expiry: str | None, requested_end: datetime | None) -> datetime:
    end_dt = requested_end or datetime.now()
    expiry_text = str(expiry or "").strip()
    if not re.fullmatch(r"\d{6}", expiry_text):
        return end_dt
    year = int(expiry_text[:4])
    month = int(expiry_text[4:])
    if not (1 <= month <= 12):
        return end_dt
    next_month = datetime(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    expiry_month_end = next_month - timedelta(seconds=1)
    return min(end_dt, expiry_month_end)


def _expiry_month_end(expiry: str | None) -> datetime:
    expiry_text = str(expiry or "").strip()
    if not re.fullmatch(r"\d{6}", expiry_text):
        raise ValueError(f"Neplatna expirace: {expiry}")
    year = int(expiry_text[:4])
    month = int(expiry_text[4:])
    if not (1 <= month <= 12):
        raise ValueError(f"Neplatna expirace: {expiry}")
    next_month = datetime(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    return next_month - timedelta(seconds=1)


def effective_download_start_for_expiry(
    expiry: str | None,
    requested_start: datetime,
    previous_expiry: str | None = None,
    *,
    overlap_days: int = ROLL_OVERLAP_DAYS,
) -> datetime:
    start_dt = requested_start
    if start_dt.tzinfo is not None:
        start_dt = start_dt.replace(tzinfo=None)
    if not previous_expiry:
        return start_dt
    overlap_anchor = _expiry_month_end(previous_expiry) - timedelta(days=int(overlap_days))
    return max(start_dt, overlap_anchor)


def bar_size_to_minutes(bar_size: str | None) -> int:
    text = str(bar_size or "").strip().lower()
    if text in {"5 mins", "5 min", "5m"}:
        return 5
    if text in {"15 mins", "15 min", "15m"}:
        return 15
    if text in {"30 mins", "30 min", "30m"}:
        return 30
    if text in {"1 hour", "1h", "60 mins", "60 min"}:
        return 60
    return 5


def bar_size_to_code(bar_size: str | None) -> str:
    minutes = bar_size_to_minutes(bar_size)
    if minutes == 60:
        return "1h"
    return f"{minutes}m"


def read_ohlc_csv_strict(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV je prazdne.")

    time_col = None
    for candidate in ("date", "time", "datetime", "timestamp"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError("CSV nema casovy sloupec (date/time/datetime/timestamp).")

    out = df.copy()
    out["date"] = pd.to_datetime(out[time_col], errors="coerce", utc=True).dt.tz_localize(None)
    for column in ("open", "high", "low", "close"):
        if column not in out.columns:
            raise ValueError(f"CSV nema povinny sloupec '{column}'.")
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    else:
        out["volume"] = 0.0
    out = out.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    return out.reset_index(drop=True)


def _detect_flat_mask(df: pd.DataFrame) -> pd.Series:
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    return (
        np.isclose(open_, high, equal_nan=False)
        & np.isclose(high, low, equal_nan=False)
        & np.isclose(low, close, equal_nan=False)
    )


def detect_flat_zero_mask(df: pd.DataFrame) -> pd.Series:
    volume = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
    return _detect_flat_mask(df) & (volume <= 0.0)


def _true_runs(mask: pd.Series) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    size = 0
    values = pd.Series(mask).fillna(False).astype(bool).tolist()
    for idx, flag in enumerate(values):
        if flag:
            if start is None:
                start = idx
                size = 1
            else:
                size += 1
        elif start is not None:
            runs.append((start, idx - 1, size))
            start = None
            size = 0
    if start is not None:
        runs.append((start, len(values) - 1, size))
    return runs


def _max_run_info(mask: pd.Series, ts: pd.Series) -> dict[str, Any]:
    runs = _true_runs(mask)
    if not runs:
        return {"count": 0, "start": None, "end": None}
    start_idx, end_idx, count = max(runs, key=lambda item: item[2])
    return {
        "count": int(count),
        "start": pd.Timestamp(ts.iloc[start_idx]).isoformat() if len(ts) > start_idx else None,
        "end": pd.Timestamp(ts.iloc[end_idx]).isoformat() if len(ts) > end_idx else None,
    }


def audit_ohlc_frame(
    df: pd.DataFrame,
    *,
    expected_step_min: int,
    duplicate_count: int = 0,
) -> dict[str, Any]:
    if df.empty:
        return {
            "row_count": 0,
            "duplicate_count": int(duplicate_count),
            "invalid_ohlc_count": 0,
            "flat_ratio": 0.0,
            "flat_zero_ratio": 0.0,
            "zero_volume_ratio": 0.0,
            "max_flat_zero_run": 0,
            "median_step_min": None,
            "gap_ratio_gt_2x": None,
            "first_ts": None,
            "last_ts": None,
        }

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    flat_mask = _detect_flat_mask(work)
    flat_zero_mask = detect_flat_zero_mask(work)
    volume = pd.to_numeric(work.get("volume", 0.0), errors="coerce").fillna(0.0)
    bad_ohlc = (
        (work["high"] < work["low"])
        | (work["open"] < work["low"])
        | (work["open"] > work["high"])
        | (work["close"] < work["low"])
        | (work["close"] > work["high"])
    )
    dt_min = work["date"].diff().dropna().dt.total_seconds().div(60.0)
    max_run = _max_run_info(flat_zero_mask, work["date"])
    gap_ratio = None
    median_step = None
    if not dt_min.empty:
        median_step = float(dt_min.median())
        gap_ratio = float((dt_min > float(expected_step_min) * 2.0).mean())

    return {
        "row_count": int(len(work)),
        "duplicate_count": int(duplicate_count),
        "invalid_ohlc_count": int(bad_ohlc.sum()),
        "flat_ratio": float(flat_mask.mean()),
        "flat_zero_ratio": float(flat_zero_mask.mean()),
        "zero_volume_ratio": float((volume <= 0.0).mean()),
        "max_flat_zero_run": int(max_run["count"]),
        "max_flat_zero_run_start": max_run["start"],
        "max_flat_zero_run_end": max_run["end"],
        "median_step_min": median_step,
        "gap_ratio_gt_2x": gap_ratio,
        "first_ts": pd.Timestamp(work["date"].iloc[0]).isoformat(),
        "last_ts": pd.Timestamp(work["date"].iloc[-1]).isoformat(),
    }


def clean_ohlc_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    duplicate_count = int(work.duplicated(subset=["date"]).sum())
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    work = work.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    flat_zero_mask = detect_flat_zero_mask(work)
    removed_runs = [run for run in _true_runs(flat_zero_mask) if run[2] >= 3]
    cleaned = work.loc[~flat_zero_mask].copy().reset_index(drop=True)
    return cleaned, {
        "removed_flat_zero_rows": int(flat_zero_mask.sum()),
        "removed_flat_zero_run_count_ge_3": int(len(removed_runs)),
        "duplicate_rows_removed": int(duplicate_count),
    }


def _extract_expiry_from_path(path: str | Path) -> str:
    name = Path(path).name
    match = re.search(r"_(\d{6})(?:_|\.csv$)", name, flags=re.IGNORECASE)
    return str(match.group(1)) if match else "UNKNOWN"


def _read_csv_time_bounds(path: str | Path) -> dict[str, Any]:
    df = pd.read_csv(path, usecols=["date"])
    ts = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None).dropna()
    if ts.empty:
        raise ValueError(f"Soubor {Path(path).name} nema validni casovy sloupec.")
    return {
        "path": str(Path(path).expanduser().resolve()),
        "start": pd.Timestamp(ts.min()).to_pydatetime(),
        "end": pd.Timestamp(ts.max()).to_pydatetime(),
        "rows": int(len(ts)),
        "mtime": float(Path(path).stat().st_mtime),
    }


def _find_existing_raw_contract_csv(
    raw_root: str | Path,
    *,
    symbol: str,
    expiry: str,
    bar_size: str,
) -> dict[str, Any] | None:
    root = Path(raw_root).expanduser().resolve()
    if not root.exists():
        return None
    pattern = f"{str(symbol).upper()}_{expiry}_{bar_size_to_code(bar_size)}_*bars_*.csv"
    candidates: list[dict[str, Any]] = []
    for path in root.glob(pattern):
        try:
            candidates.append(_read_csv_time_bounds(path))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["end"], -item["start"].timestamp(), item["mtime"]), reverse=True)
    return candidates[0]


def _write_contract_history_csv(
    df: pd.DataFrame,
    *,
    symbol: str,
    expiry: str,
    bar_size: str,
    requested_start: datetime,
    output_dir: str | Path,
) -> str:
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    bar_code = bar_size_to_code(bar_size)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = (
        f"{str(symbol).upper()}_{expiry}_{bar_code}_{len(df)}bars_"
        f"{pd.Timestamp(requested_start).strftime('%Y%m%d')}_{ts}.csv"
    )
    out_path = out_root / fname
    out_cols = ["date", "open", "high", "low", "close", "volume"]
    for opt in ("average", "barCount"):
        if opt in df.columns:
            out_cols.append(opt)
    df[out_cols].to_csv(out_path, index=False)
    return str(out_path)


def _merge_contract_csvs(existing_path: str | Path, incoming_path: str | Path) -> pd.DataFrame:
    existing = read_ohlc_csv_strict(existing_path)
    incoming = read_ohlc_csv_strict(incoming_path)
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return merged


def _first_active_ts(df: pd.DataFrame) -> pd.Timestamp:
    if df.empty:
        return pd.Timestamp.min
    volume = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
    active = df.loc[volume > 0.0]
    source = active if not active.empty else df
    return pd.Timestamp(source["date"].min())


def _daily_volume(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    work["day"] = work["date"].dt.floor("D")
    volume = pd.to_numeric(work.get("volume", 0.0), errors="coerce").fillna(0.0)
    work["volume"] = volume
    daily = work.groupby("day")["volume"].sum().sort_index()
    return daily[daily > 0.0]


def _resolve_roll_event(
    current_df: pd.DataFrame,
    next_df: pd.DataFrame,
    *,
    confirm_days: int = 2,
) -> tuple[pd.Timestamp, str]:
    current_daily = _daily_volume(current_df)
    next_daily = _daily_volume(next_df)
    overlap_days = sorted(set(current_daily.index).intersection(set(next_daily.index)))
    if overlap_days:
        streak = 0
        streak_days: list[pd.Timestamp] = []
        for day in overlap_days:
            if float(next_daily.get(day, 0.0)) > float(current_daily.get(day, 0.0)):
                streak += 1
                streak_days.append(pd.Timestamp(day))
                if streak >= int(confirm_days):
                    return pd.Timestamp(streak_days[0]), "volume_crossover_2d"
            else:
                streak = 0
                streak_days = []
    return _first_active_ts(next_df), "fallback_next_first_active"


def _prepared_row_stats(raw_df: pd.DataFrame) -> dict[str, Any]:
    if raw_df.empty:
        return {"prepared_rows": 0, "prepared_retention_ratio": 0.0}
    work = raw_df.rename(columns={"date": "timestamp"}).copy()
    prepared = prepare_dataset_with_targets(work)
    prepared_rows = int(len(prepared)) if isinstance(prepared, pd.DataFrame) else 0
    ratio = float(prepared_rows / max(len(raw_df), 1))
    return {
        "prepared_rows": int(prepared_rows),
        "prepared_retention_ratio": float(ratio),
    }


def _quality_gate(audit: dict[str, Any], prepared_stats: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(audit.get("duplicate_count", 0)) > 0:
        reasons.append(f"duplicate_count={int(audit.get('duplicate_count', 0))}")
    if int(audit.get("invalid_ohlc_count", 0)) > 0:
        reasons.append(f"invalid_ohlc_count={int(audit.get('invalid_ohlc_count', 0))}")
    if float(audit.get("flat_zero_ratio", 0.0) or 0.0) > 0.05:
        reasons.append(f"flat_zero_ratio={float(audit.get('flat_zero_ratio', 0.0)):.4f}>0.0500")
    if int(audit.get("max_flat_zero_run", 0)) > 12:
        reasons.append(f"max_flat_zero_run={int(audit.get('max_flat_zero_run', 0))}>12")
    if float(prepared_stats.get("prepared_retention_ratio", 0.0) or 0.0) < 0.90:
        reasons.append(
            f"prepared_retention_ratio={float(prepared_stats.get('prepared_retention_ratio', 0.0)):.4f}<0.9000"
        )
    return len(reasons) == 0, reasons


def build_gc_roll_chain_dataset(
    contract_csv_paths: list[str | Path],
    *,
    output_dir: str | Path | None = None,
    symbol: str = "GC",
    exchange: str = "COMEX",
    bar_size: str = "5 mins",
    session_mode: str = "full_24h",
    progress_cb=None,
) -> dict[str, Any]:
    if len(contract_csv_paths) < 2:
        raise ValueError("Pro roll-chain builder jsou potreba alespon 2 expirace.")

    expected_step = bar_size_to_minutes(bar_size)
    contracts: list[dict[str, Any]] = []
    for raw_path in contract_csv_paths:
        path = Path(raw_path).expanduser().resolve()
        df_raw = read_ohlc_csv_strict(path)
        duplicate_count = int(df_raw.duplicated(subset=["date"]).sum())
        raw_audit = audit_ohlc_frame(df_raw, expected_step_min=expected_step, duplicate_count=duplicate_count)
        cleaned_df, clean_stats = clean_ohlc_frame(df_raw)
        clean_audit = audit_ohlc_frame(cleaned_df, expected_step_min=expected_step)
        expiry = _extract_expiry_from_path(path)
        if expiry == "UNKNOWN":
            raise ValueError(
                f"Nelze urcit expiraci z nazvu souboru {path.name}. "
                "Raw FUT soubor musi obsahovat segment _YYYYMM_."
            )
        if cleaned_df.empty:
            raise ValueError(f"Kontrakt {expiry} nema po vycisteni zadna pouzitelna data.")
        cleaned_df["source_expiry"] = expiry
        contracts.append(
            {
                "expiry": expiry,
                "path": str(path),
                "raw_df": df_raw,
                "clean_df": cleaned_df,
                "raw_audit": raw_audit,
                "clean_audit": clean_audit,
                "clean_stats": clean_stats,
            }
        )
        if callable(progress_cb):
            progress_cb(
                f"[ROLL] {path.name}: raw={len(df_raw)} clean={len(cleaned_df)} "
                f"flat_zero={raw_audit['flat_zero_ratio']:.3f}"
            )

    contracts.sort(key=lambda item: item["expiry"])

    segments: list[pd.DataFrame] = []
    roll_events: list[dict[str, Any]] = []
    segment_start: pd.Timestamp | None = None
    for idx, contract in enumerate(contracts):
        current_df = contract["clean_df"]
        if idx + 1 < len(contracts):
            next_contract = contracts[idx + 1]
            roll_ts, reason = _resolve_roll_event(current_df, next_contract["clean_df"])
            roll_events.append(
                {
                    "from_expiry": contract["expiry"],
                    "to_expiry": next_contract["expiry"],
                    "roll_timestamp": pd.Timestamp(roll_ts).isoformat(),
                    "reason": reason,
                }
            )
            mask = current_df["date"] < roll_ts
            if segment_start is not None:
                mask &= current_df["date"] >= segment_start
            segment = current_df.loc[mask].copy()
            segment_start = pd.Timestamp(roll_ts)
        else:
            segment = current_df.loc[current_df["date"] >= segment_start].copy() if segment_start is not None else current_df.copy()
        segments.append(segment)

    final_df = pd.concat([segment for segment in segments if not segment.empty], ignore_index=True)
    final_df = final_df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    final_df["roll_segment_id"] = (
        final_df["source_expiry"].ne(final_df["source_expiry"].shift()).cumsum().astype(int)
    )

    final_audit = audit_ohlc_frame(final_df, expected_step_min=expected_step)
    prepared_stats = _prepared_row_stats(final_df[["date", "open", "high", "low", "close", "volume"]].copy())
    quality_passed, quality_reasons = _quality_gate(final_audit, prepared_stats)

    output_root = Path(output_dir or PROCESSED_DIR).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_tag = pd.to_datetime(final_df["date"]).min().strftime("%Y%m%d")
    end_tag = pd.to_datetime(final_df["date"]).max().strftime("%Y%m%d")
    bar_code = bar_size_to_code(bar_size)
    csv_name = f"{symbol}_{bar_code}_rollchain_{len(final_df)}bars_{start_tag}_{end_tag}_{ts}.csv"
    csv_path = output_root / csv_name
    final_df.to_csv(csv_path, index=False)

    meta = {
        "dataset_kind": "gc_roll_chain",
        "canonical": True,
        "quality_gate_passed": bool(quality_passed),
        "quality_gate_reasons": list(quality_reasons),
        "instrument": str(symbol).upper(),
        "exchange": str(exchange).upper(),
        "timeframe": bar_code,
        "bar_size": str(bar_size),
        "session_mode": str(session_mode),
        "roll_rule": {
            "type": "volume_crossover_2d",
            "confirm_days": 2,
            "fallback": "next_first_active",
        },
        "expiries_used": [item["expiry"] for item in contracts],
        "source_contracts": [
            {
                "expiry": item["expiry"],
                "csv_path": item["path"],
                "raw_audit": item["raw_audit"],
                "clean_audit": item["clean_audit"],
                "clean_stats": item["clean_stats"],
            }
            for item in contracts
        ],
        "roll_events": roll_events,
        "raw_rows_total": int(sum(len(item["raw_df"]) for item in contracts)),
        "raw_rows_after_clean": int(sum(len(item["clean_df"]) for item in contracts)),
        "chain_rows": int(len(final_df)),
        "prepared_rows": int(prepared_stats["prepared_rows"]),
        "prepared_retention_ratio": float(prepared_stats["prepared_retention_ratio"]),
        "quality_report": final_audit,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = write_dataset_sidecar_meta(csv_path, meta)

    return {
        "csv_path": str(csv_path),
        "meta_path": str(meta_path),
        "chart_df": final_df[["date", "open", "high", "low", "close", "volume"]].copy(),
        "status_text": (
            f"Canonical GC roll-chain ulozen: {csv_path.name} | rows={len(final_df)} | "
            f"quality={'OK' if quality_passed else 'FAIL'}"
        ),
        "quality_gate_passed": bool(quality_passed),
        "quality_gate_reasons": list(quality_reasons),
        "meta": meta,
    }


def download_and_build_gc_roll_chain(
    *,
    start_date: datetime,
    end_date: datetime | None,
    expiries: list[str],
    bar_size: str,
    output_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    from ibkr_trading_bot.utils.download_ibkr_data import download_ibkr_by_date_range

    if len(expiries) < 2:
        raise ValueError("Pro GC roll-chain je potreba zadat alespon 2 expirace.")

    raw_root = Path(raw_dir or RAW_DIR).expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    downloaded_paths: list[str] = []
    download_summary = {
        "requested_contracts": int(len(expiries)),
        "fresh_downloads": 0,
        "incremental_updates": 0,
        "reused_existing": 0,
        "no_new_data_reuse": 0,
    }
    for idx, expiry in enumerate(expiries, start=1):
        previous_expiry = expiries[idx - 2] if idx > 1 else None
        contract_start = effective_download_start_for_expiry(
            expiry,
            start_date,
            previous_expiry,
            overlap_days=ROLL_OVERLAP_DAYS,
        )
        contract_end = effective_download_end_for_expiry(expiry, end_date)
        existing = _find_existing_raw_contract_csv(
            raw_root,
            symbol="GC",
            expiry=str(expiry),
            bar_size=bar_size,
        )
        step_delta = timedelta(minutes=bar_size_to_minutes(bar_size))
        fresh_enough_end = contract_end - step_delta
        if existing and existing["start"] <= contract_start + step_delta and existing["end"] >= fresh_enough_end:
            if callable(progress_cb):
                progress_cb(
                    f"[ROLL] Reuse {expiry}: {Path(existing['path']).name} | "
                    f"{pd.Timestamp(existing['start']).strftime('%Y-%m-%d')} -> "
                    f"{pd.Timestamp(existing['end']).strftime('%Y-%m-%d')}"
                )
            download_summary["reused_existing"] += 1
            downloaded_paths.append(str(existing["path"]))
            continue
        if callable(progress_cb):
            progress_cb(
                f"[ROLL] Expirace {expiry} ({idx}/{len(expiries)}): "
                f"{contract_start.strftime('%Y-%m-%d')} -> {contract_end.strftime('%Y-%m-%d')}"
            )
        try:
            if existing and existing["start"] <= contract_start + step_delta and existing["end"] < fresh_enough_end:
                overlap_delta = timedelta(minutes=bar_size_to_minutes(bar_size) * RAW_UPDATE_OVERLAP_BARS)
                fetch_start = max(contract_start, existing["end"] - overlap_delta)
                if callable(progress_cb):
                    progress_cb(
                        f"[ROLL] Aktualizuji {expiry}: doplnuji od {fetch_start.strftime('%Y-%m-%d %H:%M')}"
                    )
                incremental_path = download_ibkr_by_date_range(
                    symbol="GC",
                    start_date=fetch_start,
                    end_date=contract_end,
                    bar_size=bar_size,
                    contract_mode="FUT",
                    expiry=str(expiry),
                    output_dir=str(raw_root),
                    max_bars_per_batch=5000,
                    on_progress=(
                        (lambda bn, _tb, rec, expiry_label=expiry: progress_cb(
                            f"[IBKR][{expiry_label}] Batch {bn}: {rec} baru"
                        ))
                        if callable(progress_cb)
                        else None
                    ),
                )
                merged_df = _merge_contract_csvs(existing["path"], incremental_path)
                merged_path = _write_contract_history_csv(
                    merged_df,
                    symbol="GC",
                    expiry=str(expiry),
                    bar_size=bar_size,
                    requested_start=contract_start,
                    output_dir=raw_root,
                )
                download_summary["incremental_updates"] += 1
                downloaded_paths.append(merged_path)
                continue

            path = download_ibkr_by_date_range(
                symbol="GC",
                start_date=contract_start,
                end_date=contract_end,
                bar_size=bar_size,
                contract_mode="FUT",
                expiry=str(expiry),
                output_dir=str(raw_root),
                max_bars_per_batch=5000,
                on_progress=(
                    (lambda bn, _tb, rec, expiry_label=expiry: progress_cb(
                        f"[IBKR][{expiry_label}] Batch {bn}: {rec} baru"
                    ))
                    if callable(progress_cb)
                    else None
                ),
            )
            download_summary["fresh_downloads"] += 1
            downloaded_paths.append(path)
        except RuntimeError as exc:
            if existing and "Žádná data se nestáhla" in str(exc):
                if callable(progress_cb):
                    progress_cb(
                        f"[ROLL] Bez novych dat pro {expiry}, pouzivam existujici {Path(existing['path']).name}"
                    )
                download_summary["no_new_data_reuse"] += 1
                downloaded_paths.append(str(existing["path"]))
                continue
            raise

    result = build_gc_roll_chain_dataset(
        downloaded_paths,
        output_dir=output_dir,
        symbol="GC",
        exchange="COMEX",
        bar_size=bar_size,
        progress_cb=progress_cb,
    )
    summary_text = (
        f"fresh={download_summary['fresh_downloads']} | "
        f"update={download_summary['incremental_updates']} | "
        f"reuse={download_summary['reused_existing']} | "
        f"no_new={download_summary['no_new_data_reuse']}"
    )
    if callable(progress_cb):
        progress_cb(f"[ROLL] Souhrn: {summary_text}")
    result["download_summary"] = dict(download_summary)
    result["status_text"] = f"{result['status_text']} | {summary_text}"
    if isinstance(result.get("meta"), dict):
        result["meta"]["download_summary"] = dict(download_summary)
    return result
