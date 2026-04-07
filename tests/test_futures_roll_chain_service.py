import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ibkr_trading_bot.core.services.dataset_service import DatasetService
from ibkr_trading_bot.core.services.futures_roll_chain_service import (
    build_gc_roll_chain_dataset,
    dataset_sidecar_meta_path,
    download_and_build_gc_roll_chain,
    effective_download_end_for_expiry,
    effective_download_start_for_expiry,
    parse_expiry_list,
    update_gc_roll_chain_latest_contract,
    validate_training_dataset_metadata,
)


def _contract_frame(
    start: str,
    *,
    periods: int,
    base_price: float,
    volume_base: float,
    flat_zero_slice: tuple[int, int] | None = None,
) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq="5min")
    wave = np.sin(np.arange(periods) / 9.0)
    close = base_price + np.arange(periods) * 0.05 + wave
    df = pd.DataFrame(
        {
            "date": idx,
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": volume_base + (np.arange(periods) % 11) * 3.0,
        }
    )
    if flat_zero_slice is not None:
        start_idx, end_idx = flat_zero_slice
        px = float(df.loc[start_idx, "close"])
        df.loc[start_idx:end_idx, ["open", "high", "low", "close"]] = px
        df.loc[start_idx:end_idx, "volume"] = 0.0
    return df
def test_parse_expiry_list_accepts_comma_and_space_separated_values():
    assert parse_expiry_list("202504, 202506;202508 202506") == ["202504", "202506", "202508"]


def test_effective_download_end_for_expiry_caps_past_month():
    requested_end = pd.Timestamp("2026-03-16 12:00:00").to_pydatetime()
    effective_end = effective_download_end_for_expiry("202412", requested_end)

    assert effective_end == pd.Timestamp("2024-12-31 23:59:59").to_pydatetime()


def test_effective_download_start_for_expiry_uses_overlap_window():
    effective_start = effective_download_start_for_expiry(
        "202502",
        pd.Timestamp("2024-10-01 00:00:00").to_pydatetime(),
        "202412",
    )

    assert effective_start == pd.Timestamp("2024-11-16 23:59:59").to_pydatetime()


def test_build_gc_roll_chain_dataset_writes_canonical_csv_and_meta(tmp_path):
    contract_a = _contract_frame(
        "2025-01-01 00:00:00",
        periods=1000,
        base_price=2700.0,
        volume_base=80.0,
        flat_zero_slice=(40, 44),
    )
    contract_b = _contract_frame(
        "2025-01-03 00:00:00",
        periods=1000,
        base_price=2725.0,
        volume_base=120.0,
        flat_zero_slice=(25, 28),
    )

    path_a = tmp_path / "GC_202504_5m.csv"
    path_b = tmp_path / "GC_202506_5m.csv"
    contract_a.to_csv(path_a, index=False)
    contract_b.to_csv(path_b, index=False)

    result = build_gc_roll_chain_dataset(
        [path_a, path_b],
        output_dir=tmp_path,
        symbol="GC",
        exchange="COMEX",
        bar_size="5 mins",
    )

    csv_path = Path(result["csv_path"])
    meta_path = dataset_sidecar_meta_path(csv_path)
    assert csv_path.exists()
    assert meta_path.exists()
    assert result["quality_gate_passed"] is True

    out_df = pd.read_csv(csv_path)
    assert "source_expiry" in out_df.columns
    assert {str(value) for value in out_df["source_expiry"].unique()} == {"202504", "202506"}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["dataset_kind"] == "gc_roll_chain"
    assert meta["canonical"] is True
    assert meta["quality_gate_passed"] is True
    assert meta["prepared_retention_ratio"] >= 0.90
    assert meta["quality_report"]["flat_zero_ratio"] == pytest.approx(0.0)
    assert len(meta["roll_events"]) == 1


def test_download_and_build_gc_roll_chain_caps_end_date_per_expiry(monkeypatch, tmp_path):
    captured_end_dates: list[tuple[str, object]] = []

    def _fake_downloader(**kwargs):
        expiry = str(kwargs["expiry"])
        captured_end_dates.append((expiry, kwargs["end_date"]))
        csv_path = tmp_path / f"GC_{expiry}_5m.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=4, freq="5min"),
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [1.1, 2.1, 3.1, 4.1],
                "low": [0.9, 1.9, 2.9, 3.9],
                "close": [1.0, 2.0, 3.0, 4.0],
                "volume": [1.0, 1.0, 1.0, 1.0],
            }
        ).to_csv(csv_path, index=False)
        return str(csv_path)

    def _fake_builder(paths, **kwargs):
        return {
            "csv_path": str(tmp_path / "GC_5m_rollchain.csv"),
            "meta_path": str(tmp_path / "GC_5m_rollchain_meta.json"),
            "chart_df": pd.DataFrame(
                {
                    "date": pd.date_range("2025-01-01", periods=2, freq="5min"),
                    "open": [1.0, 2.0],
                    "high": [1.1, 2.1],
                    "low": [0.9, 1.9],
                    "close": [1.0, 2.0],
                    "volume": [1.0, 1.0],
                }
            ),
            "status_text": "ok",
            "quality_gate_passed": True,
            "quality_gate_reasons": [],
            "meta": {},
        }

    monkeypatch.setattr(
        "ibkr_trading_bot.utils.download_ibkr_data.download_ibkr_by_date_range",
        _fake_downloader,
    )
    monkeypatch.setattr(
        "ibkr_trading_bot.core.services.futures_roll_chain_service.build_gc_roll_chain_dataset",
        _fake_builder,
    )

    download_and_build_gc_roll_chain(
        start_date=pd.Timestamp("2024-06-01").to_pydatetime(),
        end_date=pd.Timestamp("2026-03-16 12:00:00").to_pydatetime(),
        expiries=["202412", "202606"],
        bar_size="5 mins",
        output_dir=tmp_path,
        raw_dir=tmp_path,
    )

    assert captured_end_dates == [
        ("202412", pd.Timestamp("2024-12-31 23:59:59").to_pydatetime()),
        ("202606", pd.Timestamp("2026-03-16 12:00:00").to_pydatetime()),
    ]


def test_download_and_build_gc_roll_chain_reuses_old_contract_and_updates_only_latest(monkeypatch, tmp_path):
    older_path = tmp_path / "GC_202512_5m_10bars_20241001_20260316_000001.csv"
    latest_path = tmp_path / "GC_202602_5m_10bars_20251116_20260316_000001.csv"
    pd.DataFrame(
        {
            "date": ["2025-10-01 00:00:00", "2025-12-31 23:55:00"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [1.0, 1.0],
        }
    ).to_csv(older_path, index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2025-11-16", periods=10, freq="5min"),
            "open": np.linspace(10.0, 11.0, 10),
            "high": np.linspace(10.1, 11.1, 10),
            "low": np.linspace(9.9, 10.9, 10),
            "close": np.linspace(10.0, 11.0, 10),
            "volume": np.ones(10),
        }
    ).to_csv(latest_path, index=False)

    calls: list[dict[str, object]] = []
    messages: list[str] = []

    def _fake_downloader(**kwargs):
        calls.append(
            {
                "expiry": kwargs["expiry"],
                "start_date": kwargs["start_date"],
                "end_date": kwargs["end_date"],
            }
        )
        csv_path = tmp_path / f"GC_{kwargs['expiry']}_5m_5bars_20260316_999999.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2026-03-16 00:00:00", periods=5, freq="5min"),
                "open": np.linspace(20.0, 21.0, 5),
                "high": np.linspace(20.1, 21.1, 5),
                "low": np.linspace(19.9, 20.9, 5),
                "close": np.linspace(20.0, 21.0, 5),
                "volume": np.ones(5),
            }
        ).to_csv(csv_path, index=False)
        return str(csv_path)

    def _fake_builder(paths, **kwargs):
        return {
            "csv_path": str(tmp_path / "GC_5m_rollchain.csv"),
            "meta_path": str(tmp_path / "GC_5m_rollchain_meta.json"),
            "chart_df": pd.DataFrame(
                {
                    "date": pd.date_range("2025-01-01", periods=2, freq="5min"),
                    "open": [1.0, 2.0],
                    "high": [1.1, 2.1],
                    "low": [0.9, 1.9],
                    "close": [1.0, 2.0],
                    "volume": [1.0, 1.0],
                }
            ),
            "status_text": "ok",
            "quality_gate_passed": True,
            "quality_gate_reasons": [],
            "meta": {},
        }

    monkeypatch.setattr(
        "ibkr_trading_bot.utils.download_ibkr_data.download_ibkr_by_date_range",
        _fake_downloader,
    )
    monkeypatch.setattr(
        "ibkr_trading_bot.core.services.futures_roll_chain_service.build_gc_roll_chain_dataset",
        _fake_builder,
    )

    result = download_and_build_gc_roll_chain(
        start_date=pd.Timestamp("2025-10-01").to_pydatetime(),
        end_date=pd.Timestamp("2026-03-16 12:00:00").to_pydatetime(),
        expiries=["202512", "202602"],
        bar_size="5 mins",
        output_dir=tmp_path,
        raw_dir=tmp_path,
        progress_cb=messages.append,
    )

    assert len(calls) == 1
    assert calls[0]["expiry"] == "202602"
    assert pd.Timestamp(calls[0]["start_date"]) > pd.Timestamp("2025-11-01 00:00:00")
    assert result["download_summary"] == {
        "requested_contracts": 2,
        "fresh_downloads": 0,
        "incremental_updates": 1,
        "reused_existing": 1,
        "no_new_data_reuse": 0,
    }
    assert "fresh=0 | update=1 | reuse=1 | no_new=0" in result["status_text"]
    assert any("[ROLL] Souhrn: fresh=0 | update=1 | reuse=1 | no_new=0" in msg for msg in messages)


def test_update_gc_roll_chain_latest_contract_reuses_history_and_refreshes_only_latest(monkeypatch, tmp_path):
    older_a = tmp_path / "GC_202410_5m_10bars_20240801_20260325_000001.csv"
    older_b = tmp_path / "GC_202412_5m_10bars_20240916_20260325_000001.csv"
    latest_existing = tmp_path / "GC_202502_5m_10bars_20241116_20260325_000001.csv"

    pd.DataFrame(
        {
            "date": ["2024-08-01 00:00:00", "2024-10-28 16:55:00"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [1.0, 1.0],
        }
    ).to_csv(older_a, index=False)
    pd.DataFrame(
        {
            "date": ["2024-09-17 00:00:00", "2024-12-27 16:55:00"],
            "open": [3.0, 4.0],
            "high": [3.1, 4.1],
            "low": [2.9, 3.9],
            "close": [3.0, 4.0],
            "volume": [1.0, 1.0],
        }
    ).to_csv(older_b, index=False)
    pd.DataFrame(
        {
            "date": ["2024-11-17 00:00:00", "2025-02-26 16:55:00"],
            "open": [10.0, 11.0],
            "high": [10.1, 11.1],
            "low": [9.9, 10.9],
            "close": [10.0, 11.0],
            "volume": [1.0, 1.0],
        }
    ).to_csv(latest_existing, index=False)

    calls: list[dict[str, object]] = []
    messages: list[str] = []
    captured_paths: list[str] = []

    def _fake_downloader(**kwargs):
        calls.append(
            {
                "expiry": kwargs["expiry"],
                "start_date": kwargs["start_date"],
                "end_date": kwargs["end_date"],
            }
        )
        csv_path = tmp_path / "GC_202502_5m_5bars_20260325_999999.csv"
        pd.DataFrame(
            {
                "date": ["2025-02-26 16:50:00", "2025-02-28 16:55:00"],
                "open": [20.0, 21.0],
                "high": [20.1, 21.1],
                "low": [19.9, 20.9],
                "close": [20.0, 21.0],
                "volume": [1.0, 1.0],
            }
        ).to_csv(csv_path, index=False)
        return str(csv_path)

    def _fake_builder(paths, **kwargs):
        captured_paths[:] = list(paths)
        return {
            "csv_path": str(tmp_path / "GC_5m_rollchain.csv"),
            "meta_path": str(tmp_path / "GC_5m_rollchain_meta.json"),
            "chart_df": pd.DataFrame(
                {
                    "date": pd.date_range("2025-01-01", periods=2, freq="5min"),
                    "open": [1.0, 2.0],
                    "high": [1.1, 2.1],
                    "low": [0.9, 1.9],
                    "close": [1.0, 2.0],
                    "volume": [1.0, 1.0],
                }
            ),
            "status_text": "ok",
            "quality_gate_passed": True,
            "quality_gate_reasons": [],
            "meta": {},
        }

    monkeypatch.setattr(
        "ibkr_trading_bot.utils.download_ibkr_data.download_ibkr_by_date_range",
        _fake_downloader,
    )
    monkeypatch.setattr(
        "ibkr_trading_bot.core.services.futures_roll_chain_service.build_gc_roll_chain_dataset",
        _fake_builder,
    )

    result = update_gc_roll_chain_latest_contract(
        start_date=pd.Timestamp("2024-08-01").to_pydatetime(),
        end_date=pd.Timestamp("2025-02-28 23:59:59").to_pydatetime(),
        expiries=["202410", "202412", "202502"],
        bar_size="5 mins",
        output_dir=tmp_path,
        raw_dir=tmp_path,
        preferred_contract_paths={
            "202410": str(older_a),
            "202412": str(older_b),
            "202502": str(latest_existing),
        },
        progress_cb=messages.append,
    )

    assert len(calls) == 1
    assert calls[0]["expiry"] == "202502"
    assert captured_paths[0] == str(older_a.resolve())
    assert captured_paths[1] == str(older_b.resolve())
    assert Path(captured_paths[2]).name.startswith("GC_202502_5m_")
    assert Path(captured_paths[2]).name != latest_existing.name
    assert result["download_summary"] == {
        "requested_contracts": 3,
        "fresh_downloads": 0,
        "incremental_updates": 1,
        "reused_existing": 2,
        "no_new_data_reuse": 0,
    }
    assert "fresh=0 | update=1 | reuse=2 | no_new=0" in result["status_text"]
    assert any("Aktualizuji pouze posledni expiraci 202502" in msg for msg in messages)


def test_validate_training_dataset_metadata_rejects_failed_roll_chain(tmp_path):
    csv_path = tmp_path / "GC_5m_rollchain_bad.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5, freq="5min"),
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [1, 1, 1, 1, 1],
        }
    ).to_csv(csv_path, index=False)
    dataset_sidecar_meta_path(csv_path).write_text(
        json.dumps(
            {
                "dataset_kind": "gc_roll_chain",
                "canonical": True,
                "quality_gate_passed": False,
                "quality_gate_reasons": ["flat_zero_ratio=0.5500>0.0500"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="quality gate"):
        validate_training_dataset_metadata(csv_path)


def test_build_gc_roll_chain_dataset_rejects_paths_without_expiry_token(tmp_path):
    bad_path = tmp_path / "GC_5m_bad.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20, freq="5min"),
            "open": np.linspace(10.0, 20.0, 20),
            "high": np.linspace(10.2, 20.2, 20),
            "low": np.linspace(9.8, 19.8, 20),
            "close": np.linspace(10.1, 20.1, 20),
            "volume": np.ones(20),
        }
    ).to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="Nelze urcit expiraci"):
        build_gc_roll_chain_dataset([bad_path, bad_path], output_dir=tmp_path, symbol="GC", exchange="COMEX")


def test_dataset_service_blocks_failed_canonical_dataset(tmp_path):
    csv_path = tmp_path / "GC_5m_rollchain_bad.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=40, freq="5min"),
            "open": np.linspace(10.0, 20.0, 40),
            "high": np.linspace(10.2, 20.2, 40),
            "low": np.linspace(9.8, 19.8, 40),
            "close": np.linspace(10.1, 20.1, 40),
            "volume": np.ones(40),
        }
    ).to_csv(csv_path, index=False)
    dataset_sidecar_meta_path(csv_path).write_text(
        json.dumps(
            {
                "dataset_kind": "gc_roll_chain",
                "canonical": True,
                "quality_gate_passed": False,
                "quality_gate_reasons": ["prepared_retention_ratio=0.6500<0.9000"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="quality gate"):
        DatasetService().prepare_from_csv(str(csv_path))
