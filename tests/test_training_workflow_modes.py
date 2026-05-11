import pandas as pd

from ibkr_trading_bot.gui.tab_model_training import AutoSearchWorker


def test_auto_search_worker_normalizes_legacy_workflow_aliases(tmp_path):
    worker = AutoSearchWorker(
        csv_path="dummy.csv",
        holdout_pct=0.1,
        holdout_min_bars=1000,
        holdout_max_bars=6000,
        training_profiles={},
        candidate_top_n=5,
        candidate_fresh_ratio=0.3,
        state_path=(tmp_path / "state.json").as_posix(),
        search_profile="full",
    )

    assert worker.workflow_mode == "explore"


def test_auto_search_worker_builds_explore_queue_state(tmp_path):
    worker = AutoSearchWorker(
        csv_path="tv_GC_COMEX_5m_sample.csv",
        holdout_pct=0.1,
        holdout_min_bars=1000,
        holdout_max_bars=6000,
        training_profiles={},
        candidate_top_n=5,
        candidate_fresh_ratio=0.3,
        state_path=(tmp_path / "state.json").as_posix(),
        search_profile="explore",
    )

    state = worker._new_state()

    assert state["workflow_mode"] == "explore"
    assert state["phase"] == "explore"
    assert state["queue_idx"] == 0
    assert len(state["queue"]) > 0
    assert all(row["phase"] == "explore" for row in state["queue"])


def test_auto_search_worker_builds_refine_queue_from_region_summary(tmp_path):
        artifact_dir = tmp_path
        (artifact_dir / "tv_GC_COMEX_5m_sample_region_summary.json").write_text(
                """
{
    "version": 1,
    "mode": "explore",
    "approved_regions": [
        {
            "region_id": "lgb_h12_tp50_sl50",
            "models": ["lgb"],
            "horizon_values": [12],
            "tp_bps_min": 50.0,
            "tp_bps_max": 55.0,
            "sl_bps_min": 50.0,
            "sl_bps_max": 55.0,
            "criteria": ["balanced", "profit_first"]
        }
    ]
}
                """.strip(),
                encoding="utf-8",
        )

        worker = AutoSearchWorker(
                csv_path="tv_GC_COMEX_5m_sample.csv",
                holdout_pct=0.1,
                holdout_min_bars=1000,
                holdout_max_bars=6000,
                training_profiles={},
                candidate_top_n=5,
                candidate_fresh_ratio=0.3,
                state_path=(artifact_dir / "state.json").as_posix(),
                search_profile="refine",
        )

        state = worker._new_state()

        assert state["workflow_mode"] == "refine"
        assert len(state["queue"]) == 16
        assert all(row["phase"] == "refine" for row in state["queue"])


def test_auto_search_worker_builds_refresh_queue_from_shortlist(tmp_path):
        artifact_dir = tmp_path
        (artifact_dir / "tv_GC_COMEX_5m_sample_shortlist.json").write_text(
                """
{
    "version": 1,
    "mode": "refine",
    "candidates": [
        {
            "candidate_id": "lgb_h12_tp50_sl50_balanced",
            "model": "lgb",
            "criterion": "balanced",
            "horizon": 12,
            "tp_bps": 50.0,
            "sl_bps": 50.0
        },
        {
            "candidate_id": "hgbt_h16_tp60_sl50_profit_first",
            "model": "hgbt",
            "criterion": "profit_first",
            "horizon": 16,
            "tp_bps": 60.0,
            "sl_bps": 50.0
        }
    ]
}
                """.strip(),
                encoding="utf-8",
        )

        worker = AutoSearchWorker(
                csv_path="tv_GC_COMEX_5m_sample.csv",
                holdout_pct=0.1,
                holdout_min_bars=1000,
                holdout_max_bars=6000,
                training_profiles={},
                candidate_top_n=5,
                candidate_fresh_ratio=0.3,
                state_path=(artifact_dir / "state.json").as_posix(),
                search_profile="refresh",
        )

        state = worker._new_state()

        assert state["workflow_mode"] == "refresh"
        assert len(state["queue"]) == 2
        assert all(row["phase"] == "refresh" for row in state["queue"])


def test_auto_search_worker_migrates_legacy_full_state_to_explore(tmp_path):
        state_path = tmp_path / "tv_GC_COMEX_5m_sample_explore_state.json"
        state_path.write_text(
                """
{
    "version": 1,
    "created_at": "2026-04-10T10:00:00Z",
    "updated_at": "2026-04-10T10:00:00Z",
    "csv_path": "tv_GC_COMEX_5m_sample.csv",
    "spec": {
        "version": 1,
        "search_profile": "full"
    },
    "phase": "quick",
    "quick_queue": [
        {
            "phase": "quick",
            "model": "lgb",
            "criterion": "balanced",
            "horizon": 12,
            "tp_bps": 50.0,
            "sl_bps": 50.0
        }
    ],
    "quick_idx": 0,
    "results": [],
    "stopped": false,
    "completed": false
}
                """.strip(),
                encoding="utf-8",
        )

        worker = AutoSearchWorker(
                csv_path="tv_GC_COMEX_5m_sample.csv",
                holdout_pct=0.1,
                holdout_min_bars=1000,
                holdout_max_bars=6000,
                training_profiles={},
                candidate_top_n=5,
                candidate_fresh_ratio=0.3,
                state_path=state_path.as_posix(),
                search_profile="explore",
        )

        state, resumed = worker._load_or_init_state()

        assert resumed is True
        assert state["workflow_mode"] == "explore"
        assert state["migrated_from"] == "full"
        assert state["queue"][0]["phase"] == "explore"


def test_auto_search_worker_widens_explore_winner_into_refine_region(tmp_path):
        worker = AutoSearchWorker(
            csv_path="tv_GC_COMEX_5m_sample.csv",
            holdout_pct=0.1,
            holdout_min_bars=1000,
            holdout_max_bars=6000,
            training_profiles={},
            candidate_top_n=5,
            candidate_fresh_ratio=0.3,
            state_path=(tmp_path / "state.json").as_posix(),
            search_profile="explore",
        )

        state = worker._new_state()
        state["results"] = [
            {
                "phase": "explore",
                "model": "lgb",
                "criterion": "balanced",
                "horizon": 12,
                "tp_bps": 50.0,
                "sl_bps": 50.0,
                "status": "ok",
                "profit_net": 200.0,
                "sharpe": 0.006,
                "pf": 1.10,
                "qg_reasons": [],
                "meta_obj": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 114856,
                    "n_holdout_bars": 6000,
                },
            }
        ]

        worker._write_region_summary(state)
        payload = worker._load_json_file(worker._region_summary_path())
        region = payload["approved_regions"][0]

        assert region["models"] == ["lgb"]
        assert region["horizon_values"] == [8, 12, 16]
        assert region["tp_bps_min"] == 40.0
        assert region["tp_bps_max"] == 60.0
        assert region["sl_bps_min"] == 40.0
        assert region["sl_bps_max"] == 60.0


def test_auto_search_worker_shortlist_dedupes_criteria_variants(tmp_path):
        worker = AutoSearchWorker(
            csv_path="tv_GC_COMEX_5m_sample.csv",
            holdout_pct=0.1,
            holdout_min_bars=1000,
            holdout_max_bars=6000,
            training_profiles={},
            candidate_top_n=5,
            candidate_fresh_ratio=0.3,
            state_path=(tmp_path / "state.json").as_posix(),
            search_profile="refine",
        )

        state = {
            "spec": {"shortlist_top_k": 5},
            "results": [
                {
                    "status": "ok",
                    "model": "hgbt",
                    "criterion": "balanced",
                    "horizon": 8,
                    "tp_bps": 55.0,
                    "sl_bps": 30.0,
                    "profit_net": 100.0,
                    "sharpe": 0.01,
                    "pf": 1.10,
                    "trades": 200,
                    "meta_obj": {"instrument": "GC", "exchange": "COMEX", "timeframe": "5m", "n_total_bars": 10000, "n_holdout_bars": 1000},
                },
                {
                    "status": "ok",
                    "model": "hgbt",
                    "criterion": "profit_first",
                    "horizon": 8,
                    "tp_bps": 55.0,
                    "sl_bps": 30.0,
                    "profit_net": 100.0,
                    "sharpe": 0.01,
                    "pf": 1.10,
                    "trades": 200,
                },
                {
                    "status": "ok",
                    "model": "lgb",
                    "criterion": "balanced",
                    "horizon": 12,
                    "tp_bps": 80.0,
                    "sl_bps": 55.0,
                    "profit_net": 90.0,
                    "sharpe": 0.009,
                    "pf": 1.05,
                    "trades": 300,
                },
            ],
        }

        worker._write_shortlist(state)
        payload = worker._load_json_file(worker._shortlist_path())

        assert len(payload["candidates"]) == 2
        assert payload["candidates"][0]["candidate_id"] == "hgbt_h8_tp55_sl30"
        assert payload["candidates"][1]["candidate_id"] == "lgb_h12_tp80_sl55"


def test_auto_search_worker_refresh_set_dedupes_criteria_variants(tmp_path):
        worker = AutoSearchWorker(
            csv_path="tv_GC_COMEX_5m_sample.csv",
            holdout_pct=0.1,
            holdout_min_bars=1000,
            holdout_max_bars=6000,
            training_profiles={},
            candidate_top_n=5,
            candidate_fresh_ratio=0.3,
            state_path=(tmp_path / "state.json").as_posix(),
            search_profile="refresh",
        )

        state = {
            "results": [
                {
                    "status": "ok",
                    "model": "hgbt",
                    "criterion": "balanced",
                    "horizon": 8,
                    "tp_bps": 55.0,
                    "sl_bps": 30.0,
                    "profit_net": 100.0,
                    "sharpe": 0.01,
                    "pf": 1.10,
                    "trades": 200,
                },
                {
                    "status": "ok",
                    "model": "hgbt",
                    "criterion": "profit_first",
                    "horizon": 8,
                    "tp_bps": 55.0,
                    "sl_bps": 30.0,
                    "profit_net": 100.0,
                    "sharpe": 0.01,
                    "pf": 1.10,
                    "trades": 200,
                },
                {
                    "status": "ok",
                    "model": "lgb",
                    "criterion": "balanced",
                    "horizon": 12,
                    "tp_bps": 80.0,
                    "sl_bps": 55.0,
                    "profit_net": 90.0,
                    "sharpe": 0.009,
                    "pf": 1.05,
                    "trades": 300,
                },
            ],
        }

        worker._write_refresh_set(state)
        payload = worker._load_json_file(worker._refresh_set_path())

        assert len(payload["refresh_candidates"]) == 2
        assert payload["refresh_candidates"][0]["candidate_id"] == "hgbt_h8_tp55_sl30"
        assert payload["refresh_candidates"][1]["candidate_id"] == "lgb_h12_tp80_sl55"


def test_train_worker_forwards_side_prediction_guard_thresholds(monkeypatch, tmp_path):
    from ibkr_trading_bot.gui import tab_model_training as tab_model_training_module

    captured: dict[str, object] = {}

    def _fake_train_and_evaluate_model(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tab_model_training_module, "train_and_evaluate_model", _fake_train_and_evaluate_model)
    monkeypatch.setattr(tab_model_training_module, "_model_dir", lambda: tmp_path.as_posix())

    worker = tab_model_training_module.TrainWorker(
        df_full=pd.DataFrame({"timestamp": []}),
        holdout_bars=100,
        estimator="hgbt",
        name_prefix="test",
        meta_extra={"label_lookahead_bars": 12},
        training_profile={
            "quality_min_side_prediction_share": 0.07,
            "quality_min_side_prediction_count": 11,
        },
    )

    worker.run()

    assert captured["quality_min_side_prediction_share"] == 0.07
    assert captured["quality_min_side_prediction_count"] == 11