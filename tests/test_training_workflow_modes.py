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


def test_auto_search_worker_builds_refine_queue_from_explicit_source_artifact(tmp_path):
        artifact_dir = tmp_path
        region_summary_path = artifact_dir / "approved_refine_source_region_summary.json"
        region_summary_path.write_text(
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
                csv_path="tv_GC_COMEX_5m_source.csv",
                holdout_pct=0.1,
                holdout_min_bars=1000,
                holdout_max_bars=6000,
                training_profiles={},
                candidate_top_n=5,
                candidate_fresh_ratio=0.3,
                state_path=(artifact_dir / "approved_refine_source_refine_state.json").as_posix(),
                source_artifact_path=region_summary_path.as_posix(),
                search_profile="refine",
        )

        state = worker._new_state()

        assert state["workflow_mode"] == "refine"
        assert state["spec"]["source_region_summary"] == region_summary_path.as_posix()
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


def test_auto_search_worker_builds_refresh_queue_from_explicit_source_artifact(tmp_path):
        artifact_dir = tmp_path
        shortlist_path = artifact_dir / "approved_refresh_source_shortlist.json"
        shortlist_path.write_text(
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
                csv_path="tv_GC_COMEX_5m_source.csv",
                holdout_pct=0.1,
                holdout_min_bars=1000,
                holdout_max_bars=6000,
                training_profiles={},
                candidate_top_n=5,
                candidate_fresh_ratio=0.3,
                state_path=(artifact_dir / "tv_GC_COMEX_5m_target_refresh_state.json").as_posix(),
                source_artifact_path=shortlist_path.as_posix(),
                refresh_csv_path="tv_GC_COMEX_5m_target.csv",
                search_profile="refresh",
        )

        state = worker._new_state()

        assert state["workflow_mode"] == "refresh"
        assert state["spec"]["source_artifact"] == shortlist_path.as_posix()
        assert state["spec"]["source_artifact_kind"] == "shortlist"
        assert state["spec"]["target_csv_path"] == "tv_GC_COMEX_5m_target.csv"
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


def test_auto_search_worker_migrates_legacy_fast_state_to_refine(tmp_path):
        state_path = tmp_path / "tv_GC_COMEX_5m_sample_fast_state.json"
        state_path.write_text(
                """
{
    "version": 1,
    "created_at": "2026-04-10T10:00:00Z",
    "updated_at": "2026-04-10T10:00:00Z",
    "csv_path": "tv_GC_COMEX_5m_sample.csv",
    "spec": {
        "version": 1,
        "search_profile": "fast",
        "criteria": ["balanced", "profit_first"],
        "label_horizon_bars": [12],
        "label_tp_bps": [50.0, 55.0],
        "label_sl_bps": [50.0, 55.0]
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
    "quick_idx": 1,
    "results": [
        {
            "status": "ok",
            "model": "lgb"
        }
    ],
    "stopped": true,
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
                search_profile="refine",
        )

        state, resumed = worker._load_or_init_state()

        assert resumed is True
        assert state["workflow_mode"] == "refine"
        assert state["migrated_from"] == "fast"
        assert state["queue_idx"] == 1
        assert state["queue"][0]["phase"] == "refine"


def test_auto_search_worker_resumes_refine_state_when_csv_path_format_changes(tmp_path):
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
                csv_path=str((artifact_dir / "tv_GC_COMEX_5m_sample.csv").resolve()),
                holdout_pct=0.1,
                holdout_min_bars=1000,
                holdout_max_bars=6000,
                training_profiles={},
                candidate_top_n=5,
                candidate_fresh_ratio=0.3,
                state_path=(artifact_dir / "tv_GC_COMEX_5m_sample_refine_state.json").as_posix(),
                search_profile="refine",
        )

        state = worker._new_state()
        state["queue_idx"] = 3
        state["csv_path"] = str((artifact_dir / "tv_GC_COMEX_5m_sample.csv").resolve()).replace("\\", "/")
        worker._save_state(state)

        resumed_state, resumed = worker._load_or_init_state()

        assert resumed is True
        assert resumed_state["queue_idx"] == 3
        assert resumed_state["csv_path"] == worker.csv_path


def test_auto_search_worker_keeps_incomplete_progress_when_spec_changes(tmp_path):
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
                state_path=(artifact_dir / "tv_GC_COMEX_5m_sample_refine_state.json").as_posix(),
                search_profile="refine",
        )

        state = worker._new_state()
        state["queue_idx"] = 4
        state["spec"] = dict(state["spec"])
        state["spec"]["shortlist_top_k"] = 99
        worker._save_state(state)

        resumed_state, resumed = worker._load_or_init_state()

        assert resumed is True
        assert resumed_state["queue_idx"] == 4
        assert resumed_state["spec"]["shortlist_top_k"] == 99


def test_auto_search_worker_recovers_zeroed_refine_progress_from_saved_artifacts(monkeypatch, tmp_path):
        from ibkr_trading_bot.gui import tab_model_training as tab_model_training_module

        artifact_dir = tmp_path
        csv_path = str(artifact_dir / "tv_GC_COMEX_5m_sample.csv")
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

        monkeypatch.setattr(tab_model_training_module, "_model_dir", lambda: artifact_dir.as_posix())
        monkeypatch.setattr(
            tab_model_training_module.DatasetService,
            "prepare_from_csv",
            lambda self, *args, **kwargs: pd.DataFrame({"timestamp": list(range(10))}),
        )

        worker = AutoSearchWorker(
            csv_path=csv_path,
            holdout_pct=0.1,
            holdout_min_bars=1000,
            holdout_max_bars=6000,
            training_profiles={},
            candidate_top_n=5,
            candidate_fresh_ratio=0.3,
            state_path=(artifact_dir / "tv_GC_COMEX_5m_sample_refine_state.json").as_posix(),
            search_profile="refine",
        )

        state = worker._new_state()
        worker._save_state(state)

        first_meta = {
            "created_at": "20260519_070000",
            "created_at_iso": "2026-05-19T07:00:00+00:00",
            "estimator_name": "lgb",
            "workflow_mode": "refine",
            "training_mode": "refine",
            "training_profile": {
                "candidate_selection_criterion": "balanced",
                "training_mode": "standard",
            },
            "n_total_bars": 10,
            "label_horizon_bars": 12,
            "label_take_profit_bps": 50.0,
            "label_stop_loss_bps": 50.0,
            "metrics_holdout": {
                "profit_net": 125.0,
                "sharpe": 0.02,
                "pf": 1.4,
                "num_trades": 42,
            },
            "quality_gate": {"evaluated": True, "passed": True, "reasons": []},
            "tab5_holdout_ranking": {"csv_path": csv_path},
            "search_plan": {"search_backend_requested": "grid", "search_backend_used": "grid"},
        }
        second_meta = {
            "created_at": "20260519_071500",
            "created_at_iso": "2026-05-19T07:15:00+00:00",
            "estimator_name": "lgb",
            "workflow_mode": "refine",
            "training_mode": "refine",
            "training_profile": {
                "candidate_selection_criterion": "balanced",
                "training_mode": "standard",
            },
            "n_total_bars": 10,
            "label_horizon_bars": 12,
            "label_take_profit_bps": 50.0,
            "label_stop_loss_bps": 55.0,
            "metrics_holdout": {
                "profit_net": 95.0,
                "sharpe": 0.01,
                "pf": 1.2,
                "num_trades": 35,
            },
            "quality_gate": {"evaluated": True, "passed": True, "reasons": []},
            "tab5_holdout_ranking": {"csv_path": csv_path},
            "search_plan": {"search_backend_requested": "grid", "search_backend_used": "grid"},
        }

        (artifact_dir / "GC_COMEX_5m_10bars_lgb_20260519_070000_meta.json").write_text(
            __import__("json").dumps(first_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (artifact_dir / "GC_COMEX_5m_10bars_lgb_20260519_071500_meta.json").write_text(
            __import__("json").dumps(second_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        resumed_state, resumed = worker._load_or_init_state()

        assert resumed is True
        assert resumed_state["queue_idx"] == 0
        assert len(resumed_state["queue"]) == len(state["queue"]) - 2
        assert len(resumed_state["results"]) == 2
        assert resumed_state["results"][0]["tp_bps"] == 50.0
        assert resumed_state["results"][0]["sl_bps"] == 50.0
        assert resumed_state["results"][1]["tp_bps"] == 50.0
        assert resumed_state["results"][1]["sl_bps"] == 55.0
        assert worker._last_recovered_results_count == 2


def test_auto_search_worker_reconciles_duplicate_results_and_prunes_completed_queue(tmp_path):
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

        (tmp_path / "tv_GC_COMEX_5m_sample_region_summary.json").write_text(
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
            "tp_bps_max": 50.0,
            "sl_bps_min": 50.0,
            "sl_bps_max": 50.0,
            "criteria": ["balanced"]
        }
    ]
}
            """.strip(),
            encoding="utf-8",
        )

        state = {
            "version": 2,
            "csv_path": worker.csv_path,
            "workflow_mode": "refine",
            "spec": {"workflow_mode": "refine"},
            "phase": "refine",
            "queue": [
                {"phase": "refine", "model": "lgb", "criterion": " balanced ", "horizon": 12, "tp_bps": 50.0, "sl_bps": 50.0},
                {"phase": "refine", "model": "lgb", "criterion": "profit_first", "horizon": 16, "tp_bps": 55.0, "sl_bps": 50.0},
                {"phase": "refine", "model": "hgbt", "criterion": "balanced", "horizon": 20, "tp_bps": 60.0, "sl_bps": 55.0},
            ],
            "queue_idx": 2,
            "results": [
                {
                    "phase": "refine",
                    "model": "LGB",
                    "criterion": "balanced",
                    "horizon": 12,
                    "tp_bps": 50.0,
                    "sl_bps": 50.0,
                    "status": "ok",
                    "created_at": "2026-05-20T09:00:00Z",
                },
                {
                    "phase": "refine",
                    "model": "lgb",
                    "criterion": "balanced",
                    "horizon": 12,
                    "tp_bps": 50.0,
                    "sl_bps": 50.0,
                    "status": "rejected",
                    "created_at": "2026-05-20T09:05:00Z",
                },
                {
                    "phase": "refine",
                    "model": "lgb",
                    "criterion": "profit_first",
                    "horizon": 16,
                    "tp_bps": 55.0,
                    "sl_bps": 50.0,
                    "status": "ok",
                    "created_at": "2026-05-20T09:10:00Z",
                },
            ],
            "stopped": False,
            "completed": False,
        }
        worker._save_state(state)

        resumed_state, resumed = worker._load_or_init_state()

        assert resumed is True
        assert resumed_state["queue_idx"] == 0
        assert len(resumed_state["results"]) == 2
        assert resumed_state["results"][0]["status"] == "ok"
        assert len(resumed_state["queue"]) == 1
        assert resumed_state["queue"][0]["model"] == "hgbt"
        assert worker._last_reconciled_duplicate_results_count == 1
        assert worker._last_reconciled_pruned_queue_count == 2


def test_auto_search_worker_skips_duplicate_candidate_in_run(monkeypatch, tmp_path):
        worker = AutoSearchWorker(
            csv_path="tv_GC_COMEX_5m_sample.csv",
            holdout_pct=0.1,
            holdout_min_bars=1000,
            holdout_max_bars=6000,
            training_profiles={"refine": {}},
            candidate_top_n=5,
            candidate_fresh_ratio=0.3,
            state_path=(tmp_path / "state.json").as_posix(),
            search_profile="refine",
        )

        duplicate_cfg = {
            "phase": "refine",
            "model": "lgb",
            "criterion": "balanced",
            "horizon": 12,
            "tp_bps": 50.0,
            "sl_bps": 50.0,
        }
        next_cfg = {
            "phase": "refine",
            "model": "hgbt",
            "criterion": "balanced",
            "horizon": 16,
            "tp_bps": 55.0,
            "sl_bps": 50.0,
        }
        state = {
            "spec": {"workflow_mode": "refine"},
            "phase": "refine",
            "queue": [duplicate_cfg, next_cfg],
            "queue_idx": 0,
            "results": [dict(duplicate_cfg, status="ok")],
            "stopped": False,
            "completed": False,
        }
        saved_states: list[dict[str, object]] = []
        messages: list[str] = []
        trained: list[dict[str, object]] = []

        monkeypatch.setattr(worker, "_load_or_init_state", lambda: (state, True))
        monkeypatch.setattr(worker, "_save_state", lambda payload: saved_states.append(dict(payload)))
        monkeypatch.setattr(worker, "_finalize_workflow", lambda payload: None)

        def _fake_train_one(cfg):
            trained.append(dict(cfg))
            return dict(cfg, status="ok")

        monkeypatch.setattr(worker, "_train_one", _fake_train_one)
        worker.message.connect(messages.append)

        worker.run()

        assert trained == [next_cfg]
        assert any("skip duplicate candidate" in msg for msg in messages)
        assert saved_states[0]["queue_idx"] == 1


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

        assert payload["source_csv_path"] == worker.csv_path
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

        assert payload["source_csv_path"] == worker.csv_path
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
            "spec": {
                "source_artifact": "C:/artifact_dir/approved_shortlist.json",
                "source_artifact_kind": "shortlist",
                "source_shortlist": "C:/artifact_dir/approved_shortlist.json",
                "target_csv_path": "tv_GC_COMEX_5m_target.csv",
            },
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

        assert payload["source_csv_path"] == worker.csv_path
        assert payload["target_csv_path"] == "tv_GC_COMEX_5m_target.csv"
        assert payload["source_artifact"] == "C:/artifact_dir/approved_shortlist.json"
        assert payload["source_artifact_kind"] == "shortlist"
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
            "fee_per_trade": 2.5,
            "slippage_bps": 4.0,
        },
    )

    worker.run()

    assert captured["quality_min_side_prediction_share"] == 0.07
    assert captured["quality_min_side_prediction_count"] == 11
    assert captured["fee_per_trade"] == 2.5
    assert captured["slippage_bps"] == 4.0


def test_auto_search_worker_interrupts_current_candidate_when_stop_is_requested(monkeypatch, tmp_path):
    from ibkr_trading_bot.gui import tab_model_training as tab_model_training_module

    worker = AutoSearchWorker(
        csv_path="tv_GC_COMEX_5m_sample.csv",
        holdout_pct=0.1,
        holdout_min_bars=1000,
        holdout_max_bars=6000,
        training_profiles={"explore": {}},
        candidate_top_n=5,
        candidate_fresh_ratio=0.3,
        state_path=(tmp_path / "state.json").as_posix(),
        search_profile="explore",
    )

    state = {
        "spec": {"workflow_mode": "explore"},
        "phase": "explore",
        "queue": [
            {
                "phase": "explore",
                "model": "lgb",
                "criterion": "balanced",
                "horizon": 12,
                "tp_bps": 50.0,
                "sl_bps": 50.0,
            }
        ],
        "queue_idx": 0,
        "results": [],
        "stopped": False,
        "completed": False,
    }
    saved_states: list[dict[str, object]] = []
    finished_states: list[tuple[str, bool]] = []
    messages: list[str] = []

    monkeypatch.setattr(worker, "_load_or_init_state", lambda: (state, True))
    monkeypatch.setattr(worker, "_save_state", lambda payload: saved_states.append(dict(payload)))

    def _fake_run_training_job(**kwargs):
        should_continue = kwargs.get("should_continue")
        assert callable(should_continue)
        worker.request_stop()
        assert should_continue() is False
        raise InterruptedError("training cancelled by caller")

    monkeypatch.setattr(tab_model_training_module, "run_training_job", _fake_run_training_job)
    worker.finished_state.connect(lambda path, completed: finished_states.append((path, completed)))
    worker.message.connect(messages.append)

    worker.run()

    assert saved_states[-1]["queue_idx"] == 0
    assert saved_states[-1]["stopped"] is True
    assert saved_states[-1]["completed"] is False
    assert finished_states == [((tmp_path / "state.json").as_posix(), False)]
    assert any("stop acknowledged" in msg for msg in messages)


def test_auto_search_worker_refresh_dispatches_run_training_job_to_target_csv(monkeypatch, tmp_path):
    from ibkr_trading_bot.gui import tab_model_training as tab_model_training_module

    captured: dict[str, object] = {}

    def _fake_run_training_job(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(tab_model_training_module, "run_training_job", _fake_run_training_job)

    worker = AutoSearchWorker(
        csv_path="tv_GC_COMEX_5m_source.csv",
        holdout_pct=0.1,
        holdout_min_bars=1000,
        holdout_max_bars=6000,
        training_profiles={"refresh": {}},
        candidate_top_n=5,
        candidate_fresh_ratio=0.3,
        state_path=(tmp_path / "state.json").as_posix(),
        refresh_csv_path="tv_GC_COMEX_5m_target.csv",
        search_profile="refresh",
    )

    result = worker._train_one(
        {
            "phase": "refresh",
            "model": "lgb",
            "criterion": "balanced",
            "horizon": 12,
            "tp_bps": 50.0,
            "sl_bps": 50.0,
        }
    )

    assert result["status"] == "ok"
    assert captured["csv_path"] == "tv_GC_COMEX_5m_target.csv"
    assert captured["phase"] == "refresh"


def test_auto_search_worker_resume_logs_continue_from_completed_position(monkeypatch, tmp_path):
    worker = AutoSearchWorker(
        csv_path="tv_GC_COMEX_5m_sample.csv",
        holdout_pct=0.1,
        holdout_min_bars=1000,
        holdout_max_bars=6000,
        training_profiles={"refine": {}},
        candidate_top_n=5,
        candidate_fresh_ratio=0.3,
        state_path=(tmp_path / "state.json").as_posix(),
        search_profile="refine",
    )

    state = {
        "spec": {"workflow_mode": "refine"},
        "phase": "refine",
        "queue": [
            {
                "phase": "refine",
                "model": "lgb",
                "criterion": "balanced",
                "horizon": 12,
                "tp_bps": 50.0,
                "sl_bps": 55.0,
            }
        ],
        "queue_idx": 0,
        "results": [
            {
                "phase": "refine",
                "model": "lgb",
                "criterion": "balanced",
                "horizon": 12,
                "tp_bps": 50.0,
                "sl_bps": 50.0,
                "status": "ok",
            }
        ],
        "stopped": False,
        "completed": False,
    }
    messages: list[str] = []
    saved_states: list[dict[str, object]] = []
    finished_states: list[tuple[str, bool]] = []

    monkeypatch.setattr(worker, "_load_or_init_state", lambda: (state, True))
    monkeypatch.setattr(worker, "_save_state", lambda payload: saved_states.append(dict(payload)))
    monkeypatch.setattr(worker, "_finalize_workflow", lambda payload: None)
    monkeypatch.setattr(
        worker,
        "_train_one",
        lambda cfg: {
            "phase": "refine",
            "model": cfg["model"],
            "criterion": cfg["criterion"],
            "horizon": cfg["horizon"],
            "tp_bps": cfg["tp_bps"],
            "sl_bps": cfg["sl_bps"],
            "status": "ok",
        },
    )
    worker.message.connect(messages.append)
    worker.finished_state.connect(lambda path, completed: finished_states.append((path, completed)))

    worker.run()

    assert any("queue=1/2" in msg for msg in messages)
    assert any("Workflow run [2/2]" in msg for msg in messages)
    assert saved_states[-1]["completed"] is True
    assert finished_states == [((tmp_path / "state.json").as_posix(), True)]