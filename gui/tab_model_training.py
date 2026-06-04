# gui/tab_model_training.py
import json as jsonlib
import os
import re
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ibkr_trading_bot.core.services.dataset_service import DatasetService
from ibkr_trading_bot.core.services.futures_roll_chain_service import read_dataset_sidecar_meta
from ibkr_trading_bot.core.services.model_training_service import (
    candidate_selection_criterion_from_meta,
    canonical_workflow_mode,
    compatibility_training_mode,
    compute_holdout_bars as runtime_compute_holdout_bars,
    dataset_snapshot_signature_from_csv,
    name_and_meta_from_csv as runtime_name_and_meta_from_csv,
    run_training_job,
    training_profile_for_mode,
)
from ibkr_trading_bot.model.train_models import (
    HAS_OPTUNA,
    _align_X_for_estimator,
    _call_with_feature_name_warning_suppressed,
    _model_dir,
    _select_feature_columns,
    _ternary_predict_mapped,
    train_and_evaluate_model,
)


LAST_TRAINING_CSV_PATH_KEY = "last_training_csv_path"
TRAINING_FEE_PER_TRADE_KEY = "training_fee_per_trade"
TRAINING_SLIPPAGE_BPS_KEY = "training_slippage_bps"
REFINE_SOURCE_ARTIFACT_PATH_KEY = "refine_source_artifact_path"
REFRESH_SOURCE_ARTIFACT_PATH_KEY = "refresh_source_artifact_path"
REFRESH_TARGET_CSV_PATH_KEY = "refresh_target_csv_path"
AUTO_SEARCH_PROFILE_ALIASES = {
    "fast": "refine",
    "full": "explore",
    "weekly": "refresh",
}
AUTO_SEARCH_PROFILE_VALUES = {"explore", "refine", "refresh"}


def _normalize_auto_search_profile(profile: str) -> str:
    txt = str(profile or "").strip().lower()
    if txt in AUTO_SEARCH_PROFILE_VALUES:
        return txt
    return AUTO_SEARCH_PROFILE_ALIASES.get(txt, "explore")


def _normalized_resume_path(path: Any) -> str:
    txt = str(path or "").strip()
    if not txt:
        return ""
    expanded = os.path.expanduser(txt)
    return os.path.normcase(os.path.normpath(expanded))


def _auto_search_state_score(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return (-1, -1, -1)

    queue_idx = 0
    completed = False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip().lower()
                match = re.search(r'"(?:queue_idx|quick_idx)"\s*:\s*(\d+)', line)
                if match:
                    queue_idx = max(queue_idx, int(match.group(1)))
                if '"completed"' in line:
                    completed = "true" in line
    except Exception:
        pass

    try:
        size = int(path.stat().st_size)
    except Exception:
        size = 0
    return (0 if completed else 1, int(queue_idx), int(size))


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h_l = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def label_triple_barrier(df: pd.DataFrame, K: int, tp_atr: float, sl_atr: float) -> pd.Series:
    """
    Returns {-1, 0, +1}:
      +1 = TP touched first, -1 = SL touched first, 0 = timeout.
    """
    close = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    n = len(df)
    y = np.zeros(n, dtype=np.int8)

    for i in range(n):
        jmax = min(n - 1, i + K)
        base = close[i]
        tp = base + tp_atr * atr[i]
        sl = base - sl_atr * atr[i]

        hit = 0
        for j in range(i + 1, jmax + 1):
            hi = df["high"].iat[j]
            lo = df["low"].iat[j]
            if hi >= tp:
                hit = +1
                break
            if lo <= sl:
                hit = -1
                break
        y[i] = hit
    return pd.Series(y, index=df.index, name="y_tb")


class TrainWorker(QThread):
    progress = Signal(int, int, dict, float, float)
    phase = Signal(str)
    model_ready = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        df_full: pd.DataFrame,
        holdout_bars: int,
        estimator: str,
        name_prefix: str,
        meta_extra: dict,
        holdout_pct: float | None = None,
        holdout_min_bars: int | None = None,
        holdout_max_bars: int | None = None,
        training_profile: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.df_full = df_full
        self.holdout_bars = int(holdout_bars)
        self.holdout_pct = (float(holdout_pct) if holdout_pct is not None else None)
        self.holdout_min_bars = (int(holdout_min_bars) if holdout_min_bars is not None else None)
        self.holdout_max_bars = (int(holdout_max_bars) if holdout_max_bars is not None else None)
        self.estimator = estimator
        self.name_prefix = name_prefix
        self.meta_extra = meta_extra
        self.training_profile = dict(training_profile or {})

    def run(self):
        try:
            def cb(idx, total, params, mean_f1, std_f1):
                self.progress.emit(int(idx), int(total), dict(params), float(mean_f1), float(std_f1))

            profile = dict(self.training_profile or {})
            n_splits = int(profile.get("n_splits", 5))
            top_k_features = int(profile.get("top_k_features", 12))
            mc_enabled = bool(profile.get("mc_enabled", True))
            mc_iters = int(profile.get("mc_iters", 200))
            quality_gate_enabled = bool(profile.get("quality_gate_enabled", True))
            quality_gate_hard_reject = bool(profile.get("quality_gate_hard_reject", True))
            quality_min_trades = int(profile.get("quality_min_trades", 8))
            quality_min_side_recall = float(profile.get("quality_min_side_recall", 0.01))
            quality_min_side_prediction_share = float(profile.get("quality_min_side_prediction_share", 0.0))
            quality_min_side_prediction_count = int(profile.get("quality_min_side_prediction_count", 0))
            quality_require_mc_nonnegative = bool(profile.get("quality_require_mc_nonnegative", True))
            quality_min_mc_sharpe_p50 = float(profile.get("quality_min_mc_sharpe_p50", -0.02))
            quality_min_profit_net = float(profile.get("quality_min_profit_net", 0.0))
            quality_min_holdout_sharpe = float(profile.get("quality_min_holdout_sharpe", 0.0))
            fee_per_trade = float(profile.get("fee_per_trade", 0.0))
            slippage_bps = float(profile.get("slippage_bps", 0.0))
            training_mode = str(
                profile.get("training_mode")
                or profile.get("compatibility_mode")
                or "standard"
            ).strip().lower()
            candidate_chain_enabled = bool(profile.get("candidate_chain_enabled", True))
            candidate_selection_criterion = str(
                profile.get("candidate_selection_criterion", "balanced")
            ).strip().lower()
            candidate_top_n = int(profile.get("candidate_top_n", 5))
            candidate_fresh_ratio = float(profile.get("candidate_fresh_ratio", 0.30))
            max_param_candidates = profile.get("max_param_candidates")
            try:
                max_param_candidates = int(max_param_candidates) if max_param_candidates is not None else None
                if max_param_candidates is not None and max_param_candidates <= 0:
                    max_param_candidates = None
            except Exception:
                max_param_candidates = None
            param_sample_seed = int(profile.get("param_sample_seed", 42))
            mc_block_len = int(profile.get("mc_block_len", self.meta_extra.get("mc_block_len", 100)))
            search_backend = profile.get("search_backend", "grid")
            optuna_trials = profile.get("optuna_trials")
            optuna_timeout_seconds = profile.get("optuna_timeout_seconds")
            self.phase.emit(str(search_backend))

            train_and_evaluate_model(
                self.df_full,
                estimator_name=self.estimator,
                param_grid=None,
                on_progress=cb,
                n_splits=n_splits,
                holdout_bars=self.holdout_bars,
                holdout_pct=self.holdout_pct,
                holdout_min_bars=self.holdout_min_bars,
                holdout_max_bars=self.holdout_max_bars,
                name_prefix=self.name_prefix,
                meta_extra=self.meta_extra,
                mc_enabled=mc_enabled,
                mc_iters=mc_iters,
                mc_block_len=mc_block_len,
                annualize_sharpe=True,
                top_k_features=top_k_features,
                feature_stability_threshold=profile.get("feature_stability_threshold"),
                label_lookahead_bars=int(self.meta_extra.get("label_lookahead_bars", 0)),
                quality_gate_enabled=quality_gate_enabled,
                quality_gate_hard_reject=quality_gate_hard_reject,
                quality_min_trades=quality_min_trades,
                quality_min_side_recall=quality_min_side_recall,
                quality_min_side_prediction_share=quality_min_side_prediction_share,
                quality_min_side_prediction_count=quality_min_side_prediction_count,
                quality_require_mc_nonnegative=quality_require_mc_nonnegative,
                quality_min_mc_sharpe_p50=quality_min_mc_sharpe_p50,
                quality_min_profit_net=quality_min_profit_net,
                quality_min_holdout_sharpe=quality_min_holdout_sharpe,
                fee_per_trade=fee_per_trade,
                slippage_bps=slippage_bps,
                max_param_candidates=max_param_candidates,
                param_sample_seed=param_sample_seed,
                search_backend=search_backend,
                optuna_trials=optuna_trials,
                optuna_timeout_seconds=optuna_timeout_seconds,
                training_mode=training_mode,
                candidate_chain_enabled=candidate_chain_enabled,
                candidate_selection_criterion=candidate_selection_criterion,
                candidate_top_n=candidate_top_n,
                candidate_fresh_ratio=candidate_fresh_ratio,
            )

            self.phase.emit("save")
            out_dir = Path(_model_dir())
            patt = f"{self.name_prefix}_{self.estimator}_*.pkl" if self.name_prefix else f"{self.estimator}_*.pkl"
            files = sorted(out_dir.glob(patt), key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = files[0].as_posix() if files else ""
            self.model_ready.emit(model_path)
            self.phase.emit("done")
        except Exception as e:
            self.error.emit(str(e))
            self.phase.emit("error")


class AutoSearchWorker(QThread):
    """Long-running auto search with checkpoint + resume."""

    message = Signal(str)
    result = Signal(dict)
    finished_state = Signal(str, bool)  # state_path, completed
    error = Signal(str)

    def __init__(
        self,
        *,
        csv_path: str,
        holdout_pct: float,
        holdout_min_bars: int,
        holdout_max_bars: int,
        training_profiles: dict[str, dict[str, Any]],
        candidate_top_n: int,
        candidate_fresh_ratio: float,
        state_path: str,
        source_artifact_path: str | None = None,
        refresh_csv_path: str | None = None,
        search_profile: str = "explore",
    ):
        super().__init__()
        self.csv_path = str(csv_path)
        self.holdout_pct = float(holdout_pct)
        self.holdout_min_bars = int(holdout_min_bars)
        self.holdout_max_bars = int(holdout_max_bars)
        self.training_profiles = {
            str(k): dict(v or {}) for k, v in (training_profiles or {}).items()
        }
        self.candidate_top_n = int(max(1, candidate_top_n))
        self.candidate_fresh_ratio = float(np.clip(candidate_fresh_ratio, 0.05, 0.80))
        self.state_path = Path(state_path)
        self.source_artifact_path = str(source_artifact_path or "").strip()
        self.refresh_csv_path = str(refresh_csv_path or "").strip()
        self.workflow_mode = self._normalize_search_profile(search_profile)
        self._stop_requested = False
        self._last_recovered_results_count = 0
        self._last_reconciled_duplicate_results_count = 0
        self._last_reconciled_pruned_queue_count = 0

    def request_stop(self):
        self._stop_requested = True

    @staticmethod
    def _now_str() -> str:
        return str(pd.Timestamp.now(tz="UTC").isoformat())

    @staticmethod
    def _compute_holdout_bars(n_total: int, pct: float, min_bars: int, max_bars: int) -> int:
        n = int(max(0, n_total))
        n_hold = int(round(float(n) * float(np.clip(pct, 0.0, 0.95))))
        n_hold = max(int(min_bars), n_hold)
        n_hold = min(int(max_bars), n_hold)
        n_hold = min(max(0, n_hold), max(n - 50, 0))
        return int(n_hold)

    @staticmethod
    def _name_and_meta_from_csv(path: str, n_total: int, n_train: int, n_hold: int) -> tuple[str, dict[str, Any]]:
        base = os.path.basename(path)
        instrument, exchange, timeframe = ("UNKNOWN", "UNK", "UNK")
        m = re.match(r"tv_([^_]+)_([^_]+)_([^_]+)_.+\.csv$", base)
        if m:
            instrument, exchange, timeframe = m.group(1), m.group(2), m.group(3)
        else:
            m = re.match(r"([A-Z0-9]+)_([0-9]+m|[0-9]+h|[0-9]+d)_(.+\.csv)$", base)
            if m:
                instrument = m.group(1)
                timeframe = m.group(2)
                exchange = "COMEX"
        name_prefix = f"{instrument}_{exchange}_{timeframe}_{n_total}bars"
        meta_extra = {
            "instrument": instrument,
            "exchange": exchange,
            "timeframe": timeframe,
            "n_total_bars": int(n_total),
            "n_train_bars": int(n_train),
            "n_holdout_bars": int(n_hold),
        }
        return name_prefix, meta_extra

    @staticmethod
    def _normalize_search_profile(profile: str) -> str:
        return _normalize_auto_search_profile(profile)

    @staticmethod
    def _same_csv_path(left: Any, right: Any) -> bool:
        return _normalized_resume_path(left) == _normalized_resume_path(right)

    def _artifact_stem(self) -> str:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(self.csv_path).stem).strip("_")
        return stem or "dataset"

    def _artifact_dir(self) -> Path:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        return self.state_path.parent

    def _explicit_source_artifact(self) -> Path | None:
        path_txt = str(self.source_artifact_path or "").strip()
        if not path_txt:
            return None
        return Path(path_txt)

    def _target_csv_path_for_phase(self, phase: str) -> str:
        if str(phase or "").strip().lower() == "refresh":
            refresh_csv = str(self.refresh_csv_path or "").strip()
            if refresh_csv:
                return refresh_csv
        return self.csv_path

    @staticmethod
    def _refresh_candidates_from_shortlist_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
        refresh_candidates: list[dict[str, Any]] = []
        for candidate in list(payload.get("candidates") or []):
            refresh_candidates.append(
                {
                    "enabled": True,
                    "candidate_id": candidate.get("candidate_id"),
                    "model": candidate.get("model"),
                    "criterion": candidate.get("criterion"),
                    "horizon": candidate.get("horizon"),
                    "tp_bps": candidate.get("tp_bps"),
                    "sl_bps": candidate.get("sl_bps"),
                }
            )
        return refresh_candidates

    def _region_summary_path(self) -> Path:
        return self._artifact_dir() / f"{self._artifact_stem()}_region_summary.json"

    def _shortlist_path(self) -> Path:
        return self._artifact_dir() / f"{self._artifact_stem()}_shortlist.json"

    def _refresh_set_path(self) -> Path:
        return self._artifact_dir() / f"{self._artifact_stem()}_refresh_set.json"

    def _refine_source_path(self) -> Path:
        explicit = self._explicit_source_artifact()
        if explicit is not None:
            return explicit
        return self._region_summary_path()

    def _refresh_source_payload(self) -> tuple[list[dict[str, Any]], Path | None, str]:
        explicit = self._explicit_source_artifact()
        if explicit is not None:
            payload = self._load_json_file(explicit)
            if "refresh_candidates" in payload:
                return list(payload.get("refresh_candidates") or []), explicit, "refresh_set"
            return self._refresh_candidates_from_shortlist_payload(payload), explicit, "shortlist"

        refresh_set_path = self._refresh_set_path()
        shortlist_path = self._shortlist_path()
        if refresh_set_path.exists():
            payload = self._load_json_file(refresh_set_path)
            return list(payload.get("refresh_candidates") or []), refresh_set_path, "refresh_set"
        if shortlist_path.exists():
            payload = self._load_json_file(shortlist_path)
            return self._refresh_candidates_from_shortlist_payload(payload), shortlist_path, "shortlist"
        return [], None, ""

    @staticmethod
    def _load_json_file(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path.as_posix())
        payload = jsonlib.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSON artifact: {path.as_posix()}")
        return payload

    @staticmethod
    def _grid_values(min_value: float, max_value: float, step: float) -> list[float]:
        lo = float(min(min_value, max_value))
        hi = float(max(min_value, max_value))
        step_value = float(max(step, 1.0))
        values: list[float] = []
        current = lo
        while current <= hi + 1e-9:
            values.append(round(current, 6))
            current += step_value
        return values or [round(lo, 6)]

    @staticmethod
    def _neighbor_values(center: int, ordered_values: list[int], radius: int) -> list[int]:
        values = [int(v) for v in ordered_values]
        if center not in values:
            values.append(int(center))
            values = sorted(set(values))
        if not values:
            return [int(center)]
        idx = values.index(int(center))
        lo = max(0, idx - int(max(0, radius)))
        hi = min(len(values), idx + int(max(0, radius)) + 1)
        return values[lo:hi]

    @staticmethod
    def _unique_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str, int, float, float]] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            key = AutoSearchWorker._candidate_key(row)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    @staticmethod
    def _candidate_key(row: dict[str, Any]) -> tuple[str, str, str, int, float, float]:
        return (
            str(row.get("phase") or ""),
            str(row.get("model") or "").strip().lower(),
            str(row.get("criterion") or "balanced").strip().lower(),
            int(row.get("horizon") or 0),
            float(row.get("tp_bps") or 0.0),
            float(row.get("sl_bps") or 0.0),
        )

    @staticmethod
    def _collect_csv_paths(payload: Any) -> set[str]:
        paths: set[str] = set()
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key == "csv_path":
                        normalized = _normalized_resume_path(value)
                        if normalized:
                            paths.add(normalized)
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(item, list):
                for value in item:
                    if isinstance(value, (dict, list)):
                        stack.append(value)
        return paths

    @staticmethod
    def _workflow_progress_counts(state: dict[str, Any]) -> tuple[int, int]:
        queue = list(state.get("queue") or []) if isinstance(state, dict) else []
        try:
            queue_idx = int(state.get("queue_idx", 0) or 0) if isinstance(state, dict) else 0
        except Exception:
            queue_idx = 0
        queue_idx = max(0, min(queue_idx, len(queue)))
        completed_count = len(list(state.get("results") or [])) if isinstance(state, dict) else 0
        completed_count = max(completed_count, queue_idx)
        total_count = completed_count + max(0, len(queue) - queue_idx)
        return int(completed_count), int(total_count)

    def _queue_from_spec(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        mode = str(spec.get("workflow_mode") or self.workflow_mode)
        queue: list[dict[str, Any]] = []
        if mode == "explore":
            for model_name, horizon, tp_bps, sl_bps in product(
                spec.get("models") or ["lgb", "hgbt"],
                spec.get("label_horizon_bars") or [8, 12, 16],
                spec.get("label_tp_bps") or [40.0, 50.0, 60.0],
                spec.get("label_sl_bps") or [40.0, 50.0, 60.0],
            ):
                queue.append(
                    {
                        "phase": "explore",
                        "model": str(model_name),
                        "criterion": str(spec.get("coarse_criterion") or "balanced"),
                        "horizon": int(horizon),
                        "tp_bps": float(tp_bps),
                        "sl_bps": float(sl_bps),
                    }
                )
            return self._unique_queue(queue)

        if mode == "refine":
            regions = list(spec.get("approved_regions") or [])
            criteria = list(spec.get("criteria") or ["balanced", "profit_first", "robustness_first", "recall_balance"])
            fine_step = float(spec.get("fine_step_bps") or 5.0)
            for region in regions:
                models = list(region.get("models") or [])
                horizons = list(region.get("horizon_values") or [])
                tp_values = self._grid_values(
                    float(region.get("tp_bps_min") or 50.0),
                    float(region.get("tp_bps_max") or 50.0),
                    fine_step,
                )
                sl_values = self._grid_values(
                    float(region.get("sl_bps_min") or 50.0),
                    float(region.get("sl_bps_max") or 50.0),
                    fine_step,
                )
                for model_name, criterion, horizon, tp_bps, sl_bps in product(models, criteria, horizons, tp_values, sl_values):
                    queue.append(
                        {
                            "phase": "refine",
                            "model": str(model_name),
                            "criterion": str(criterion),
                            "horizon": int(horizon),
                            "tp_bps": float(tp_bps),
                            "sl_bps": float(sl_bps),
                        }
                    )
            return self._unique_queue(queue)

        for candidate in list(spec.get("refresh_candidates") or []):
            if not bool(candidate.get("enabled", True)):
                continue
            queue.append(
                {
                    "phase": "refresh",
                    "model": str(candidate.get("model") or "lgb"),
                    "criterion": str(candidate.get("criterion") or "balanced"),
                    "horizon": int(candidate.get("horizon") or 12),
                    "tp_bps": float(candidate.get("tp_bps") or 50.0),
                    "sl_bps": float(candidate.get("sl_bps") or 50.0),
                }
            )
        return self._unique_queue(queue)

    def _build_spec(self) -> dict[str, Any]:
        if self.workflow_mode == "explore":
            return {
                "version": 2,
                "workflow_mode": "explore",
                "models": ["lgb", "hgbt"],
                "coarse_criterion": "balanced",
                "label_horizon_bars": [8, 12, 16, 20],
                "label_tp_bps": [30.0, 40.0, 50.0, 60.0, 80.0],
                "label_sl_bps": [30.0, 40.0, 50.0, 60.0, 80.0],
                "promote_top_k": 3,
                "approved_model_top_k": 2,
                "fine_step_bps": 5.0,
                "region_horizon_neighbors": 1,
                "region_tp_band_bps": 10.0,
                "region_sl_band_bps": 10.0,
                "refine_criteria": ["balanced", "profit_first", "robustness_first", "recall_balance"],
            }
        if self.workflow_mode == "refine":
            source_region_summary_path = self._refine_source_path()
            region_summary = self._load_json_file(source_region_summary_path)
            approved_regions = list(region_summary.get("approved_regions") or [])
            if not approved_regions:
                raise ValueError(
                    f"Refine vyzaduje region_summary s approved_regions: {source_region_summary_path.as_posix()}"
                )
            return {
                "version": 2,
                "workflow_mode": "refine",
                "source_region_summary": source_region_summary_path.as_posix(),
                "approved_regions": approved_regions,
                "criteria": ["balanced", "profit_first", "robustness_first", "recall_balance"],
                "fine_step_bps": 5.0,
                "shortlist_top_k": int(max(1, self.candidate_top_n)),
            }
        refresh_candidates, source_artifact_path, source_artifact_kind = self._refresh_source_payload()
        if not refresh_candidates:
            refresh_set_path = self._refresh_set_path()
            shortlist_path = self._shortlist_path()
            raise ValueError(
                f"Refresh vyzaduje refresh_set nebo shortlist: {refresh_set_path.as_posix()} / {shortlist_path.as_posix()}"
            )
        target_csv_path = self._target_csv_path_for_phase("refresh")
        return {
            "version": 2,
            "workflow_mode": "refresh",
            "source_artifact": source_artifact_path.as_posix() if source_artifact_path is not None else "",
            "source_artifact_kind": source_artifact_kind,
            "source_shortlist": source_artifact_path.as_posix() if source_artifact_path is not None else "",
            "target_csv_path": target_csv_path,
            "refresh_candidates": refresh_candidates[: int(max(1, self.candidate_top_n))],
        }

    def _new_state(self) -> dict[str, Any]:
        spec = self._build_spec()
        queue = self._queue_from_spec(spec)
        explicit_source_artifact = self._explicit_source_artifact()
        return {
            "version": 2,
            "created_at": self._now_str(),
            "updated_at": self._now_str(),
            "csv_path": self.csv_path,
            "source_artifact_path": explicit_source_artifact.as_posix() if explicit_source_artifact is not None else "",
            "refresh_csv_path": str(self.refresh_csv_path or ""),
            "workflow_mode": self.workflow_mode,
            "spec": spec,
            "phase": self.workflow_mode,
            "queue": queue,
            "queue_idx": 0,
            "results": [],
            "stopped": False,
            "completed": False,
        }

    def _recover_result_from_meta_path(
        self,
        meta_path: Path,
        *,
        expected_csv_path: str,
        expected_n_total: int,
    ) -> dict[str, Any] | None:
        try:
            meta_obj = jsonlib.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(meta_obj, dict):
            return None

        workflow_raw = str(
            meta_obj.get("workflow_mode")
            or meta_obj.get("training_mode_requested")
            or meta_obj.get("training_mode")
            or ""
        ).strip().lower()
        workflow_mode = canonical_workflow_mode(workflow_raw) if workflow_raw else ""
        if workflow_mode != self.workflow_mode:
            return None

        try:
            n_total = int(meta_obj.get("n_total_bars") or 0)
        except Exception:
            return None
        if n_total != int(expected_n_total):
            return None

        csv_paths = self._collect_csv_paths(meta_obj)
        if expected_csv_path not in csv_paths:
            return None

        estimator_name = str(meta_obj.get("estimator_name") or "").strip().lower()
        if not estimator_name:
            return None
        criterion = candidate_selection_criterion_from_meta(meta_obj, default="balanced")
        try:
            horizon = int(meta_obj.get("label_horizon_bars") or meta_obj.get("label_lookahead_bars") or 0)
            tp_bps = float(meta_obj.get("label_take_profit_bps") or 0.0)
            sl_bps = float(meta_obj.get("label_stop_loss_bps") or 0.0)
        except Exception:
            return None
        if horizon <= 0:
            return None

        metrics = self._meta_metrics(meta_obj)
        quality_gate = meta_obj.get("quality_gate") if isinstance(meta_obj.get("quality_gate"), dict) else {}
        qg_reasons = list(quality_gate.get("reasons") or [])
        trade_count = metrics.get("num_trades")
        if trade_count is None:
            raw_trades = metrics.get("trades")
            if isinstance(raw_trades, list):
                trade_count = len(raw_trades)
            else:
                trade_count = raw_trades

        raw_status = str(meta_obj.get("status") or "").strip().lower()
        status = "ok"
        if meta_path.name.endswith("_rejected_meta.json") or raw_status.startswith("rejected"):
            status = "rejected"
        elif raw_status == "error":
            status = "error"
        elif quality_gate.get("evaluated") and not bool(quality_gate.get("passed", False)):
            status = "rejected"

        training_profile = meta_obj.get("training_profile") if isinstance(meta_obj.get("training_profile"), dict) else {}
        search_plan = meta_obj.get("search_plan") if isinstance(meta_obj.get("search_plan"), dict) else {}
        compatibility_mode = str(
            meta_obj.get("training_mode_compatibility")
            or compatibility_training_mode(workflow_mode)
            or ""
        )
        runtime_training_mode = str(
            training_profile.get("training_mode")
            or compatibility_mode
            or workflow_mode
        )

        model_path = ""
        if meta_path.name.endswith("_meta.json") and not meta_path.name.endswith("_rejected_meta.json"):
            model_candidate = meta_path.with_name(meta_path.name[: -len("_meta.json")] + ".pkl")
            if model_candidate.exists():
                model_path = model_candidate.as_posix()

        return {
            "phase": workflow_mode,
            "workflow_mode": workflow_mode,
            "compatibility_mode": compatibility_mode,
            "runtime_training_mode": runtime_training_mode,
            "model": estimator_name,
            "criterion": criterion,
            "horizon": horizon,
            "tp_bps": tp_bps,
            "sl_bps": sl_bps,
            "status": status,
            "error": "" if status == "ok" else str(meta_obj.get("status") or raw_status),
            "model_path": model_path,
            "meta_path": meta_path.as_posix(),
            "meta_obj": meta_obj,
            "search_plan": dict(search_plan),
            "search_backend_requested": search_plan.get("search_backend_requested"),
            "search_backend_used": search_plan.get("search_backend_used"),
            "search_backend_fallback_reason": search_plan.get("search_backend_fallback_reason"),
            "profit_net": metrics.get("profit_net"),
            "sharpe": metrics.get("sharpe"),
            "pf": metrics.get("pf"),
            "trades": trade_count,
            "num_trades_short": metrics.get("num_trades_short"),
            "num_trades_long": metrics.get("num_trades_long"),
            "qg_reasons": qg_reasons,
            "created_at": str(meta_obj.get("created_at_iso") or self._now_str()),
        }

    def _recover_empty_state_from_artifacts(self, state: dict[str, Any]) -> dict[str, Any]:
        self._last_recovered_results_count = 0
        if not isinstance(state, dict):
            return state

        queue = list(state.get("queue") or [])
        if not queue:
            return state
        try:
            if int(state.get("queue_idx", 0) or 0) > 0:
                return state
        except Exception:
            return state
        if list(state.get("results") or []):
            return state

        seed_cfg = dict(queue[0])
        try:
            df = DatasetService().prepare_from_csv(
                self.csv_path,
                labeling="triple_barrier",
                target_mode="ternary",
                horizon=int(seed_cfg.get("horizon") or 12),
                take_profit_bps=float(seed_cfg.get("tp_bps") or 50.0),
                stop_loss_bps=float(seed_cfg.get("sl_bps") or 50.0),
                same_bar_policy="neutral",
            ).sort_values("timestamp").reset_index(drop=True)
        except Exception:
            return state

        n_total = int(len(df))
        if n_total <= 0:
            return state
        n_hold = self._compute_holdout_bars(
            n_total,
            self.holdout_pct,
            self.holdout_min_bars,
            self.holdout_max_bars,
        )
        name_prefix, _ = self._name_and_meta_from_csv(self.csv_path, n_total, max(0, n_total - n_hold), n_hold)

        queue_keys = {self._candidate_key(row) for row in queue}
        recovered_by_key: dict[tuple[str, str, str, int, float, float], dict[str, Any]] = {}
        recovered_mtime: dict[tuple[str, str, str, int, float, float], float] = {}
        expected_csv_path = _normalized_resume_path(self.csv_path)
        for meta_path in Path(_model_dir()).glob(f"{name_prefix}_*_meta.json"):
            row = self._recover_result_from_meta_path(
                meta_path,
                expected_csv_path=expected_csv_path,
                expected_n_total=n_total,
            )
            if row is None:
                continue
            key = self._candidate_key(row)
            if key not in queue_keys:
                continue
            try:
                mtime = float(meta_path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            if key not in recovered_by_key or mtime >= recovered_mtime.get(key, float("-inf")):
                recovered_by_key[key] = row
                recovered_mtime[key] = mtime

        if not recovered_by_key:
            return state

        recovered_results = [recovered_by_key[self._candidate_key(row)] for row in queue if self._candidate_key(row) in recovered_by_key]
        if not recovered_results:
            return state

        state["queue"] = [row for row in queue if self._candidate_key(row) not in recovered_by_key]
        state["results"] = recovered_results
        state["queue_idx"] = 0
        self._save_state(state)
        self._last_recovered_results_count = len(recovered_results)
        return state

    @staticmethod
    def _result_status_rank(row: dict[str, Any]) -> int:
        status = str(row.get("status") or "").strip().lower()
        if status == "ok":
            return 3
        if status == "rejected":
            return 2
        if status == "error":
            return 1
        return 0

    @staticmethod
    def _result_artifact_rank(row: dict[str, Any]) -> tuple[int, int]:
        meta_path = str(row.get("meta_path") or "").strip()
        model_path = str(row.get("model_path") or "").strip()
        meta_exists = int(bool(meta_path) and Path(meta_path).exists())
        model_exists = int(bool(model_path) and Path(model_path).exists())
        return meta_exists, model_exists

    @staticmethod
    def _result_created_sort_value(row: dict[str, Any]) -> str:
        return str(row.get("created_at") or "").strip()

    def _dedupe_result_rows(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        best_row_by_key: dict[tuple[str, str, str, int, float, float], dict[str, Any]] = {}
        best_rank_by_key: dict[tuple[str, str, str, int, float, float], tuple[int, tuple[int, int], str, int]] = {}
        best_index_by_key: dict[tuple[str, str, str, int, float, float], int] = {}

        for idx, row in enumerate(rows):
            key = self._candidate_key(row)
            rank = (
                self._result_status_rank(row),
                self._result_artifact_rank(row),
                self._result_created_sort_value(row),
                idx,
            )
            if key not in best_rank_by_key or rank > best_rank_by_key[key]:
                best_rank_by_key[key] = rank
                best_row_by_key[key] = row
                best_index_by_key[key] = idx

        deduped = [best_row_by_key[key] for _, key in sorted((idx, key) for key, idx in best_index_by_key.items())]
        duplicate_count = max(0, len(rows) - len(deduped))
        return deduped, duplicate_count

    def _reconcile_state_progress(self, state: dict[str, Any]) -> dict[str, Any]:
        self._last_reconciled_duplicate_results_count = 0
        self._last_reconciled_pruned_queue_count = 0
        if not isinstance(state, dict):
            return state

        queue = list(state.get("queue") or [])
        results = list(state.get("results") or [])
        deduped_results, duplicate_count = self._dedupe_result_rows(results)
        result_keys = {self._candidate_key(row) for row in deduped_results}

        try:
            queue_idx = int(state.get("queue_idx", 0) or 0)
        except Exception:
            queue_idx = 0
        queue_idx = max(0, min(queue_idx, len(queue)))

        should_prune_completed = bool(result_keys) and (queue_idx == 0 or len(deduped_results) >= queue_idx)
        pruned_queue_count = 0
        new_queue = queue
        new_queue_idx = queue_idx

        if should_prune_completed:
            new_queue = []
            for pos, row in enumerate(queue):
                if pos < queue_idx:
                    pruned_queue_count += 1
                    continue
                if self._candidate_key(row) in result_keys:
                    pruned_queue_count += 1
                    continue
                new_queue.append(row)
            new_queue_idx = 0

        state_changed = duplicate_count > 0 or pruned_queue_count > 0 or new_queue_idx != queue_idx
        if state_changed:
            state["results"] = deduped_results
            state["queue"] = new_queue
            state["queue_idx"] = new_queue_idx
            self._save_state(state)

        self._last_reconciled_duplicate_results_count = duplicate_count
        self._last_reconciled_pruned_queue_count = pruned_queue_count
        return state

    def _save_state(self, state: dict[str, Any]):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = self._now_str()
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(jsonlib.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _migrate_legacy_state(self, legacy_state: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(legacy_state, dict) or legacy_state.get("workflow_mode"):
            return None
        spec = legacy_state.get("spec") or {}
        legacy_profile = str(spec.get("search_profile") or "").strip().lower()
        migrated_mode = _normalize_auto_search_profile(legacy_profile)
        if legacy_profile not in AUTO_SEARCH_PROFILE_ALIASES or migrated_mode != self.workflow_mode:
            return None

        queue: list[dict[str, Any]] = []
        for row in list(legacy_state.get("quick_queue") or []):
            queue.append(
                {
                    "phase": migrated_mode,
                    "model": str(row.get("model") or "lgb"),
                    "criterion": str(row.get("criterion") or "balanced"),
                    "horizon": int(row.get("horizon") or 12),
                    "tp_bps": float(row.get("tp_bps") or 50.0),
                    "sl_bps": float(row.get("sl_bps") or 50.0),
                }
            )

        if migrated_mode == "explore":
            migrated_spec = self._build_spec()
        else:
            migrated_spec = {
                "version": 1,
                "workflow_mode": migrated_mode,
                "legacy_search_profile": legacy_profile,
                "migrated_queue_source": "quick_queue",
                "criteria": list(spec.get("criteria") or []),
                "label_horizon_bars": list(spec.get("label_horizon_bars") or []),
                "label_tp_bps": list(spec.get("label_tp_bps") or []),
                "label_sl_bps": list(spec.get("label_sl_bps") or []),
                "shortlist_top_k": int(max(1, self.candidate_top_n)),
            }

        migrated = {
            "version": 2,
            "created_at": legacy_state.get("created_at") or self._now_str(),
            "updated_at": self._now_str(),
            "csv_path": self.csv_path,
            "workflow_mode": migrated_mode,
            "spec": migrated_spec,
            "phase": migrated_mode,
            "queue": queue,
            "queue_idx": int(legacy_state.get("quick_idx", 0) or 0),
            "results": list(legacy_state.get("results") or []),
            "stopped": bool(legacy_state.get("stopped", False)),
            "completed": False,
            "migrated_from": legacy_profile,
        }
        self._save_state(migrated)
        return migrated

    def _load_or_init_state(self) -> tuple[dict[str, Any], bool]:
        self._last_recovered_results_count = 0
        self._last_reconciled_duplicate_results_count = 0
        self._last_reconciled_pruned_queue_count = 0
        if not self.state_path.exists():
            st = self._new_state()
            self._save_state(st)
            return st, False
        try:
            st = jsonlib.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            st = self._new_state()
            self._save_state(st)
            return st, False
        migrated = self._migrate_legacy_state(st)
        if migrated is not None:
            return migrated, True
        spec_expected = self._build_spec()
        state_has_progress = False
        if isinstance(st, dict):
            try:
                state_has_progress = int(st.get("queue_idx", 0) or 0) > 0
            except Exception:
                state_has_progress = False
            if not state_has_progress:
                state_has_progress = bool(st.get("results"))
        allow_spec_mismatch = bool(isinstance(st, dict) and st.get("migrated_from")) or state_has_progress
        if (
            not isinstance(st, dict)
            or not self._same_csv_path(st.get("csv_path"), self.csv_path)
            or str(st.get("workflow_mode") or "") != self.workflow_mode
            or st.get("phase") == "done"
        ):
            st = self._new_state()
            self._save_state(st)
            return st, False
        if st.get("spec") != spec_expected and not allow_spec_mismatch:
            st = self._new_state()
            self._save_state(st)
            return st, False
        if str(st.get("csv_path") or "") != self.csv_path:
            st["csv_path"] = self.csv_path
            self._save_state(st)
        st = self._recover_empty_state_from_artifacts(st)
        st = self._reconcile_state_progress(st)
        return st, True

    @staticmethod
    def _score_result(r: dict[str, Any]) -> tuple[float, float, float]:
        def _fv(k: str, d: float = float("-inf")) -> float:
            try:
                v = float(r.get(k, d))
                return float(v) if np.isfinite(v) else float(d)
            except Exception:
                return float(d)
        return (_fv("profit_net"), _fv("sharpe"), _fv("pf"))

    @staticmethod
    def _dataset_signature_from_row(row: dict[str, Any]) -> dict[str, Any]:
        meta = row.get("meta_obj") or {}
        if not isinstance(meta, dict):
            meta = {}
        return {
            "instrument": meta.get("instrument") or "UNKNOWN",
            "exchange": meta.get("exchange") or "UNK",
            "timeframe": meta.get("timeframe") or "UNK",
            "n_total_bars": int(meta.get("n_total_bars") or 0),
            "n_holdout_bars": int(meta.get("n_holdout_bars") or 0),
        }

    def _write_json_artifact(self, path: Path, payload: dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(jsonlib.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path.as_posix()

    def _write_region_summary(self, state: dict[str, Any]) -> str:
        spec = dict(state.get("spec") or {})
        ok_rows = [r for r in list(state.get("results") or []) if str(r.get("status")) == "ok"]
        ok_rows.sort(key=self._score_result, reverse=True)
        horizon_values = [int(v) for v in list(spec.get("label_horizon_bars") or [])]
        tp_values = [float(v) for v in list(spec.get("label_tp_bps") or [])]
        sl_values = [float(v) for v in list(spec.get("label_sl_bps") or [])]
        horizon_radius = int(max(0, int(spec.get("region_horizon_neighbors", 1))))
        tp_band = float(max(0.0, float(spec.get("region_tp_band_bps", 10.0))))
        sl_band = float(max(0.0, float(spec.get("region_sl_band_bps", 10.0))))
        approved_models: list[str] = []
        for row in ok_rows:
            model = str(row.get("model") or "")
            if model and model not in approved_models:
                approved_models.append(model)
            if len(approved_models) >= int(max(1, int(spec.get("approved_model_top_k", 2)))):
                break

        approved_regions: list[dict[str, Any]] = []
        seen_region_keys: set[tuple[str, int]] = set()
        for row in ok_rows:
            model = str(row.get("model") or "")
            horizon = int(row.get("horizon") or 0)
            region_key = (model, horizon)
            if model not in approved_models or region_key in seen_region_keys:
                continue
            seen_region_keys.add(region_key)
            horizon_neighbors = self._neighbor_values(horizon, horizon_values, horizon_radius)
            tp_center = float(row.get("tp_bps") or 0.0)
            sl_center = float(row.get("sl_bps") or 0.0)
            tp_min = max(min(tp_values or [tp_center]), tp_center - tp_band)
            tp_max = min(max(tp_values or [tp_center]), tp_center + tp_band)
            sl_min = max(min(sl_values or [sl_center]), sl_center - sl_band)
            sl_max = min(max(sl_values or [sl_center]), sl_center + sl_band)
            approved_regions.append(
                {
                    "region_id": (
                        f"{model}_h{min(horizon_neighbors)}_{max(horizon_neighbors)}"
                        f"_tp{int(tp_min)}_{int(tp_max)}_sl{int(sl_min)}_{int(sl_max)}"
                    ),
                    "models": [model],
                    "horizon_values": horizon_neighbors,
                    "tp_bps_min": float(tp_min),
                    "tp_bps_max": float(tp_max),
                    "sl_bps_min": float(sl_min),
                    "sl_bps_max": float(sl_max),
                    "criteria": list(spec.get("refine_criteria") or ["balanced", "profit_first", "robustness_first", "recall_balance"]),
                    "evidence": {
                        "num_runs": 1,
                        "num_profitable_runs": 1,
                        "median_profit_net": float(row.get("profit_net") or 0.0),
                        "median_sharpe": float(row.get("sharpe") or 0.0),
                        "median_pf": float(row.get("pf") or 0.0),
                        "trade_band_ok_ratio": 1.0,
                        "local_stability_score": 0.50,
                    },
                    "notes": "Explore winner widened into a local refine region.",
                }
            )
            if len(approved_regions) >= int(max(1, int(spec.get("promote_top_k", 4)))):
                break

        blocked_regions: list[dict[str, Any]] = []
        for row in list(state.get("results") or []):
            reasons = [str(x) for x in list(row.get("qg_reasons") or [])]
            if not any("trade" in reason.lower() for reason in reasons):
                continue
            blocked_regions.append(
                {
                    "reason": "trade_gate",
                    "models": [str(row.get("model") or "")],
                    "horizon_values": [int(row.get("horizon") or 0)],
                    "tp_bps_min": float(row.get("tp_bps") or 0.0),
                    "tp_bps_max": float(row.get("tp_bps") or 0.0),
                    "sl_bps_min": float(row.get("sl_bps") or 0.0),
                    "sl_bps_max": float(row.get("sl_bps") or 0.0),
                }
            )

        payload = {
            "version": 1,
            "mode": "explore",
            "created_at": self._now_str(),
            "dataset_signature": self._dataset_signature_from_row(ok_rows[0]) if ok_rows else {},
            "source_csv_path": str(state.get("csv_path") or self.csv_path),
            "source_checkpoint": self.state_path.as_posix(),
            "model_families": {
                "approved": approved_models,
                "rejected": [m for m in list(spec.get("models") or []) if m not in approved_models],
            },
            "approved_regions": approved_regions,
            "blocked_regions": blocked_regions,
            "recommended_refine_budget": {
                "max_models": int(max(1, int(spec.get("approved_model_top_k", 2)))),
                "max_regions": int(max(1, int(spec.get("promote_top_k", 4)))),
                "fine_step_bps": float(spec.get("fine_step_bps") or 5.0),
            },
        }
        return self._write_json_artifact(self._region_summary_path(), payload)

    def _write_shortlist(self, state: dict[str, Any]) -> str:
        spec = dict(state.get("spec") or {})
        ok_rows = [r for r in list(state.get("results") or []) if str(r.get("status")) == "ok"]
        ok_rows.sort(key=self._score_result, reverse=True)
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, int, float, float]] = set()
        for row in ok_rows:
            key = (
                str(row.get("model") or ""),
                int(row.get("horizon") or 0),
                float(row.get("tp_bps") or 0.0),
                float(row.get("sl_bps") or 0.0),
            )
            if key in seen:
                continue
            seen.add(key)
            criterion = str(row.get("criterion") or "balanced")
            candidates.append(
                {
                    "rank": len(candidates) + 1,
                    "candidate_id": f"{key[0]}_h{key[1]}_tp{int(key[2])}_sl{int(key[3])}",
                    "model": key[0],
                    "criterion": criterion,
                    "horizon": key[1],
                    "tp_bps": key[2],
                    "sl_bps": key[3],
                    "selection_score": float(row.get("profit_net") or 0.0),
                    "holdout_metrics": {
                        "profit_net": row.get("profit_net"),
                        "sharpe_net": row.get("sharpe"),
                        "pf": row.get("pf"),
                        "trades": row.get("trades"),
                    },
                    "status": "approved_for_refresh",
                    "refresh_priority": len(candidates) + 1,
                    "notes": f"Generated from Refine run; representative criterion={criterion}.",
                }
            )
            if len(candidates) >= int(max(1, int(spec.get("shortlist_top_k", self.candidate_top_n)))):
                break

        payload = {
            "version": 1,
            "mode": "refine",
            "created_at": self._now_str(),
            "dataset_signature": self._dataset_signature_from_row(ok_rows[0]) if ok_rows else {},
            "source_csv_path": str(state.get("csv_path") or self.csv_path),
            "source_region_summary": str(spec.get("source_region_summary") or self._region_summary_path().as_posix()),
            "candidates": candidates,
        }
        return self._write_json_artifact(self._shortlist_path(), payload)

    def _write_refresh_set(self, state: dict[str, Any]) -> str:
        spec = dict(state.get("spec") or {})
        ok_rows = [r for r in list(state.get("results") or []) if str(r.get("status")) == "ok"]
        ok_rows.sort(key=self._score_result, reverse=True)
        refresh_candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, int, float, float]] = set()
        for row in ok_rows:
            key = (
                str(row.get("model") or ""),
                int(row.get("horizon") or 0),
                float(row.get("tp_bps") or 0.0),
                float(row.get("sl_bps") or 0.0),
            )
            if key in seen:
                continue
            seen.add(key)
            criterion = str(row.get("criterion") or "balanced")
            refresh_candidates.append(
                {
                    "refresh_priority": len(refresh_candidates) + 1,
                    "candidate_id": f"{key[0]}_h{key[1]}_tp{int(key[2])}_sl{int(key[3])}",
                    "model": key[0],
                    "criterion": criterion,
                    "horizon": key[1],
                    "tp_bps": key[2],
                    "sl_bps": key[3],
                    "enabled": True,
                }
            )
            if len(refresh_candidates) >= int(max(1, self.candidate_top_n)):
                break

        payload = {
            "version": 1,
            "mode": "refresh",
            "created_at": self._now_str(),
            "dataset_signature": self._dataset_signature_from_row(ok_rows[0]) if ok_rows else {},
            "source_csv_path": str(state.get("csv_path") or self.csv_path),
            "target_csv_path": str(spec.get("target_csv_path") or self._target_csv_path_for_phase("refresh")),
            "source_artifact": str(spec.get("source_artifact") or spec.get("source_shortlist") or self._shortlist_path().as_posix()),
            "source_artifact_kind": str(spec.get("source_artifact_kind") or "shortlist"),
            "source_shortlist": str(spec.get("source_shortlist") or self._shortlist_path().as_posix()),
            "refresh_candidates": refresh_candidates,
        }
        return self._write_json_artifact(self._refresh_set_path(), payload)

    def _finalize_workflow(self, state: dict[str, Any]) -> str | None:
        mode = str(state.get("workflow_mode") or self.workflow_mode)
        if mode == "explore":
            return self._write_region_summary(state)
        if mode == "refine":
            return self._write_shortlist(state)
        if mode == "refresh":
            return self._write_refresh_set(state)
        return None

    @staticmethod
    def _meta_metrics(meta: dict[str, Any]) -> dict[str, Any]:
        mh = (meta.get("metrics_holdout") or meta.get("metrics") or {}) if isinstance(meta, dict) else {}
        if not isinstance(mh, dict):
            mh = {}
        return mh

    def _train_one(self, cfg: dict[str, Any]) -> dict[str, Any]:
        phase = str(cfg.get("phase", self.workflow_mode or "explore"))
        estimator_name = str(cfg.get("model", "lgb")).strip().lower()
        criterion = str(cfg.get("criterion", "balanced")).strip().lower()
        horizon = int(cfg.get("horizon", 12))
        tp_bps = float(cfg.get("tp_bps", 50.0))
        sl_bps = float(cfg.get("sl_bps", 50.0))
        return run_training_job(
            csv_path=self._target_csv_path_for_phase(phase),
            holdout_pct=float(self.holdout_pct),
            holdout_min_bars=int(self.holdout_min_bars),
            holdout_max_bars=int(self.holdout_max_bars),
            phase=phase,
            estimator_name=estimator_name,
            criterion=criterion,
            horizon=horizon,
            tp_bps=tp_bps,
            sl_bps=sl_bps,
            candidate_top_n=int(self.candidate_top_n),
            candidate_fresh_ratio=float(self.candidate_fresh_ratio),
            training_profile=dict(self.training_profiles.get(phase) or {}),
            should_continue=lambda: not self._stop_requested,
        )

    def run(self):
        state: dict[str, Any] | None = None
        try:
            state, resumed = self._load_or_init_state()
            spec = (state.get("spec") or {}) if isinstance(state, dict) else {}
            workflow_mode = str(spec.get("workflow_mode") or self.workflow_mode)
            completed_count, total_count = self._workflow_progress_counts(state)
            self.message.emit(
                f"INFO Workflow {'resume' if resumed else 'start'}: {self.state_path.as_posix()} "
                f"| mode={workflow_mode} "
                f"| phase={state.get('phase')} queue={completed_count}/{total_count}"
            )
            if self._last_recovered_results_count > 0:
                self.message.emit(
                    f"INFO Workflow recovery: restored {self._last_recovered_results_count} completed candidates from saved artifacts."
                )
            if self._last_reconciled_duplicate_results_count > 0 or self._last_reconciled_pruned_queue_count > 0:
                self.message.emit(
                    "INFO Workflow reconcile: "
                    f"removed {self._last_reconciled_duplicate_results_count} duplicate result rows "
                    f"and pruned {self._last_reconciled_pruned_queue_count} completed candidates from queue."
                )

            while not self._stop_requested:
                q = list(state.get("queue") or [])
                i = int(state.get("queue_idx", 0))
                if i >= len(q):
                    artifact_path = self._finalize_workflow(state)
                    state["phase"] = "done"
                    state["completed"] = True
                    self._save_state(state)
                    if artifact_path:
                        self.message.emit(f"INFO Workflow artifact saved: {artifact_path}")
                    break

                cfg = dict(q[i])
                candidate_key = self._candidate_key(cfg)
                completed_keys = {self._candidate_key(row) for row in list(state.get("results") or [])}
                if candidate_key in completed_keys:
                    state["queue_idx"] = i + 1
                    self._save_state(state)
                    self.message.emit(
                        f"INFO Workflow skip duplicate candidate: mode={workflow_mode} "
                        f"model={cfg.get('model')} criterion={cfg.get('criterion')} "
                        f"horizon={cfg.get('horizon')} tp={cfg.get('tp_bps')} sl={cfg.get('sl_bps')}"
                    )
                    continue
                completed_count, total_count = self._workflow_progress_counts(state)
                self.message.emit(
                    f"INFO Workflow run [{completed_count+1}/{total_count}] mode={workflow_mode} "
                    f"model={cfg.get('model')} criterion={cfg.get('criterion')} "
                    f"horizon={cfg.get('horizon')} tp={cfg.get('tp_bps')} sl={cfg.get('sl_bps')}"
                )
                row = self._train_one(cfg)
                state.setdefault("results", []).append(row)
                state["queue_idx"] = i + 1
                self._save_state(state)
                self.result.emit(dict(row))

            if self._stop_requested:
                state["stopped"] = True
                state["completed"] = False
                self._save_state(state)

            completed = bool(state.get("phase") == "done" and state.get("completed"))
            self.finished_state.emit(self.state_path.as_posix(), completed)
        except InterruptedError:
            if isinstance(state, dict):
                state["stopped"] = True
                state["completed"] = False
                self._save_state(state)
            self.message.emit("INFO Workflow: stop acknowledged, aktualni kandidat prerusen.")
            self.finished_state.emit(self.state_path.as_posix(), False)
        except Exception as e:
            self.error.emit(str(e))


class ModelTrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset: pd.DataFrame | None = None
        self.csv_path: str | None = None
        self._ui_settings = QSettings("ibkr_trading_bot", "model_training_tab")
        self._pending_restore_csv_path: str | None = None
        self.worker: TrainWorker | None = None
        self.auto_worker: AutoSearchWorker | None = None
        self.refine_source_artifact_path: str | None = None
        self.refresh_source_artifact_path: str | None = None
        self.refresh_target_csv_path: str | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_test: pd.Series | None = None
        self._is_ternary_target: bool = False
        self._label_horizon_bars: int = 12
        self._label_take_profit_bps: float = 50.0
        self._label_stop_loss_bps: float = 50.0
        self.holdout_pct_default = 0.10
        self.holdout_min_bars_default = 1000
        self.holdout_max_bars_default = 6000

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        box1 = QGroupBox("1) Nacteni dat pro trenink")
        box1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay1 = QHBoxLayout(box1)
        self.lbl_csv = QLabel("Vybrany soubor: -")
        self.btn_csv = QPushButton("Vybrat CSV...")
        self.btn_csv.clicked.connect(self.pick_csv)
        lay1.addWidget(self.lbl_csv)
        lay1.addStretch(1)
        lay1.addWidget(self.btn_csv)
        root.addWidget(box1)

        box2 = QGroupBox("2) Workflow hledani modelu")
        box2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay2 = QVBoxLayout(box2)

        row = QHBoxLayout()
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["hgbt", "rf", "et", "xgb", "lgb", "svm"])
        self.cmb_model.hide()
        self.cmb_training_mode = QComboBox()
        self.cmb_training_mode.addItems(["quick", "standard"])
        self.cmb_training_mode.setCurrentText("standard")
        self.cmb_training_mode.currentTextChanged.connect(lambda _: self._refresh_train_button_text())
        self.cmb_training_mode.hide()
        self.cmb_candidate_criterion = QComboBox()
        self.cmb_candidate_criterion.addItems(
            ["balanced", "profit_first", "robustness_first", "recall_balance"]
        )
        self.cmb_candidate_criterion.setCurrentText("balanced")
        self.cmb_candidate_criterion.hide()
        row.addWidget(QLabel("Top N:"))
        self.cmb_candidate_top_n = QComboBox()
        self.cmb_candidate_top_n.addItems(["3", "5", "8", "12"])
        self.cmb_candidate_top_n.setCurrentText("5")
        row.addWidget(self.cmb_candidate_top_n)
        row.addWidget(QLabel("Fresh %:"))
        self.cmb_candidate_fresh_pct = QComboBox()
        self.cmb_candidate_fresh_pct.addItems(["20", "30", "40", "50"])
        self.cmb_candidate_fresh_pct.setCurrentText("30")
        row.addWidget(self.cmb_candidate_fresh_pct)
        row.addWidget(QLabel("Backend:"))
        self.cmb_search_backend = QComboBox()
        self.cmb_search_backend.addItems(["grid", "optuna"])
        self.cmb_search_backend.setCurrentText("grid")
        self.cmb_search_backend.currentTextChanged.connect(self._on_search_backend_changed)
        row.addWidget(self.cmb_search_backend)
        std_profile = training_profile_for_mode("standard")
        row.addWidget(QLabel("Trials:"))
        self.spn_optuna_trials = QSpinBox()
        self.spn_optuna_trials.setRange(1, 1000)
        self.spn_optuna_trials.setValue(int(std_profile.get("optuna_trials", 24)))
        row.addWidget(self.spn_optuna_trials)
        row.addWidget(QLabel("Timeout [s]:"))
        self.spn_optuna_timeout = QSpinBox()
        self.spn_optuna_timeout.setRange(1, 86400)
        self.spn_optuna_timeout.setValue(int(std_profile.get("optuna_timeout_seconds", 300)))
        row.addWidget(self.spn_optuna_timeout)
        row.addWidget(QLabel("Auto profil:"))
        self.cmb_auto_search_profile = QComboBox()
        self.cmb_auto_search_profile.addItems(["Explore", "Refine", "Refresh"])
        self.cmb_auto_search_profile.setCurrentText("Explore")
        self.cmb_auto_search_profile.currentTextChanged.connect(self._on_auto_search_profile_changed)
        row.addWidget(self.cmb_auto_search_profile)
        row.addWidget(QLabel("Fee/trade:"))
        self.spn_training_fee = QDoubleSpinBox()
        self.spn_training_fee.setRange(0.0, 100000.0)
        self.spn_training_fee.setDecimals(2)
        self.spn_training_fee.setSingleStep(0.25)
        self.spn_training_fee.setValue(0.0)
        self.spn_training_fee.valueChanged.connect(lambda *_: self._persist_training_cost_settings())
        row.addWidget(self.spn_training_fee)
        row.addWidget(QLabel("Slippage [bps]:"))
        self.spn_training_slippage = QDoubleSpinBox()
        self.spn_training_slippage.setRange(0.0, 10000.0)
        self.spn_training_slippage.setDecimals(2)
        self.spn_training_slippage.setSingleStep(0.25)
        self.spn_training_slippage.setValue(0.0)
        self.spn_training_slippage.valueChanged.connect(lambda *_: self._persist_training_cost_settings())
        row.addWidget(self.spn_training_slippage)
        row.addStretch(1)

        self.btn_train = QPushButton("Trenovat (standard)")
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self.run_training)
        self.btn_train.hide()
        self.btn_auto_search = QPushButton("Workflow (resume)")
        self.btn_auto_search.setEnabled(False)
        self.btn_auto_search.clicked.connect(self.run_auto_search)
        row.addWidget(self.btn_auto_search)
        self.btn_auto_stop = QPushButton("Stop auto")
        self.btn_auto_stop.setEnabled(False)
        self.btn_auto_stop.clicked.connect(self.stop_auto_search)
        row.addWidget(self.btn_auto_stop)
        lay2.addLayout(row)

        self.refresh_overrides_widget = QWidget(self)
        refresh_overrides_layout = QVBoxLayout(self.refresh_overrides_widget)
        refresh_overrides_layout.setContentsMargins(0, 0, 0, 0)
        refresh_overrides_layout.setSpacing(6)

        refresh_artifact_row = QHBoxLayout()
        self.lbl_workflow_source_caption = QLabel("Workflow source:")
        refresh_artifact_row.addWidget(self.lbl_workflow_source_caption)
        self.lbl_refresh_source_artifact = QLabel("automaticky podle aktivniho datasetu")
        refresh_artifact_row.addWidget(self.lbl_refresh_source_artifact, 1)
        self.btn_refresh_source_artifact = QPushButton("Vybrat artifact...")
        self.btn_refresh_source_artifact.clicked.connect(self.pick_refresh_source_artifact)
        refresh_artifact_row.addWidget(self.btn_refresh_source_artifact)
        self.btn_refresh_source_artifact_clear = QPushButton("Auto")
        self.btn_refresh_source_artifact_clear.clicked.connect(self.clear_refresh_source_artifact)
        refresh_artifact_row.addWidget(self.btn_refresh_source_artifact_clear)
        refresh_overrides_layout.addLayout(refresh_artifact_row)

        refresh_target_row = QHBoxLayout()
        self.lbl_refresh_target_caption = QLabel("Refresh target CSV:")
        refresh_target_row.addWidget(self.lbl_refresh_target_caption)
        self.lbl_refresh_target_csv = QLabel("stejny jako aktivni dataset")
        refresh_target_row.addWidget(self.lbl_refresh_target_csv, 1)
        self.btn_refresh_target_csv = QPushButton("Vybrat CSV...")
        self.btn_refresh_target_csv.clicked.connect(self.pick_refresh_target_csv)
        refresh_target_row.addWidget(self.btn_refresh_target_csv)
        self.btn_refresh_target_csv_clear = QPushButton("Stejny dataset")
        self.btn_refresh_target_csv_clear.clicked.connect(self.clear_refresh_target_csv)
        refresh_target_row.addWidget(self.btn_refresh_target_csv_clear)
        refresh_overrides_layout.addLayout(refresh_target_row)

        lay2.addWidget(self.refresh_overrides_widget)

        self.prog = QProgressBar()
        self.prog.setRange(0, 1)
        self.prog.hide()

        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["#", "mean score (CV)", "std", "params"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tbl.hide()
        root.addWidget(box2)

        box3 = QGroupBox("3) Konzole")
        lay3 = QVBoxLayout(box3)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Hlasky treninku a evaluace...")
        lay3.addWidget(self.log)
        root.addWidget(box3, 1)
        self._restore_training_cost_settings()
        self._restore_refresh_workflow_paths()
        self._on_search_backend_changed()
        self._update_refresh_override_labels()
        self._update_refresh_overrides_visibility()
        self._restore_last_csv_path()

    def _on_auto_search_profile_changed(self, *_args) -> None:
        self._update_refresh_override_labels()
        self._update_refresh_overrides_visibility()

    def _current_training_mode(self) -> str:
        txt = (self.cmb_training_mode.currentText() or "").strip().lower()
        return txt if txt in {"quick", "standard"} else "standard"

    def _training_profile_for_mode(self, mode: str) -> dict[str, Any]:
        return training_profile_for_mode(mode)

    def _selected_search_backend(self) -> str:
        txt = (self.cmb_search_backend.currentText() or "").strip().lower()
        return txt if txt in {"grid", "optuna"} else "grid"

    def _selected_optuna_trials(self) -> int:
        try:
            return int(self.spn_optuna_trials.value())
        except Exception:
            return 24

    def _selected_optuna_timeout_seconds(self) -> int:
        try:
            return int(self.spn_optuna_timeout.value())
        except Exception:
            return 300

    def _current_fee_per_trade(self) -> float:
        try:
            return float(self.spn_training_fee.value())
        except Exception:
            return 0.0

    def _current_slippage_bps(self) -> float:
        try:
            return float(self.spn_training_slippage.value())
        except Exception:
            return 0.0

    def _persist_training_cost_settings(self) -> None:
        self._ui_settings.setValue(TRAINING_FEE_PER_TRADE_KEY, float(self._current_fee_per_trade()))
        self._ui_settings.setValue(TRAINING_SLIPPAGE_BPS_KEY, float(self._current_slippage_bps()))
        self._ui_settings.sync()

    def _restore_training_cost_settings(self) -> None:
        try:
            fee_value = float(self._ui_settings.value(TRAINING_FEE_PER_TRADE_KEY, self.spn_training_fee.value()) or 0.0)
        except Exception:
            fee_value = float(self.spn_training_fee.value())
        try:
            slippage_value = float(self._ui_settings.value(TRAINING_SLIPPAGE_BPS_KEY, self.spn_training_slippage.value()) or 0.0)
        except Exception:
            slippage_value = float(self.spn_training_slippage.value())
        self.spn_training_fee.blockSignals(True)
        self.spn_training_slippage.blockSignals(True)
        self.spn_training_fee.setValue(float(max(0.0, fee_value)))
        self.spn_training_slippage.setValue(float(max(0.0, slippage_value)))
        self.spn_training_fee.blockSignals(False)
        self.spn_training_slippage.blockSignals(False)

    def _current_source_artifact_path_for_profile(self, profile: str) -> str | None:
        profile_norm = _normalize_auto_search_profile(profile)
        if profile_norm == "refine":
            return self.refine_source_artifact_path
        if profile_norm == "refresh":
            return self.refresh_source_artifact_path
        return None

    def _current_source_artifact_path(self) -> str | None:
        return self._current_source_artifact_path_for_profile(self._current_auto_search_profile())

    def _default_source_artifact_label(self, profile: str) -> str:
        profile_norm = _normalize_auto_search_profile(profile)
        if profile_norm == "refine":
            return "region_summary podle aktivniho datasetu"
        if profile_norm == "refresh":
            return "shortlist/refresh_set podle aktivniho datasetu"
        return "automaticky podle aktivniho datasetu"

    def _update_refresh_override_labels(self) -> None:
        profile = self._current_auto_search_profile()
        source_txt = str(self._current_source_artifact_path_for_profile(profile) or "").strip()
        target_txt = str(self.refresh_target_csv_path or "").strip()
        self.lbl_workflow_source_caption.setText(
            "Refine source:" if profile == "refine" else "Refresh source:" if profile == "refresh" else "Workflow source:"
        )
        self.lbl_refresh_source_artifact.setText(
            os.path.basename(source_txt) if source_txt else self._default_source_artifact_label(profile)
        )
        self.lbl_refresh_target_csv.setText(
            os.path.basename(target_txt) if target_txt else "stejny jako aktivni dataset"
        )

    def _update_refresh_overrides_visibility(self) -> None:
        profile = self._current_auto_search_profile()
        source_visible = profile in {"refine", "refresh"}
        refresh_visible = profile == "refresh"
        self.refresh_overrides_widget.setVisible(source_visible)
        self.lbl_refresh_target_caption.setVisible(refresh_visible)
        self.lbl_refresh_target_csv.setVisible(refresh_visible)
        self.btn_refresh_target_csv.setVisible(refresh_visible)
        self.btn_refresh_target_csv_clear.setVisible(refresh_visible)

    def _persist_refresh_workflow_paths(self) -> None:
        self._ui_settings.setValue(REFINE_SOURCE_ARTIFACT_PATH_KEY, str(self.refine_source_artifact_path or ""))
        self._ui_settings.setValue(REFRESH_SOURCE_ARTIFACT_PATH_KEY, str(self.refresh_source_artifact_path or ""))
        self._ui_settings.setValue(REFRESH_TARGET_CSV_PATH_KEY, str(self.refresh_target_csv_path or ""))
        self._ui_settings.sync()

    def _restore_refresh_workflow_paths(self) -> None:
        refine_source_artifact = str(self._ui_settings.value(REFINE_SOURCE_ARTIFACT_PATH_KEY, "") or "").strip()
        source_artifact = str(self._ui_settings.value(REFRESH_SOURCE_ARTIFACT_PATH_KEY, "") or "").strip()
        target_csv = str(self._ui_settings.value(REFRESH_TARGET_CSV_PATH_KEY, "") or "").strip()
        if refine_source_artifact:
            try:
                candidate = Path(refine_source_artifact).expanduser().resolve()
                if candidate.exists():
                    self.refine_source_artifact_path = str(candidate)
            except Exception:
                self.refine_source_artifact_path = None
        if source_artifact:
            try:
                candidate = Path(source_artifact).expanduser().resolve()
                if candidate.exists():
                    self.refresh_source_artifact_path = str(candidate)
            except Exception:
                self.refresh_source_artifact_path = None
        if target_csv:
            try:
                candidate = Path(target_csv).expanduser().resolve()
                if candidate.exists():
                    self.refresh_target_csv_path = str(candidate)
            except Exception:
                self.refresh_target_csv_path = None

    def _set_source_artifact_path_for_profile(self, profile: str, path: str | None, *, persist: bool) -> None:
        normalized_path = ""
        if path:
            candidate = Path(path).expanduser().resolve()
            normalized_path = str(candidate) if candidate.exists() else ""
        if _normalize_auto_search_profile(profile) == "refine":
            self.refine_source_artifact_path = normalized_path or None
        else:
            self.refresh_source_artifact_path = normalized_path or None
        self._update_refresh_override_labels()
        if persist:
            self._persist_refresh_workflow_paths()

    def _set_refine_source_artifact_path(self, path: str | None, *, persist: bool) -> None:
        self._set_source_artifact_path_for_profile("refine", path, persist=persist)

    def _set_refresh_source_artifact_path(self, path: str | None, *, persist: bool) -> None:
        self._set_source_artifact_path_for_profile("refresh", path, persist=persist)

    def _set_refresh_target_csv_path(self, path: str | None, *, persist: bool) -> None:
        normalized_path = ""
        if path:
            candidate = Path(path).expanduser().resolve()
            normalized_path = str(candidate) if candidate.exists() else ""
        self.refresh_target_csv_path = normalized_path or None
        self._update_refresh_override_labels()
        if persist:
            self._persist_refresh_workflow_paths()

    def clear_refresh_source_artifact(self) -> None:
        profile = self._current_auto_search_profile()
        if profile not in {"refine", "refresh"}:
            return
        self._set_source_artifact_path_for_profile(profile, None, persist=True)

    def clear_refresh_target_csv(self) -> None:
        self._set_refresh_target_csv_path(None, persist=True)

    def _default_refresh_source_artifact_dialog_dir(self) -> str:
        current_source_artifact = self._current_source_artifact_path()
        if current_source_artifact:
            candidate_dir = Path(current_source_artifact).expanduser().resolve().parent
            if candidate_dir.exists():
                return candidate_dir.as_posix()
        return (Path(_model_dir()) / "auto_search").as_posix()

    def pick_refresh_source_artifact(self) -> None:
        profile = self._current_auto_search_profile()
        if profile not in {"refine", "refresh"}:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Vyber refine source artifact" if profile == "refine" else "Vyber refresh source artifact",
            self._default_refresh_source_artifact_dialog_dir(),
            "Workflow JSON (*.json)",
        )
        if not path:
            return
        self._set_source_artifact_path_for_profile(profile, path, persist=True)

    def pick_refresh_target_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Vyber refresh target CSV",
            self._default_csv_dialog_dir(),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        self._set_refresh_target_csv_path(path, persist=True)

    def _on_search_backend_changed(self, *_args):
        optuna_enabled = self._selected_search_backend() == "optuna"
        self.spn_optuna_trials.setEnabled(optuna_enabled)
        self.spn_optuna_timeout.setEnabled(optuna_enabled)
        self._refresh_train_button_text()

    def _apply_search_backend_profile_overrides(self, profile: dict[str, Any] | None) -> dict[str, Any]:
        out = dict(profile or {})
        out["search_backend"] = self._selected_search_backend()
        out["optuna_trials"] = int(self._selected_optuna_trials())
        out["optuna_timeout_seconds"] = int(self._selected_optuna_timeout_seconds())
        out["fee_per_trade"] = float(self._current_fee_per_trade())
        out["slippage_bps"] = float(self._current_slippage_bps())
        return out

    def _log_search_backend_hint(self, *, estimators: list[str] | None = None):
        backend = self._selected_search_backend()
        if backend != "optuna":
            return
        if not HAS_OPTUNA:
            self.log.appendPlainText(
                "WARN Optuna backend byl zvolen, ale Optuna neni nainstalovana. Pipeline fallbackne na grid."
            )
            return
        if estimators is None:
            self.log.appendPlainText(
                "INFO Optuna backend se pouzije pro hgbt/lgb; ostatni estimatory fallbacknou na grid."
            )
            return
        unsupported = sorted(
            {
                str(est).strip().lower()
                for est in (estimators or [])
                if str(est).strip().lower() not in {"hgbt", "histgb", "histgradientboosting", "lgb", "lightgbm"}
            }
        )
        if unsupported:
            self.log.appendPlainText(
                "INFO Optuna backend podporuje jen hgbt/lgb; "
                f"estimator(y) {unsupported} fallbacknou na grid."
            )

    def _refresh_train_button_text(self):
        mode = self._current_training_mode()
        running = self._is_training_running()
        if self.dataset is None:
            self.btn_train.setText(f"Trenovat ({mode})")
            self.btn_train.setEnabled(False)
            self.btn_auto_search.setText("Workflow (resume)")
            self.btn_auto_search.setEnabled(False)
            self.btn_auto_stop.setEnabled(self._is_auto_search_running())
            return
        n_rows = int(len(self.dataset))
        n_hold = self._compute_holdout_bars(n_rows)
        profile = self._current_auto_search_profile()
        self.btn_train.setText(
            f"Trenovat [{mode}] ({n_rows} rows, holdout {n_hold} [{float(self.holdout_pct_default) * 100.0:.1f}%])"
        )
        self.btn_train.setEnabled(not running)
        self.btn_auto_search.setText(f"Workflow [{profile}] (resume) [{n_rows} rows, holdout {n_hold}]")
        self.btn_auto_search.setEnabled(not running)
        self.btn_auto_stop.setEnabled(self._is_auto_search_running())

    def _is_training_running(self) -> bool:
        return bool(
            (self.worker is not None and self.worker.isRunning())
            or (self.auto_worker is not None and self.auto_worker.isRunning())
        )

    def _is_auto_search_running(self) -> bool:
        return bool(self.auto_worker is not None and self.auto_worker.isRunning())

    def _set_controls_running(self, running: bool):
        self.btn_csv.setEnabled(not running)
        self.cmb_model.setEnabled(not running)
        self.cmb_training_mode.setEnabled(not running)
        self.cmb_candidate_criterion.setEnabled(not running)
        self.cmb_candidate_top_n.setEnabled(not running)
        self.cmb_candidate_fresh_pct.setEnabled(not running)
        self.cmb_search_backend.setEnabled(not running)
        self.spn_optuna_trials.setEnabled((not running) and (self._selected_search_backend() == "optuna"))
        self.spn_optuna_timeout.setEnabled((not running) and (self._selected_search_backend() == "optuna"))
        self.spn_training_fee.setEnabled(not running)
        self.spn_training_slippage.setEnabled(not running)
        self.cmb_auto_search_profile.setEnabled(not running)
        self.btn_refresh_source_artifact.setEnabled(not running)
        self.btn_refresh_source_artifact_clear.setEnabled(not running)
        self.btn_refresh_target_csv.setEnabled(not running)
        self.btn_refresh_target_csv_clear.setEnabled(not running)
        self.btn_train.setEnabled((not running) and (self.dataset is not None))
        self.btn_auto_search.setEnabled((not running) and (self.dataset is not None))
        self.btn_auto_stop.setEnabled(self._is_auto_search_running())

    @staticmethod
    def _artifact_stem_for_path(path: str | None) -> str:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path or "dataset").stem).strip("_")
        return stem or "dataset"

    @staticmethod
    def _auto_search_artifact_dir() -> Path:
        return Path(_model_dir()) / "auto_search"

    def _region_summary_artifact_path_for_csv(self, csv_path: str) -> Path:
        return self._auto_search_artifact_dir() / f"{self._artifact_stem_for_path(csv_path)}_region_summary.json"

    def _shortlist_artifact_path_for_csv(self, csv_path: str) -> Path:
        return self._auto_search_artifact_dir() / f"{self._artifact_stem_for_path(csv_path)}_shortlist.json"

    def _refresh_set_artifact_path_for_csv(self, csv_path: str) -> Path:
        return self._auto_search_artifact_dir() / f"{self._artifact_stem_for_path(csv_path)}_refresh_set.json"

    def _resolve_source_artifact_path(self, profile: str) -> Path | None:
        profile_norm = _normalize_auto_search_profile(profile)
        explicit_source = str(self._current_source_artifact_path_for_profile(profile_norm) or "").strip()
        if explicit_source:
            return Path(explicit_source)
        if not self.csv_path:
            return None
        if profile_norm == "refine":
            candidate = self._region_summary_artifact_path_for_csv(self.csv_path)
            return candidate if candidate.exists() else None
        if profile_norm == "refresh":
            refresh_set_path = self._refresh_set_artifact_path_for_csv(self.csv_path)
            if refresh_set_path.exists():
                return refresh_set_path
            shortlist_path = self._shortlist_artifact_path_for_csv(self.csv_path)
            return shortlist_path if shortlist_path.exists() else None
        return None

    @staticmethod
    def _load_workflow_artifact_payload(path: Path) -> dict[str, Any]:
        payload = jsonlib.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid workflow artifact: {path.as_posix()}")
        return payload

    @staticmethod
    def _artifact_dataset_triplet(payload: dict[str, Any]) -> tuple[str, str, str] | None:
        signature = payload.get("dataset_signature") or {}
        if not isinstance(signature, dict):
            return None
        instrument = str(signature.get("instrument") or "").strip().upper()
        exchange = str(signature.get("exchange") or "").strip().upper()
        timeframe = str(signature.get("timeframe") or "").strip().lower()
        if not instrument or not exchange or not timeframe:
            return None
        return (instrument, exchange, timeframe)

    @staticmethod
    def _format_dataset_triplet(triplet: tuple[str, str, str] | None) -> str:
        if not triplet:
            return "unknown"
        return f"{triplet[0]}_{triplet[1]}_{triplet[2]}"

    @staticmethod
    def _dataset_triplet_from_csv_path(csv_path: str | None) -> tuple[str, str, str] | None:
        if not csv_path:
            return None
        _, meta = runtime_name_and_meta_from_csv(str(csv_path), 1, 0, 0)
        instrument = str(meta.get("instrument") or "").strip().upper()
        exchange = str(meta.get("exchange") or "").strip().upper()
        timeframe = str(meta.get("timeframe") or "").strip().lower()
        if not instrument or not exchange or not timeframe:
            return None
        return (instrument, exchange, timeframe)

    def _validate_auto_search_inputs(
        self,
        profile: str,
        source_artifact_path: Path | None,
        refresh_target_csv: str | None,
    ) -> None:
        profile_norm = _normalize_auto_search_profile(profile)
        if profile_norm == "explore":
            return
        if source_artifact_path is None:
            if profile_norm == "refine":
                expected = self._region_summary_artifact_path_for_csv(self.csv_path or "dataset")
                raise ValueError(
                    "Refine vyzaduje region_summary artifact pro aktivni dataset. "
                    f"expected={expected.as_posix()} | Pro novejsi data nejdriv spust Explore; "
                    "pro pretrenovani finalistu pouzij Refresh."
                )
            refresh_set_path = self._refresh_set_artifact_path_for_csv(self.csv_path or "dataset")
            shortlist_path = self._shortlist_artifact_path_for_csv(self.csv_path or "dataset")
            raise ValueError(
                f"Refresh vyzaduje refresh_set nebo shortlist: {refresh_set_path.as_posix()} / {shortlist_path.as_posix()}"
            )

        payload = self._load_workflow_artifact_payload(source_artifact_path)
        if profile_norm == "refine":
            approved_regions = list(payload.get("approved_regions") or [])
            if not approved_regions:
                raise ValueError(
                    f"Refine source artifact musi obsahovat approved_regions: {source_artifact_path.as_posix()}"
                )
            source_csv_path = _normalized_resume_path(payload.get("source_csv_path"))
            current_csv_path = _normalized_resume_path(self.csv_path)
            if source_csv_path and current_csv_path and source_csv_path != current_csv_path:
                raise ValueError(
                    "Refine source artifact patri k jinemu CSV. "
                    f"artifact={source_artifact_path.as_posix()} current_csv={self.csv_path} | "
                    "Refine je vazany na zdrojovy dataset; pro novejsi data pouzij Explore nebo Refresh."
                )
            if self.dataset is not None and self.csv_path:
                current_signature = dataset_snapshot_signature_from_csv(self.csv_path, len(self.dataset))
                artifact_signature = payload.get("dataset_signature") or {}
                if current_signature and isinstance(artifact_signature, dict):
                    artifact_exact = dataset_snapshot_signature_from_csv(self.csv_path, int(artifact_signature.get("n_total_bars") or 0))
                    artifact_triplet = self._artifact_dataset_triplet(payload)
                    current_triplet = (current_signature[0], current_signature[1], current_signature[2])
                    if artifact_triplet and artifact_triplet != current_triplet:
                        raise ValueError(
                            "Refine source artifact neodpovida aktivnimu datasetu. "
                            f"artifact={self._format_dataset_triplet(artifact_triplet)} current={self._format_dataset_triplet(current_triplet)} | "
                            "Refine je vazany na zdrojovy dataset; pro novejsi data pouzij Explore nebo Refresh."
                        )
                    try:
                        artifact_bars = int(artifact_signature.get("n_total_bars") or 0)
                    except Exception:
                        artifact_bars = 0
                    if artifact_bars > 0 and artifact_bars != int(len(self.dataset)):
                        raise ValueError(
                            "Refine source artifact musi mit stejny rozsah pripravenych dat jako aktivni dataset. "
                            f"artifact_bars={artifact_bars} current_bars={len(self.dataset)} | "
                            "Refine je vazany na zdrojovy dataset; pro novejsi data pouzij Explore nebo Refresh."
                        )
            return

        refresh_candidates = list(payload.get("refresh_candidates") or [])
        shortlist_candidates = list(payload.get("candidates") or [])
        if not refresh_candidates and not shortlist_candidates:
            raise ValueError(
                f"Refresh source artifact musi obsahovat refresh_candidates nebo candidates: {source_artifact_path.as_posix()}"
            )
        target_csv = str(refresh_target_csv or self.csv_path or "").strip()
        artifact_triplet = self._artifact_dataset_triplet(payload)
        target_triplet = self._dataset_triplet_from_csv_path(target_csv)
        if artifact_triplet and target_triplet and artifact_triplet != target_triplet:
            raise ValueError(
                "Refresh source artifact neodpovida target datasetu. "
                f"artifact={self._format_dataset_triplet(artifact_triplet)} target={self._format_dataset_triplet(target_triplet)}"
            )

    def _current_candidate_criterion(self) -> str:
        txt = (self.cmb_candidate_criterion.currentText() or "").strip().lower()
        allowed = {"balanced", "profit_first", "robustness_first", "recall_balance"}
        return txt if txt in allowed else "balanced"

    def _current_candidate_top_n(self) -> int:
        try:
            return int(self.cmb_candidate_top_n.currentText())
        except Exception:
            return 5

    def _current_candidate_fresh_ratio(self) -> float:
        try:
            return float(self.cmb_candidate_fresh_pct.currentText()) / 100.0
        except Exception:
            return 0.30

    def _current_auto_search_profile(self) -> str:
        return _normalize_auto_search_profile(self.cmb_auto_search_profile.currentText() or "")

    def _compute_holdout_bars(self, n_total: int) -> int:
        return runtime_compute_holdout_bars(
            int(n_total),
            float(self.holdout_pct_default),
            int(self.holdout_min_bars_default),
            int(self.holdout_max_bars_default),
        )

    def on_tab_activated(self):
        if self.dataset is not None or not self._pending_restore_csv_path:
            return
        restore_path = self._pending_restore_csv_path
        self._pending_restore_csv_path = None
        self._load_csv_path(restore_path, persist=False, restored=True)

    def _restore_last_csv_path(self) -> None:
        saved_path = str(self._ui_settings.value(LAST_TRAINING_CSV_PATH_KEY, "") or "").strip()
        if not saved_path:
            return
        candidate = Path(saved_path).expanduser()
        try:
            if not candidate.exists():
                return
            normalized_path = str(candidate.resolve())
        except Exception:
            return
        self._pending_restore_csv_path = normalized_path
        self.lbl_csv.setText(f"Vybrany soubor: {os.path.basename(normalized_path)}")

    def _default_csv_dialog_dir(self) -> str:
        candidate_path = self.csv_path or self._pending_restore_csv_path
        if candidate_path:
            candidate_dir = Path(candidate_path).expanduser().resolve().parent
            if candidate_dir.exists():
                return candidate_dir.as_posix()
        base_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
        return base_dir.as_posix()

    def _load_csv_path(self, path: str, *, persist: bool, restored: bool = False) -> bool:
        normalized_path = str(Path(path).expanduser().resolve())
        self.csv_path = normalized_path
        self.lbl_csv.setText(f"Vybrany soubor: {os.path.basename(normalized_path)}")
        if restored:
            self.log.appendPlainText(f"INFO Obnovuji posledni CSV: {normalized_path}")

        try:
            svc = DatasetService()
            df = svc.prepare_from_csv(
                normalized_path,
                labeling="triple_barrier",
                target_mode="ternary",
                horizon=int(self._label_horizon_bars),
                take_profit_bps=float(self._label_take_profit_bps),
                stop_loss_bps=float(self._label_stop_loss_bps),
                same_bar_policy="neutral",
            )
            self.dataset = df
            n_rows = len(df)
            uniq = sorted(pd.Series(df["target"]).dropna().astype(int).unique().tolist()) if "target" in df.columns else []
            dist = pd.Series(df["target"]).dropna().astype(int).value_counts().to_dict() if "target" in df.columns else {}
            self.log.appendPlainText(f"INFO Target classes: {uniq}")
            self.log.appendPlainText(f"INFO Target distribution: {dist}")
            self.log.appendPlainText(
                f"INFO Label params: horizon={int(self._label_horizon_bars)} "
                f"tp_bps={float(self._label_take_profit_bps):.1f} "
                f"sl_bps={float(self._label_stop_loss_bps):.1f} same_bar=neutral"
            )
            dataset_meta = read_dataset_sidecar_meta(normalized_path)
            if dataset_meta:
                self.log.appendPlainText(
                    "INFO Dataset meta: "
                    f"kind={dataset_meta.get('dataset_kind')} "
                    f"canonical={dataset_meta.get('canonical')} "
                    f"quality={dataset_meta.get('quality_gate_passed')}"
                )
                if "prepared_retention_ratio" in dataset_meta:
                    self.log.appendPlainText(
                        "INFO Dataset quality: "
                        f"prepared_ratio={float(dataset_meta.get('prepared_retention_ratio', 0.0)):.3f} "
                        f"flat_zero_ratio={float((dataset_meta.get('quality_report') or {}).get('flat_zero_ratio', 0.0)):.3f}"
                    )
            self.log.appendPlainText(f"OK Nacteno: {normalized_path} | radku={n_rows}")
            self._log_dataset_audit(df)
            self._set_controls_running(False)
            self._refresh_train_button_text()
            self.tbl.setRowCount(0)
            self.prog.setRange(0, 1)
            self.prog.setValue(0)
            self._pending_restore_csv_path = None
            if persist:
                self._ui_settings.setValue(LAST_TRAINING_CSV_PATH_KEY, normalized_path)
                self._ui_settings.sync()
            return True
        except Exception as e:
            self.log.appendPlainText(f"ERROR Chyba nacteni/pripravy dat: {e}")
            self._set_controls_running(False)
            return False

    def pick_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Vyber CSV s daty",
            self._default_csv_dialog_dir(),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        self._load_csv_path(path, persist=True)

    def run_training(self):
        if self.dataset is None:
            self.log.appendPlainText("WARN Nejprve vyber CSV.")
            return
        if self._is_training_running():
            self.log.appendPlainText("WARN Jiz probiha trenink nebo workflow. Pockej na dokonceni.")
            return

        est = self.cmb_model.currentText().strip().lower()
        mode = self._current_training_mode()
        profile = self._apply_search_backend_profile_overrides(self._training_profile_for_mode(mode))
        workflow_mode = str(profile.get("workflow_mode") or canonical_workflow_mode(mode))
        compatibility_mode = profile.get("compatibility_mode")
        if compatibility_mode is None:
            compatibility_mode = compatibility_training_mode(mode)
        runtime_training_mode = str(compatibility_mode or mode)
        profile["training_mode"] = runtime_training_mode
        profile["training_mode_requested"] = mode
        profile["workflow_mode"] = workflow_mode
        profile["compatibility_mode"] = compatibility_mode
        profile["runtime_training_mode"] = runtime_training_mode
        profile["candidate_chain_enabled"] = True
        profile["candidate_selection_criterion"] = self._current_candidate_criterion()
        profile["candidate_top_n"] = int(max(1, self._current_candidate_top_n()))
        profile["candidate_fresh_ratio"] = float(
            np.clip(self._current_candidate_fresh_ratio(), 0.05, 0.80)
        )
        self._log_search_backend_hint(estimators=[est])
        df = self.dataset.copy().sort_values("timestamp").reset_index(drop=True)
        n_total = len(df)
        n_hold = self._compute_holdout_bars(n_total)
        df_train = df.iloc[: n_total - n_hold].reset_index(drop=True)
        df_hold = df.iloc[n_total - n_hold :].reset_index(drop=True) if n_hold > 0 else None
        split_idx = int(max(0, n_total - n_hold))
        self.log.appendPlainText(
            "INFO Holdout policy: "
            f"pct={float(self.holdout_pct_default) * 100.0:.1f}% "
            f"min={int(self.holdout_min_bars_default)} max={int(self.holdout_max_bars_default)} "
            f"=> holdout={int(n_hold)} train={int(len(df_train))} split_idx={split_idx} (no_overlap=True)"
        )

        feats_all = _select_feature_columns(df_train)
        X_cols = feats_all
        uniq_target = sorted(pd.Series(df["target"]).dropna().astype(int).unique().tolist())
        self._is_ternary_target = set(uniq_target).issubset({-1, 0, 1}) and len(uniq_target) == 3
        if df_hold is not None:
            self.X_test = df_hold[X_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            y_hold = df_hold["target"].astype(float)
            if self._is_ternary_target:
                self.y_test = y_hold.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
            else:
                self.y_test = (y_hold > 0).astype(int)
        else:
            self.X_test, self.y_test = None, None

        name_prefix, meta_extra = self._name_and_meta_from_csv(
            self.csv_path, n_total, len(df_train), len(df_hold) if df_hold is not None else 0
        )
        self.log.appendPlainText(f"INFO name_prefix={name_prefix} | meta_extra={meta_extra}")

        tf = (meta_extra.get("timeframe") or "").lower()
        mc_block = 100
        if tf in ("5min", "5m"):
            mc_block = 40
        elif tf in ("15min", "15m"):
            mc_block = 80
        elif tf in ("30min", "30m"):
            mc_block = 120
        elif tf in ("1hour", "1h"):
            mc_block = 150
        profile["mc_block_len"] = int(mc_block)
        self.log.appendPlainText(
            "INFO Training mode: "
            f"{mode} | workflow={workflow_mode} | compat={compatibility_mode} | runtime={runtime_training_mode} | cv={int(profile.get('n_splits', 5))} "
            f"top_k={int(profile.get('top_k_features', 12))} "
            f"search_backend={profile.get('search_backend', 'grid')} "
            f"optuna_trials={profile.get('optuna_trials')} "
            f"optuna_timeout={profile.get('optuna_timeout_seconds')}s "
            f"grid_used<={profile.get('max_param_candidates')} "
            f"mc_enabled={bool(profile.get('mc_enabled', True))} "
            f"mc_iters={int(profile.get('mc_iters', 200))} "
            f"qgate={bool(profile.get('quality_gate_enabled', True))}/hard={bool(profile.get('quality_gate_hard_reject', True))} "
            f"qprofit>={float(profile.get('quality_min_profit_net', 0.0)):.2f} "
            f"qsharpe>={float(profile.get('quality_min_holdout_sharpe', 0.0)):.4f} "
            f"qmc>={float(profile.get('quality_min_mc_sharpe_p50', -0.02)):.4f} "
            f"fee={float(profile.get('fee_per_trade', 0.0)):.2f} "
            f"slippage_bps={float(profile.get('slippage_bps', 0.0)):.2f} "
            f"chain={bool(profile.get('candidate_chain_enabled', True))} "
            f"crit={profile.get('candidate_selection_criterion')} "
            f"topN={int(profile.get('candidate_top_n', 5))} "
            f"fresh={float(profile.get('candidate_fresh_ratio', 0.30)):.2f}"
        )
        meta_extra["mc_block_len"] = mc_block
        meta_extra["label_horizon_bars"] = int(self._label_horizon_bars)
        meta_extra["label_take_profit_bps"] = float(self._label_take_profit_bps)
        meta_extra["label_stop_loss_bps"] = float(self._label_stop_loss_bps)
        meta_extra["label_same_bar_policy"] = "neutral"
        meta_extra["label_lookahead_bars"] = int(self._label_horizon_bars) if self._is_ternary_target else 1
        meta_extra["holdout_mode"] = "pct"
        meta_extra["holdout_pct"] = float(self.holdout_pct_default)
        meta_extra["holdout_min_bars"] = int(self.holdout_min_bars_default)
        meta_extra["holdout_max_bars"] = int(self.holdout_max_bars_default)
        meta_extra["training_mode"] = mode
        meta_extra["training_mode_requested"] = mode
        meta_extra["training_mode_compatibility"] = compatibility_mode
        meta_extra["training_mode_runtime"] = runtime_training_mode
        meta_extra["workflow_mode"] = workflow_mode
        meta_extra["training_profile"] = dict(profile)

        self.tbl.setRowCount(0)
        self.prog.setRange(0, 1)
        self.prog.setValue(0)
        self.worker = TrainWorker(
            df_full=df,
            holdout_bars=n_hold,
            estimator=est,
            name_prefix=name_prefix,
            meta_extra=meta_extra,
            holdout_pct=float(self.holdout_pct_default),
            holdout_min_bars=int(self.holdout_min_bars_default),
            holdout_max_bars=int(self.holdout_max_bars_default),
            training_profile=profile,
        )
        self.worker.progress.connect(self._on_progress_row)
        self.worker.phase.connect(lambda ph: None)
        self.worker.model_ready.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self._set_controls_running(True)
        self.worker.start()

    def _auto_search_state_path(
        self,
        profile: str,
        *,
        source_artifact_path: str | None = None,
        refresh_target_csv_path: str | None = None,
    ) -> Path:
        csv_src = self.csv_path or "dataset"
        stem = Path(csv_src).stem
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
        if not safe:
            safe = "dataset"
        profile_norm = _normalize_auto_search_profile(profile)
        state_dir = Path(_model_dir()) / "auto_search"
        if profile_norm == "refine":
            refine_source = str(source_artifact_path or "").strip()
            if refine_source:
                source_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(refine_source).stem).strip("_") or safe
                return state_dir / f"{source_safe}_refine_state.json"
        if profile_norm == "refresh":
            refresh_target = str(refresh_target_csv_path or "").strip()
            source_artifact = str(source_artifact_path or "").strip()
            if refresh_target or source_artifact:
                source_stem = Path(source_artifact).stem if source_artifact else safe
                source_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem).strip("_") or safe
                if refresh_target:
                    target_stem = Path(refresh_target).stem
                    target_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", target_stem).strip("_") or "dataset"
                    return state_dir / f"{source_safe}__to__{target_safe}_refresh_state.json"
                return state_dir / f"{source_safe}_refresh_state.json"
        prof_path = state_dir / f"{safe}_{profile_norm}_state.json"
        legacy_candidates = {
            "explore": [
                state_dir / f"{safe}_state.json",
                state_dir / f"{safe}_full_state.json",
            ],
            "refine": [state_dir / f"{safe}_fast_state.json"],
            "refresh": [state_dir / f"{safe}_weekly_state.json"],
        }
        candidates = [prof_path, *legacy_candidates.get(profile_norm, [])]
        existing = [path for path in candidates if path.exists()]
        if not existing:
            return prof_path
        return max(existing, key=_auto_search_state_score)

    def run_auto_search(self):
        if self.dataset is None or not self.csv_path:
            self.log.appendPlainText("WARN Nejprve vyber CSV.")
            return
        if self._is_training_running():
            self.log.appendPlainText("WARN Jiz probiha trenink nebo workflow. Pockej na dokonceni.")
            return

        auto_profile = self._current_auto_search_profile()
        source_artifact_path = self._resolve_source_artifact_path(auto_profile)
        refresh_target_csv = self.refresh_target_csv_path if auto_profile == "refresh" else None
        try:
            self._validate_auto_search_inputs(auto_profile, source_artifact_path, refresh_target_csv)
        except Exception as exc:
            self.log.appendPlainText(f"ERROR Workflow: {exc}")
            return
        state_path = self._auto_search_state_path(
            auto_profile,
            source_artifact_path=source_artifact_path.as_posix() if source_artifact_path is not None else None,
            refresh_target_csv_path=refresh_target_csv,
        )
        profiles = {
            "explore": self._apply_search_backend_profile_overrides(self._training_profile_for_mode("explore")),
            "refine": self._apply_search_backend_profile_overrides(self._training_profile_for_mode("refine")),
            "refresh": self._apply_search_backend_profile_overrides(self._training_profile_for_mode("refresh")),
        }
        top_n = int(max(1, self._current_candidate_top_n()))
        fresh_ratio = float(np.clip(self._current_candidate_fresh_ratio(), 0.05, 0.80))
        self._log_search_backend_hint()

        self.auto_worker = AutoSearchWorker(
            csv_path=self.csv_path,
            holdout_pct=float(self.holdout_pct_default),
            holdout_min_bars=int(self.holdout_min_bars_default),
            holdout_max_bars=int(self.holdout_max_bars_default),
            training_profiles=profiles,
            candidate_top_n=top_n,
            candidate_fresh_ratio=fresh_ratio,
            state_path=state_path.as_posix(),
            source_artifact_path=source_artifact_path.as_posix() if source_artifact_path is not None else None,
            refresh_csv_path=refresh_target_csv,
            search_profile=auto_profile,
        )
        self.auto_worker.message.connect(self.log.appendPlainText)
        self.auto_worker.result.connect(self._on_auto_result)
        self.auto_worker.finished_state.connect(self._on_auto_finished_state)
        self.auto_worker.error.connect(self._on_auto_error)
        self.auto_worker.finished.connect(self._on_auto_worker_finished)
        self.log.appendPlainText(
            f"INFO Workflow start: mode={auto_profile} checkpoint={state_path.as_posix()} "
            f"| backend={self._selected_search_backend()} "
            f"optuna_trials={self._selected_optuna_trials()} "
            f"optuna_timeout={self._selected_optuna_timeout_seconds()}s "
            f"| topN={top_n} fresh={fresh_ratio:.2f}"
        )
        if auto_profile in {"refine", "refresh"}:
            self.log.appendPlainText(
                "INFO Workflow source: "
                f"artifact={(source_artifact_path.as_posix() if source_artifact_path is not None else 'auto')} "
                f"target_csv={refresh_target_csv or self.csv_path}"
            )
        self.auto_worker.start()
        self._set_controls_running(True)
        self.btn_auto_stop.setEnabled(True)

    def stop_auto_search(self):
        if self._is_auto_search_running() and self.auto_worker is not None:
            self.auto_worker.request_stop()
            self.btn_auto_stop.setEnabled(False)
            self.log.appendPlainText("INFO Workflow: stop requested (ulozim checkpoint a ukoncim beh).")

    def _on_auto_result(self, row: dict[str, Any]):
        if not isinstance(row, dict):
            return

        def _fmt(v: Any, nd: int = 4) -> str:
            try:
                if v is None:
                    return "n/a"
                fv = float(v)
                return f"{fv:.{nd}f}" if np.isfinite(fv) else "n/a"
            except Exception:
                return str(v)

        phase = row.get("phase")
        model = row.get("model")
        criterion = row.get("criterion")
        horizon = row.get("horizon")
        tp_bps = row.get("tp_bps")
        sl_bps = row.get("sl_bps")
        status = row.get("status")
        self.log.appendPlainText(
            "INFO Workflow result: "
            f"phase={phase} model={model} crit={criterion} "
            f"h={horizon} tp={tp_bps} sl={sl_bps} "
            f"status={status} profit_net={_fmt(row.get('profit_net'), 2)} "
            f"sharpe={_fmt(row.get('sharpe'), 4)} pf={_fmt(row.get('pf'), 4)} "
            f"trades={int(row.get('trades', 0) or 0)} "
            f"backend={row.get('search_backend_used') or 'n/a'} "
            f"requested={row.get('search_backend_requested') or 'n/a'} "
            f"fallback={row.get('search_backend_fallback_reason') or 'none'}"
        )
        if status != "ok":
            reasons = row.get("qg_reasons") or []
            self.log.appendPlainText(
                f"INFO Workflow reject/error: reasons={reasons} meta={row.get('meta_path') or 'n/a'}"
            )

    def _on_auto_finished_state(self, state_path: str, completed: bool):
        state = "completed" if completed else "paused"
        self.log.appendPlainText(f"INFO Workflow {state}: checkpoint={state_path}")

    def _on_auto_error(self, msg: str):
        self.log.appendPlainText(f"ERROR Workflow: {msg}")

    def _on_auto_worker_finished(self):
        self.auto_worker = None
        self._set_controls_running(False)
        self._refresh_train_button_text()

    def _name_and_meta_from_csv(self, path: str, n_total: int, n_train: int, n_hold: int):
        return runtime_name_and_meta_from_csv(path, n_total, n_train, n_hold)

    def _on_progress_row(self, idx: int, total: int, params: dict, mean_f1: float, std_f1: float):
        if self.prog.maximum() != total:
            self.prog.setRange(0, total)
        self.prog.setValue(idx)
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, QTableWidgetItem(str(idx)))
        self.tbl.setItem(r, 1, QTableWidgetItem(f"{mean_f1:.4f}"))
        self.tbl.setItem(r, 2, QTableWidgetItem(f"{std_f1:.4f}"))
        self.tbl.setItem(r, 3, QTableWidgetItem(jsonlib.dumps(params, ensure_ascii=False)))

    def _on_finished(self, model_path: str):
        try:
            if not model_path or not os.path.exists(model_path):
                self.log.appendPlainText("INFO Model ulozen, ale cesta nenalezena.")
                return
            self.log.appendPlainText(f"INFO Nacitam model: {os.path.basename(model_path)}")
            obj = joblib.load(model_path)
            if isinstance(obj, dict):
                mdl = obj.get("model") or obj
                thr = float(obj.get("decision_threshold", 0.45))
                thr_short = float(obj.get("ternary_threshold_short", thr))
                thr_long = float(obj.get("ternary_threshold_long", thr))
                feats = obj.get("features")
            else:
                mdl = obj
                thr = 0.5
                thr_short = thr
                thr_long = thr
                feats = None

            meta = {}
            try:
                meta_path = Path(model_path).with_name(Path(model_path).stem + "_meta.json")
                if meta_path.exists():
                    with meta_path.open("r", encoding="utf-8") as fh:
                        meta = jsonlib.load(fh)
                    feats = feats or meta.get("trained_features")
                    if isinstance(meta.get("decision_threshold"), (int, float)):
                        thr = float(meta["decision_threshold"])
                    if isinstance(meta.get("ternary_threshold_short"), (int, float)):
                        thr_short = float(meta["ternary_threshold_short"])
                    if isinstance(meta.get("ternary_threshold_long"), (int, float)):
                        thr_long = float(meta["ternary_threshold_long"])
                    n_total_meta = int(meta.get("n_total_bars", 0) or 0)
                    n_train_eff_meta = int(meta.get("n_train_bars", 0) or 0)
                    n_train_pre_meta = int(meta.get("n_train_bars_pre_guard", n_train_eff_meta) or n_train_eff_meta)
                    n_train_core_meta = int(meta.get("n_train_core_bars", 0) or 0)
                    n_cal_meta = int(meta.get("n_threshold_calibration_bars", 0) or 0)
                    n_hold_meta = int(meta.get("n_holdout_bars", 0) or 0)
                    self.log.appendPlainText(
                        f"INFO Meta: timeframe={meta.get('timeframe')} total/train_eff/hold="
                        f"{n_total_meta}/{n_train_eff_meta}/{n_hold_meta}"
                    )
                    self.log.appendPlainText(
                        f"INFO Train bars detail: pre_guard={n_train_pre_meta} "
                        f"effective={n_train_eff_meta} core={n_train_core_meta} calib={n_cal_meta}"
                    )
                    hs = meta.get("holdout_selection") or {}
                    if isinstance(hs, dict) and hs:
                        self.log.appendPlainText(
                            "INFO Holdout split: "
                            f"mode={hs.get('mode')} requested_bars={hs.get('requested_bars')} "
                            f"requested_pct={hs.get('requested_pct')} applied_bars={hs.get('applied_bars')} "
                            f"min={hs.get('min_bars')} max={hs.get('max_bars')} guard_train_min={hs.get('min_train_bars_guard')}"
                        )
                    tc = meta.get("threshold_calibration_selection") or {}
                    if isinstance(tc, dict) and tc.get("enabled"):
                        self.log.appendPlainText(
                            "INFO Threshold calibration: "
                            f"mode={tc.get('mode')} requested_bars={tc.get('requested_bars')} "
                            f"requested_pct={tc.get('requested_pct')} applied_bars={tc.get('applied_bars')} "
                            f"train_core={tc.get('train_core_bars')} train_full={tc.get('train_full_bars')} "
                            f"guard={tc.get('train_min_guard')}"
                        )
                    if meta.get("annualize_sharpe") is True:
                        self.log.appendPlainText("INFO Sharpe je anualizovany (prepocet na rok).")
                    self.log.appendPlainText(
                        f"INFO Labeling: horizon={meta.get('label_horizon_bars')} "
                        f"tp_bps={meta.get('label_take_profit_bps')} "
                        f"sl_bps={meta.get('label_stop_loss_bps')} "
                        f"lookahead={meta.get('label_lookahead_bars')} embargo_eff={meta.get('effective_embargo')} "
                        f"same_bar={meta.get('label_same_bar_policy')}"
                    )
                    cd = meta.get("class_distribution") or {}
                    if isinstance(cd, dict) and cd:
                        self.log.appendPlainText(
                            f"INFO Class dist all/train/hold: {cd.get('all')} / {cd.get('train')} / {cd.get('holdout')}"
                        )
                    qg = meta.get("quality_gate") or {}
                    if isinstance(qg, dict) and qg.get("enabled"):
                        qg_passed = qg.get("passed")
                        qg_state = "PASS" if qg_passed is True else ("FAIL" if qg_passed is False else "N/A")
                        self.log.appendPlainText(
                            f"INFO Quality gate: {qg_state} "
                            f"(min_f1_lift={qg.get('min_f1_lift')}, min_trades={qg.get('min_trades')}, "
                            f"min_profit_net={qg.get('min_profit_net')}, min_sharpe={qg.get('min_holdout_sharpe')}) "
                            f"reasons={qg.get('reasons') or []}"
                        )
                    sp = meta.get("search_plan") or {}
                    if isinstance(sp, dict) and sp:
                        self.log.appendPlainText(
                            "INFO Search plan: "
                            f"backend={sp.get('search_backend_used')} "
                            f"requested={sp.get('search_backend_requested')} "
                            f"fallback={sp.get('search_backend_fallback_reason')} "
                            f"optuna_trials={sp.get('optuna_trials_effective') or sp.get('optuna_trials_requested')} "
                            f"optuna_timeout={sp.get('optuna_timeout_seconds')} "
                            f"optuna_done={sp.get('optuna_completed_trials')} "
                            f"optuna_pruned={sp.get('optuna_pruned_trials')} "
                            f"grid_total={sp.get('grid_total_candidates')} "
                            f"grid_used={sp.get('grid_used_candidates')} "
                            f"sampled={sp.get('sampled_candidates')} "
                            f"seed={sp.get('param_sample_seed')}"
                        )
                        cc = sp.get("candidate_chain") or {}
                        if isinstance(cc, dict) and cc:
                            src_crit = cc.get("source_criterion")
                            reranked = bool(cc.get("reranked_with_current_criterion", False))
                            extra = ""
                            if src_crit:
                                extra += f" src_crit={src_crit}"
                            if reranked:
                                extra += " reranked=True"
                            self.log.appendPlainText(
                                "INFO Candidate chain: "
                                f"mode={cc.get('mode')} source={cc.get('source_mode')} "
                                f"used={cc.get('used', False)} "
                                f"carry={cc.get('carry_count', 0)} "
                                f"fresh={cc.get('fresh_count', 0)} "
                                f"criterion={cc.get('criterion')} top_n={cc.get('top_n')} "
                                f"fresh_ratio={cc.get('fresh_ratio')}"
                                f"{extra}"
                            )
                    mc = meta.get("mc") or {}
                    if mc:
                        s = mc.get("sharpe", {})
                        dd = mc.get("max_drawdown", {})
                        note = mc.get("note")
                        self.log.appendPlainText(
                            f"MC: Sharpe p50={s.get('p50', 'n/a')} "
                            f"[{s.get('p10', 'n/a')} - {s.get('p90', 'n/a')}], "
                            f"MaxDD p90={dd.get('p90', 'n/a')}, iters={mc.get('iters')} block={mc.get('block_len')}"
                            + (f" ({note})" if note else "")
                        )
                    met = meta.get("metrics") or {}
                    if "sharpe" in met:
                        self.log.appendPlainText(
                            f"HOLDOUT Sharpe: {met['sharpe']:.4f} (annualized={bool(meta.get('annualize_sharpe'))})"
                        )
                    self._log_holdout_key_metrics(meta)
                    tt = meta.get("threshold_tuning") or {}
                    if isinstance(tt, dict) and tt:
                        fs = tt.get("oof_selected_final") or {}
                        self.log.appendPlainText(
                            "INFO Threshold tuning: "
                            f"mode={tt.get('selected_mode', 'n/a')} base={tt.get('selected_mode_base', 'n/a')} source={tt.get('source', 'n/a')} "
                            f"oof_dir={fs.get('n_dir', 'n/a')} "
                            f"short={fs.get('n_short', 'n/a')} long={fs.get('n_long', 'n/a')} "
                            f"dom={fs.get('dominance', 'n/a')}"
                        )
                    self.log.appendPlainText(
                        f"INFO Thresholds: decision={thr:.3f}, short={thr_short:.3f}, long={thr_long:.3f}"
                    )
            except Exception:
                pass

            if mdl is not None and self.X_test is not None and self.y_test is not None:
                X_eval = self.X_test.reindex(columns=feats, fill_value=0.0) if feats else self.X_test
                X_eval_use = _align_X_for_estimator(mdl, X_eval)
                if hasattr(mdl, "predict_proba"):
                    pr = _call_with_feature_name_warning_suppressed(mdl.predict_proba, X_eval_use)
                    if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 3:
                        p_short = pr[:, 0]
                        p_long = pr[:, 2]
                        y_pred = _ternary_predict_mapped(p_short, p_long, float(thr_short), float(thr_long))
                    else:
                        p1 = (
                            pr[:, 1]
                            if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 2
                            else np.asarray(pr).ravel()
                        )
                        y_pred = (p1 >= thr).astype(int)
                elif hasattr(mdl, "decision_function"):
                    z = np.asarray(mdl.decision_function(X_eval_use)).ravel()
                    p1 = 1.0 / (1.0 + np.exp(-z))
                    y_pred = (p1 >= thr).astype(int)
                else:
                    y_pred = _call_with_feature_name_warning_suppressed(mdl.predict, X_eval_use)

                acc = accuracy_score(self.y_test, y_pred)
                if len(np.unique(np.asarray(self.y_test))) >= 3:
                    prec = precision_score(self.y_test, y_pred, average="macro", zero_division=0)
                    rec = recall_score(self.y_test, y_pred, average="macro", zero_division=0)
                    f1 = f1_score(self.y_test, y_pred, average="macro", zero_division=0)
                else:
                    prec = precision_score(self.y_test, y_pred, zero_division=0)
                    rec = recall_score(self.y_test, y_pred, zero_division=0)
                    f1 = f1_score(self.y_test, y_pred, zero_division=0)
                self.log.appendPlainText(
                    f"HOLDOUT (poslednich {len(self.y_test)} baru): "
                    f"Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}"
                )

        except Exception as e:
            self.log.appendPlainText(f"ERROR Chyba pri dokonceni/evaluaci: {e}")

    def _on_error(self, msg: str):
        if "QUALITY_GATE_REJECT" in str(msg):
            self.log.appendPlainText(f"ERROR Model odmitnut quality gate: {msg}")
            self._log_reject_summary_from_diag_meta(str(msg))
        else:
            self.log.appendPlainText(f"ERROR Chyba trenovani: {msg}")

    def _on_worker_finished(self):
        self.worker = None
        self._set_controls_running(False)
        self._refresh_train_button_text()

    def shutdown(self) -> bool:
        ok = True
        if self.auto_worker is not None:
            try:
                self.auto_worker.request_stop()
            except Exception:
                pass
            if self.auto_worker.isRunning() and not self.auto_worker.wait(3000):
                ok = False
            elif not self.auto_worker.isRunning():
                try:
                    self.auto_worker.deleteLater()
                except Exception:
                    pass
                self.auto_worker = None
        if self.worker is not None:
            if self.worker.isRunning():
                ok = False
            else:
                try:
                    self.worker.deleteLater()
                except Exception:
                    pass
                self.worker = None
        return ok

    def _log_reject_summary_from_diag_meta(self, msg: str):
        try:
            m = re.search(r"\|\s*diag_meta=(.+)$", str(msg))
            if not m:
                return
            meta_path = m.group(1).strip().strip('"').strip("'")
            if not meta_path:
                return
            p = Path(meta_path)
            if not p.exists():
                return
            with p.open("r", encoding="utf-8") as fh:
                diag = jsonlib.load(fh)
        except Exception:
            return

        def _f(v, nd: int = 4) -> str:
            try:
                if v is None:
                    return "n/a"
                fv = float(v)
                return f"{fv:.{nd}f}"
            except Exception:
                return str(v)

        try:
            mh = (diag.get("metrics_holdout") or {}) if isinstance(diag, dict) else {}
            qg = (diag.get("quality_gate") or {}) if isinstance(diag, dict) else {}
            mc = (diag.get("mc_summary") or {}) if isinstance(diag, dict) else {}
            tt = (diag.get("threshold_tuning") or {}) if isinstance(diag, dict) else {}
            pc = (mh.get("per_class_3") or {}) if isinstance(mh, dict) else {}
            rs = ((pc.get("-1") or {}).get("recall")) if isinstance(pc, dict) else None
            rl = ((pc.get("1") or {}).get("recall")) if isinstance(pc, dict) else None
            mc_p50 = None
            if isinstance(mc, dict):
                mc_sh = mc.get("sharpe") or {}
                if isinstance(mc_sh, dict):
                    mc_p50 = mc_sh.get("p50")

            self.log.appendPlainText(
                "INFO Reject summary: "
                f"profit_net={_f(mh.get('profit_net'), 2)} | "
                f"sharpe={_f(mh.get('sharpe'), 4)} | "
                f"trades={int(mh.get('num_trades', mh.get('trades', 0)) or 0)} "
                f"(short={int(mh.get('num_trades_short', 0) or 0)}, long={int(mh.get('num_trades_long', 0) or 0)}) | "
                f"rec_short={_f(rs, 4)} rec_long={_f(rl, 4)} | "
                f"mc_p50={_f(mc_p50, 4)} | "
                f"thr_short={_f(diag.get('ternary_threshold_short'), 3)} "
                f"thr_long={_f(diag.get('ternary_threshold_long'), 3)} | "
                f"mode={tt.get('selected_mode', 'n/a')} | "
                f"reasons={qg.get('reasons') or []}"
            )
            chunks = qg.get("holdout_chunks") if isinstance(qg, dict) else None
            if isinstance(chunks, list) and chunks:
                parts: list[str] = []
                for chunk in chunks[:3]:
                    if not isinstance(chunk, dict):
                        continue
                    pb = chunk.get("prediction_balance") if isinstance(chunk.get("prediction_balance"), dict) else {}
                    parts.append(
                        f"c{int(chunk.get('chunk_index', len(parts) + 1) or (len(parts) + 1))} "
                        f"pnl={_f(chunk.get('profit_net'), 2)} "
                        f"trades={int(chunk.get('num_trades', 0) or 0)} "
                        f"S/L={int(pb.get('n_short', 0) or 0)}/{int(pb.get('n_long', 0) or 0)}"
                    )
                if parts:
                    self.log.appendPlainText("INFO Reject holdout chunks: " + " | ".join(parts))
        except Exception:
            return

    def _log_holdout_key_metrics(self, meta: dict[str, Any]):
        if not isinstance(meta, dict):
            return

        def _f(v, nd: int = 4) -> str:
            try:
                if v is None:
                    return "n/a"
                fv = float(v)
                return f"{fv:.{nd}f}"
            except Exception:
                return str(v)

        try:
            mh = meta.get("metrics") or {}
            if not isinstance(mh, dict) or not mh:
                return

            pc = mh.get("per_class_3") or {}
            rs = ((pc.get("-1") or {}).get("recall")) if isinstance(pc, dict) else None
            rl = ((pc.get("1") or {}).get("recall")) if isinstance(pc, dict) else None

            mc = meta.get("mc") or {}
            mc_p50 = None
            if isinstance(mc, dict):
                mc_sh = mc.get("sharpe") or {}
                if isinstance(mc_sh, dict):
                    mc_p50 = mc_sh.get("p50")

            self.log.appendPlainText(
                "INFO Holdout key: "
                f"profit_net={_f(mh.get('profit_net'), 2)} | "
                f"trades={int(mh.get('num_trades', mh.get('trades', 0)) or 0)} "
                f"(short={int(mh.get('num_trades_short', 0) or 0)}, long={int(mh.get('num_trades_long', 0) or 0)}) | "
                f"pf={_f(mh.get('pf'), 4)} | "
                f"max_dd={_f(mh.get('max_dd'), 2)} | "
                f"rec_short={_f(rs, 4)} rec_long={_f(rl, 4)} | "
                f"mc_p50={_f(mc_p50, 4)}"
            )

            qg = meta.get("quality_gate") or {}
            chunks = qg.get("holdout_chunks") if isinstance(qg, dict) else None
            if isinstance(chunks, list) and chunks:
                parts: list[str] = []
                for chunk in chunks[:3]:
                    if not isinstance(chunk, dict):
                        continue
                    pb = chunk.get("prediction_balance") if isinstance(chunk.get("prediction_balance"), dict) else {}
                    parts.append(
                        f"c{int(chunk.get('chunk_index', len(parts) + 1) or (len(parts) + 1))} "
                        f"pnl={_f(chunk.get('profit_net'), 2)} "
                        f"trades={int(chunk.get('num_trades', 0) or 0)} "
                        f"S/L={int(pb.get('n_short', 0) or 0)}/{int(pb.get('n_long', 0) or 0)}"
                    )
                if parts:
                    self.log.appendPlainText("INFO Holdout chunks: " + " | ".join(parts))
        except Exception:
            return

    def _log_dataset_audit(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        try:
            ts = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
            ts_valid = ts.dropna()
            if ts_valid.empty:
                return
            n_rows = int(len(df))
            span_days = float((ts_valid.iloc[-1] - ts_valid.iloc[0]).total_seconds() / 86400.0)
            dt_min = ts_valid.diff().dropna().dt.total_seconds() / 60.0
            med_step = float(dt_min.median()) if not dt_min.empty else float("nan")
            gap_ratio = 0.0
            if np.isfinite(med_step) and med_step > 0.0 and not dt_min.empty:
                gap_ratio = float((dt_min > (2.0 * med_step)).mean())
            self.log.appendPlainText(
                f"INFO Data audit: rows={n_rows} span_days={span_days:.1f} "
                f"median_step_min={med_step:.2f} gap_ratio_gt_2x={gap_ratio:.3f}"
            )
            if n_rows < 5000:
                self.log.appendPlainText("WARN Data audit: dataset je kratky (<5000 radku).")
            if gap_ratio > 0.05:
                self.log.appendPlainText("WARN Data audit: vyssi podil casovych mezer (>5%).")

            if "close" in df.columns:
                close = pd.to_numeric(df["close"], errors="coerce")
                close_ok = close.dropna()
                if not close_ok.empty:
                    c_min = float(close_ok.min())
                    c_med = float(close_ok.median())
                    c_max = float(close_ok.max())
                    non_pos = int((close_ok <= 0.0).sum())
                    self.log.appendPlainText(
                        f"INFO Data audit price: close_min={c_min:.4f} close_med={c_med:.4f} close_max={c_max:.4f}"
                    )
                    if non_pos > 0:
                        self.log.appendPlainText(
                            f"WARN Data audit: close<=0 count={non_pos} (mozna transformovana/nekvalitni cena)."
                        )

            req_cols = {"open", "high", "low", "close"}
            if req_cols.issubset(set(df.columns)):
                o = pd.to_numeric(df["open"], errors="coerce")
                h = pd.to_numeric(df["high"], errors="coerce")
                l = pd.to_numeric(df["low"], errors="coerce")
                c = pd.to_numeric(df["close"], errors="coerce")
                bad_hl = int((h < l).sum())
                bad_oc = int(((o < l) | (o > h) | (c < l) | (c > h)).sum())
                if bad_hl > 0 or bad_oc > 0:
                    self.log.appendPlainText(
                        f"WARN Data audit OHLC: high<low={bad_hl}, open/close_out_of_range={bad_oc}"
                    )
        except Exception:
            return
