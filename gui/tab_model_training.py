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
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ibkr_trading_bot.core.services.dataset_service import DatasetService
from ibkr_trading_bot.core.services.futures_roll_chain_service import read_dataset_sidecar_meta
from ibkr_trading_bot.core.services.model_training_service import (
    compute_holdout_bars as runtime_compute_holdout_bars,
    name_and_meta_from_csv as runtime_name_and_meta_from_csv,
    run_training_job,
    training_profile_for_mode,
)
from ibkr_trading_bot.model.train_models import (
    _model_dir,
    _select_feature_columns,
    _ternary_predict_mapped,
    train_and_evaluate_model,
)


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
            self.phase.emit("grid")

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
            quality_require_mc_nonnegative = bool(profile.get("quality_require_mc_nonnegative", True))
            quality_min_mc_sharpe_p50 = float(profile.get("quality_min_mc_sharpe_p50", -0.02))
            quality_min_profit_net = float(profile.get("quality_min_profit_net", 0.0))
            quality_min_holdout_sharpe = float(profile.get("quality_min_holdout_sharpe", 0.0))
            training_mode = str(profile.get("training_mode", "standard")).strip().lower()
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
                label_lookahead_bars=int(self.meta_extra.get("label_lookahead_bars", 0)),
                quality_gate_enabled=quality_gate_enabled,
                quality_gate_hard_reject=quality_gate_hard_reject,
                quality_min_trades=quality_min_trades,
                quality_min_side_recall=quality_min_side_recall,
                quality_require_mc_nonnegative=quality_require_mc_nonnegative,
                quality_min_mc_sharpe_p50=quality_min_mc_sharpe_p50,
                quality_min_profit_net=quality_min_profit_net,
                quality_min_holdout_sharpe=quality_min_holdout_sharpe,
                max_param_candidates=max_param_candidates,
                param_sample_seed=param_sample_seed,
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
        search_profile: str = "fast",
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
        self.search_profile = self._normalize_search_profile(search_profile)
        self._stop_requested = False

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
        p = str(profile or "").strip().lower()
        return p if p in {"fast", "full", "weekly"} else "fast"

    def _build_spec(self) -> dict[str, Any]:
        # full keeps the original exhaustive search; fast trims redundant branches.
        if self.search_profile == "weekly":
            return {
                "version": 1,
                "search_profile": "weekly",
                "quick_models": ["lgb", "hgbt", "xgb", "et", "rf"],
                "criteria": ["profit_first"],
                "label_horizon_bars": [8, 12, 16],
                "label_tp_bps": [40.0, 50.0, 60.0],
                "label_sl_bps": [40.0, 50.0, 60.0],
                "promote_top_k": 10,
            }
        if self.search_profile == "full":
            return {
                "version": 1,
                "search_profile": "full",
                "quick_models": ["lgb", "hgbt"],
                "criteria": ["balanced", "profit_first", "robustness_first", "recall_balance"],
                "label_horizon_bars": [8, 12, 16],
                "label_tp_bps": [40.0, 50.0, 60.0],
                "label_sl_bps": [40.0, 50.0, 60.0],
                "promote_top_k": 8,
            }
        return {
            "version": 1,
            "search_profile": "fast",
            "quick_models": ["lgb", "hgbt"],
            "criteria": ["profit_first"],
            "label_horizon_bars": [8, 12],
            "label_tp_bps": [40.0, 50.0, 60.0],
            "label_sl_bps": [40.0, 50.0, 60.0],
            "promote_top_k": 5,
        }

    def _new_state(self) -> dict[str, Any]:
        spec = self._build_spec()
        quick_queue: list[dict[str, Any]] = []
        for model_name, horizon, tp_bps, sl_bps in product(
            spec["quick_models"],
            spec["label_horizon_bars"],
            spec["label_tp_bps"],
            spec["label_sl_bps"],
        ):
            quick_queue.append(
                {
                    "phase": "quick",
                    "model": str(model_name),
                    "criterion": "balanced",
                    "horizon": int(horizon),
                    "tp_bps": float(tp_bps),
                    "sl_bps": float(sl_bps),
                }
            )
        return {
            "version": 1,
            "created_at": self._now_str(),
            "updated_at": self._now_str(),
            "csv_path": self.csv_path,
            "spec": spec,
            "phase": "quick",
            "quick_queue": quick_queue,
            "quick_idx": 0,
            "standard_queue": [],
            "standard_idx": 0,
            "results": [],
            "stopped": False,
            "completed": False,
        }

    def _save_state(self, state: dict[str, Any]):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = self._now_str()
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(jsonlib.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load_or_init_state(self) -> tuple[dict[str, Any], bool]:
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
        spec_expected = self._build_spec()
        if (
            not isinstance(st, dict)
            or str(st.get("csv_path")) != self.csv_path
            or st.get("spec") != spec_expected
            or st.get("phase") == "done"
        ):
            st = self._new_state()
            self._save_state(st)
            return st, False
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

    def _promote_to_standard(self, state: dict[str, Any]) -> dict[str, Any]:
        quick_rows = [r for r in list(state.get("results") or []) if str(r.get("phase")) == "quick"]
        quick_rows.sort(key=self._score_result, reverse=True)
        top_k = int(max(1, int((state.get("spec") or {}).get("promote_top_k", 8))))
        picked_base: list[dict[str, Any]] = []
        seen: set[tuple[str, int, float, float]] = set()
        for r in quick_rows:
            key = (
                str(r.get("model", "")),
                int(r.get("horizon", 0) or 0),
                float(r.get("tp_bps", 0.0) or 0.0),
                float(r.get("sl_bps", 0.0) or 0.0),
            )
            if key in seen:
                continue
            seen.add(key)
            picked_base.append(
                {
                    "model": key[0],
                    "horizon": key[1],
                    "tp_bps": key[2],
                    "sl_bps": key[3],
                }
            )
            if len(picked_base) >= top_k:
                break

        criteria = list((state.get("spec") or {}).get("criteria") or ["balanced"])
        std_queue: list[dict[str, Any]] = []
        for b in picked_base:
            for crit in criteria:
                std_queue.append(
                    {
                        "phase": "standard",
                        "model": str(b["model"]),
                        "criterion": str(crit),
                        "horizon": int(b["horizon"]),
                        "tp_bps": float(b["tp_bps"]),
                        "sl_bps": float(b["sl_bps"]),
                    }
                )
        state["phase"] = "standard"
        state["standard_queue"] = std_queue
        state["standard_idx"] = 0
        return state

    @staticmethod
    def _meta_metrics(meta: dict[str, Any]) -> dict[str, Any]:
        mh = (meta.get("metrics_holdout") or meta.get("metrics") or {}) if isinstance(meta, dict) else {}
        if not isinstance(mh, dict):
            mh = {}
        return mh

    def _train_one(self, cfg: dict[str, Any]) -> dict[str, Any]:
        phase = str(cfg.get("phase", "quick"))
        estimator_name = str(cfg.get("model", "lgb")).strip().lower()
        criterion = str(cfg.get("criterion", "balanced")).strip().lower()
        horizon = int(cfg.get("horizon", 12))
        tp_bps = float(cfg.get("tp_bps", 50.0))
        sl_bps = float(cfg.get("sl_bps", 50.0))
        return run_training_job(
            csv_path=self.csv_path,
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
        )

    def run(self):
        try:
            state, resumed = self._load_or_init_state()
            spec = (state.get("spec") or {}) if isinstance(state, dict) else {}
            search_profile = str(spec.get("search_profile") or self.search_profile)
            self.message.emit(
                f"INFO Auto-search {'resume' if resumed else 'start'}: {self.state_path.as_posix()} "
                f"| profile={search_profile} "
                f"| phase={state.get('phase')} quick={state.get('quick_idx', 0)}/{len(state.get('quick_queue') or [])} "
                f"standard={state.get('standard_idx', 0)}/{len(state.get('standard_queue') or [])}"
            )

            while not self._stop_requested:
                phase = str(state.get("phase") or "quick")
                if phase == "quick":
                    q = list(state.get("quick_queue") or [])
                    i = int(state.get("quick_idx", 0))
                    if i >= len(q):
                        state = self._promote_to_standard(state)
                        self._save_state(state)
                        self.message.emit(
                            f"INFO Auto-search promote: quick_done={len(q)} -> standard_queue={len(state.get('standard_queue') or [])}"
                        )
                        continue
                    cfg = dict(q[i])
                    self.message.emit(
                        f"INFO Auto-search run [{i+1}/{len(q)}] phase=quick "
                        f"model={cfg.get('model')} horizon={cfg.get('horizon')} tp={cfg.get('tp_bps')} sl={cfg.get('sl_bps')}"
                    )
                    row = self._train_one(cfg)
                    state.setdefault("results", []).append(row)
                    state["quick_idx"] = i + 1
                    self._save_state(state)
                    self.result.emit(dict(row))
                    continue

                if phase == "standard":
                    q = list(state.get("standard_queue") or [])
                    i = int(state.get("standard_idx", 0))
                    if i >= len(q):
                        state["phase"] = "done"
                        state["completed"] = True
                        self._save_state(state)
                        break
                    cfg = dict(q[i])
                    self.message.emit(
                        f"INFO Auto-search run [{i+1}/{len(q)}] phase=standard "
                        f"model={cfg.get('model')} criterion={cfg.get('criterion')} "
                        f"horizon={cfg.get('horizon')} tp={cfg.get('tp_bps')} sl={cfg.get('sl_bps')}"
                    )
                    row = self._train_one(cfg)
                    state.setdefault("results", []).append(row)
                    state["standard_idx"] = i + 1
                    self._save_state(state)
                    self.result.emit(dict(row))
                    continue

                break

            if self._stop_requested:
                state["stopped"] = True
                state["completed"] = False
                self._save_state(state)

            completed = bool(state.get("phase") == "done" and state.get("completed"))
            self.finished_state.emit(self.state_path.as_posix(), completed)
        except Exception as e:
            self.error.emit(str(e))


class ModelTrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset: pd.DataFrame | None = None
        self.csv_path: str | None = None
        self.worker: TrainWorker | None = None
        self.auto_worker: AutoSearchWorker | None = None
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

        box2 = QGroupBox("2) Auto-search modelu")
        box2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay2 = QVBoxLayout(box2)

        row = QHBoxLayout()
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["hgbt", "rf", "et", "xgb", "lgb", "svm"])
        self.cmb_model.hide()
        self.cmb_training_mode = QComboBox()
        self.cmb_training_mode.addItems(["quick", "standard", "strict"])
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
        row.addWidget(QLabel("Auto profil:"))
        self.cmb_auto_search_profile = QComboBox()
        self.cmb_auto_search_profile.addItems(["fast", "full", "weekly"])
        self.cmb_auto_search_profile.setCurrentText("fast")
        row.addWidget(self.cmb_auto_search_profile)
        row.addStretch(1)

        self.btn_train = QPushButton("Trenovat (standard)")
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self.run_training)
        self.btn_train.hide()
        self.btn_auto_search = QPushButton("Auto-search (resume)")
        self.btn_auto_search.setEnabled(False)
        self.btn_auto_search.clicked.connect(self.run_auto_search)
        row.addWidget(self.btn_auto_search)
        self.btn_auto_stop = QPushButton("Stop auto")
        self.btn_auto_stop.setEnabled(False)
        self.btn_auto_stop.clicked.connect(self.stop_auto_search)
        row.addWidget(self.btn_auto_stop)
        lay2.addLayout(row)

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

    def _current_training_mode(self) -> str:
        txt = (self.cmb_training_mode.currentText() or "").strip().lower()
        return txt if txt in {"quick", "standard", "strict"} else "standard"

    def _training_profile_for_mode(self, mode: str) -> dict[str, Any]:
        return training_profile_for_mode(mode)

    def _refresh_train_button_text(self):
        mode = self._current_training_mode()
        running = self._is_training_running()
        if self.dataset is None:
            self.btn_train.setText(f"Trenovat ({mode})")
            self.btn_train.setEnabled(False)
            self.btn_auto_search.setText("Auto-search (resume)")
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
        self.btn_auto_search.setText(f"Auto-search [{profile}] (resume) [{n_rows} rows, holdout {n_hold}]")
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
        self.cmb_auto_search_profile.setEnabled(not running)
        self.btn_train.setEnabled((not running) and (self.dataset is not None))
        self.btn_auto_search.setEnabled((not running) and (self.dataset is not None))
        self.btn_auto_stop.setEnabled(self._is_auto_search_running())

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
        txt = (self.cmb_auto_search_profile.currentText() or "").strip().lower()
        return txt if txt in {"fast", "full", "weekly"} else "fast"

    def _compute_holdout_bars(self, n_total: int) -> int:
        return runtime_compute_holdout_bars(
            int(n_total),
            float(self.holdout_pct_default),
            int(self.holdout_min_bars_default),
            int(self.holdout_max_bars_default),
        )

    def pick_csv(self):
        base_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
        path, _ = QFileDialog.getOpenFileName(self, "Vyber CSV s daty", base_dir.as_posix(), "CSV Files (*.csv)")
        if not path:
            return
        self.csv_path = path
        self.lbl_csv.setText(f"Vybrany soubor: {os.path.basename(path)}")

        try:
            svc = DatasetService()
            df = svc.prepare_from_csv(
                path,
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
            dataset_meta = read_dataset_sidecar_meta(path)
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
            self.log.appendPlainText(f"OK Nacteno: {path} | radku={n_rows}")
            self._log_dataset_audit(df)
            self._set_controls_running(False)
            self._refresh_train_button_text()
            self.tbl.setRowCount(0)
            self.prog.setRange(0, 1)
            self.prog.setValue(0)
        except Exception as e:
            self.log.appendPlainText(f"ERROR Chyba nacteni/pripravy dat: {e}")
            self._set_controls_running(False)

    def run_training(self):
        if self.dataset is None:
            self.log.appendPlainText("WARN Nejprve vyber CSV.")
            return
        if self._is_training_running():
            self.log.appendPlainText("WARN Jiz probiha trenink/auto-search. Pockej na dokonceni.")
            return

        est = self.cmb_model.currentText().strip().lower()
        mode = self._current_training_mode()
        profile = self._training_profile_for_mode(mode)
        profile["training_mode"] = mode
        profile["candidate_chain_enabled"] = True
        profile["candidate_selection_criterion"] = self._current_candidate_criterion()
        profile["candidate_top_n"] = int(max(1, self._current_candidate_top_n()))
        profile["candidate_fresh_ratio"] = float(
            np.clip(self._current_candidate_fresh_ratio(), 0.05, 0.80)
        )
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
            f"{mode} | cv={int(profile.get('n_splits', 5))} "
            f"top_k={int(profile.get('top_k_features', 12))} "
            f"grid_used<={profile.get('max_param_candidates')} "
            f"mc_enabled={bool(profile.get('mc_enabled', True))} "
            f"mc_iters={int(profile.get('mc_iters', 200))} "
            f"qgate={bool(profile.get('quality_gate_enabled', True))}/hard={bool(profile.get('quality_gate_hard_reject', True))} "
            f"qprofit>={float(profile.get('quality_min_profit_net', 0.0)):.2f} "
            f"qsharpe>={float(profile.get('quality_min_holdout_sharpe', 0.0)):.4f} "
            f"qmc>={float(profile.get('quality_min_mc_sharpe_p50', -0.02)):.4f} "
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

    def _auto_search_state_path(self, profile: str) -> Path:
        csv_src = self.csv_path or "dataset"
        stem = Path(csv_src).stem
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
        if not safe:
            safe = "dataset"
        profile_norm = str(profile or "").strip().lower()
        if profile_norm not in {"fast", "full", "weekly"}:
            profile_norm = "fast"
        state_dir = Path(_model_dir()) / "auto_search"
        prof_path = state_dir / f"{safe}_{profile_norm}_state.json"
        legacy_path = state_dir / f"{safe}_state.json"
        # Backward compatibility: older runs stored only one state file without profile suffix.
        if profile_norm == "full" and legacy_path.exists() and not prof_path.exists():
            return legacy_path
        return prof_path

    def run_auto_search(self):
        if self.dataset is None or not self.csv_path:
            self.log.appendPlainText("WARN Nejprve vyber CSV.")
            return
        if self._is_training_running():
            self.log.appendPlainText("WARN Jiz probiha trenink/auto-search. Pockej na dokonceni.")
            return

        auto_profile = self._current_auto_search_profile()
        state_path = self._auto_search_state_path(auto_profile)
        profiles = {
            "quick": self._training_profile_for_mode("quick"),
            "standard": self._training_profile_for_mode("standard"),
        }
        top_n = int(max(1, self._current_candidate_top_n()))
        fresh_ratio = float(np.clip(self._current_candidate_fresh_ratio(), 0.05, 0.80))

        self.auto_worker = AutoSearchWorker(
            csv_path=self.csv_path,
            holdout_pct=float(self.holdout_pct_default),
            holdout_min_bars=int(self.holdout_min_bars_default),
            holdout_max_bars=int(self.holdout_max_bars_default),
            training_profiles=profiles,
            candidate_top_n=top_n,
            candidate_fresh_ratio=fresh_ratio,
            state_path=state_path.as_posix(),
            search_profile=auto_profile,
        )
        self.auto_worker.message.connect(self.log.appendPlainText)
        self.auto_worker.result.connect(self._on_auto_result)
        self.auto_worker.finished_state.connect(self._on_auto_finished_state)
        self.auto_worker.error.connect(self._on_auto_error)
        self.auto_worker.finished.connect(self._on_auto_worker_finished)
        self.log.appendPlainText(
            f"INFO Auto-search start: profile={auto_profile} checkpoint={state_path.as_posix()} "
            f"| topN={top_n} fresh={fresh_ratio:.2f}"
        )
        self.auto_worker.start()
        self._set_controls_running(True)
        self.btn_auto_stop.setEnabled(True)

    def stop_auto_search(self):
        if self._is_auto_search_running() and self.auto_worker is not None:
            self.auto_worker.request_stop()
            self.log.appendPlainText("INFO Auto-search: stop requested (ulozim checkpoint a ukoncim beh).")

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
            "INFO Auto result: "
            f"phase={phase} model={model} crit={criterion} "
            f"h={horizon} tp={tp_bps} sl={sl_bps} "
            f"status={status} profit_net={_fmt(row.get('profit_net'), 2)} "
            f"sharpe={_fmt(row.get('sharpe'), 4)} pf={_fmt(row.get('pf'), 4)} "
            f"trades={int(row.get('trades', 0) or 0)}"
        )
        if status != "ok":
            reasons = row.get("qg_reasons") or []
            self.log.appendPlainText(
                f"INFO Auto reject/error: reasons={reasons} meta={row.get('meta_path') or 'n/a'}"
            )

    def _on_auto_finished_state(self, state_path: str, completed: bool):
        state = "completed" if completed else "paused"
        self.log.appendPlainText(f"INFO Auto-search {state}: checkpoint={state_path}")

    def _on_auto_error(self, msg: str):
        self.log.appendPlainText(f"ERROR Auto-search: {msg}")

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
                if hasattr(mdl, "predict_proba"):
                    pr = mdl.predict_proba(X_eval)
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
                    z = np.asarray(mdl.decision_function(X_eval)).ravel()
                    p1 = 1.0 / (1.0 + np.exp(-z))
                    y_pred = (p1 >= thr).astype(int)
                else:
                    y_pred = mdl.predict(X_eval)

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
