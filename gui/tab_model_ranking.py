from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ibkr_trading_bot.core.services import model_eval_service as model_eval_runtime
from ibkr_trading_bot.core.services.model_service import model_sidecar_meta_path, read_sidecar_model_meta, write_sidecar_model_meta
from ibkr_trading_bot.core.services.model_training_service import (
    candidate_selection_criterion_from_meta,
    dataset_snapshot_signature_from_csv,
    dataset_snapshot_signature_from_meta,
    run_training_job,
    training_profile_for_mode,
)
from ibkr_trading_bot.gui.components.workers import TaskWorker
from ibkr_trading_bot.model.feature_stability import compute_feature_stability_score

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "model_outputs"
RANKING_NOTE_META_KEY = "model_ranking_note"
COL_MODEL = 0
COL_MODE = 1
COL_STATUS = 2
COL_PROFIT_OPT = 3
COL_PROFIT = 4
COL_ENTRY = 5
COL_EXIT = 6
COL_TRADES = 7
COL_SHARPE = 8
COL_FEATURES = 9
COL_STABILITY = 10
COL_CREATED = 11
COL_NOTE = 12

STABILITY_FILTER_ALL = "all"
STABILITY_FILTER_GOOD = "good"
STABILITY_FILTER_EXCELLENT = "excellent"
STABILITY_FILTER_LABELS = {
    STABILITY_FILTER_ALL: "All",
    STABILITY_FILTER_GOOD: "Good (>0.4)",
    STABILITY_FILTER_EXCELLENT: "Excellent (>0.5)",
}
STABILITY_FILTER_THRESHOLDS = {
    STABILITY_FILTER_GOOD: 0.4,
    STABILITY_FILTER_EXCELLENT: 0.5,
}


def _as_float(x: Any, default: float = float("-inf")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            value = x.strip().replace(",", ".")
            if not value:
                return default
            return float(value)
        return float(x)
    except Exception:
        return default


def _as_timestamp(created: Any, fallback_path: Path) -> float:
    try:
        if isinstance(created, datetime):
            return float(created.timestamp())
        if isinstance(created, (int, float)):
            return float(created)
        if isinstance(created, str) and created:
            try:
                return datetime.fromisoformat(created).timestamp()
            except Exception:
                return float(created)
    except Exception:
        pass
    try:
        return float(fallback_path.stat().st_mtime)
    except Exception:
        return 0.0


def _top_feature_from_meta(meta: dict[str, Any]) -> str:
    feat_imp = meta.get("feature_importance", {})
    if not isinstance(feat_imp, dict) or not feat_imp:
        return ""
    try:
        return str(max(feat_imp.keys(), key=lambda key: feat_imp[key]))[:12]
    except Exception:
        return ""


def _training_mode_label(meta: dict[str, Any]) -> str:
    value = str(meta.get("training_mode") or "").strip().lower()
    return value if value else "-"


def _ranking_note_from_meta(meta: dict[str, Any]) -> str:
    raw = str(meta.get(RANKING_NOTE_META_KEY) or "")
    return raw.replace("\r", " ").replace("\n", " ").strip()


def _table_item(
    text: str,
    *,
    editable: bool = False,
    alignment: int | None = None,
    tooltip: str | None = None,
) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    flags = item.flags()
    if not editable:
        flags &= ~Qt.ItemIsEditable
    item.setFlags(flags)
    if alignment is not None:
        item.setTextAlignment(int(alignment))
    if tooltip:
        item.setToolTip(tooltip)
    return item


def _directory_snapshot(dir_path: Path) -> tuple[tuple[str, int, int], ...]:
    if not dir_path.exists():
        return ()
    snapshot: list[tuple[str, int, int]] = []
    for path in dir_path.glob("*.pkl"):
        try:
            st = path.stat()
        except OSError:
            continue
        snapshot.append((path.name, int(st.st_size), int(st.st_mtime_ns)))
    snapshot.sort()
    return tuple(snapshot)


def _metrics_from_meta(meta: dict[str, Any]) -> dict[str, float]:
    metrics_raw = meta.get("metrics_holdout") or meta.get("metrics") or {}
    metrics: dict[str, float] = {}
    if not isinstance(metrics_raw, dict):
        return metrics
    for key, value in metrics_raw.items():
        try:
            if isinstance(value, (int, float)) and np.isfinite(value):
                metrics[key] = float(value)
            elif isinstance(value, str):
                value_clean = value.strip()
                if value_clean.lower() in ("nan", "infinity", "inf", "-infinity"):
                    continue
                metrics[key] = float(value_clean)
            elif isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (int, float)):
                if np.isfinite(value[0]):
                    metrics[key] = float(value[0])
        except (TypeError, ValueError):
            pass
    return metrics


@dataclass
class RankingRecord:
    model_path: Path
    meta_path: Path | None
    meta: dict[str, Any]
    created: str
    metrics: dict[str, float]
    features_n: int
    classes: list[str]
    top_feature: str
    file_size: int
    file_mtime_ns: int

    @property
    def ranking(self) -> dict[str, Any] | None:
        return model_eval_runtime.get_tab5_holdout_ranking(self.meta)


@dataclass
class StrictCandidate:
    record: RankingRecord
    source_rank_position: int
    dataset_signature: tuple[str, str, str, int]
    estimator_name: str
    criterion: str
    horizon: int
    tp_bps: float
    sl_bps: float
    tier: int
    optimized_profit: float
    base_profit: float
    sharpe: float
    trades: float
    created_ts: float

    @property
    def tier_label(self) -> str:
        return "A" if self.tier >= 2 else "B"

    @property
    def source_metrics(self) -> dict[str, Any]:
        return {
            "profit_h_opt": float(self.optimized_profit),
            "profit_h": float(self.base_profit),
            "sharpe_h": float(self.sharpe),
            "trades_h": float(self.trades),
            "tier": self.tier_label,
        }

    def sort_key(self) -> tuple[int, float, float, float, float, float]:
        return (
            int(self.tier),
            float(self.optimized_profit),
            float(self.base_profit),
            float(self.sharpe),
            float(self.trades),
            float(self.created_ts),
        )

    def dedupe_key(self) -> tuple[Any, ...]:
        return (
            str(self.estimator_name).strip().lower(),
            int(self.horizon),
            float(self.tp_bps),
            float(self.sl_bps),
            self.dataset_signature,
        )


@dataclass
class StrictRejection:
    record: RankingRecord
    reason: str


@dataclass(frozen=True)
class FeatureStabilityRow:
    feature_name: str
    mean: float | None
    std: float | None
    stability_score: float | None


@dataclass(frozen=True)
class FeatureStabilityDetail:
    rows: list[FeatureStabilityRow]
    average_score: float | None
    original_feature_count: int
    filtered_feature_count: int
    filter_threshold: float | None
    filter_applied: bool
    fallback_reason: str | None

    @property
    def has_data(self) -> bool:
        return bool(self.rows)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _finite_float_or_none(value: Any) -> float | None:
    parsed = model_eval_runtime.safe_float(value)
    if parsed is None or not np.isfinite(parsed):
        return None
    return float(parsed)


def _feature_stability_scores_from_meta(meta: dict[str, Any]) -> dict[str, float]:
    raw_scores = meta.get("feature_stability_score")
    if not isinstance(raw_scores, dict) or not raw_scores:
        raw_scores = compute_feature_stability_score(meta.get("feature_stability"))

    scores: dict[str, float] = {}
    if not isinstance(raw_scores, dict):
        return scores

    for feature_name, score_raw in raw_scores.items():
        score = _finite_float_or_none(score_raw)
        if score is None:
            continue
        scores[str(feature_name)] = float(np.clip(score, 0.0, 1.0))
    return scores


def _feature_stability_detail_from_meta(meta: dict[str, Any]) -> FeatureStabilityDetail:
    stability_stats = meta.get("feature_stability")
    if not isinstance(stability_stats, dict):
        stability_stats = {}

    score_map = _feature_stability_scores_from_meta(meta)
    feature_names: set[str] = set(score_map)
    feature_names.update(str(feature_name) for feature_name in stability_stats)
    if not feature_names:
        return FeatureStabilityDetail(
            rows=[],
            average_score=None,
            original_feature_count=0,
            filtered_feature_count=0,
            filter_threshold=_finite_float_or_none(meta.get("feature_stability_threshold")),
            filter_applied=bool(meta.get("feature_stability_filter_applied")),
            fallback_reason=(str(meta.get("feature_stability_filter_fallback_reason") or "").strip() or None),
        )

    rows: list[FeatureStabilityRow] = []
    for feature_name in feature_names:
        stats = stability_stats.get(feature_name)
        stats_map = stats if isinstance(stats, dict) else {}
        rows.append(
            FeatureStabilityRow(
                feature_name=str(feature_name),
                mean=_finite_float_or_none(stats_map.get("mean")),
                std=_finite_float_or_none(stats_map.get("std")),
                stability_score=score_map.get(str(feature_name)),
            )
        )

    rows.sort(
        key=lambda row: (
            row.stability_score is None,
            -(row.stability_score if row.stability_score is not None else 0.0),
            row.feature_name.lower(),
        )
    )

    scores = [score for score in score_map.values() if np.isfinite(score)]
    trained_features = _string_list(meta.get("trained_features") or meta.get("features"))
    kept_features = _string_list(meta.get("features_kept_by_stability"))
    removed_features = _string_list(meta.get("features_removed_by_stability"))

    if kept_features or removed_features:
        original_count = len(set(kept_features + removed_features))
        filtered_count = len(kept_features) if kept_features else len(trained_features)
    else:
        original_names = set(trained_features)
        original_names.update(feature_names)
        original_count = len(original_names)
        filtered_count = len(trained_features) if trained_features else len(original_names)

    return FeatureStabilityDetail(
        rows=rows,
        average_score=(float(np.mean(scores)) if scores else None),
        original_feature_count=int(original_count),
        filtered_feature_count=int(filtered_count),
        filter_threshold=_finite_float_or_none(meta.get("feature_stability_threshold")),
        filter_applied=bool(meta.get("feature_stability_filter_applied")),
        fallback_reason=(str(meta.get("feature_stability_filter_fallback_reason") or "").strip() or None),
    )


def _feature_stability_tooltip(detail: FeatureStabilityDetail) -> str:
    if not detail.has_data:
        return "Top 5 stable features: N/A"

    lines = ["Top 5 stable features:"]
    top_rows = [row for row in detail.rows if row.stability_score is not None][:5]
    if not top_rows:
        lines.append("- N/A")
    else:
        for row in top_rows:
            lines.append(f"- {row.feature_name}: {row.stability_score:.3f}")
    lines.append("")
    lines.append(f"Features: {detail.original_feature_count} -> {detail.filtered_feature_count}")
    return "\n".join(lines)


class FeatureStabilityDetailDialog(QDialog):
    def __init__(self, parent: QWidget | None, *, model_name: str, detail: FeatureStabilityDetail):
        super().__init__(parent)
        self.setWindowTitle(f"Feature Stability - {model_name}")
        self.setModal(True)
        self.resize(760, 420)

        summary_parts = [
            f"Puvodni featury: {detail.original_feature_count}",
            f"Po filtrovani: {detail.filtered_feature_count}",
        ]
        if detail.filter_threshold is not None:
            summary_parts.append(f"Threshold: {detail.filter_threshold:.3f}")
        if detail.filter_applied:
            summary_parts.append("Filtr aplikovan")
        elif detail.filter_threshold is not None:
            summary_parts.append("Filtr neaplikovan")
        if detail.fallback_reason:
            summary_parts.append(f"Fallback: {detail.fallback_reason}")

        summary = QLabel(" | ".join(summary_parts), self)
        summary.setWordWrap(True)

        table = QTableWidget(self)
        table.setColumnCount(4)
        table.setRowCount(len(detail.rows))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setHorizontalHeaderLabels(["Feature", "Mean", "Std", "Stability"])
        table.horizontalHeader().setStretchLastSection(True)
        for row_idx, row in enumerate(detail.rows):
            table.setItem(row_idx, 0, _table_item(row.feature_name))
            table.setItem(
                row_idx,
                1,
                _table_item(
                    "-" if row.mean is None else f"{row.mean:.3f}",
                    alignment=Qt.AlignRight | Qt.AlignVCenter,
                ),
            )
            table.setItem(
                row_idx,
                2,
                _table_item(
                    "-" if row.std is None else f"{row.std:.3f}",
                    alignment=Qt.AlignRight | Qt.AlignVCenter,
                ),
            )
            table.setItem(
                row_idx,
                3,
                _table_item(
                    "-" if row.stability_score is None else f"{row.stability_score:.3f}",
                    alignment=Qt.AlignCenter,
                ),
            )

        btn_close = QPushButton("Zavrit", self)
        btn_close.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)

        layout = QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(table)
        layout.addLayout(btn_row)


def discover_ranking_models(dir_path: Path) -> list[RankingRecord]:
    records: list[RankingRecord] = []
    if not dir_path.exists():
        return records
    for model_path in dir_path.glob("*.pkl"):
        try:
            st = model_path.stat()
        except OSError:
            continue
        meta_path = model_sidecar_meta_path(model_path)
        meta = read_sidecar_model_meta(model_path)
        classes = meta.get("model_classes")
        if not isinstance(classes, list) and isinstance(meta.get("classes"), list):
            classes = list(meta.get("classes"))
        features = meta.get("trained_features")
        if not isinstance(features, list) and isinstance(meta.get("features"), list):
            features = list(meta.get("features"))
        records.append(
            RankingRecord(
                model_path=model_path,
                meta_path=(meta_path if meta_path.exists() else None),
                meta=meta,
                created=str(meta.get("created_at_iso") or meta.get("created_at") or ""),
                metrics=_metrics_from_meta(meta),
                features_n=len(features or []),
                classes=[str(item) for item in list(classes or [])],
                top_feature=_top_feature_from_meta(meta),
                file_size=int(st.st_size),
                file_mtime_ns=int(st.st_mtime_ns),
            )
        )
    records.sort(key=ranking_record_sort_key, reverse=True)
    return records


def ranking_record_sort_key(record: RankingRecord) -> tuple[int, float, float, float]:
    ranking = record.ranking or {}
    status = str(ranking.get("status") or "").strip().lower()
    sharpe = _as_float(record.metrics.get("sharpe"), default=float("-inf"))
    ts = _as_timestamp(record.created, record.model_path)
    if status == "ok":
        optimized_profit = _as_float(ranking.get("profit_h"), default=float("-inf"))
        return (2, optimized_profit, sharpe, ts)
    if status in {"unsupported", "error"}:
        return (0, float("-inf"), sharpe, ts)
    base_profit = _as_float(
        record.metrics.get("profit_net", record.metrics.get("profit_gross")),
        default=float("-inf"),
    )
    return (1, base_profit, sharpe, ts)


class ModelRankingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tab_model_ranking")

        self.dir_edit = QLineEdit(self)
        self.dir_edit.setText(str(DEFAULT_MODEL_DIR))
        self.btn_browse = QPushButton("Zvolit slozku s modely...", self)
        self.btn_refresh = QPushButton("Refresh", self)
        self.btn_recompute_all = QPushButton("Prepocitat Profit(H) pro vsechny", self)
        self.btn_strict_top5 = QPushButton("Spustit strict Top 5", self)
        self.btn_delete = QPushButton("Smazat vybrane", self)
        self.btn_load = QPushButton("Nacist vybrany model", self)
        self.cmb_stability_filter = QComboBox(self)
        self.cmb_stability_filter.addItem(STABILITY_FILTER_LABELS[STABILITY_FILTER_ALL], STABILITY_FILTER_ALL)
        self.cmb_stability_filter.addItem(STABILITY_FILTER_LABELS[STABILITY_FILTER_GOOD], STABILITY_FILTER_GOOD)
        self.cmb_stability_filter.addItem(
            STABILITY_FILTER_LABELS[STABILITY_FILTER_EXCELLENT],
            STABILITY_FILTER_EXCELLENT,
        )
        self.btn_delete.hide()
        self.btn_load.hide()

        self.lbl_context = QLabel("CSV z Tab 4: - | fee/trade: - | Entry/Exit start: -/-", self)
        self.lbl_status = QLabel("Status: pripraveno", self)
        self.lbl_selected = QLabel("Vybrano: -", self)

        self.tbl = QTableWidget(self)
        self.tbl.setColumnCount(13)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.tbl.setHorizontalHeaderLabels(
            [
                "Model",
                "Rezim",
                "Status",
                "Profit(H opt)",
                "Profit(H)",
                "Entry(H)",
                "Exit(H)",
                "Trades(H)",
                "Sharpe(H)",
                "#Feats",
                "Stability",
                "Vytvoren",
                "Poznamka",
            ]
        )
        self.tbl.horizontalHeader().setStretchLastSection(True)

        top = QHBoxLayout()
        top.addWidget(QLabel("Slozka modelu:"))
        top.addWidget(self.dir_edit)
        top.addWidget(self.btn_browse)
        top.addWidget(self.btn_refresh)
        top.addWidget(QLabel("Stability:"))
        top.addWidget(self.cmb_stability_filter)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_recompute_all)
        actions.addWidget(self.btn_strict_top5)
        actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.lbl_context)
        layout.addWidget(self.tbl)
        layout.addWidget(self.lbl_selected)
        layout.addLayout(actions)
        layout.addWidget(self.lbl_status)

        self.records: list[RankingRecord] = []
        self._visible_records: list[RankingRecord] = []
        self._last_snapshot: tuple[tuple[str, int, int], ...] = ()
        self._last_context_fingerprint: tuple[Any, ...] | None = None
        self._ranking_worker: TaskWorker | None = None
        self._ranking_request_id = 0
        self._strict_worker: TaskWorker | None = None
        self._strict_request_id = 0

        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_recompute_all.clicked.connect(self._on_recompute_all_clicked)
        self.btn_strict_top5.clicked.connect(self._on_strict_top5_clicked)
        self.btn_delete.clicked.connect(self._on_delete_selected)
        self.btn_load.clicked.connect(self._on_load_selected)
        self.cmb_stability_filter.currentIndexChanged.connect(self._on_stability_filter_changed)
        self.tbl.itemSelectionChanged.connect(self._on_selection_changed)
        self.tbl.cellClicked.connect(self._on_cell_clicked)
        self.tbl.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.tbl.itemChanged.connect(self._on_table_item_changed)

        self._delete_shortcut = QShortcut(QKeySequence.Delete, self)
        self._delete_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(self._on_delete_selected)

        self.timer = QTimer(self)
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self._refresh_list(force=True)
        self._update_context_label()

    def _on_browse(self) -> None:
        start_dir = str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR.exists() else str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Zvol slozku s modely", start_dir)
        if not directory:
            return
        self.dir_edit.setText(directory)
        self._refresh_list(force=True)
        self._refresh_pending_hint()

    def _on_refresh_clicked(self) -> None:
        self._refresh_list(force=True)
        self._refresh_pending_hint()

    def _models_dir(self) -> Path | None:
        txt = self.dir_edit.text().strip()
        return Path(txt) if txt else None

    def _current_eval_context(self) -> dict[str, Any] | None:
        win = self.window()
        if win is None:
            return None
        try:
            ensure_tab_loaded = getattr(win, "_ensure_tab_loaded", None)
            if callable(ensure_tab_loaded):
                ensure_tab_loaded(3)
        except Exception:
            pass

        tab_eval = getattr(win, "tab_eval", None)
        if tab_eval is not None and hasattr(tab_eval, "current_ranking_context"):
            try:
                ctx = tab_eval.current_ranking_context()
                return ctx if isinstance(ctx, dict) else None
            except Exception:
                return None
        return None

    def _context_fingerprint(self, context: dict[str, Any] | None) -> tuple[Any, ...] | None:
        if not isinstance(context, dict):
            return None
        data_path = context.get("data_path")
        if not data_path:
            return None
        try:
            normalized_path = model_eval_runtime.normalize_path(data_path)
            st = Path(normalized_path).stat()
        except OSError:
            return None
        return (
            normalized_path,
            int(st.st_size),
            int(st.st_mtime_ns),
            float(context.get("fee_per_trade", 0.0) or 0.0),
            float(context.get("entry_threshold", 0.0) or 0.0),
            float(context.get("exit_threshold", 0.0) or 0.0),
        )

    def _update_context_label(self) -> None:
        context = self._current_eval_context()
        if not isinstance(context, dict):
            self.lbl_context.setText("CSV z Tab 4: - | fee/trade: - | Entry/Exit start: -/-")
            return
        data_path = str(context.get("data_path") or "-")
        fee = float(context.get("fee_per_trade", 0.0) or 0.0)
        entry = float(context.get("entry_threshold", 0.0) or 0.0)
        exit_thr = float(context.get("exit_threshold", 0.0) or 0.0)
        self.lbl_context.setText(
            f"CSV z Tab 4: {Path(data_path).name} | fee/trade: {fee:.3f} | Entry/Exit start: {entry:.2f}/{exit_thr:.2f}"
        )

    def _refresh_list(self, *, force: bool = False) -> bool:
        models_dir = self._models_dir()
        if not models_dir:
            return False
        snapshot = _directory_snapshot(models_dir)
        if not force and snapshot == self._last_snapshot:
            return False
        self._last_snapshot = snapshot
        self.records = discover_ranking_models(models_dir)
        self._render_table()
        self._update_context_label()
        return True

    def set_stability_filter(self, mode: str | None) -> None:
        normalized_mode = str(mode or STABILITY_FILTER_ALL).strip().lower()
        available_modes = {
            str(self.cmb_stability_filter.itemData(idx))
            for idx in range(self.cmb_stability_filter.count())
        }
        if normalized_mode not in available_modes:
            normalized_mode = STABILITY_FILTER_ALL
        idx = self.cmb_stability_filter.findData(normalized_mode)
        if idx >= 0 and idx != self.cmb_stability_filter.currentIndex():
            self.cmb_stability_filter.setCurrentIndex(idx)
        elif idx < 0 and self.cmb_stability_filter.currentIndex() != 0:
            self.cmb_stability_filter.setCurrentIndex(0)

    def _stability_filter_threshold(self) -> float | None:
        mode = str(self.cmb_stability_filter.currentData() or STABILITY_FILTER_ALL)
        return STABILITY_FILTER_THRESHOLDS.get(mode)

    def _filtered_records(self) -> list[RankingRecord]:
        threshold = self._stability_filter_threshold()
        if threshold is None:
            return list(self.records)
        return [
            record
            for record in self.records
            if (
                (detail := _feature_stability_detail_from_meta(record.meta)).average_score is not None
                and detail.average_score > threshold
            )
        ]

    def _on_stability_filter_changed(self) -> None:
        self._render_table()
        self._on_selection_changed()

    def _render_table(self) -> None:
        selected = self._selected_record()
        selected_path = None
        if selected is not None:
            try:
                selected_path = selected.model_path.resolve()
            except Exception:
                selected_path = selected.model_path

        visible_records = self._filtered_records()
        self._visible_records = list(visible_records)
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(visible_records))
        for row, record in enumerate(visible_records):
            ranking = record.ranking or {}
            stability_detail = _feature_stability_detail_from_meta(record.meta)
            training_mode = _training_mode_label(record.meta)
            note = _ranking_note_from_meta(record.meta)
            status = str(ranking.get("status") or ("meta" if record.metrics else "-")).strip().lower()
            optimized_profit = model_eval_runtime.safe_float(ranking.get("profit_h"))
            base_profit = model_eval_runtime.safe_float(
                model_eval_runtime.pick_metric(record.metrics, "profit_net", "profit_gross", "profit")
            )
            entry = model_eval_runtime.safe_float(ranking.get("entry_threshold"))
            exit_thr = model_eval_runtime.safe_float(ranking.get("exit_threshold"))
            trades = model_eval_runtime.safe_float(ranking.get("trades_h"))
            sharpe = model_eval_runtime.safe_float(record.metrics.get("sharpe"))

            self.tbl.setItem(row, COL_MODEL, _table_item(record.model_path.name))
            self.tbl.setItem(row, COL_MODE, _table_item(training_mode))
            self.tbl.setItem(row, COL_STATUS, _table_item(status or "-"))
            self.tbl.setItem(
                row,
                COL_PROFIT_OPT,
                _table_item("-" if optimized_profit is None else f"{optimized_profit:.2f}"),
            )
            self.tbl.setItem(
                row,
                COL_PROFIT,
                _table_item("-" if base_profit is None else f"{base_profit:.2f}"),
            )
            self.tbl.setItem(row, COL_ENTRY, _table_item("-" if entry is None else f"{entry:.2f}"))
            self.tbl.setItem(row, COL_EXIT, _table_item("-" if exit_thr is None else f"{exit_thr:.2f}"))
            self.tbl.setItem(row, COL_TRADES, _table_item("-" if trades is None else f"{trades:.0f}"))
            self.tbl.setItem(row, COL_SHARPE, _table_item("-" if sharpe is None else f"{sharpe:.3f}"))
            self.tbl.setItem(row, COL_FEATURES, _table_item(str(record.features_n)))
            self.tbl.setItem(
                row,
                COL_STABILITY,
                _table_item(
                    "-" if stability_detail.average_score is None else f"{stability_detail.average_score:.3f}",
                    alignment=Qt.AlignCenter,
                    tooltip=_feature_stability_tooltip(stability_detail),
                ),
            )
            self.tbl.setItem(row, COL_CREATED, _table_item(record.created or ""))
            self.tbl.setItem(row, COL_NOTE, _table_item(note, editable=True))
        self.tbl.blockSignals(False)

        if selected_path is not None:
            for row, record in enumerate(visible_records):
                try:
                    record_path = record.model_path.resolve()
                except Exception:
                    record_path = record.model_path
                if record_path == selected_path:
                    self.tbl.setCurrentCell(row, 0)
                    return

    def _record_for_row(self, row: int) -> RankingRecord | None:
        if row < 0 or row >= len(self._visible_records):
            return None
        return self._visible_records[row]

    def _build_feature_stability_detail(self, record: RankingRecord) -> FeatureStabilityDetail:
        return _feature_stability_detail_from_meta(record.meta)

    def _show_feature_stability_detail(self, record: RankingRecord) -> None:
        detail = self._build_feature_stability_detail(record)
        if not detail.has_data:
            QMessageBox.information(
                self,
                "Feature Stability",
                f"Model {record.model_path.name} nema dostupna stability metadata.",
            )
            return

        dialog = FeatureStabilityDetailDialog(
            self,
            model_name=record.model_path.name,
            detail=detail,
        )
        dialog.exec()

    def _on_cell_clicked(self, row: int, column: int) -> None:
        if column != COL_STABILITY:
            return
        record = self._record_for_row(row)
        if record is not None:
            self._show_feature_stability_detail(record)

    def _on_cell_double_clicked(self, row: int, column: int) -> None:
        if column == COL_NOTE:
            item = self.tbl.item(row, column)
            if item is not None:
                self.tbl.editItem(item)
            return
        if column == COL_STABILITY:
            return
        record = self._record_for_row(row)
        if record is not None:
            self._load_record_into_tabs(record, activate_eval_tab=True, auto_evaluate=True)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != COL_NOTE:
            return
        row = item.row()
        if row < 0 or row >= len(self._visible_records):
            return

        record = self._visible_records[row]
        new_note = str(item.text() or "").replace("\r", " ").replace("\n", " ").strip()
        old_note = _ranking_note_from_meta(record.meta)
        if new_note == old_note:
            if item.text() != new_note:
                self.tbl.blockSignals(True)
                item.setText(new_note)
                self.tbl.blockSignals(False)
            return

        meta = dict(record.meta or {})
        if new_note:
            meta[RANKING_NOTE_META_KEY] = new_note
        else:
            meta.pop(RANKING_NOTE_META_KEY, None)

        try:
            meta_path = write_sidecar_model_meta(record.model_path, meta)
        except Exception as exc:
            self.tbl.blockSignals(True)
            item.setText(old_note)
            self.tbl.blockSignals(False)
            self.lbl_status.setText(f"Status: nepodarilo se ulozit poznamku: {exc}")
            return

        record.meta = meta
        record.meta_path = meta_path

        if item.text() != new_note:
            self.tbl.blockSignals(True)
            item.setText(new_note)
            self.tbl.blockSignals(False)
        self.lbl_status.setText(f"Status: ulozena poznamka pro {record.model_path.name}")

    def _selected_record(self) -> RankingRecord | None:
        row = self.tbl.currentRow()
        if row < 0 or row >= len(self._visible_records):
            return None
        return self._visible_records[row]

    def _selected_rows(self) -> list[int]:
        rows: list[int] = []
        selection_model = self.tbl.selectionModel()
        if selection_model is not None:
            rows = sorted({index.row() for index in selection_model.selectedRows()})
        current_row = self.tbl.currentRow()
        if not rows and current_row >= 0:
            rows = [current_row]
        return [row for row in rows if 0 <= row < len(self._visible_records)]

    def _selected_records(self) -> list[RankingRecord]:
        return [self._visible_records[row] for row in self._selected_rows()]

    def _on_selection_changed(self) -> None:
        records = self._selected_records()
        if not records:
            self.lbl_selected.setText("Vybrano: -")
            return
        if len(records) > 1:
            self.lbl_selected.setText(f"Vybrano: {len(records)} modelu")
            return
        record = records[0]
        ranking = record.ranking or {}
        status = str(ranking.get("status") or "-")
        self.lbl_selected.setText(f"Vybrano: {record.model_path.name} | status={status}")

    def _is_busy(self) -> bool:
        return self._ranking_worker is not None or self._strict_worker is not None

    def _current_training_context(self) -> dict[str, Any] | None:
        win = self.window()
        if win is None:
            return None
        try:
            ensure_tab_loaded = getattr(win, "_ensure_tab_loaded", None)
            if callable(ensure_tab_loaded):
                ensure_tab_loaded(1)
        except Exception:
            pass

        tab_train = getattr(win, "tab_train", None)
        if tab_train is None:
            return None

        csv_path = getattr(tab_train, "csv_path", None)
        dataset = getattr(tab_train, "dataset", None)
        if not csv_path or dataset is None:
            return None

        try:
            normalized_path = model_eval_runtime.normalize_path(csv_path)
        except Exception:
            normalized_path = str(Path(csv_path).expanduser().resolve())

        try:
            n_total_bars = int(len(dataset))
        except Exception:
            return None

        snapshot_signature = dataset_snapshot_signature_from_csv(normalized_path, n_total_bars)
        if snapshot_signature is None:
            return None

        strict_profile = training_profile_for_mode("strict")
        candidate_top_n = int(strict_profile.get("candidate_top_n", 5) or 5)
        candidate_fresh_ratio = float(strict_profile.get("candidate_fresh_ratio", 0.30) or 0.30)

        try:
            if hasattr(tab_train, "_current_candidate_top_n"):
                candidate_top_n = int(max(1, tab_train._current_candidate_top_n()))
        except Exception:
            pass
        try:
            if hasattr(tab_train, "_current_candidate_fresh_ratio"):
                candidate_fresh_ratio = float(np.clip(tab_train._current_candidate_fresh_ratio(), 0.05, 0.80))
        except Exception:
            pass

        return {
            "csv_path": normalized_path,
            "snapshot_signature": snapshot_signature,
            "holdout_pct": float(getattr(tab_train, "holdout_pct_default", 0.10) or 0.10),
            "holdout_min_bars": int(getattr(tab_train, "holdout_min_bars_default", 1000) or 1000),
            "holdout_max_bars": int(getattr(tab_train, "holdout_max_bars_default", 6000) or 6000),
            "candidate_top_n": int(candidate_top_n),
            "candidate_fresh_ratio": float(candidate_fresh_ratio),
        }

    @staticmethod
    def _strict_tier(base_profit: float, sharpe: float) -> int:
        if base_profit > 0.0 and sharpe > 0.0:
            return 2
        if (base_profit > 0.0) != (sharpe > 0.0):
            return 1
        return 0

    @classmethod
    def _build_strict_shortlist(
        cls,
        *,
        records: list[RankingRecord],
        context: dict[str, Any],
        current_snapshot_signature: tuple[str, str, str, int],
        limit: int = 5,
    ) -> dict[str, Any]:
        selected_candidates: list[StrictCandidate] = []
        rejections: list[StrictRejection] = []
        data_path = str(context.get("data_path") or "")
        fee_per_trade = float(context.get("fee_per_trade", 0.0) or 0.0)

        for rank_position, record in enumerate(records, start=1):
            ranking = record.ranking or {}
            status = str(ranking.get("status") or "").strip().lower()
            if status != "ok":
                rejections.append(StrictRejection(record=record, reason="ranking nema status ok"))
                continue

            training_mode = str(record.meta.get("training_mode") or "").strip().lower()
            if training_mode == "strict":
                rejections.append(StrictRejection(record=record, reason="model uz je strict"))
                continue

            try:
                stale = model_eval_runtime.is_tab5_holdout_ranking_stale(
                    record.meta,
                    data_path=data_path,
                    fee_per_trade=fee_per_trade,
                    model_path=record.model_path,
                    meta_path=record.meta_path,
                )
            except OSError:
                stale = True
            if stale:
                rejections.append(StrictRejection(record=record, reason="ranking je stale pro aktualni CSV/fee"))
                continue

            dataset_signature = dataset_snapshot_signature_from_meta(record.meta)
            if dataset_signature != current_snapshot_signature:
                rejections.append(StrictRejection(record=record, reason="model nepatri do aktualniho dataset snapshotu"))
                continue

            optimized_profit = model_eval_runtime.safe_float(ranking.get("profit_h"))
            if optimized_profit is None or optimized_profit <= 0.0:
                rejections.append(StrictRejection(record=record, reason="Profit(H opt) musi byt kladny"))
                continue

            base_profit = model_eval_runtime.safe_float(
                model_eval_runtime.pick_metric(record.metrics, "profit_net", "profit_gross", "profit")
            )
            sharpe = model_eval_runtime.safe_float(record.metrics.get("sharpe"))
            if base_profit is None:
                base_profit = float("-inf")
            if sharpe is None:
                sharpe = float("-inf")
            if base_profit < 0.0 and sharpe <= 0.0:
                rejections.append(StrictRejection(record=record, reason="Profit(H) < 0 a Sharpe(H) <= 0"))
                continue

            tier = cls._strict_tier(float(base_profit), float(sharpe))
            if tier <= 0:
                rejections.append(StrictRejection(record=record, reason="model neprosel tier A/B filtrem"))
                continue

            estimator_name = str(record.meta.get("estimator_name") or "").strip().lower()
            try:
                horizon = int(record.meta.get("label_horizon_bars"))
                tp_bps = float(record.meta.get("label_take_profit_bps"))
                sl_bps = float(record.meta.get("label_stop_loss_bps"))
            except Exception:
                horizon = 0
                tp_bps = 0.0
                sl_bps = 0.0
            if not estimator_name or horizon <= 0 or tp_bps <= 0.0 or sl_bps <= 0.0:
                rejections.append(StrictRejection(record=record, reason="chybi estimator nebo label metadata"))
                continue

            trades = model_eval_runtime.safe_float(ranking.get("trades_h"))
            created_ts = _as_timestamp(record.created, record.model_path)
            selected_candidates.append(
                StrictCandidate(
                    record=record,
                    source_rank_position=rank_position,
                    dataset_signature=current_snapshot_signature,
                    estimator_name=estimator_name,
                    criterion=candidate_selection_criterion_from_meta(record.meta, default="balanced"),
                    horizon=int(horizon),
                    tp_bps=float(tp_bps),
                    sl_bps=float(sl_bps),
                    tier=int(tier),
                    optimized_profit=float(optimized_profit),
                    base_profit=float(base_profit),
                    sharpe=float(sharpe),
                    trades=float(trades if trades is not None else 0.0),
                    created_ts=float(created_ts),
                )
            )

        selected_candidates.sort(key=lambda candidate: candidate.sort_key(), reverse=True)

        deduped: list[StrictCandidate] = []
        seen_keys: dict[tuple[Any, ...], StrictCandidate] = {}
        for candidate in selected_candidates:
            dedupe_key = candidate.dedupe_key()
            existing = seen_keys.get(dedupe_key)
            if existing is not None:
                rejections.append(
                    StrictRejection(
                        record=candidate.record,
                        reason=f"duplicitni konfigurace, ponechan {existing.record.model_path.name}",
                    )
                )
                continue
            seen_keys[dedupe_key] = candidate
            deduped.append(candidate)

        shortlist = deduped[: max(1, int(limit))]
        for candidate in deduped[len(shortlist) :]:
            rejections.append(StrictRejection(record=candidate.record, reason="mimo Top 5 po serazeni"))

        return {
            "selected": shortlist,
            "rejected": rejections,
        }

    @staticmethod
    def _strict_preview_text(selected: list[StrictCandidate], rejected: list[StrictRejection]) -> str:
        lines = ["Vybrane strict kandidaty:"]
        if not selected:
            lines.append("- zadny kandidat")
        else:
            for idx, candidate in enumerate(selected, start=1):
                lines.append(
                    f"{idx}. {candidate.record.model_path.name} | tier={candidate.tier_label} "
                    f"| Profit(H opt)={candidate.optimized_profit:.2f} | Profit(H)={candidate.base_profit:.2f} "
                    f"| Sharpe(H)={candidate.sharpe:.3f} | Trades(H)={candidate.trades:.0f} "
                    f"| {candidate.estimator_name} h={candidate.horizon} tp={candidate.tp_bps:.0f} sl={candidate.sl_bps:.0f}"
                )

        lines.append("")
        lines.append("Vyrazene modely:")
        if not rejected:
            lines.append("- zadny")
        else:
            for rejection in rejected[:15]:
                lines.append(f"- {rejection.record.model_path.name}: {rejection.reason}")
            more = len(rejected) - 15
            if more > 0:
                lines.append(f"- ... a dalsich {more}")

        lines.append("")
        lines.append(f"Spustit strict batch pro {len(selected)} modelu?")
        return "\n".join(lines)

    def _records_requiring_ranking(self, context: dict[str, Any]) -> list[RankingRecord]:
        stale: list[RankingRecord] = []
        data_path = str(context.get("data_path") or "")
        fee_per_trade = float(context.get("fee_per_trade", 0.0) or 0.0)
        if not data_path:
            return stale
        for record in self.records:
            try:
                is_stale = model_eval_runtime.is_tab5_holdout_ranking_stale(
                    record.meta,
                    data_path=data_path,
                    fee_per_trade=fee_per_trade,
                    model_path=record.model_path,
                    meta_path=record.meta_path,
                )
            except OSError:
                is_stale = True
            if is_stale:
                stale.append(record)
        return stale

    def _refresh_pending_hint(self) -> None:
        if self._is_busy():
            return
        context = self._current_eval_context()
        if not isinstance(context, dict):
            return
        pending = self._records_requiring_ranking(context)
        if pending:
            self.lbl_status.setText(
                f"Status: aktualni CSV kontext zmeni ranking u {len(pending)} modelu | klikni 'Prepocitat Profit(H)'"
            )
        else:
            self.lbl_status.setText("Status: ranking je aktualni pro aktualni CSV kontext")

    def _start_incremental_if_needed(self) -> None:
        context = self._current_eval_context()
        if not isinstance(context, dict):
            return
        pending = self._records_requiring_ranking(context)
        if pending:
            self._start_batch_worker(records=pending, context=context, full_recompute=False)

    def _on_recompute_all_clicked(self) -> None:
        if self._is_busy():
            return
        context = self._current_eval_context()
        if not isinstance(context, dict):
            QMessageBox.warning(self, "Model Ranking", "V Tab 4 musi byt nactene CSV s historickymi daty.")
            return
        if not self.records:
            self.lbl_status.setText("Status: zadne modely k prepocitu")
            return
        pending = self._records_requiring_ranking(context)
        if not pending:
            self.lbl_status.setText("Status: vsechny modely uz maji aktualni Profit(H)")
            return
        self._start_batch_worker(records=pending, context=context, full_recompute=False)

    def _tick(self) -> None:
        changed = self._refresh_list()
        context = self._current_eval_context()
        context_fp = self._context_fingerprint(context)
        context_changed = context_fp != self._last_context_fingerprint
        if context_changed:
            self._last_context_fingerprint = context_fp
            self._update_context_label()

        if self._is_busy():
            return
        if changed:
            self._start_incremental_if_needed()
        elif context_changed:
            self._refresh_pending_hint()

    def _start_batch_worker(
        self,
        *,
        records: list[RankingRecord],
        context: dict[str, Any],
        full_recompute: bool,
    ) -> None:
        if self._is_busy():
            return
        if not records:
            self.lbl_status.setText("Status: zadne modely k prepocitu")
            return
        data_path = str(context.get("data_path") or "")
        if not data_path:
            QMessageBox.warning(self, "Model Ranking", "V Tab 4 chybi CSV dataset.")
            return

        self._ranking_request_id += 1
        req_id = self._ranking_request_id
        self.lbl_status.setText(
            f"Status: {'plny' if full_recompute else 'inkrementalni'} prepocet {len(records)} modelu..."
        )
        self.btn_recompute_all.setEnabled(False)
        self.btn_strict_top5.setEnabled(False)

        worker = TaskWorker(
            self._task_compute_rankings,
            model_paths=[str(record.model_path) for record in records],
            data_path=data_path,
            fee_per_trade=float(context.get("fee_per_trade", 0.0) or 0.0),
            current_entry=float(context.get("entry_threshold", 0.0) or 0.0),
            current_exit=float(context.get("exit_threshold", 0.0) or 0.0),
            full_recompute=bool(full_recompute),
        )
        self._ranking_worker = worker
        worker.progress_text.connect(lambda text: self.lbl_status.setText(f"Status: {text}"))
        worker.result.connect(lambda result, rid=req_id: self._on_batch_result(rid, result))
        worker.error.connect(lambda msg, rid=req_id: self._on_batch_error(rid, msg))
        worker.finished.connect(lambda rid=req_id: self._on_batch_finished(rid))
        worker.start()

    @staticmethod
    def _task_compute_rankings(
        *,
        model_paths: list[str],
        data_path: str,
        fee_per_trade: float,
        current_entry: float,
        current_exit: float,
        full_recompute: bool,
        progress_cb=None,
        should_run=None,
    ) -> dict[str, Any]:
        prepared = model_eval_runtime.load_prepared_evaluation_data(data_path, progress_cb=progress_cb)
        updated = 0
        failures = 0
        total = len(model_paths)

        for idx, model_path in enumerate(model_paths, start=1):
            if callable(should_run) and not should_run():
                break
            if callable(progress_cb):
                progress_cb(f"Ranking {idx}/{total}: {Path(model_path).name}")

            meta = read_sidecar_model_meta(model_path)
            try:
                loaded = model_eval_runtime.load_predictor_with_merged_meta(model_path)
                evaluation = model_eval_runtime.run_model_evaluation(
                    model=loaded.predictor,
                    metadata=loaded.metadata,
                    prepared_data=prepared,
                    scope_mode="holdout",
                    fee_per_trade=float(fee_per_trade),
                    entry_threshold=float(current_entry),
                    exit_threshold=float(current_exit),
                )
                search = model_eval_runtime.run_auto_threshold_search_from_context(
                    y_pred_raw=evaluation.y_pred_raw,
                    confidence_arr=evaluation.confidence_arr,
                    y_true_current=evaluation.y_true_current,
                    df_current=evaluation.df_current,
                    fee_per_trade=float(fee_per_trade),
                    current_entry=float(current_entry),
                    current_exit=float(current_exit),
                    should_run=should_run,
                )
                _, optimized_metrics = model_eval_runtime.recalculate_metrics_from_predictions(
                    y_pred_raw=evaluation.y_pred_raw,
                    confidence_arr=evaluation.confidence_arr,
                    y_true_current=evaluation.y_true_current,
                    df_current=evaluation.df_current,
                    fee_per_trade=float(fee_per_trade),
                    entry_threshold=float(search.best_entry),
                    exit_threshold=float(search.best_exit),
                )
                meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = model_eval_runtime.build_tab5_holdout_ranking_payload(
                    data_path=data_path,
                    fee_per_trade=float(fee_per_trade),
                    entry_threshold=float(search.best_entry),
                    exit_threshold=float(search.best_exit),
                    metrics=optimized_metrics,
                    status="ok",
                )
                write_sidecar_model_meta(model_path, meta)
                updated += 1
            except Exception as exc:
                failures += 1
                status = model_eval_runtime.ranking_status_from_error_message(str(exc))
                meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = model_eval_runtime.build_tab5_holdout_ranking_payload(
                    data_path=data_path,
                    fee_per_trade=float(fee_per_trade),
                    entry_threshold=None,
                    exit_threshold=None,
                    metrics=None,
                    status=status,
                    error=str(exc),
                )
                write_sidecar_model_meta(model_path, meta)

        return {
            "updated": int(updated),
            "failures": int(failures),
            "requested": int(total),
            "full_recompute": bool(full_recompute),
        }

    def _on_batch_result(self, req_id: int, result: dict[str, Any]) -> None:
        if req_id != self._ranking_request_id:
            return
        self._refresh_list(force=True)
        updated = int(result.get("updated", 0) or 0)
        failures = int(result.get("failures", 0) or 0)
        mode = "Plny" if bool(result.get("full_recompute")) else "Inkrementalni"
        self.lbl_status.setText(f"Status: {mode.lower()} prepocet hotov | updated={updated}, failures={failures}")

    def _on_batch_error(self, req_id: int, msg: str) -> None:
        if req_id != self._ranking_request_id:
            return
        self.lbl_status.setText(f"Status: chyba batch rankingu: {msg}")

    def _on_batch_finished(self, req_id: int) -> None:
        if req_id != self._ranking_request_id:
            return
        self._ranking_worker = None
        self.btn_recompute_all.setEnabled(True)
        self.btn_strict_top5.setEnabled(True)

    def _on_strict_top5_clicked(self) -> None:
        if self._is_busy():
            return
        context = self._current_eval_context()
        if not isinstance(context, dict):
            QMessageBox.warning(self, "Model Ranking", "V Tab 4 musi byt nactene CSV s historickymi daty.")
            return

        training_context = self._current_training_context()
        if not isinstance(training_context, dict):
            QMessageBox.warning(
                self,
                "Model Ranking",
                "Pro strict Top 5 musi byt v Tab 2 nactene odpovidajici treninkove CSV.",
            )
            return

        eval_csv = str(context.get("data_path") or "")
        train_csv = str(training_context.get("csv_path") or "")
        if not eval_csv or not train_csv:
            QMessageBox.warning(self, "Model Ranking", "Chybi CSV kontext pro ranking nebo trenink.")
            return

        try:
            eval_csv_norm = model_eval_runtime.normalize_path(eval_csv)
            train_csv_norm = model_eval_runtime.normalize_path(train_csv)
        except Exception:
            eval_csv_norm = eval_csv
            train_csv_norm = train_csv
        if eval_csv_norm != train_csv_norm:
            QMessageBox.warning(
                self,
                "Model Ranking",
                "Strict Top 5 lze spustit jen kdyz Tab 2 a Tab 4 pouzivaji stejny CSV snapshot.",
            )
            return

        shortlist = self._build_strict_shortlist(
            records=list(self.records),
            context=context,
            current_snapshot_signature=training_context["snapshot_signature"],
            limit=5,
        )
        selected = list(shortlist.get("selected") or [])
        rejected = list(shortlist.get("rejected") or [])
        if not selected:
            QMessageBox.warning(
                self,
                "Model Ranking",
                "Pro strict Top 5 se nenasel zadny vhodny kandidat v aktualnim ranking kontextu.",
            )
            return

        preview = self._strict_preview_text(selected, rejected)
        reply = QMessageBox.question(
            self,
            "Strict Top 5",
            preview,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._start_strict_worker(selected=selected, training_context=training_context)

    def _start_strict_worker(self, *, selected: list[StrictCandidate], training_context: dict[str, Any]) -> None:
        if self._is_busy():
            return
        if not selected:
            self.lbl_status.setText("Status: strict shortlist je prazdny")
            return

        self._strict_request_id += 1
        req_id = self._strict_request_id
        batch_id = f"strict_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}"
        self.lbl_status.setText(f"Status: strict batch start pro {len(selected)} modelu...")
        self.btn_recompute_all.setEnabled(False)
        self.btn_strict_top5.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.btn_load.setEnabled(False)

        jobs = []
        for candidate in selected:
            jobs.append(
                {
                    "source_model_path": str(candidate.record.model_path),
                    "source_rank_position": int(candidate.source_rank_position),
                    "estimator_name": str(candidate.estimator_name),
                    "criterion": str(candidate.criterion),
                    "horizon": int(candidate.horizon),
                    "tp_bps": float(candidate.tp_bps),
                    "sl_bps": float(candidate.sl_bps),
                    "strict_source_metrics": dict(candidate.source_metrics),
                }
            )

        worker = TaskWorker(
            self._task_run_strict_batch,
            jobs=jobs,
            training_csv_path=str(training_context.get("csv_path") or ""),
            holdout_pct=float(training_context.get("holdout_pct", 0.10) or 0.10),
            holdout_min_bars=int(training_context.get("holdout_min_bars", 1000) or 1000),
            holdout_max_bars=int(training_context.get("holdout_max_bars", 6000) or 6000),
            candidate_top_n=int(training_context.get("candidate_top_n", 5) or 5),
            candidate_fresh_ratio=float(training_context.get("candidate_fresh_ratio", 0.30) or 0.30),
            batch_id=batch_id,
        )
        self._strict_worker = worker
        worker.progress_text.connect(lambda text: self.lbl_status.setText(f"Status: {text}"))
        worker.result.connect(lambda result, rid=req_id: self._on_strict_result(rid, result))
        worker.error.connect(lambda msg, rid=req_id: self._on_strict_error(rid, msg))
        worker.finished.connect(lambda rid=req_id: self._on_strict_finished(rid))
        worker.start()

    @staticmethod
    def _task_run_strict_batch(
        *,
        jobs: list[dict[str, Any]],
        training_csv_path: str,
        holdout_pct: float,
        holdout_min_bars: int,
        holdout_max_bars: int,
        candidate_top_n: int,
        candidate_fresh_ratio: float,
        batch_id: str,
        progress_cb=None,
        should_run=None,
    ) -> dict[str, Any]:
        created = 0
        rejected = 0
        failures = 0
        results: list[dict[str, Any]] = []
        total = len(jobs)

        for idx, job in enumerate(jobs, start=1):
            if callable(should_run) and not should_run():
                break
            if callable(progress_cb):
                progress_cb(f"Strict {idx}/{total}: {Path(str(job.get('source_model_path') or '')).name}")

            provenance = {
                "training_mode": "strict",
                "strict_source_model_path": str(job.get("source_model_path") or ""),
                "strict_source_rank_position": int(job.get("source_rank_position", idx) or idx),
                "strict_batch_id": str(batch_id),
                "strict_trigger": "ranking_top5",
                "strict_source_metrics": dict(job.get("strict_source_metrics") or {}),
            }

            strict_profile = training_profile_for_mode("strict")
            result = run_training_job(
                csv_path=str(training_csv_path),
                holdout_pct=float(holdout_pct),
                holdout_min_bars=int(holdout_min_bars),
                holdout_max_bars=int(holdout_max_bars),
                phase="strict",
                estimator_name=str(job.get("estimator_name") or ""),
                criterion=str(job.get("criterion") or "balanced"),
                horizon=int(job.get("horizon", 12) or 12),
                tp_bps=float(job.get("tp_bps", 50.0) or 50.0),
                sl_bps=float(job.get("sl_bps", 50.0) or 50.0),
                candidate_top_n=int(max(1, candidate_top_n)),
                candidate_fresh_ratio=float(np.clip(candidate_fresh_ratio, 0.05, 0.80)),
                training_profile=dict(strict_profile),
                extra_meta=provenance,
            )

            model_path = str(result.get("model_path") or "")
            if model_path:
                meta = read_sidecar_model_meta(model_path)
                if isinstance(meta, dict):
                    meta.update(provenance)
                    write_sidecar_model_meta(model_path, meta)

            status = str(result.get("status") or "").strip().lower()
            if status == "ok" and model_path:
                created += 1
            elif status == "rejected":
                rejected += 1
            else:
                failures += 1

            row = dict(result)
            row.update(provenance)
            results.append(row)

        return {
            "requested": int(total),
            "created": int(created),
            "rejected": int(rejected),
            "failures": int(failures),
            "results": results,
            "batch_id": str(batch_id),
        }

    def _on_strict_result(self, req_id: int, result: dict[str, Any]) -> None:
        if req_id != self._strict_request_id:
            return
        self._refresh_list(force=True)
        created = int(result.get("created", 0) or 0)
        rejected = int(result.get("rejected", 0) or 0)
        failures = int(result.get("failures", 0) or 0)
        self.lbl_status.setText(
            f"Status: strict batch hotov | created={created}, rejected={rejected}, failures={failures}"
        )
        if created > 0:
            self._start_incremental_if_needed()

    def _on_strict_error(self, req_id: int, msg: str) -> None:
        if req_id != self._strict_request_id:
            return
        self.lbl_status.setText(f"Status: chyba strict batch: {msg}")

    def _on_strict_finished(self, req_id: int) -> None:
        if req_id != self._strict_request_id:
            return
        self._strict_worker = None
        self.btn_recompute_all.setEnabled(True)
        self.btn_strict_top5.setEnabled(True)
        self.btn_delete.setEnabled(True)
        self.btn_load.setEnabled(True)
        self._refresh_list(force=True)
        self._start_incremental_if_needed()

    def _propagate_model_path(
        self,
        record: RankingRecord,
        *,
        activate_eval_tab: bool = False,
        auto_evaluate: bool = False,
    ) -> None:
        win = self.window()
        if win is None:
            return
        model_path = record.model_path

        try:
            ensure_tab_loaded = getattr(win, "_ensure_tab_loaded", None)
            if callable(ensure_tab_loaded):
                ensure_tab_loaded(3)
                ensure_tab_loaded(4)
        except Exception:
            pass

        if activate_eval_tab:
            try:
                tabs = getattr(win, "tabs", None)
                if tabs is not None:
                    tabs.setCurrentIndex(3)
            except Exception:
                pass

        tab_eval = getattr(win, "tab_eval", None)
        if tab_eval is not None and hasattr(tab_eval, "set_model_path"):
            try:
                tab_eval.set_model_path(str(model_path))
            except Exception:
                pass
            else:
                ranking = record.ranking or {}
                entry_threshold = model_eval_runtime.safe_float(ranking.get("entry_threshold"))
                exit_threshold = model_eval_runtime.safe_float(ranking.get("exit_threshold"))
                try:
                    if entry_threshold is not None and hasattr(tab_eval, "et_spin"):
                        tab_eval.et_spin.setValue(float(entry_threshold))
                    if exit_threshold is not None and hasattr(tab_eval, "ext_spin"):
                        tab_eval.ext_spin.setValue(float(exit_threshold))
                except Exception:
                    pass
                if auto_evaluate and hasattr(tab_eval, "on_evaluate_clicked"):
                    try:
                        tab_eval.on_evaluate_clicked()
                    except Exception:
                        pass

        tab_live = getattr(win, "tab_live", None)
        if tab_live is not None:
            try:
                if hasattr(tab_live, "set_model_paths"):
                    tab_live.set_model_paths([str(model_path)])
                elif hasattr(tab_live, "le_model_path"):
                    tab_live.le_model_path.setText(str(model_path))
            except Exception:
                pass

    def _load_record_into_tabs(
        self,
        record: RankingRecord | None,
        *,
        activate_eval_tab: bool = False,
        auto_evaluate: bool = False,
    ) -> None:
        if record is None:
            QMessageBox.warning(self, "Model Ranking", "Nejprve vyber model v tabulce.")
            return
        self._propagate_model_path(
            record,
            activate_eval_tab=bool(activate_eval_tab),
            auto_evaluate=bool(auto_evaluate),
        )
        if auto_evaluate:
            self.lbl_status.setText(f"Status: model aplikovan do Tab 4/5 a spusteno vyhodnoceni -> {record.model_path.name}")
        else:
            self.lbl_status.setText(f"Status: model aplikovan do Tab 4/5 -> {record.model_path.name}")

    def _load_selected_into_tabs(self) -> None:
        self._load_record_into_tabs(self._selected_record())

    def _on_load_selected(self) -> None:
        self._load_selected_into_tabs()

    def _on_delete_selected(self) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        records = [self.records[row] for row in rows]
        if len(records) == 1:
            prompt = f"Opravdu smazat model?\n\n{records[0].model_path.name}"
        else:
            preview = [record.model_path.name for record in records[:5]]
            more_count = len(records) - len(preview)
            prompt = "Opravdu smazat vybrane modely?\n\n" + "\n".join(preview)
            if more_count > 0:
                prompt += f"\n... a dalsich {more_count}"

        reply = QMessageBox.question(
            self,
            "Smazat modely",
            prompt,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        deleted = 0
        failures: list[str] = []
        for record in records:
            sidecar_path = model_sidecar_meta_path(record.model_path)
            try:
                if sidecar_path.exists():
                    sidecar_path.unlink()
                record.model_path.unlink()
                deleted += 1
            except Exception as exc:
                failures.append(f"{record.model_path.name}: {exc}")

        self._refresh_list(force=True)
        if self.records:
            next_row = min(rows[0], len(self.records) - 1)
            self.tbl.setCurrentCell(next_row, 0)

        self.lbl_status.setText(
            f"Status: smazano {deleted} modelu, chyby {len(failures)}"
        )
        if failures:
            QMessageBox.warning(
                self,
                "Mazani modelu",
                "Nektere modely se nepodarilo smazat:\n\n" + "\n".join(failures[:10]),
            )
