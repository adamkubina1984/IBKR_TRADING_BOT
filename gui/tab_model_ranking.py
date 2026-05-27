from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import traceback
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ibkr_trading_bot.core.services import model_eval_service as model_eval_runtime
from ibkr_trading_bot.core.services.model_service import model_sidecar_meta_path, read_sidecar_model_meta, write_sidecar_model_meta
from ibkr_trading_bot.core.services.model_training_service import candidate_selection_criterion_from_meta, normalize_candidate_criterion
from ibkr_trading_bot.gui.components.workers import TaskWorker
from ibkr_trading_bot.model.feature_stability import compute_feature_stability_score

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "model_outputs"
LOG_DIR = Path(__file__).parent.parent / "logs"
RANKING_NOTE_META_KEY = "model_ranking_note"
_BATCH_PROGRESS_RE = re.compile(r"^Ranking\s+(\d+)/(\d+):\s+(.+)$")
COL_MODEL = 0
COL_MODE = 1
COL_STATUS = 2
COL_PROFIT_OPT = 3
COL_PROFIT = 4
COL_ENTRY = 5
COL_EXIT = 6
COL_TRADES = 7
COL_SHARPE = 8
COL_BIAS = 9
COL_FEATURES = 10
COL_STABILITY = 11
COL_CREATED = 12
COL_CHECK = 13
COL_NOTE = 14
MAX_COMPARISON_MODELS = 6

COMPARISON_METRIC_SPECS: tuple[tuple[str, str], ...] = (
    ("mode", "Rezim"),
    ("status", "Status"),
    ("freshness", "Ranking v CSV"),
    ("optimized_profit", "Profit(H opt)"),
    ("base_profit", "Profit(H)"),
    ("entry", "Entry(H)"),
    ("exit", "Exit(H)"),
    ("trades", "Trades(H)"),
    ("sharpe", "Sharpe(H)"),
    ("bias", "Bias S-L"),
    ("stability", "Stability"),
    ("criterion", "Criterion"),
    ("horizon", "Horizon"),
    ("tp_bps", "TP bps"),
    ("sl_bps", "SL bps"),
    ("created", "Vytvoren"),
    ("note", "Poznamka"),
)

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

BIAS_FILTER_ALL = "all"
BIAS_FILTER_TIGHT = "tight"
BIAS_FILTER_MODERATE = "moderate"
BIAS_FILTER_LABELS = {
    BIAS_FILTER_ALL: "All",
    BIAS_FILTER_TIGHT: "Abs bias <= 0.10",
    BIAS_FILTER_MODERATE: "Abs bias <= 0.25",
}
BIAS_FILTER_THRESHOLDS = {
    BIAS_FILTER_TIGHT: 0.10,
    BIAS_FILTER_MODERATE: 0.25,
}

WORKFLOW_MODE_EXPLORE = "explore"
WORKFLOW_MODE_REFINE = "refine"
WORKFLOW_MODE_REFRESH = "refresh"
WORKFLOW_MODE_LABELS = {
    WORKFLOW_MODE_EXPLORE: "Explore",
    WORKFLOW_MODE_REFINE: "Refine",
    WORKFLOW_MODE_REFRESH: "Refresh",
}
WORKFLOW_MODE_ORDER = (
    WORKFLOW_MODE_EXPLORE,
    WORKFLOW_MODE_REFINE,
    WORKFLOW_MODE_REFRESH,
)
LEGACY_TRAINING_MODE_TO_WORKFLOW_MODE = {
    "quick": WORKFLOW_MODE_EXPLORE,
    "standard": WORKFLOW_MODE_REFINE,
    "strict": WORKFLOW_MODE_REFRESH,
}


class _SortAwareTableItem(QTableWidgetItem):
    def __init__(self, text: str, *, sort_value: Any = None):
        super().__init__(text)
        self._sort_value = sort_value

    def __lt__(self, other: object) -> bool:
        if isinstance(other, _SortAwareTableItem):
            left = self._sort_value
            right = other._sort_value
            if left is not None and right is not None:
                try:
                    return left < right
                except Exception:
                    pass
            return self.text().casefold() < other.text().casefold()
        return self.text().casefold() < QTableWidgetItem.text(other).casefold() if isinstance(other, QTableWidgetItem) else False


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
    mode_key = _training_mode_key(meta)
    if mode_key is not None:
        return WORKFLOW_MODE_LABELS[mode_key]
    value = str(meta.get("workflow_mode") or meta.get("training_mode") or "").strip()
    return value if value else "-"


def _normalize_workflow_mode(value: Any) -> str | None:
    txt = str(value or "").strip().lower()
    if not txt:
        return None
    if txt in WORKFLOW_MODE_LABELS:
        return txt
    return LEGACY_TRAINING_MODE_TO_WORKFLOW_MODE.get(txt)


def _training_mode_key(meta: dict[str, Any]) -> str | None:
    candidates = [
        meta.get("workflow_mode"),
        meta.get("training_mode"),
        meta.get("training_mode_requested"),
        meta.get("training_mode_compatibility"),
    ]
    for value in candidates:
        normalized = _normalize_workflow_mode(value)
        if normalized is not None:
            return normalized
    return None


def _ranking_note_from_meta(meta: dict[str, Any]) -> str:
    raw = str(meta.get(RANKING_NOTE_META_KEY) or "")
    return raw.replace("\r", " ").replace("\n", " ").strip()


def _table_item(
    text: str,
    *,
    editable: bool = False,
    alignment: int | None = None,
    tooltip: str | None = None,
    sort_value: Any = None,
) -> QTableWidgetItem:
    item = _SortAwareTableItem(text, sort_value=sort_value)
    flags = item.flags()
    if not editable:
        flags &= ~Qt.ItemIsEditable
    item.setFlags(flags)
    if alignment is not None:
        item.setTextAlignment(int(alignment))
    if tooltip:
        item.setToolTip(tooltip)
    return item


def _write_ranking_crash_log(context: str, exc: BaseException) -> Path | None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = LOG_DIR / f"ranking_crash_{context}_{stamp}.log"
        path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        return path
    except Exception:
        return None


def _check_item(checked: bool) -> QTableWidgetItem:
    item = QTableWidgetItem("")
    item.setFlags((item.flags() & ~Qt.ItemIsEditable) | Qt.ItemIsUserCheckable)
    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
    item.setTextAlignment(int(Qt.AlignCenter))
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


def _shortlist_directory_paths(models_dir: Path) -> list[Path]:
    candidates = [models_dir / "auto_search", models_dir.parent / "auto_search"]
    seen: set[str] = set()
    out: list[Path] = []
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_dir():
            out.append(candidate)
    return out


def _shortlist_snapshot(models_dir: Path | None) -> tuple[tuple[str, int, int], ...]:
    if models_dir is None:
        return ()
    snapshot: list[tuple[str, int, int]] = []
    for dir_path in _shortlist_directory_paths(models_dir):
        for path in dir_path.glob("*_shortlist.json"):
            try:
                st = path.stat()
            except OSError:
                continue
            snapshot.append((str(path), int(st.st_size), int(st.st_mtime_ns)))
    snapshot.sort()
    return tuple(snapshot)


def _dataset_signature_tuple(
    instrument: Any,
    exchange: Any,
    timeframe: Any,
    n_total_bars: Any,
) -> tuple[str, str, str, int] | None:
    instrument_txt = str(instrument or "").strip().lower()
    exchange_txt = str(exchange or "").strip().lower()
    timeframe_txt = str(timeframe or "").strip().lower()
    try:
        bars = int(n_total_bars or 0)
    except Exception:
        bars = 0
    if not instrument_txt or not exchange_txt or not timeframe_txt or bars <= 0:
        return None
    return (instrument_txt, exchange_txt, timeframe_txt, bars)


def _candidate_identity_key(
    *,
    estimator_name: Any,
    criterion: Any,
    horizon: Any,
    tp_bps: Any,
    sl_bps: Any,
) -> tuple[str, str, int, float, float]:
    return (
        str(estimator_name or "").strip().lower(),
        normalize_candidate_criterion(str(criterion or "balanced"), default="balanced"),
        int(horizon or 0),
        float(tp_bps or 0.0),
        float(sl_bps or 0.0),
    )


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


def _per_class_recall_from_meta(meta: dict[str, Any], *label_keys: str) -> float | None:
    metrics_raw = meta.get("metrics_holdout") or meta.get("metrics") or {}
    if not isinstance(metrics_raw, dict):
        return None
    per_class = metrics_raw.get("per_class_3")
    if not isinstance(per_class, dict):
        return None
    for label_key in label_keys:
        try:
            recall = float(((per_class.get(label_key) or {}).get("recall")))
        except Exception:
            continue
        if np.isfinite(recall):
            return float(recall)
    return None


def _bias_score_from_meta(meta: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    rec_short = _per_class_recall_from_meta(meta, "-1", "0")
    rec_long = _per_class_recall_from_meta(meta, "1", "2")
    if rec_short is None or rec_long is None:
        return None, rec_short, rec_long
    return float(rec_short - rec_long), rec_short, rec_long


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


@dataclass(frozen=True)
class ShortlistArtifact:
    path: Path
    created: str
    dataset_signature: tuple[str, str, str, int] | None
    candidate_keys: frozenset[tuple[str, str, int, float, float]]

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_keys)


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


def _load_shortlist_artifact(path: Path) -> ShortlistArtifact | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    candidates_raw = payload.get("candidates")
    if not isinstance(candidates_raw, list):
        return None

    candidate_keys: set[tuple[str, str, int, float, float]] = set()
    for candidate in candidates_raw:
        if not isinstance(candidate, dict):
            continue
        candidate_keys.add(
            _candidate_identity_key(
                estimator_name=candidate.get("model"),
                criterion=candidate.get("criterion"),
                horizon=candidate.get("horizon"),
                tp_bps=candidate.get("tp_bps"),
                sl_bps=candidate.get("sl_bps"),
            )
        )

    dataset_signature_raw = payload.get("dataset_signature")
    dataset_signature = None
    if isinstance(dataset_signature_raw, dict):
        dataset_signature = _dataset_signature_tuple(
            dataset_signature_raw.get("instrument"),
            dataset_signature_raw.get("exchange"),
            dataset_signature_raw.get("timeframe"),
            dataset_signature_raw.get("n_total_bars"),
        )

    return ShortlistArtifact(
        path=path,
        created=str(payload.get("created_at") or ""),
        dataset_signature=dataset_signature,
        candidate_keys=frozenset(candidate_keys),
    )


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


def _format_optional_float(value: Any, *, digits: int = 2, placeholder: str = "-") -> str:
    parsed = model_eval_runtime.safe_float(value)
    if parsed is None or not np.isfinite(parsed):
        return placeholder
    return f"{float(parsed):.{digits}f}"


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


@dataclass(frozen=True)
class ModelComparisonSnapshot:
    record_key: str
    model_name: str
    values: dict[str, str]


class ModelComparisonDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        *,
        snapshots: list[ModelComparisonSnapshot],
        summary_text: str,
        on_load_record_key=None,
        on_export=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Porovnani modelu ({len(snapshots)})")
        self.setModal(True)
        self.resize(1080, 520)

        self._snapshots = list(snapshots)
        self._on_load_record_key = on_load_record_key
        self._on_export = on_export

        summary = QLabel(summary_text, self)
        summary.setWordWrap(True)

        self.table = QTableWidget(self)
        self.table.setColumnCount(len(self._snapshots) + 1)
        self.table.setRowCount(len(COMPARISON_METRIC_SPECS))
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setHorizontalHeaderLabels(["Metrika", *[snapshot.model_name for snapshot in self._snapshots]])
        self.table.horizontalHeader().setStretchLastSection(True)

        right_aligned_keys = {
            "optimized_profit",
            "base_profit",
            "entry",
            "exit",
            "trades",
            "sharpe",
            "bias",
            "stability",
            "criterion",
            "horizon",
            "tp_bps",
            "sl_bps",
        }
        for row_idx, (value_key, label) in enumerate(COMPARISON_METRIC_SPECS):
            self.table.setItem(row_idx, 0, _table_item(label))
            for col_idx, snapshot in enumerate(self._snapshots, start=1):
                alignment = Qt.AlignRight | Qt.AlignVCenter if value_key in right_aligned_keys else Qt.AlignLeft | Qt.AlignVCenter
                self.table.setItem(
                    row_idx,
                    col_idx,
                    _table_item(snapshot.values.get(value_key, "-"), alignment=alignment),
                )

        self.table.cellClicked.connect(self._on_cell_clicked)

        self.cmb_load = QComboBox(self)
        for snapshot in self._snapshots:
            self.cmb_load.addItem(snapshot.model_name, snapshot.record_key)

        self.btn_load = QPushButton("Nacist do Tab 4/5", self)
        self.btn_load.setEnabled(callable(self._on_load_record_key) and bool(self._snapshots))
        self.btn_load.clicked.connect(self._on_load_clicked)

        self.btn_export_csv = QPushButton("Export CSV", self)
        self.btn_export_csv.setEnabled(callable(self._on_export) and bool(self._snapshots))
        self.btn_export_csv.clicked.connect(lambda: self._on_export_clicked("csv"))

        self.btn_export_json = QPushButton("Export JSON", self)
        self.btn_export_json.setEnabled(callable(self._on_export) and bool(self._snapshots))
        self.btn_export_json.clicked.connect(lambda: self._on_export_clicked("json"))

        btn_close = QPushButton("Zavrit", self)
        btn_close.clicked.connect(self.accept)

        actions = QHBoxLayout()
        actions.addWidget(QLabel("Detail modelu:", self))
        actions.addWidget(self.cmb_load)
        actions.addWidget(self.btn_load)
        actions.addWidget(self.btn_export_csv)
        actions.addWidget(self.btn_export_json)
        actions.addStretch(1)
        actions.addWidget(btn_close)

        layout = QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(self.table)
        layout.addLayout(actions)

    def _on_cell_clicked(self, _row: int, column: int) -> None:
        if column <= 0:
            return
        combo_index = column - 1
        if 0 <= combo_index < self.cmb_load.count():
            self.cmb_load.setCurrentIndex(combo_index)

    def _on_load_clicked(self) -> None:
        if not callable(self._on_load_record_key):
            return
        record_key = str(self.cmb_load.currentData() or "")
        if not record_key:
            return
        self._on_load_record_key(record_key)
        self.accept()

    def _on_export_clicked(self, file_format: str) -> None:
        if not callable(self._on_export):
            return
        self._on_export(str(file_format or "").strip().lower())


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
        optimized_payload = model_eval_runtime.get_tab5_holdout_optimized_payload(ranking)
        optimized_profit = _as_float(optimized_payload.get("profit_h"), default=float("-inf"))
        return (2, optimized_profit, sharpe, ts)
    if status in {"unsupported", "error"}:
        return (0, float("-inf"), sharpe, ts)
    base_payload = model_eval_runtime.get_tab5_holdout_base_payload(ranking, fallback_metrics=record.metrics)
    base_profit = _as_float(base_payload.get("profit_h"), default=float("-inf"))
    return (1, base_profit, sharpe, ts)


def _payload_value_with_flat_fallback(payload: dict[str, Any], ranking: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is not None:
        return value
    return ranking.get(key)


class ModelRankingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tab_model_ranking")

        self.dir_edit = QLineEdit(self)
        self.dir_edit.setText(str(DEFAULT_MODEL_DIR))
        self.btn_browse = QPushButton("Zvolit slozku s modely...", self)
        self.btn_refresh = QPushButton("Refresh", self)
        self.btn_recompute_profit_opt = QPushButton("Prepocitat neaktualni (H opt)", self)
        self.btn_stop_recompute = QPushButton("Zastavit prepocet", self)
        self.btn_stop_recompute.setEnabled(False)
        self.btn_check_filtered = QPushButton("Zatrhnout filtrovane", self)
        self.btn_clear_checked = QPushButton("Vymazat zatrzeni", self)
        self.btn_compare_filtered = QPushButton("Porovnat filtrovane", self)
        self.btn_compare = QPushButton("Porovnat vybrane", self)
        self.btn_delete = QPushButton("Smazat vybrane", self)
        self.btn_load = QPushButton("Nacist vybrany model", self)
        self.cmb_shortlist_filter = QComboBox(self)
        self.cmb_stability_filter = QComboBox(self)
        self.cmb_stability_filter.addItem(STABILITY_FILTER_LABELS[STABILITY_FILTER_ALL], STABILITY_FILTER_ALL)
        self.cmb_stability_filter.addItem(STABILITY_FILTER_LABELS[STABILITY_FILTER_GOOD], STABILITY_FILTER_GOOD)
        self.cmb_stability_filter.addItem(
            STABILITY_FILTER_LABELS[STABILITY_FILTER_EXCELLENT],
            STABILITY_FILTER_EXCELLENT,
        )
        self.cmb_bias_filter = QComboBox(self)
        self.cmb_bias_filter.addItem(BIAS_FILTER_LABELS[BIAS_FILTER_ALL], BIAS_FILTER_ALL)
        self.cmb_bias_filter.addItem(BIAS_FILTER_LABELS[BIAS_FILTER_TIGHT], BIAS_FILTER_TIGHT)
        self.cmb_bias_filter.addItem(BIAS_FILTER_LABELS[BIAS_FILTER_MODERATE], BIAS_FILTER_MODERATE)
        self.mode_filter_button = QToolButton(self)
        self.mode_filter_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.mode_filter_menu = QMenu(self.mode_filter_button)
        self.mode_filter_menu.aboutToShow.connect(self._rebuild_mode_filter_menu)
        self.mode_filter_button.setMenu(self.mode_filter_menu)
        self.btn_delete.hide()
        self.btn_load.hide()

        self.lbl_context = QLabel("CSV z Tab 4: - | fee/trade: - | Entry/Exit start: -/-", self)
        self.lbl_shortlist = QLabel("Shortlist: bez filtru", self)
        self.lbl_status = QLabel("Status: pripraveno", self)
        self.lbl_selected = QLabel("Vybrano: -", self)
        self.lbl_batch_progress = QLabel("Prubeh batch: -", self)

        self.tbl = QTableWidget(self)
        self.tbl.setColumnCount(15)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.tbl.setSortingEnabled(True)
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
                "Bias S-L",
                "#Feats",
                "Stability",
                "Vytvoren",
                "Vyber",
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
        top.addWidget(QLabel("Bias:"))
        top.addWidget(self.cmb_bias_filter)
        top.addWidget(QLabel("Rezimy:"))
        top.addWidget(self.mode_filter_button)
        top.addWidget(QLabel("Shortlist:"))
        top.addWidget(self.cmb_shortlist_filter)

        actions = QHBoxLayout()
        actions.addWidget(self.btn_recompute_profit_opt)
        actions.addWidget(self.btn_stop_recompute)
        actions.addWidget(self.btn_check_filtered)
        actions.addWidget(self.btn_clear_checked)
        actions.addWidget(self.btn_compare_filtered)
        actions.addWidget(self.btn_compare)
        actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.lbl_context)
        layout.addWidget(self.lbl_shortlist)
        layout.addWidget(self.tbl)
        layout.addWidget(self.lbl_selected)
        layout.addWidget(self.lbl_batch_progress)
        layout.addLayout(actions)
        layout.addWidget(self.lbl_status)

        self.records: list[RankingRecord] = []
        self._visible_records: list[RankingRecord] = []
        self._last_snapshot: tuple[tuple[str, int, int], ...] = ()
        self._last_shortlist_snapshot: tuple[tuple[str, int, int], ...] = ()
        self._last_context_fingerprint: tuple[Any, ...] | None = None
        self._ranking_worker: TaskWorker | None = None
        self._ranking_request_id = 0
        self._batch_total = 0
        self._batch_completed = 0
        self._batch_current_index = 0
        self._batch_current_model = ""
        self._batch_cancel_requested = False
        self._batch_result_received = False
        self._checked_model_paths: set[str] = set()
        self._mode_filter_keys: set[str] = set()
        self._records_by_key: dict[str, RankingRecord] = {}
        self._shortlist_artifacts_by_path: dict[str, ShortlistArtifact] = {}
        self._active_shortlist_path = ""
        self._latest_shortlist_path = ""
        self._delete_in_progress = False
        self._tab_active = False

        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_recompute_profit_opt.clicked.connect(self._on_recompute_profit_opt_clicked)
        self.btn_stop_recompute.clicked.connect(self._on_stop_recompute_clicked)
        self.btn_check_filtered.clicked.connect(self._on_check_filtered_clicked)
        self.btn_clear_checked.clicked.connect(self._on_clear_checked_clicked)
        self.btn_compare_filtered.clicked.connect(self._on_compare_filtered_clicked)
        self.btn_compare.clicked.connect(self._on_compare_selected_clicked)
        self.btn_delete.clicked.connect(self._on_delete_selected)
        self.btn_load.clicked.connect(self._on_load_selected)
        self.cmb_shortlist_filter.currentIndexChanged.connect(self._on_shortlist_filter_changed)
        self.cmb_stability_filter.currentIndexChanged.connect(self._on_stability_filter_changed)
        self.cmb_bias_filter.currentIndexChanged.connect(self._on_bias_filter_changed)
        self.tbl.itemSelectionChanged.connect(self._on_selection_changed)
        self.tbl.cellClicked.connect(self._on_cell_clicked)
        self.tbl.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.tbl.itemChanged.connect(self._on_table_item_changed)
        self.tbl.installEventFilter(self)
        self.tbl.viewport().installEventFilter(self)

        self._delete_shortcut = QShortcut(QKeySequence.Delete, self)
        self._delete_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(self._delete_selected_via_shortcut)

        self.timer = QTimer(self)
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self._tick)

        self._update_mode_filter_button()
        self._refresh_list(force=True)

    def on_tab_activated(self) -> None:
        self._tab_active = True
        if not self.timer.isActive():
            self.timer.start()
        self._update_context_label()

    def on_tab_deactivated(self) -> None:
        self._tab_active = False
        if self.timer.isActive():
            self.timer.stop()

    def eventFilter(self, watched: object, event: object) -> bool:
        if watched in {self.tbl, self.tbl.viewport()} and isinstance(event, QEvent):
            if event.type() == QEvent.KeyPress and getattr(event, "key", lambda: None)() == Qt.Key_Delete:
                self._delete_selected_via_shortcut()
                return True
        return super().eventFilter(watched, event)

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
        shortlist_snapshot = _shortlist_snapshot(models_dir)
        if not force and snapshot == self._last_snapshot and shortlist_snapshot == self._last_shortlist_snapshot:
            return False
        self._last_snapshot = snapshot
        self._last_shortlist_snapshot = shortlist_snapshot
        self.records = discover_ranking_models(models_dir)
        self._records_by_key = {self._record_key(record): record for record in self.records}
        self._mode_filter_keys = {
            mode_key for mode_key in self._mode_filter_keys if mode_key in WORKFLOW_MODE_LABELS
        }
        self._refresh_shortlist_options(models_dir)
        self._update_mode_filter_button()
        self._render_table()
        self._update_context_label()
        return True

    def set_shortlist_filter(self, path: str | None) -> None:
        normalized_path = str(path or "").strip()
        idx = self.cmb_shortlist_filter.findData(normalized_path)
        if idx < 0:
            idx = 0
        if idx != self.cmb_shortlist_filter.currentIndex():
            self.cmb_shortlist_filter.setCurrentIndex(idx)
            return
        self._active_shortlist_path = str(self.cmb_shortlist_filter.currentData() or "")
        self._render_table()
        self._on_selection_changed()

    def _active_shortlist_artifact(self) -> ShortlistArtifact | None:
        if not self._active_shortlist_path:
            return None
        return self._shortlist_artifacts_by_path.get(self._active_shortlist_path)

    def _latest_shortlist_artifact(self) -> ShortlistArtifact | None:
        if not self._latest_shortlist_path:
            return None
        return self._shortlist_artifacts_by_path.get(self._latest_shortlist_path)

    def _refresh_shortlist_options(self, models_dir: Path) -> None:
        artifacts: list[ShortlistArtifact] = []
        for dir_path in _shortlist_directory_paths(models_dir):
            for path in dir_path.glob("*_shortlist.json"):
                info = _load_shortlist_artifact(path)
                if info is not None:
                    artifacts.append(info)

        artifacts.sort(
            key=lambda info: (_as_timestamp(info.created, info.path), str(info.path).casefold()),
            reverse=True,
        )
        self._shortlist_artifacts_by_path = {str(info.path): info for info in artifacts}
        self._latest_shortlist_path = str(artifacts[0].path) if artifacts else ""

        current_path = self._active_shortlist_path
        self.cmb_shortlist_filter.blockSignals(True)
        self.cmb_shortlist_filter.clear()
        self.cmb_shortlist_filter.addItem("Bez shortlistu", "")
        for info in artifacts:
            label = info.path.name if info.candidate_count <= 0 else f"{info.path.name} ({info.candidate_count})"
            self.cmb_shortlist_filter.addItem(label, str(info.path))
        idx = self.cmb_shortlist_filter.findData(current_path)
        if idx < 0:
            idx = 0
        self.cmb_shortlist_filter.setCurrentIndex(idx)
        self.cmb_shortlist_filter.setEnabled(self.cmb_shortlist_filter.count() > 1)
        self.cmb_shortlist_filter.blockSignals(False)

        self._active_shortlist_path = str(self.cmb_shortlist_filter.currentData() or "")
        self._update_shortlist_label()

    @staticmethod
    def _record_dataset_signature(record: RankingRecord) -> tuple[str, str, str, int] | None:
        return _dataset_signature_tuple(
            record.meta.get("instrument"),
            record.meta.get("exchange"),
            record.meta.get("timeframe"),
            record.meta.get("n_total_bars"),
        )

    @staticmethod
    def _record_candidate_identity(record: RankingRecord) -> tuple[str, str, int, float, float]:
        return _candidate_identity_key(
            estimator_name=record.meta.get("estimator_name"),
            criterion=candidate_selection_criterion_from_meta(record.meta),
            horizon=record.meta.get("label_horizon_bars") or record.meta.get("label_lookahead_bars"),
            tp_bps=record.meta.get("label_take_profit_bps"),
            sl_bps=record.meta.get("label_stop_loss_bps"),
        )

    def _record_matches_shortlist(self, record: RankingRecord, shortlist: ShortlistArtifact) -> bool:
        if self._record_candidate_identity(record) not in shortlist.candidate_keys:
            return False
        record_signature = self._record_dataset_signature(record)
        if shortlist.dataset_signature is not None and record_signature is not None:
            return record_signature == shortlist.dataset_signature
        return True

    def _update_shortlist_label(self) -> None:
        shortlist = self._active_shortlist_artifact()
        if shortlist is None:
            latest_shortlist = self._latest_shortlist_artifact()
            if self.cmb_shortlist_filter.count() <= 1 or latest_shortlist is None:
                self.lbl_shortlist.setText("Shortlist: nenalezen")
            else:
                matched_rows = sum(1 for record in self.records if self._record_matches_shortlist(record, latest_shortlist))
                self.lbl_shortlist.setText(
                    f"Shortlist: bez filtru | pripraven posledni: {latest_shortlist.path.name} | kandidati {latest_shortlist.candidate_count} | modelove radky {matched_rows}"
                )
            return

        matched_rows = sum(1 for record in self.records if self._record_matches_shortlist(record, shortlist))
        self.lbl_shortlist.setText(
            f"Shortlist: {shortlist.path.name} | kandidati {shortlist.candidate_count} | modelove radky {matched_rows} | zobrazeno {len(self._visible_records)}"
        )

    def _on_shortlist_filter_changed(self) -> None:
        self._active_shortlist_path = str(self.cmb_shortlist_filter.currentData() or "")
        self._render_table()
        self._on_selection_changed()

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

    def set_bias_filter(self, mode: str | None) -> None:
        normalized_mode = str(mode or BIAS_FILTER_ALL).strip().lower()
        available_modes = {
            str(self.cmb_bias_filter.itemData(idx))
            for idx in range(self.cmb_bias_filter.count())
        }
        if normalized_mode not in available_modes:
            normalized_mode = BIAS_FILTER_ALL
        idx = self.cmb_bias_filter.findData(normalized_mode)
        if idx >= 0 and idx != self.cmb_bias_filter.currentIndex():
            self.cmb_bias_filter.setCurrentIndex(idx)
        elif idx < 0 and self.cmb_bias_filter.currentIndex() != 0:
            self.cmb_bias_filter.setCurrentIndex(0)

    def _bias_filter_threshold(self) -> float | None:
        mode = str(self.cmb_bias_filter.currentData() or BIAS_FILTER_ALL)
        return BIAS_FILTER_THRESHOLDS.get(mode)

    def set_mode_filter(self, modes: list[str] | tuple[str, ...] | set[str] | None) -> None:
        normalized_modes = {
            normalized
            for mode in (modes or [])
            if (normalized := _normalize_workflow_mode(mode)) is not None
        }
        self._apply_mode_filter_modes(normalized_modes)

    def _apply_mode_filter_modes(self, mode_keys: set[str]) -> None:
        normalized_modes = {mode_key for mode_key in mode_keys if mode_key in WORKFLOW_MODE_LABELS}
        if normalized_modes == set(WORKFLOW_MODE_ORDER):
            normalized_modes = set()
        if normalized_modes == self._mode_filter_keys:
            return
        self._mode_filter_keys = normalized_modes
        self._update_mode_filter_button()
        self._render_table()
        self._on_selection_changed()

    def _clear_mode_filter(self) -> None:
        self._apply_mode_filter_modes(set())

    def _rebuild_mode_filter_menu(self) -> None:
        self.mode_filter_menu.clear()

        clear_action = self.mode_filter_menu.addAction("Vsechny rezimy")
        clear_action.setEnabled(bool(self._mode_filter_keys))
        clear_action.triggered.connect(self._clear_mode_filter)

        available_modes = {
            mode_key
            for record in self.records
            if (mode_key := _training_mode_key(record.meta)) is not None
        }

        if not self.records:
            empty_action = self.mode_filter_menu.addAction("Zadne modely")
            empty_action.setEnabled(False)
            return

        self.mode_filter_menu.addSeparator()
        for mode_key in WORKFLOW_MODE_ORDER:
            action = self.mode_filter_menu.addAction(WORKFLOW_MODE_LABELS[mode_key])
            action.setCheckable(True)
            action.setChecked(mode_key in self._mode_filter_keys)
            action.setEnabled(mode_key in available_modes)
            action.toggled.connect(
                lambda checked, key=mode_key: self._apply_mode_filter_modes(
                    (self._mode_filter_keys | {key}) if checked else (self._mode_filter_keys - {key})
                )
            )

    def _update_mode_filter_button(self) -> None:
        total = len(self.records)
        active_modes = [mode_key for mode_key in WORKFLOW_MODE_ORDER if mode_key in self._mode_filter_keys]
        if total <= 0:
            self.mode_filter_button.setText("Zadne")
            self.mode_filter_button.setToolTip("Ve slozce nejsou zadne modely.")
            self.mode_filter_button.setEnabled(False)
            return

        self.mode_filter_button.setEnabled(True)
        if not active_modes:
            self.mode_filter_button.setText("Vsechny")
            self.mode_filter_button.setToolTip("Bez filtru rezimu.")
            return

        selected_labels = [WORKFLOW_MODE_LABELS[mode_key] for mode_key in active_modes]
        if len(selected_labels) <= 2:
            self.mode_filter_button.setText(" / ".join(selected_labels))
        else:
            self.mode_filter_button.setText(f"{len(selected_labels)}/{len(WORKFLOW_MODE_ORDER)}")
        tooltip = "Aktivni filtr rezimu:\n" + "\n".join(selected_labels)
        self.mode_filter_button.setToolTip(tooltip)

    def _filtered_records(self) -> list[RankingRecord]:
        filtered = list(self.records)

        shortlist = self._active_shortlist_artifact()
        if shortlist is not None:
            filtered = [record for record in filtered if self._record_matches_shortlist(record, shortlist)]

        if self._mode_filter_keys:
            filtered = [
                record
                for record in filtered
                if _training_mode_key(record.meta) in self._mode_filter_keys
            ]

        stability_threshold = self._stability_filter_threshold()
        if stability_threshold is not None:
            filtered = [
                record
                for record in filtered
                if (
                    (detail := _feature_stability_detail_from_meta(record.meta)).average_score is not None
                    and detail.average_score > stability_threshold
                )
            ]

        bias_threshold = self._bias_filter_threshold()
        if bias_threshold is not None:
            filtered = [
                record
                for record in filtered
                if (
                    (bias_score := _bias_score_from_meta(record.meta)[0]) is not None
                    and abs(float(bias_score)) <= float(bias_threshold)
                )
            ]

        return filtered

    def _on_stability_filter_changed(self) -> None:
        self._render_table()
        self._on_selection_changed()

    def _on_bias_filter_changed(self) -> None:
        self._render_table()
        self._on_selection_changed()

    @staticmethod
    def _record_key(record: RankingRecord) -> str:
        try:
            return str(record.model_path.resolve())
        except Exception:
            return str(record.model_path)

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
        sorting_enabled = self.tbl.isSortingEnabled()
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(len(visible_records))
        for row, record in enumerate(visible_records):
            ranking = record.ranking or {}
            base_payload = model_eval_runtime.get_tab5_holdout_base_payload(ranking, fallback_metrics=record.metrics)
            optimized_payload = model_eval_runtime.get_tab5_holdout_optimized_payload(ranking)
            stability_detail = _feature_stability_detail_from_meta(record.meta)
            training_mode = _training_mode_label(record.meta)
            note = _ranking_note_from_meta(record.meta)
            bias_score, rec_short, rec_long = _bias_score_from_meta(record.meta)
            status = str(ranking.get("status") or ("meta" if record.metrics else "-")).strip().lower()
            optimized_profit = model_eval_runtime.safe_float(optimized_payload.get("profit_h"))
            base_profit = model_eval_runtime.safe_float(base_payload.get("profit_h"))
            entry = model_eval_runtime.safe_float(_payload_value_with_flat_fallback(optimized_payload, ranking, "entry_threshold"))
            exit_thr = model_eval_runtime.safe_float(_payload_value_with_flat_fallback(optimized_payload, ranking, "exit_threshold"))
            trades = model_eval_runtime.safe_float(_payload_value_with_flat_fallback(optimized_payload, ranking, "trades_h"))
            sharpe = model_eval_runtime.safe_float(record.metrics.get("sharpe"))

            model_item = _table_item(record.model_path.name)
            model_item.setData(Qt.UserRole, self._record_key(record))
            self.tbl.setItem(row, COL_MODEL, model_item)
            self.tbl.setItem(row, COL_MODE, _table_item(training_mode))
            self.tbl.setItem(row, COL_STATUS, _table_item(status or "-"))
            self.tbl.setItem(
                row,
                COL_PROFIT_OPT,
                _table_item(
                    "-" if optimized_profit is None else f"{optimized_profit:.2f}",
                    sort_value=(float("-inf") if optimized_profit is None else float(optimized_profit)),
                ),
            )
            self.tbl.setItem(
                row,
                COL_PROFIT,
                _table_item(
                    "-" if base_profit is None else f"{base_profit:.2f}",
                    sort_value=(float("-inf") if base_profit is None else float(base_profit)),
                ),
            )
            self.tbl.setItem(
                row,
                COL_ENTRY,
                _table_item("-" if entry is None else f"{entry:.2f}", sort_value=entry),
            )
            self.tbl.setItem(
                row,
                COL_EXIT,
                _table_item("-" if exit_thr is None else f"{exit_thr:.2f}", sort_value=exit_thr),
            )
            self.tbl.setItem(
                row,
                COL_TRADES,
                _table_item("-" if trades is None else f"{trades:.0f}", sort_value=trades),
            )
            self.tbl.setItem(
                row,
                COL_SHARPE,
                _table_item("-" if sharpe is None else f"{sharpe:.3f}", sort_value=sharpe),
            )
            self.tbl.setItem(
                row,
                COL_BIAS,
                _table_item(
                    "-" if bias_score is None else f"{bias_score:+.3f}",
                    alignment=Qt.AlignCenter,
                    tooltip=(
                        None
                        if rec_short is None or rec_long is None
                        else f"SHORT recall: {rec_short:.3f}\nLONG recall: {rec_long:.3f}"
                    ),
                    sort_value=(float("inf") if bias_score is None else abs(float(bias_score))),
                ),
            )
            self.tbl.setItem(row, COL_FEATURES, _table_item(str(record.features_n)))
            self.tbl.setItem(
                row,
                COL_STABILITY,
                _table_item(
                    "-" if stability_detail.average_score is None else f"{stability_detail.average_score:.3f}",
                    alignment=Qt.AlignCenter,
                    tooltip=_feature_stability_tooltip(stability_detail),
                    sort_value=stability_detail.average_score,
                ),
            )
            self.tbl.setItem(row, COL_CREATED, _table_item(record.created or ""))
            self.tbl.setItem(
                row,
                COL_CHECK,
                _check_item(self._record_key(record) in self._checked_model_paths),
            )
            self.tbl.setItem(row, COL_NOTE, _table_item(note, editable=True))
        self.tbl.setSortingEnabled(sorting_enabled)
        self.tbl.blockSignals(False)
        self._update_shortlist_label()

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
        if row < 0 or row >= self.tbl.rowCount():
            return None
        item = self.tbl.item(row, COL_MODEL)
        if item is None:
            return None
        record_key = str(item.data(Qt.UserRole) or "").strip()
        if not record_key:
            return None
        return self._records_by_key.get(record_key)

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
        if column == COL_CHECK:
            return
        if column == COL_STABILITY:
            return
        record = self._record_for_row(row)
        if record is not None:
            self._load_record_into_tabs(record, activate_eval_tab=True, auto_evaluate=True)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == COL_CHECK:
            record = self._record_for_row(item.row())
            if record is None:
                return
            record_key = self._record_key(record)
            if item.checkState() == Qt.Checked:
                self._checked_model_paths.add(record_key)
            else:
                self._checked_model_paths.discard(record_key)
            self._on_selection_changed()
            return

        if item.column() != COL_NOTE:
            return
        record = self._record_for_row(item.row())
        if record is None:
            return
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
        return self._record_for_row(row)

    def _selected_rows(self) -> list[int]:
        rows: list[int] = []
        selection_model = self.tbl.selectionModel()
        if selection_model is not None:
            rows = sorted({index.row() for index in selection_model.selectedRows()})
        current_row = self.tbl.currentRow()
        if not rows and current_row >= 0:
            rows = [current_row]
        return [row for row in rows if 0 <= row < self.tbl.rowCount()]

    def _selected_records(self) -> list[RankingRecord]:
        records: list[RankingRecord] = []
        seen_keys: set[str] = set()
        for row in self._selected_rows():
            record = self._record_for_row(row)
            if record is None:
                continue
            record_key = self._record_key(record)
            if record_key in seen_keys:
                continue
            seen_keys.add(record_key)
            records.append(record)
        return records

    def _checked_records(self) -> list[RankingRecord]:
        return [record for record in self.records if self._record_key(record) in self._checked_model_paths]

    def _set_checked_scope(self, records: list[RankingRecord]) -> None:
        self._checked_model_paths = {self._record_key(record) for record in records}
        self._render_table()
        self._on_selection_changed()

    def _on_check_filtered_clicked(self) -> None:
        if self._is_busy():
            self.lbl_status.setText("Status: zatrzeni nelze menit behem batch prepoctu")
            return
        if not self._has_active_filters():
            self.lbl_status.setText("Status: nejprve zapni shortlist nebo jiny filtr a pak zatrhni scope")
            return

        filtered_records = self._filtered_records()
        if not filtered_records:
            self.lbl_status.setText("Status: aktualni filtr nevratil zadne modely k zatrzeni")
            return

        self._set_checked_scope(filtered_records)
        self.lbl_status.setText(f"Status: zatrzeno {len(filtered_records)} filtrovanych modelu")

    def _on_clear_checked_clicked(self) -> None:
        if self._is_busy():
            self.lbl_status.setText("Status: zatrzeni nelze menit behem batch prepoctu")
            return
        if not self._checked_model_paths:
            self.lbl_status.setText("Status: zadne zatrzene modely k vymazani")
            return
        self._set_checked_scope([])
        self.lbl_status.setText("Status: zatrzeni vymazano")

    def _comparison_target_records(self) -> tuple[list[RankingRecord], str]:
        checked_records = self._checked_records()
        if checked_records:
            return checked_records, "checked"
        selected_records = self._selected_records()
        if selected_records:
            return selected_records, "selected"
        return [], "none"

    def _comparison_freshness_label(self, record: RankingRecord, context: dict[str, Any] | None) -> str:
        if not isinstance(context, dict) or not str(context.get("data_path") or "").strip():
            return "bez CSV"
        if record.ranking is None:
            return "chybi"
        try:
            is_stale = model_eval_runtime.is_tab5_holdout_ranking_stale(
                record.meta,
                data_path=str(context.get("data_path") or ""),
                fee_per_trade=float(context.get("fee_per_trade", 0.0) or 0.0),
                model_path=record.model_path,
                meta_path=record.meta_path,
            )
        except OSError:
            return "chybi"
        return "neaktualni" if is_stale else "aktualni"

    def _comparison_snapshots(
        self,
        records: list[RankingRecord],
        context: dict[str, Any] | None,
    ) -> list[ModelComparisonSnapshot]:
        snapshots: list[ModelComparisonSnapshot] = []
        for record in records:
            ranking = record.ranking or {}
            base_payload = model_eval_runtime.get_tab5_holdout_base_payload(ranking, fallback_metrics=record.metrics)
            optimized_payload = model_eval_runtime.get_tab5_holdout_optimized_payload(ranking)

            base_profit = _payload_value_with_flat_fallback(base_payload, ranking, "profit_h")
            if base_profit is None:
                base_profit = record.metrics.get("profit_net")

            bias_score, _, _ = _bias_score_from_meta(record.meta)
            stability_detail = _feature_stability_detail_from_meta(record.meta)
            note = _ranking_note_from_meta(record.meta)
            criterion = str(
                record.meta.get("criterion")
                or record.meta.get("selection_criterion")
                or record.meta.get("scoring_criterion")
                or "-"
            ).strip() or "-"

            snapshots.append(
                ModelComparisonSnapshot(
                    record_key=self._record_key(record),
                    model_name=record.model_path.name,
                    values={
                        "mode": _training_mode_label(record.meta),
                        "status": str(ranking.get("status") or "-"),
                        "freshness": self._comparison_freshness_label(record, context),
                        "optimized_profit": _format_optional_float(
                            _payload_value_with_flat_fallback(optimized_payload, ranking, "profit_h")
                        ),
                        "base_profit": _format_optional_float(base_profit),
                        "entry": _format_optional_float(
                            _payload_value_with_flat_fallback(optimized_payload, ranking, "entry_threshold")
                        ),
                        "exit": _format_optional_float(
                            _payload_value_with_flat_fallback(optimized_payload, ranking, "exit_threshold")
                        ),
                        "trades": _format_optional_float(
                            _payload_value_with_flat_fallback(optimized_payload, ranking, "trades_h")
                        ),
                        "sharpe": _format_optional_float(record.metrics.get("sharpe"), digits=3),
                        "bias": "-" if bias_score is None else f"{bias_score:+.3f}",
                        "stability": (
                            "-"
                            if stability_detail.average_score is None
                            else f"{stability_detail.average_score:.3f}"
                        ),
                        "criterion": criterion,
                        "horizon": str(record.meta.get("label_horizon_bars") or "-"),
                        "tp_bps": _format_optional_float(record.meta.get("label_take_profit_bps")),
                        "sl_bps": _format_optional_float(record.meta.get("label_stop_loss_bps")),
                        "created": record.created or "-",
                        "note": note or "-",
                    },
                )
            )
        return snapshots

    def _comparison_summary_text(
        self,
        snapshots: list[ModelComparisonSnapshot],
        context: dict[str, Any],
    ) -> str:
        data_path = str(context.get("data_path") or "")
        fee = float(context.get("fee_per_trade", 0.0) or 0.0)
        entry = float(context.get("entry_threshold", 0.0) or 0.0)
        exit_thr = float(context.get("exit_threshold", 0.0) or 0.0)
        freshness_counts = {
            label: sum(1 for snapshot in snapshots if snapshot.values.get("freshness") == label)
            for label in ("aktualni", "neaktualni", "chybi", "bez CSV")
        }
        parts = [
            f"CSV: {Path(data_path).name}",
            f"fee/trade: {fee:.3f}",
            f"Entry/Exit start: {entry:.2f}/{exit_thr:.2f}",
        ]
        if freshness_counts["aktualni"]:
            parts.append(f"aktualni: {freshness_counts['aktualni']}")
        if freshness_counts["neaktualni"]:
            parts.append(f"neaktualni: {freshness_counts['neaktualni']}")
        if freshness_counts["chybi"]:
            parts.append(f"bez rankingu: {freshness_counts['chybi']}")
        return " | ".join(parts)

    @staticmethod
    def _comparison_export_rows(
        snapshots: list[ModelComparisonSnapshot],
        context: dict[str, Any],
        *,
        exported_at: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        data_path = str(context.get("data_path") or "")
        fee_per_trade = float(context.get("fee_per_trade", 0.0) or 0.0)
        entry_threshold = float(context.get("entry_threshold", 0.0) or 0.0)
        exit_threshold = float(context.get("exit_threshold", 0.0) or 0.0)
        for snapshot in snapshots:
            row: dict[str, Any] = {
                "exported_at": exported_at,
                "scope_mode": "holdout",
                "model_path": snapshot.record_key,
                "model_name": snapshot.model_name,
                "data_path": data_path,
                "fee_per_trade": fee_per_trade,
                "ranking_context_entry_threshold": entry_threshold,
                "ranking_context_exit_threshold": exit_threshold,
            }
            for metric_key, _metric_label in COMPARISON_METRIC_SPECS:
                row[metric_key] = snapshot.values.get(metric_key, "-")
            rows.append(row)
        return rows

    @staticmethod
    def _write_comparison_export(
        out_path: Path,
        *,
        file_format: str,
        rows: list[dict[str, Any]],
        context: dict[str, Any],
        exported_at: str,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = str(file_format or "").strip().lower()
        if fmt == "csv":
            fieldnames = [
                "exported_at",
                "scope_mode",
                "model_path",
                "model_name",
                "data_path",
                "fee_per_trade",
                "ranking_context_entry_threshold",
                "ranking_context_exit_threshold",
                *[metric_key for metric_key, _metric_label in COMPARISON_METRIC_SPECS],
            ]
            with out_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return
        if fmt == "json":
            payload = {
                "exported_at": exported_at,
                "comparison_context": {
                    "scope_mode": "holdout",
                    "data_path": str(context.get("data_path") or ""),
                    "fee_per_trade": float(context.get("fee_per_trade", 0.0) or 0.0),
                    "ranking_context_entry_threshold": float(context.get("entry_threshold", 0.0) or 0.0),
                    "ranking_context_exit_threshold": float(context.get("exit_threshold", 0.0) or 0.0),
                },
                "models": rows,
            }
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            return
        raise ValueError(f"Unsupported comparison export format: {file_format}")

    def _comparison_export_suggested_path(self, *, file_format: str, context: dict[str, Any]) -> Path:
        models_dir = self._models_dir() or DEFAULT_MODEL_DIR
        out_dir = models_dir / "evals"
        stem = Path(str(context.get("data_path") or "comparison")).stem or "comparison"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = ".json" if str(file_format).lower() == "json" else ".csv"
        return out_dir / f"compare_{stem}_{stamp}{suffix}"

    def _export_comparison_snapshots(
        self,
        snapshots: list[ModelComparisonSnapshot],
        context: dict[str, Any],
        *,
        file_format: str,
    ) -> None:
        fmt = str(file_format or "").strip().lower()
        if fmt not in {"csv", "json"}:
            QMessageBox.warning(self, "Model Ranking", f"Neznamy exportni format: {file_format}")
            return
        if not snapshots:
            QMessageBox.information(self, "Model Ranking", "Neni co exportovat.")
            return

        suggested_path = self._comparison_export_suggested_path(file_format=fmt, context=context)
        filter_text = "CSV (*.csv)" if fmt == "csv" else "JSON (*.json)"
        title = "Ulozit compare export (CSV)" if fmt == "csv" else "Ulozit compare export (JSON)"
        out_path_str, _selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            suggested_path.as_posix(),
            filter_text,
        )
        if not out_path_str:
            return

        out_path = Path(out_path_str)
        exported_at = model_eval_runtime.utc_now_iso()
        rows = self._comparison_export_rows(snapshots, context, exported_at=exported_at)
        try:
            self._write_comparison_export(
                out_path,
                file_format=fmt,
                rows=rows,
                context=context,
                exported_at=exported_at,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Model Ranking", f"Export compare selhal: {exc}")
            return
        self.lbl_status.setText(f"Status: compare export ulozen -> {out_path.name}")

    def _show_comparison_dialog(self, records: list[RankingRecord], context: dict[str, Any]) -> None:
        records_by_key = {self._record_key(record): record for record in records}
        snapshots = self._comparison_snapshots(records, context)
        dialog = ModelComparisonDialog(
            self,
            snapshots=snapshots,
            summary_text=self._comparison_summary_text(snapshots, context),
            on_load_record_key=lambda record_key: self._load_record_into_tabs(
                records_by_key.get(record_key),
                activate_eval_tab=True,
                auto_evaluate=False,
            ),
            on_export=lambda file_format: self._export_comparison_snapshots(
                snapshots,
                context,
                file_format=file_format,
            ),
        )
        dialog.exec()

    def _open_comparison_for_records(
        self,
        *,
        records: list[RankingRecord],
        source_label: str,
        empty_message: str,
    ) -> None:
        if self._is_busy():
            self.lbl_status.setText("Status: porovnani nelze otevrit behem batch prepoctu")
            return

        context = self._current_eval_context()
        if not isinstance(context, dict) or not str(context.get("data_path") or "").strip():
            QMessageBox.warning(self, "Model Ranking", "V Tab 4 musi byt nactene CSV s historickymi daty.")
            return

        if len(records) < 2:
            self.lbl_status.setText(empty_message)
            return
        if len(records) > MAX_COMPARISON_MODELS:
            QMessageBox.information(
                self,
                "Model Ranking",
                f"Porovnat lze najednou max {MAX_COMPARISON_MODELS} modelu. Omez scope a akci zopakuj.",
            )
            self.lbl_status.setText(
                f"Status: porovnani omezeno na max {MAX_COMPARISON_MODELS} modelu na jednu akci"
            )
            return

        self.lbl_status.setText(f"Status: oteviram porovnani pro {len(records)} modelu ({source_label})")
        self._show_comparison_dialog(records, context)

    def _on_compare_selected_clicked(self) -> None:
        records, source = self._comparison_target_records()
        source_label = "zatrzene" if source == "checked" else "oznacene"
        self._open_comparison_for_records(
            records=records,
            source_label=source_label,
            empty_message="Status: pro porovnani vyber alespon 2 modely (zatrzene nebo oznacene)",
        )

    def _on_compare_filtered_clicked(self) -> None:
        if self._has_active_filters():
            self._open_comparison_for_records(
                records=self._filtered_records(),
                source_label="filtrovane",
                empty_message="Status: pro porovnani potrebuji alespon 2 filtrovane modely",
            )
            return

        latest_shortlist = self._latest_shortlist_artifact()
        if latest_shortlist is not None:
            shortlist_records = [
                record for record in self.records if self._record_matches_shortlist(record, latest_shortlist)
            ]
            self._open_comparison_for_records(
                records=shortlist_records,
                source_label="posledni shortlist",
                empty_message="Status: posledni shortlist nema alespon 2 odpovidajici modely k porovnani",
            )
            return

        self.lbl_status.setText("Status: pro 'Porovnat filtrovane' nejprve zapni shortlist nebo jiny filtr")

    def _has_active_filters(self) -> bool:
        return (
            bool(self._active_shortlist_path)
            or
            bool(self._mode_filter_keys)
            or
            str(self.cmb_stability_filter.currentData() or STABILITY_FILTER_ALL) != STABILITY_FILTER_ALL
            or str(self.cmb_bias_filter.currentData() or BIAS_FILTER_ALL) != BIAS_FILTER_ALL
        )

    def _on_selection_changed(self) -> None:
        records = self._selected_records()
        checked_count = len(self._checked_records())
        if not records:
            self.lbl_selected.setText("Vybrano: -" if checked_count <= 0 else f"Vybrano: - | zatrzeno: {checked_count}")
            return
        if len(records) > 1:
            suffix = f" | zatrzeno: {checked_count}" if checked_count > 0 else ""
            self.lbl_selected.setText(f"Vybrano: {len(records)} modelu{suffix}")
            return
        record = records[0]
        ranking = record.ranking or {}
        status = str(ranking.get("status") or "-")
        suffix = f" | zatrzeno: {checked_count}" if checked_count > 0 else ""
        self.lbl_selected.setText(f"Vybrano: {record.model_path.name} | status={status}{suffix}")

    def _update_batch_progress_label(self) -> None:
        total = max(int(self._batch_total or 0), 0)
        if total <= 0:
            self.lbl_batch_progress.setText("Prubeh batch: -")
            return

        completed = max(0, min(int(self._batch_completed or 0), total))
        if self._is_busy():
            if self._batch_current_index > 0 and self._batch_current_model:
                self.lbl_batch_progress.setText(
                    f"Prubeh batch: hotovo {completed}/{total} | aktualne {self._batch_current_index}/{total}: {self._batch_current_model}"
                )
            else:
                self.lbl_batch_progress.setText(f"Prubeh batch: hotovo {completed}/{total}")
            return

        if self._batch_cancel_requested and not self._batch_result_received:
            suffix = f" | preruseno pri {self._batch_current_model}" if self._batch_current_model else ""
            self.lbl_batch_progress.setText(f"Prubeh batch: zastaveno po {completed}/{total}{suffix}")
            return

        self.lbl_batch_progress.setText(f"Prubeh batch: hotovo {completed}/{total}")

    def _on_batch_progress_text(self, text: str) -> None:
        message = str(text or "").strip()
        if not message:
            return

        match = _BATCH_PROGRESS_RE.match(message)
        if match is not None:
            current_index = max(int(match.group(1)), 0)
            total = max(int(match.group(2)), current_index)
            self._batch_total = max(self._batch_total, total)
            self._batch_current_index = current_index
            self._batch_completed = max(self._batch_completed, current_index - 1)
            self._batch_current_model = match.group(3).strip()
            self.lbl_status.setText(f"Status: {message}")
            self._update_batch_progress_label()
            return

        if self._batch_current_index > 0 and self._batch_total > 0 and self._batch_current_model:
            self.lbl_status.setText(
                f"Status: model {self._batch_current_index}/{self._batch_total} ({self._batch_current_model}) | {message}"
            )
        else:
            self.lbl_status.setText(f"Status: {message}")
        self._update_batch_progress_label()

    def _on_stop_recompute_clicked(self) -> None:
        worker = self._ranking_worker
        if worker is None or self._batch_cancel_requested:
            return

        self._batch_cancel_requested = True
        self.btn_stop_recompute.setEnabled(False)
        self.lbl_status.setText("Status: zastavuji prepocet...")
        self._update_batch_progress_label()
        worker.stop()

    def _is_busy(self) -> bool:
        return self._ranking_worker is not None

    def _records_requiring_ranking_from(
        self,
        records: list[RankingRecord],
        context: dict[str, Any],
    ) -> list[RankingRecord]:
        stale: list[RankingRecord] = []
        data_path = str(context.get("data_path") or "")
        fee_per_trade = float(context.get("fee_per_trade", 0.0) or 0.0)
        if not data_path:
            return stale
        for record in records:
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

    def _records_requiring_ranking(self, context: dict[str, Any]) -> list[RankingRecord]:
        return self._records_requiring_ranking_from(list(self.records), context)

    def _trigger_recompute_for_records(
        self,
        *,
        candidate_records: list[RankingRecord],
        empty_message: str,
        up_to_date_message: str,
        force_recompute: bool = False,
    ) -> None:
        if self._is_busy():
            return
        context = self._current_eval_context()
        if not isinstance(context, dict):
            QMessageBox.warning(self, "Model Ranking", "V Tab 4 musi byt nactene CSV s historickymi daty.")
            return
        if not candidate_records:
            self.lbl_status.setText(empty_message)
            return
        pending = list(candidate_records) if force_recompute else self._records_requiring_ranking_from(candidate_records, context)
        if not pending:
            self.lbl_status.setText(up_to_date_message)
            return
        self._start_batch_worker(records=pending, context=context, full_recompute=force_recompute)

    def _refresh_pending_hint(self) -> None:
        if self._is_busy():
            return
        context = self._current_eval_context()
        if not isinstance(context, dict):
            return
        pending = self._records_requiring_ranking(context)
        if pending:
            self.lbl_status.setText(
                f"Status: aktualni CSV kontext zmeni ranking u {len(pending)} modelu | klikni 'Prepocitat neaktualni (H opt)'"
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

    def _recompute_profit_opt_target(self) -> tuple[list[RankingRecord], str, str]:
        checked_records = self._checked_records()
        if checked_records:
            return (
                checked_records,
                "Status: zadne zatrzene modely k prepocitu",
                "Status: zatrzene modely uz maji aktualni Profit(H opt)",
            )
        if self._has_active_filters():
            return (
                self._filtered_records(),
                "Status: zadne filtrovane modely k prepocitu",
                "Status: filtrovane modely uz maji aktualni Profit(H opt)",
            )
        return (
            list(self.records),
            "Status: zadne modely k prepocitu",
            "Status: vsechny modely uz maji aktualni Profit(H opt)",
        )

    def _on_recompute_profit_opt_clicked(self) -> None:
        candidate_records, empty_message, up_to_date_message = self._recompute_profit_opt_target()
        self._trigger_recompute_for_records(
            candidate_records=candidate_records,
            empty_message=empty_message,
            up_to_date_message=up_to_date_message,
            force_recompute=False,
        )

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
        self._batch_total = len(records)
        self._batch_completed = 0
        self._batch_current_index = 0
        self._batch_current_model = ""
        self._batch_cancel_requested = False
        self._batch_result_received = False
        self._update_batch_progress_label()
        self.lbl_status.setText(
            f"Status: {'plny' if full_recompute else 'inkrementalni'} prepocet {len(records)} modelu..."
        )
        self.btn_recompute_profit_opt.setEnabled(False)
        self.btn_stop_recompute.setEnabled(True)

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
        worker.progress_text.connect(self._on_batch_progress_text)
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
        prepared_cache: dict[tuple[Any, ...], model_eval_runtime.PreparedEvaluationData] = {}
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
                prepared_key = model_eval_runtime.prepared_evaluation_cache_key(data_path, loaded.metadata)
                prepared = prepared_cache.get(prepared_key)
                if prepared is None:
                    prepared = model_eval_runtime.load_prepared_evaluation_data(
                        data_path,
                        metadata=loaded.metadata,
                        progress_cb=progress_cb,
                    )
                    prepared_cache[prepared_key] = prepared
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
                    exit_policy=evaluation.exit_policy,
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
                    exit_policy=evaluation.exit_policy,
                )
                ranking_payload = model_eval_runtime.build_tab5_holdout_ranking_payload(
                    data_path=data_path,
                    fee_per_trade=float(fee_per_trade),
                    metadata=loaded.metadata,
                    exit_policy=evaluation.exit_policy,
                    base_entry_threshold=float(current_entry),
                    base_exit_threshold=float(current_exit),
                    base_metrics=evaluation.results,
                    optimized_entry_threshold=float(search.best_entry),
                    optimized_exit_threshold=float(search.best_exit),
                    optimized_metrics=optimized_metrics,
                    entry_threshold=float(search.best_entry),
                    exit_threshold=float(search.best_exit),
                    metrics=optimized_metrics,
                    status="ok",
                )
                model_eval_runtime.set_tab5_holdout_ranking(
                    meta,
                    ranking_payload,
                    exit_policy=evaluation.exit_policy,
                )
                write_sidecar_model_meta(model_path, meta)
                updated += 1
            except Exception as exc:
                failures += 1
                status = model_eval_runtime.ranking_status_from_error_message(str(exc))
                ranking_payload = model_eval_runtime.build_tab5_holdout_ranking_payload(
                    data_path=data_path,
                    fee_per_trade=float(fee_per_trade),
                    metadata=meta,
                    entry_threshold=None,
                    exit_threshold=None,
                    metrics=None,
                    status=status,
                    error=str(exc),
                )
                model_eval_runtime.set_tab5_holdout_ranking(meta, ranking_payload)
                write_sidecar_model_meta(model_path, meta)

        return {
            "updated": int(updated),
            "failures": int(failures),
            "requested": int(total),
            "full_recompute": bool(full_recompute),
        }

    def _on_batch_result(self, req_id: int, result: dict[str, Any] | None) -> None:
        if req_id != self._ranking_request_id or not isinstance(result, dict):
            return

        self._batch_result_received = True
        updated = int(result.get("updated", 0) or 0)
        failures = int(result.get("failures", 0) or 0)
        requested = int(result.get("requested", updated + failures) or 0)
        mode = "plny" if bool(result.get("full_recompute")) else "inkrementalni"
        self._batch_total = max(self._batch_total, requested)
        self._batch_completed = max(self._batch_completed, min(self._batch_total, updated + failures))
        self._batch_current_index = self._batch_completed
        self._batch_current_model = ""

        self._refresh_list(force=True)
        self._on_selection_changed()
        self._update_batch_progress_label()
        self.lbl_status.setText(
            f"Status: {mode} prepocet dokoncen | aktualizovano {updated}/{requested}, chyby {failures}"
        )

    def _on_batch_error(self, req_id: int, msg: str) -> None:
        if req_id != self._ranking_request_id:
            return
        self.lbl_status.setText(f"Status: prepocet selhal - {msg}")
        self._update_batch_progress_label()
        QMessageBox.warning(self, "Model Ranking", msg)

    def _on_batch_finished(self, req_id: int) -> None:
        if req_id != self._ranking_request_id:
            return
        canceled = self._batch_cancel_requested and not self._batch_result_received
        self._ranking_worker = None
        self.btn_recompute_profit_opt.setEnabled(True)
        self.btn_stop_recompute.setEnabled(False)
        if canceled:
            self.lbl_status.setText(
                f"Status: prepocet zastaven | dokonceno nejmene {self._batch_completed}/{self._batch_total}"
            )
        self._update_batch_progress_label()

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
                optimized_payload = model_eval_runtime.get_tab5_holdout_optimized_payload(ranking)
                entry_threshold = model_eval_runtime.safe_float(
                    _payload_value_with_flat_fallback(optimized_payload, ranking, "entry_threshold")
                )
                exit_threshold = model_eval_runtime.safe_float(
                    _payload_value_with_flat_fallback(optimized_payload, ranking, "exit_threshold")
                )
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

    def _clear_deleted_model_references(self, deleted_records: list[RankingRecord]) -> None:
        if not deleted_records:
            return
        deleted_paths = {self._record_key(record) for record in deleted_records}

        win = self.window()
        if win is None:
            return

        tab_eval = getattr(win, "tab_eval", None)
        if tab_eval is not None:
            try:
                eval_path = getattr(tab_eval, "model_path", None)
                if eval_path:
                    eval_key = str(Path(str(eval_path)).resolve())
                else:
                    eval_key = ""
            except Exception:
                eval_key = str(getattr(tab_eval, "model_path", "") or "")
            if eval_key in deleted_paths:
                try:
                    setattr(tab_eval, "model_path", None)
                except Exception:
                    pass
                try:
                    setattr(tab_eval, "loaded_model", None)
                except Exception:
                    pass
                try:
                    setattr(tab_eval, "model_metadata", None)
                except Exception:
                    pass
                try:
                    if hasattr(tab_eval, "model_label"):
                        tab_eval.model_label.setText("Model: —")
                except Exception:
                    pass

        tab_live = getattr(win, "tab_live", None)
        if tab_live is not None:
            try:
                if hasattr(tab_live, "le_model_path"):
                    raw = str(tab_live.le_model_path.text() or "")
                    kept = []
                    for part in re.split(r"[;\r\n]+", raw):
                        txt = str(part or "").strip()
                        if not txt:
                            continue
                        try:
                            key = str(Path(txt).resolve())
                        except Exception:
                            key = txt
                        if key not in deleted_paths:
                            kept.append(txt)
                    tab_live.le_model_path.setText(";".join(kept))
            except Exception:
                pass

    def _delete_selected_via_shortcut(self) -> None:
        if self._delete_in_progress:
            return
        try:
            self._on_delete_selected()
        except Exception as exc:
            log_path = _write_ranking_crash_log("delete", exc)
            detail = f"\n\nCrash log: {log_path}" if log_path is not None else ""
            QMessageBox.critical(
                self,
                "Model Ranking",
                f"Mazani modelu selhalo neocekavanou chybou: {exc}{detail}",
            )

    def _on_delete_selected(self) -> None:
        if self._delete_in_progress:
            return
        self._delete_in_progress = True
        try:
            if self._is_busy():
                self.lbl_status.setText("Status: nelze mazat modely behem probihajiciho batch procesu")
                return

            records = list(self._selected_records())
            if not records:
                return
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

            deleted_keys = {self._record_key(record) for record in records}
            selected_rows = self._selected_rows()
            timer_was_active = self.timer.isActive()
            if timer_was_active:
                self.timer.stop()

            deleted = 0
            failures: list[str] = []
            try:
                self.tbl.blockSignals(True)
                self.tbl.clearSelection()
                for record in records:
                    sidecar_path = model_sidecar_meta_path(record.model_path)
                    try:
                        if sidecar_path.exists():
                            sidecar_path.unlink()
                        if record.model_path.exists():
                            record.model_path.unlink()
                        deleted += 1
                    except Exception as exc:
                        failures.append(f"{record.model_path.name}: {exc}")
                self._checked_model_paths.difference_update(deleted_keys)
                self._clear_deleted_model_references(records)
                self._refresh_list(force=True)
            finally:
                self.tbl.blockSignals(False)
                if timer_was_active:
                    self.timer.start()

            if self.tbl.rowCount() > 0:
                anchor_row = selected_rows[0] if selected_rows else 0
                next_row = min(anchor_row, self.tbl.rowCount() - 1)
                self.tbl.setCurrentCell(next_row, 0)
            else:
                self._on_selection_changed()

            self.lbl_status.setText(
                f"Status: smazano {deleted} modelu, chyby {len(failures)}"
            )
            if failures:
                QMessageBox.warning(
                    self,
                    "Mazani modelu",
                    "Nektere modely se nepodarilo smazat:\n\n" + "\n".join(failures[:10]),
                )
        finally:
            self._delete_in_progress = False
