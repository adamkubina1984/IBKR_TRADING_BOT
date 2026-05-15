from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
LOG_DIR = Path(__file__).parent.parent / "logs"
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
COL_BIAS = 9
COL_FEATURES = 10
COL_STABILITY = 11
COL_CREATED = 12
COL_CHECK = 13
COL_NOTE = 14

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
    trades_band_score: float
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

    def sort_key(self) -> tuple[int, float, float, float, float, float, float]:
        return (
            int(self.tier),
            float(self.trades_band_score),
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


def _trade_count_preference_score(
    n_trades: float,
    *,
    hard_min: float = 60.0,
    preferred_low: float = 150.0,
    preferred_high: float = 300.0,
    soft_max: float = 450.0,
) -> float:
    try:
        value = float(n_trades)
    except Exception:
        return 0.0
    if not np.isfinite(value) or value < hard_min:
        return 0.0
    if value <= preferred_low:
        span = max(preferred_low - hard_min, 1.0)
        return float(np.clip((value - hard_min) / span, 0.0, 1.0))
    if value <= preferred_high:
        return 1.0
    if value <= soft_max:
        span = max(soft_max - preferred_high, 1.0)
        return float(np.clip(1.0 - ((value - preferred_high) / span), 0.0, 1.0))
    return 0.0


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
        self.btn_recompute_all = QPushButton("Prepocitat Profit(H) pro vsechny", self)
        self.btn_recompute_filtered = QPushButton("Prepocitat Profit(H) pro filtrovane", self)
        self.btn_recompute_checked = QPushButton("Prepocitat Profit(H) pro zatrzene", self)
        self.btn_strict_top5 = QPushButton("Spustit strict Top 10", self)
        self.btn_delete = QPushButton("Smazat vybrane", self)
        self.btn_load = QPushButton("Nacist vybrany model", self)
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
        self.btn_delete.hide()
        self.btn_load.hide()

        self.lbl_context = QLabel("CSV z Tab 4: - | fee/trade: - | Entry/Exit start: -/-", self)
        self.lbl_status = QLabel("Status: pripraveno", self)
        self.lbl_selected = QLabel("Vybrano: -", self)

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

        actions = QHBoxLayout()
        actions.addWidget(self.btn_recompute_all)
        actions.addWidget(self.btn_recompute_filtered)
        actions.addWidget(self.btn_recompute_checked)
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
        self._checked_model_paths: set[str] = set()
        self._records_by_key: dict[str, RankingRecord] = {}
        self._delete_in_progress = False

        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        self.btn_recompute_all.clicked.connect(self._on_recompute_all_clicked)
        self.btn_recompute_filtered.clicked.connect(self._on_recompute_filtered_clicked)
        self.btn_recompute_checked.clicked.connect(self._on_recompute_checked_clicked)
        self.btn_strict_top5.clicked.connect(self._on_strict_top5_clicked)
        self.btn_delete.clicked.connect(self._on_delete_selected)
        self.btn_load.clicked.connect(self._on_load_selected)
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
        self.timer.start()

        self._refresh_list(force=True)
        self._update_context_label()

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
        if not force and snapshot == self._last_snapshot:
            return False
        self._last_snapshot = snapshot
        self.records = discover_ranking_models(models_dir)
        self._records_by_key = {self._record_key(record): record for record in self.records}
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

    def _filtered_records(self) -> list[RankingRecord]:
        filtered = list(self.records)

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
        limit: int = 10,
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

            base_payload = model_eval_runtime.get_tab5_holdout_base_payload(ranking, fallback_metrics=record.metrics)
            optimized_payload = model_eval_runtime.get_tab5_holdout_optimized_payload(ranking)
            optimized_profit = model_eval_runtime.safe_float(optimized_payload.get("profit_h"))
            if optimized_profit is None or optimized_profit <= 0.0:
                rejections.append(StrictRejection(record=record, reason="Profit(H opt) musi byt kladny"))
                continue

            base_profit = model_eval_runtime.safe_float(base_payload.get("profit_h"))
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

            trades = model_eval_runtime.safe_float(_payload_value_with_flat_fallback(optimized_payload, ranking, "trades_h"))
            trades_value = float(trades if trades is not None else 0.0)
            if trades is None or trades_value < 60.0:
                rejections.append(StrictRejection(record=record, reason="Trades(H) je pod minimem 60 pro strict shortlist"))
                continue
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
                    trades=trades_value,
                    trades_band_score=_trade_count_preference_score(trades_value),
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
            rejections.append(StrictRejection(record=candidate.record, reason=f"mimo Top {int(max(1, int(limit)))} po serazeni"))

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
        pending = self._records_requiring_ranking_from(candidate_records, context)
        if not pending:
            self.lbl_status.setText(up_to_date_message)
            return
        self._start_batch_worker(records=pending, context=context, full_recompute=False)

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
        self._trigger_recompute_for_records(
            candidate_records=list(self.records),
            empty_message="Status: zadne modely k prepocitu",
            up_to_date_message="Status: vsechny modely uz maji aktualni Profit(H)",
        )

    def _on_recompute_filtered_clicked(self) -> None:
        self._trigger_recompute_for_records(
            candidate_records=self._filtered_records(),
            empty_message="Status: zadne filtrovane modely k prepocitu",
            up_to_date_message="Status: filtrovane modely uz maji aktualni Profit(H)",
        )

    def _on_recompute_checked_clicked(self) -> None:
        self._trigger_recompute_for_records(
            candidate_records=self._checked_records(),
            empty_message="Status: zadne zatrzene modely k prepocitu",
            up_to_date_message="Status: zatrzene modely uz maji aktualni Profit(H)",
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
        self.lbl_status.setText(
            f"Status: {'plny' if full_recompute else 'inkrementalni'} prepocet {len(records)} modelu..."
        )
        self.btn_recompute_all.setEnabled(False)
        self.btn_recompute_filtered.setEnabled(False)
        self.btn_recompute_checked.setEnabled(False)
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
        self.btn_recompute_filtered.setEnabled(True)
        self.btn_recompute_checked.setEnabled(True)
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
                "Pro strict Top 10 musi byt v Tab 2 nactene odpovidajici treninkove CSV.",
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
                "Strict Top 10 lze spustit jen kdyz Tab 2 a Tab 4 pouzivaji stejny CSV snapshot.",
            )
            return

        shortlist = self._build_strict_shortlist(
            records=list(self.records),
            context=context,
            current_snapshot_signature=training_context["snapshot_signature"],
            limit=10,
        )
        selected = list(shortlist.get("selected") or [])
        rejected = list(shortlist.get("rejected") or [])
        if not selected:
            QMessageBox.warning(
                self,
                "Model Ranking",
                "Pro strict Top 10 se nenasel zadny vhodny kandidat v aktualnim ranking kontextu.",
            )
            return

        preview = self._strict_preview_text(selected, rejected)
        reply = QMessageBox.question(
            self,
            "Strict Top 10",
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
        self.btn_recompute_filtered.setEnabled(False)
        self.btn_recompute_checked.setEnabled(False)
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
                "strict_trigger": "ranking_top10",
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
        self.btn_recompute_filtered.setEnabled(True)
        self.btn_recompute_checked.setEnabled(True)
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
