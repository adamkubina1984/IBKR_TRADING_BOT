# gui/tab_model_evaluation.py
# Záložka 3 – Kontrola modelu na historických datech
# ---------------------------------------------------
# Opravy / vylepšení:
# - Normalizace predikcí po prahování na {-1,0,+1} pro stabilní metriky/grafy.
# - Respektování expected_features z metadat uloženého modelu (pokud jsou k dispozici).
# - Grafy kreslené lokálně se zachováním robustních fallbacků.

import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ibkr_trading_bot.core.services import model_eval_service as model_eval_runtime
from ibkr_trading_bot.core.services.signal_policy import (
    DEFAULT_EXIT_POLICY,
    apply_confidence_entry_threshold,
    apply_exit_confidence_threshold,
    normalize_signal_array,
    resolve_exit_policy_setting,
)
from ibkr_trading_bot.core.services.auto_threshold_search import run_auto_threshold_search
from ibkr_trading_bot.core.services.evaluation_service import EvaluationService
from ibkr_trading_bot.core.services.trade_executor import replay_signals_over_market_data
from ibkr_trading_bot.core.services.model_service import (
    build_sklearn_version_warning,
)
from ibkr_trading_bot.gui.components.workers import TaskWorker

# Import zůstává (pro případ budoucího přepnutí), ale grafy kreslíme lokálně
try:
    from ibkr_trading_bot.core.utils.plotting import (
        draw_equity_chart as _utils_draw_eq,
        draw_histogram as _utils_draw_hist,
        draw_rolling_chart as _utils_draw_roll,
    )
except Exception:
    _utils_draw_eq = _utils_draw_hist = _utils_draw_roll = None

from ibkr_trading_bot.gui.timeframe import DEFAULT_TIMEFRAME, TIMEFRAME_OPTIONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR   = PROJECT_ROOT / "model_outputs"
RAW_DIR     = PROJECT_ROOT / "data" / "raw"
LAST_DATA_CSV_PATH_KEY = "last_data_csv_path"


@dataclass
class EvaluationPayload:
    X_current: pd.DataFrame | np.ndarray
    y_true_current: np.ndarray
    df_current: pd.DataFrame
    close_series: pd.Series | None
    confidence_arr: np.ndarray
    y_pred_raw: np.ndarray
    y_pred_used: np.ndarray
    results: dict
    scope_info: dict
    threshold_source: str
    thr_short: float
    thr_long: float
    entry_threshold: float
    exit_threshold: float


@dataclass
class AutoThresholdPayload:
    best_entry: float
    best_exit: float
    best_score: float
    best_metrics: dict | None


def _safe_float_eval(value):
    try:
        if value is None:
            return None
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return None


def _feature_names_for_model_eval(model) -> list[str] | None:
    try:
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            return [str(x) for x in list(names)]
    except Exception:
        pass
    try:
        steps = getattr(model, "steps", None)
        if steps:
            last = steps[-1][1]
            names = getattr(last, "feature_names_in_", None)
            if names is not None:
                return [str(x) for x in list(names)]
    except Exception:
        pass
    return None


def _tail_rows_eval(obj, n_rows: int):
    if obj is None:
        return None
    n = int(max(0, n_rows))
    if isinstance(obj, pd.DataFrame):
        return obj.tail(n).reset_index(drop=True)
    if isinstance(obj, pd.Series):
        return obj.tail(n).reset_index(drop=True)
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr
    return arr[-n:] if n < arr.shape[0] else arr


def _infer_holdout_bars_from_metadata_eval(meta: dict, n_rows: int) -> int | None:
    n = int(max(0, n_rows))
    if n <= 0:
        return None
    try:
        nh = int(meta.get("n_holdout_bars", 0))
        if nh > 0:
            return int(min(n, nh))
    except Exception:
        pass
    hold_sel = meta.get("holdout_selection") if isinstance(meta, dict) else None
    if isinstance(hold_sel, dict):
        try:
            applied = int(hold_sel.get("applied_bars", 0))
            if applied > 0:
                return int(min(n, applied))
        except Exception:
            pass
        try:
            pct = float(hold_sel.get("requested_pct"))
            if np.isfinite(pct) and pct > 0.0:
                calc = int(round(float(n) * float(np.clip(pct, 0.0, 0.95))))
                if calc > 0:
                    return int(min(n, calc))
        except Exception:
            pass
    return None


def _apply_eval_scope_eval(X, y_true, df_for_metrics, scope_mode: str, metadata: dict):
    lengths = []
    try:
        lengths.append(int(len(X)))
    except Exception:
        pass
    if y_true is not None:
        try:
            lengths.append(int(len(y_true)))
        except Exception:
            pass
    if df_for_metrics is not None:
        try:
            lengths.append(int(len(df_for_metrics)))
        except Exception:
            pass
    if not lengths:
        raise ValueError("Nelze urcit delku datasetu pro evaluaci.")

    n_base = int(max(0, min(lengths)))
    if n_base <= 0:
        raise ValueError("Dataset pro evaluaci je prazdny.")

    X_aligned = _tail_rows_eval(X, n_base)
    y_aligned = _tail_rows_eval(y_true, n_base) if y_true is not None else None
    df_aligned = _tail_rows_eval(df_for_metrics, n_base) if df_for_metrics is not None else None

    mode = scope_mode if scope_mode in {"holdout", "full"} else "holdout"
    if mode == "holdout":
        n_hold = _infer_holdout_bars_from_metadata_eval(metadata or {}, n_base)
        if n_hold is not None and n_hold > 0:
            n_eval = int(min(n_base, n_hold))
            X_eval = _tail_rows_eval(X_aligned, n_eval)
            y_eval = _tail_rows_eval(y_aligned, n_eval) if y_aligned is not None else None
            df_eval = _tail_rows_eval(df_aligned, n_eval) if df_aligned is not None else None
        else:
            n_eval = n_base
            X_eval, y_eval, df_eval = X_aligned, y_aligned, df_aligned
    else:
        n_eval = n_base
        X_eval, y_eval, df_eval = X_aligned, y_aligned, df_aligned

    scope_info = {
        "mode": mode,
        "applied_rows": int(n_eval),
        "total_rows": int(n_base),
    }
    return X_eval, y_eval, df_eval, scope_info


def _coerce_features_for_model_eval(X, model, metadata: dict):
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

    for c in list(dfX.columns):
        if (not pd.api.types.is_bool_dtype(dfX[c])) and (not pd.api.types.is_numeric_dtype(dfX[c])):
            dfX.drop(columns=[c], inplace=True, errors="ignore")

    exp = None
    if isinstance(metadata, dict):
        exp = metadata.get("expected_features") or metadata.get("features")
    if isinstance(exp, (list, tuple)) and all(isinstance(k, str) for k in exp):
        for k in exp:
            if k not in dfX.columns:
                dfX[k] = 0.0
        dfX = dfX[list(exp)]
        med = dfX.median(numeric_only=True)
        dfX = dfX.fillna(med).fillna(0.0)
        for c in dfX.columns:
            if not pd.api.types.is_bool_dtype(dfX[c]):
                dfX[c] = dfX[c].astype("float32", copy=False)
        return dfX

    names = _feature_names_for_model_eval(model)
    if names is not None:
        for k in names:
            if k not in dfX.columns:
                dfX[k] = 0.0
        dfX = dfX[names]

    med = dfX.median(numeric_only=True)
    dfX = dfX.fillna(med).fillna(0.0)
    for c in dfX.columns:
        if not pd.api.types.is_bool_dtype(dfX[c]):
            dfX[c] = dfX[c].astype("float32", copy=False)
    return dfX


def _align_X_for_model_eval(model, X):
    if isinstance(X, pd.DataFrame):
        Xdf = X.copy()
    else:
        Xdf = pd.DataFrame(X)

    names = _feature_names_for_model_eval(model)
    if names:
        for c in names:
            if c not in Xdf.columns:
                Xdf[c] = 0.0
        Xdf = Xdf.reindex(columns=names, fill_value=0.0)

    med = Xdf.median(numeric_only=True)
    Xdf = Xdf.fillna(med).fillna(0.0)
    for c in Xdf.columns:
        if not pd.api.types.is_bool_dtype(Xdf[c]):
            Xdf[c] = Xdf[c].astype("float32", copy=False)
    return Xdf


def _resolve_ternary_thresholds_eval(metadata: dict) -> tuple[float, float, str]:
    meta = metadata if isinstance(metadata, dict) else {}
    tshort = _safe_float_eval(meta.get("ternary_threshold_short"))
    tlong = _safe_float_eval(meta.get("ternary_threshold_long"))
    user = meta.get("user_settings")
    if isinstance(user, dict):
        if tshort is None:
            tshort = _safe_float_eval(user.get("ternary_threshold_short_eval"))
        if tlong is None:
            tlong = _safe_float_eval(user.get("ternary_threshold_long_eval"))
    if not isinstance(tshort, (int, float)) or not isinstance(tlong, (int, float)):
        raise ValueError(
            "Model neobsahuje platne ternarni prahy (ternary_threshold_short/long). "
            "Nahraj model natrenovany v nove pipeline."
        )
    return float(tshort), float(tlong), "model"


def _apply_confidence_threshold_eval(raw_pred, confidence, threshold):
    return apply_confidence_entry_threshold(raw_pred, confidence, threshold)


def _apply_exit_threshold_eval(y_pred: np.ndarray, confidence: np.ndarray, exit_thr: float) -> np.ndarray:
    return apply_exit_confidence_threshold(y_pred, confidence, exit_thr)


def _normalize_pred_eval(arr):
    return normalize_signal_array(arr)


def _safe_close_series_eval(df: pd.DataFrame | None):
    if not isinstance(df, pd.DataFrame):
        return None
    for c in ["close", "Close", "CLOSE", "adj_close", "Adj Close"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return None


# ---------------- Pomocné komponenty pro grafy ----------------
class MplCanvas(FigureCanvas):
    """Jednoduché plátno pro 1 osu."""
    def __init__(self, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


# ---------------- Hlavní widget záložky ----------------
class ModelEvaluationTab(QWidget):
    _eval_service = EvaluationService(None, None, None)

    def _open_folder(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            p = str(path)
            if sys.platform.startswith("win"):
                os.startfile(p)  # Windows
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])  # macOS
            else:
                subprocess.Popen(["xdg-open", p])  # Linux
        except Exception as e:
            QMessageBox.warning(self, "Nelze otevřít", f"Nešlo otevřít složku:\n{p}\n\n{e}")

    def __init__(self):
        super().__init__()

        # --- stavové proměnné ---
        self.model_path = None
        self.data_path = None
        self.loaded_model = None
        self.model_metadata = None

        self.df_current = None
        self.close_series = None

        self.X_current = None
        self.y_true_current = None
        self.y_pred_raw = None         # predikce bez filtru
        self.y_pred_used = None        # predikce po filtru (−1/0/+1)
        self.confidence_arr = None     # jistoty (0..1) pro filtr

        self.last_metrics = None       # poslední metriky (po filtru a nákladech)
        self.eval_scope_info = {"mode": "holdout", "applied_rows": 0, "total_rows": 0}
        self._last_ternary_threshold_source = "model"
        self._eval_worker: TaskWorker | None = None
        self._params_worker: TaskWorker | None = None
        self._auto_threshold_worker: TaskWorker | None = None
        self._retired_workers: list[TaskWorker] = []
        self._eval_request_id = 0
        self._params_request_id = 0
        self._auto_request_id = 0
        self._pending_auto_threshold_search = False
        self._pending_auto_threshold_dialog: AutoThresholdPayload | None = None
        self._ui_settings = QSettings("ibkr_trading_bot", "model_evaluation_tab")

        # --- UI layouty ---
        main_layout = QVBoxLayout(self)

        # ====== 1) SKUPINA: Vstupy ======
        files_group = QGroupBox("Vstupy")
        files_layout = QVBoxLayout()

        # řádek: výběr modelu
        model_row = QHBoxLayout()
        self.model_label = QLabel("Model: —")
        self.btn_open_model = QPushButton("Otevřít model…")
        self.btn_open_model.clicked.connect(self.on_open_model_clicked)
        model_row.addWidget(self.model_label, 1)
        model_row.addWidget(self.btn_open_model, 0)

        # řádek: výběr dat
        data_row = QHBoxLayout()
        self.data_label = QLabel("Data (CSV): —")
        self.btn_open_data = QPushButton("Otevřít data…")
        self.btn_open_data.clicked.connect(self.on_open_data_clicked)
        data_row.addWidget(self.data_label, 1)
        data_row.addWidget(self.btn_open_data, 0)

        # řádek: akce
        action_row = QHBoxLayout()
        self.status_label = QLabel("Status: Připraveno")

        self.btn_evaluate = QPushButton("Vyhodnotit")
        self.btn_evaluate.clicked.connect(self.on_evaluate_clicked)

        self.btn_export_trades = QPushButton("Exportovat obchody (CSV)")
        self.btn_export_trades.setEnabled(False)                 # zapne se až po vyhodnocení
        self.btn_export_trades.clicked.connect(self._export_trades_csv)

        action_row.addWidget(self.status_label, 1)
        action_row.addWidget(self.btn_evaluate, 0)
        action_row.addWidget(self.btn_export_trades, 0)

        files_layout.addLayout(model_row)
        files_layout.addLayout(data_row)
        files_layout.addLayout(action_row)
        files_group.setLayout(files_layout)

        # ====== 2) SKUPINA: Parametry vyhodnocení ======
        params_group = QGroupBox("Parametry vyhodnocení")
        params_layout = QHBoxLayout()

        self.cmb_tf = QComboBox()
        self.cmb_tf.addItems(TIMEFRAME_OPTIONS)
        self.cmb_tf.setCurrentText(DEFAULT_TIMEFRAME)

        # Náklady/obchod
        self.cost_label = QLabel("Náklady/obchod")
        self.cost_spin = QDoubleSpinBox()
        self.cost_spin.setRange(0.0, 1e9)
        self.cost_spin.setSingleStep(0.1)
        self.cost_spin.setDecimals(3)
        self.cost_spin.setValue(0.0)
        self.cost_spin.setToolTip("Komise + slippage v měně na každý obchod (odečítá se z PnL).")
        self.cost_spin.valueChanged.connect(self.on_params_changed)

        # Rolling metrika
        self.roll_label = QLabel("Rolling metr.")
        self.roll_combo = QComboBox()
        self.roll_combo.addItems(["Rolling profit (mean)", "Rolling Sharpe", "Rolling max DD"])
        self.roll_combo.setCurrentIndex(0)
        self.roll_combo.currentIndexChanged.connect(self.on_params_changed)

        # Rozsah evaluace
        self.scope_label = QLabel("Rozsah eval.")
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Holdout z modelu (Doporuceno)", userData="holdout")
        self.scope_combo.addItem("Cely dataset (Diagnostika)", userData="full")
        self.scope_combo.setCurrentIndex(0)
        self.scope_combo.setToolTip(
            "Doporuceno: vyhodnocovat pouze holdout segment podle metadat modelu.\n"
            "Cely dataset pouzij jen pro diagnostiku."
        )
        self.scope_combo.currentIndexChanged.connect(self._on_eval_scope_changed)

        params_layout.addWidget(self.cost_label)
        params_layout.addWidget(self.cost_spin)
        params_layout.addSpacing(16)
        params_layout.addWidget(self.roll_label)
        params_layout.addWidget(self.roll_combo)
        params_layout.addSpacing(16)
        params_layout.addWidget(self.scope_label)
        params_layout.addWidget(self.scope_combo)
        params_layout.addStretch(1)
        params_group.setLayout(params_layout)

        # ====== 2b) SKUPINA: Nastavení modelu (pro uložení do metadat) ======
        model_settings_group = QGroupBox("Nastavení modelu (uloží se do meta)")
        model_settings_layout = QHBoxLayout()

        # Entry threshold (pro live bot)
        et_label = QLabel("Entry Threshold:")
        self.et_spin = QDoubleSpinBox()
        self.et_spin.setRange(0.0, 1.0)
        self.et_spin.setSingleStep(0.01)
        self.et_spin.setDecimals(2)
        self.et_spin.setValue(0.6)
        self.et_spin.setToolTip("Minimální confidence pro otevření pozice")
        self.et_spin.valueChanged.connect(self._on_model_settings_changed)

        # Exit threshold
        ext_label = QLabel("Exit Threshold:")
        self.ext_spin = QDoubleSpinBox()
        self.ext_spin.setRange(0.0, 1.0)
        self.ext_spin.setSingleStep(0.01)
        self.ext_spin.setDecimals(2)
        self.ext_spin.setValue(0.7)
        self.ext_spin.setToolTip("Minimální confidence pro zavření pozice (0=vypnuto). Pokud confidence klesne pod tuto hodnotu, pozice se zavře.")
        self.ext_spin.valueChanged.connect(self._on_model_settings_changed)

        self.lbl_threshold_preview = QLabel("T(model): short=— long=— | T(active): short=— long=— src=model | Entry=0.600 Exit=0.700")
        self.lbl_threshold_preview.setToolTip(
            "T(model) = prahy ulozene v metadatech modelu.\n"
            "T(active) = prahy realne pouzite pri evaluaci podle aktualniho nastaveni."
        )

        # Tlačítko pro uložení
        self.btn_auto_thresholds = QPushButton("Auto Entry/Exit (max profit)")
        self.btn_auto_thresholds.setToolTip(
            "Najde automaticky Entry/Exit thresholdy, ktere maximalizuji profit_net "
            "na aktualnim eval scope (holdout/full)."
        )
        self.btn_auto_thresholds.clicked.connect(self._on_auto_thresholds_clicked)
        self.btn_save_model_settings = QPushButton("💾 Uložit nastavení do modelu")
        self.btn_save_model_settings.setToolTip("Uloží aktuální thresholdy do meta.json modelu pro reload v Tab 4")
        self.btn_save_model_settings.clicked.connect(self._on_save_model_settings)

        model_settings_layout.addWidget(et_label)
        model_settings_layout.addWidget(self.et_spin)
        model_settings_layout.addSpacing(12)
        model_settings_layout.addWidget(ext_label)
        model_settings_layout.addWidget(self.ext_spin)
        model_settings_layout.addSpacing(12)
        model_settings_layout.addWidget(self.lbl_threshold_preview, 1)
        model_settings_layout.addSpacing(12)
        model_settings_layout.addWidget(self.btn_auto_thresholds)
        model_settings_layout.addSpacing(8)
        model_settings_layout.addWidget(self.btn_save_model_settings)
        model_settings_layout.addStretch(1)
        model_settings_group.setLayout(model_settings_layout)

        # ====== 3) SKUPINA: Metriky ======
        metrics_group = QGroupBox("Metriky modelu")
        metrics_layout = QVBoxLayout()
        self.metrics_table = QTableWidget(0, 3)
        self.metrics_table.setHorizontalHeaderLabels(["Metrika", "Hodnota", "Význam"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.metrics_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.metrics_table.setToolTip(
            "Vyber radky metrik a stiskni Ctrl+C, nebo pouzij tlacitko Kopirovat metriky."
        )
        self.btn_copy_metrics = QPushButton("Kopirovat metriky")
        self.btn_copy_metrics.setToolTip(
            "Zkopiruje vybrane radky metrik; kdyz nic nevyberes, zkopiruje cely panel."
        )
        self.btn_copy_metrics.clicked.connect(self._copy_selected_metrics)
        self.btn_copy_key_metrics = QPushButton("Kopirovat klicove metriky")
        self.btn_copy_key_metrics.setToolTip(
            "Zkopiruje jen klicove metriky (profit, sharpe, PF, DD, long/short, trades)."
        )
        self.btn_copy_key_metrics.clicked.connect(self._copy_key_metrics)
        copy_row = QHBoxLayout()
        copy_row.addWidget(self.btn_copy_metrics, 0)
        copy_row.addWidget(self.btn_copy_key_metrics, 0)
        copy_row.addStretch(1)
        metrics_layout.addLayout(copy_row)
        metrics_layout.addWidget(self.metrics_table)
        
        metrics_group.setLayout(metrics_layout)

        # ====== 4) SKUPINA: Grafy ======
        charts_group = QGroupBox("Vizualizace")
        charts_layout = QVBoxLayout()
        self.canvas_equity = MplCanvas()
        self.canvas_hist = MplCanvas()
        self.canvas_rolling = MplCanvas()
        charts_layout.addWidget(self.canvas_equity)
        charts_layout.addWidget(self.canvas_hist)
        charts_layout.addWidget(self.canvas_rolling)
        charts_group.setLayout(charts_layout)

        # velikostní chování (1/3 metriky : 2/3 grafy)
        metrics_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        charts_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ====== Sestavení hlavního layoutu ======
        main_layout.addWidget(files_group)
        main_layout.addWidget(params_group)
        main_layout.addWidget(model_settings_group)

        content_row = QHBoxLayout()
        content_row.addWidget(metrics_group, 1)
        content_row.addWidget(charts_group, 2)
        content_row.setStretch(0, 1)
        content_row.setStretch(1, 2)
        main_layout.addLayout(content_row)

        # popisy metrik
        self.metric_descriptions = self._build_metric_descriptions()
        self.metrics_copy_shortcut = QShortcut(QKeySequence.Copy, self.metrics_table)
        self.metrics_copy_shortcut.activated.connect(self._copy_selected_metrics)
        self._params_timer = QTimer(self)
        self._params_timer.setSingleShot(True)
        self._params_timer.setInterval(150)
        self._params_timer.timeout.connect(self._run_params_recalc)
        self._refresh_threshold_preview()
        self._restore_last_data_path()

    # ---------------- Event handlery ----------------
    def on_open_model_clicked(self):
        # Dynamický a záložní start dir (po změně kořene projektu)
        try:
            project_root = Path(__file__).resolve().parents[1]
        except Exception:
            project_root = Path(os.getcwd())
        model_dir_dyn = project_root / "model_outputs"
        model_dir_abs = Path(r"C:\Users\adamk\Můj disk\Trader\ibkr_trading_bot\model_outputs")

        if model_dir_dyn.is_dir():
            start_dir = str(model_dir_dyn)
        elif model_dir_abs.is_dir():
            start_dir = str(model_dir_abs)
        else:
            start_dir = str(project_root)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Vyber model (.pkl)", start_dir, "Pickle model (*.pkl)"
        )
        if not file_path:
            return
        self.set_model_path(file_path)

    @staticmethod
    def _safe_float(value):
        try:
            if value is None:
                return None
            out = float(value)
            if np.isfinite(out):
                return out
        except Exception:
            pass
        return None

    def _meta_threshold_values(self):
        meta = self.model_metadata if isinstance(self.model_metadata, dict) else {}
        tshort = self._safe_float(meta.get("ternary_threshold_short"))
        tlong = self._safe_float(meta.get("ternary_threshold_long"))
        user = meta.get("user_settings")
        if isinstance(user, dict):
            if tshort is None:
                tshort = self._safe_float(user.get("ternary_threshold_short_eval"))
            if tlong is None:
                tlong = self._safe_float(user.get("ternary_threshold_long_eval"))
        return None, tshort, tlong

    def _refresh_threshold_preview(self) -> None:
        if not hasattr(self, "lbl_threshold_preview"):
            return
        _, tshort_model, tlong_model = self._meta_threshold_values()
        try:
            thr_short, thr_long = self._resolve_ternary_thresholds()
        except Exception:
            thr_short, thr_long = None, None

        def _fmt(value):
            if value is None:
                return "—"
            return f"{float(value):.3f}"

        src = getattr(self, "_last_ternary_threshold_source", "model")
        entry = self._safe_float(self.et_spin.value()) if hasattr(self, "et_spin") else None
        exit_thr = self._safe_float(self.ext_spin.value()) if hasattr(self, "ext_spin") else None
        self.lbl_threshold_preview.setText(
            f"T(model): short={_fmt(tshort_model)} long={_fmt(tlong_model)} | "
            f"T(active): short={_fmt(thr_short)} long={_fmt(thr_long)} src={src} | "
            f"Entry={_fmt(entry)} Exit={_fmt(exit_thr)}"
        )

    def _active_exit_policy(self) -> str:
        metadata = self.model_metadata if isinstance(self.model_metadata, dict) else {}
        return resolve_exit_policy_setting(metadata, default=DEFAULT_EXIT_POLICY)

    def on_open_data_clicked(self):
        # Dynamický a záložní start dir (po změně kořene projektu)
        try:
            project_root = Path(__file__).resolve().parents[1]
        except Exception:
            project_root = Path(os.getcwd())
        processed_dir_dyn = project_root / "data" / "processed"
        processed_dir_abs = Path(r"C:\Users\adamk\Můj disk\Trader\ibkr_trading_bot\data\processed")

        if processed_dir_dyn.is_dir():
            start_dir = str(processed_dir_dyn)
        elif processed_dir_abs.is_dir():
            start_dir = str(processed_dir_abs)
        else:
            start_dir = str(project_root)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Vyber CSV s historickými daty", start_dir, "CSV soubory (*.csv)"
        )
        if not file_path:
            return
        if not os.path.isfile(file_path):
            self._error("Soubor neexistuje.")
            return

        self.set_data_path(file_path)

    def set_data_path(self, file_path: str | Path, *, update_status: bool = True) -> None:
        normalized = str(Path(file_path).expanduser().resolve())
        self.data_path = normalized
        self.data_label.setText(f"Data (CSV): {normalized}")
        try:
            self._ui_settings.setValue(LAST_DATA_CSV_PATH_KEY, normalized)
            self._ui_settings.sync()
        except Exception:
            pass
        if update_status:
            self._set_status("Data připravena.")

    def _restore_last_data_path(self) -> None:
        try:
            saved = self._ui_settings.value(LAST_DATA_CSV_PATH_KEY, "")
        except Exception:
            saved = ""
        candidate = str(saved or "").strip()
        if not candidate:
            return
        path = Path(candidate).expanduser()
        if path.exists():
            self.set_data_path(path, update_status=False)

    def _on_eval_scope_changed(self, *_):
        mode = self._eval_scope_mode()
        if self.loaded_model is not None and self.model_path and self.data_path:
            self.on_evaluate_clicked()
        else:
            self._set_status(f"Rozsah evaluace nastaven: {mode}")

    def _eval_scope_mode(self) -> str:
        try:
            mode = self.scope_combo.currentData()
            if isinstance(mode, str) and mode in {"holdout", "full"}:
                return mode
        except Exception:
            pass
        return "holdout"

    def current_ranking_context(self) -> dict | None:
        if not self.data_path:
            return None
        return {
            "data_path": str(self.data_path),
            "fee_per_trade": float(self.cost_spin.value()),
            "entry_threshold": float(self.et_spin.value()),
            "exit_threshold": float(self.ext_spin.value()),
        }

    def _infer_holdout_bars_from_metadata(self, n_rows: int) -> int | None:
        n = int(max(0, n_rows))
        if n <= 0:
            return None
        meta = self.model_metadata if isinstance(self.model_metadata, dict) else {}

        try:
            nh = int(meta.get("n_holdout_bars", 0))
            if nh > 0:
                return int(min(n, nh))
        except Exception:
            pass

        hold_sel = meta.get("holdout_selection") if isinstance(meta, dict) else None
        if isinstance(hold_sel, dict):
            try:
                applied = int(hold_sel.get("applied_bars", 0))
                if applied > 0:
                    return int(min(n, applied))
            except Exception:
                pass
            try:
                pct = float(hold_sel.get("requested_pct"))
                if np.isfinite(pct) and pct > 0.0:
                    calc = int(round(float(n) * float(np.clip(pct, 0.0, 0.95))))
                    if calc > 0:
                        return int(min(n, calc))
            except Exception:
                pass
        return None

    @staticmethod
    def _tail_rows(obj, n_rows: int):
        if obj is None:
            return None
        n = int(max(0, n_rows))
        if isinstance(obj, pd.DataFrame):
            return obj.tail(n).reset_index(drop=True)
        if isinstance(obj, pd.Series):
            return obj.tail(n).reset_index(drop=True)
        arr = np.asarray(obj)
        if arr.ndim == 0:
            return arr
        return arr[-n:] if n < arr.shape[0] else arr

    def _apply_eval_scope(self, X, y_true, df_for_metrics):
        lengths = []
        try:
            lengths.append(int(len(X)))
        except Exception:
            pass
        if y_true is not None:
            try:
                lengths.append(int(len(y_true)))
            except Exception:
                pass
        if df_for_metrics is not None:
            try:
                lengths.append(int(len(df_for_metrics)))
            except Exception:
                pass
        if not lengths:
            raise ValueError("Nelze určit délku datasetu pro evaluaci.")

        n_base = int(max(0, min(lengths)))
        if n_base <= 0:
            raise ValueError("Dataset pro evaluaci je prázdný.")

        X_aligned = self._tail_rows(X, n_base)
        y_aligned = self._tail_rows(y_true, n_base) if y_true is not None else None
        df_aligned = self._tail_rows(df_for_metrics, n_base) if df_for_metrics is not None else None

        mode = self._eval_scope_mode()
        if mode == "holdout":
            n_hold = self._infer_holdout_bars_from_metadata(n_base)
            if n_hold is not None and n_hold > 0:
                n_eval = int(min(n_base, n_hold))
                X_eval = self._tail_rows(X_aligned, n_eval)
                y_eval = self._tail_rows(y_aligned, n_eval) if y_aligned is not None else None
                df_eval = self._tail_rows(df_aligned, n_eval) if df_aligned is not None else None
            else:
                n_eval = n_base
                X_eval, y_eval, df_eval = X_aligned, y_aligned, df_aligned
        else:
            n_eval = n_base
            X_eval, y_eval, df_eval = X_aligned, y_aligned, df_aligned

        scope_info = {
            "mode": mode,
            "applied_rows": int(n_eval),
            "total_rows": int(n_base),
        }
        self.eval_scope_info = scope_info
        return X_eval, y_eval, df_eval, scope_info

    @staticmethod
    def _pick_metric(metrics: dict, *keys):
        for key in keys:
            if key in metrics:
                value = metrics.get(key)
                if value is None:
                    continue
                if isinstance(value, float) and not np.isfinite(value):
                    continue
                return value
        return None

    def _format_key_metrics_line(self, metrics: dict) -> str:
        if not isinstance(metrics, dict) or not metrics:
            return "key=NA"

        def _f(value, digits=4):
            if value is None:
                return "NA"
            if isinstance(value, float):
                return f"{value:.{digits}f}"
            return str(value)

        profit_net = self._pick_metric(metrics, "profit_net", "profit_gross", "profit")
        pf = self._pick_metric(metrics, "pf", "profit_factor", "profit_factor_net")
        trades = self._pick_metric(metrics, "trades", "num_trades")
        long_trades = self._pick_metric(metrics, "num_trades_long", "long_trades", "long_net_trades")
        short_trades = self._pick_metric(metrics, "num_trades_short", "short_trades", "short_net_trades")

        return (
            "key: "
            f"profit_net={_f(profit_net, 2)} "
            f"pf={_f(pf, 4)} "
            f"trades={_f(trades, 0)} "
            f"long={_f(long_trades, 0)} "
            f"short={_f(short_trades, 0)}"
        )

    def _trade_pnls_for_plot(self, results: dict):
        trade_pnls_plot = None
        if isinstance(results, dict):
            trade_pnls_plot = results.get("trade_pnls_net") or results.get("trade_pnls")
        return trade_pnls_plot

    def _resolve_ternary_thresholds(self) -> tuple[float, float]:
        """
        Resolve active ternary short/long thresholds from model metadata.
        Tab 3 is strict ternary-only flow: no Decision-threshold fallback.
        """
        _, tshort, tlong = self._meta_threshold_values()
        if not isinstance(tshort, (int, float)) or not isinstance(tlong, (int, float)):
            raise ValueError(
                "Model neobsahuje platne ternarni prahy (ternary_threshold_short/long). "
                "Nahraj model natrenovany v nove pipeline."
            )
        self._last_ternary_threshold_source = "model"
        return float(tshort), float(tlong)

    def _feature_names_for_loaded_model(self) -> list[str] | None:
        try:
            names = getattr(self.loaded_model, "feature_names_in_", None)
            if names is not None:
                return [str(x) for x in list(names)]
        except Exception:
            pass
        try:
            steps = getattr(self.loaded_model, "steps", None)
            if steps:
                last = steps[-1][1]
                names = getattr(last, "feature_names_in_", None)
                if names is not None:
                    return [str(x) for x in list(names)]
        except Exception:
            pass
        return None

    def _align_X_for_loaded_model(self, X):
        if isinstance(X, pd.DataFrame):
            Xdf = X.copy()
        else:
            Xdf = pd.DataFrame(X)

        names = self._feature_names_for_loaded_model()
        if names:
            for c in names:
                if c not in Xdf.columns:
                    Xdf[c] = 0.0
            Xdf = Xdf.reindex(columns=names, fill_value=0.0)

        med = Xdf.median(numeric_only=True)
        Xdf = Xdf.fillna(med).fillna(0.0)
        for c in Xdf.columns:
            if not pd.api.types.is_bool_dtype(Xdf[c]):
                Xdf[c] = Xdf[c].astype("float32", copy=False)
        return Xdf

    # ---------------- Helpery: confidence / threshold ----------------
    def _get_model_scores(self, X):
        """Vrací (proba, classes, decision) podle možností modelu."""
        proba = None
        classes = None
        decision = None
        X_pred = self._align_X_for_loaded_model(X)
        try:
            if hasattr(self.loaded_model, "predict_proba"):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"X does not have valid feature names, but .* was fitted with feature names",
                        category=UserWarning,
                    )
                    proba = self.loaded_model.predict_proba(X_pred)
                classes = getattr(self.loaded_model, "classes_", None)
        except Exception:
            proba = None
        try:
            if hasattr(self.loaded_model, "decision_function"):
                decision = self.loaded_model.decision_function(X_pred)
        except Exception:
            decision = None
        return proba, classes, decision

    def _confidence_from_scores(self, proba, decision):
        """Confidence = max class probability, nebo sigmoida(abs(margin)). Když nic není, samé 1."""
        if proba is not None:
            proba = np.asarray(proba, dtype=float)
            if proba.ndim == 2:
                return np.max(proba, axis=1).astype(float)
            return proba.astype(float)
        if decision is not None:
            z = np.asarray(decision, dtype=float)
            if z.ndim == 2:
                z = np.max(np.abs(z), axis=1)
            return 1.0 / (1.0 + np.exp(-z))
        self._set_status("Model nepodporuje predict_proba/decision_function – confidence filtr se neaplikuje.")
        return np.ones(len(self.y_pred_raw), dtype=float)

    def _apply_confidence_threshold(self, raw_pred, confidence, threshold):
        """Pod prahem confidence nastaví predikci na 0 (flat)."""
        return apply_confidence_entry_threshold(raw_pred, confidence, threshold)

    def _apply_exit_threshold(self, y_pred: np.ndarray, confidence: np.ndarray, exit_thr: float) -> np.ndarray:
        """
        Aplikuj exit threshold: pokud máme otevřenou pozici (LONG/SHORT) 
        a confidence klesne pod exit_thr, zavři ji (vrátí FLAT=0).
        """
        return apply_exit_confidence_threshold(y_pred, confidence, exit_thr)

    # --- NEW: normalizace predikcí po prahování na {-1,0,+1} ---
    def _normalize_pred(self, arr):
        """
        Převede libovolné predikce na {-1, 0, +1}.
        Podporuje čísla, booly i texty ('long'/'short'/...).
        """
        return normalize_signal_array(arr)

    # ---------------- Helpery: PnL a breakdown ----------------
    def _build_positions(self, y_pred):
        arr = np.asarray(y_pred).astype(float)
        pos = np.zeros_like(arr, dtype=int)
        pos[arr > 0] = 1
        pos[arr < 0] = -1
        return pos

    def _compute_trade_pnls_from_signals(self, fee_per_trade: float = None):
        if self.close_series is None or self.y_pred_used is None:
            return []

        close = np.asarray(self.close_series, dtype=float)
        sig = self._build_positions(self.y_pred_used)
        n = min(len(close), len(sig))
        if n < 2:
            return []

        if fee_per_trade is None and hasattr(self, "cost_spin"):
            fee_per_trade = float(self.cost_spin.value() or 0.0)
        fee = float(fee_per_trade or 0.0)
        replay = replay_signals_over_market_data(sig[:n], close[:n], force_close=True)
        return [float(p) - fee for p in (replay.get("trade_pnls") or [])]

    # ---------------- Helpery: grafy ----------------
    def _draw_equity_chart(self, results: dict):
        ax = self.canvas_equity.ax
        ax.clear()
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_title("Equity křivka")

        # 1) Načti křivky z výsledků ze společného executoru
        eq = results.get("equity_curve")
        eq_net = results.get("equity_curve_net")

        # 2) Pokud chybí, fallback z PnL per-trade
        if eq is None and results.get("trade_pnls") is not None:
            eq = np.cumsum(np.asarray(results["trade_pnls"], dtype=float))
        if eq_net is None and results.get("trade_pnls_net") is not None:
            eq_net = np.cumsum(np.asarray(results["trade_pnls_net"], dtype=float))

        def _len(x):
            try:
                return len(x)
            except Exception:
                return None

        # 3) vykreslení
        plotted = False
        if eq is not None and _len(eq):
            ax.plot(np.arange(_len(eq)), np.asarray(eq, dtype=float), label="Equity (gross)")
            plotted = True
        if eq_net is not None and _len(eq_net):
            ax.plot(np.arange(_len(eq_net)), np.asarray(eq_net, dtype=float), label="Equity (net)")
            plotted = True

        # baseline (buy&hold z close)
        if self.close_series is not None and len(self.close_series) > 0:
            arr = self.close_series.to_numpy(dtype=float)
            base = arr - arr[0]
            ax.plot(np.arange(len(base)), base, linestyle="--", alpha=0.6, label="Buy & Hold")
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "Žádná data pro equity", ha="center", va="center", transform=ax.transAxes)

        ax.legend(loc="best")
        self.canvas_equity.draw_idle()

    def _draw_histogram(self, trade_pnls):
        ax = self.canvas_hist.ax
        ax.clear()
        ax.set_title("Distribuce PnL obchodů")
        ax.grid(True, linestyle=":", alpha=0.4)

        if trade_pnls is None or len(trade_pnls) == 0:
            ax.text(0.5, 0.5, "Žádné obchody", ha="center", va="center", transform=ax.transAxes)
            self.canvas_hist.draw_idle()
            return

        vals = np.asarray(trade_pnls, dtype=float)
        ax.hist(vals, bins=50, alpha=0.85)
        ax.axvline(0.0, linestyle="--")
        mu = float(np.nanmean(vals)) if vals.size else 0.0
        med = float(np.nanmedian(vals)) if vals.size else 0.0
        ax.axvline(mu, linestyle=":")
        ax.axvline(med, linestyle=":")
        ax.legend(["0", f"mean={mu:.3f}", f"median={med:.3f}"])
        self.canvas_hist.draw_idle()

    def _draw_rolling_chart(self, trade_pnls, window=20):
        ax = self.canvas_rolling.ax
        ax.clear()
        ax.grid(True, linestyle=":", alpha=0.4)
        mode = self.roll_combo.currentText()
        ax.set_title(mode)

        if trade_pnls is None or len(trade_pnls) == 0:
            ax.text(0.5, 0.5, "Žádné obchody", ha="center", va="center", transform=ax.transAxes)
            self.canvas_rolling.draw_idle()
            return

        s = pd.Series(trade_pnls, dtype=float)
        if mode.startswith("Rolling profit"):
            r = s.rolling(window, min_periods=1).mean()
            ax.plot(r.index, r.values, label=f"Rolling mean ({window})")
        elif mode.startswith("Rolling Sharpe"):
            mu = s.rolling(window, min_periods=1).mean()
            sd = s.rolling(window, min_periods=2).std(ddof=1)
            sharpe = mu / sd.replace(0, np.nan)
            sharpe = sharpe.fillna(0.0)
            ax.plot(sharpe.index, sharpe.values, label=f"Sharpe ({window})")
        else:  # Rolling max DD (na kum. equity)
            eq = s.cumsum()
            def _roll_max_dd(x: pd.Series) -> float:
                peak = -np.inf
                max_dd = 0.0
                for v in x:
                    if v > peak:
                        peak = v
                    dd = peak - v
                    if dd > max_dd:
                        max_dd = dd
                return float(max_dd)
            rdd = eq.rolling(window, min_periods=1).apply(_roll_max_dd, raw=False)
            ax.plot(rdd.index, rdd.values, label=f"Max DD ({window})")

        ax.legend(loc="best")
        self.canvas_rolling.draw_idle()

    # ---------------- Helpery: equity/baseline, DD, close ----------------
    @staticmethod
    def _max_drawdown(equity):
        peak = -np.inf
        max_dd = 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
        return float(max_dd)

    # ---------------- Metriky a tabulka ----------------
    def _build_metric_descriptions(self):
        return {
            # klasické
            "accuracy": "Podíl správných predikcí napříč všemi třídami; u nevyvážených tříd sleduj hlavně i F1.",
            "accuracy_binary": "Binární accuracy po zjednodušení na signál vs. bez signálu.",
            "f1": "Vyváženost mezi precision a recall – jak dobře model predikuje obchodní signály.",
            "f1_binary": "Binární F1 po zjednodušení na signál vs. bez signálu.",
            "f1_macro_3": "Průměrné F1 přes SHORT, HOLD a LONG stejnou vahou.",
            "f1_micro_3": "Celkové F1 přes všechny 3 třídy dohromady.",
            "f1_weighted_3": "F1 přes 3 třídy vážené podle četnosti jednotlivých tříd.",
            "precision": "Kolik z predikovaných obchodů bylo skutečně správně.",
            "precision_binary": "Jak čisté jsou binární signály; kolik aktivních signálů bylo správně.",
            "recall": "Kolik správných obchodů model dokázal najít.",
            "recall_binary": "Kolik skutečných binárních signálů model zachytil.",
            "profit": "Celkový zisk/ztráta (hrubý).",
            "profit_gross": "Celkový hrubý zisk před náklady.",
            "sharpe": "Sharpe bez anualizace – poměr výnosu k volatilitě na úrovni použité řady.",
            "sharpe_ann": "Anualizovaný Sharpe pro snazší srovnání modelů mezi obdobími.",
            "sharpe_ratio": "Zisk očištěný o volatilitu – vyšší = lepší poměr zisku k riziku.",
            "max_dd": "Kratší alias pro max drawdown – největší pokles equity od maxima.",
            "max_drawdown": "Největší pokles equity od maxima (riziko velké ztráty).",
            "winrate": "Procento ziskových obchodů.",
            "pf": "Kratší alias pro profit factor.",
            "profit_factor": "Poměr hrubého zisku ku hrubé ztrátě (>1 = zisková strategie).",
            "signal_stability": "Jak moc se model přepíná mezi signály – vyšší = stabilnější.",
            "signals": "Počet barů, kde model generoval aktivní obchodní signál.",
            "num_trades": "Počet provedených obchodů.",
            "num_trades_long": "Počet long obchodů.",
            "num_trades_short": "Počet short obchodů.",
            "avg_pnl_trade": "Průměrný hrubý PnL na obchod.",
            "median_pnl_trade": "Medián hrubého PnL na obchod; méně citlivý na extrémy.",
            "avg_profit_per_trade": "Průměrný zisk na obchod (hrubý).",
            "median_profit_per_trade": "Medián zisku na obchod (hrubý).",
            "var_95": "Value at Risk 95 % (hrubé) – odhad ztráty v 95 % případů.",
            "cvar_95": "Conditional VaR 95 % (hrubé) – průměr ztrát v nejhorších 5 %.",
            "tp": "True positives – správně zachycené aktivní signály.",
            "fp": "False positives – signály, které model otevřel chybně.",
            "tn": "True negatives – správně rozpoznané situace bez signálu.",
            "fn": "False negatives – zmeškané skutečné signály.",
            # netto
            "profit_net": "Celkový zisk po nákladech.",
            "sharpe_net": "Sharpe po nákladech bez anualizace.",
            "sharpe_net_ann": "Anualizovaný Sharpe po nákladech.",
            "sharpe_ratio_net": "Sharpe po nákladech.",
            "max_drawdown_net": "Max drawdown po nákladech.",
            "winrate_net": "Winrate po nákladech.",
            "profit_factor_net": "Profit factor po nákladech.",
            "num_trades_net": "Počet obchodů (netto = shodný, jen pro přehled).",
            "avg_profit_per_trade_net": "Průměrný zisk/obchod po nákladech.",
            "median_profit_per_trade_net": "Medián zisk/obchod po nákladech.",
            "var_95_net": "VaR 95 % po nákladech.",
            "cvar_95_net": "CVaR 95 % po nákladech.",
            # breakdown
            "long_trades": "Počet long obchodů (hrubě).",
            "short_trades": "Počet short obchodů (hrubě).",
            "long_profit": "Součet zisků longů (hrubě).",
            "short_profit": "Součet zisků shortů (hrubě).",
            "long_winrate": "Winrate long (hrubě).",
            "short_winrate": "Winrate short (hrubě).",
            "long_profit_factor": "Profit factor long (hrubě).",
            "short_profit_factor": "Profit factor short (hrubě).",
            "long_net_trades": "Počet long obchodů (netto).",
            "short_net_trades": "Počet short obchodů (netto).",
            "long_net_profit": "Součet zisků longů po nákladech.",
            "short_net_profit": "Součet zisků shortů po nákladech.",
            "long_net_winrate": "Winrate long po nákladech.",
            "short_net_winrate": "Winrate short po nákladech.",
            "long_net_profit_factor": "Profit factor long po nákladech.",
            "short_net_profit_factor": "Profit factor short po nákladech.",
            # další diagnostika
            "n_signals_holdout": "Počet aktivních signálů jen na holdout části datasetu.",
        }

    def _populate_metrics_table(self, metrics: dict):
        order = [
            "f1", "precision", "recall",
            "profit", "sharpe_ratio", "max_drawdown",
            "winrate", "profit_factor", "signal_stability",
            "num_trades", "avg_profit_per_trade", "median_profit_per_trade",
            "var_95", "cvar_95",
            # netto
            "profit_net", "sharpe_ratio_net", "max_drawdown_net",
            "winrate_net", "profit_factor_net",
            "avg_profit_per_trade_net", "median_profit_per_trade_net",
            "var_95_net", "cvar_95_net",
            # breakdown gross
            "long_trades", "short_trades",
            "long_profit", "short_profit",
            "long_winrate", "short_winrate",
            "long_profit_factor", "short_profit_factor",
            # breakdown net
            "long_net_trades", "short_net_trades",
            "long_net_profit", "short_net_profit",
            "long_net_winrate", "short_net_winrate",
            "long_net_profit_factor", "short_net_profit_factor",
        ]

        self.metrics_table.setRowCount(0)

        def _fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        for key in order:
            if key in metrics:
                self._add_metric_row(self._pretty_metric_name(key), _fmt(metrics[key]),
                                     self.metric_descriptions.get(key, ""))

        # přidej i další scalar hodnoty, které nejsou v seznamu order
        for key, value in metrics.items():
            if key in order:
                continue
            if isinstance(value, (list, tuple, np.ndarray, pd.Series, dict)):
                continue
            self._add_metric_row(self._pretty_metric_name(key), _fmt(value),
                                 self.metric_descriptions.get(key, ""))

        self.metrics_table.resizeColumnsToContents()

    def _add_metric_row(self, name: str, value: str, desc: str):
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(value))
        item_desc = QTableWidgetItem(desc)
        item_desc.setFlags(item_desc.flags() & ~Qt.ItemIsEditable)
        self.metrics_table.setItem(row, 2, item_desc)

    def _copy_selected_metrics(self):
        """Copy selected metric rows (or all rows when nothing is selected) to clipboard as TSV."""
        try:
            row_count = int(self.metrics_table.rowCount())
        except Exception:
            row_count = 0
        if row_count <= 0:
            self._set_status("Metriky nejsou k dispozici pro kopirovani.")
            return

        selected_rows = sorted({idx.row() for idx in self.metrics_table.selectedIndexes()})
        rows = selected_rows if selected_rows else list(range(row_count))

        lines = ["Metrika\tHodnota\tVyznam"]
        for r in rows:
            vals = []
            for c in range(3):
                it = self.metrics_table.item(int(r), int(c))
                vals.append(it.text() if it is not None else "")
            lines.append("\t".join(vals))

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        mode = "vybrane" if selected_rows else "vsechny"
        self._set_status(f"Zkopirovano {len(rows)} radku metrik ({mode}).")

    def _copy_key_metrics(self):
        """Copy compact key metrics summary to clipboard."""
        metrics = self.last_metrics if isinstance(self.last_metrics, dict) else {}
        if not metrics:
            self._set_status("Nejprve proved vyhodnoceni modelu.")
            return

        def _pick(*keys):
            for k in keys:
                if k in metrics:
                    v = metrics.get(k)
                    if v is None:
                        continue
                    if isinstance(v, float) and not np.isfinite(v):
                        continue
                    return v
            return None

        def _fmt(v):
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        scope = self.eval_scope_info if isinstance(self.eval_scope_info, dict) else {}
        scope_mode = scope.get("mode", "")
        scope_rows = scope.get("applied_rows", "")
        scope_total = scope.get("total_rows", "")

        lines = [
            "Metrika\tHodnota",
            f"scope\t{scope_mode} ({scope_rows}/{scope_total})",
            f"profit_net\t{_fmt(_pick('profit_net', 'profit_gross', 'profit'))}",
            f"sharpe\t{_fmt(_pick('sharpe_ann', 'sharpe_net_ann', 'sharpe', 'sharpe_net', 'sharpe_ratio'))}",
            f"profit_factor\t{_fmt(_pick('pf', 'profit_factor', 'profit_factor_net'))}",
            f"max_drawdown\t{_fmt(_pick('max_dd', 'max_drawdown', 'max_drawdown_trade_net', 'max_drawdown_trade_gross'))}",
            f"trades\t{_fmt(_pick('trades', 'num_trades'))}",
            f"num_trades_long\t{_fmt(_pick('num_trades_long', 'long_trades', 'long_net_trades'))}",
            f"num_trades_short\t{_fmt(_pick('num_trades_short', 'short_trades', 'short_net_trades'))}",
            f"winrate\t{_fmt(_pick('winrate', 'winrate_net'))}",
            f"f1_macro_3\t{_fmt(_pick('f1_macro_3', 'f1'))}",
        ]
        QApplication.clipboard().setText("\n".join(lines))
        self._set_status("Zkopirovany klicove metriky do schranky.")

    @staticmethod
    def _pretty_metric_name(key: str) -> str:
        mapping = {
            "f1": "F1 score", "precision": "Precision", "recall": "Recall",
            "profit": "Profit", "sharpe_ratio": "Sharpe ratio", "max_drawdown": "Max drawdown",
            "winrate": "Winrate", "profit_factor": "Profit factor", "signal_stability": "Signal stability",
            "num_trades": "Počet obchodů", "avg_profit_per_trade": "Průměrný zisk/obchod",
            "median_profit_per_trade": "Medián zisk/obchod", "var_95": "VaR (95 %)", "cvar_95": "CVaR (95 %)",
            "profit_net": "Profit (netto)", "sharpe_ratio_net": "Sharpe (netto)",
            "max_drawdown_net": "Max drawdown (netto)", "winrate_net": "Winrate (netto)",
            "profit_factor_net": "Profit factor (netto)",
            "avg_profit_per_trade_net": "Průměrný zisk/obchod (netto)",
            "median_profit_per_trade_net": "Medián zisk/obchod (netto)",
            "var_95_net": "VaR 95 % netto", "cvar_95_net": "CVaR 95 % netto",
            "long_trades": "Long – počet", "short_trades": "Short – počet",
            "long_profit": "Long – profit", "short_profit": "Short – profit",
            "long_winrate": "Long – winrate", "short_winrate": "Short – winrate",
            "long_profit_factor": "Long – PF", "short_profit_factor": "Short – PF",
            "long_net_trades": "Long – počet (netto)", "short_net_trades": "Short – počet (netto)",
            "long_net_profit": "Long – profit (netto)", "short_net_profit": "Short – profit (netto)",
            "long_net_winrate": "Long – winrate (netto)", "short_net_winrate": "Short – winrate (netto)",
            "long_net_profit_factor": "Long – PF (netto)", "short_net_profit_factor": "Short – PF (netto)",
        }
        return mapping.get(key, key)

    # ---------------- Status & dialogy ----------------
    def _set_status(self, text: str):
        self.status_label.setText(f"Status: {text}")

    def _warn(self, msg: str):
        QMessageBox.warning(self, "Upozornění", msg)
        self._set_status(msg)

    def _error(self, msg: str):
        QMessageBox.critical(self, "Chyba", msg)
        self._set_status(msg)

    def _export_trades_csv(self):
        import os
        import time
        from pathlib import Path

        from PySide6.QtWidgets import QMessageBox

        df = getattr(self, "trades_df", None)
        if df is None or df.empty:
            QMessageBox.information(self, "Export", "Nejsou žádné obchody k exportu. Nejprve klikni na „Vyhodnotit“.")
            return

        out_dir = Path(__file__).resolve().parents[2] / "model_outputs" / "evals"
        out_dir.mkdir(parents=True, exist_ok=True)

        model_name = os.path.splitext(os.path.basename(getattr(self, "model_path", "model.pkl")))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{model_name}__trades_{ts}.csv"

        # nebo se zeptej na umístění:
        # out_path_str, _ = QFileDialog.getSaveFileName(self, "Uložit CSV s obchody", out_path.as_posix(), "CSV (*.csv)")
        # if not out_path_str: return
        # out_path = Path(out_path_str)

        df.to_csv(out_path.as_posix(), index=False, encoding="utf-8")
        QMessageBox.information(self, "Export", f"Uloženo: {out_path}")

    def _extract_trades_df(self, results: dict):
        """
        Vrátí DataFrame s obchody ze slovníku results.
        Podporuje:
          - 'trades_df' (DataFrame)
          - 'trades' / 'trades_list' (list[dict])
          - fallback: jen PnL per trade → sloupec 'pnl'
        """
        try:
            df = results.get("trades_df")
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.copy()

            lst = results.get("trades") or results.get("trades_list")
            if isinstance(lst, (list, tuple)) and lst and isinstance(lst[0], dict):
                out = pd.DataFrame(lst)
                preferred = [
                    "entry_time", "exit_time", "direction", "entry_price", "exit_price", "pnl", "pnl_net", "exit_reason",
                ]
                cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
                return out[cols]

            pnls = results.get("trade_pnls_net") or results.get("trade_pnls")
            if pnls is not None:
                arr = np.asarray(pnls, dtype=float).reshape(-1)
                return pd.DataFrame({"pnl": arr})
        except Exception:
            pass
        return None

    def _show_overfitting_diagnostics(self):
        """Zobrazí diagnostiku overfittingu porovnáním train vs holdout metrik."""
        if not isinstance(self.model_metadata, dict):
            self.overfitting_console.setPlainText("(Metadata modelu nejsou dostupná)")
            return

        train_metrics = self.model_metadata.get("metrics_train", {})
        holdout_metrics = self.model_metadata.get("metrics_holdout", {})
        
        if not train_metrics or not holdout_metrics:
            text = "(Tréninkové a holdout metriky nejsou k dispozici)\n"
            if train_metrics:
                text += f"Train metrics: {train_metrics}\n"
            if holdout_metrics:
                text += f"Holdout metrics: {holdout_metrics}"
            self.overfitting_console.setPlainText(text)
            return

        # Extrahuj accuracy metriky
        train_acc = train_metrics.get("accuracy")
        holdout_acc = holdout_metrics.get("accuracy")
        
        if train_acc is None or holdout_acc is None:
            self.overfitting_console.setPlainText(
                f"Train: {train_metrics}\nHoldout: {holdout_metrics}"
            )
            return

        train_acc = float(train_acc)
        holdout_acc = float(holdout_acc)
        diff = train_acc - holdout_acc

        # F1 metriky
        train_f1 = train_metrics.get("f1")
        holdout_f1 = holdout_metrics.get("f1")
        
        # Počet signálů
        train_signals = train_metrics.get("n_signals_train", 0)
        holdout_signals = holdout_metrics.get("n_signals_holdout", 0)

        # Sestav diagnostickou zprávu
        lines = [
            "╔═══════════════════════════════════════════════════════╗",
            "║         DIAGNOSTIKA OVERFITTINGU                      ║",
            "╠═══════════════════════════════════════════════════════╣",
            f"║ Train Accuracy:    {train_acc:7.4f}                         ║",
            f"║ Holdout Accuracy:  {holdout_acc:7.4f}                         ║",
            f"║ ─────────────────────────────────────────────────────║",
            f"║ Rozdíl (Train-Out): {diff:+7.4f}                       ║",
        ]
        
        if train_f1 is not None and holdout_f1 is not None:
            train_f1 = float(train_f1)
            holdout_f1 = float(holdout_f1)
            diff_f1 = train_f1 - holdout_f1
            lines.extend([
                f"║ Train F1:          {train_f1:7.4f}                         ║",
                f"║ Holdout F1:        {holdout_f1:7.4f}                         ║",
                f"║ Rozdíl F1:         {diff_f1:+7.4f}                       ║",
            ])

        lines.extend([
            f"║ Train Signals:     {train_signals:7d}                       ║",
            f"║ Holdout Signals:   {holdout_signals:7d}                       ║",
            "╠═══════════════════════════════════════════════════════╣",
        ])

        # Diagnóza – kontrola F1, accuracy a overfittingu
        # 1. Nejdřív zkontroluj F1 score (nutné pro použitelný model)
        if holdout_f1 is not None and holdout_f1 < 0.05:
            lines.append("║ ❌ NEPOUŽITELNÝ MODEL: F1 score téměř 0             ║")
            lines.append("║    → Model nepredikuje LONG/SHORT signály!          ║")
        elif holdout_acc > 0.98 and (holdout_f1 is None or holdout_f1 < 0.1):
            lines.append("║ ❌ PODEZŘELÝ: Vysoká accuracy ale nízké F1          ║")
            lines.append("║    → Model predikuje jen majoritní třídu (NEUTRAL)  ║")
        # 2. Pak zkontroluj overfitting podle rozdílu
        elif abs(diff) < 0.05:
            lines.append("║ ✅ DOBRÝ MODEL: Minimální přefitting                 ║")
        elif abs(diff) < 0.10:
            lines.append("║ ⚠️  MÍRNÝ OVERFITTING: Rozdíl < 10%                ║")
        elif abs(diff) < 0.15:
            lines.append("║ ⚠️  STŘEDOVÝ OVERFITTING: Rozdíl 10-15%            ║")
        else:
            lines.append("║ ❌ VÁŽNĚJŠÍ OVERFITTING: Rozdíl > 15%              ║")

        lines.append("╚═══════════════════════════════════════════════════════╝")

        self.overfitting_console.setPlainText("\n".join(lines))

    def _on_save_model_settings(self) -> None:
        """Uloží aktuální nastavení modelu do metadat."""
        if not self.model_path:
            QMessageBox.warning(self, "Chyba", "Nejdřív načti model!")
            return
        
        try:
            import json as jsonlib
            from pathlib import Path
            
            # Najdi meta.json
            model_path = Path(self.model_path)
            meta_path = model_path.with_name(model_path.stem + "_meta.json")
            
            # Načti meta (nebo vytvoř nový)
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as fh:
                    metadata = jsonlib.load(fh)
            else:
                metadata = {}
            
            # Přidej user_settings
            ts_eval, tl_eval = self._resolve_ternary_thresholds()
            metadata["user_settings"] = {
                "ternary_threshold_short_eval": float(ts_eval),
                "ternary_threshold_long_eval": float(tl_eval),
                "entry_threshold": float(self.et_spin.value()),
                "exit_threshold": float(self.ext_spin.value()),
                "use_and_ensemble": False,
                "use_ma_only": False,
                "use_macd_filter": False,
                "updated_at": str(pd.Timestamp.now(tz="UTC")),
            }
            
            # Ulož metadat zpět
            with meta_path.open("w", encoding="utf-8") as fh:
                jsonlib.dump(metadata, fh, indent=2, default=str)
            
            self._set_status(f"✅ Nastavení uloženo: {meta_path.name}")
            QMessageBox.information(
                self, "✅ Hotovo",
                f"Nastavení modelu úspěšně uloženo!\n\n"
                f"T-short eval: {ts_eval}\n"
                f"T-long eval: {tl_eval}\n"
                f"T-source: {self._last_ternary_threshold_source}\n"
                f"Entry Threshold: {self.et_spin.value()}\n"
                f"Exit Threshold: {self.ext_spin.value()}\n"
                f"AND Ensemble: False\n"
                f"MA-Only: False\n"
                f"MACD Filter: False\n\n"
                f"Soubor: {meta_path.name}"
            )
        except Exception as e:
            self._error(f"Chyba při uložení nastavení:\n{e}")

    def _track_retired_worker(self, worker: TaskWorker) -> None:
        if worker in self._retired_workers:
            return
        self._retired_workers.append(worker)

        def _cleanup_retired_worker() -> None:
            try:
                self._retired_workers.remove(worker)
            except ValueError:
                pass
            try:
                worker.deleteLater()
            except Exception:
                pass

        worker.finished.connect(_cleanup_retired_worker)

    def _stop_worker_attr(self, attr_name: str, *, wait_ms: int = 250, allow_background: bool = True) -> bool:
        worker = getattr(self, attr_name, None)
        if worker is None:
            return True
        setattr(self, attr_name, None)
        for signal_name in ("progress_text", "result", "error", "finished"):
            try:
                getattr(worker, signal_name).disconnect()
            except Exception:
                pass
        try:
            worker.stop()
        except Exception:
            pass
        if worker.isRunning() and not worker.wait(wait_ms):
            if allow_background:
                self._track_retired_worker(worker)
                return False
            setattr(self, attr_name, worker)
            return False
        try:
            worker.deleteLater()
        except Exception:
            pass
        return True

    def shutdown(self) -> bool:
        self._pending_auto_threshold_search = False
        self._pending_auto_threshold_dialog = None
        try:
            self._params_timer.stop()
        except Exception:
            pass
        ok = True
        ok = self._stop_worker_attr("_auto_threshold_worker", wait_ms=2000, allow_background=False) and ok
        ok = self._stop_worker_attr("_params_worker", wait_ms=2000, allow_background=False) and ok
        ok = self._stop_worker_attr("_eval_worker", wait_ms=2000, allow_background=False) and ok
        return ok

    def _apply_results_to_ui(self, results: dict) -> None:
        self.last_metrics = results
        self.trades_df = self._extract_trades_df(results)
        self.btn_export_trades.setEnabled(
            isinstance(self.trades_df, pd.DataFrame) and not self.trades_df.empty
        )
        self._populate_metrics_table(results)
        self._draw_equity_chart(results)
        trade_pnls_plot = self._trade_pnls_for_plot(results)
        self._draw_histogram(trade_pnls_plot)
        self._draw_rolling_chart(trade_pnls_plot)

    def on_evaluate_clicked(self):
        if self.loaded_model is None or self.model_path is None:
            self._warn("Nejprve vyber model (.pkl).")
            return
        if self.data_path is None:
            self._warn("Nejprve vyber CSV s historickymi daty.")
            return
        self._start_evaluation_worker()

    def on_params_changed(self, *_):
        if self.y_pred_raw is None or self.confidence_arr is None or self.df_current is None:
            return
        self._params_timer.start()

    def _start_evaluation_worker(self) -> None:
        self._pending_auto_threshold_dialog = None
        self._params_timer.stop()
        self._stop_worker_attr("_params_worker")
        self._stop_worker_attr("_auto_threshold_worker")
        self._stop_worker_attr("_eval_worker")
        self._eval_request_id += 1
        req_id = self._eval_request_id
        self._set_status("Vyhodnoceni: nacitam CSV...")

        worker = TaskWorker(
            self._task_run_full_evaluation,
            model=self.loaded_model,
            metadata=self.model_metadata,
            data_path=self.data_path,
            scope_mode=self._eval_scope_mode(),
            fee_per_trade=float(self.cost_spin.value()),
            entry_threshold=float(self.et_spin.value()),
            exit_threshold=float(self.ext_spin.value()),
            exit_policy=self._active_exit_policy(),
        )
        self._eval_worker = worker
        worker.progress_text.connect(self._set_status)
        worker.result.connect(lambda payload, rid=req_id: self._on_evaluation_result(rid, payload))
        worker.error.connect(lambda msg, rid=req_id: self._on_evaluation_error(rid, msg))
        worker.finished.connect(lambda rid=req_id: self._on_evaluation_finished(rid))
        worker.start()

    @staticmethod
    def _task_run_full_evaluation(
        *,
        model,
        metadata,
        data_path: str,
        scope_mode: str,
        fee_per_trade: float,
        entry_threshold: float,
        exit_threshold: float,
        exit_policy: str | None = None,
        progress_cb=None,
    ) -> EvaluationPayload:
        return model_eval_runtime.run_model_evaluation(
            model=model,
            metadata=metadata,
            data_path=data_path,
            scope_mode=scope_mode,
            fee_per_trade=float(fee_per_trade),
            entry_threshold=float(entry_threshold),
            exit_threshold=float(exit_threshold),
            exit_policy=exit_policy,
            progress_cb=progress_cb,
        )

    def _on_evaluation_result(self, req_id: int, payload: EvaluationPayload) -> None:
        if req_id != self._eval_request_id or payload is None:
            return
        self.X_current = payload.X_current
        self.y_true_current = payload.y_true_current
        self.df_current = payload.df_current
        self.close_series = payload.close_series
        self.confidence_arr = payload.confidence_arr
        self.y_pred_raw = payload.y_pred_raw
        self.y_pred_used = payload.y_pred_used
        self.eval_scope_info = payload.scope_info
        self._last_ternary_threshold_source = payload.threshold_source
        self._apply_results_to_ui(payload.results)

        key_line = self._format_key_metrics_line(payload.results)
        scope_info = payload.scope_info if isinstance(payload.scope_info, dict) else {}
        self._set_status(
            f"Hotovo. Vyhodnoceni dokonceno (naklady/obchod {self.cost_spin.value():.3f}, "
            f"T-src={payload.threshold_source}, "
            f"scope={scope_info.get('mode')} {scope_info.get('applied_rows')}/{scope_info.get('total_rows')}, "
            f"{key_line})."
        )

        if self._pending_auto_threshold_search:
            self._pending_auto_threshold_search = False
            self._start_auto_threshold_worker()
            return

        if not np.isclose(float(self.et_spin.value()), payload.entry_threshold) or not np.isclose(float(self.ext_spin.value()), payload.exit_threshold):
            self._params_timer.start()

    def _on_evaluation_error(self, req_id: int, msg: str) -> None:
        if req_id != self._eval_request_id:
            return
        self._pending_auto_threshold_search = False
        self._error(msg)

    def _on_evaluation_finished(self, req_id: int) -> None:
        if req_id == self._eval_request_id:
            self._eval_worker = None

    @staticmethod
    def _task_recalculate_metrics(
        *,
        y_pred_raw,
        confidence_arr,
        y_true_current,
        df_current,
        fee_per_trade: float,
        entry_threshold: float,
        exit_threshold: float,
        exit_policy: str = DEFAULT_EXIT_POLICY,
        progress_cb=None,
    ) -> tuple[np.ndarray, dict]:
        return model_eval_runtime.recalculate_metrics_from_predictions(
            y_pred_raw=y_pred_raw,
            confidence_arr=confidence_arr,
            y_true_current=y_true_current,
            df_current=df_current,
            fee_per_trade=float(fee_per_trade),
            entry_threshold=float(entry_threshold),
            exit_threshold=float(exit_threshold),
            exit_policy=exit_policy,
            progress_cb=progress_cb,
        )

    def _run_params_recalc(self) -> None:
        if self.y_pred_raw is None or self.confidence_arr is None or self.df_current is None:
            return
        self._stop_worker_attr("_params_worker")
        self._params_request_id += 1
        req_id = self._params_request_id

        worker = TaskWorker(
            self._task_recalculate_metrics,
            y_pred_raw=np.asarray(self.y_pred_raw),
            confidence_arr=np.asarray(self.confidence_arr),
            y_true_current=np.asarray(self.y_true_current),
            df_current=self.df_current,
            fee_per_trade=float(self.cost_spin.value()),
            entry_threshold=float(self.et_spin.value()),
            exit_threshold=float(self.ext_spin.value()),
            exit_policy=self._active_exit_policy(),
        )
        self._params_worker = worker
        worker.progress_text.connect(self._set_status)
        worker.result.connect(lambda result, rid=req_id: self._on_params_result(rid, result))
        worker.error.connect(lambda msg, rid=req_id: self._on_params_error(rid, msg))
        worker.finished.connect(lambda rid=req_id: self._on_params_finished(rid))
        worker.start()

    def _on_params_result(self, req_id: int, result: tuple[np.ndarray, dict]) -> None:
        if req_id != self._params_request_id or result is None:
            return
        y_pred_used, results = result
        self.y_pred_used = np.asarray(y_pred_used)
        self._apply_results_to_ui(results)

        try:
            thr_short, thr_long = self._resolve_ternary_thresholds()
        except Exception:
            thr_short, thr_long = float("nan"), float("nan")
        entry_threshold = float(self.et_spin.value())
        exit_threshold = float(self.ext_spin.value())
        scope_info = self.eval_scope_info if isinstance(self.eval_scope_info, dict) else {}
        key_line = self._format_key_metrics_line(results)
        self._set_status(
            f"Prepociteno (T-short={thr_short:.2f}, T-long={thr_long:.2f}, "
            f"T-src={self._last_ternary_threshold_source}, "
            f"Entry={entry_threshold:.2f}, Exit={exit_threshold:.2f}, "
            f"naklady/obchod {self.cost_spin.value():.3f}, "
            f"scope={scope_info.get('mode')} {scope_info.get('applied_rows')}/{scope_info.get('total_rows')}, "
            f"{key_line})"
        )

        if self._pending_auto_threshold_dialog is not None:
            payload = self._pending_auto_threshold_dialog
            self._pending_auto_threshold_dialog = None
            shown_metrics = self.last_metrics if isinstance(self.last_metrics, dict) else payload.best_metrics
            best_profit = self._pick_metric(shown_metrics or {}, "profit_net", "profit_gross", "profit")
            best_trades = self._pick_metric(shown_metrics or {}, "trades", "num_trades")
            QMessageBox.information(
                self,
                "Auto Entry/Exit hotovo",
                "Nalezeny prahy pro max profit_net:\n\n"
                f"Entry Threshold: {payload.best_entry:.2f}\n"
                f"Exit Threshold: {payload.best_exit:.2f}\n"
                f"profit_net: {best_profit}\n"
                f"trades: {best_trades}",
            )

    def _on_params_error(self, req_id: int, msg: str) -> None:
        if req_id == self._params_request_id:
            self._error(msg)

    def _on_params_finished(self, req_id: int) -> None:
        if req_id == self._params_request_id:
            self._params_worker = None

    def _on_model_settings_changed(self) -> None:
        self._refresh_threshold_preview()
        self.on_params_changed()

    @staticmethod
    def _task_auto_threshold_search(
        *,
        y_pred_raw,
        confidence_arr,
        y_true_current,
        df_current,
        fee_per_trade: float,
        current_entry: float,
        current_exit: float,
        exit_policy: str = DEFAULT_EXIT_POLICY,
        progress_cb=None,
        should_run=None,
    ) -> AutoThresholdPayload:
        return model_eval_runtime.run_auto_threshold_search_from_context(
            y_pred_raw=y_pred_raw,
            confidence_arr=confidence_arr,
            y_true_current=y_true_current,
            df_current=df_current,
            fee_per_trade=float(fee_per_trade),
            current_entry=float(current_entry),
            current_exit=float(current_exit),
            exit_policy=exit_policy,
            progress_cb=progress_cb,
            should_run=should_run,
        )

    def _start_auto_threshold_worker(self) -> None:
        self._stop_worker_attr("_auto_threshold_worker")
        self._auto_request_id += 1
        req_id = self._auto_request_id
        self._set_status("Auto Entry/Exit: hledam nejlepsi kombinaci pro max profit_net...")

        worker = TaskWorker(
            self._task_auto_threshold_search,
            y_pred_raw=np.asarray(self.y_pred_raw),
            confidence_arr=np.asarray(self.confidence_arr),
            y_true_current=np.asarray(self.y_true_current),
            df_current=self.df_current,
            fee_per_trade=float(self.cost_spin.value()),
            current_entry=float(self.et_spin.value()),
            current_exit=float(self.ext_spin.value()),
            exit_policy=self._active_exit_policy(),
        )
        self._auto_threshold_worker = worker
        worker.progress_text.connect(self._set_status)
        worker.result.connect(lambda payload, rid=req_id: self._on_auto_threshold_result(rid, payload))
        worker.error.connect(lambda msg, rid=req_id: self._on_auto_threshold_error(rid, msg))
        worker.finished.connect(lambda rid=req_id: self._on_auto_threshold_finished(rid))
        worker.start()

    def _on_auto_thresholds_clicked(self) -> None:
        if self.loaded_model is None or self.model_path is None:
            self._warn("Nejprve vyber model (.pkl).")
            return
        if self.data_path is None:
            self._warn("Nejprve vyber CSV s historickymi daty.")
            return
        if self.y_pred_raw is None or self.confidence_arr is None or self.df_current is None:
            self._pending_auto_threshold_search = True
            self._start_evaluation_worker()
            return
        self._pending_auto_threshold_search = False
        self._start_auto_threshold_worker()

    def _on_auto_threshold_result(self, req_id: int, payload: AutoThresholdPayload) -> None:
        if req_id != self._auto_request_id or payload is None:
            return
        self.et_spin.blockSignals(True)
        self.ext_spin.blockSignals(True)
        self.et_spin.setValue(float(payload.best_entry))
        self.ext_spin.setValue(float(payload.best_exit))
        self.et_spin.blockSignals(False)
        self.ext_spin.blockSignals(False)

        self._refresh_threshold_preview()
        self._pending_auto_threshold_dialog = payload
        self._run_params_recalc()

    def _on_auto_threshold_error(self, req_id: int, msg: str) -> None:
        if req_id == self._auto_request_id:
            self._warn(msg)

    def _on_auto_threshold_finished(self, req_id: int) -> None:
        if req_id == self._auto_request_id:
            self._auto_threshold_worker = None

    def set_model_path(self, file_path: str) -> None:
        if not file_path:
            return

        try:
            normalized_path = str(Path(file_path))
        except Exception:
            normalized_path = str(file_path)

        if self.model_path == normalized_path and self.loaded_model is not None:
            self.model_label.setText(f"Model: {normalized_path}")
            return

        try:
            loaded = model_eval_runtime.load_predictor_with_merged_meta(normalized_path)
            self.loaded_model = loaded.predictor
            self.model_metadata = loaded.metadata
            if isinstance(self.model_metadata, dict):
                _, tshort, tlong = self._meta_threshold_values()
                if not isinstance(tshort, (int, float)) or not isinstance(tlong, (int, float)):
                    raise ValueError(
                        "Model neobsahuje ternarni prahy (ternary_threshold_short/long). "
                        "Tab 3 je nyni pouze pro ternarni modely."
                    )
            self.model_path = normalized_path
            self.model_label.setText(f"Model: {normalized_path}")
            self._refresh_threshold_preview()
            version_warning = loaded.version_warning or build_sklearn_version_warning(
                self.model_metadata, model_path=normalized_path
            )
            if version_warning:
                self._set_status(f"Model nacten. {version_warning}")
            else:
                self._set_status("Model nacten.")
        except Exception as e:
            self._error(f"Nepodarilo se nacist model:\n{e}")
