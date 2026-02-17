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
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ibkr_trading_bot.core.services.evaluation_service import EvaluationService
from ibkr_trading_bot.features.feature_engineering import prepare_dataset_with_targets

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

        # Confidence threshold
        self.thr_label = QLabel("Confidence ≥")
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.0, 1.0)
        self.thr_spin.setSingleStep(0.01)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(0.65)
        self.thr_spin.setToolTip("Minimální jistota predikce. Pod prahem bude signál zrušen (flat).")
        self.thr_spin.valueChanged.connect(self.on_params_changed)

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

        params_layout.addWidget(self.thr_label)
        params_layout.addWidget(self.thr_spin)
        params_layout.addSpacing(16)
        params_layout.addWidget(self.cost_label)
        params_layout.addWidget(self.cost_spin)
        params_layout.addSpacing(16)
        params_layout.addWidget(self.roll_label)
        params_layout.addWidget(self.roll_combo)
        params_layout.addStretch(1)
        params_group.setLayout(params_layout)

        # ====== 3) SKUPINA: Metriky ======
        metrics_group = QGroupBox("Metriky modelu")
        metrics_layout = QVBoxLayout()
        self.metrics_table = QTableWidget(0, 3)
        self.metrics_table.setHorizontalHeaderLabels(["Metrika", "Hodnota", "Význam"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
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

        content_row = QHBoxLayout()
        content_row.addWidget(metrics_group, 1)
        content_row.addWidget(charts_group, 2)
        content_row.setStretch(0, 1)
        content_row.setStretch(1, 2)
        main_layout.addLayout(content_row)

        # popisy metrik
        self.metric_descriptions = self._build_metric_descriptions()

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
        try:
            obj = joblib.load(file_path)
            predictor, metadata = self._extract_predictor_from_object(obj)
            self.loaded_model = predictor
            self.model_metadata = metadata if isinstance(metadata, dict) else (metadata or {})
            self.model_path = file_path
            self.model_label.setText(f"Model: {file_path}")
            self._set_status("Model načten.")
        except Exception as e:
            self._error(f"Nepodařilo se získat estimator z načteného souboru:\n{e}")

    def on_open_data_clicked(self):
        # Dynamický a záložní start dir (po změně kořene projektu)
        try:
            project_root = Path(__file__).resolve().parents[1]
        except Exception:
            project_root = Path(os.getcwd())
        raw_dir_dyn = project_root / "data" / "raw"
        raw_dir_abs = Path(r"C:\Users\adamk\Můj disk\Trader\ibkr_trading_bot\data\raw")

        if raw_dir_dyn.is_dir():
            start_dir = str(raw_dir_dyn)
        elif raw_dir_abs.is_dir():
            start_dir = str(raw_dir_abs)
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

        self.data_path = file_path
        self.data_label.setText(f"Data (CSV): {file_path}")
        self._set_status("Data připravena.")

    def on_evaluate_clicked(self):
        if self.loaded_model is None or self.model_path is None:
            self._warn("Nejprve vyber model (.pkl).")
            return
        if self.data_path is None:
            self._warn("Nejprve vyber CSV s historickými daty.")
            return

        # 1) CSV
        try:
            df = pd.read_csv(self.data_path, encoding="utf-8", engine="python")
            self.df_current = df
            self.close_series = self._safe_close_series(df)
        except Exception as e:
            self._error(f"Chyba při načítání CSV:\n{e}")
            return

        # 2) Dataset
        try:
            prepared = prepare_dataset_with_targets(df)
            X, y_true = self._extract_X_y(prepared)
            X = self._coerce_features_for_model(X)
            self.X_current = X
            self.y_true_current = y_true
        except Exception as e:
            self._error(f"Chyba při přípravě datasetu:\n{e}")
            return

        # 3) Predikce
        try:
            if not hasattr(self.loaded_model, "predict"):
                raise AttributeError("Načtený objekt nemá metodu `.predict`.")
            y_pred = self.loaded_model.predict(self.X_current)
            self.y_pred_raw = np.asarray(y_pred)

            # confidence (pro threshold)
            proba, classes, decision = self._get_model_scores(self.X_current)
            self.confidence_arr = self._confidence_from_scores(proba, decision)

            # aplikace prahu + NORMALIZACE NA −1/0/+1
            self.y_pred_used = self._apply_confidence_threshold(
                raw_pred=self.y_pred_raw,
                confidence=self.confidence_arr,
                threshold=self.thr_spin.value()
            )
            self.y_pred_used = self._normalize_pred(self.y_pred_used)

            self._set_status(f"Predikce OK. Aplikován confidence filtr ≥ {self.thr_spin.value():.2f}.")
        except Exception as e:
            self._error(f"Chyba při predikci modelem:\n{e}")
            return

        # 4–5) Metriky vč. nákladů/obchod
        try:
            results = self._eval_service.calculate_metrics(
                y_true=self.y_true_current,
                y_pred=self.y_pred_used,
                df=self.df_current,
                fee_per_trade=float(self.cost_spin.value()),
                slippage_bps=0.0,
                rolling_window=200,
                annualize_sharpe=False
            )
        except Exception as e:
            self._error(f"Chyba při výpočtu metrik:\n{e}")
            return

        if not isinstance(results, dict) or not results:
            self._error("Výpočet metrik vrátil prázdný výsledek.")
            return

        # 6) UI výstup
        self.last_metrics = results

        self.trades_df = self._extract_trades_df(results)
        self.btn_export_trades.setEnabled(
        isinstance(self.trades_df, pd.DataFrame) and not self.trades_df.empty
        )
        self._populate_metrics_table(results)
        self._draw_equity_chart(results)
        trade_pnls_plot = results.get("trade_pnls_net") or results.get("trade_pnls")
        if not trade_pnls_plot:
            trade_pnls_plot = self._compute_trade_pnls_from_signals()

        self._draw_histogram(trade_pnls_plot)
        self._draw_rolling_chart(trade_pnls_plot)

        self._set_status(
            f"Hotovo. Vyhodnocení dokončeno (náklady/obchod {self.cost_spin.value():.3f})."
        )

    def on_params_changed(self, *_):
        if self.X_current is None or self.y_pred_raw is None or self.df_current is None:
            return

        # aplikace nového filtru + NORMALIZACE NA −1/0/+1
        self.y_pred_used = self._apply_confidence_threshold(
            raw_pred=self.y_pred_raw,
            confidence=self.confidence_arr,
            threshold=self.thr_spin.value()
        )
        self.y_pred_used = self._normalize_pred(self.y_pred_used)

        try:
            results = self._eval_service.calculate_metrics(
                y_true=self.y_true_current,
                y_pred=self.y_pred_used,
                df=self.df_current,
                fee_per_trade=float(self.cost_spin.value()),
                slippage_bps=0.0,
                rolling_window=200,
                annualize_sharpe=False
            )
        except Exception as e:
            self._error(f"Chyba při výpočtu metrik (po změně parametrů):\n{e}")
            return

        if not isinstance(results, dict) or not results:
            self._error("Výpočet metrik vrátil prázdný výsledek (po změně parametrů).")
            return

        self.last_metrics = results
        self.trades_df = self._extract_trades_df(results)
        self.btn_export_trades.setEnabled(
            isinstance(self.trades_df, pd.DataFrame) and not self.trades_df.empty
        )
        self._populate_metrics_table(results)
        self._draw_equity_chart(results)
        trade_pnls_plot = results.get("trade_pnls_net") or results.get("trade_pnls")
        self._draw_histogram(trade_pnls_plot)
        self._draw_rolling_chart(trade_pnls_plot)

        self._set_status(
            f"Přepočteno (confidence ≥ {self.thr_spin.value():.2f}, "
            f"náklady/obchod {self.cost_spin.value():.3f})"
        )

    # ---------------- Helpery: model / dataset ----------------
    def _extract_predictor_from_object(self, obj):
        """Vrátí (model, metadata_dict_nebo_None). Podporuje estimator, dict, tuple/list."""
        if hasattr(obj, "predict"):
            return obj, None
        if isinstance(obj, dict):
            for k in ["model", "estimator", "pipeline", "clf", "best_estimator_", "sk_model", "predictor"]:
                if k in obj and hasattr(obj[k], "predict"):
                    return obj[k], obj
            # případně projdeme hodnoty
            for _, v in obj.items():
                if hasattr(v, "predict"):
                    return v, obj
            raise ValueError("Ve slovníku není žádný objekt s `.predict`.")
        if isinstance(obj, (tuple, list)):
            for v in obj:
                if hasattr(v, "predict"):
                    return v, None
            raise ValueError("V tuple/listu není žádná položka s `.predict`.")
        raise ValueError(f"Neočekávaný typ uloženého modelu: {type(obj).__name__}.")

    def _extract_X_y(self, prepared):
        if isinstance(prepared, (tuple, list)):
            if len(prepared) >= 2:
                return prepared[0], prepared[1]
            return prepared[0], None
        if isinstance(prepared, dict):
            X = prepared.get("X") or prepared.get("features") or prepared.get("data") or prepared.get("df")
            y = prepared.get("y") or prepared.get("target") or prepared.get("y_true")
            if X is None:
                raise ValueError("V dictu chybí klíč 'X'/'features'/'data'.")
            return X, y
        if isinstance(prepared, pd.DataFrame):
            X, y = prepared, None
            for cand in ["target", "y", "label"]:
                if cand in prepared.columns:
                    y = prepared[cand].values
                    X = prepared.drop(columns=[cand])
                    break
            return X, y
        if isinstance(prepared, np.ndarray):
            return prepared, None
        raise ValueError("Neočekávaný návratový typ z prepare_dataset_with_targets(df).")

    def _coerce_features_for_model(self, X):
        """Připraví X podle očekávání modelu (číselné typy, doplnění chybějících, pořadí).
           1) Pokud metadata obsahují expected_features, použijí se přednostně (v daném pořadí).
           2) Jinak se použije feature_names_in_ (pokud je k dispozici).
        """
        if not isinstance(X, pd.DataFrame):
            return X

        dfX = X.copy()

        # 0) Převody typů: datumy → int, stringy zkus na datetime → int (jinak ponech)
        for col in dfX.columns:
            if pd.api.types.is_datetime64_any_dtype(dfX[col]):
                dfX[col] = dfX[col].astype("int64") // 10**6
            elif dfX[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(dfX[col], errors="raise")
                    dfX[col] = parsed.astype("int64") // 10**6
                except Exception:
                    pass

        # 1) Zahodit jasně nenumerické sloupce (ponecháme bool a čísla)
        for c in list(dfX.columns):
            if (not pd.api.types.is_bool_dtype(dfX[c])) and (not pd.api.types.is_numeric_dtype(dfX[c])):
                dfX.drop(columns=[c], inplace=True, errors="ignore")

        # 2) METADATA expected_features (pokud jsou k dispozici v dictu modelu)
        try:
            if isinstance(self.model_metadata, dict):
                exp = self.model_metadata.get("expected_features") or self.model_metadata.get("features")
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
        except Exception:
            # když metadata nejsou nebo jsou jiného tvaru, pokračujeme původní cestou
            pass

        # 3) Pokud model zná feature_names_in_, zarovnáme pořadí a doplníme chybějící
        names = getattr(self.loaded_model, "feature_names_in_", None)
        if names is not None:
            names = [str(x) for x in list(names)]
            for k in names:
                if k not in dfX.columns:
                    dfX[k] = 0.0
            dfX = dfX[names]

        # 4) doplnění NaN a typů
        med = dfX.median(numeric_only=True)
        dfX = dfX.fillna(med).fillna(0.0)
        for c in dfX.columns:
            if not pd.api.types.is_bool_dtype(dfX[c]):
                dfX[c] = dfX[c].astype("float32", copy=False)
        return dfX

    # ---------------- Helpery: confidence / threshold ----------------
    def _get_model_scores(self, X):
        """Vrací (proba, classes, decision) podle možností modelu."""
        proba = None
        classes = None
        decision = None
        try:
            if hasattr(self.loaded_model, "predict_proba"):
                proba = self.loaded_model.predict_proba(X)
                classes = getattr(self.loaded_model, "classes_", None)
        except Exception:
            proba = None
        try:
            if hasattr(self.loaded_model, "decision_function"):
                decision = self.loaded_model.decision_function(X)
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
        arr = np.asarray(raw_pred).copy()
        conf = np.asarray(confidence).reshape(-1)
        thr = float(threshold)
        mask_low = conf < thr
        try:
            arr[mask_low] = 0
        except Exception:
            # kdyby byl typ objektový, uděláme bezpečný převod
            tmp = np.array(arr, dtype=object)
            tmp[mask_low] = 0
            arr = tmp
        return arr

    # --- NEW: normalizace predikcí po prahování na {-1,0,+1} ---
    def _normalize_pred(self, arr):
        """
        Převede libovolné predikce na {-1, 0, +1}.
        Podporuje čísla, booly i texty ('long'/'short'/...).
        """
        a = np.asarray(arr, dtype=object)
        out = np.zeros(a.shape, dtype=float)
        # číselné typy
        num_mask = np.array([isinstance(x, (int, float, np.number)) for x in a], dtype=bool)
        out[num_mask] = np.sign(a[num_mask].astype(float))
        # texty
        txt = np.char.lower(a.astype(str))
        out[(txt == "long") | (txt == "buy") | (txt == "up") | (txt == "1") | (txt == "+1")] = 1.0
        out[(txt == "short") | (txt == "sell") | (txt == "down") | (txt == "-1")] = -1.0
        return out

    # ---------------- Helpery: PnL a breakdown ----------------
    def _build_positions(self, y_pred):
        arr = np.asarray(y_pred).astype(float)
        pos = np.zeros_like(arr, dtype=int)
        pos[arr > 0] = 1
        pos[arr < 0] = -1
        return pos

    def _compute_trade_pnls_from_signals(self, fee_per_trade: float = None):
        """
        Rekonstruuje seznam PnL po obchodech z y_pred_used a close.
        LONG: vstup při přechodu -> +1, výstup při změně na 0/−1
        SHORT: vstup při přechodu -> −1, výstup při změně na 0/+1
        fee_per_trade = náklad za *každou* změnu pozice (open nebo close).
        Vrací list netto PnL (po nákladech).
        """
        if self.close_series is None or self.y_pred_used is None:
            return []

        close = np.asarray(self.close_series, dtype=float)
        pos   = self._build_positions(self.y_pred_used)  # −1/0/+1
        n = min(len(close), len(pos))
        if n < 2:
            return []

        if fee_per_trade is None and hasattr(self, "cost_spin"):
            fee_per_trade = float(self.cost_spin.value() or 0.0)
        fee = float(fee_per_trade or 0.0)

        trade_pnls = []
        cur_pos = 0
        entry_px = None

        for i in range(1, n):
            p_prev, p_now = pos[i-1], pos[i]
            # otevření?
            if cur_pos == 0 and p_prev == 0 and p_now != 0:
                cur_pos = p_now
                entry_px = close[i]
                # náklad za open
                if fee:
                    trade_pnls.append(-fee)  # evidujeme náklad (můžeme i odložit; varianta: odečíst až v nettě)
                    trade_pnls.pop()         # ne, raději započítáme až do PnL obchodu níže
            # uzavření (přechod do 0 nebo flip)
            if cur_pos != 0 and (p_now == 0 or np.sign(p_now) != np.sign(cur_pos)):
                exit_px = close[i]
                if cur_pos == +1:
                    gross = exit_px - entry_px
                else:
                    gross = entry_px - exit_px
                net = gross - 2*fee  # open + close
                trade_pnls.append(net)
                cur_pos = 0
                entry_px = None
            # flip → současně otevřeme novou v opačném směru
            if cur_pos == 0 and p_now != 0 and (p_prev != p_now):
                cur_pos = p_now
                entry_px = close[i]
        return trade_pnls

    # ---------------- Helpery: grafy ----------------
    def _draw_equity_chart(self, results: dict):
        ax = self.canvas_equity.ax
        ax.clear()
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_title("Equity křivka")

        # 1) Načti křivky z výsledků (pokud existují)
        eq = results.get("equity_curve")
        eq_net = results.get("equity_curve_net")

        # 2) Pokud chybí, fallback z PnL per-trade
        if eq is None and results.get("trade_pnls") is not None:
            eq = np.cumsum(np.asarray(results["trade_pnls"], dtype=float))
        if eq_net is None and results.get("trade_pnls_net") is not None:
            eq_net = np.cumsum(np.asarray(results["trade_pnls_net"], dtype=float))

        # 3) Odhad, zda přepočítat per-BAR křivky (per-trade bývá krátké)
        n_bars = None
        if self.close_series is not None:
            n_bars = len(self.close_series)
        elif self.X_current is not None:
            try:
                n_bars = len(self.X_current)
            except Exception:
                n_bars = None

        def _len(x):
            try:
                return len(x)
            except Exception:
                return None

        need_bar_equity = False
        if eq is None or eq_net is None:
            need_bar_equity = True
        else:
            Le, Ln = _len(eq), _len(eq_net)
            if Le is None or Ln is None:
                need_bar_equity = True
            elif Le != Ln:
                need_bar_equity = True
            elif n_bars is not None and (Le < 0.7 * n_bars or Ln < 0.7 * n_bars):
                need_bar_equity = True

        # 4) Per-bar výpočet z close & signálů (stejná délka pro gross i net)
        if need_bar_equity and self.close_series is not None and self.y_pred_used is not None:
            n = min(len(self.close_series), len(self.y_pred_used))
            if n > 1:
                close = self.close_series.iloc[:n].to_numpy(dtype=float)
                pos = self._build_positions(self.y_pred_used)[:n]
                pos_prev = np.concatenate(([0], pos[:-1]))
                close_prev = np.concatenate(([close[0]], close[:-1]))
                gross = (close - close_prev) * pos_prev
                trades = np.abs(pos - pos_prev)
                fee = float(self.cost_spin.value()) if hasattr(self, 'cost_spin') else 0.0
                net = gross - fee * trades
                eq = np.cumsum(gross)
                eq_net = np.cumsum(net)

        # 5) vykreslení
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
    def _safe_close_series(self, df: pd.DataFrame):
        for c in ["close", "Close", "CLOSE", "adj_close", "Adj Close"]:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce")
        return None

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
            "f1": "Vyváženost mezi precision a recall – jak dobře model predikuje obchodní signály.",
            "precision": "Kolik z predikovaných obchodů bylo skutečně správně.",
            "recall": "Kolik správných obchodů model dokázal najít.",
            "profit": "Celkový zisk/ztráta (hrubý).",
            "sharpe_ratio": "Zisk očištěný o volatilitu – vyšší = lepší poměr zisku k riziku.",
            "max_drawdown": "Největší pokles equity od maxima (riziko velké ztráty).",
            "winrate": "Procento ziskových obchodů.",
            "profit_factor": "Poměr hrubého zisku ku hrubé ztrátě (>1 = zisková strategie).",
            "signal_stability": "Jak moc se model přepíná mezi signály – vyšší = stabilnější.",
            "num_trades": "Počet provedených obchodů.",
            "avg_profit_per_trade": "Průměrný zisk na obchod (hrubý).",
            "median_profit_per_trade": "Medián zisku na obchod (hrubý).",
            "var_95": "Value at Risk 95 % (hrubé) – odhad ztráty v 95 % případů.",
            "cvar_95": "Conditional VaR 95 % (hrubé) – průměr ztrát v nejhorších 5 %.",
            # netto
            "profit_net": "Celkový zisk po nákladech.",
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
                return pd.DataFrame(lst)

            pnls = results.get("trade_pnls_net") or results.get("trade_pnls")
            if pnls is not None:
                arr = np.asarray(pnls, dtype=float).reshape(-1)
                return pd.DataFrame({"pnl": arr})
        except Exception:
            pass
        return None

