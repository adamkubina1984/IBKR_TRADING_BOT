# gui/tab_model_training.py
import json as jsonlib
import os
import re
from pathlib import Path

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
from ibkr_trading_bot.model.train_models import _model_dir, _select_feature_columns, train_and_evaluate_model


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h_l  = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def label_triple_barrier(df: pd.DataFrame, K: int, tp_atr: float, sl_atr: float) -> pd.Series:
    """
    Vrací {-1, 0, +1}:
      +1 = TP dosaženo jako první, -1 = SL dosaženo jako první, 0 = timeout (ani TP ani SL)
    """
    close = df["close"].to_numpy()
    atr   = df["atr"].to_numpy()
    n = len(df)
    y = np.zeros(n, dtype=np.int8)

    for i in range(n):
        jmax = min(n - 1, i + K)
        base = close[i]
        tp = base + tp_atr * atr[i]
        sl = base - sl_atr * atr[i]

        hit = 0
        # procházíme dopředu, první dotyk rozhoduje
        for j in range(i + 1, jmax + 1):
            hi = df["high"].iat[j]
            lo = df["low"].iat[j]
            if hi >= tp:
                hit = +1; break
            if lo <= sl:
                hit = -1; break
        y[i] = hit  # 0 = timeout
    return pd.Series(y, index=df.index, name="y_tb")


class TrainWorker(QThread):
    progress = Signal(int, int, dict, float, float)
    phase = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, df_full: pd.DataFrame, holdout_bars: int, estimator: str, name_prefix: str, meta_extra: dict):
        super().__init__()
        self.df_full = df_full
        self.holdout_bars = int(holdout_bars)
        self.estimator = estimator
        self.name_prefix = name_prefix
        self.meta_extra = meta_extra

    def run(self):
        try:
            self.phase.emit("grid")
            def cb(idx, total, params, mean_f1, std_f1):
                self.progress.emit(int(idx), int(total), dict(params), float(mean_f1), float(std_f1))

            train_and_evaluate_model(
                self.df_full,
                estimator_name=self.estimator,
                param_grid=None,
                on_progress=cb,
                holdout_bars=self.holdout_bars,
                name_prefix=self.name_prefix,
                meta_extra=self.meta_extra,
                mc_enabled=True,
                mc_iters=200,
                mc_block_len=int(self.meta_extra.get("mc_block_len", 100)),
                annualize_sharpe=True,   # ← nově přidáno
            )

            self.phase.emit("save")
            out_dir = Path(_model_dir())
            patt = f"{self.name_prefix}_{self.estimator}_*.pkl" if self.name_prefix else f"{self.estimator}_*.pkl"
            files = sorted(out_dir.glob(patt), key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = files[0].as_posix() if files else ""
            self.finished.emit(model_path)
            self.phase.emit("done")
        except Exception as e:
            self.error.emit(str(e))
            self.phase.emit("error")

class ModelTrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset: pd.DataFrame | None = None
        self.csv_path: str | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_test: pd.Series | None = None
        self.holdout_bars_default = 500

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # 1) CSV
        box1 = QGroupBox("1) Načtení dat pro trénink")
        lay1 = QHBoxLayout(box1)
        self.lbl_csv = QLabel("Vybraný soubor: —")
        self.btn_csv = QPushButton("Vybrat CSV…")
        self.btn_csv.clicked.connect(self.pick_csv)
        lay1.addWidget(self.lbl_csv); lay1.addStretch(1); lay1.addWidget(self.btn_csv)
        root.addWidget(box1)

        # 2) Trénování
        box2 = QGroupBox("2) Trénování modelu")
        lay2 = QVBoxLayout(box2)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["hgbt", "rf", "et", "xgb"])
        row.addWidget(self.cmb_model); row.addStretch(1)

        self.btn_train = QPushButton("Trénovat (5000 bars, holdout 500)")
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self.run_training)
        row.addWidget(self.btn_train)
        lay2.addLayout(row)

        self.prog = QProgressBar(); self.prog.setRange(0, 1); lay2.addWidget(self.prog)

        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["#", "mean score (CV)", "std", "params"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay2.addWidget(self.tbl, 1)

        root.addWidget(box2, 2)

        # 3) Konzole
        box3 = QGroupBox("3) Konzole")
        lay3 = QVBoxLayout(box3)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setPlaceholderText("Hlášky tréninku a evaluace…")
        lay3.addWidget(self.log); root.addWidget(box3, 1)

    # ---------- I/O ----------
    def pick_csv(self):
        base_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
        path, _ = QFileDialog.getOpenFileName(self, "Vyber CSV s daty", base_dir.as_posix(), "CSV Files (*.csv)")
        if not path: return
        self.csv_path = path; self.lbl_csv.setText(f"Vybraný soubor: {os.path.basename(path)}")

        try:
            svc = DatasetService()
            df = svc.prepare_from_csv(path, labeling="triple_barrier")
            if len(df) > 5000: df = df.iloc[-5000:].reset_index(drop=True)
            self.dataset = df
            self.log.appendPlainText(f"✅ Načteno: {path} | řádků={len(df)}")
            self.btn_train.setEnabled(True)
            self.tbl.setRowCount(0); self.prog.setRange(0, 1); self.prog.setValue(0)
        except Exception as e:
            self.log.appendPlainText(f"❌ Chyba načtení/přípravy dat: {e}")
            self.btn_train.setEnabled(False)

    # ---------- Train ----------
    def run_training(self):
        if self.dataset is None:
            self.log.appendPlainText("⚠️ Nejprve vyber CSV."); return

        est = self.cmb_model.currentText().strip().lower()
        df = self.dataset.copy().sort_values("timestamp").reset_index(drop=True)
        n_total = len(df)
        n_hold = min(self.holdout_bars_default, max(n_total - 50, 0))
        df_train = df.iloc[: n_total - n_hold].reset_index(drop=True)
        df_hold  = df.iloc[n_total - n_hold :].reset_index(drop=True) if n_hold > 0 else None

        feats_all = _select_feature_columns(df_train)
        X_cols = feats_all
        if df_hold is not None:
            self.X_test = df_hold[X_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            self.y_test = (df_hold["target"].astype(float) > 0).astype(int)
        else:
            self.X_test, self.y_test = None, None

        name_prefix, meta_extra = self._name_and_meta_from_csv(
            self.csv_path, n_total, len(df_train), len(df_hold) if df_hold is not None else 0
        )
        self.log.appendPlainText(f"🧾 name_prefix={name_prefix} | meta_extra={meta_extra}")

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
        meta_extra["mc_block_len"] = mc_block

        self.tbl.setRowCount(0)
        self.prog.setRange(0, 1)
        self.prog.setValue(0)
        self.worker = TrainWorker(
            df_full=df,
            holdout_bars=n_hold,
            estimator=est,
            name_prefix=name_prefix,
            meta_extra=meta_extra,
        )
        self.worker.progress.connect(self._on_progress_row)
        self.worker.phase.connect(lambda ph: None)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(lambda msg: self._on_error(msg))
        self.btn_train.setEnabled(False)
        self.worker.start()

    def _name_and_meta_from_csv(self, path: str, n_total: int, n_train: int, n_hold: int):
        base = os.path.basename(path)
        m = re.match(r"tv_([^_]+)_([^_]+)_([^_]+)_.+\.csv$", base)
        instrument, exchange, timeframe = ("UNKNOWN", "UNK", "UNK")
        if m:
            instrument = m.group(1)
            exchange = m.group(2)
            timeframe = m.group(3)
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

    # ---------- callbacks ----------
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
                self.log.appendPlainText("ℹ️ Model uložen, ale cesta nenalezena.")
                return
            self.log.appendPlainText(f"ℹ️ Načítám model: {os.path.basename(model_path)}")
            obj = joblib.load(model_path)
            if isinstance(obj, dict):
                mdl = obj.get("model") or obj
                thr = float(obj.get("decision_threshold", 0.45))
                feats = obj.get("features")
            else:
                mdl = obj
                thr = 0.5
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
                    self.log.appendPlainText(
                        f"ℹ️ Meta: timeframe={meta.get('timeframe')} total/train/hold="
                        f"{meta.get('n_total_bars')}/{meta.get('n_train_bars')}/{meta.get('n_holdout_bars')}"
                    )
                    if meta.get("annualize_sharpe") is True:
                        self.log.appendPlainText("ℹ️ Sharpe je anualizovaný (přepočet na rok).")
                    mc = meta.get("mc") or {}
                    if mc:
                        s = mc.get("sharpe", {})
                        dd = mc.get("max_drawdown", {})
                        note = mc.get("note")
                        self.log.appendPlainText(
                            f"🧪 MC: Sharpe p50={s.get('p50','n/a')} "
                            f"[{s.get('p10','n/a')}–{s.get('p90','n/a')}], "
                            f"MaxDD p90={dd.get('p90','n/a')}, iters={mc.get('iters')} block={mc.get('block_len')}"
                            + (f" ({note})" if note else "")
                        )
                    # pokud jsou v meta metriky se Sharpe, vypiš je hned
                    met = meta.get("metrics") or {}
                    if "sharpe" in met:
                        self.log.appendPlainText(f"📌 HOLDOUT Sharpe: {met['sharpe']:.4f} (annualized={bool(meta.get('annualize_sharpe'))})")
            except Exception:
                pass

            # TEST evaluace
            if mdl is not None and self.X_test is not None and self.y_test is not None:
                X_eval = self.X_test.reindex(columns=feats, fill_value=0.0) if feats else self.X_test
                if hasattr(mdl, "predict_proba"):
                    pr = mdl.predict_proba(X_eval)
                    p1 = pr[:, 1] if isinstance(pr, np.ndarray) and pr.ndim == 2 and pr.shape[1] >= 2 else np.asarray(pr).ravel()
                    y_pred = (p1 >= thr).astype(int)
                elif hasattr(mdl, "decision_function"):
                    z = np.asarray(mdl.decision_function(X_eval)).ravel()
                    p1 = 1.0 / (1.0 + np.exp(-z))
                    y_pred = (p1 >= thr).astype(int)
                else:
                    y_pred = mdl.predict(X_eval)
                acc = accuracy_score(self.y_test, y_pred)
                prec = precision_score(self.y_test, y_pred, zero_division=0)
                rec = recall_score(self.y_test, y_pred, zero_division=0)
                f1 = f1_score(self.y_test, y_pred, zero_division=0)
                self.log.appendPlainText(
                    f"📈 HOLDOUT (posledních {len(self.y_test)} barů): Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}"
                )

        except Exception as e:
            self.log.appendPlainText(f"❌ Chyba při dokončení/evaluaci: {e}")
        finally:
            self.btn_train.setEnabled(True)

    def _on_error(self, msg: str):
        self.log.appendPlainText(f"❌ Chyba trénování: {msg}")
        self.btn_train.setEnabled(True)
