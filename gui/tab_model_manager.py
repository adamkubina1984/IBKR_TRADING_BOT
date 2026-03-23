from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
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

from ibkr_trading_bot.core.services.model_service import load_model_with_meta

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "model_outputs"


def _as_float(x: Any, default: float = float("-inf")) -> float:
    if x is None:
        return default
    try:
        if isinstance(x, str):
            x2 = x.strip().replace(",", ".")
            if x2 == "":
                return default
            return float(x2)
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


@dataclass
class ModelRecord:
    model_path: Path
    meta_path: Path | None
    sha1: str
    created: str
    metrics: dict[str, float]
    features_n: int
    classes: list[str]
    top_feature: str
    file_size: int
    file_mtime_ns: int


def _record_rank_key(r: ModelRecord) -> tuple[float, float, float, float]:
    profit = _as_float(
        r.metrics.get("profit_net", r.metrics.get("profit_gross", None)),
        default=float("-inf"),
    )
    sharpe = _as_float(r.metrics.get("sharpe", None), default=float("-inf"))
    trades = _as_float(r.metrics.get("trades", None), default=0.0)
    ts = _as_timestamp(r.created, r.model_path)
    return (profit, sharpe, trades, ts)


def _top_feature_from_meta(meta: dict[str, Any]) -> str:
    feat_imp = meta.get("feature_importance", {})
    if not isinstance(feat_imp, dict) or not feat_imp:
        return ""
    try:
        return str(max(feat_imp.keys(), key=lambda x: feat_imp[x]))[:12]
    except Exception:
        return ""


def _directory_snapshot(dir_path: Path) -> tuple[tuple[str, int, int], ...]:
    if not dir_path.exists():
        return ()
    snapshot: list[tuple[str, int, int]] = []
    for p in dir_path.glob("*.pkl"):
        try:
            st = p.stat()
        except OSError:
            continue
        snapshot.append((p.name, int(st.st_size), int(st.st_mtime_ns)))
    snapshot.sort()
    return tuple(snapshot)


def discover_models(dir_path: Path) -> list[ModelRecord]:
    recs: list[ModelRecord] = []
    if not dir_path.exists():
        return recs

    for p in dir_path.glob("*.pkl"):
        try:
            p_stat = p.stat()
        except OSError:
            continue

        meta_candidates = [p.with_name(p.stem + "_meta.json"), p.parent / "model_meta.json"]
        meta: dict[str, Any] = {}
        meta_path = None
        for m in meta_candidates:
            if not m.exists():
                continue
            try:
                loaded = json.loads(m.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(loaded, dict):
                meta = loaded
                meta_path = m
                break

        classes = meta.get("model_classes")
        if not isinstance(classes, list) and isinstance(meta.get("classes"), list):
            classes = list(meta.get("classes"))
        features = meta.get("trained_features")
        if not isinstance(features, list) and isinstance(meta.get("features"), list):
            features = list(meta.get("features"))
        sha1 = str(meta.get("sha1") or meta.get("model_sha1") or "")

        holdout_metrics_raw = meta.get("metrics_holdout") or {}
        fallback_metrics_raw = meta.get("metrics") or {}
        metrics_raw = holdout_metrics_raw if holdout_metrics_raw else fallback_metrics_raw
        metrics: dict[str, float] = {}
        for k, v in metrics_raw.items():
            try:
                if isinstance(v, (int, float)):
                    if np.isfinite(v):
                        metrics[k] = float(v)
                elif isinstance(v, str):
                    v_clean = v.strip()
                    if v_clean.lower() in ("nan", "infinity", "inf", "-infinity"):
                        continue
                    metrics[k] = float(v_clean)
                elif isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (int, float)):
                    if np.isfinite(v[0]):
                        metrics[k] = float(v[0])
            except (ValueError, TypeError):
                pass

        recs.append(
            ModelRecord(
                model_path=p,
                meta_path=meta_path,
                sha1=sha1,
                created=str(meta.get("created_at_iso") or meta.get("created_at") or ""),
                metrics=metrics,
                features_n=len(features or []),
                classes=[str(c) for c in list(classes or [])],
                top_feature=_top_feature_from_meta(meta),
                file_size=int(p_stat.st_size),
                file_mtime_ns=int(p_stat.st_mtime_ns),
            )
        )

    recs.sort(key=_record_rank_key, reverse=True)
    return recs


class ModelManagerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tab_model_manager")

        self.dir_edit = QLineEdit(self)
        self.dir_edit.setText(str(DEFAULT_MODEL_DIR))
        self.btn_browse = QPushButton("Zvolit složku s modely...", self)
        self.chk_auto = QCheckBox("Auto-load nejnovější/best model")
        self.chk_auto.setChecked(True)

        self.tbl = QTableWidget(self)
        self.tbl.setColumnCount(8)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.setHorizontalHeaderLabels(
            ["Model", "SHA1", "Vytvořen", "Sharpe(H)", "Profit(H)", "PF(H)", "#Feats", "Top Feature"]
        )
        self.tbl.horizontalHeader().setStretchLastSection(True)

        self.lbl_loaded = QLabel("Načten: -")

        self.sens = QDoubleSpinBox(self)
        self.sens.setRange(0.01, 0.99)
        self.sens.setSingleStep(0.01)
        self.sens.setValue(0.50)

        self.btn_load = QPushButton("Načíst vybraný model", self)
        self.btn_validate = QPushButton("Ověřit shodu featur (live vs. trénink)", self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Složka modelů:"))
        top.addWidget(self.dir_edit)
        top.addWidget(self.btn_browse)

        mid = QHBoxLayout()
        mid.addWidget(self.chk_auto)
        mid.addStretch(1)
        mid.addWidget(QLabel("Citlivost:"))
        mid.addWidget(self.sens)

        bottom = QHBoxLayout()
        bottom.addWidget(self.btn_load)
        bottom.addWidget(self.btn_validate)
        bottom.addStretch(1)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(mid)
        lay.addWidget(self.tbl)
        lay.addWidget(self.lbl_loaded)
        lay.addLayout(bottom)

        self.records: list[ModelRecord] = []
        self.loaded = None
        self._last_snapshot: tuple[tuple[str, int, int], ...] = ()

        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_load.clicked.connect(self._on_load_selected)
        self.btn_validate.clicked.connect(self._on_validate)
        self.sens.valueChanged.connect(self._on_sens)
        self.tbl.itemSelectionChanged.connect(self._on_selection_changed)

        self._delete_shortcut = QShortcut(QKeySequence.Delete, self)
        self._delete_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(self._on_delete_selected)

        self.timer = QTimer(self)
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self._refresh_list(force=True)
        if self.chk_auto.isChecked():
            self._auto_load_best()

    def _on_browse(self):
        start_dir = str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR.exists() else str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Zvol složku s modely", start_dir)
        if d:
            self.dir_edit.setText(d)
            self._refresh_list(force=True)
            if self.chk_auto.isChecked():
                self._auto_load_best()

    def _on_sens(self, v: float):
        pass

    def _on_load_selected(self):
        rec = self._selected_record()
        if rec is None:
            QMessageBox.warning(self, "Model", "Nejprve vyber model v tabulce.")
            return
        self._load_and_propagate_model(rec)

    def _on_selection_changed(self):
        rec = self._selected_record()
        if rec is None:
            return
        self._propagate_model_path(rec.model_path)

    def _on_delete_selected(self):
        rec = self._selected_record()
        row = self.tbl.currentRow()
        if rec is None or row < 0:
            return

        reply = QMessageBox.question(
            self,
            "Smazat model",
            f"Opravdu smazat model?\n\n{rec.model_path.name}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        sidecar_path = rec.model_path.with_name(rec.model_path.stem + "_meta.json")
        try:
            if sidecar_path.exists():
                sidecar_path.unlink()
            rec.model_path.unlink()
        except Exception as e:
            QMessageBox.critical(self, "Mazani selhalo", str(e))
            return

        loaded = getattr(self, "loaded", None)
        if loaded is not None:
            try:
                loaded_path = Path(loaded.path).resolve()
            except Exception:
                loaded_path = None
            try:
                rec_path = rec.model_path.resolve()
            except Exception:
                rec_path = rec.model_path
            if loaded_path == rec_path:
                self.loaded = None
                self.lbl_loaded.setText("Nacten: -")

        self._refresh_list(force=True)
        if self.records:
            next_row = min(row, len(self.records) - 1)
            self.tbl.setCurrentCell(next_row, 0)

    def _selected_record(self) -> ModelRecord | None:
        row = self.tbl.currentRow()
        if row < 0 or row >= len(self.records):
            return None
        return self.records[row]

    def _on_validate(self):
        from PySide6.QtWidgets import QApplication

        if not getattr(self, "loaded", None):
            QMessageBox.information(self, "Validace", "Nejprve načti model v této záložce.")
            return

        candidates = []
        w = self
        while w is not None:
            candidates.append(w)
            w = w.parent()
        candidates += [self.window(), QApplication.activeWindow()]

        get_live = None
        for obj in filter(None, candidates):
            for name in ("get_live_features_df", "get_live_feature_df"):
                if hasattr(obj, name) and callable(getattr(obj, name)):
                    get_live = getattr(obj, name)
                    break
            if get_live:
                break

        if not get_live:
            QMessageBox.warning(self, "Validace", "Hlavní okno neumí poskytnout live featury.")
            return

        try:
            live_df = get_live()
        except Exception as e:
            QMessageBox.critical(self, "Validace", f"Získání live featur selhalo: {e}")
            return

        if live_df is None or (isinstance(live_df, pd.DataFrame) and live_df.empty):
            QMessageBox.information(self, "Validace", "Zadna live data. Otevri Tab 5 a nacti snapshot.")
            return

        trained_feats = list(self.loaded.meta.get("trained_features", []) or [])
        if not trained_feats:
            try:
                mdl_obj = joblib.load(self.loaded.path)
                mdl = mdl_obj.get("model") if isinstance(mdl_obj, dict) and "model" in mdl_obj else mdl_obj
                if hasattr(mdl, "feature_names_in_") and mdl.feature_names_in_ is not None:
                    trained_feats = [str(c) for c in list(mdl.feature_names_in_)]
            except Exception:
                pass

        if not trained_feats:
            QMessageBox.warning(self, "Validace", "Model nemá uložené featury (meta ani feature_names_in_).")
            return

        live_cols = [str(c) for c in list(live_df.columns)]
        missing = [c for c in trained_feats if c not in live_cols]
        extra = [c for c in live_cols if c not in trained_feats]

        if missing or extra:
            msg = []
            if missing:
                msg.append("Chybí (expected -> live není): " + ", ".join(missing[:20]) + ("..." if len(missing) > 20 else ""))
            if extra:
                msg.append("Navíc (live -> trénink nezná): " + ", ".join(extra[:20]) + ("..." if len(extra) > 20 else ""))
            QMessageBox.critical(self, "Featury nesedí", "\n".join(msg))
            return

        if live_cols != trained_feats:
            diffs = []
            for i, (a, b) in enumerate(zip(trained_feats, live_cols)):
                if a != b:
                    diffs.append(f"{i}: expected='{a}' vs live='{b}'")
                if len(diffs) >= 10:
                    break
            QMessageBox.warning(
                self,
                "Pořadí featur",
                "Sloupce sedí, ale pořadí je jiné.\n" + "\n".join(diffs),
            )
            return

        QMessageBox.information(self, "Validace", "OK - featury i pořadí sedí.")

    def _tick(self):
        if not self.chk_auto.isChecked():
            return
        d = self._models_dir()
        if not d:
            return
        if not self._refresh_list():
            return
        best = self._pick_best(self.records)
        if best and self._should_auto_load(best):
            self._load_and_propagate_model(best)

    def _models_dir(self) -> Path | None:
        t = self.dir_edit.text().strip()
        return Path(t) if t else None

    def _refresh_list(self, *, force: bool = False) -> bool:
        d = self._models_dir()
        if not d:
            return False
        snapshot = _directory_snapshot(d)
        if not force and snapshot == self._last_snapshot:
            return False
        self._last_snapshot = snapshot
        self.records = discover_models(d)
        self._render_table()
        return True

    def _render_table(self):
        selected = self._selected_record()
        selected_path = None
        if selected is not None:
            try:
                selected_path = selected.model_path.resolve()
            except Exception:
                selected_path = selected.model_path

        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(self.records))
        for i, r in enumerate(self.records):
            self.tbl.setItem(i, 0, QTableWidgetItem(r.model_path.name))
            self.tbl.setItem(i, 1, QTableWidgetItem(r.sha1[:8] if r.sha1 else "-"))
            self.tbl.setItem(i, 2, QTableWidgetItem(r.created or ""))

            sharpe_val = r.metrics.get("sharpe")
            sharpe_txt = f"{sharpe_val:.3f}" if sharpe_val is not None else "-"
            self.tbl.setItem(i, 3, QTableWidgetItem(sharpe_txt))

            profit_val = r.metrics.get("profit_net", r.metrics.get("profit_gross"))
            profit_txt = f"{profit_val:.0f}" if profit_val is not None else "-"
            self.tbl.setItem(i, 4, QTableWidgetItem(profit_txt))

            pf_val = r.metrics.get("profit_factor") or r.metrics.get("pf")
            pf_txt = f"{pf_val:.2f}" if pf_val is not None else "-"
            self.tbl.setItem(i, 5, QTableWidgetItem(pf_txt))

            self.tbl.setItem(i, 6, QTableWidgetItem(str(r.features_n)))
            self.tbl.setItem(i, 7, QTableWidgetItem(r.top_feature))
        self.tbl.blockSignals(False)

        if selected_path is None:
            return

        for i, rec in enumerate(self.records):
            try:
                rec_path = rec.model_path.resolve()
            except Exception:
                rec_path = rec.model_path
            if rec_path == selected_path:
                self.tbl.setCurrentCell(i, 0)
                return

    def _pick_best(self, recs: list[ModelRecord]) -> ModelRecord | None:
        if not recs:
            return None
        return max(recs, key=_record_rank_key)

    def _auto_load_best(self):
        best = self._pick_best(self.records)
        if best:
            self._load_and_propagate_model(best)

    def _should_auto_load(self, rec: ModelRecord) -> bool:
        loaded = getattr(self, "loaded", None)
        if loaded is None:
            return True
        try:
            loaded_path = Path(loaded.path).resolve()
        except Exception:
            loaded_path = None
        try:
            rec_path = rec.model_path.resolve()
        except Exception:
            rec_path = rec.model_path
        if loaded_path != rec_path:
            return True
        try:
            loaded_stat = Path(loaded.path).stat()
        except OSError:
            return True
        return (
            int(loaded_stat.st_size) != int(rec.file_size)
            or int(loaded_stat.st_mtime_ns) != int(rec.file_mtime_ns)
        )

    def _load_model(self, rec: ModelRecord):
        try:
            self.loaded = load_model_with_meta(str(rec.model_path))
            classes = self.loaded.meta.get("model_classes")
            version_suffix = ""
            if getattr(self.loaded, "version_warning", None):
                model_ver = self.loaded.meta.get("sklearn_version") or "?"
                runtime_ver = "?"
                try:
                    runtime_ver = str(self.loaded.version_warning).split("runtime=")[-1].split(". Compatibility")[0]
                except Exception:
                    pass
                version_suffix = f" | sklearn={model_ver}/runtime={runtime_ver}"
            self.lbl_loaded.setText(
                f"Načten: {rec.model_path.name} | sha1={self.loaded.sha1[:8]} | třídy={classes}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Načtení selhalo", str(e))
    def _load_and_propagate_model(self, rec: ModelRecord) -> None:
        before = getattr(self, "loaded", None)
        self._load_model(rec)
        after = getattr(self, "loaded", None)
        self._refresh_loaded_label(rec)
        if after is not None and after is not before:
            self._propagate_model_path(rec.model_path)

    def _refresh_loaded_label(self, rec: ModelRecord) -> None:
        loaded = getattr(self, "loaded", None)
        if loaded is None:
            return
        classes = loaded.meta.get("model_classes")
        version_suffix = ""
        if getattr(loaded, "version_warning", None):
            model_ver = loaded.meta.get("sklearn_version") or "?"
            runtime_ver = "?"
            try:
                runtime_ver = str(loaded.version_warning).split("runtime=")[-1].split(". Compatibility")[0]
            except Exception:
                pass
            version_suffix = f" | sklearn={model_ver}/runtime={runtime_ver}"
        self.lbl_loaded.setText(
            f"Nacten: {rec.model_path.name} | sha1={loaded.sha1[:8]} | tridy={classes}{version_suffix}"
        )

    def _propagate_model_path(self, model_path: Path) -> None:
        win = self.window()
        if win is None:
            return

        try:
            ensure_tab_loaded = getattr(win, "_ensure_tab_loaded", None)
            if callable(ensure_tab_loaded):
                ensure_tab_loaded(3)
                ensure_tab_loaded(4)
        except Exception:
            pass

        tab_eval = getattr(win, "tab_eval", None)
        if tab_eval is not None and hasattr(tab_eval, "set_model_path"):
            try:
                tab_eval.set_model_path(str(model_path))
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
