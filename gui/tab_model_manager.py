from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
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

# Výchozí (default) složka s modely
DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "model_outputs"

def _as_float(x: Any, default: float = float("-inf")) -> float:
    """Bezpečně převeď na float (např. '0', '0.0', None, 'nan', '0,1')."""
    if x is None:
        return default
    try:
        # evropská desetinná čárka → tečka
        if isinstance(x, str):
            x2 = x.strip().replace(",", ".")
            if x2 == "":
                return default
            return float(x2)
        return float(x)
    except Exception:
        return default

def _as_timestamp(created: Any, fallback_path) -> float:
    """
    Vrátí unix timestamp z různých tvarů 'created':
    - datetime → .timestamp()
    - číslo/string → parse na float/ISO
    - jinak spadne na mtime souboru
    """
    try:
        if isinstance(created, datetime):
            return float(created.timestamp())
        if isinstance(created, (int, float)):
            return float(created)
        if isinstance(created, str) and created:
            # zkus ISO (např. 2025-10-07T11:09:22)
            try:
                return datetime.fromisoformat(created).timestamp()
            except Exception:
                # zkus čisté číslo (sekundy)
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


def sha1_file(path: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()



def discover_models(dir_path: Path) -> list[ModelRecord]:
    recs: list[ModelRecord] = []
    if not dir_path.exists():
        return recs

    for p in dir_path.glob("*.pkl"):
        sha1 = sha1_file(p)
        meta_candidates = [p.with_name(p.stem + "_meta.json"), p.parent / "model_meta.json"]
        meta = {}
        meta_path = None
        for m in meta_candidates:
            if m.exists():
                try:
                    meta = json.loads(m.read_text(encoding="utf-8"))
                    meta_path = m
                    break
                except Exception:
                    pass

        # doplnění z modelu (fallbacky na classes/features)
        try:
            mdl = joblib.load(p)
            predictor = mdl.get("model") if isinstance(mdl, dict) and "model" in mdl else mdl
            if not meta.get("model_classes"):
                if isinstance(meta.get("classes"), list):
                    meta["model_classes"] = list(meta.get("classes"))
                elif hasattr(predictor, "classes_"):
                    meta["model_classes"] = list(getattr(predictor, "classes_"))
            if not meta.get("trained_features"):
                if isinstance(meta.get("features"), list):
                    meta["trained_features"] = list(meta.get("features"))
                elif hasattr(predictor, "feature_names_in_"):
                    meta["trained_features"] = list(getattr(predictor, "feature_names_in_"))
        except Exception:
            pass

        # --- robustní parsování metrik na skaláry, s handlerováním Infinity a NaN ---
        metrics_raw = meta.get("metrics") or {}
        metrics: dict[str, float] = {}
        for k, v in metrics_raw.items():
            try:
                if isinstance(v, (int, float)):
                    if np.isfinite(v):
                        metrics[k] = float(v)
                elif isinstance(v, str):
                    v_clean = v.strip()
                    if v_clean.lower() in ("nan", "infinity", "inf", "-infinity"):
                        continue  # skip NaN, Infinity
                    metrics[k] = float(v_clean)
                elif isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (int, float)):
                    if np.isfinite(v[0]):
                        metrics[k] = float(v[0])
                # jinák ignoruj (např. listy equity, dicty s kvantily apod.)
            except (ValueError, TypeError):
                pass

        # --- čas vytvoření (string pro UI + timestamp pro třídění) ---
        created_str = str(meta.get("created_at_iso") or meta.get("created_at") or "")
        def _to_ts(s: str) -> float:
            if not s:
                return p.stat().st_mtime
            try:
                return datetime.fromisoformat(s).timestamp()
            except Exception:
                try:
                    return float(s)
                except Exception:
                    return p.stat().st_mtime

        recs.append(
            ModelRecord(
                model_path=p,
                meta_path=meta_path,
                sha1=sha1,
                created=created_str,                               # pro zobrazení
                metrics=metrics,                                   # už očištěné
                features_n=len(meta.get("trained_features", [])),
                classes=list(meta.get("model_classes", [])),
            )
        )

    # třídění: nejdřív Sharpe (desc), pak čas (desc), pak profit (desc)
    def _sort_key(r: ModelRecord):
        sharpe = r.metrics.get("sharpe", float("-inf"))
        profit = r.metrics.get("profit_net", float("-inf"))
        try:
            ts = datetime.fromisoformat(r.created).timestamp()
        except Exception:
            try:
                ts = float(r.created)
            except Exception:
                ts = r.model_path.stat().st_mtime
        # Primary: sharpe (desc), Secondary: profit (desc), Tertiary: timestamp (desc)
        return (sharpe, profit, ts)

    recs.sort(key=_sort_key, reverse=True)
    return recs



class ModelManagerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("tab_model_manager")

        self.dir_edit = QLineEdit(self)
        self.dir_edit.setText(str(DEFAULT_MODEL_DIR))  # Nainicializuj na default cestu
        self.btn_browse = QPushButton("Zvolit složku s modely…", self)
        self.chk_auto = QCheckBox("Auto-load nejnovější/best model")
        self.chk_auto.setChecked(True)

        self.tbl = QTableWidget(self)
        self.tbl.setColumnCount(8)
        self.tbl.setHorizontalHeaderLabels(["Model", "SHA1", "Vytvořen", "Sharpe", "Profit", "PF", "#Feats", "Top Feature"])
        self.tbl.horizontalHeader().setStretchLastSection(True)

        self.lbl_loaded = QLabel("Načten: –")

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

        # Stav
        self.records: list[ModelRecord] = []
        self.loaded = None  # výsledek load_model_with_meta(...)

        # Signály
        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_load.clicked.connect(self._on_load_selected)
        self.btn_validate.clicked.connect(self._on_validate)
        self.sens.valueChanged.connect(self._on_sens)

        # Periodický refresh pro auto-load
        self.timer = QTimer(self)
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        # Načti modely z default cesty a auto-load nejlepší
        self._refresh_list()
        if self.chk_auto.isChecked():
            self._auto_load_best()

    # ---------- UI callbacks ----------
    def _on_browse(self):
        # Defaultně otevři dialog v DEFAULT_MODEL_DIR
        start_dir = str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR.exists() else str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Zvol složku s modely", start_dir)
        if d:
            self.dir_edit.setText(d)
            self._refresh_list()
            if self.chk_auto.isChecked():
                self._auto_load_best()

    def _on_sens(self, v: float):
        # tady jen držíme číslo; předat do live tabu můžeš přes MainWindow, pokud chceš
        pass

    def _on_load_selected(self):
        row = self.tbl.currentRow()
        if row < 0 or row >= len(self.records):
            QMessageBox.warning(self, "Model", "Nejprve vyber model v tabulce.")
            return
        self._load_model(self.records[row])

    def _on_validate(self):
        """
        Porovná featury očekávané modelem (trained_features / feature_names_in_)
        s live featurami z Tab 4. Vypíše konkrétní rozdíly (missing/extra/pořadí).
        """
        import joblib
        from PySide6.QtWidgets import QApplication, QMessageBox

        # 1) Musí být načtený model
        if not getattr(self, "loaded", None):
            QMessageBox.information(self, "Validace", "Nejprve načti model v této záložce.")
            return

        # 2) Najdi callback v MainWindow – robustně (projdi parent chain + activeWindow)
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

        # 3) Získej live featury z Tab 4
        try:
            live_df = get_live()
        except Exception as e:
            QMessageBox.critical(self, "Validace", f"Získání live featur selhalo: {e}")
            return

        if live_df is None or (isinstance(live_df, pd.DataFrame) and live_df.empty):
            QMessageBox.information(self, "Validace", "Žádná live data. Otevři Tab 4 a načti snapshot.")
            return

        # 4) Očekávané featury z metadat (fallback: feature_names_in_ z modelu)
        trained_feats = list(self.loaded.meta.get("trained_features", []) or [])
        if not trained_feats:
            # fallback – zkus načíst model a vzít feature_names_in_
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

        # 5) Porovnání
        live_cols = [str(c) for c in list(live_df.columns)]
        missing = [c for c in trained_feats if c not in live_cols]
        extra   = [c for c in live_cols if c not in trained_feats]

        if missing or extra:
            msg = []
            if missing:
                msg.append("Chybí (expected → live není): " + ", ".join(missing[:20]) + ("…" if len(missing) > 20 else ""))
            if extra:
                msg.append("Navíc (live → trénink nezná): " + ", ".join(extra[:20]) + ("…" if len(extra) > 20 else ""))
            QMessageBox.critical(self, "Featury nesedí", "\n".join(msg))
            return

        # 6) Kontrola pořadí (jen když se množiny shodují)
        if live_cols != trained_feats:
            # najdi prvních pár rozdílů v pořadí
            diffs = []
            for i, (a, b) in enumerate(zip(trained_feats, live_cols)):
                if a != b:
                    diffs.append(f"{i}: expected='{a}' vs live='{b}'")
                if len(diffs) >= 10:
                    break
            QMessageBox.warning(self, "Pořadí featur",
                                "Sloupce sedí, ale pořadí je jiné.\n" + "\n".join(diffs))
            return

        # 7) Vše OK
        QMessageBox.information(self, "Validace", "OK – featury i pořadí sedí.")

    # ---------- periodic auto-load ----------
    def _tick(self):
        if not self.chk_auto.isChecked():
            return
        d = self._models_dir()
        if not d:
            return
        prev_sha = getattr(self.loaded, "sha1", None)
        self._refresh_list()
        best = self._pick_best(self.records)
        if best and best.sha1 != prev_sha:
            self._load_model(best)

    # ---------- helpers ----------
    def _models_dir(self) -> Path | None:
        t = self.dir_edit.text().strip()
        return Path(t) if t else None

    def _refresh_list(self):
        d = self._models_dir()
        if not d:
            return
        recs = discover_models(d)
        self.records = recs
        self._render_table()

    def _render_table(self):
        self.tbl.setRowCount(len(self.records))
        for i, r in enumerate(self.records):
            self.tbl.setItem(i, 0, QTableWidgetItem(r.model_path.name))
            self.tbl.setItem(i, 1, QTableWidgetItem(r.sha1[:8]))
            self.tbl.setItem(i, 2, QTableWidgetItem(r.created or ""))
            
            # Sharpe - zobraz N/A pokud chybí
            sharpe_val = r.metrics.get("sharpe")
            sharpe_txt = f"{sharpe_val:.3f}" if sharpe_val is not None else "–"
            self.tbl.setItem(i, 3, QTableWidgetItem(sharpe_txt))
            
            # Profit - zobraz N/A pokud chybí
            profit_val = r.metrics.get("profit_net")
            profit_txt = f"{profit_val:.0f}" if profit_val is not None else "–"
            self.tbl.setItem(i, 4, QTableWidgetItem(profit_txt))
            
            # Profit Factor - zobraz N/A pokud chybí
            pf_val = r.metrics.get("profit_factor") or r.metrics.get("pf")
            pf_txt = f"{pf_val:.2f}" if pf_val is not None else "–"
            self.tbl.setItem(i, 5, QTableWidgetItem(pf_txt))
            
            self.tbl.setItem(i, 6, QTableWidgetItem(str(r.features_n)))
            # Top feature z feature_importance
            top_feat = ""
            if r.model_path.exists():
                try:
                    meta_path = r.model_path.with_name(f"{r.model_path.stem}_meta.json")
                    if meta_path.exists():
                        meta_txt = meta_path.read_text(encoding="utf-8")
                        meta = json.loads(meta_txt)
                        feat_imp = meta.get("feature_importance", {})
                        if feat_imp and isinstance(feat_imp, dict):
                            # najdi feature s největší importante
                            top_feat = max(feat_imp.keys(), key=lambda x: feat_imp[x])[:12]  # první feature, zkrácený
                except Exception as e:
                    pass
            self.tbl.setItem(i, 7, QTableWidgetItem(top_feat))

    def _pick_best(self, recs: list[ModelRecord]) -> ModelRecord | None:
        if not recs:
            return None
        recs2 = sorted(
            recs,
            key=lambda r: (
                _as_float(r.metrics.get("sharpe", None)),
                _as_timestamp(r.created, r.model_path),
            ),
            reverse=True,
        )
        return recs2[0]

    def _auto_load_best(self):
        best = self._pick_best(self.records)
        if best:
            self._load_model(best)

    def _load_model(self, rec: ModelRecord):
        try:
            self.loaded = load_model_with_meta(str(rec.model_path))
            classes = self.loaded.meta.get("model_classes")
            self.lbl_loaded.setText(
                f"Načten: {rec.model_path.name} | sha1={self.loaded.sha1[:8]} | třídy={classes}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Načtení selhalo", str(e))
