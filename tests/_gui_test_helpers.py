import json
from pathlib import Path

import joblib
import numpy as np
from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit, QVBoxLayout, QWidget

from ibkr_trading_bot.core.services import model_eval_service as model_eval_runtime


class DummyTernaryPredictor:
    def __init__(self):
        self.classes_ = np.array([0, 1, 2])
        self.feature_names_in_ = np.array(["feat_a", "feat_b"], dtype=object)

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        arr = np.zeros((len(X), 3), dtype=float)
        arr[:, 1] = 1.0
        return arr


class StubCanvas(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def draw_idle(self):
        pass


class StubLiveTab(QWidget):
    def __init__(self):
        super().__init__()
        self.le_model_path = QLineEdit(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.le_model_path)

    def set_model_paths(self, paths):
        if isinstance(paths, str):
            items = [paths]
        else:
            items = [str(path) for path in paths]
        self.le_model_path.setText(";".join(items))


class StubTrainTab(QWidget):
    def __init__(self, csv_path: str | None = None, *, n_total_bars: int = 44268):
        super().__init__()
        self.csv_path = csv_path
        self.dataset = [0] * int(n_total_bars) if csv_path else None
        self.holdout_pct_default = 0.10
        self.holdout_min_bars_default = 1000
        self.holdout_max_bars_default = 6000

    def _current_candidate_top_n(self):
        return 5

    def _current_candidate_fresh_ratio(self):
        return 0.30


class StubEvalRankingTab(QWidget):
    def __init__(self, csv_path: str, *, fee_per_trade: float = 0.25):
        super().__init__()
        self._context = {
            "data_path": csv_path,
            "fee_per_trade": fee_per_trade,
            "entry_threshold": 0.55,
            "exit_threshold": 0.60,
        }

    def current_ranking_context(self):
        return dict(self._context)


class StubEvalTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model_path = None
        self.et_spin = QDoubleSpinBox(self)
        self.ext_spin = QDoubleSpinBox(self)
        self.eval_calls = 0
        layout = QVBoxLayout(self)
        layout.addWidget(self.et_spin)
        layout.addWidget(self.ext_spin)

    def set_model_path(self, path: str) -> None:
        self.model_path = path

    def on_evaluate_clicked(self) -> None:
        self.eval_calls += 1


def write_model(tmp_path: Path, name: str, *, ranking=None, profit_net: float = 1.0) -> Path:
    model_path = tmp_path / name
    joblib.dump(DummyTernaryPredictor(), model_path)
    meta = {
        "created_at": "2026-03-06T12:00:00",
        "created_at_iso": "2026-03-06T12:00:00",
        "trained_features": ["feat_a", "feat_b"],
        "classes": [0, 1, 2],
        "ternary_threshold_short": 0.40,
        "ternary_threshold_long": 0.60,
        "metrics_holdout": {"profit_net": profit_net},
    }
    if ranking is not None:
        model_eval_runtime.set_tab5_holdout_ranking(
            meta,
            ranking,
            exit_policy=str(ranking.get("exit_policy") or "hold_until_opposite"),
        )
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path


def ranking_payload(csv_path: Path, *, fee: float, profit_h: float, trades_h: float) -> dict[str, object]:
    return model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=fee,
        exit_policy="hold_until_opposite",
        entry_threshold=0.55,
        exit_threshold=0.60,
        metrics={"profit_net": profit_h, "max_dd": -10.0, "trades": trades_h},
        status="ok",
    )