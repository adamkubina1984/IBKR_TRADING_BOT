import sys

from dotenv import load_dotenv
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget

load_dotenv()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IBKR Trading Bot - Aplikace")
        self.setMinimumSize(1200, 900)

        self.tabs = QTabWidget()
        self.tab_data = None
        self.tab_train = None
        self.tab_model_ranking = None
        self.tab_eval = None
        self.tab_live = None
        self.tab_model_manager = None

        self._tab_specs = [
            ("tab_data", "1) Stazeni dat z TreadingView", self._create_data_tab),
            ("tab_train", "2) Trenovani modelu", self._create_train_tab),
            ("tab_model_ranking", "3) Model Ranking", self._create_model_ranking_tab),
            ("tab_eval", "4) Kontrola modelu", self._create_eval_tab),
            ("tab_live", "5) Live trading bot", self._create_live_tab),
        ]
        self._tab_containers: list[QWidget] = []

        for _, label, _ in self._tab_specs:
            container = QWidget(self.tabs)
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self._tab_containers.append(container)
            self.tabs.addTab(container, label)

        self.tabs.currentChanged.connect(self._ensure_tab_loaded)
        self.setCentralWidget(self.tabs)
        self._ensure_tab_loaded(0)

    def _create_data_tab(self):
        from ibkr_trading_bot.gui.tab_data_download import DataDownloadTab

        return DataDownloadTab()

    def _create_train_tab(self):
        from ibkr_trading_bot.gui.tab_model_training import ModelTrainingTab

        return ModelTrainingTab()

    def _create_eval_tab(self):
        from ibkr_trading_bot.gui.tab_model_evaluation import ModelEvaluationTab

        return ModelEvaluationTab()

    def _create_live_tab(self):
        from ibkr_trading_bot.gui.tab_live_bot import LiveBotTab

        return LiveBotTab()

    def _create_model_manager_tab(self):
        from ibkr_trading_bot.gui.tab_model_manager_legacy import ModelManagerTab

        return ModelManagerTab(self)

    def _create_model_ranking_tab(self):
        from ibkr_trading_bot.gui.tab_model_ranking import ModelRankingTab

        return ModelRankingTab(self)

    def _ensure_tab_loaded(self, index: int) -> None:
        if index < 0 or index >= len(self._tab_specs):
            return
        attr_name, _, factory = self._tab_specs[index]
        if getattr(self, attr_name) is not None:
            return

        widget = factory()
        setattr(self, attr_name, widget)
        container = self._tab_containers[index]
        layout = container.layout()
        if layout is not None:
            layout.addWidget(widget)

    def get_live_features_df(self):
        try:
            if self.tab_live is None:
                self._ensure_tab_loaded(4)
            if self.tab_live is not None:
                if hasattr(self.tab_live, "export_live_features_df"):
                    return self.tab_live.export_live_features_df()
                return self.tab_live._build_features_for_all()
        except Exception:
            pass
        import pandas as pd

        return pd.DataFrame()

    def get_live_feature_df(self):
        return self.get_live_features_df()

    def closeEvent(self, event: QCloseEvent) -> None:
        for attr_name, _, _ in self._tab_specs:
            widget = getattr(self, attr_name, None)
            if widget is None:
                continue
            shutdown = getattr(widget, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
