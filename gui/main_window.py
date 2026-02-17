# Tento soubor je součástí projektu IBKR Trading Bot
# gui/main_window.py – zachovaný design + přidaná záložka „Model Manager“

import sys

from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from ibkr_trading_bot.gui.tab_data_download import DataDownloadTab
from ibkr_trading_bot.gui.tab_live_bot import LiveBotTab
from ibkr_trading_bot.gui.tab_model_evaluation import ModelEvaluationTab
from ibkr_trading_bot.gui.tab_model_manager import ModelManagerTab
from ibkr_trading_bot.gui.tab_model_training import ModelTrainingTab

load_dotenv()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IBKR Trading Bot – Aplikace")
        self.setMinimumSize(1200, 900)

        self.tabs = QTabWidget()

        # 1) Stažení dat
        self.tab_data = DataDownloadTab()
        self.tabs.addTab(self.tab_data, "1) Stažení dat z TreadingView")

        # 2) Trénování modelu
        self.tab_train = ModelTrainingTab()
        self.tabs.addTab(self.tab_train, "2) Trénování modelu")

        # 3) Kontrola modelu
        self.tab_eval = ModelEvaluationTab()
        self.tabs.addTab(self.tab_eval, "3) Kontrola modelu")

        # 4) Live trading bot (držíme instanci kvůli přístupu k live featurám)
        self.tab_live = LiveBotTab()
        self.tabs.addTab(self.tab_live, "4) Live trading bot")

        # 5) Model Manager – auto výběr/načtení modelu a validace featur (nová záložka)
        #    Záložka očekává, že MainWindow umí poskytnout live featury (viz metoda níže).
        self.tab_model_manager = ModelManagerTab(self)
        self.tabs.addTab(self.tab_model_manager, "5) Model Manager")

        self.setCentralWidget(self.tabs)

    # ------ Poskytnutí live featur pro Model Manager (pro tlačítko "Ověřit shodu featur") ------
    def get_live_features_df(self):
        """Vrátí featury z live tabu, zarovnané na model (pro validaci v Tab5)."""
        try:
            if hasattr(self, "tab_live") and self.tab_live is not None:
                # preferuj veřejný alias; fallback na interní builder
                if hasattr(self.tab_live, "export_live_features_df"):
                    return self.tab_live.export_live_features_df()
                return self.tab_live._build_features_for_all()
        except Exception:
            pass
        import pandas as pd
        return pd.DataFrame()

    def get_live_feature_df(self):  # alias kvůli starší hlášce
        return self.get_live_features_df()


    # starší text hlášky volal jednou i singulár bez 's' – necháme alias, ať to nikdy neselže
    def get_live_feature_df(self):
        return self.get_live_features_df()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
