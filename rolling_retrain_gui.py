# rolling_retrain_gui.py (rozšířená verze)
# GUI pro rolling retrain, vizualizaci výsledků, porovnání modelů, export nejlepšího modelu a spuštění online simulace

import os
import sys

import joblib
import pandas as pd
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Přidání základní online simulace
from sklearn.metrics import classification_report

from ibkr_trading_bot.features.feature_engineering import prepare_dataset_with_targets
from ibkr_trading_bot.model.train_models import train_and_evaluate_model
from ibkr_trading_bot.utils.metrics import calculate_metrics


def run_online_simulation_with_model(model_path):
    model = joblib.load(model_path)
    print(f"✅ Model načten ze souboru: {model_path}")

    # Načti poslední CSV soubor z data/raw
    def load_latest_csv(path="data/raw"):
        files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError("Žádný CSV soubor v data/raw")
        latest = max(files, key=lambda f: os.path.getmtime(os.path.join(path, f)))
        return pd.read_csv(os.path.join(path, latest), parse_dates=["date"])

    df = load_latest_csv()
    df = df.rename(columns={"date": "datetime"})
    df = df.sort_values("datetime").reset_index(drop=True)

    prepared = prepare_dataset_with_targets(df)
    if prepared.empty:
        print("⚠️ Připravený dataset je prázdný.\n")
        return

    X = prepared.drop(columns=["target"]).select_dtypes(include=["number"])
    y = prepared["target"]
    preds = model.predict(X)

    print("\n📊 Výsledky online simulace:\n")
    print(classification_report(y, preds))

class RollingRetrainGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rolling Retrain – IBKR Trading Bot")
        self.setMinimumSize(1000, 750)

        layout = QVBoxLayout()

        self.file_label = QLabel("Vybraný soubor: žádný")
        self.select_file_button = QPushButton("Vybrat CSV soubor")
        self.select_file_button.clicked.connect(self.select_file)

        param_layout = QHBoxLayout()
        self.train_input = QLineEdit("5")
        self.test_input = QLineEdit("1")

        self.model_select = QComboBox()
        self.model_select.addItems(["Random Forest", "LightGBM", "XGBoost", "Všechny"])

        param_layout.addWidget(QLabel("Trénovací okno (dny):"))
        param_layout.addWidget(self.train_input)
        param_layout.addWidget(QLabel("Testovací okno (dny):"))
        param_layout.addWidget(self.test_input)
        param_layout.addWidget(QLabel("Model:"))
        param_layout.addWidget(self.model_select)

        self.run_button = QPushButton("Spustit rolling retrain")
        self.run_button.clicked.connect(self.run_rolling_retrain)

        self.simulate_button = QPushButton("Spustit online simulaci s nejlepším modelem")
        self.simulate_button.clicked.connect(self.run_simulation)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.table = QTableWidget()

        layout.addWidget(self.file_label)
        layout.addWidget(self.select_file_button)
        layout.addLayout(param_layout)
        layout.addWidget(self.run_button)
        layout.addWidget(self.simulate_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.table)

        self.setLayout(layout)
        self.csv_path = None
        self.best_model_path = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Vybrat CSV soubor", "data/raw", "*.csv")
        if file_path:
            self.csv_path = file_path
            self.file_label.setText(f"Vybraný soubor: {os.path.basename(file_path)}")

    def run_rolling_retrain(self):
        if not self.csv_path:
            self.file_label.setText("⚠️ Nejprve vyber soubor!")
            return

        train_days = int(self.train_input.text())
        test_days = int(self.test_input.text())

        df = pd.read_csv(self.csv_path, parse_dates=["date"])
        df = df.rename(columns={"date": "datetime"})
        df = df.sort_values("datetime").reset_index(drop=True)

        df_per_day = df.groupby(df["datetime"].dt.date)
        dates = sorted(df_per_day.groups.keys())

        model_map = {
            "Random Forest": "rf",
            "LightGBM": "lgbm",
            "XGBoost": "xgb"
        }
        selected_option = self.model_select.currentText()
        model_names = list(model_map.values()) if selected_option == "Všechny" else [model_map.get(selected_option, "lgbm")]

        all_results = {}
        best_avg_f1 = -1
        best_model = None
        best_model_name = ""

        for model_name in model_names:
            results = []
            num_iterations = len(dates) - train_days - test_days
            self.progress_bar.setMaximum(num_iterations)

            for i in range(num_iterations):
                train_start = dates[i]
                train_end = dates[i + train_days - 1]
                test_start = dates[i + train_days]
                test_end = dates[i + train_days + test_days - 1]

                train_df = df[(df["datetime"].dt.date >= train_start) & (df["datetime"].dt.date <= train_end)].copy()
                test_df = df[(df["datetime"].dt.date >= test_start) & (df["datetime"].dt.date <= test_end)].copy()

                train_prepared = prepare_dataset_with_targets(train_df)
                test_prepared = prepare_dataset_with_targets(test_df)

                if train_prepared.empty or test_prepared.empty:
                    continue

                X_train = train_prepared.drop(columns=["target"]).select_dtypes(include=["number"])
                y_train = train_prepared["target"]

                model, _ = train_and_evaluate_model(X_train, y_train, model_name, {})

                X_test = test_prepared.drop(columns=["target"]).select_dtypes(include=["number"])
                y_test = test_prepared["target"]
                preds = model.predict(X_test)

                metrics = calculate_metrics(y_test, preds)
                results.append({
                    "Train Start": train_start,
                    "Train End": train_end,
                    "Test Start": test_start,
                    "Test End": test_end,
                    **metrics
                })

                self.progress_bar.setValue(i + 1)

            all_results[model_name] = results

            avg_f1 = sum(r["F1"] for r in results) / len(results) if results else 0
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                best_model = model
                best_model_name = model_name

        if best_model:
            os.makedirs("model_outputs", exist_ok=True)
            self.best_model_path = f"model_outputs/{best_model_name}_best_model.pkl"
            joblib.dump(best_model, self.best_model_path)

        self.show_results(all_results)

        for model_name, result in all_results.items():
            df_export = pd.DataFrame(result)
            os.makedirs("logs", exist_ok=True)
            df_export.to_csv(f"logs/rolling_results_{model_name}.csv", index=False)

    def show_results(self, all_results):
        if not all_results:
            return

        model_names = list(all_results.keys())
        results = all_results[model_names[0]]

        keys = list(results[0].keys())
        self.table.setColumnCount(len(keys))
        self.table.setRowCount(len(results))
        self.table.setHorizontalHeaderLabels(keys)

        for row_idx, row_data in enumerate(results):
            for col_idx, key in enumerate(keys):
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(row_data[key])))

    def run_simulation(self):
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            self.file_label.setText("❌ Nejprve spusť rolling retrain a ulož nejlepší model!")
            return

        run_online_simulation_with_model(self.best_model_path)
        self.file_label.setText("✅ Online simulace dokončena.")

# Spuštění aplikace
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RollingRetrainGUI()
    window.show()
    sys.exit(app.exec())
