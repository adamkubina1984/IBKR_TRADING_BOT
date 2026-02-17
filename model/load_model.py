# model/load_model.py
"""
Načítání nejlepšího modelu z disku pro použití v online simulaci.
"""

import os

import joblib


def load_latest_model(model_dir="model_outputs") -> object:
    """
    Načte naposledy uložený model podle času vytvoření.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Složka {model_dir} neexistuje")

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("Nebyl nalezen žádný modelový soubor .pkl")

    # Seřazení podle data poslední úpravy
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    latest_model_path = os.path.join(model_dir, model_files[0])

    model = joblib.load(latest_model_path)
    return model
