from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]

class paths:
    @staticmethod
    def project_root() -> Path:
        return PKG_DIR.parent

    @staticmethod
    def data_raw() -> Path:
        p = PKG_DIR / "data" / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def data_processed() -> Path:
        p = PKG_DIR / "data" / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def models_dir() -> Path:
        p = PKG_DIR / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def results_dir() -> Path:
        p = PKG_DIR / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def logs_dir() -> Path:
        p = PKG_DIR / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p
