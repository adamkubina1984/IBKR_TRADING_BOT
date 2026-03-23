# ibkr_trading_bot/main.py
"""
Hlavní rozcestník pro celý IBKR trading projekt (CLI).
Podporované příkazy:
  - gui        : spustí desktopové GUI (PySide6)
  - download   : stažení historických dat z IBKR
  - generate   : výpočet featur z historických (nebo syntetických) dat
      └─ --generate-synthetic : vygeneruje syntetická OHLC a hned nad nimi spočítá featury
  - train      : rychlý trénink modelu nad features.csv
  - evaluate   : vyhodnocení uloženého modelu
  - select-best: výběr nejlepšího modelu z results.csv

Poznámky:
- Importy jsou balíčkové (ibkr_trading_bot.*), aby fungovalo `python -m ibkr_trading_bot.main`.
- Cesty k datům a výstupům řešíme relativně k tomuto souboru, takže funguje spouštění odkudkoli.
"""

import argparse
import os
from pathlib import Path

from ibkr_trading_bot.features.feature_engineering import main as generate_features_main

# ✅ Balíčkové importy
from ibkr_trading_bot.utils.download_ibkr_data import download_data


def main():
    parser = argparse.ArgumentParser(description="IBKR Trading Bot CLI")
    parser.add_argument("--quiet", action="store_true", help="Méně výpisů na stdout")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Příkaz")

    # === GUI ===
    subparsers.add_parser("gui", help="Spustí desktopové GUI")

    # === GENERATE FEATURES ===
    generate_parser = subparsers.add_parser(
        "generate", help="Vygeneruje featury z historických (nebo syntetických) dat"
    )
    generate_parser.add_argument(
        "--calibrate-from", type=str, default=None,
        help="CSV s reálnými OHLCV (timestamp, open, high, low, close, volume) pro kalibraci syntetiky."
    )
    generate_parser.add_argument(
        "--generate-synthetic", action="store_true",
        help="Vygeneruje syntetická OHLC data a rovnou na nich spočítá featury."
    )
    generate_parser.add_argument("--n-samples", type=int, default=5000, help="Počet syntetických svíček")
    generate_parser.add_argument("--noise-level", type=float, default=0.05, help="Úroveň šumu syntetiky")
    generate_parser.add_argument("--input", type=str, default="data/raw/ohlc_data.csv", help="Cesta k CSV s OHLC (timestamp, open, high, low, close, volume)")

    # === TRAIN ===
    train_parser = subparsers.add_parser("train", help="Natrénuje jednoduchý model nad features.csv")
    train_parser.add_argument("--features", type=str, default=None,
                              help="Cesta k CSV s featurami (default: data/processed/features.csv)")
    train_parser.add_argument("--model-out", type=str, default=None,
                              help="Cesta k uloženému modelu (default: model_outputs/model.joblib)")

    # === EVALUATE ===
    eval_parser = subparsers.add_parser("evaluate", help="Vyhodnotí model a zapíše results.csv")
    eval_parser.add_argument("--features", type=str, default=None, help="Cesta k CSV s featurami")
    eval_parser.add_argument("--model", type=str, required=True, help="Cesta k uloženému modelu (*.joblib)")
    eval_parser.add_argument("--results-out", type=str, default=None,
                             help="Kam zapsat results.csv (default: results/results.csv)")
    eval_parser.add_argument("--holdout-only", action="store_true",
                             help="Vyhodnotí jen poslední holdout segment (tail).")
    eval_parser.add_argument("--holdout-bars", type=int, default=None,
                             help="Počet posledních barů pro holdout evaluaci (přepíše metadata modelu).")

    # === SELECT-BEST ===
    select_parser = subparsers.add_parser(
        "select-best", help="Vybere nejlepší model (primárně podle profit, sekundárně F1)"
    )
    select_parser.add_argument("--results", type=str, required=True, help="Cesta k results.csv")
    select_parser.add_argument("--min-trades", type=int, default=20, help="Min. počet obchodů pro výběr")

    # === DOWNLOAD DATA ===
    download_parser = subparsers.add_parser("download", help="Stažení historických dat z IBKR")
    download_parser.add_argument('--symbol', type=str, required=True, help='Např. GC')
    download_parser.add_argument('--days', type=int, default=30, help='Počet dnů zpět')
    download_parser.add_argument('--barSize', type=str, default='5 mins', help='Granularita, např. "5 mins"')
    download_parser.add_argument('--output', type=str, default='data/raw', help='Cílová složka pro CSV')
    download_parser.add_argument('--expiry', type=str, required=True, help='Expirace kontraktu, např. 202509')

    args = parser.parse_args()

    # Základní adresář balíčku (cesty relativně k němu)
    BASE_DIR = Path(__file__).resolve().parent

    # --- Subpříkazy ---
    if args.command == "gui":
        try:
            from PySide6.QtWidgets import QApplication

            from ibkr_trading_bot.gui.main_window import MainWindow
        except Exception as e:
            print("❌ GUI nelze spustit. Zkontroluj instalaci PySide6 a modul gui/main_window.py")
            print(f"Detail chyby: {e}")
            return

        # potlačení informačních logů FFmpeg (volitelné)
        os.environ["QT_LOGGING_RULES"] = "qt.multimedia.ffmpeg.debug=false;qt.multimedia.ffmpeg.info=false"

        import sys
        app = QApplication(sys.argv)
        w = MainWindow()
        w.show()
        sys.exit(app.exec())

    if args.command == "generate":
        if args.generate_synthetic:
            print(f"🧪 Generuji syntetická data: {args.n_samples} vzorků, šum {args.noise_level}...")
            # Odložený import, pokud modul není potřeba jinde
            import pandas as pd

            from ibkr_trading_bot.data.generate_synthetic import generate_synthetic_data

            syn_dir = (BASE_DIR / "data" / "synthetic")
            syn_dir.mkdir(parents=True, exist_ok=True)
            synthetic_csv = syn_dir / "synthetic_dataset.csv"

            # Generuj a ulož
            # Načti kalibrační data (pokud jsou zadána)
            df_base = None
            if args.calibrate_from:
                base_path = (BASE_DIR / args.calibrate_from) if not Path(args.calibrate_from).is_absolute() else Path(args.calibrate_from)
                if not base_path.exists():
                    raise FileNotFoundError(f"CSV pro kalibraci neexistuje: {base_path}")
                df_base = pd.read_csv(base_path.as_posix(), parse_dates=["timestamp"])
            required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
            missing = required_cols.difference(set(map(str.lower, df_base.columns)))
            if missing:
                raise ValueError(f"Kalibrační CSV postrádá sloupce: {sorted(missing)}")

            synth = generate_synthetic_data(
                df_base=df_base,
                n_samples=args.n_samples,
                noise_level=args.noise_level,
                bar_minutes=5,
                seed=42,
                calibrate=True,
            )
            synth.to_csv(synthetic_csv.as_posix(), index=False)
            print(f"✅ Syntetická data uložena do: {synthetic_csv}")

            print("🛠 Spouštím generování featur ze syntetických dat…")
            generate_features_main(input_path=synthetic_csv.as_posix())
            return

        # Standard: featury nad existujícím CSV
        input_path = (BASE_DIR / args.input) if not Path(args.input).is_absolute() else Path(args.input)
        if not input_path.exists():
            print(f"❌ Vstupní CSV s OHLC neexistuje: {input_path}")
            print("💡 Tip: Můžeš spustit syntetiku: `python -m ibkr_trading_bot.main generate --generate-synthetic`")
            return
        print(f"🛠 Spouštím generování featur z: {input_path}")
        generate_features_main(input_path=input_path.as_posix())
        return

    if args.command == "download":
        allowed = {"1 min","2 mins","3 mins","5 mins","10 mins","15 mins","30 mins","1 hour","4 hours","1 day"}
        if args.barSize not in allowed:
            print(f"⚠️  Neobvyklá hodnota --barSize: '{args.barSize}'. Běžné hodnoty: {sorted(allowed)}")
        print(f"⬇️  Stahuji data pro {args.symbol} s expirací {args.expiry}…")
        output_dir = (BASE_DIR / args.output) if not Path(args.output).is_absolute() else Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        download_data(
            symbol=args.symbol,
            expiry=args.expiry,
            days_back=args.days,
            bar_size=args.barSize,
            output_dir=output_dir.as_posix(),
        )
        print(f"✅ Data stažena do složky: {output_dir.as_posix()}")
        return

    if args.command == "train":
        features_csv = Path(args.features) if args.features else (BASE_DIR / "data" / "processed" / "features.csv")
        model_out = Path(args.model_out) if args.model_out else (BASE_DIR / "model_outputs" / "model.joblib")
        print(f"ℹ️  features_csv = {features_csv.as_posix()}")
        print(f"ℹ️  model_out   = {model_out.as_posix()}")

        if not features_csv.exists():
            raise FileNotFoundError(f"Soubor s featurami neexistuje: {features_csv}")

        import pandas as pd
        _df_head = pd.read_csv(features_csv.as_posix(), nrows=1)
        if _df_head.shape[1] < 2:
            raise ValueError(f"CSV s featurami vypadá prázdné nebo bez featur: {features_csv}")

        print(f"🏋️  Trénuji model z: {features_csv}")
        from ibkr_trading_bot.model.train_models import train_simple_model
        model_out.parent.mkdir(parents=True, exist_ok=True)
        model_path = train_simple_model(features_csv=features_csv.as_posix(), model_out=model_out.as_posix())
        print(f"✅ Model uložen: {model_path}")
        return

    if args.command == "evaluate":
        model_path = Path(args.model)

        if not model_path.exists():
            raise FileNotFoundError(f"Uložený model neexistuje: {model_path}")

        features_csv = Path(args.features) if args.features else (BASE_DIR / "data" / "processed" / "features.csv")
        results_out = Path(args.results_out) if args.results_out else (BASE_DIR / "results" / "results.csv")

        if not features_csv.exists():
            raise FileNotFoundError(f"Soubor s featurami neexistuje: {features_csv}")

        print(f"📊 Vyhodnocuji model {args.model} na {features_csv}")
        from ibkr_trading_bot.model.evaluate_models import evaluate_model_once
        results_out.parent.mkdir(parents=True, exist_ok=True)
        evaluate_model_once(
            features_csv=features_csv.as_posix(),
            model_path=args.model,
            results_out=results_out.as_posix(),
            holdout_only=bool(args.holdout_only),
            holdout_bars=args.holdout_bars,
        )
        print(f"✅ Výsledky uloženy do: {results_out}")
        return

    if args.command == "select-best":
        print(f"🔎 Vybírám nejlepší model z: {args.results}")
        from ibkr_trading_bot.model.selection import select_best_model
        try:
            best = select_best_model(results_csv=args.results, min_trades=args.min_trades)
        except Exception as e:
            print(f"❌ Nepodařilo se vybrat nejlepší model z '{args.results}'.")
            print(f"Detail chyby: {e}")
            return

        if best is None:
            print("⚠️  Nebyl nalezen žádný vhodný model (možná málo obchodů).")
        else:
            print(f"🏆 Nejlepší model: {best.model_path} | profit={best.profit:.2f} | f1={best.f1:.4f}")

        return


if __name__ == "__main__":
    main()
