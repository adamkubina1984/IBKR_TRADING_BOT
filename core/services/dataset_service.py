from typing import Literal

import pandas as pd

from ibkr_trading_bot.features.feature_engineering import prepare_dataset_with_targets
from ibkr_trading_bot.labels import make_triple_barrier_labels
from ibkr_trading_bot.utils.io_helpers import load_dataframe


class DatasetService:
    """
    Připraví dataset pro trénink 1:1 s logikou GUI s možností volby labelingu.

    Kroky:
      1) načte CSV
      2) normalizuje čas (UTC), seřadí
      3) zavolá prepare_dataset_with_targets(raw)
      4) doplní timestamp
      5) vytvoří / konsoliduje 'target' podle zvoleného režimu (původní nebo triple-barrier)
      6) binarizuje target do {0,1}
      7) volitelně vyčistí řádky s NaN ve featurách

    Pozn.: Tato služba NEMÁ míchat timestamp do featur (to řeší tréninková část).
    """

    def prepare_from_csv(
        self,
        path: str,
        labeling: Literal["prepared", "triple_barrier"] = "prepared",
        horizon: int = 12,
        take_profit_bps: float = 60.0,
        stop_loss_bps: float = 40.0,
        fee_per_trade: float = 0.0,
        slippage_bps: float = 0.0,
        dropna_minimal: bool = True,
        min_non_na_ratio: float = 1.0,
        ensure_unique_timestamp: bool = True,
    ) -> pd.DataFrame:

        # --- 1) načtení
        raw = load_dataframe(path)

        # --- 2) timestamp normalizace (UTC) + sort
        time_col = self._detect_time_col(raw)
        if time_col is None:
            raise ValueError(
                "CSV musí obsahovat časový sloupec 'date' nebo 'datetime' nebo 'timestamp'."
            )

        raw["timestamp"] = pd.to_datetime(raw[time_col], errors="coerce", utc=True)
        raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if ensure_unique_timestamp:
            # Případné duplikáty času (např. spoje z různých zdrojů) – vezmeme poslední záznam.
            raw = raw.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

        # --- 3) základní příprava featur + (možná) target z prepare_*
        prepared = prepare_dataset_with_targets(raw)

        df_prepared, y_series = self._normalize_prepare_output(prepared)

        # --- 4) zajisti timestamp ve výstupu
        if "timestamp" not in df_prepared.columns:
            df_prepared["timestamp"] = raw["timestamp"].values

        # --- 5) labeling: buď respektuj target z prepare_* (prepared),
        #         nebo vygeneruj triple-barrier (přepíše existující y/target)
        if labeling == "triple_barrier":
            tb = make_triple_barrier_labels(
                df=raw if set(["close"]).issubset(raw.columns) else df_prepared,
                horizon=horizon,
                take_profit_bps=take_profit_bps,
                stop_loss_bps=stop_loss_bps,
                fee_per_trade=fee_per_trade,
                slippage_bps=slippage_bps,
            )
            df_prepared["target"] = tb.values
        else:
            # Vezmi target z prepare_* pokud existuje; případně z y_series
            if "target" not in df_prepared.columns:
                if y_series is None:
                    raise ValueError("Po přípravě chybí sloupec 'target' a nebyl vrácen ani 'y'.")
                df_prepared["target"] = pd.Series(y_series, index=df_prepared.index)

        # --- 6) final binarizace do {0,1}
        # Připouští původní {-1,0,1} i spojité hodnoty – kladné -> 1, jinak 0
        df_prepared["target"] = pd.Series(df_prepared["target"]).astype(float)
        df_prepared["target"] = (df_prepared["target"] > 0).astype(int)

        if df_prepared["target"].dropna().empty:
            raise ValueError("Po přípravě chybí platné hodnoty pro 'target'.")

        # --- 7) volitelné čištění NaN ve featurách (mimo timestamp/target)
        feature_cols = [c for c in df_prepared.columns if c not in ("timestamp", "target")]
        if dropna_minimal and feature_cols:
            if min_non_na_ratio >= 1.0:
                df_prepared = df_prepared.dropna(subset=feature_cols)
            else:
                # Povolit řádky s alespoň daným poměrem ne-NaN hodnot ve featurách
                ok = (df_prepared[feature_cols].notna().sum(axis=1) >= min_non_na_ratio * len(feature_cols))
                df_prepared = df_prepared.loc[ok]

        # Finální seřazení podle timestamp pro jistotu
        df_prepared = df_prepared.sort_values("timestamp").reset_index(drop=True)

        # --- sanity diagnostiky (nevyhazují chybu; jen pro log/print, když používáš GUI log)
        self._sanity_log(df_prepared, feature_cols)

        return df_prepared

    # ---------- helpers ----------

    @staticmethod
    def _detect_time_col(df: pd.DataFrame) -> str | None:
        for cand in ("timestamp", "datetime", "date"):
            if cand in df.columns:
                return cand
        return None

    @staticmethod
    def _normalize_prepare_output(prepared):
        """
        Akceptuje:
          - DataFrame
          - (X, y)
          - {"X": df, "y": y} nebo jiné aliasy
        Vrací (df_prepared, y_series|None)
        """
        df_prepared = None
        y_series = None

        if isinstance(prepared, pd.DataFrame):
            df_prepared = prepared.copy()

        elif isinstance(prepared, (tuple, list)):
            if len(prepared) >= 1 and isinstance(prepared[0], pd.DataFrame):
                df_prepared = prepared[0].copy()
            if len(prepared) >= 2 and prepared[1] is not None:
                y_series = pd.Series(prepared[1])

        elif isinstance(prepared, dict):
            Xcand = prepared.get("X") or prepared.get("features") or prepared.get("data") or prepared.get("df")
            if isinstance(Xcand, pd.DataFrame):
                df_prepared = Xcand.copy()
            y_series = prepared.get("y") or prepared.get("target")

        if df_prepared is None:
            raise ValueError("prepare_dataset_with_targets() nevrátil DataFrame ani (X, y).")

        return df_prepared, y_series

    @staticmethod
    def _sanity_log(df: pd.DataFrame, feature_cols: list[str]) -> None:
        # Tohle si můžeš přepojit na tvůj logger
        n = len(df)
        n_pos = int(df["target"].sum())
        n_neg = int(n - n_pos)
        msg = (
            f"[DatasetService] rows={n}  pos={n_pos}  neg={n_neg}  "
            f"pos_ratio={n_pos/max(n,1):.3f}  features={len(feature_cols)}"
        )
        try:
            print(msg)
        except Exception:
            pass
