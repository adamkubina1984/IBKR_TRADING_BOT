from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from ibkr_trading_bot.core.datasource.tradingview_client import TradingViewClient
from dotenv import load_dotenv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QDate, QProcess, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ibkr_trading_bot.config.settings import paths
from ibkr_trading_bot.core.services.futures_roll_chain_service import (
    download_and_build_gc_roll_chain,
    parse_expiry_list,
    read_dataset_sidecar_meta,
    update_gc_roll_chain_latest_contract,
)
from ibkr_trading_bot.core.utils.plotting import plot_candles, prepare_for_chart
from ibkr_trading_bot.gui.components.log_console import LogConsole
from ibkr_trading_bot.gui.components.workers import TaskWorker

load_dotenv()

RAW_DIR = str(paths.data_raw())
PROCESSED_DIR = str(paths.data_processed())
LOG_DIR = str(paths.logs_dir())
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_log_file(prefix: str, content: str) -> str:
    fname = f"{prefix}_{_now_stamp()}.log"
    fpath = os.path.join(LOG_DIR, fname)
    try:
        with open(fpath, "w", encoding="utf-8", errors="replace") as f:
            f.write(content or "")
    except Exception:
        import tempfile

        fpath = os.path.join(tempfile.gettempdir(), fname)
        with open(fpath, "w", encoding="utf-8", errors="replace") as f:
            f.write(content or "")
    return fpath


@dataclass
class DownloadTaskPayload:
    operation: str
    chart_df: pd.DataFrame | None
    status_text: str
    status_ok: bool
    output_path: str | None = None
    auto_plot: bool = False


class DataDownloadTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.df: pd.DataFrame | None = None
        self._proc: QProcess | None = None
        self._proc_variants: list[list[str]] = []
        self._proc_idx: int = 0
        self._buf_out: str = ""
        self._buf_err: str = ""
        self._log_queue: deque[str] = deque()
        self._log_timer = QTimer(self)
        self._log_timer.setInterval(100)
        self._log_timer.timeout.connect(self._flush_log_queue)
        self._task_worker: TaskWorker | None = None
        self._retired_task_workers: list[TaskWorker] = []
        self.tv_client: TradingViewClient | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        box_ibkr = QGroupBox("IBKR Stahovani - Od data do Ted (5000-bar batchu)")
        lay_ibkr = QVBoxLayout()
        lay_ibkr.setContentsMargins(10, 8, 10, 8)
        lay_ibkr.setSpacing(8)

        def _add_inline_field(row_layout: QHBoxLayout, label_text: str, widget: QWidget) -> None:
            pair = QHBoxLayout()
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(4)
            pair.addWidget(QLabel(label_text))
            pair.addWidget(widget)
            row_layout.addLayout(pair)

        lay_ibkr_row1 = QHBoxLayout()
        lay_ibkr_row1.setContentsMargins(0, 0, 0, 0)
        lay_ibkr_row1.setSpacing(10)
        self.dt_start = QDateEdit()
        self.dt_start.setCalendarPopup(True)
        self.dt_start.setDisplayFormat("dd.MM.yyyy")
        self.dt_start.setDate(QDate.currentDate().addDays(-30))
        self.dt_start.setFixedWidth(130)
        _add_inline_field(lay_ibkr_row1, "Start datum:", self.dt_start)

        self.ed_ibkr_symbol = QLineEdit("GC")
        self.ed_ibkr_symbol.setFixedWidth(80)
        _add_inline_field(lay_ibkr_row1, "Symbol:", self.ed_ibkr_symbol)

        self.cmb_ibkr_mode = QComboBox()
        self.cmb_ibkr_mode.addItems(["CONT", "FUT"])
        self.cmb_ibkr_mode.setCurrentText("FUT")
        self.cmb_ibkr_mode.setFixedWidth(90)
        _add_inline_field(lay_ibkr_row1, "Mode:", self.cmb_ibkr_mode)

        self.ed_ibkr_expiry = QLineEdit("202602")
        self.ed_ibkr_expiry.setPlaceholderText("202602 nebo 202504,202506,202508")
        self.ed_ibkr_expiry.setFixedWidth(220)
        _add_inline_field(lay_ibkr_row1, "Expiry:", self.ed_ibkr_expiry)
        lay_ibkr_row1.addStretch(1)

        lay_ibkr.addLayout(lay_ibkr_row1)

        lay_ibkr_row2 = QHBoxLayout()
        lay_ibkr_row2.addWidget(QLabel("Bar Size:"))
        self.cmb_ibkr_bar = QComboBox()
        self.cmb_ibkr_bar.addItems(["5 mins", "15 mins", "30 mins", "1 hour"])
        self.cmb_ibkr_bar.setCurrentText("5 mins")
        lay_ibkr_row2.addWidget(self.cmb_ibkr_bar)

        self.btn_ibkr_dl = QPushButton("Stahnout z IBKR")
        self.btn_ibkr_dl.clicked.connect(self.on_download_ibkr)
        self.btn_ibkr_update = QPushButton("Aktualizovat existujici CSV")
        self.btn_ibkr_update.clicked.connect(self.on_update_ibkr_csv)
        lay_ibkr_row2.addStretch(1)
        lay_ibkr_row2.addWidget(self.btn_ibkr_update)
        lay_ibkr_row2.addWidget(self.btn_ibkr_dl)

        lay_ibkr.addLayout(lay_ibkr_row2)
        box_ibkr.setLayout(lay_ibkr)
        root.addWidget(box_ibkr)

        box_dl = QGroupBox("Stazeni dat z TradingView")
        lay_dl = QHBoxLayout()
        lay_dl.setContentsMargins(10, 8, 10, 8)
        lay_dl.setSpacing(8)

        lbl_since = QLabel("Od:")
        self.dt_since = QDateEdit()
        self.dt_since.setCalendarPopup(True)
        self.dt_since.setDisplayFormat("dd.MM.yyyy")
        self.dt_since.setDate(QDate.currentDate().addDays(-30))
        lbl_until = QLabel("Do:")
        self.dt_until = QDateEdit()
        self.dt_until.setCalendarPopup(True)
        self.dt_until.setDisplayFormat("dd.MM.yyyy")
        self.dt_until.setDate(QDate.currentDate())

        self.ed_symbol = QLineEdit("GOLD")
        self.ed_symbol.setFixedWidth(80)
        self.ed_expiry = QLineEdit("TVC")
        self.ed_expiry.setFixedWidth(90)

        self.cmb_bars = QComboBox()
        self.cmb_bars.addItems(["5 min", "15 min", "30 min", "1 hour"])
        self.cmb_bars.setCurrentText("1 hour")

        self.btn_download = QPushButton("Stahnout poslednich 5 000 baru")
        self.btn_download.clicked.connect(self.on_download_tv)
        self.btn_stop = QPushButton("Zastavit")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop_download)

        lay_dl.addWidget(lbl_since)
        lay_dl.addWidget(self.dt_since)
        lay_dl.addWidget(lbl_until)
        lay_dl.addWidget(self.dt_until)
        lay_dl.addSpacing(6)
        lay_dl.addWidget(QLabel("Symbol:"))
        lay_dl.addWidget(self.ed_symbol)
        lay_dl.addWidget(QLabel("Exchange:"))
        lay_dl.addWidget(self.ed_expiry)
        lay_dl.addWidget(QLabel("Svicka:"))
        lay_dl.addWidget(self.cmb_bars)
        lay_dl.addStretch(1)
        lay_dl.addWidget(self.btn_stop)
        lay_dl.addWidget(self.btn_download)
        box_dl.setLayout(lay_dl)
        root.addWidget(box_dl)

        box_view = QGroupBox("Zobrazeni dat")
        lay_view = QVBoxLayout()
        lay_view.setContentsMargins(10, 8, 10, 8)
        self.lbl_status = QLabel("")
        self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay_view.addWidget(self.lbl_status)

        row_view = QHBoxLayout()
        row_view.addStretch(1)
        self.btn_load = QPushButton("Nacist CSV...")
        self.btn_load.clicked.connect(self.on_load_csv)
        row_view.addWidget(self.btn_load)
        lay_view.addLayout(row_view)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        lay_view.addWidget(sep)

        self.fig = Figure(figsize=(9, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay_view.addWidget(self.canvas, 1)
        self.ax = self.fig.add_subplot(111)
        self._style_axes_empty()
        box_view.setLayout(lay_view)
        root.addWidget(box_view, 1)

        box_log = QGroupBox("Log zprav")
        lay_log = QVBoxLayout()
        self.log_view = LogConsole()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("Hlasky stahovani/nacitani...")
        lay_log.addWidget(self.log_view)
        box_log.setLayout(lay_log)
        root.addWidget(box_log)

    def _set_status(self, text: str, ok: bool | None = None) -> None:
        prefix = "⟳" if ok is None else ("✅" if ok else "❌")
        self.lbl_status.setText(f"{prefix} {text}")

    def log_msg(self, text: str) -> None:
        if not hasattr(self, "log_view") or self.log_view is None:
            return
        self._log_queue.append(str(text))
        if not self._log_timer.isActive():
            self._log_timer.start()

    def _lock_buttons(self, locked: bool) -> None:
        self.btn_download.setEnabled(not locked if False else not locked)
        self.btn_stop.setEnabled(locked)
        self.btn_load.setEnabled(not locked)
        if hasattr(self, "btn_ibkr_dl"):
            self.btn_ibkr_dl.setEnabled(not locked)
        if hasattr(self, "btn_ibkr_update"):
            self.btn_ibkr_update.setEnabled(not locked)

    def _style_axes_empty(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#111111")
        self.fig.patch.set_facecolor("#111111")
        self.ax.grid(color="#333333", linestyle=":", linewidth=0.6)
        self.ax.set_title("Svičkovy graf", color="#e0e0e0", pad=10, fontsize=11)
        self.ax.tick_params(axis="x", colors="#cccccc", labelsize=9)
        self.ax.tick_params(axis="y", colors="#cccccc", labelsize=9)
        self.canvas.draw_idle()

    def _new_tv_client(self) -> TradingViewClient:
        return TradingViewClient(username=os.getenv("TV_USERNAME"), password=os.getenv("TV_PASSWORD"))

    def _track_retired_task_worker(self, worker: TaskWorker) -> None:
        if worker in self._retired_task_workers:
            return
        self._retired_task_workers.append(worker)

        def _cleanup_retired_worker() -> None:
            try:
                self._retired_task_workers.remove(worker)
            except ValueError:
                pass
            try:
                worker.deleteLater()
            except Exception:
                pass

        worker.finished.connect(_cleanup_retired_worker)

    def _stop_task_worker(self, *, wait_ms: int = 1500, allow_background: bool = False) -> bool:
        worker = self._task_worker
        if worker is None:
            return True
        self._task_worker = None
        try:
            worker.progress_text.disconnect(self.log_msg)
        except Exception:
            pass
        try:
            worker.finished.disconnect(self._on_task_finished)
        except Exception:
            pass
        try:
            worker.stop()
        except Exception:
            pass
        if worker.isRunning() and not worker.wait(wait_ms):
            if allow_background:
                self._track_retired_task_worker(worker)
                return False
            self._task_worker = worker
            return False
        try:
            worker.deleteLater()
        except Exception:
            pass
        return True

    def _start_task(self, worker: TaskWorker) -> None:
        self._stop_task_worker(wait_ms=250, allow_background=True)
        self._task_worker = worker
        worker.progress_text.connect(self.log_msg)
        worker.finished.connect(self._on_task_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_task_finished(self) -> None:
        self._task_worker = None
        self._flush_log_queue(force=True)

    def _apply_download_payload(self, payload: DownloadTaskPayload) -> None:
        if payload.auto_plot and payload.chart_df is not None:
            self.df = payload.chart_df
            self._plot_candles(payload.chart_df)
        self._set_status(payload.status_text, ok=payload.status_ok)

    def _task_load_csv(self, path: str) -> DownloadTaskPayload:
        df = pd.read_csv(path)
        df_chart = self._prepare_for_chart(df)
        return DownloadTaskPayload(
            operation="load_csv",
            chart_df=df_chart,
            status_text=f"Nacteno {len(df_chart)} radku z {os.path.basename(path)}",
            status_ok=True,
            output_path=path,
            auto_plot=True,
        )

    def _task_download_tv(self, symbol: str, exchange: str, tf_label: str, limit: int) -> DownloadTaskPayload:
        df = self._new_tv_client().get_history(
            symbol=symbol,
            exchange=exchange,
            timeframe_label=tf_label,
            limit=limit,
        )
        if df is None or df.empty:
            raise ValueError("TradingView vratil prazdna data.")

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])
        df_out = df.rename(columns={"time": "date"})
        df_out["average"] = df_out[["open", "high", "low", "close"]].mean(axis=1).astype(float)
        df_out["barCount"] = 0
        cols = ["date", "open", "high", "low", "close", "volume", "average", "barCount"]
        df_out = df_out[cols].sort_values("date").reset_index(drop=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"tv_{symbol.replace('!','')}_{exchange}_{tf_label.replace(' ','')}_{ts}.csv"
        fpath = os.path.join(RAW_DIR, fname)
        try:
            os.makedirs(RAW_DIR, exist_ok=True)
            df_out.to_csv(fpath, index=False)
        except Exception as e:
            pass

        return DownloadTaskPayload(
            operation="download_tv",
            chart_df=None,
            status_text=(
                f"Stazeno {len(df_out)} radku do {Path(fpath).name}. "
                "Graf se vykresli az po kliknuti na Nacist CSV."
            ),
            status_ok=True,
            output_path=fpath,
        )

    def _task_download_ibkr_legacy(
        self,
        symbol: str,
        start_date: datetime,
        mode: str,
        expiry: str | None,
        bar_size: str,
        progress_cb=None,
    ) -> DownloadTaskPayload:
        expiries = parse_expiry_list(expiry)
        if str(symbol).upper() == "GC" and str(mode).upper() == "FUT" and len(expiries) > 1:
            result = download_and_build_gc_roll_chain(
                start_date=start_date,
                end_date=datetime.now(),
                expiries=expiries,
                bar_size=bar_size,
                output_dir=PROCESSED_DIR,
                raw_dir=RAW_DIR,
                progress_cb=progress_cb,
            )
            df_chart = self._prepare_for_chart(result["chart_df"])
            status_text = str(result["status_text"]) + f" | file={Path(str(result['csv_path'])).name}"
            return DownloadTaskPayload(
                operation="download_ibkr_roll_chain",
                chart_df=df_chart,
                status_text=status_text,
                status_ok=bool(result.get("quality_gate_passed", False)),
            )

        from ibkr_trading_bot.utils.download_ibkr_data import download_ibkr_by_date_range

        output_path = download_ibkr_by_date_range(
            symbol=symbol,
            start_date=start_date,
            end_date=datetime.now(),
            bar_size=bar_size,
            contract_mode=mode,
            expiry=expiry,
            output_dir=RAW_DIR,
            max_bars_per_batch=5000,
            on_progress=lambda bn, tb, rec: progress_cb(f"[IBKR] Batch {bn}: {rec} barů") if progress_cb else None,
        )

        df = pd.read_csv(output_path)
        df["date"] = pd.to_datetime(df["date"])
        df_plot = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df_chart = self._prepare_for_chart(df_plot)
        return DownloadTaskPayload(
            operation="download_ibkr",
            chart_df=df_chart,
            status_text=f"OK: {len(df)} barů ze souboru {Path(output_path).name}",
            status_ok=True,
        )

    def _task_update_ibkr_csv_legacy(
        self,
        csv_path: str,
        mode: str,
        expiry: str | None,
        progress_cb=None,
    ) -> DownloadTaskPayload:
        roll_meta = self._load_canonical_roll_chain_meta(csv_path)
        if roll_meta is not None:
            return self._task_update_roll_chain_csv(csv_path, roll_meta, progress_cb=progress_cb)
        if self._looks_like_roll_chain_csv(csv_path):
            raise ValueError(
                "Vybrany roll-chain CSV nema validni canonical _meta.json. "
                "Aktualizace obycejnym raw merge by nebyla bezpecna."
            )

        existing = self._read_ohlc_csv_strict(csv_path)
        file_symbol, file_bar_size = self._infer_symbol_bar_from_filename(csv_path)
        bar_min = self._bar_size_to_minutes(file_bar_size)
        self._validate_ohlc_integrity(existing, bar_min, "Puvodni dataset")

        last_ts = pd.to_datetime(existing["date"]).max().to_pydatetime()
        overlap_bars = 200
        fetch_start = last_ts - timedelta(minutes=bar_min * overlap_bars)
        if progress_cb:
            progress_cb(
                f"[IBKR][UPDATE] Symbol={file_symbol} TF={file_bar_size} "
                f"last_ts={last_ts} fetch_start={fetch_start} overlap_bars={overlap_bars}"
            )

        if mode == "FUT" and not expiry:
            raise ValueError("Pro FUT rezim je nutna expirace.")

        from ibkr_trading_bot.utils.download_ibkr_data import download_ibkr_by_date_range

        downloaded_path = download_ibkr_by_date_range(
            symbol=file_symbol,
            start_date=fetch_start,
            end_date=datetime.now(),
            bar_size=file_bar_size,
            contract_mode=mode,
            expiry=expiry,
            output_dir=RAW_DIR,
            max_bars_per_batch=5000,
            on_progress=lambda bn, tb, rec: progress_cb(f"[IBKR][UPDATE] Batch {bn}: {rec} barů") if progress_cb else None,
        )
        if progress_cb:
            progress_cb(f"[IBKR][UPDATE] Dočasně staženo: {downloaded_path}")

        incoming = self._read_ohlc_csv_strict(downloaded_path)
        self._validate_ohlc_integrity(incoming, bar_min, "Nove stazena data")

        merged = pd.concat([existing, incoming], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        self._validate_ohlc_integrity(merged, bar_min, "Slouceny dataset")

        old_n = int(len(existing))
        new_n = int(len(merged))
        added = int(max(0, new_n - old_n))
        new_max = pd.to_datetime(merged["date"]).max().to_pydatetime()
        old_max = pd.to_datetime(existing["date"]).max().to_pydatetime()

        bar_tag = file_bar_size.replace(" mins", "m").replace(" min", "m").replace(" hour", "h").replace(" ", "")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_tag = pd.to_datetime(merged["date"]).min().strftime("%Y%m%d")
        end_tag = pd.to_datetime(merged["date"]).max().strftime("%Y%m%d")
        out_name = f"{file_symbol}_{bar_tag}_{len(merged)}bars_{start_tag}_{end_tag}_{ts}.csv"
        out_path = Path(RAW_DIR) / out_name

        out_cols = ["date", "open", "high", "low", "close", "volume"]
        for opt in ("average", "barCount"):
            if opt in merged.columns:
                out_cols.append(opt)
        merged[out_cols].to_csv(out_path, index=False)

        df_plot = merged[["date", "open", "high", "low", "close", "volume"]].copy()
        df_chart = self._prepare_for_chart(df_plot)
        if progress_cb:
            progress_cb(
                f"[IBKR][UPDATE] OK: old={old_n} new={new_n} added={added} "
                f"old_max={old_max} new_max={new_max}"
            )
            progress_cb(f"[IBKR][UPDATE] Ulozeno: {out_path}")

        status_text = (
            "Update dokoncen, ale bez novejsich timestampu (jen prepsany overlap)."
            if new_max <= old_max
            else f"Update OK: +{added} radku, posledni bar {new_max}."
        )
        return DownloadTaskPayload(
            operation="update_ibkr_csv",
            chart_df=df_chart,
            status_text=status_text,
            status_ok=True,
        )

    def on_load_csv(self) -> None:
        start_dir = PROCESSED_DIR if os.path.isdir(PROCESSED_DIR) else RAW_DIR
        path, _ = QFileDialog.getOpenFileName(self, "Vyber CSV s daty (OHLC)", start_dir, "CSV (*.csv)")
        if not path:
            return
        self._set_status(f"Nacitam data z {path}...", ok=None)
        self.log_msg(f"[LOAD] Soubor: {path}")
        self._lock_buttons(True)

        worker = TaskWorker(self._task_load_csv, path)
        worker.result.connect(self._on_load_csv_success)
        worker.error.connect(self._on_load_csv_error)
        self._start_task(worker)

    def _on_load_csv_success(self, payload: DownloadTaskPayload) -> None:
        self._apply_download_payload(payload)
        self.log_msg(f"[LOAD] Nacteno {len(payload.chart_df)} radku.")
        self._lock_buttons(False)

    def _on_load_csv_error(self, message: str) -> None:
        self._style_axes_empty()
        self._set_status(f"Chyba nacteni: {message}", ok=False)
        self.log_msg(f"[LOAD][ERROR] {message}")
        self._lock_buttons(False)

    def on_download_tv(self) -> None:
        self._lock_buttons(True)
        self._set_status("Stahuji data z TradingView...", ok=None)
        symbol = (self.ed_symbol.text() or "GOLD").strip()
        exchange = (self.ed_expiry.text() or "TVC").strip()
        tf_label = self.cmb_bars.currentText().replace("mins", "min")
        limit = 5000

        worker = TaskWorker(self._task_download_tv, symbol, exchange, tf_label, limit)
        worker.result.connect(self._on_download_tv_success)
        worker.error.connect(self._on_download_tv_error)
        self._start_task(worker)

    def _on_download_tv_success(self, payload: DownloadTaskPayload) -> None:
        self._apply_download_payload(payload)
        self.log_msg(f"[TV] OK: {payload.status_text}")
        self._lock_buttons(False)

    def _on_download_tv_error(self, message: str) -> None:
        self._style_axes_empty()
        self._set_status(f"Chyba TV: {message}", ok=False)
        self.log_msg(f"[TV][ERROR] {message}")
        self._lock_buttons(False)

    def on_stop_download(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc = None
            self._set_status("Stahovani zruseno.", ok=False)
            self._lock_buttons(False)

    def on_download_ibkr(self) -> None:
        start_date_q = self.dt_start.date()
        start_date = datetime.combine(
            datetime(start_date_q.year(), start_date_q.month(), start_date_q.day()).date(),
            datetime.min.time(),
        )

        symbol = (self.ed_ibkr_symbol.text() or "GC").strip()
        mode = self.cmb_ibkr_mode.currentText()
        expiry = (self.ed_ibkr_expiry.text() or "").strip() if mode == "FUT" else None
        bar_size = self.cmb_ibkr_bar.currentText()
        expiries = parse_expiry_list(expiry)

        if mode == "FUT" and not expiry:
            self._set_status("Chyba: Expiry je povinna pro FUT mod", ok=False)
            return

        if mode == "FUT" and str(symbol).upper() == "GC" and len(expiries) > 1:
            self._set_status(f"Stahuji GC expirace a skladam roll-chain od {start_date.date()}...", ok=None)
            self.log_msg(
                f"[ROLL] Start: {start_date.date()} | Symbol: {symbol} | Expiries: {','.join(expiries)} | Bar: {bar_size}"
            )
        else:
            self._set_status(f"Stahuji {symbol} z IBKR od {start_date.date()}...", ok=None)
            self.log_msg(f"[IBKR] Start: {start_date.date()} | Symbol: {symbol} | Mode: {mode} | Bar: {bar_size}")
        self._lock_buttons(True)

        worker = TaskWorker(self._task_download_ibkr, symbol, start_date, mode, expiry, bar_size)
        worker.result.connect(self._on_download_ibkr_success)
        worker.error.connect(self._on_download_ibkr_error)
        self._start_task(worker)

    def _on_download_ibkr_success(self, payload: DownloadTaskPayload) -> None:
        self._apply_download_payload(payload)
        self.log_msg(f"[IBKR] ✅ Hotovo: {payload.status_text}")
        self._lock_buttons(False)

    def _on_download_ibkr_error(self, message: str) -> None:
        self._style_axes_empty()
        self._set_status(f"Chyba IBKR: {message}", ok=False)
        self.log_msg(f"[IBKR][ERROR] {message}")
        self._lock_buttons(False)

    def _on_download_ibkr_success(self, payload: DownloadTaskPayload) -> None:
        self._apply_download_payload(payload)
        if payload.operation == "download_ibkr_roll_chain":
            prefix = "[ROLL] OK:" if payload.status_ok else "[ROLL][WARN]"
            self.log_msg(f"{prefix} {payload.status_text}")
        else:
            self.log_msg(f"[IBKR] OK: {payload.status_text}")
        self._lock_buttons(False)

    def _bar_size_to_minutes(self, bar_size: str) -> int:
        s = str(bar_size or "").strip().lower()
        if s in {"5 mins", "5 min", "5m"}:
            return 5
        if s in {"15 mins", "15 min", "15m"}:
            return 15
        if s in {"30 mins", "30 min", "30m"}:
            return 30
        if s in {"1 hour", "1h", "60 mins"}:
            return 60
        return 5

    def _infer_symbol_bar_from_filename(self, path: str) -> tuple[str, str]:
        name = Path(path).name
        m = re.match(r"^([A-Za-z0-9]+)_(?:\d{6}_)?([0-9]+m|[0-9]+h|[0-9]+d)_.+\.csv$", name)
        symbol = (self.ed_ibkr_symbol.text() or "GC").strip().upper()
        bar_size = self.cmb_ibkr_bar.currentText()
        if not m:
            return symbol, bar_size

        symbol = m.group(1).upper()
        tf_code = m.group(2).lower()
        tf_map = {
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
        }
        return symbol, tf_map.get(tf_code, bar_size)

    def _looks_like_roll_chain_csv(self, path: str) -> bool:
        return "rollchain" in Path(path).stem.lower()

    def _load_canonical_roll_chain_meta(self, path: str) -> dict[str, object] | None:
        meta = read_dataset_sidecar_meta(path)
        if not isinstance(meta, dict) or not meta:
            return None
        if str(meta.get("dataset_kind") or "").strip().lower() != "gc_roll_chain":
            return None
        if meta.get("canonical") is not True:
            raise ValueError("Vybrany roll-chain dataset neni oznacen jako canonical.")
        return meta

    def _roll_chain_expiries_from_meta(self, meta: dict[str, object]) -> list[str]:
        raw_items = meta.get("expiries_used") or []
        items = raw_items if isinstance(raw_items, (list, tuple)) else parse_expiry_list(str(raw_items))
        expiries: list[str] = []
        seen: set[str] = set()
        for item in items:
            expiry = str(item).strip()
            if not re.fullmatch(r"\d{6}", expiry) or expiry in seen:
                continue
            seen.add(expiry)
            expiries.append(expiry)
        return expiries

    def _roll_chain_contract_paths_from_meta(self, meta: dict[str, object]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        raw_items = meta.get("source_contracts") or []
        if not isinstance(raw_items, list):
            return mapping
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            expiry = str(item.get("expiry") or "").strip()
            csv_path = str(item.get("csv_path") or "").strip()
            if re.fullmatch(r"\d{6}", expiry) and csv_path:
                mapping[expiry] = csv_path
        return mapping

    def _task_update_roll_chain_csv_legacy(
        self,
        csv_path: str,
        meta: dict[str, object],
        progress_cb=None,
    ) -> DownloadTaskPayload:
        instrument = str(meta.get("instrument") or "GC").strip().upper()
        if instrument != "GC":
            raise ValueError("Update canonical roll-chain datasetu je zatim podporen jen pro GC.")

        expiries = self._roll_chain_expiries_from_meta(meta)
        if len(expiries) < 2:
            raise ValueError("Canonical roll-chain meta neobsahuje alespon 2 validni expirace.")

        bar_size = str(meta.get("bar_size") or "").strip()
        if not bar_size:
            _, bar_size = self._infer_symbol_bar_from_filename(csv_path)

        existing = self._read_ohlc_csv_strict(csv_path)
        bar_min = self._bar_size_to_minutes(bar_size)
        self._validate_ohlc_integrity(existing, bar_min, "Puvodni roll-chain dataset")
        start_date = pd.to_datetime(existing["date"]).min().to_pydatetime()
        output_dir = str(Path(csv_path).expanduser().resolve().parent)

        if progress_cb:
            progress_cb(
                f"[ROLL][UPDATE] Rebuild {Path(csv_path).name} | "
                f"start={start_date:%Y-%m-%d} | expiries={','.join(expiries)} | bar={bar_size}"
            )

        result = download_and_build_gc_roll_chain(
            start_date=start_date,
            end_date=datetime.now(),
            expiries=expiries,
            bar_size=bar_size,
            output_dir=output_dir,
            raw_dir=RAW_DIR,
            progress_cb=progress_cb,
        )
        df_chart = self._prepare_for_chart(result["chart_df"])
        status_text = str(result["status_text"]) + f" | file={Path(str(result['csv_path'])).name}"
        return DownloadTaskPayload(
            operation="update_ibkr_roll_chain",
            chart_df=df_chart,
            status_text=status_text,
            status_ok=bool(result.get("quality_gate_passed", False)),
        )

    def _task_download_ibkr(
        self,
        symbol: str,
        start_date: datetime,
        mode: str,
        expiry: str | None,
        bar_size: str,
        progress_cb=None,
    ) -> DownloadTaskPayload:
        expiries = parse_expiry_list(expiry)
        if str(symbol).upper() == "GC" and str(mode).upper() == "FUT" and len(expiries) > 1:
            result = download_and_build_gc_roll_chain(
                start_date=start_date,
                end_date=datetime.now(),
                expiries=expiries,
                bar_size=bar_size,
                output_dir=PROCESSED_DIR,
                raw_dir=RAW_DIR,
                progress_cb=progress_cb,
            )
            status_text = str(result["status_text"]) + f" | file={Path(str(result['csv_path'])).name}"
            return DownloadTaskPayload(
                operation="download_ibkr_roll_chain",
                chart_df=None,
                status_text=status_text + " | Pro vykresleni klikni na Nacist CSV.",
                status_ok=bool(result.get("quality_gate_passed", False)),
                output_path=str(result["csv_path"]),
            )

        from ibkr_trading_bot.utils.download_ibkr_data import download_ibkr_by_date_range

        output_path = download_ibkr_by_date_range(
            symbol=symbol,
            start_date=start_date,
            end_date=datetime.now(),
            bar_size=bar_size,
            contract_mode=mode,
            expiry=expiry,
            output_dir=RAW_DIR,
            max_bars_per_batch=5000,
            on_progress=lambda bn, tb, rec: progress_cb(f"[IBKR] Batch {bn}: {rec} baru") if progress_cb else None,
        )
        return DownloadTaskPayload(
            operation="download_ibkr",
            chart_df=None,
            status_text=f"OK: ulozeno do {Path(output_path).name}. Pro vykresleni klikni na Nacist CSV.",
            status_ok=True,
            output_path=str(output_path),
        )

    def _task_update_ibkr_csv(
        self,
        csv_path: str,
        mode: str,
        expiry: str | None,
        progress_cb=None,
    ) -> DownloadTaskPayload:
        roll_meta = self._load_canonical_roll_chain_meta(csv_path)
        if roll_meta is not None:
            return self._task_update_roll_chain_csv(csv_path, roll_meta, progress_cb=progress_cb)
        if self._looks_like_roll_chain_csv(csv_path):
            raise ValueError(
                "Vybrany roll-chain CSV nema validni canonical _meta.json. "
                "Aktualizace obycejnym raw merge by nebyla bezpecna."
            )

        existing = self._read_ohlc_csv_strict(csv_path)
        file_symbol, file_bar_size = self._infer_symbol_bar_from_filename(csv_path)
        bar_min = self._bar_size_to_minutes(file_bar_size)
        self._validate_ohlc_integrity(existing, bar_min, "Puvodni dataset")

        last_ts = pd.to_datetime(existing["date"]).max().to_pydatetime()
        overlap_bars = 200
        fetch_start = last_ts - timedelta(minutes=bar_min * overlap_bars)
        if progress_cb:
            progress_cb(
                f"[IBKR][UPDATE] Symbol={file_symbol} TF={file_bar_size} "
                f"last_ts={last_ts} fetch_start={fetch_start} overlap_bars={overlap_bars}"
            )

        if mode == "FUT" and not expiry:
            raise ValueError("Pro FUT rezim je nutna expirace.")

        from ibkr_trading_bot.utils.download_ibkr_data import download_ibkr_by_date_range

        downloaded_path = download_ibkr_by_date_range(
            symbol=file_symbol,
            start_date=fetch_start,
            end_date=datetime.now(),
            bar_size=file_bar_size,
            contract_mode=mode,
            expiry=expiry,
            output_dir=RAW_DIR,
            max_bars_per_batch=5000,
            on_progress=lambda bn, tb, rec: progress_cb(f"[IBKR][UPDATE] Batch {bn}: {rec} baru") if progress_cb else None,
        )
        if progress_cb:
            progress_cb(f"[IBKR][UPDATE] Docasne stazeno: {downloaded_path}")

        incoming = self._read_ohlc_csv_strict(downloaded_path)
        self._validate_ohlc_integrity(incoming, bar_min, "Nove stazena data")

        merged = pd.concat([existing, incoming], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        self._validate_ohlc_integrity(merged, bar_min, "Slouceny dataset")

        old_n = int(len(existing))
        new_n = int(len(merged))
        added = int(max(0, new_n - old_n))
        new_max = pd.to_datetime(merged["date"]).max().to_pydatetime()
        old_max = pd.to_datetime(existing["date"]).max().to_pydatetime()

        bar_tag = file_bar_size.replace(" mins", "m").replace(" min", "m").replace(" hour", "h").replace(" ", "")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_tag = pd.to_datetime(merged["date"]).min().strftime("%Y%m%d")
        end_tag = pd.to_datetime(merged["date"]).max().strftime("%Y%m%d")
        out_name = f"{file_symbol}_{bar_tag}_{len(merged)}bars_{start_tag}_{end_tag}_{ts}.csv"
        out_path = Path(RAW_DIR) / out_name

        out_cols = ["date", "open", "high", "low", "close", "volume"]
        for opt in ("average", "barCount"):
            if opt in merged.columns:
                out_cols.append(opt)
        merged[out_cols].to_csv(out_path, index=False)

        if progress_cb:
            progress_cb(
                f"[IBKR][UPDATE] OK: old={old_n} new={new_n} added={added} "
                f"old_max={old_max} new_max={new_max}"
            )
            progress_cb(f"[IBKR][UPDATE] Ulozeno: {out_path}")

        status_text = (
            "Update dokoncen, ale bez novejsich timestampu (jen prepsany overlap)."
            if new_max <= old_max
            else f"Update OK: +{added} radku, posledni bar {new_max}."
        )
        return DownloadTaskPayload(
            operation="update_ibkr_csv",
            chart_df=None,
            status_text=f"{status_text} | file={out_path.name} | Pro vykresleni klikni na Nacist CSV.",
            status_ok=True,
            output_path=str(out_path),
        )

    def _task_update_roll_chain_csv(
        self,
        csv_path: str,
        meta: dict[str, object],
        progress_cb=None,
    ) -> DownloadTaskPayload:
        instrument = str(meta.get("instrument") or "GC").strip().upper()
        if instrument != "GC":
            raise ValueError("Update canonical roll-chain datasetu je zatim podporen jen pro GC.")

        expiries = self._roll_chain_expiries_from_meta(meta)
        if len(expiries) < 2:
            raise ValueError("Canonical roll-chain meta neobsahuje alespon 2 validni expirace.")

        bar_size = str(meta.get("bar_size") or "").strip()
        if not bar_size:
            _, bar_size = self._infer_symbol_bar_from_filename(csv_path)

        existing = self._read_ohlc_csv_strict(csv_path)
        bar_min = self._bar_size_to_minutes(bar_size)
        self._validate_ohlc_integrity(existing, bar_min, "Puvodni roll-chain dataset")
        start_date = pd.to_datetime(existing["date"]).min().to_pydatetime()
        output_dir = str(Path(csv_path).expanduser().resolve().parent)
        preferred_contract_paths = self._roll_chain_contract_paths_from_meta(meta)

        if progress_cb:
            progress_cb(
                f"[ROLL][UPDATE] Rebuild {Path(csv_path).name} | "
                f"start={start_date:%Y-%m-%d} | expiries={','.join(expiries)} | "
                f"bar={bar_size} | update_mode=latest_only"
            )

        result = update_gc_roll_chain_latest_contract(
            start_date=start_date,
            end_date=datetime.now(),
            expiries=expiries,
            bar_size=bar_size,
            output_dir=output_dir,
            raw_dir=RAW_DIR,
            preferred_contract_paths=preferred_contract_paths,
            progress_cb=progress_cb,
        )
        status_text = str(result["status_text"]) + f" | file={Path(str(result['csv_path'])).name}"
        return DownloadTaskPayload(
            operation="update_ibkr_roll_chain",
            chart_df=None,
            status_text=status_text + " | Pro vykresleni klikni na Nacist CSV.",
            status_ok=bool(result.get("quality_gate_passed", False)),
            output_path=str(result["csv_path"]),
        )

    def _read_ohlc_csv_strict(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV je prazdne.")

        time_col = None
        for c in ("date", "time", "datetime", "timestamp"):
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            raise ValueError("CSV nema casovy sloupec (date/time/datetime/timestamp).")

        out = df.copy()
        out["date"] = pd.to_datetime(out[time_col], errors="coerce", utc=True).dt.tz_localize(None)
        req = ["open", "high", "low", "close"]
        for c in req:
            if c not in out.columns:
                raise ValueError(f"CSV nema povinny sloupec '{c}'.")
            out[c] = pd.to_numeric(out[c], errors="coerce")
        if "volume" in out.columns:
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
        else:
            out["volume"] = 0.0

        out = out.dropna(subset=["date", "open", "high", "low", "close"]).copy()
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if out.empty:
            raise ValueError("Po ocisteni nezustala zadna validni OHLC data.")
        return out

    def _validate_ohlc_integrity(self, df: pd.DataFrame, expected_step_min: int, label: str) -> None:
        if df.empty:
            raise ValueError(f"{label}: prazdna data.")
        bad_hl = int((df["high"] < df["low"]).sum())
        bad_oc = int(
            (
                (df["open"] < df["low"])
                | (df["open"] > df["high"])
                | (df["close"] < df["low"])
                | (df["close"] > df["high"])
            ).sum()
        )
        if bad_hl > 0 or bad_oc > 0:
            raise ValueError(f"{label}: nevalidni OHLC (high<low={bad_hl}, open/close out-of-range={bad_oc}).")

        dt_min = df["date"].diff().dropna().dt.total_seconds().div(60.0)
        if not dt_min.empty:
            med = float(dt_min.median())
            if med <= 0:
                raise ValueError(f"{label}: nevalidni casovy krok (median={med}).")
            if abs(med - float(expected_step_min)) > max(1.0, float(expected_step_min) * 0.6):
                raise ValueError(
                    f"{label}: nesedi timeframe (median step {med:.2f} min vs ocek. {expected_step_min} min)."
                )

    def on_update_ibkr_csv(self) -> None:
        if os.path.isdir(PROCESSED_DIR):
            start_dir = PROCESSED_DIR
        elif os.path.isdir(RAW_DIR):
            start_dir = RAW_DIR
        else:
            start_dir = str(Path.home())
        csv_path, _ = QFileDialog.getOpenFileName(self, "Vyber existujici CSV pro update", start_dir, "CSV (*.csv)")
        if not csv_path:
            return

        mode = self.cmb_ibkr_mode.currentText()
        expiry = (self.ed_ibkr_expiry.text() or "").strip() if mode == "FUT" else None

        self._lock_buttons(True)
        self._set_status("Aktualizuji existujici dataset...", ok=None)
        self.log_msg(f"[IBKR][UPDATE] Zdrojovy CSV: {csv_path}")

        worker = TaskWorker(self._task_update_ibkr_csv, csv_path, mode, expiry)
        worker.result.connect(self._on_update_ibkr_success)
        worker.error.connect(self._on_update_ibkr_error)
        self._start_task(worker)

    def _on_update_ibkr_success(self, payload: DownloadTaskPayload) -> None:
        self._apply_download_payload(payload)
        if payload.operation == "update_ibkr_roll_chain":
            prefix = "[ROLL][UPDATE] OK:" if payload.status_ok else "[ROLL][UPDATE][WARN]"
            self.log_msg(f"{prefix} {payload.status_text}")
        else:
            self.log_msg(f"[IBKR][UPDATE] {payload.status_text}")
        self._lock_buttons(False)

    def _on_update_ibkr_error(self, message: str) -> None:
        self._style_axes_empty()
        self._set_status(f"Chyba update CSV: {message}", ok=False)
        self.log_msg(f"[IBKR][UPDATE][ERROR] {message}")
        self._lock_buttons(False)

    def _flush_log_queue(self, force: bool = False) -> None:
        if not self._log_queue:
            if self._log_timer.isActive():
                self._log_timer.stop()
            return
        chunk: list[str] = []
        max_lines = 200 if not force else len(self._log_queue)
        while self._log_queue and len(chunk) < max_lines:
            chunk.append(self._log_queue.popleft())
        if chunk:
            self.log_view.append_line("\n".join(chunk))
            cur = self.log_view.textCursor()
            cur.movePosition(QTextCursor.End)
            self.log_view.setTextCursor(cur)
            self.log_view.ensureCursorVisible()
        if not self._log_queue and self._log_timer.isActive():
            self._log_timer.stop()

    def _prepare_for_chart(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_for_chart(df)

    def _plot_candles(self, df: pd.DataFrame) -> None:
        return plot_candles(self.fig, self.ax, self.canvas, df)

    def shutdown(self) -> bool:
        if self._log_timer.isActive():
            self._log_timer.stop()
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.waitForFinished(1000)
            except Exception:
                pass
            self._proc = None
        return self._stop_task_worker(wait_ms=2000, allow_background=False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.shutdown():
            QMessageBox.warning(
                self,
                "Stahovani dat",
                "Probihajici operace se jeste neukoncila. Pockej na dokonceni nebo ji zastav a zkus zavreni znovu.",
            )
            event.ignore()
            return
        super().closeEvent(event)
