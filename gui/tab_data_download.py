# gui/tab_data_download.py
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
from core.data_sources.tradingview_client import TradingViewClient

# --- Matplotlib (QtAgg) ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- PySide6 ---
from PySide6.QtCore import QDate, QProcess, Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ibkr_trading_bot.config.settings import paths

# --- Naše ---
from ibkr_trading_bot.core.utils.plotting import plot_candles, prepare_for_chart
from ibkr_trading_bot.gui.components.log_console import LogConsole

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
        self.tv_client = TradingViewClient(username=os.getenv("TV_USERNAME"), password=os.getenv("TV_PASSWORD"))
        self._log_timer.timeout.connect(self._flush_log_queue)

        root = QVBoxLayout(self); root.setContentsMargins(12, 10, 12, 10); root.setSpacing(10)

        # === IBKR STAHOVÁNÍ ===
        box_ibkr = QGroupBox("IBKR Stahování - Od data do Teď (5000-bar batchů)")
        lay_ibkr = QVBoxLayout(); lay_ibkr.setContentsMargins(10, 8, 10, 8); lay_ibkr.setSpacing(8)
        
        lay_ibkr_row1 = QHBoxLayout()
        lay_ibkr_row1.addWidget(QLabel("Start datum:"))
        self.dt_start = QDateEdit(); self.dt_start.setCalendarPopup(True)
        self.dt_start.setDisplayFormat("dd.MM.yyyy"); self.dt_start.setDate(QDate.currentDate().addDays(-30))
        lay_ibkr_row1.addWidget(self.dt_start)
        
        lay_ibkr_row1.addWidget(QLabel("Symbol:"))
        self.ed_ibkr_symbol = QLineEdit("GC"); self.ed_ibkr_symbol.setFixedWidth(80)
        lay_ibkr_row1.addWidget(self.ed_ibkr_symbol)
        
        lay_ibkr_row1.addWidget(QLabel("Mode:"))
        self.cmb_ibkr_mode = QComboBox()
        self.cmb_ibkr_mode.addItems(["CONT", "FUT"])
        lay_ibkr_row1.addWidget(self.cmb_ibkr_mode)
        
        lay_ibkr_row1.addWidget(QLabel("Expiry:"))
        self.ed_ibkr_expiry = QLineEdit("202602"); self.ed_ibkr_expiry.setFixedWidth(80)
        lay_ibkr_row1.addWidget(self.ed_ibkr_expiry)
        
        lay_ibkr.addLayout(lay_ibkr_row1)
        
        lay_ibkr_row2 = QHBoxLayout()
        lay_ibkr_row2.addWidget(QLabel("Bar Size:"))
        self.cmb_ibkr_bar = QComboBox()
        self.cmb_ibkr_bar.addItems(["5 mins", "15 mins", "30 mins", "1 hour"])
        self.cmb_ibkr_bar.setCurrentText("5 mins")
        lay_ibkr_row2.addWidget(self.cmb_ibkr_bar)
        
        self.btn_ibkr_dl = QPushButton("🔽 Stáhnout z IBKR")
        self.btn_ibkr_dl.clicked.connect(self.on_download_ibkr)
        lay_ibkr_row2.addStretch(1)
        lay_ibkr_row2.addWidget(self.btn_ibkr_dl)
        
        lay_ibkr.addLayout(lay_ibkr_row2)
        box_ibkr.setLayout(lay_ibkr); root.addWidget(box_ibkr)

        # --- Stažení z TradingView ---
        box_dl = QGroupBox("Stažení dat z TradingView")
        lay_dl = QHBoxLayout(); lay_dl.setContentsMargins(10, 8, 10, 8); lay_dl.setSpacing(8)

        lbl_since = QLabel("Od:"); self.dt_since = QDateEdit(); self.dt_since.setCalendarPopup(True)
        self.dt_since.setDisplayFormat("dd.MM.yyyy"); self.dt_since.setDate(QDate.currentDate().addDays(-30))
        lbl_until = QLabel("Do:"); self.dt_until = QDateEdit(); self.dt_until.setCalendarPopup(True)
        self.dt_until.setDisplayFormat("dd.MM.yyyy"); self.dt_until.setDate(QDate.currentDate())

        self.ed_symbol = QLineEdit("GOLD"); self.ed_symbol.setFixedWidth(80)
        self.ed_expiry = QLineEdit("TVC"); self.ed_expiry.setFixedWidth(90)

        self.cmb_bars = QComboBox(); self.cmb_bars.addItems(["5 min","15 min","30 min","1 hour"])
        self.cmb_bars.setCurrentText("1 hour")

        self.btn_download = QPushButton("Stáhnout posledních 5 000 barů")
        self.btn_download.clicked.connect(self.on_download_tv)
        self.btn_stop = QPushButton("Zastavit"); self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop_download)

        lay_dl.addWidget(lbl_since); lay_dl.addWidget(self.dt_since)
        lay_dl.addWidget(lbl_until); lay_dl.addWidget(self.dt_until)
        lay_dl.addSpacing(6)
        lay_dl.addWidget(QLabel("Symbol:")); lay_dl.addWidget(self.ed_symbol)
        lay_dl.addWidget(QLabel("Exchange:")); lay_dl.addWidget(self.ed_expiry)
        lay_dl.addWidget(QLabel("Svíčka:")); lay_dl.addWidget(self.cmb_bars)
        lay_dl.addStretch(1)
        lay_dl.addWidget(self.btn_stop); lay_dl.addWidget(self.btn_download)
        box_dl.setLayout(lay_dl); root.addWidget(box_dl)

        # --- Zobrazení dat ---
        box_view = QGroupBox("Zobrazení dat"); lay_view = QVBoxLayout(); lay_view.setContentsMargins(10, 8, 10, 8)
        self.lbl_status = QLabel(""); self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay_view.addWidget(self.lbl_status)

        row_view = QHBoxLayout(); row_view.addStretch(1)
        self.btn_load = QPushButton("Načíst CSV…"); self.btn_load.clicked.connect(self.on_load_csv)
        row_view.addWidget(self.btn_load); lay_view.addLayout(row_view)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setFrameShadow(QFrame.Sunken); lay_view.addWidget(sep)

        self.fig = Figure(figsize=(9, 4), dpi=100); self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay_view.addWidget(self.canvas, 1)
        self.ax = self.fig.add_subplot(111); self._style_axes_empty()
        box_view.setLayout(lay_view); root.addWidget(box_view, 1)

        # --- Log ---
        box_log = QGroupBox("Log zpráv"); lay_log = QVBoxLayout()
        self.log_view = LogConsole(); self.log_view.setReadOnly(True); self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("Hlášky stahování/načítání…"); lay_log.addWidget(self.log_view)
        box_log.setLayout(lay_log); root.addWidget(box_log)

    def _set_status(self, text: str, ok: bool | None = None) -> None:
        prefix = "⏳" if ok is None else ("✅" if ok else "❌")
        self.lbl_status.setText(f"{prefix} {text}")

    def log_msg(self, text: str) -> None:
        if not hasattr(self, "log_view") or self.log_view is None:
            return
        self.log_view.append_line(text)
        cur = self.log_view.textCursor(); cur.movePosition(QTextCursor.End)
        self.log_view.setTextCursor(cur); self.log_view.ensureCursorVisible()

    def _lock_buttons(self, locked: bool) -> None:
        self.btn_download.setEnabled(not locked if False else not locked)  # py: keep previous behavior
        self.btn_stop.setEnabled(locked); self.btn_load.setEnabled(not locked)

    def _style_axes_empty(self) -> None:
        self.ax.clear(); self.ax.set_facecolor("#111111"); self.fig.patch.set_facecolor("#111111")
        self.ax.grid(color="#333333", linestyle=":", linewidth=0.6)
        self.ax.set_title("Svíčkový graf", color="#e0e0e0", pad=10, fontsize=11)
        self.ax.tick_params(axis='x', colors="#cccccc", labelsize=9)
        self.ax.tick_params(axis='y', colors="#cccccc", labelsize=9)
        self.canvas.draw_idle()

    def on_load_csv(self) -> None:
        start_dir = PROCESSED_DIR if os.path.isdir(PROCESSED_DIR) else RAW_DIR
        path, _ = QFileDialog.getOpenFileName(self, "Vyber CSV s daty (OHLC)", start_dir, "CSV (*.csv)")
        if not path: return
        self._set_status(f"Načítám data z {path}…", ok=None)
        self.log_msg(f"[LOAD] Soubor: {path}"); self._lock_buttons(True)
        try:
            df = pd.read_csv(path); df = self._prepare_for_chart(df); self.df = df; self._plot_candles(df)
            self._set_status(f"Načteno {len(df)} řádků z {os.path.basename(path)}", ok=True)
            self.log_msg(f"[LOAD] Načteno {len(df)} řádků.")
        except Exception as e:
            self._style_axes_empty(); self._set_status(f"Chyba načtení: {e}", ok=False); self.log_msg(f"[LOAD][ERROR] {e}")
        finally:
            self._lock_buttons(False)

    # ---------------- TradingView: vždy posledních 5000 barů ----------------
    def on_download_tv(self) -> None:
        """Stáhne **posledních 5000 barů** z TradingView pro zvolený TF a vykreslí graf."""
        try:
            self._lock_buttons(True); self._set_status("Stahuji data z TradingView…", ok=None)
            symbol = (self.ed_symbol.text() or "GOLD").strip()
            exchange = (self.ed_expiry.text() or "TVC").strip()
            tf_label = self.cmb_bars.currentText().replace("mins", "min")

            # pevné pravidlo: vždy chceme posledních 5000 barů
            limit = 5000
            df = self.tv_client.get_history(symbol=symbol, exchange=exchange, timeframe_label=tf_label, limit=limit)
            if df is None or df.empty:
                self._style_axes_empty(); self._set_status("TradingView vrátil prázdná data.", ok=False)
                self.log_msg("[TV] Empty dataframe."); return

            # převod do „IBKR-like“ schématu
            df["time"] = pd.to_datetime(df["time"])
            df_out = df.copy().rename(columns={"time": "date"})
            df_out["average"]  = df_out[["open","high","low","close"]].mean(axis=1).astype(float)
            df_out["barCount"] = 0
            cols = ["date","open","high","low","close","volume","average","barCount"]
            df_out = df_out[cols].sort_values("date").reset_index(drop=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"tv_{symbol.replace('!','')}_{exchange}_{tf_label.replace(' ','')}_{ts}.csv"
            fpath = os.path.join(RAW_DIR, fname)
            try:
                os.makedirs(RAW_DIR, exist_ok=True); df_out.to_csv(fpath, index=False)
            except Exception as e:
                self.log_msg(f"[TV][WARN] Soubor neuložen: {e}")

            df_plot = df_out[["date","open","high","low","close","volume"]]
            df_chart = self._prepare_for_chart(df_plot); self.df = df_chart; self._plot_candles(df_chart)
            self._set_status(f"Staženo {len(df_out)} řádků (posledních {limit} barů).", ok=True)
            self.log_msg(f"[TV] OK: {len(df_out)} řádků. Soubor: {os.path.basename(fpath)}")
        except Exception as e:
            self._style_axes_empty(); self._set_status(f"Chyba TV: {e}", ok=False); self.log_msg(f"[TV][ERROR] {e}")
        finally:
            self._lock_buttons(False)

    def on_stop_download(self) -> None:
        if self._proc is not None:
            self._proc.kill(); self._proc = None
            self._set_status("Stahování zrušeno.", ok=False); self._lock_buttons(False)

    def on_download_ibkr(self) -> None:
        """Stahování z IBKR od zadaného data do teď, po 5000-bar batchích."""
        try:
            start_date_q = self.dt_start.date()
            start_date = datetime.combine(
                datetime(start_date_q.year(), start_date_q.month(), start_date_q.day()).date(),
                datetime.min.time()
            )
            
            symbol = (self.ed_ibkr_symbol.text() or "GC").strip()
            mode = self.cmb_ibkr_mode.currentText()
            expiry = (self.ed_ibkr_expiry.text() or "").strip() if mode == "FUT" else None
            bar_size = self.cmb_ibkr_bar.currentText()
            
            if mode == "FUT" and not expiry:
                self._set_status("Chyba: Expiry je povinná pro FUT mód", ok=False)
                return
            
            self._set_status(f"Stahuji {symbol} z IBKR od {start_date.date()}...", ok=None)
            self.log_msg(f"[IBKR] Start: {start_date.date()} | Symbol: {symbol} | Mode: {mode} | Bar: {bar_size}")
            self._lock_buttons(True)
            
            # Import v callbacku
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
                on_progress=lambda bn, tb, rec: self.log_msg(f"[IBKR] Batch {bn}: {rec} barů")
            )
            
            # Načtení a zobrazení
            df = pd.read_csv(output_path)
            df["date"] = pd.to_datetime(df["date"])
            df_plot = df[["date", "open", "high", "low", "close", "volume"]]
            df_chart = self._prepare_for_chart(df_plot)
            self.df = df_chart
            self._plot_candles(df_chart)
            
            self._set_status(f"OK: {len(df)} barů ze souboru {Path(output_path).name}", ok=True)
            self.log_msg(f"[IBKR] ✅ Hotovo: {output_path}")
            
        except Exception as e:
            self._style_axes_empty()
            self._set_status(f"Chyba IBKR: {e}", ok=False)
            self.log_msg(f"[IBKR][ERROR] {e}")
            import traceback
            self.log_msg(f"[IBKR] {traceback.format_exc()}")
        
        finally:
            self._lock_buttons(False)

    def _flush_log_queue(self, force: bool = False) -> None:
        if not self._log_queue:
            if self._log_timer.isActive(): self._log_timer.stop()
            return
        chunk: list[str] = []; max_lines = 200 if not force else len(self._log_queue)
        while self._log_queue and len(chunk) < max_lines:
            chunk.append(self._log_queue.popleft())
        if chunk:
            self.log_view.append_line("\n".join(chunk))
            cur = self.log_view.textCursor(); cur.movePosition(QTextCursor.End)
            self.log_view.setTextCursor(cur); self.log_view.ensureCursorVisible()
        if not self._log_queue and self._log_timer.isActive():
            self._log_timer.stop()

    def _prepare_for_chart(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_for_chart(df)

    def _plot_candles(self, df: pd.DataFrame) -> None:
        return plot_candles(self.fig, self.ax, self.canvas, df)
