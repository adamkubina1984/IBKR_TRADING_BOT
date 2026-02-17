
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


class Plotting:
    def candle_macd(self, df):
        # Placeholder: zachováváme stávající kreslení v GUI; tohle je připravené API pro pozdější přesun
        return None

    def equity_curve(self, report):
        fig, ax = plt.subplots()
        if hasattr(report, "equity"):
            report.equity.plot(ax=ax)
        ax.set_title("Equity Curve")
        return fig

    def payout_histogram(self, report):
        fig, ax = plt.subplots()
        if hasattr(report, "trade_returns"):
            ax.hist(report.trade_returns.dropna())
        ax.set_title("Trade Returns Histogram")
        return fig


# === Extracted from GUI (1:1 logic preserved) ===
    def prepare_for_chart(df: pd.DataFrame) -> pd.DataFrame:
        """
        Připraví DataFrame pro kreslení svíček:
        - najde timestamp (timestamp/datetime/date/…),
        - najde OHLC (open/high/low/close),
        - volitelně najde volume (volume/vol/qty/quantity/barCount/…),
        - vrací sloupce: ['timestamp','open','high','low','close', ('volume' pokud je)].
        """
        import numpy as np
        import pandas as pd

        if df is None or df.empty:
            raise ValueError("Soubor je prázdný.")

        # --- case-insensitive mapa ---
        cols_lower = {c.lower().strip(): c for c in df.columns}

        def pick(*cands: str) -> str | None:
            for name in cands:
                key = name.lower().strip()
                if key in cols_lower:
                    return cols_lower[key]
            return None

        # 1) timestamp
        ts_col = pick("timestamp", "datetime", "date_time", "bar_time", "bar_start", "start_time", "date")
        if ts_col is None:
            raise ValueError("Chybí časový sloupec (zkus: timestamp/datetime/date_time/bar_time/bar_start/start_time/date).")
        ts = df[ts_col]

        # převod na datetime (podpora epoch s/ms i textu)
        try:
            if np.issubdtype(ts.dtype, np.number):
                maxv = float(np.nanmax(ts.values))
                unit = "ms" if maxv > 1e12 else "s"
                ts_parsed = pd.to_datetime(ts, unit=unit, errors="coerce", utc=False)
            else:
                ts_parsed = pd.to_datetime(ts, errors="coerce", utc=False)
                if ts_parsed.isna().mean() > 0.5:
                    as_num = pd.to_numeric(ts.astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")
                    maxv = float(np.nanmax(as_num.values)) if np.isfinite(as_num.max()) else 0.0
                    if maxv > 0:
                        unit = "ms" if maxv > 1e12 else "s"
                        ts2 = pd.to_datetime(as_num, unit=unit, errors="coerce", utc=False)
                        if ts2.notna().sum() > ts_parsed.notna().sum():
                            ts_parsed = ts2
        except Exception:
            ts_parsed = pd.to_datetime(ts, errors="coerce", utc=False)

        if ts_parsed.isna().all():
            raise ValueError(f"Nepodařilo se převést '{ts_col}' na datetime.")

        # 2) OHLC
        open_col  = pick("open", "o", "open_price", "openprice", "price_open")
        high_col  = pick("high", "h", "high_price", "highprice", "price_high")
        low_col   = pick("low",  "l", "low_price",  "lowprice",  "price_low")
        close_col = pick("close","c", "close_price","closeprice","price_close","last","last_price","adj_close")

        missing = [name for name, col in {
            "open": open_col, "high": high_col, "low": low_col, "close": close_col
        }.items() if col is None]
        if missing:
            raise ValueError(f"Chybí sloupce pro OHLC: {missing}. Dostupné: {list(df.columns)}")

        # 3) Volume (volitelné)
        volume_col = pick("volume", "vol", "qty", "quantity", "barcount", "bar_count", "trade_volume", "volume_sum")

        # 4) Sestavení výstupu
        out = pd.DataFrame({
            "timestamp": ts_parsed,
            "open":  pd.to_numeric(df[open_col],  errors="coerce"),
            "high":  pd.to_numeric(df[high_col],  errors="coerce"),
            "low":   pd.to_numeric(df[low_col],   errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        })

        if volume_col is not None:
            out["volume"] = pd.to_numeric(df[volume_col], errors="coerce")

        # 5) Očista + řazení
        out = out.dropna(subset=["timestamp"])
        out = out.sort_values("timestamp").reset_index(drop=True)
        out[["open", "high", "low", "close"]] = (
            out[["open", "high", "low", "close"]]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(method="ffill")
            .fillna(method="bfill")
        )
        if "volume" in out:
            out["volume"] = out["volume"].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 6) Log mapování
        try:
            log_msg(
                "[MAP] "
                f"timestamp='{ts_col}', open='{open_col}', high='{high_col}', "
                f"low='{low_col}', close='{close_col}', volume='{volume_col or '—'}'"
            )
        except Exception:
            pass

        return out


    def plot_candles(fig, ax, canvas, df: pd.DataFrame) -> None:
        """
        Vykreslí svíčkový graf + objemy do figure (2 panely pod sebou).
        Očekává sloupce: ['timestamp','open','high','low','close'] a volitelně ['volume'].
        """
        import matplotlib.dates as mdates
        import numpy as np
        import pandas as pd
        from matplotlib.patches import Rectangle

        # --- validace ---
        if df is None or df.empty:
            self._style_axes_empty()
            return

        # --- načtení + čištění dat ---
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        # zobrazíme lokální (naivní) čas – žádný posun
        try:
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_localize(None)
        except Exception:
            try:
                ts = ts.dt.tz_convert(None)
            except Exception:
                pass

        o = pd.to_numeric(df["open"],  errors="coerce")
        h = pd.to_numeric(df["high"],  errors="coerce")
        l = pd.to_numeric(df["low"],   errors="coerce")
        c = pd.to_numeric(df["close"], errors="coerce")

        # volume je volitelné (najdeme case-insensitive)
        vol_col = None
        for col in df.columns:
            if str(col).lower().strip() in {"volume", "vol", "qty", "quantity"}:
                vol_col = col
                break
        v = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else None

        mask = ts.notna() & o.notna() & h.notna() & l.notna() & c.notna()
        if not mask.any():
            self._style_axes_empty()
            return
        ts, o, h, l, c = ts[mask], o[mask], h[mask], l[mask], c[mask]
        if v is not None:
            v = v[mask].fillna(0)

        # mdates čísla
        times = mdates.date2num(ts.dt.to_pydatetime())

        # --- odhad šířky svíčky z typického intervalu ---
        if len(times) > 1:
            diffs = np.diff(times)
            diffs = diffs[diffs > 0]
            if diffs.size:
                p90 = np.percentile(diffs, 90)
                core = diffs[diffs <= p90]
                base = np.median(core) if core.size else np.median(diffs)
                dx = float(np.clip(base, 1.0 / (24 * 60), 1.0))   # min 1 min, max 1 den
            else:
                dx = 1.0 / (24 * 60)
        else:
            dx = 1.0 / (24 * 60)
        w = dx * 0.75
        w_vol = dx * 0.70

        # --- připrav figure se 2 panely ---
        fig.clear()
        ax_price, ax_vol = fig.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
        )
        # ulož referenci (kvůli ostatním metodám)
        ax = ax_price
        ax_vol = ax_vol

        # tmavé pozadí a mřížka
        for ax in (ax_price, ax_vol):
            ax.set_facecolor("#111111")
            fig.patch.set_facecolor("#111111")
            ax.grid(color="#333333", linestyle=":", linewidth=0.6)
            ax.tick_params(axis='x', colors="#cccccc", labelsize=9)
            ax.tick_params(axis='y', colors="#cccccc", labelsize=9)
        ax_price.set_title("Svíčkový graf", color="#e0e0e0", pad=10, fontsize=11)

        # --- vykreslení svíček ---
        greens = "#26a69a"
        reds   = "#ef5350"
        o_np, h_np, l_np, c_np = o.to_numpy(), h.to_numpy(), l.to_numpy(), c.to_numpy()

        for t, open_, high_, low_, close_ in zip(times, o_np, h_np, l_np, c_np):
            color = greens if close_ >= open_ else reds
            # knot
            ax_price.vlines(t, low_, high_, linewidth=0.9, color=color, alpha=0.95)
            # tělo
            y = min(open_, close_)
            height = max(abs(close_ - open_), 1e-8)
            ax_price.add_patch(
                Rectangle(
                    (t - w / 2, y),
                    w,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0.6,
                    alpha=0.95,
                )
            )

        # y-lim s paddingem
        y_min = float(np.nanmin(l_np))
        y_max = float(np.nanmax(h_np))
        pad = max((y_max - y_min) * 0.02, 1e-6)
        ax_price.set_ylim(y_min - pad, y_max + pad)

        # --- objemy (pokud jsou) ---
        if v is not None and v.notna().any():
            colors = np.where(c_np >= o_np, greens, reds)
            ax_vol.bar(
                times,
                v.to_numpy(),
                width=w_vol,
                color=colors,
                edgecolor=colors,
                alpha=0.7,
                align="center",
            )
            ax_vol.set_ylabel("Volume", color="#cccccc", labelpad=4)
            ax_vol.yaxis.set_major_locator(mticker.AutoLocator())
            # trochu prostoru nahoře
            vmax = float(np.nanmax(v.to_numpy())) if len(v) else 1.0
            ax_vol.set_ylim(0, vmax * 1.15 if vmax > 0 else 1.0)
        else:
            # když objem není, zruš spodní panel a zvětši horní
            ax_vol.remove()
            fig.clear()
            ax_price = fig.add_subplot(111)
            ax = ax_price
            # znovu nastavit styl pro jediný panel
            ax_price.set_facecolor("#111111")
            fig.patch.set_facecolor("#111111")
            ax_price.grid(color="#333333", linestyle=":", linewidth=0.6)
            ax_price.tick_params(axis='x', colors="#cccccc", labelsize=9)
            ax_price.tick_params(axis='y', colors="#cccccc", labelsize=9)
            ax_price.set_title("Svíčkový graf", color="#e0e0e0", pad=10, fontsize=11)
            # znovu dokreslit svíčky
            for t, open_, high_, low_, close_ in zip(times, o_np, h_np, l_np, c_np):
                color = greens if close_ >= open_ else reds
                ax_price.vlines(t, low_, high_, linewidth=0.9, color=color, alpha=0.95)
                y = min(open_, close_)
                height = max(abs(close_ - open_), 1e-8)
                ax_price.add_patch(
                    Rectangle(
                        (t - w / 2, y),
                        w,
                        height,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.6,
                        alpha=0.95,
                    )
                )
            ax_price.set_ylim(y_min - pad, y_max + pad)

        # --- společné nastavení osy X ---
        ax = ax  # horní osa
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.yaxis.set_major_locator(mticker.AutoLocator())
        ax.margins(x=0.01, y=0.05)

        if len(times):
            ax.set_xlim(times.min() - dx, times.max() + dx)

        canvas.draw_idle()

def _parse_time_series(s: pd.Series) -> pd.DatetimeIndex:
    """Bezpečně naparsuje různé formáty času (stringy, epoch s/ms/ns). Vrací UTC DatetimeIndex."""
    s = s.dropna()
    # stringy → standardně
    if s.dtype == "O":
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        return pd.DatetimeIndex(dt)

    # numerické epochy (heuristika podle řádu)
    if np.issubdtype(s.dtype, np.integer) or np.issubdtype(s.dtype, np.floating):
        v = pd.Series(s.astype("float64"))
        # odhad řádu
        vmax = float(np.nanmax(v.values)) if len(v) else 0.0
        if vmax > 1e13:   # ns
            dt = pd.to_datetime(v.astype("int64"), errors="coerce", utc=True)
        elif vmax > 1e11: # ms
            dt = pd.to_datetime((v*1e6).astype("int64"), errors="coerce", utc=True)
        elif vmax > 1e9:  # s
            dt = pd.to_datetime((v*1e9).astype("int64"), errors="coerce", utc=True)
        else:
            # malé hodnoty – nebudeme je násilím konvertovat (pravděpodobně index 0..N)
            dt = pd.to_datetime(v, errors="coerce", utc=True)
        return pd.DatetimeIndex(dt)

    # fallback
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return pd.DatetimeIndex(dt)


def draw_equity_chart(ax, results: dict, baseline=None):
    ax.clear()
    ax.set_title("Equity curve (Strategy vs. Baseline)")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity (×)")
    import numpy as np
    # Strategy (net)
    eq_net = results.get("equity_bar_net")
    if isinstance(eq_net, (list, np.ndarray)) and len(eq_net):
        ax.plot(eq_net, label="Strategy (net)")
    else:
        trade_pnls_plot = results.get("trade_pnls_net") or results.get("trade_pnls")
        if trade_pnls_plot:
            eq = np.cumprod([1.0] + [1.0 + float(x) for x in trade_pnls_plot])
            ax.plot(eq, label="Strategy (approx)")
    # Baseline
    if baseline is not None:
        try:
            ax.plot(baseline, label="Baseline (buy&hold)", linestyle="--")
        except Exception:
            pass
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

def draw_histogram(ax, trade_pnls):
    ax.clear()
    ax.set_title("Histogram zisků/ztrát obchodů")
    ax.set_xlabel("Zisk na obchod")
    ax.set_ylabel("četnost")
    import numpy as np
    if trade_pnls and len(trade_pnls):
        arr = np.array(trade_pnls, dtype=float)
        ax.hist(arr, bins=min(50, max(10, int(len(arr) ** 0.5))), alpha=0.9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Seznam 'trade_pnls' není k dispozici.", ha="center", va="center", transform=ax.transAxes)

def draw_rolling_chart(ax, results: dict, mode: str, window: int = 20):
    ax.clear()
    ax.set_title(f"{mode} (okno {window})")
    ax.set_xlabel("Bar / Trade index")
    import numpy as np
    eq_bar_net = results.get("equity_bar_net")
    if mode.lower().startswith("profit"):
        trade_pnls = results.get("trade_pnls_net") or results.get("trade_pnls")
        if trade_pnls:
            arr = np.array(trade_pnls, dtype=float)
            if window > 1 and len(arr) >= window:
                roll = pd.Series(arr).rolling(window).sum()
                ax.plot(roll.values, label=f"Rolling sum ({window})")
            else:
                ax.plot(arr, label="PnL per trade")
    elif "sharpe" in mode.lower():
        rolling_sh = results.get("rolling_sharpe")
        if rolling_sh is not None:
            ax.plot(rolling_sh, label="Rolling Sharpe")
    elif "drawdown" in mode.lower():
        rolling_dd = results.get("rolling_max_dd")
        if rolling_dd is not None:
            ax.plot(rolling_dd, label="Rolling Max Drawdown")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    # --- ADD at the bottom of ibkr_trading_bot/core/utils/plotting.py ---

def prepare_for_chart(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustní příprava DF pro kreslení:
    - najde časový sloupec (timestamp|datetime|date), naparsuje a dá jako index,
    - srovná, odfiltruje NaT/duplikáty,
    - zajistí numeric OHLC(V).
    """
    if df is None or df.empty:
        raise ValueError("prepare_for_chart: prázdný dataframe")

    df = df.copy()

    # 1) najdi časový sloupec
    time_col = None
    for cand in ("timestamp", "datetime", "date", "time", "Date", "Datetime"):
        if cand in df.columns:
            time_col = cand
            break

    if time_col is not None:
        dtidx = _parse_time_series(df[time_col])
    else:
        # zkusíme použít index, pokud je už datetime-like
        if isinstance(df.index, pd.DatetimeIndex):
            dtidx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        else:
            # když je index 0..N, nesnaž se ho „převádět“ – to dělá ty nesmysly kolem 1970
            raise ValueError("prepare_for_chart: nenašel jsem časový sloupec (timestamp/datetime/date)")

    df.index = dtidx
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # 2) zajisti numeric OHLCV
    need = ["open", "high", "low", "close"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"prepare_for_chart: chybí sloupec '{c}'")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])

    return df


def plot_candles(fig, ax, canvas, df: pd.DataFrame):
    """Svíčky s rozumnou šířkou dle mediánu kroku v čase."""
    if df is None or df.empty:
        ax.clear()
        ax.set_title("Žádná data")
        canvas.draw_idle()
        return

    df = prepare_for_chart(df)

    # převod času
    x = mdates.date2num(df.index.to_pydatetime())
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")

    ax.clear()

    # odhad šířky svíčky
    if len(x) >= 3:
        step = np.median(np.diff(x))
    elif len(x) == 2:
        step = x[1] - x[0]
    else:
        step = 1.0
    w = float(step) * 0.8 if step > 0 else 0.8

    # wicky + těla
    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        ax.vlines(xi, li, hi, linewidth=1)
        up = ci >= oi
        y = min(oi, ci)
        height = abs(ci - oi) or 1e-12
        rect = Rectangle((xi - w/2, y), w, height, fill=True, linewidth=0, alpha=0.85)
        # barvy nechávám defaultní – pokud chceš, lze rozlišit up/down
        # rect.set_facecolor('tab:green' if up else 'tab:red')
        ax.add_patch(rect)

    ax.xaxis_date()
    ax.set_title("Candles")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    canvas.draw_idle()

