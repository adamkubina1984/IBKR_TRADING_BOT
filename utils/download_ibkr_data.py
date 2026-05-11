# ibkr_trading_bot/utils/download_ibkr_data.py
"""
Rolling stahování historických dat z IBKR (např. GC futures) pomocí ib_insync.
Spustitelné přes main.py nebo samostatně s parametry.

Nové: download_ibkr_by_date_range() - stahování od data Do do teď po 5000 záznamech
"""

import argparse
import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _ensure_thread_event_loop() -> tuple[asyncio.AbstractEventLoop, bool]:
    """
    ib_insync expects an asyncio event loop even in worker threads.
    QThread-backed workers do not have one by default on Python 3.10+.
    """
    try:
        return asyncio.get_event_loop(), False
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop, True


def _import_ib_insync():
    from ib_insync import IB, Future, ContFuture, util

    return IB, Future, ContFuture, util


def download_ibkr_by_date_range(
    symbol: str,
    start_date: datetime,
    end_date: datetime | None = None,
    bar_size: str = "5 mins",
    contract_mode: str = "CONT",
    expiry: str | None = None,
    exchange: str | None = None,
    output_dir: str = "data/raw",
    max_bars_per_batch: int = 5000,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    on_progress=None,
) -> str:
    """
    Stahuje IBKR data od start_date do end_date (teď) po batchích.
    
    Args:
        symbol: Ticker (např. 'GC')
        start_date: Od kdy stahovat (datetime)
        end_date: Až kdy (default=nyní). Stahuje se od start_date k end_date
        bar_size: Granularita ('5 mins', '1 hour', atd.)
        contract_mode: 'CONT' (kontinuální futures) nebo 'FUT' (s expirací)
        expiry: Expirace (nutná pro FUT, např. '202602')
        output_dir: Cílová složka
        max_bars_per_batch: Počet barů na batch (max 5000)
        host, port, client_id: TWS/Gateway para
        on_progress: Callback fn(batch_num, total_batches, records_downloaded)
    
    Returns:
        Cesta k finálnímu sloučenému CSV souboru
    """
    loop, created_loop = _ensure_thread_event_loop()
    IB, Future, ContFuture, util = _import_ib_insync()
    if end_date is None:
        end_date = datetime.now()
    
    # Normalizace na naive (bez timezone) pro porovnávání
    if start_date.tzinfo is not None:
        start_date = start_date.replace(tzinfo=None)
    if end_date.tzinfo is not None:
        end_date = end_date.replace(tzinfo=None)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Připojení
    ib = IB()
    attempted_ports: list[int] = []
    connect_errors: list[str] = []
    candidate_ports: list[int] = [int(port)]
    candidate_client_ids: list[int] = [int(client_id), int(client_id) + 1, int(client_id) + 2, 11, 21]
    candidate_client_ids = list(dict.fromkeys(cid for cid in candidate_client_ids if cid >= 0))
    # Allow historical data download in both paper and live API modes.
    # TWS: paper=7497, live=7496 | IB Gateway: paper=4002, live=4001.
    candidate_ports.extend([7497, 4002, 7496, 4001])

    def _format_connect_error(exc: Exception) -> str:
        text = str(exc).strip()
        if not text:
            args = getattr(exc, "args", ())
            args_text = ", ".join(repr(a) for a in args if a is not None)
            text = f"args={args_text}" if args_text else repr(exc)
        return f"{type(exc).__name__}: {text}"

    seen_ports: set[int] = set()
    used_port: int | None = None
    used_client_id: int | None = None
    for candidate in candidate_ports:
        if candidate in seen_ports:
            continue
        seen_ports.add(candidate)
        attempted_ports.append(candidate)
        for candidate_client_id in candidate_client_ids:
            try:
                if ib.isConnected():
                    ib.disconnect()
                ib.connect(host, candidate, clientId=candidate_client_id, timeout=4)
                used_port = candidate
                used_client_id = candidate_client_id
                break
            except Exception as exc:
                connect_errors.append(f"{candidate}/{candidate_client_id}: {_format_connect_error(exc)}")
        if used_port is not None:
            break

    if used_port is None or not ib.isConnected():
        attempted = ", ".join(str(p) for p in attempted_ports)
        details = " | ".join(connect_errors) if connect_errors else "bez detailu"
        raise ConnectionError(
            "IBKR API pripojeni selhalo. "
            f"Host={host}, zkousene porty=[{attempted}], clientId={client_id}, fallbackClientIds={candidate_client_ids}. "
            "Zkontroluj TWS/Gateway API nastaveni (Enable ActiveX and Socket Clients, Trusted IP) "
            "a ze bezi API port (paper: 7497/4002, live: 7496/4001). "
            f"Detaily: {details}"
        )
    if used_client_id is not None and used_client_id != int(client_id):
        print(
            f"ℹ️  IBKR connect fallback: pozadovany clientId={client_id}, pouzit clientId={used_client_id} na portu {used_port}."
        )
    
    try:
        # Definice kontraktu
        what_to_show = "TRADES"
        # Special mapping: GOLD on TVC -> use GC future with expiry 202602 (explicit FUT)
        if exchange is not None and str(exchange).upper() == "TVC" and str(symbol).upper() == "GOLD":
            # Map GOLD/TVC to COMEX gold future GC (example expiry 202602)
            expiry_used = expiry or "202602"
            contract = Future("GC", expiry_used, "COMEX", currency="USD")
            contract.includeExpired = True
            what_to_show = "TRADES"
        else:
            if contract_mode.upper() == "CONT":
                contract = ContFuture(symbol, "COMEX")
                what_to_show = "TRADES"
            else:  # FUT
                if not expiry:
                    raise ValueError("Expirace (--expiry) je povinná pro kontrakty FUT")
                contract = Future(symbol, expiry, "COMEX", currency="USD")
                contract.includeExpired = True
                what_to_show = "TRADES"
        
        ib.qualifyContracts(contract)

        # Pokud je to kontinuální future, IB zakazuje nastavit endDateTime
        # proto použijeme konkrétní FUT kontrakt podle resolved expiry
        if isinstance(contract, ContFuture):
            expiry_resolved = getattr(contract, "lastTradeDateOrContractMonth", None) or expiry
            if not expiry_resolved:
                raise RuntimeError(
                    "Kontinuální futures nelze stáhnout s endDateTime; zadejte prosím expiraci (FUT mode) nebo použijte jiný symbol."
                )
            contract = Future(
                contract.symbol,
                expiry_resolved,
                getattr(contract, "exchange", "COMEX"),
                currency=getattr(contract, "currency", "USD"),
            )
            contract.includeExpired = True
            ib.qualifyContracts(contract)

        # Stahování po batchích
        all_batches = []
        batch_num = 0
        current_end = end_date
        
        temp_dir = Path(tempfile.gettempdir()) / f"ibkr_download_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Stahování {symbol} od {start_date.date()} do {end_date.date()}")
        print(f"   Mode: {contract_mode} | Bar: {bar_size} | Max: {max_bars_per_batch} barů/batch")
        
        while current_end > start_date:
            batch_num += 1
            end_str = current_end.strftime("%Y%m%d %H:%M:%S")
            
            # Vypočti durationStr tak, aby pokryl přibližně max_bars_per_batch
            # např. pokud bar_size='5 mins' a max_bars_per_batch=5000 => 5000*5 minutes
            def _minutes_per_bar(bs: str) -> int:
                s = bs.strip().lower()
                if s.endswith('min') or s.endswith('mins'):
                    return int(s.split()[0]) if ' ' in s else int(s.replace('mins','').replace('min',''))
                if s.endswith('hour') or s.endswith('hours'):
                    return int(s.split()[0]) * 60
                if s.endswith('day') or s.endswith('days'):
                    return int(s.split()[0]) * 60 * 24
                # fallback assume minutes
                try:
                    return int(s)
                except Exception:
                    return 5

            minutes_per_bar = _minutes_per_bar(bar_size)
            total_minutes = max_bars_per_batch * max(1, minutes_per_bar)
            if total_minutes >= 60 * 24:
                days = max(1, (total_minutes + 60*24 - 1) // (60*24))
                durationStr = f"{days} D"
            elif total_minutes >= 60:
                hours = max(1, (total_minutes + 59) // 60)
                durationStr = f"{hours} H"
            else:
                durationStr = f"{max(1, total_minutes)} S"

            # Stažení data
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr=durationStr,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=False,
                formatDate=1,
                keepUpToDate=False,
            )
            
            if not bars:
                print(f"  ⚠️  Batch {batch_num}: Žádná data, konec.")
                break
            
            df = util.df(bars)
            if df is None or df.empty:
                print(f"  ⚠️  Batch {batch_num}: Prázdná data, konec.")
                break
            
            # Okleštění na max_bars_per_batch pokud je více
            if len(df) > max_bars_per_batch:
                df = df.tail(max_bars_per_batch).copy()
            
            # Filtrování: vezmi jen záznamy >= start_date
            df["date"] = pd.to_datetime(df["date"])
            # Normalizuj date na naive (bez timezone) pro porovnání
            df["date"] = df["date"].dt.tz_localize(None)
            df = df[df["date"] >= start_date].copy()
            
            if df.empty:
                print(f"  ✓ Batch {batch_num}: Dosažen začátek ({start_date.date()}), konec.")
                break
            
            # Uložení
            batch_path = temp_dir / f"batch_{batch_num:04d}.csv"
            df.to_csv(batch_path, index=False)
            all_batches.append(df)
            
            print(f"  ✓ Batch {batch_num}: {len(df)} barů (od {df['date'].min().date()})")
            
            if on_progress:
                on_progress(batch_num, None, len(df))
            
            # Posun back
            current_end = df["date"].min() - timedelta(minutes=1)
            
            # Pacing
            import time
            time.sleep(1.0)
        
        # Merge všech batchí
        if not all_batches:
            raise RuntimeError("Žádná data se nestáhla!")
        
        print(f"\n🔗 Slučuji {len(all_batches)} batchů...")
        df_merged = pd.concat(all_batches, ignore_index=True)
        df_merged["date"] = pd.to_datetime(df_merged["date"])
        # Normalizuj na naive (bez timezone)
        df_merged["date"] = df_merged["date"].dt.tz_localize(None)
        df_merged = df_merged.sort_values("date").drop_duplicates(subset=["date"])
        
        # Filtrování přesně na rozsah
        df_merged = df_merged[(df_merged["date"] >= start_date) & (df_merged["date"] <= end_date)]
        
        # Uložení
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bar_tag = bar_size.replace(" ", "").replace("mins", "m")
        if contract_mode.upper() == "FUT" and expiry:
            fname = (
                f"{symbol}_{expiry}_{bar_tag}_{len(df_merged)}bars_"
                f"{start_date.strftime('%Y%m%d')}_{timestamp}.csv"
            )
        else:
            fname = f"{symbol}_{bar_tag}_{len(df_merged)}bars_{start_date.strftime('%Y%m%d')}_{timestamp}.csv"
        output_path = Path(output_dir) / fname
        
        # Formát: date,open,high,low,close,volume
        cols = ["date", "open", "high", "low", "close", "volume"]
        if "average" in df_merged.columns:
            cols.append("average")
        if "barCount" in df_merged.columns:
            cols.append("barCount")
        
        df_out = df_merged[cols].copy()
        df_out.to_csv(output_path, index=False)
        
        # Čištění temp dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"\n✅ Hotovo: {len(df_merged)} barů")
        print(f"   Rozsah: {df_merged['date'].min().date()} až {df_merged['date'].max().date()}")
        print(f"   Soubor: {output_path}")
        
        return str(output_path)
    
    finally:
        try:
            ib.disconnect()
        finally:
            if created_loop:
                try:
                    loop.stop()
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass




def download_data(symbol: str, expiry: str, days_back: int, bar_size: str, output_dir: str = "data/raw"):
    loop, created_loop = _ensure_thread_event_loop()
    IB, Future, _ContFuture, util = _import_ib_insync()
    # Připojení k IB Gateway
    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=1)

    # Definice kontraktu s expirací
    contract = Future(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiry,
        exchange='COMEX',
        currency='USD'
    )

    os.makedirs(output_dir, exist_ok=True)
    all_data = []

    print(f"Stahuji data pro {symbol} ({expiry}) za posledních {days_back} dnů...")

    for i in range(days_back):
        day = datetime.now() - timedelta(days=i)
        day_str = day.strftime('%Y%m%d %H:%M:%S')
        print(f"  ➤ Den {i+1}: {day.date()}")

        bars = ib.reqHistoricalData(
            contract,
            endDateTime=day_str,
            durationStr='1 D',
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )

        if not bars:
            print(f"    ⚠️  Žádná data pro den {day.date()}")
            continue

        df = util.df(bars)
        df.drop_duplicates(subset='date', inplace=True)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # Kontrola časových mezer
        df['delta'] = df.index.to_series().diff()
        if (df['delta'] > pd.Timedelta(minutes=5)).any():
            print("    ⚠️  Mezery mezi svíčkami detekovány")
        df.drop(columns='delta', inplace=True)

        all_data.append(df)

    if all_data:
        df_all = pd.concat(all_data)
        df_all = df_all[~df_all.index.duplicated()]
        df_all.sort_index(inplace=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        bar_tag = bar_size.replace(' ', '')
        filename = f"{output_dir}/{symbol}_{expiry}_{bar_tag}_{days_back}d_{timestamp}.csv"
        df_all.to_csv(filename)

        shutil.copy2(filename, f"{output_dir}/ohlc_data.csv")
        print(f"📄 Zkopírováno také jako: {output_dir}/ohlc_data.csv")

        print(f"\n✅ Staženo {len(df_all)} svíček")
        print(f"Rozsah: {df_all.index.min()} až {df_all.index.max()}")
        print(f"Uloženo do: {filename}")
    else:
        print("❌ Nepodařilo se stáhnout žádná data.")

    try:
        ib.disconnect()
    finally:
        if created_loop:
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass


# Volitelně: samostatné spouštění pro test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stažení historických dat z IBKR.")
    parser.add_argument('--symbol', type=str, required=True, help='Např. GC')
    parser.add_argument('--expiry', type=str, required=True, help='Např. 202509')
    parser.add_argument('--days', type=int, default=30, help='Počet dnů zpět')
    parser.add_argument('--barSize', type=str, default='5 mins', help='Granularita, např. "5 mins"')
    parser.add_argument('--output', type=str, default='data/raw', help='Složka pro CSV')

    args = parser.parse_args()

    download_data(
        symbol=args.symbol,
        expiry=args.expiry,
        days_back=args.days,
        bar_size=args.barSize,
        output_dir=args.output
    )
