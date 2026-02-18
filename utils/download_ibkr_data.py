# ibkr_trading_bot/utils/download_ibkr_data.py
"""
Rolling stahování historických dat z IBKR (např. GC futures) pomocí ib_insync.
Spustitelné přes main.py nebo samostatně s parametry.

Nové: download_ibkr_by_date_range() - stahování od data Do do teď po 5000 záznamech
"""

import argparse
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from ib_insync import IB, Future, ContFuture, util


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
    port: int = 7496,
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
    ib.connect(host, port, clientId=client_id)
    
    try:
        # Definice kontraktu
        what_to_show = "TRADES"
        # Special mapping: GOLD on TVC -> use GC future with expiry 202602 (explicit FUT)
        if exchange is not None and str(exchange).upper() == "TVC" and str(symbol).upper() == "GOLD":
            # Map GOLD/TVC to COMEX gold future GC (example expiry 202602)
            expiry_used = expiry or "202602"
            contract = Future("GC", expiry_used, "COMEX", currency="USD")
            what_to_show = "TRADES"
        else:
            if contract_mode.upper() == "CONT":
                contract = ContFuture(symbol, "COMEX")
                what_to_show = "TRADES"
            else:  # FUT
                if not expiry:
                    raise ValueError("Expirace (--expiry) je povinná pro kontrakty FUT")
                contract = Future(symbol, expiry, "COMEX", currency="USD")
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
        ib.disconnect()




def download_data(symbol: str, expiry: str, days_back: int, bar_size: str, output_dir: str = "data/raw"):
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

    ib.disconnect()


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
