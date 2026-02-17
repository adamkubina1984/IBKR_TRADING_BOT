# ibkr_trading_bot/utils/download_ibkr_data.py
"""
Rolling stahování historických dat z IBKR (např. GC futures) pomocí ib_insync.
Spustitelné přes main.py nebo samostatně s parametry.
"""

import argparse
import os
import shutil
from datetime import datetime, timedelta

import pandas as pd
from ib_insync import IB, Future, util


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
