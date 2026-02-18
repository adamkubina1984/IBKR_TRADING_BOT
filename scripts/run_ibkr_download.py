from datetime import datetime, timedelta
import os
import sys

# Ujistíme se, že rodičovský adresář projektu je v sys.path (pro import balíčku)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ibkr_trading_bot.utils.download_ibkr_data import download_ibkr_by_date_range

# Parametry
symbol = "GOLD"
exchange = "TVC"
start_date = datetime.utcnow() - timedelta(days=30)
end_date = datetime.utcnow()
bar_size = "5 mins"
contract_mode = "CONT"
output_dir = "data/raw"

print(f"START: symbol={symbol}, start={start_date}, end={end_date}, bar={bar_size}")

out = download_ibkr_by_date_range(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    bar_size=bar_size,
    contract_mode=contract_mode,
    expiry=None,
    exchange=exchange,
    output_dir=output_dir,
    max_bars_per_batch=5000,
    host="127.0.0.1",
    port=7496,
    client_id=1,
    on_progress=lambda bn, tb, rec: print(f"BATCH {bn}: {rec} bars")
)

print('DONE. Output:', out)
