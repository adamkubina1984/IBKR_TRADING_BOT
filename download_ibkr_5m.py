import argparse
import time
from datetime import datetime, timezone

import pandas as pd
from ib_insync import IB, ContFuture, Forex, Future, util


def parse_dt(s: str) -> datetime:
    # očekává "YYYY-MM-DD HH:MM"
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)

def bars_to_df(bars) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    df = util.df(bars)
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    keep = [c for c in ["date","open","high","low","close","volume"] if c in df.columns]
    return df[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help='UTC "YYYY-MM-DD HH:MM"')
    ap.add_argument("--end", required=True, help='UTC "YYYY-MM-DD HH:MM"')
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--bar", default="5 mins")
    ap.add_argument("--chunk", default="7 D", help='IBKR durationStr, e.g. "7 D", "14 D"')
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496, help="7497 paper, 7496 live")
    ap.add_argument("--clientId", type=int, default=7)

    # instrument volby:
    ap.add_argument("--mode", choices=["CONT_GC", "FUT_GC", "XAUUSD"], default="CONT_GC")
    ap.add_argument("--contract", default="202602", help='jen pro FUT_GC, např. 202512 / 202602')
    args = ap.parse_args()

    start = parse_dt(args.start)
    end = parse_dt(args.end)

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.clientId)

    # Vyber instrument:
    if args.mode == "CONT_GC":
        contract = ContFuture("GC", "COMEX")  # kontinuální futures (front)
        what = "TRADES"
    elif args.mode == "FUT_GC":
        contract = Future("GC", args.contract, "COMEX")  # konkrétní kontrakt
        what = "TRADES"
    else:
        contract = Forex("XAUUSD")  # spot XAUUSD (pokud máš na IB povolené)
        what = "MIDPOINT"

    ib.qualifyContracts(contract)

    all_parts = []
    cur_end = end

    # Taháme zpětně od end → start po blocích
    while cur_end > start:
        end_str = cur_end.strftime("%Y%m%d-%H:%M:%S")  # IB format
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_str,
            durationStr=args.chunk,
            barSizeSetting=args.bar,
            whatToShow=what,
            useRTH=False,
            formatDate=1,
            keepUpToDate=False,
        )
        df = bars_to_df(bars)
        if df.empty:
            # když narazíš na pacing / prázdno, zkus menší chunk
            break

        all_parts.append(df)

        # posuň konec na začátek staženého bloku (minus 1 bar)
        cur_end = df["date"].min().replace(tzinfo=timezone.utc)
        # pacing ochrana
        time.sleep(1.0)

    ib.disconnect()

    if not all_parts:
        raise SystemExit("Nic se nestáhlo (zkontroluj TWS/Gateway, port, instrument, permissions).")

    out = pd.concat(all_parts, ignore_index=True)
    out = out.sort_values("date").drop_duplicates("date")

    # ořízni přesně na požadovaný interval
    start_naive = start.replace(tzinfo=None)
    end_naive = end.replace(tzinfo=None)
    out = out[(out["date"] >= start_naive) & (out["date"] <= end_naive)]

    out.to_csv(args.out, index=False)
    print(f"OK: uloženo {len(out)} barů do {args.out}")

if __name__ == "__main__":
    main()
