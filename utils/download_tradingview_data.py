import argparse

from ibkr_trading_bot.core.datasource.tv_source import TradingViewSource, TVConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="GC1!")
    ap.add_argument("--exchange", default="COMEX")
    ap.add_argument("--interval", default="15 mins")
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--username", default=None)
    ap.add_argument("--password", default=None)
    args = ap.parse_args()

    src = TradingViewSource(TVConfig(username=args.username, password=args.password))
    df = src.get_hist(args.symbol, args.exchange, args.interval, n_bars=args.bars)
    df.rename(columns={"datetime":"timestamp"}, inplace=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
