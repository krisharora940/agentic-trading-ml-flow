from __future__ import annotations

import argparse
from pathlib import Path

from trading_ml.stage2_data import load_ohlcv_file, regular_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache an opening-hours subset of OHLCV data as Parquet.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--timeframe", default="30s")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--window-end", default="11:30:00")
    args = parser.parse_args()

    bars = load_ohlcv_file(
        args.source,
        symbol=args.symbol,
        timeframe=args.timeframe,
        timezone=args.timezone,
    )
    subset = regular_session(bars).between_time("09:30:00", args.window_end, inclusive="both")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output)
    print(f"wrote {output}")
    print({"rows": int(len(subset)), "date_start": subset.index.min().isoformat(), "date_end": subset.index.max().isoformat()})


if __name__ == "__main__":
    main()
