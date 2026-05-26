from __future__ import annotations

import argparse
from pathlib import Path

from trading_ml.stage2_data import build_data_quality_report, load_ohlcv_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize OHLCV source data and cache it as Parquet."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--timeframe", default="30s")
    parser.add_argument("--timezone", default="America/New_York")
    args = parser.parse_args()

    bars = load_ohlcv_file(
        args.source,
        symbol=args.symbol,
        timeframe=args.timeframe,
        timezone=args.timezone,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(output)
    report = build_data_quality_report(
        bars,
        source_path=output,
        symbol=args.symbol,
        timeframe=args.timeframe,
        timezone=args.timezone,
    )
    print(f"wrote {output}")
    print(report.to_dict())


if __name__ == "__main__":
    main()
