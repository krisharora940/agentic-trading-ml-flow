from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ml.schemas import utc_now_iso
from trading_ml.stage2_data import build_data_quality_report, load_ohlcv_file


DEFAULT_SOURCES = [
    "/Users/radhikaarora/Documents/Trading ML/ML V2/data/mnq_30s_2026-01-01_to_2026-02-28.parquet",
    "/Users/radhikaarora/Documents/Trading ML/ML V2/data/mnq_30s_2026-03-20_to_2026-03-27.parquet",
    "/Users/radhikaarora/Documents/Trading ML/ML V2/data/mnq_1m_2026-03-20_to_2026-03-27.parquet",
]


def infer_timeframe(path: Path) -> str:
    name = path.name.lower()
    if "_30s_" in name or "30s" in name:
        return "30s"
    if "_1m_" in name or "1m" in name:
        return "1m"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a real Databento manifest from MNQ Parquet files."
    )
    parser.add_argument(
        "--output", default="data/manifests/databento_mnq_manifest.json"
    )
    parser.add_argument("--source", action="append", dest="sources")
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--timezone", default="America/New_York")
    args = parser.parse_args()

    entries = []
    for source in args.sources or DEFAULT_SOURCES:
        path = Path(source)
        timeframe = infer_timeframe(path)
        bars = load_ohlcv_file(
            path, symbol=args.symbol, timeframe=timeframe, timezone=args.timezone
        )
        quality = build_data_quality_report(
            bars,
            source_path=path,
            symbol=args.symbol,
            timeframe=timeframe,
            timezone=args.timezone,
        )
        entries.append(
            {
                "symbol": args.symbol,
                "instrument": "Micro E-mini Nasdaq-100 futures",
                "provider": "databento",
                "timeframe": timeframe,
                "schema": f"ohlcv-{timeframe}",
                "source_path": str(path),
                "timezone": args.timezone,
                "date_start": quality.date_start,
                "date_end": quality.date_end,
                "rows": quality.rows,
                "sessions": quality.sessions,
                "opening_zone_sessions": quality.opening_zone_sessions,
                "duplicate_timestamps": quality.duplicate_timestamps,
                "missing_regular_session_bars": quality.missing_regular_session_bars,
                "quality_flags": quality.quality_flags,
            }
        )

    manifest = {
        "dataset_name": "databento_mnq_bnr",
        "provider": "databento",
        "symbol": args.symbol,
        "asset_class": "futures",
        "primary_setup": "BNR",
        "zone_window": "09:30:00-09:30:59 America/New_York",
        "decision_available_at": "09:31:00 America/New_York",
        "timezone": args.timezone,
        "generated_at": utc_now_iso(),
        "files": entries,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {output}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
