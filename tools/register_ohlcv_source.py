from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ml.config import load_databento_manifest
from trading_ml.schemas import utc_now_iso
from trading_ml.stage2_data import build_data_quality_report, load_ohlcv_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a normalized OHLCV source in the MNQ manifest."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--manifest", default="data/manifests/databento_mnq_manifest.json"
    )
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--timeframe", default="30s")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--provider", default="external_csv")
    parser.add_argument("--instrument", default="Micro E-mini Nasdaq-100 futures")
    parser.add_argument("--schema")
    parser.add_argument("--stage2-priority", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source)
    manifest_path = Path(args.manifest)
    manifest = (
        load_databento_manifest(manifest_path.name)
        if manifest_path.exists()
        else {
            "dataset_name": "databento_mnq_bnr",
            "provider": args.provider,
            "symbol": args.symbol,
            "asset_class": "futures",
            "primary_setup": "BNR",
            "zone_window": "09:30:00-09:30:59 America/New_York",
            "decision_available_at": "09:31:00 America/New_York",
            "timezone": args.timezone,
            "files": [],
        }
    )

    bars = load_ohlcv_file(
        source,
        symbol=args.symbol,
        timeframe=args.timeframe,
        timezone=args.timezone,
    )
    quality = build_data_quality_report(
        bars,
        source_path=source,
        symbol=args.symbol,
        timeframe=args.timeframe,
        timezone=args.timezone,
    )

    entry = {
        "symbol": args.symbol,
        "instrument": args.instrument,
        "provider": args.provider,
        "timeframe": args.timeframe,
        "schema": args.schema or f"ohlcv-{args.timeframe}",
        "source_path": str(source),
        "timezone": args.timezone,
        "stage2_priority": args.stage2_priority,
        "date_start": quality.date_start,
        "date_end": quality.date_end,
        "rows": quality.rows,
        "sessions": quality.sessions,
        "opening_zone_sessions": quality.opening_zone_sessions,
        "duplicate_timestamps": quality.duplicate_timestamps,
        "missing_regular_session_bars": quality.missing_regular_session_bars,
        "quality_flags": quality.quality_flags,
        "earliest_session_end": quality.earliest_session_end,
        "latest_session_end": quality.latest_session_end,
    }

    files = [
        item
        for item in manifest.get("files", [])
        if item.get("source_path") != str(source)
    ]
    files.append(entry)
    files.sort(
        key=lambda item: (
            item.get("timeframe") != "30s",
            -(item.get("sessions", 0)),
            -(item.get("rows", 0)),
        )
    )
    manifest["files"] = files
    manifest["generated_at"] = utc_now_iso()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"updated {manifest_path}")
    print(json.dumps(entry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
