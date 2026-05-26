from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ml.config import load_databento_manifest, load_evidence_boundary_config
from trading_ml.evidence_sources import select_manifest_source_path
from trading_ml.paths import DATA_DIR, MANIFESTS_DIR
from trading_ml.schemas import utc_now_iso
from trading_ml.stage2_data import build_data_quality_report, load_ohlcv_file


def _filter_window(df, *, start: str, end: str):
    dates = df.index.date
    mask = [(str(date) >= start) and (str(date) <= end) for date in dates]
    return df[mask].copy()


def _write_window_cache(df, *, symbol: str, timeframe: str, role: str) -> Path:
    output = DATA_DIR / "cache" / f"{symbol.lower()}_{timeframe}_{role}_opening.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    return output


def _manifest_entry(
    *,
    path: Path,
    bars,
    symbol: str,
    timeframe: str,
    timezone: str,
    role: str,
    priority: int,
) -> dict:
    quality = build_data_quality_report(
        bars,
        source_path=path,
        symbol=symbol,
        timeframe=timeframe,
        timezone=timezone,
    )
    return {
        "symbol": symbol,
        "instrument": "Micro E-mini Nasdaq-100 futures",
        "provider": "boundary_cache",
        "timeframe": timeframe,
        "schema": f"ohlcv-{timeframe}",
        "source_path": str(path),
        "timezone": timezone,
        "boundary_role": role,
        "stage2_priority": priority,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exploration/validation caches from the opening-hours MNQ source."
    )
    parser.add_argument("--manifest", default="databento_mnq_manifest.json")
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--timeframe", default="30s")
    parser.add_argument("--timezone", default="America/New_York")
    args = parser.parse_args()

    manifest = load_databento_manifest(args.manifest)
    boundary = load_evidence_boundary_config()
    source_path = select_manifest_source_path(manifest, timeframe=args.timeframe) or ""
    if not source_path:
        raise RuntimeError("No source_path available for boundary cache build.")

    bars = load_ohlcv_file(
        source_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        timezone=args.timezone,
    )
    exploration_bars = _filter_window(
        bars,
        start=boundary["exploration"]["start"],
        end=boundary["exploration"]["end"],
    )
    validation_bars = _filter_window(
        bars,
        start=boundary["validation"]["start"],
        end=boundary["validation"]["end"],
    )

    exploration_path = _write_window_cache(
        exploration_bars,
        symbol=args.symbol,
        timeframe=args.timeframe,
        role="exploration",
    )
    validation_path = _write_window_cache(
        validation_bars, symbol=args.symbol, timeframe=args.timeframe, role="validation"
    )

    entries = [
        _manifest_entry(
            path=exploration_path,
            bars=exploration_bars,
            symbol=args.symbol,
            timeframe=args.timeframe,
            timezone=args.timezone,
            role="exploration",
            priority=300,
        ),
        _manifest_entry(
            path=validation_path,
            bars=validation_bars,
            symbol=args.symbol,
            timeframe=args.timeframe,
            timezone=args.timezone,
            role="validation",
            priority=200,
        ),
    ]

    for file_entry in manifest.get("files", []):
        path = str(file_entry.get("source_path", ""))
        if "/2026-" in path or "2026-" in path:
            file_entry["boundary_role"] = "holdout"

    manifest["files"] = [
        file_entry
        for file_entry in manifest.get("files", [])
        if str(file_entry.get("source_path", ""))
        not in {str(exploration_path), str(validation_path)}
    ] + entries
    manifest["generated_at"] = utc_now_iso()
    manifest["files"].sort(
        key=lambda item: (
            item.get("timeframe") != "30s",
            {"exploration": 0, "validation": 1, "holdout": 2}.get(
                item.get("boundary_role", ""), 3
            ),
            -(item.get("stage2_priority", 0)),
            -(item.get("sessions", 0)),
            -(item.get("rows", 0)),
        )
    )

    manifest_path = MANIFESTS_DIR / args.manifest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    payload = {
        "source_path": source_path,
        "exploration_path": str(exploration_path),
        "validation_path": str(validation_path),
        "exploration_sessions": entries[0]["sessions"],
        "validation_sessions": entries[1]["sessions"],
        "holdout_marked_paths": [
            item["source_path"]
            for item in manifest["files"]
            if item.get("boundary_role") == "holdout"
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
