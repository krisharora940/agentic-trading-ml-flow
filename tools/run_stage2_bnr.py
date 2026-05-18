from __future__ import annotations

import argparse
from pathlib import Path

from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine, write_stage2_report


DEFAULT_SOURCE = "/Users/radhikaarora/Documents/Trading ML/ML V2/data/mnq_30s_2026-01-01_to_2026-02-28.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 2 BNR research engine on MNQ 30s data.")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--symbol", default="MNQ")
    parser.add_argument("--timeframe", default="30s")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--output", default="reports/stage2_bnr_report.json")
    parser.add_argument("--earliest-trigger-time", default="09:32:00")
    parser.add_argument("--latest-trigger-time", default="11:00:00")
    parser.add_argument("--horizon-bars", type=int, default=20)
    parser.add_argument("--stop-multiple", type=float, default=1.0)
    parser.add_argument("--target-multiple", type=float, default=1.5)
    parser.add_argument("--break-buffer-points", type=float, default=0.0)
    args = parser.parse_args()

    config = Stage2Config(
        source_path=args.source,
        symbol=args.symbol,
        timeframe=args.timeframe,
        timezone=args.timezone,
        earliest_trigger_time=args.earliest_trigger_time,
        latest_trigger_time=args.latest_trigger_time,
        horizon_bars=args.horizon_bars,
        stop_multiple=args.stop_multiple,
        target_multiple=args.target_multiple,
        break_buffer_points=args.break_buffer_points,
    )
    result = run_stage2_research_engine(config)
    write_stage2_report(result, Path(args.output))
    print(f"wrote {args.output}")
    print(f"zones={result['zone_count']} candidates={result['candidate_count']}")
    print(f"labels={result['label_summary']}")
    print(f"model={result['model_summary']}")


if __name__ == "__main__":
    main()
