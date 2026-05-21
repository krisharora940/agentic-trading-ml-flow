from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

from trading_ml.ml4t_backtest_adapter import run_market_state_v1_ml4t_backtest


def main() -> None:
    boundary_role = sys.argv[1] if len(sys.argv) > 1 else "exploration"
    bundle = run_market_state_v1_ml4t_backtest(boundary_role=boundary_role)
    summary = {
        "output": str(bundle.output_path),
        "run_dir": str(bundle.run_dir),
        "benchmark_name": bundle.report["benchmark_name"],
        "boundary_role": bundle.report["boundary_role"],
        "walk_forward": bundle.report["walk_forward"],
        "planned_trade_stream": bundle.report["planned_trade_stream"],
        "backtest": bundle.report["backtest"],
        "translation": bundle.report["translation"],
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
