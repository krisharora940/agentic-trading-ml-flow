from __future__ import annotations

import json

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.exit_engine import run_scratch_exit_refinement_cycle


def main() -> None:
    result = run_scratch_exit_refinement_cycle(build_agent_loop_state())
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "artifact_path": result.get("artifact_path"),
                "run_artifact_path": result.get("run_artifact_path"),
                "batch_decision": result.get("batch_decision"),
                "best_policy": _policy_summary(result.get("best_policy")),
                "policies": [_policy_summary(row) for row in result.get("ranked_policies", [])],
            },
            indent=2,
            default=str,
        )
    )


def _policy_summary(row: object) -> dict[str, object]:
    if not isinstance(row, dict):
        return {}
    keys = {
        "variant",
        "decision",
        "trade_count",
        "total_pnl_r",
        "avg_trade_r",
        "median_trade_r",
        "win_rate",
        "payoff_ratio",
        "max_drawdown_r",
        "mean_cpcv_path_pnl_r",
        "median_cpcv_path_pnl_r",
        "pbo",
        "worst_3_cpcv_paths",
        "dsr_psr",
        "improvement_attribution",
    }
    return {key: value for key, value in row.items() if key in keys}


if __name__ == "__main__":
    main()
