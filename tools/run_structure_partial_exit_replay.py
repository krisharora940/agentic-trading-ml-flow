from __future__ import annotations

import json

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.exit_behavior_research import run_structure_partial_exit_replay_cycle


def main() -> None:
    result = run_structure_partial_exit_replay_cycle(build_agent_loop_state())
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "artifact_path": result.get("artifact_path"),
                "run_artifact_path": result.get("run_artifact_path"),
                "batch_decision": result.get("batch_decision"),
                "best_trial": _summary(result.get("best_trial")),
                "trials": [_summary(row) for row in result.get("ranked_trials", [])],
            },
            indent=2,
            default=str,
        )
    )


def _summary(row: object) -> dict[str, object]:
    if not isinstance(row, dict):
        return {}
    keys = {
        "variant",
        "exit_family",
        "decision",
        "trade_count",
        "total_pnl_r",
        "avg_trade_r",
        "median_trade_r",
        "win_rate",
        "payoff_ratio",
        "max_drawdown_r",
        "right_tail_p90_r",
        "mean_cpcv_path_pnl_r",
        "median_cpcv_path_pnl_r",
        "pbo",
        "worst_3_cpcv_paths",
        "dsr_psr",
        "gate_results",
    }
    return {key: value for key, value in row.items() if key in keys}


if __name__ == "__main__":
    main()
