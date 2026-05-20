from __future__ import annotations

import json

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.exit_behavior_research import run_exit_behavior_research_cycle


def main() -> None:
    result = run_exit_behavior_research_cycle(build_agent_loop_state())
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "artifact_path": result.get("artifact_path"),
                "run_artifact_path": result.get("run_artifact_path"),
                "trade_count": result.get("stage_1_trade_path_diagnostics", {}).get("trade_count"),
                "taxonomy_counts": result.get("stage_1_trade_path_diagnostics", {}).get("taxonomy_counts"),
                "candidate_exit_families": result.get("stage_3_candidate_exit_families"),
                "full_validation_shortlist": [
                    {
                        "variant": row.get("variant"),
                        "exit_family": row.get("exit_family"),
                        "total_pnl_r": row.get("total_pnl_r"),
                        "max_drawdown_r": row.get("max_drawdown_r"),
                        "pbo": row.get("pbo"),
                        "mean_cpcv_path_pnl_r": row.get("mean_cpcv_path_pnl_r"),
                    }
                    for row in result.get("stage_5_full_validation_shortlist", [])
                ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
