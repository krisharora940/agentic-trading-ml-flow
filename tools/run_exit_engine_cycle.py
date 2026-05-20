from __future__ import annotations

import json

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.exit_engine import run_exit_engine_cycle


def main() -> None:
    result = run_exit_engine_cycle(build_agent_loop_state())
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "artifact_path": result.get("artifact_path"),
                "run_artifact_path": result.get("run_artifact_path"),
                "batch_decision": result.get("batch_decision"),
                "best_policy": {
                    key: value
                    for key, value in dict(result.get("best_policy") or {}).items()
                    if key
                    in {
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
                },
                "policies": [
                    {
                        "variant": row["variant"],
                        "decision": row["decision"],
                        "trade_count": row["trade_count"],
                        "total_pnl_r": row["total_pnl_r"],
                        "avg_trade_r": row["avg_trade_r"],
                        "median_trade_r": row["median_trade_r"],
                        "win_rate": row["win_rate"],
                        "payoff_ratio": row["payoff_ratio"],
                        "max_drawdown_r": row["max_drawdown_r"],
                        "mean_cpcv_path_pnl_r": row["mean_cpcv_path_pnl_r"],
                        "median_cpcv_path_pnl_r": row["median_cpcv_path_pnl_r"],
                        "pbo": row["pbo"],
                        "worst_3_cpcv_paths": row["worst_3_cpcv_paths"],
                        "dsr_psr": row["dsr_psr"],
                        "improvement_attribution": row["improvement_attribution"],
                    }
                    for row in result.get("ranked_policies", [])
                ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
