from __future__ import annotations

import json

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.candidate_universe_expansion import run_candidate_universe_expansion_cycle


def main() -> None:
    result = run_candidate_universe_expansion_cycle(build_agent_loop_state())
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "artifact_path": result.get("artifact_path"),
                "run_artifact_path": result.get("run_artifact_path"),
                "trial_count": result.get("trial_count"),
                "selected_for_next_stage": result.get("selected_for_next_stage"),
                "variants": [
                    {
                        "variant": row["variant"],
                        "candidate_count": row["candidate_count"],
                        "new_vs_baseline": row["new_deduped_candidates_vs_baseline"],
                        "session_direction_ess": row["effective_sample_size"]["session_direction_cluster_ess"],
                        "duplicate_ratio": row["deduplication"]["duplicate_ratio"],
                        "decision": row["governance_decision"],
                    }
                    for row in result.get("variant_summaries", [])
                ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
