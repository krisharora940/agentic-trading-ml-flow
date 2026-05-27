from __future__ import annotations

import json

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.candidate_universe_expansion import (
    run_candidate_universe_shortlist_diagnostic,
)


def main() -> None:
    result = run_candidate_universe_shortlist_diagnostic(build_agent_loop_state())
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "inventory_gate": result.get("inventory_gate"),
                "artifact_path": result.get("artifact_path"),
                "run_artifact_path": result.get("run_artifact_path"),
                "ranked_variants": result.get("ranked_variants", []),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
