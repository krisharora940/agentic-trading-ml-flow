from __future__ import annotations

import json

from trading_ml.agent_workflow import pending_human_checkpoints, run_linear_stage3_pass


def main() -> None:
    state = run_linear_stage3_pass()
    summary = {
        "current_node": state["current_node"],
        "promotion_decision": state["promotion_decision"],
        "blocking_issues": state["blocking_issues"],
        "pending_human_checkpoints": pending_human_checkpoints(state),
        "run_log_entries": len(state["run_log"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
