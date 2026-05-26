from __future__ import annotations

import json

from trading_ml.agent_workflow import pending_human_checkpoints
from trading_ml.langgraph_integration import (
    build_langgraph_initial_input,
    compile_bnr_langgraph,
)


def main() -> None:
    graph = compile_bnr_langgraph()
    state = graph.invoke(
        build_langgraph_initial_input(),
        config={"configurable": {"thread_id": "stage3-demo"}},
    )
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
