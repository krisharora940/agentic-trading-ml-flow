from __future__ import annotations

import argparse
import json

from trading_ml.langgraph_integration import build_langgraph_initial_input, compile_bnr_langgraph, require_langgraph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BNR workflow through the actual LangGraph runtime.")
    parser.add_argument("--thread-id", default="bnr-langgraph")
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()

    _, _, _, (_, Command, _) = require_langgraph()
    graph = compile_bnr_langgraph(use_llm=args.use_llm)
    config = {"configurable": {"thread_id": args.thread_id}}

    result = graph.invoke(build_langgraph_initial_input(), config=config)
    while True:
        snapshot = graph.get_state(config)
        interrupts = list(getattr(snapshot, "interrupts", ()) or ())
        if not interrupts:
            print(json.dumps(result, indent=2, default=str))
            return

        payloads = [getattr(item, "value", item) for item in interrupts]
        print(json.dumps({"interrupts": payloads}, indent=2, default=str))
        if not args.auto_approve:
            return
        result = graph.invoke(Command(resume=True), config=config)


if __name__ == "__main__":
    main()
