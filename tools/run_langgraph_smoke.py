from __future__ import annotations

from trading_ml.langgraph_integration import build_langgraph_initial_input, compile_bnr_langgraph


def main() -> None:
    graph = compile_bnr_langgraph()
    config = {"configurable": {"thread_id": "bnr-smoke"}}
    result = graph.invoke(build_langgraph_initial_input(), config=config)
    print(result)


if __name__ == "__main__":
    main()
