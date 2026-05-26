from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from trading_ml.langgraph_integration import (
    build_langgraph_initial_input,
    compile_bnr_langgraph,
    require_langgraph,
)


@contextlib.contextmanager
def _suppress_runtime_stderr(enabled: bool):
    if not enabled:
        yield None
        return
    sink = tempfile.NamedTemporaryFile(
        prefix="bnr-runtime-", suffix=".stderr", delete=False
    )
    sink_path = Path(sink.name)
    sink.close()
    original_fd = os.dup(2)
    with sink_path.open("w") as handle:
        os.dup2(handle.fileno(), 2)
        try:
            yield sink_path
        finally:
            os.dup2(original_fd, 2)
            os.close(original_fd)


def _latest_log_payload(result: dict[str, Any], actor: str) -> dict[str, Any]:
    for entry in reversed(list(result.get("run_log", []) or [])):
        if entry.get("actor") == actor:
            return dict(entry.get("payload", {}) or {})
    return {}


def _build_run_summary(
    result: dict[str, Any], stderr_path: Path | None = None
) -> dict[str, Any]:
    research = dict(result.get("research_director_summary", {}) or {})
    program = dict(result.get("next_step_plan", {}) or {})
    search = dict(result.get("search_results", {}) or {})
    translation = dict(result.get("translation_summary", {}) or {})
    promotion = _latest_log_payload(result, "promotion_decision")
    iteration = _latest_log_payload(result, "iteration_controller")
    hypothesis = dict(result.get("active_hypothesis", {}) or {})
    summary = {
        "run_id": result.get("run_id"),
        "research_cycle": result.get("research_cycle"),
        "selected_family": program.get("selected_family") or search.get("family"),
        "assigned_research_action": research.get("assigned_research_action")
        or program.get("assigned_research_action"),
        "active_hypothesis_id": hypothesis.get("hypothesis_id"),
        "active_hypothesis_claim": hypothesis.get("claim"),
        "search_status": result.get("search_batch_status"),
        "trial_count": search.get("trial_count"),
        "batch_decision": search.get("batch_decision"),
        "translation_status": translation.get("translation_status"),
        "promotion_decision": promotion.get("decision")
        or result.get("promotion_decision"),
        "promotion_gate": promotion.get("promotion_gate", {}),
        "continue_iteration": iteration.get("continue_iteration"),
        "blocking_issues": result.get("blocking_issues", []),
        "recent_research_actions": list(
            result.get("research_action_history", []) or []
        )[-5:],
    }
    if stderr_path is not None and stderr_path.exists():
        lines = [
            line.strip()
            for line in stderr_path.read_text(errors="ignore").splitlines()
            if line.strip()
        ]
        if lines:
            summary["suppressed_runtime_stderr"] = {
                "line_count": len(lines),
                "sample": lines[:5],
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the BNR workflow through the actual LangGraph runtime."
    )
    parser.add_argument("--thread-id", default="bnr-langgraph")
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument(
        "--autonomous-cycle",
        action="store_true",
        help="Preapprove the bounded research checkpoints for one autonomous cycle.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Disable remote tracing and use local writable caches for autonomous runs.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Override the bounded research-cycle limit for this run.",
    )
    parser.add_argument(
        "--max-full-validations",
        type=int,
        default=None,
        help="Override the bounded full-validation budget for this run.",
    )
    parser.add_argument(
        "--max-cpcv-runs",
        type=int,
        default=None,
        help="Override the bounded CPCV budget for this run.",
    )
    parser.add_argument(
        "--max-model-trains",
        type=int,
        default=None,
        help="Override the bounded model-train budget for this run.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Override the bounded trial budget for this run.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a compact research-cycle summary instead of the full graph state.",
    )
    parser.add_argument(
        "--quiet-runtime",
        action="store_true",
        help="Suppress incidental native stderr noise during local autonomous runs.",
    )
    args = parser.parse_args()

    if args.local_only:
        cache_root = tempfile.mkdtemp(prefix="bnr-cache-")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_ENDPOINT"] = ""
        os.environ["LANGCHAIN_API_KEY"] = ""
        os.environ.setdefault("PYTHONWARNINGS", "ignore")
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("XDG_CACHE_HOME", cache_root)
        os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg-"))

    _, _, _, (_, Command, _) = require_langgraph()
    graph = compile_bnr_langgraph(use_llm=args.use_llm)
    config = {"configurable": {"thread_id": args.thread_id}}
    preapproved_checkpoints = (
        [
            "bnr_spec_approval",
            "label_approval",
            "search_space_approval",
            "frozen_spec_approval",
        ]
        if args.autonomous_cycle
        else None
    )

    budget_overrides = {
        key: value
        for key, value in {
            "max_full_validations": args.max_full_validations,
            "max_cpcv_runs": args.max_cpcv_runs,
            "max_model_trains": args.max_model_trains,
            "max_trials": args.max_trials,
        }.items()
        if value is not None
    }
    if args.autonomous_cycle:
        budget_overrides.setdefault("max_trials", 2)
        budget_overrides.setdefault("max_full_validations", 1)
        budget_overrides.setdefault("max_cpcv_runs", 1)
        budget_overrides.setdefault("max_model_trains", 4)

    suppress_runtime = bool(args.quiet_runtime or args.local_only)
    with _suppress_runtime_stderr(suppress_runtime) as stderr_path:
        result = graph.invoke(
            build_langgraph_initial_input(
                preapproved_checkpoints=preapproved_checkpoints,
                max_research_cycles=args.max_cycles,
                compute_budget_overrides=budget_overrides or None,
                runtime_profile=(
                    "bounded_autonomous" if args.autonomous_cycle else "standard"
                ),
            ),
            config=config,
        )
        while True:
            snapshot = graph.get_state(config)
            interrupts = list(getattr(snapshot, "interrupts", ()) or ())
            if not interrupts:
                payload = (
                    _build_run_summary(result, stderr_path)
                    if args.summary_only
                    else result
                )
                print(json.dumps(payload, indent=2, default=str))
                return

            payloads = [getattr(item, "value", item) for item in interrupts]
            print(json.dumps({"interrupts": payloads}, indent=2, default=str))
            if not args.auto_approve:
                return
            result = graph.invoke(Command(resume=True), config=config)


if __name__ == "__main__":
    main()
