from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from trading_ml.langgraph_integration import (
    build_bnr_research_desk_initial_input_with_profile,
    build_governor_state_from_desk_handoff,
    compile_bnr_langgraph,
    compile_bnr_research_desk_graph,
    require_langgraph,
)


@contextlib.contextmanager
def _suppress_runtime_stderr(enabled: bool):
    if not enabled:
        yield None
        return
    sink = tempfile.NamedTemporaryFile(
        prefix="bnr-desk-runtime-", suffix=".stderr", delete=False
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


def _prune_empty(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in mapping.items()
        if value not in (None, "", [], {}, ())
    }


def _proposal_review(desk_result: dict[str, Any]) -> dict[str, Any]:
    proposal = dict((desk_result.get("desk_proposals", []) or [{}])[-1])
    return _prune_empty(
        {
            "proposal_id": proposal.get("proposal_id"),
            "node": proposal.get("node"),
            "family": proposal.get("family"),
            "action_id": proposal.get("action_id"),
            "claim": proposal.get("claim"),
            "hypothesis": proposal.get("hypothesis"),
            "target_failure_cluster": proposal.get("target_failure_cluster"),
            "target_market_state": proposal.get("target_market_state"),
            "target_setup_state": proposal.get("target_setup_state"),
            "target_environment_state": proposal.get("target_environment_state"),
            "target_path_class": proposal.get("target_path_class"),
            "allowable_knobs": proposal.get("allowable_knobs"),
            "forbidden_knobs": proposal.get("forbidden_knobs"),
            "expected_metric_delta": proposal.get("expected_metric_delta"),
            "falsification_rule": proposal.get("falsification_rule"),
            "kill_criteria": proposal.get("kill_criteria"),
        }
    )


def _plan_review(desk_result: dict[str, Any]) -> dict[str, Any]:
    plan = dict(desk_result.get("research_action_plan", {}) or {})
    controller = dict(plan.get("controller_state", {}) or {})
    return _prune_empty(
        {
            "plan_id": plan.get("plan_id"),
            "proposal_id": plan.get("proposal_id"),
            "action_id": plan.get("action_id"),
            "family": plan.get("family"),
            "objective": plan.get("objective"),
            "target_failure_cluster": plan.get("target_failure_cluster"),
            "target_market_state": plan.get("target_market_state"),
            "target_setup_state": plan.get("target_setup_state"),
            "target_environment_state": plan.get("target_environment_state"),
            "target_path_class": plan.get("target_path_class"),
            "allowed_policy_atoms": plan.get("allowed_policy_atoms"),
            "allowable_knobs": plan.get("allowable_knobs"),
            "forbidden_knobs": plan.get("forbidden_knobs"),
            "search_mechanics": plan.get("search_mechanics"),
            "robustness_target": plan.get("robustness_target"),
            "falsification_rule": plan.get("falsification_rule"),
            "kill_criteria": plan.get("kill_criteria"),
            "controller_focus": _prune_empty(
                {
                    "active_family": controller.get("active_family"),
                    "focus_setup_state": controller.get("focus_setup_state"),
                    "focus_environment_state": controller.get(
                        "focus_environment_state"
                    ),
                    "focus_path_class": controller.get("focus_path_class"),
                }
            ),
        }
    )


def _trial_delta_review(trial: dict[str, Any]) -> dict[str, Any]:
    return _prune_empty(
        {
            "trial_id": trial.get("trial_id"),
            "variant": trial.get("variant"),
            "overrides": trial.get("overrides"),
            "candidate_count": trial.get("candidate_count"),
            "trade_count": trial.get("trade_count"),
            "total_pnl_r": trial.get("total_pnl_r"),
            "net_avg_pnl_r": trial.get("net_avg_pnl_r"),
            "roc_auc": trial.get("roc_auc"),
            "profit_factor": trial.get("profit_factor"),
            "mean_cpcv_path_pnl_r": trial.get("mean_cpcv_path_pnl_r"),
            "median_cpcv_path_pnl_r": trial.get("median_cpcv_path_pnl_r"),
            "worst_path_pnl_r": trial.get("worst_path_pnl_r"),
            "positive_path_rate": trial.get("positive_path_rate"),
            "retained_fraction": trial.get("retained_fraction"),
            "decision": trial.get("decision"),
        }
    )


def _summarize_research_action(action: dict[str, Any]) -> dict[str, Any]:
    best_trial = dict(action.get("best_trial", {}) or {})
    accepted_trial = dict(action.get("accepted_trial", {}) or {})
    marginal = dict(action.get("marginal_evidence", {}) or {})
    return _prune_empty(
        {
            "cycle": action.get("cycle"),
            "family": action.get("family"),
            "action_id": action.get("action_id"),
            "proposal_id": action.get("proposal_id"),
            "hypothesis_id": action.get("hypothesis_id"),
            "status": action.get("status"),
            "batch_decision": action.get("batch_decision"),
            "callable_kind": action.get("callable_kind"),
            "best_trial": _trial_delta_review(best_trial),
            "accepted_trial": _trial_delta_review(accepted_trial),
            "marginal_evidence": _prune_empty(
                {
                    "decision": marginal.get("decision"),
                    "net_delta_vs_baseline": marginal.get("net_delta_vs_baseline"),
                    "roc_auc_delta_vs_baseline": marginal.get(
                        "roc_auc_delta_vs_baseline"
                    ),
                    "status": marginal.get("status"),
                    "trial_id": marginal.get("trial_id"),
                }
            ),
        }
    )


def _build_combined_summary(
    desk_result: dict[str, Any],
    governor_result: dict[str, Any],
    stderr_path: Path | None = None,
) -> dict[str, Any]:
    desk_summary = dict(desk_result.get("desk_summary", {}) or {})
    desk_handoff = dict(desk_summary.get("desk_memory_update", {}) or {})
    latest_proposal = dict((desk_result.get("desk_proposals", []) or [{}])[-1])
    program = dict(governor_result.get("next_step_plan", {}) or {})
    promotion = _latest_log_payload(governor_result, "promotion_decision")
    recent_actions = list(governor_result.get("research_action_history", []) or [])
    summary = {
        "desk_run_id": desk_result.get("run_id"),
        "governor_run_id": governor_result.get("run_id"),
        "desk_selected_node": dict(desk_summary.get("desk_director", {}) or {}).get(
            "selected_node"
        ),
        "desk_proposal_id": latest_proposal.get("proposal_id"),
        "desk_proposal_family": latest_proposal.get("family"),
        "desk_handoff_status": desk_handoff.get("status"),
        "governor_current_node": governor_result.get("current_node"),
        "governor_selected_family": program.get("selected_family"),
        "governor_assigned_action": program.get("assigned_research_action"),
        "governor_hypothesis_id": program.get("hypothesis_id"),
        "governor_search_batch_status": governor_result.get("search_batch_status"),
        "promotion_decision": promotion.get("decision")
        or governor_result.get("promotion_decision"),
        "run_review": {
            "proposal": _proposal_review(desk_result),
            "plan": _plan_review(desk_result),
            "latest_action": (
                _summarize_research_action(recent_actions[-1]) if recent_actions else {}
            ),
        },
        "recent_research_actions": [
            _summarize_research_action(action) for action in recent_actions[-5:]
        ],
        "blocking_issues": governor_result.get("blocking_issues", []),
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


def _should_emit_early_summary(governor_result: dict[str, Any]) -> bool:
    current_node = str(governor_result.get("current_node", "") or "")
    if current_node == "audit_agent":
        return True
    if (
        current_node == "search_controller_agent"
        and governor_result.get("search_batch_status") == "complete"
    ):
        return True
    return False


def _should_emit_early_desk_summary(desk_result: dict[str, Any]) -> bool:
    current_node = str(desk_result.get("current_node", "") or "")
    if current_node == "desk_memory_update":
        return True
    summary = dict(desk_result.get("desk_summary", {}) or {})
    handoff = dict(summary.get("desk_memory_update", {}) or {})
    return handoff.get("status") == "ready_for_governor_graph"


def _run_desk(
    desk_graph: Any,
    desk_state: dict[str, Any],
    desk_config: dict[str, Any],
    *,
    summary_only: bool,
) -> dict[str, Any]:
    if not summary_only:
        return desk_graph.invoke(desk_state, config=desk_config)
    latest = dict(desk_state)
    try:
        for event in desk_graph.stream(
            desk_state, config=desk_config, stream_mode="values"
        ):
            if isinstance(event, dict):
                latest = event
                if _should_emit_early_desk_summary(latest):
                    return latest
    except Exception:
        return desk_graph.invoke(desk_state, config=desk_config)
    return latest


def _run_governor(
    governor_graph: Any,
    governor_state: dict[str, Any],
    governor_config: dict[str, Any],
    *,
    summary_only: bool,
    summary_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if not summary_only:
        return governor_graph.invoke(governor_state, config=governor_config)
    latest = dict(governor_state)
    timeout = summary_timeout_seconds
    deadline = time.monotonic() + timeout if timeout and timeout > 0 else None

    def _stream_runner() -> dict[str, Any]:
        nonlocal latest
        for event in governor_graph.stream(
            governor_state, config=governor_config, stream_mode="values"
        ):
            if isinstance(event, dict):
                latest = event
                if _should_emit_early_summary(latest):
                    return latest
        return latest

    try:
        if deadline is None:
            return _stream_runner()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_stream_runner)
        remaining = max(0.0, deadline - time.monotonic())
        try:
            return future.result(timeout=remaining)
        except concurrent.futures.TimeoutError:
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            return latest
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        return governor_graph.invoke(governor_state, config=governor_config)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the BNR research desk graph, then hand off to the governor graph."
    )
    parser.add_argument("--thread-id", default="bnr-desk-governor")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument(
        "--unattended",
        action="store_true",
        help="Run multiple bounded cycles and auto-accept only if all hard gates pass.",
    )
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--quiet-runtime", action="store_true")
    parser.add_argument("--summary-timeout-seconds", type=float, default=90.0)
    args = parser.parse_args()

    if args.local_only:
        cache_root = tempfile.mkdtemp(prefix="bnr-desk-cache-")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_ENDPOINT"] = ""
        os.environ["LANGCHAIN_API_KEY"] = ""
        os.environ.setdefault("PYTHONWARNINGS", "ignore")
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("XDG_CACHE_HOME", cache_root)
        os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg-"))

    require_langgraph()
    desk_graph = compile_bnr_research_desk_graph(use_llm=args.use_llm)
    governor_graph = compile_bnr_langgraph(use_llm=args.use_llm)
    desk_config = {"configurable": {"thread_id": f"{args.thread_id}-desk"}}
    governor_config = {"configurable": {"thread_id": f"{args.thread_id}-governor"}}
    unattended = bool(args.unattended)
    max_cycles = int(args.max_cycles or (25 if unattended else 1))
    auto_accept_robust = unattended
    desk_budget_overrides = {
        "max_trials": 1,
        "max_full_validations": 0 if not unattended else 4,
        "max_cpcv_runs": 0 if not unattended else 4,
        "max_model_trains": 1 if not unattended else 12,
    }
    governor_budget_overrides = {
        "max_trials": 1 if not unattended else max(6, max_cycles * 2),
        "max_full_validations": 1 if not unattended else 6,
        "max_cpcv_runs": 1 if not unattended else 6,
        "max_model_trains": 1 if not unattended else 20,
    }

    suppress_runtime = bool(args.quiet_runtime or args.local_only)
    with _suppress_runtime_stderr(suppress_runtime) as stderr_path:
        desk_result = _run_desk(
            desk_graph,
            build_bnr_research_desk_initial_input_with_profile(
                preapproved_checkpoints=[
                    "bnr_spec_approval",
                    "label_approval",
                    "search_space_approval",
                    "frozen_spec_approval",
                ],
                max_research_cycles=max_cycles,
                compute_budget_overrides=desk_budget_overrides,
                runtime_profile="unattended" if unattended else "bounded_autonomous",
                auto_accept_robust=auto_accept_robust,
            ),
            desk_config,
            summary_only=bool(args.summary_only),
        )
        governor_state = build_governor_state_from_desk_handoff(
            desk_result,
            max_research_cycles=max_cycles,
            compute_budget_overrides=governor_budget_overrides,
            runtime_profile="unattended" if unattended else "bounded_autonomous",
            auto_accept_robust=auto_accept_robust,
        )
        governor_result = _run_governor(
            governor_graph,
            governor_state,
            governor_config,
            summary_only=bool(args.summary_only),
            summary_timeout_seconds=float(args.summary_timeout_seconds),
        )
        payload = (
            _build_combined_summary(desk_result, governor_result, stderr_path)
            if args.summary_only
            else {"desk_result": desk_result, "governor_result": governor_result}
        )
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
