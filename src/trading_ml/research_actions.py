from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from trading_ml.market_state_quality import (
    _followthrough_confirmation_policy_gate,
    _require_pandas,
    build_market_state_setup_quality_diagnostic,
    run_market_state_policy_simulation,
)
from trading_ml.paths import REPORTS_DIR
from trading_ml.research_controller import run_governed_research_cycle
from trading_ml.setup_redesign import build_setup_redesign_plan


ResearchActionFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ResearchActionSpec:
    action_id: str
    family: str
    description: str
    callable_kind: str
    bounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_research_actions() -> dict[str, ResearchActionSpec]:
    return {
        "domain_prior_ingestion": ResearchActionSpec(
            "domain_prior_ingestion",
            "research_foundation",
            "Ingest curated BNR domain priors into measurable hypotheses.",
            "stateful_diagnostic_action",
        ),
        "setup": ResearchActionSpec(
            "setup",
            "setup",
            "Governed setup-geometry search batch.",
            "governed_research_cycle",
        ),
        "model": ResearchActionSpec(
            "model",
            "model",
            "Governed model-family search batch.",
            "governed_research_cycle",
        ),
        "feature": ResearchActionSpec(
            "feature",
            "feature",
            "Governed feature-family search batch.",
            "governed_research_cycle",
        ),
        "feature_threshold": ResearchActionSpec(
            "feature_threshold",
            "feature_threshold",
            "Governed feature plus threshold batch.",
            "governed_research_cycle",
        ),
        "threshold": ResearchActionSpec(
            "threshold",
            "threshold",
            "Governed threshold-only search batch.",
            "governed_research_cycle",
        ),
        "translation_policy": ResearchActionSpec(
            "translation_policy",
            "translation_policy",
            "Governed score-to-trade translation batch.",
            "governed_research_cycle",
        ),
        "label": ResearchActionSpec(
            "label",
            "label",
            "Governed label-policy search batch.",
            "governed_research_cycle",
        ),
        "sample_expansion": ResearchActionSpec(
            "sample_expansion",
            "sample_expansion",
            "Governed sample-expansion batch.",
            "governed_research_cycle",
        ),
        "subtype": ResearchActionSpec(
            "subtype",
            "subtype",
            "Callable subtype cycle selected by the research director.",
            "governed_research_cycle",
        ),
        "policy_gate": ResearchActionSpec(
            "policy_gate",
            "policy_gate",
            "Callable break-quality gate cycle selected by the research director.",
            "governed_research_cycle",
        ),
        "policy_meta": ResearchActionSpec(
            "policy_meta",
            "policy_meta",
            "Callable reclaim/meta filter cycle selected by the research director.",
            "governed_research_cycle",
        ),
        "tail_path_cleanup": ResearchActionSpec(
            "tail_path_cleanup",
            "tail_path_cleanup",
            "Callable CPCV tail-path cleanup cycle.",
            "governed_research_cycle",
        ),
        "market_state_setup_quality": ResearchActionSpec(
            "market_state_setup_quality",
            "market_state_setup_quality",
            "Callable market-state/setup-quality cycle.",
            "governed_research_cycle",
        ),
        "state_gate_search": ResearchActionSpec(
            "state_gate_search",
            "market_state_setup_quality",
            "Bounded state-gate search over auction-state policy variants.",
            "stateful_diagnostic_action",
        ),
        "continuation_policy_search": ResearchActionSpec(
            "continuation_policy_search",
            "exit_behavior_research",
            "Bounded continuation-lifecycle gate search over follow-through policy variants.",
            "stateful_diagnostic_action",
        ),
        "failure_reduction_search": ResearchActionSpec(
            "failure_reduction_search",
            "market_state_setup_quality",
            "Bounded failure-cluster reduction pack combining state and continuation gates.",
            "stateful_diagnostic_action",
        ),
        "execution_stress_search": ResearchActionSpec(
            "execution_stress_search",
            "research_diagnostics",
            "Replay the governed benchmark through the execution path to measure implementation resilience.",
            "stateful_diagnostic_action",
        ),
        "ablation_pack": ResearchActionSpec(
            "ablation_pack",
            "research_diagnostics",
            "Produce component ablations for state and continuation policy candidates.",
            "stateful_diagnostic_action",
        ),
        "robust_window_rescore": ResearchActionSpec(
            "robust_window_rescore",
            "research_diagnostics",
            "Rescore policy variants with a weak-window penalty and positive-path floor.",
            "stateful_diagnostic_action",
        ),
        "exit_behavior_research": ResearchActionSpec(
            "exit_behavior_research",
            "exit_behavior_research",
            "Callable exit-behavior research cycle.",
            "governed_research_cycle",
        ),
        "candidate_universe_expansion": ResearchActionSpec(
            "candidate_universe_expansion",
            "candidate_universe_expansion",
            "Callable candidate-universe expansion cycle.",
            "governed_research_cycle",
        ),
        "setup_redesign": ResearchActionSpec(
            "setup_redesign",
            "setup",
            "Prepare a BNR setup-redesign mandate from repeated structural failure.",
            "stateful_diagnostic_action",
        ),
        "validation_failure_analysis": ResearchActionSpec(
            "validation_failure_analysis",
            "research_diagnostics",
            "Summarize why the current BNR candidate failed validation and what to try next.",
            "stateful_diagnostic_action",
        ),
        "cpcv_attribution": ResearchActionSpec(
            "cpcv_attribution",
            "research_diagnostics",
            "Load CPCV failure attribution as a callable director diagnostic.",
            "stateful_diagnostic_action",
        ),
        "ml4t_backtest": ResearchActionSpec(
            "ml4t_backtest",
            "research_diagnostics",
            "Replay the current governed benchmark through the ML4T event-driven backtest path.",
            "stateful_diagnostic_action",
        ),
        "validation_window": ResearchActionSpec(
            "validation_window",
            "validation_window",
            "Callable reserved-validation confirmation cycle.",
            "governed_research_cycle",
        ),
        "holdout_confirmation": ResearchActionSpec(
            "holdout_confirmation",
            "holdout_confirmation",
            "Callable holdout confirmation cycle.",
            "governed_research_cycle",
        ),
    }


def execute_research_action(
    action_id: str,
    *,
    base_config: dict[str, Any],
    controller_state: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = available_research_actions()
    spec = registry.get(action_id)
    if spec is None:
        raise ValueError(f"Unknown research action: {action_id}")
    if spec.callable_kind == "governed_research_cycle":
        result = run_governed_research_cycle(
            base_config,
            family=spec.family,
            controller_override=controller_state,
        )
    else:
        result = _execute_stateful_action(
            action_id,
            state=state or {},
            base_config=base_config,
            controller_state=controller_state,
        )
    return {
        **result,
        "action": spec.to_dict(),
    }


def _execute_stateful_action(
    action_id: str,
    *,
    state: dict[str, Any],
    base_config: dict[str, Any],
    controller_state: dict[str, Any],
) -> dict[str, Any]:
    if action_id == "domain_prior_ingestion":
        from trading_ml.research_os import (
            build_curated_domain_priors,
            build_hypotheses_from_priors,
            build_research_backlog,
        )

        priors = build_curated_domain_priors()
        hypotheses = build_hypotheses_from_priors(priors)
        backlog = build_research_backlog(
            hypotheses,
            list(state.get("failure_memory", []) or []),
            stage2_result=dict(state.get("stage2_result", {}) or {}),
            research_action_history=list(
                state.get("research_action_history", []) or []
            ),
        )
        return {
            "family": "research_foundation",
            "status": "complete",
            "trial_count": 0,
            "batch_decision": "inform",
            "domain_priors": priors,
            "research_backlog": backlog,
            "active_hypothesis": dict(backlog[0]) if backlog else {},
        }
    if action_id == "setup_redesign":
        plan = build_setup_redesign_plan(state)
        diagnostic = build_market_state_setup_quality_diagnostic(state)
        return {
            "family": "setup",
            "status": "complete",
            "trial_count": 0,
            "batch_decision": "inform",
            "setup_redesign_plan": {
                **plan,
                "market_state_setup_quality_diagnostic": diagnostic,
            },
            "market_state_setup_quality_diagnostic": diagnostic,
        }
    if action_id == "validation_failure_analysis":
        return _validation_failure_analysis(state)
    if action_id == "cpcv_attribution":
        return _cpcv_attribution()
    if action_id == "state_gate_search":
        return _state_gate_search(state, base_config, controller_state)
    if action_id == "continuation_policy_search":
        return _continuation_policy_search(state, base_config, controller_state)
    if action_id == "failure_reduction_search":
        return _failure_reduction_search(state, base_config, controller_state)
    if action_id == "execution_stress_search":
        return _execution_stress_search(controller_state)
    if action_id == "ablation_pack":
        return _ablation_pack(state, base_config, controller_state)
    if action_id == "robust_window_rescore":
        return _robust_window_rescore(state, base_config, controller_state)
    if action_id == "ml4t_backtest":
        from trading_ml.ml4t_backtest_adapter import run_market_state_v1_ml4t_backtest

        boundary_role = str(
            controller_state.get("boundary_role", "exploration") or "exploration"
        )
        bundle = run_market_state_v1_ml4t_backtest(boundary_role=boundary_role)
        return {
            "family": "research_diagnostics",
            "status": "complete",
            "trial_count": 1,
            "batch_decision": "inform",
            "ml4t_backtest": bundle.report,
            "artifacts": {
                "output_path": str(bundle.output_path),
                "run_dir": str(bundle.run_dir),
            },
        }
    raise ValueError(f"Unsupported stateful research action: {action_id}")


def _stateful_action_state(
    state: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    next_state = dict(state)
    next_state["stage2_config"] = dict(base_config)
    return next_state


def _state_gate_search(
    state: dict[str, Any],
    base_config: dict[str, Any],
    controller_state: dict[str, Any],
) -> dict[str, Any]:
    simulation = run_market_state_policy_simulation(
        _stateful_action_state(state, base_config)
    )
    variants = list(simulation.get("policy_variants", []) or [])
    ranked = sorted(
        variants,
        key=lambda row: (_worst_path_total(row), _float(row.get("total_pnl_r"))),
        reverse=True,
    )
    best = dict(ranked[0]) if ranked else {}
    return {
        "family": "market_state_setup_quality",
        "status": str(simulation.get("status", "pending")),
        "trial_count": int(simulation.get("trial_count", len(variants)) or 0),
        "batch_decision": "inform",
        "state_gate_summary": {
            "target_market_state": controller_state.get("target_market_state"),
            "target_environment_state": controller_state.get(
                "target_environment_state"
            ),
            "variant_count": len(variants),
            "best_variant": best.get("variant"),
        },
        "best_variant": best,
        "policy_variants": ranked,
        "diagnostic_summary": simulation.get("diagnostic_summary", {}),
    }


def _continuation_policy_search(
    state: dict[str, Any],
    base_config: dict[str, Any],
    controller_state: dict[str, Any],
) -> dict[str, Any]:
    diagnostic = build_market_state_setup_quality_diagnostic(
        _stateful_action_state(state, base_config)
    )
    rows = list(diagnostic.get("_labeled_rows", []) or [])
    if not rows:
        return {
            "family": "exit_behavior_research",
            "status": "pending",
            "trial_count": 0,
            "batch_decision": "inform",
            "reason": "labeled_rows_not_available",
        }
    pd = _require_pandas()
    gate_summary = _followthrough_confirmation_policy_gate(pd.DataFrame(rows))
    variants = list(gate_summary.get("policy_variants", []) or [])
    ranked = sorted(
        variants,
        key=lambda row: (_worst_path_total(row), _float(row.get("total_pnl_r"))),
        reverse=True,
    )
    best = dict(ranked[0]) if ranked else {}
    return {
        "family": "exit_behavior_research",
        "status": "complete",
        "trial_count": len(variants),
        "batch_decision": "inform",
        "continuation_gate_summary": {
            "target_path_class": controller_state.get("target_path_class"),
            "target_setup_state": controller_state.get("target_setup_state"),
            "best_gate": best.get("gate"),
        },
        "best_gate": best,
        "policy_variants": ranked,
    }


def _failure_reduction_search(
    state: dict[str, Any],
    base_config: dict[str, Any],
    controller_state: dict[str, Any],
) -> dict[str, Any]:
    state_gate = _state_gate_search(state, base_config, controller_state)
    continuation = _continuation_policy_search(state, base_config, controller_state)
    return {
        "family": "market_state_setup_quality",
        "status": "complete",
        "trial_count": int(state_gate.get("trial_count", 0) or 0)
        + int(continuation.get("trial_count", 0) or 0),
        "batch_decision": "inform",
        "failure_reduction_summary": {
            "target_failure_cluster": controller_state.get("target_failure_cluster"),
            "state_gate_variant": dict(state_gate.get("best_variant", {}) or {}).get(
                "variant"
            ),
            "continuation_gate": dict(continuation.get("best_gate", {}) or {}).get(
                "gate"
            ),
        },
        "state_gate_summary": state_gate.get("state_gate_summary", {}),
        "continuation_gate_summary": continuation.get("continuation_gate_summary", {}),
        "state_gate_variants": state_gate.get("policy_variants", []),
        "continuation_gate_variants": continuation.get("policy_variants", []),
    }


def _execution_stress_search(controller_state: dict[str, Any]) -> dict[str, Any]:
    from trading_ml.ml4t_backtest_adapter import run_market_state_v1_ml4t_backtest

    boundary_role = str(
        controller_state.get("boundary_role", "exploration") or "exploration"
    )
    bundle = run_market_state_v1_ml4t_backtest(boundary_role=boundary_role)
    summary = dict(bundle.report.get("backtest", {}) or {})
    return {
        "family": "research_diagnostics",
        "status": "complete",
        "trial_count": 1,
        "batch_decision": "inform",
        "execution_stress_summary": {
            "sharpe": summary.get("sharpe"),
            "max_drawdown_pct": summary.get("max_drawdown_pct"),
            "total_costs": summary.get("total_costs"),
            "profit_factor": summary.get("profit_factor"),
        },
        "artifacts": {
            "output_path": str(bundle.output_path),
            "run_dir": str(bundle.run_dir),
        },
    }


def _ablation_pack(
    state: dict[str, Any],
    base_config: dict[str, Any],
    controller_state: dict[str, Any],
) -> dict[str, Any]:
    state_gate = _state_gate_search(state, base_config, controller_state)
    continuation = _continuation_policy_search(state, base_config, controller_state)
    ablations = []
    state_best = dict(state_gate.get("best_variant", {}) or {})
    continuation_best = dict(continuation.get("best_gate", {}) or {})
    if state_best:
        ablations.append(
            {
                "component": "state_gate",
                "variant": state_best.get("variant"),
                "total_pnl_r": state_best.get("total_pnl_r"),
                "worst_path_pnl_r": _worst_path_total(state_best),
            }
        )
    if continuation_best:
        ablations.append(
            {
                "component": "continuation_gate",
                "variant": continuation_best.get("variant"),
                "gate": continuation_best.get("gate"),
                "total_pnl_r": continuation_best.get("total_pnl_r"),
                "worst_path_pnl_r": _worst_path_total(continuation_best),
            }
        )
    dependence = 0.0
    if len(ablations) == 2:
        dependence = abs(
            _float(ablations[0].get("total_pnl_r"))
            - _float(ablations[1].get("total_pnl_r"))
        )
    return {
        "family": "research_diagnostics",
        "status": "complete",
        "trial_count": len(ablations),
        "batch_decision": "inform",
        "best_ablation": max(
            ablations,
            key=lambda row: (
                _float(row.get("worst_path_pnl_r")),
                _float(row.get("total_pnl_r")),
            ),
            default={},
        ),
        "ablation_dependence_score": dependence,
        "ablations": ablations,
    }


def _robust_window_rescore(
    state: dict[str, Any],
    base_config: dict[str, Any],
    controller_state: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for row in _state_gate_search(state, base_config, controller_state).get(
        "policy_variants", []
    ):
        candidates.append(_rescored_variant("state_gate", row))
    for row in _continuation_policy_search(state, base_config, controller_state).get(
        "policy_variants", []
    ):
        candidates.append(_rescored_variant("continuation_gate", row))
    ranked = sorted(candidates, key=lambda row: row["robust_score"], reverse=True)
    return {
        "family": "research_diagnostics",
        "status": "complete",
        "trial_count": len(ranked),
        "batch_decision": "inform",
        "robust_window_summary": {
            "best_name": ranked[0]["name"] if ranked else None,
            "best_kind": ranked[0]["kind"] if ranked else None,
            "scoring_rule": "total_pnl_r + worst_path_pnl_r + 5 * positive_path_rate",
        },
        "ranked_variants": ranked,
    }


def _rescored_variant(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    total = _float(row.get("total_pnl_r"))
    positive_path_rate = _float(row.get("positive_path_rate"))
    worst_path = _worst_path_total(row)
    return {
        "name": row.get("variant"),
        "kind": kind,
        "total_pnl_r": total,
        "positive_path_rate": positive_path_rate,
        "worst_path_pnl_r": worst_path,
        "robust_score": total + worst_path + 5.0 * positive_path_rate,
    }


def _worst_path_total(row: dict[str, Any]) -> float:
    worst = list(row.get("worst_3_cpcv_paths", []) or [])
    if not worst:
        return 0.0
    return _float(worst[0].get("total_pnl_r"))


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _cpcv_attribution() -> dict[str, Any]:
    attribution = _read_json(REPORTS_DIR / "cpcv_failure_attribution.json")
    return {
        "family": "research_diagnostics",
        "status": "complete" if attribution else "missing",
        "trial_count": 0,
        "batch_decision": "inform",
        "cpcv_attribution": attribution,
    }


def _validation_failure_analysis(state: dict[str, Any]) -> dict[str, Any]:
    audit = dict(state.get("audit_summary", {}) or {})
    translation = dict(state.get("translation_summary", {}) or {})
    active_hypothesis = dict(state.get("active_hypothesis", {}) or {})
    cpcv = _read_json(REPORTS_DIR / "cpcv_failure_attribution.json")
    latest_failure = dict(
        (state.get("failure_memory", []) or [])[-1]
        if state.get("failure_memory")
        else {}
    )
    report = {
        "status": "complete",
        "hypothesis_id": active_hypothesis.get("hypothesis_id"),
        "family": active_hypothesis.get("family"),
        "failure_type": latest_failure.get("failure_type"),
        "promotion_decision": state.get("promotion_decision"),
        "current_blockers": list(state.get("blocking_issues", []) or []),
        "walk_forward_status": dict(audit.get("walk_forward", {}) or {}).get("status"),
        "cpcv_status": dict(audit.get("cpcv", {}) or {}).get("status"),
        "deflated_sharpe_status": dict(audit.get("deflated_sharpe", {}) or {}).get(
            "status"
        ),
        "multiple_testing_status": dict(audit.get("multiple_testing", {}) or {}).get(
            "status"
        ),
        "calibration_status": dict(audit.get("model_diagnostics", {}) or {})
        .get("calibration_review", {})
        .get("status"),
        "translation_status": translation.get("status"),
        "dominant_failure_axes": dict(cpcv.get("dominant_failure_axes", {}) or {}),
        "recommended_next_actions": _recommended_next_actions(state, cpcv),
    }
    return {
        "family": "research_diagnostics",
        "status": "complete",
        "trial_count": 0,
        "batch_decision": "inform",
        "validation_failure_analysis": report,
    }


def _recommended_next_actions(state: dict[str, Any], cpcv: dict[str, Any]) -> list[str]:
    recent_failure = dict(
        (state.get("failure_memory", []) or [])[-1]
        if state.get("failure_memory")
        else {}
    )
    failure_type = str(recent_failure.get("failure_type", "") or "")
    actions: list[str] = []
    if failure_type == "cpcv_tail_path_fragility":
        actions.extend(["candidate_universe_expansion", "exit_behavior_research"])
    if dict(cpcv.get("dominant_failure_axes", {}) or {}).get("subtype"):
        actions.append("setup_redesign")
    if not actions:
        actions.append("domain_prior_ingestion")
    return actions
