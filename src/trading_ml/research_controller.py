from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any
from uuid import uuid4

from trading_ml.break_quality_policy import (
    apply_break_quality_policy,
    get_break_quality_policies,
)
from trading_ml.candidate_universe_expansion import (
    build_candidate_universe_expansion_space,
    run_candidate_universe_expansion_cycle,
)
from trading_ml.config import load_bnr_config, load_global_config
from trading_ml.bnr_subtypes import list_bnr_subtypes
from trading_ml.deflated_sharpe_analysis import (
    compute_sharpe_ratio,
    deflated_sharpe_probability,
)
from trading_ml.evidence_sources import select_manifest_source_path
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.exit_behavior_research import (
    build_exit_behavior_research_space,
    run_exit_behavior_research_cycle,
)
from trading_ml.feature_families import list_feature_families
from trading_ml.market_state_quality import (
    market_state_policy_variant_specs,
    run_market_state_policy_simulation,
)
from trading_ml.paths import REPORTS_DIR
from trading_ml.reclaim_meta_policy import (
    apply_reclaim_meta_policy,
    get_reclaim_meta_policies,
)
from trading_ml.registry import append_experiment_record
from trading_ml.schemas import ExperimentRecord
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.tail_path_cleanup_policy import (
    apply_tail_path_cleanup_policy,
    get_tail_path_cleanup_policies,
)
from trading_ml.translation_analysis import build_translation_analysis
from trading_ml.translation_policy import (
    get_regime_size_policies,
    get_regime_throttle_policies,
    get_sizing_policies,
)
from trading_ml.utility_analysis import compute_execution_utility
from trading_ml.validation_audit import build_validation_audit


@dataclass(slots=True)
class ControllerTrialSummary:
    trial_id: str
    family: str
    spec_version: str
    overrides: dict[str, Any]
    candidate_count: int
    positive_rate: float
    avg_pnl_r: float
    net_avg_pnl_r: float
    net_delta_vs_baseline: float
    roc_auc: float | None
    roc_auc_delta_vs_baseline: float | None
    precision: float | None
    recall: float | None
    status: str
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_search_space() -> dict[str, Any]:
    return load_bnr_config()["search_v1"]


def build_model_search_space() -> dict[str, Any]:
    return load_bnr_config()["model_search_v1"]


def build_feature_search_space() -> dict[str, Any]:
    return load_bnr_config()["feature_search_v1"]


def build_feature_threshold_search_space() -> dict[str, Any]:
    return load_bnr_config()["feature_threshold_search_v1"]


def build_threshold_search_space() -> dict[str, Any]:
    return load_bnr_config()["threshold_search_v1"]


def build_translation_policy_search_space() -> dict[str, Any]:
    return load_bnr_config()["translation_policy_search_v1"]


def build_label_search_space() -> dict[str, Any]:
    return load_bnr_config()["label_search_v1"]


def build_sample_expansion_search_space() -> dict[str, Any]:
    return load_bnr_config()["sample_expansion_search_v1"]


def build_subtype_search_space() -> dict[str, Any]:
    return load_bnr_config()["subtype_search_v1"]


def build_policy_gate_search_space() -> dict[str, Any]:
    return {
        "description": "Governed break-quality gate comparison batch.",
        "max_batch_trials": len(get_break_quality_policies()),
        "policies": [
            {"name": policy["name"]} for policy in get_break_quality_policies()
        ],
    }


def build_policy_meta_search_space() -> dict[str, Any]:
    return {
        "description": "Governed reclaim/meta policy comparison batch.",
        "max_batch_trials": len(get_reclaim_meta_policies()),
        "policies": [
            {"name": policy["name"]} for policy in get_reclaim_meta_policies()
        ],
    }


def build_tail_path_cleanup_search_space() -> dict[str, Any]:
    config = load_bnr_config()
    configured = dict(config.get("tail_path_cleanup_search_v1", {}) or {})
    policy_catalog = {
        policy["name"]: policy for policy in get_tail_path_cleanup_policies()
    }
    policies = list(
        configured.get("policies", [])
        or [{"name": policy["name"]} for policy in get_tail_path_cleanup_policies()]
    )
    executable = [
        row
        for row in policies
        if policy_catalog.get(str(row.get("name")), {}).get("executable", True)
        is not False
    ]
    diagnostic_only = [
        {
            "name": row.get("name"),
            "reason": policy_catalog.get(str(row.get("name")), {}).get(
                "governance_reason", "diagnostic_only"
            ),
        }
        for row in policies
        if policy_catalog.get(str(row.get("name")), {}).get("executable", True) is False
    ]
    return {
        "description": configured.get("description", "CPCV tail-path cleanup batch."),
        "max_batch_trials": min(
            int(configured.get("max_batch_trials", len(executable)) or len(executable)),
            len(executable),
        ),
        "diagnostic_artifact": configured.get(
            "diagnostic_artifact", "reports/cpcv_failure_attribution.json"
        ),
        "policies": executable,
        "diagnostic_only_policies": diagnostic_only,
    }


def build_market_state_setup_quality_search_space() -> dict[str, Any]:
    variants = market_state_policy_variant_specs()
    return {
        "description": "Tiny governed market-state/setup-quality policy simulation batch.",
        "max_batch_trials": len(variants),
        "family": "market_state_setup_quality",
        "variants": [{"name": variant["name"]} for variant in variants],
        "disallowed_knobs": [
            "model_family",
            "threshold",
            "holdout",
            "broad_feature_search",
        ],
    }


def build_exit_behavior_research_search_space() -> dict[str, Any]:
    return build_exit_behavior_research_space()


def build_candidate_universe_expansion_search_space() -> dict[str, Any]:
    return build_candidate_universe_expansion_space()


def load_controller_config(override: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_bnr_config()
    controller = dict(config.get("controller", {}))
    frozen = dict(config.get("frozen_benchmark", {}))
    controller.setdefault("spec_version", "bnr_spec_vA")
    controller.setdefault("active_family", "setup")
    controller.setdefault("active_model_family", "linear_baseline")
    controller.setdefault(
        "frozen_threshold", config.get("controller", {}).get("frozen_threshold", 0.45)
    )
    controller.setdefault(
        "benchmark_name",
        config.get("controller", {}).get("benchmark_name", "bnr_hybrid_linear_v1"),
    )
    controller.setdefault("benchmark_policy_gate", frozen.get("policy_gate"))
    controller.setdefault("benchmark_meta_policy", frozen.get("policy_meta"))
    controller.setdefault("benchmark_sizing_policy", frozen.get("sizing_policy"))
    controller.setdefault(
        "benchmark_regime_throttle_policy", frozen.get("regime_throttle_policy")
    )
    controller.setdefault(
        "benchmark_regime_size_policy", frozen.get("regime_size_policy")
    )
    controller.setdefault("parent_benchmark_id", controller.get("benchmark_name"))
    controller.setdefault("min_candidate_ratio_vs_baseline", 0.7)
    controller.setdefault("require_positive_net_delta", True)
    controller.setdefault("min_roc_auc_delta", 0.0)
    if override:
        controller.update(override)
    return controller


def generate_search_trials(
    base_config: dict[str, Any],
    family: str | None = None,
    controller_override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    active_family = (
        family or load_controller_config(controller_override)["active_family"]
    )
    controller = load_controller_config(controller_override)
    trials: list[dict[str, Any]] = []
    if active_family == "setup":
        search_v1 = build_search_space()
        space = search_v1["space"]
        ordered_keys = [
            "earliest_trigger_time",
            "horizon_bars",
            "target_multiple",
            "break_buffer_points",
        ]
        values = [space[key] for key in ordered_keys]
        for combo in product(*values):
            trial = dict(base_config)
            trial.update(dict(zip(ordered_keys, combo, strict=True)))
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    if active_family == "model":
        search_v1 = build_model_search_space()
        for model_family in search_v1["space"]["model_family"]:
            trial = dict(base_config)
            trial["model_family"] = model_family
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    if active_family == "feature":
        search_v1 = build_feature_search_space()
        valid_families = set(list_feature_families())
        feature_families = _focused_feature_families(
            search_v1["space"]["feature_family"], controller
        )
        for feature_family in feature_families:
            if feature_family not in valid_families:
                continue
            trial = dict(base_config)
            trial["feature_family"] = feature_family
            trial.update(_focus_trial_fields(controller))
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    if active_family == "feature_threshold":
        search_v1 = build_feature_threshold_search_space()
        valid_families = set(list_feature_families())
        for feature_family, decision_threshold in product(
            search_v1["space"]["feature_family"],
            search_v1["space"]["decision_threshold"],
        ):
            if feature_family not in valid_families:
                continue
            trial = dict(base_config)
            trial["feature_family"] = feature_family
            trial["decision_threshold"] = float(decision_threshold)
            trial.update(_focus_trial_fields(controller))
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    if active_family == "threshold":
        search_v1 = build_threshold_search_space()
        for decision_threshold in search_v1["space"]["decision_threshold"]:
            trial = dict(base_config)
            trial["decision_threshold"] = float(decision_threshold)
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    if active_family == "translation_policy":
        search_v1 = build_translation_policy_search_space()
        valid_sizing = {policy["name"] for policy in get_sizing_policies()}
        valid_throttles = {policy["name"] for policy in get_regime_throttle_policies()}
        valid_regime_sizes = {policy["name"] for policy in get_regime_size_policies()}
        for (
            decision_threshold,
            sizing_policy,
            regime_throttle_policy,
            regime_size_policy,
        ) in product(
            search_v1["space"]["decision_threshold"],
            search_v1["space"]["sizing_policy"],
            search_v1["space"]["regime_throttle_policy"],
            search_v1["space"]["regime_size_policy"],
        ):
            if (
                sizing_policy not in valid_sizing
                or regime_throttle_policy not in valid_throttles
                or regime_size_policy not in valid_regime_sizes
            ):
                continue
            trial = dict(base_config)
            trial["decision_threshold"] = float(decision_threshold)
            trial["sizing_policy"] = sizing_policy
            trial["regime_throttle_policy"] = regime_throttle_policy
            trial["regime_size_policy"] = regime_size_policy
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    if active_family == "label":
        search_v1 = build_label_search_space()
        ordered_keys = ["horizon_bars", "stop_multiple", "target_multiple"]
        values = [search_v1["space"][key] for key in ordered_keys]
        for combo in product(*values):
            trial = dict(base_config)
            trial.update(dict(zip(ordered_keys, combo, strict=True)))
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    if active_family == "sample_expansion":
        search_v1 = build_sample_expansion_search_space()
        ordered_keys = [
            "earliest_trigger_time",
            "latest_trigger_time",
            "break_buffer_points",
        ]
        values = [search_v1["space"][key] for key in ordered_keys]
        for combo in product(*values):
            trial = dict(base_config)
            trial.update(dict(zip(ordered_keys, combo, strict=True)))
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    if active_family == "subtype":
        search_v1 = build_subtype_search_space()
        valid_subtypes = set(list_bnr_subtypes())
        allowed_subtypes = set(controller.get("allowed_setup_subtypes", []) or [])
        for setup_subtype in search_v1["space"]["setup_subtype"]:
            if allowed_subtypes and setup_subtype not in allowed_subtypes:
                continue
            if setup_subtype not in valid_subtypes:
                continue
            trial = dict(base_config)
            trial["setup_subtype"] = setup_subtype
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    raise ValueError(f"Unsupported search family: {active_family}")


def run_governed_search(
    base_config: dict[str, Any], controller_override: dict[str, Any] | None = None
) -> dict[str, Any]:
    return run_governed_research_cycle(
        base_config, controller_override=controller_override
    )


def run_governed_research_cycle(
    base_config: dict[str, Any],
    family: str | None = None,
    controller_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controller = load_controller_config(controller_override)
    active_family = family or str(controller["active_family"])
    if active_family == "policy_gate":
        return run_policy_gate_cycle(base_config, controller)
    if active_family == "policy_meta":
        return run_policy_meta_cycle(base_config, controller)
    if active_family == "tail_path_cleanup":
        return run_tail_path_cleanup_cycle(base_config, controller)
    if active_family == "market_state_setup_quality":
        return run_market_state_setup_quality_cycle(base_config, controller)
    if active_family == "exit_behavior_research":
        return run_exit_behavior_research_cycle(
            {"stage2_config": base_config, "controller_state": controller}
        )
    if active_family == "candidate_universe_expansion":
        return run_candidate_universe_expansion_cycle(
            {"stage2_config": base_config, "controller_state": controller}
        )
    if active_family == "translation_policy":
        return run_translation_policy_cycle(base_config, controller)
    if active_family == "validation_window":
        return run_boundary_confirmation_cycle(
            base_config, controller, boundary_role="validation"
        )
    if active_family == "holdout_confirmation":
        return run_boundary_confirmation_cycle(
            base_config, controller, boundary_role="holdout"
        )
    result_cache: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    baseline_result = _cached_stage2_result(base_config, result_cache)
    baseline = _summarize_baseline(
        active_family, str(controller["spec_version"]), baseline_result, controller
    )
    trials = generate_search_trials(
        base_config, family=active_family, controller_override=controller
    )
    results: list[ControllerTrialSummary] = []
    for idx, trial_config in enumerate(trials, start=1):
        result = _cached_stage2_result(trial_config, result_cache)
        summary = _summarize_trial(
            trial_number=idx,
            family=active_family,
            spec_version=str(controller["spec_version"]),
            config=trial_config,
            result=result,
            baseline=baseline,
            controller=controller,
        )
        results.append(summary)
        _record_trial(active_family, summary, trial_config, result)

    ranked = sorted(
        results,
        key=lambda item: (item.net_avg_pnl_r, item.roc_auc or float("-inf")),
        reverse=True,
    )
    accepted = next((trial for trial in ranked if trial.decision == "accept"), None)
    _record_baseline(active_family, baseline, base_config, baseline_result)
    return {
        "family": active_family,
        "spec_version": controller["spec_version"],
        "focus_slice": _focus_trial_fields(controller),
        "space": (
            build_search_space()
            if active_family == "setup"
            else (
                build_model_search_space()
                if active_family == "model"
                else (
                    build_feature_search_space()
                    if active_family == "feature"
                    else (
                        build_feature_threshold_search_space()
                        if active_family == "feature_threshold"
                        else (
                            build_threshold_search_space()
                            if active_family == "threshold"
                            else (
                                build_translation_policy_search_space()
                                if active_family == "translation_policy"
                                else (
                                    build_sample_expansion_search_space()
                                    if active_family == "sample_expansion"
                                    else (
                                        build_subtype_search_space()
                                        if active_family == "subtype"
                                        else (
                                            build_market_state_setup_quality_search_space()
                                            if active_family
                                            == "market_state_setup_quality"
                                            else (
                                                build_exit_behavior_research_search_space()
                                                if active_family
                                                == "exit_behavior_research"
                                                else (
                                                    build_candidate_universe_expansion_search_space()
                                                    if active_family
                                                    == "candidate_universe_expansion"
                                                    else build_label_search_space()
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
        "controller": controller,
        "baseline": baseline.to_dict(),
        "trial_count": len(results),
        "ranked_trials": [trial.to_dict() for trial in ranked],
        "best_trial": ranked[0].to_dict() if ranked else None,
        "accepted_trial": accepted.to_dict() if accepted else None,
        "batch_decision": "accept" if accepted is not None else "revise",
    }


def _summarize_baseline(
    family: str, spec_version: str, result: dict[str, Any], controller: dict[str, Any]
) -> ControllerTrialSummary:
    label_summary = result.get("label_summary", {})
    model_summary = result.get("model_summary", {})
    metrics = model_summary.get("metrics", {})
    return ControllerTrialSummary(
        trial_id="baseline",
        family=family,
        spec_version=spec_version,
        overrides={},
        candidate_count=int(result.get("candidate_count", 0)),
        positive_rate=float(label_summary.get("positive_rate", 0.0) or 0.0),
        avg_pnl_r=float(label_summary.get("avg_pnl_r", 0.0) or 0.0),
        net_avg_pnl_r=_estimate_net_avg_pnl_r(
            result,
            threshold=(
                controller.get("frozen_threshold")
                if family in {"feature", "model", "threshold"}
                else None
            ),
        ),
        net_delta_vs_baseline=0.0,
        roc_auc=float(metrics["roc_auc"]) if "roc_auc" in metrics else None,
        roc_auc_delta_vs_baseline=0.0 if "roc_auc" in metrics else None,
        precision=float(metrics["precision"]) if "precision" in metrics else None,
        recall=float(metrics["recall"]) if "recall" in metrics else None,
        status=str(model_summary.get("status", "unknown")),
        decision="baseline",
    )


def _summarize_trial(
    *,
    trial_number: int,
    family: str,
    spec_version: str,
    config: dict[str, Any],
    result: dict[str, Any],
    baseline: ControllerTrialSummary,
    controller: dict[str, Any],
) -> ControllerTrialSummary:
    label_summary = result.get("label_summary", {})
    model_summary = result.get("model_summary", {})
    metrics = model_summary.get("metrics", {})
    threshold = (
        float(config["decision_threshold"])
        if family in {"threshold", "feature_threshold"}
        and "decision_threshold" in config
        else (
            controller.get("frozen_threshold")
            if family in {"feature", "feature_threshold", "model", "threshold", "label"}
            else None
        )
    )
    net_avg_pnl_r = _estimate_net_avg_pnl_r(result, threshold=threshold)
    roc_auc = float(metrics["roc_auc"]) if "roc_auc" in metrics else None
    roc_delta = (
        None
        if roc_auc is None or baseline.roc_auc is None
        else roc_auc - baseline.roc_auc
    )
    summary = ControllerTrialSummary(
        trial_id=f"trial-{trial_number:03d}",
        family=family,
        spec_version=spec_version,
        overrides=_trial_overrides(family, config),
        candidate_count=int(result.get("candidate_count", 0)),
        positive_rate=float(label_summary.get("positive_rate", 0.0) or 0.0),
        avg_pnl_r=float(label_summary.get("avg_pnl_r", 0.0) or 0.0),
        net_avg_pnl_r=net_avg_pnl_r,
        net_delta_vs_baseline=net_avg_pnl_r - baseline.net_avg_pnl_r,
        roc_auc=roc_auc,
        roc_auc_delta_vs_baseline=roc_delta,
        precision=float(metrics["precision"]) if "precision" in metrics else None,
        recall=float(metrics["recall"]) if "recall" in metrics else None,
        status=str(model_summary.get("status", "unknown")),
        decision="revise",
    )
    summary.decision = _decide_trial(summary, baseline, controller)
    return summary


def _trial_overrides(family: str, config: dict[str, Any]) -> dict[str, Any]:
    if family == "model":
        return {"model_family": config["model_family"]}
    if family == "feature":
        return {
            "feature_family": config["feature_family"],
            **_focus_overrides(config),
        }
    if family == "feature_threshold":
        return {
            "feature_family": config["feature_family"],
            "decision_threshold": float(config["decision_threshold"]),
            **_focus_overrides(config),
        }
    if family == "threshold":
        return {"decision_threshold": float(config["decision_threshold"])}
    if family == "translation_policy":
        return {
            "decision_threshold": float(config["decision_threshold"]),
            "sizing_policy": str(config["sizing_policy"]),
            "regime_throttle_policy": str(config["regime_throttle_policy"]),
            "regime_size_policy": str(config["regime_size_policy"]),
        }
    if family == "label":
        return {
            "horizon_bars": config["horizon_bars"],
            "stop_multiple": config["stop_multiple"],
            "target_multiple": config["target_multiple"],
        }
    if family == "sample_expansion":
        return {
            "earliest_trigger_time": config["earliest_trigger_time"],
            "latest_trigger_time": config["latest_trigger_time"],
            "break_buffer_points": config["break_buffer_points"],
        }
    if family == "subtype":
        return {"setup_subtype": config["setup_subtype"]}
    return {
        "earliest_trigger_time": config["earliest_trigger_time"],
        "horizon_bars": config["horizon_bars"],
        "target_multiple": config["target_multiple"],
        "break_buffer_points": config["break_buffer_points"],
    }


def _focus_overrides(config: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "focus_setup_state": config.get("focus_setup_state"),
        "focus_environment_state": config.get("focus_environment_state"),
        "focus_path_class": config.get("focus_path_class"),
    }
    return {
        key: value
        for key, value in mapping.items()
        if value not in {None, "", "unknown"}
    }


def _focus_trial_fields(controller: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "focus_setup_state": controller.get("focus_setup_state"),
            "focus_environment_state": controller.get("focus_environment_state"),
            "focus_path_class": controller.get("focus_path_class"),
        }.items()
        if value not in {None, "", "unknown"}
    }


def _focused_feature_families(
    feature_families: list[str], controller: dict[str, Any]
) -> list[str]:
    setup_state = str(controller.get("focus_setup_state", "") or "")
    environment_state = str(controller.get("focus_environment_state", "") or "")
    path_class = str(controller.get("focus_path_class", "") or "")
    priority: list[str] = []
    if environment_state in {
        "volatile_chop",
        "balance_chop",
        "trend_expansion",
        "trend_day",
    }:
        priority.extend(
            ["context_plus_regime", "reclaim_plus_regime", "regime_features"]
        )
    if setup_state in {
        "continuation",
        "late_followthrough",
        "repair",
        "failed_reclaim",
        "weak_confirmation",
    }:
        priority.extend(["context_plus_reclaim", "bnr_plus_context", "pivot_reclaim"])
    if path_class in {"chop", "failure", "runner", "delayed_runner"}:
        priority.extend(["context_plus_geometry", "bnr_core", "pre_trigger_context"])
    seen: set[str] = set()
    ordered: list[str] = []
    for family in [*priority, *feature_families]:
        if family in seen:
            continue
        seen.add(family)
        ordered.append(family)
    trial_limit = int(controller.get("max_batch_trials", len(ordered)) or len(ordered))
    return ordered[: max(1, trial_limit)]


def _decide_trial(
    trial: ControllerTrialSummary,
    baseline: ControllerTrialSummary,
    controller: dict[str, Any],
) -> str:
    if trial.status != "fit":
        return "reject"
    min_candidate_ratio = float(controller["min_candidate_ratio_vs_baseline"])
    if (
        baseline.candidate_count > 0
        and (trial.candidate_count / baseline.candidate_count) < min_candidate_ratio
    ):
        return "reject"
    if (
        bool(controller["require_positive_net_delta"])
        and trial.net_delta_vs_baseline <= 0
    ):
        return "reject"
    min_roc_auc_delta = float(controller["min_roc_auc_delta"])
    if (
        trial.roc_auc_delta_vs_baseline is not None
        and trial.roc_auc_delta_vs_baseline < min_roc_auc_delta
    ):
        return "reject"
    return "accept"


def _record_baseline(
    family: str,
    baseline: ControllerTrialSummary,
    config: dict[str, Any],
    result: dict[str, Any],
) -> None:
    accounting = _trial_accounting(family, config, result, selected_by="baseline")
    append_experiment_record(
        ExperimentRecord(
            experiment_id=f"{baseline.spec_version}-{family}-baseline",
            hypothesis=f"Frozen {family} baseline for {baseline.spec_version}.",
            config_ref=baseline.spec_version,
            data_slice={
                "source_path": config["source_path"],
                "symbol": config["symbol"],
                "timeframe": config["timeframe"],
            },
            result={
                "overrides": {},
                "candidate_count": result.get("candidate_count", 0),
                "label_summary": result.get("label_summary", {}),
                "model_summary": result.get("model_summary", {}),
                "net_avg_pnl_r": baseline.net_avg_pnl_r,
                "trial_accounting": accounting,
            },
            decision=baseline.decision,
            phase="exploration",
        )
    )


def _record_trial(
    family: str,
    summary: ControllerTrialSummary,
    config: dict[str, Any],
    result: dict[str, Any],
) -> None:
    accounting = _trial_accounting(family, config, result, selected_by="controller")
    append_experiment_record(
        ExperimentRecord(
            experiment_id=f"{summary.spec_version}-{family}-{summary.trial_id}",
            hypothesis=f"Test {family} overrides against frozen baseline.",
            config_ref=summary.spec_version,
            data_slice={
                "source_path": config["source_path"],
                "symbol": config["symbol"],
                "timeframe": config["timeframe"],
            },
            result={
                "overrides": summary.overrides,
                "candidate_count": result.get("candidate_count", 0),
                "label_summary": result.get("label_summary", {}),
                "model_summary": result.get("model_summary", {}),
                "net_avg_pnl_r": summary.net_avg_pnl_r,
                "net_delta_vs_baseline": summary.net_delta_vs_baseline,
                "roc_auc_delta_vs_baseline": summary.roc_auc_delta_vs_baseline,
                "trial_accounting": accounting,
            },
            decision=summary.decision,
            phase="exploration",
        )
    )


def _estimate_net_avg_pnl_r(
    result: dict[str, Any], threshold: float | None = None
) -> float:
    prediction_based = _estimate_prediction_net_avg_pnl_r(result, threshold)
    if prediction_based is not None:
        return prediction_based

    global_config = load_global_config()
    label_summary = result.get("label_summary", {})
    avg_pnl_r = float(label_summary.get("avg_pnl_r", 0.0) or 0.0)
    model_summary = result.get("model_summary", {})
    avg_risk_points = float(model_summary.get("avg_risk_points", 0.0) or 0.0)
    avg_entry_price = float(model_summary.get("avg_entry_price", 0.0) or 0.0)
    if avg_risk_points <= 0 or avg_entry_price <= 0:
        return avg_pnl_r
    cost_points = _round_trip_cost_points(avg_entry_price, global_config)
    cost_r = cost_points / avg_risk_points
    return avg_pnl_r - cost_r


def _estimate_prediction_net_avg_pnl_r(
    result: dict[str, Any], threshold: float | None
) -> float | None:
    if threshold is None:
        return None

    global_config = load_global_config()
    model_summary = dict(result.get("model_summary", {}))
    prediction_records = list(model_summary.get("prediction_records", []))
    avg_risk_points = float(model_summary.get("avg_risk_points", 0.0) or 0.0)
    avg_entry_price = float(model_summary.get("avg_entry_price", 0.0) or 0.0)
    if not prediction_records or avg_risk_points <= 0 or avg_entry_price <= 0:
        return None

    selected = [
        row
        for row in prediction_records
        if float(row.get("probability", 0.0) or 0.0) >= float(threshold)
    ]
    if not selected:
        return 0.0

    avg_pnl_r = sum(float(row.get("pnl_r", 0.0) or 0.0) for row in selected) / len(
        selected
    )
    cost_points = _round_trip_cost_points(avg_entry_price, global_config)
    cost_r = cost_points / avg_risk_points
    return avg_pnl_r - cost_r


def _round_trip_cost_points(
    avg_entry_price: float, global_config: dict[str, Any]
) -> float:
    slippage = dict(global_config.get("slippage", {}))
    model = str(slippage.get("model", "ticks"))
    if model == "ticks":
        tick_size = float(slippage.get("tick_size", 0.25) or 0.25)
        profile = str(slippage.get("profile", "base"))
        ticks_per_side = (
            float(slippage.get("stressed_ticks_per_side", 6.0) or 6.0)
            if profile == "stressed"
            else float(slippage.get("base_ticks_per_side", 3.0) or 3.0)
        )
        return 2.0 * tick_size * ticks_per_side
    entry_bps = float(slippage.get("entry_bps", 0.0) or 0.0)
    exit_bps = float(slippage.get("exit_bps", 0.0) or 0.0)
    total_bps = entry_bps + exit_bps
    return avg_entry_price * (total_bps / 10_000.0)


def _cached_stage2_result(
    config: dict[str, Any], cache: dict[tuple[tuple[str, Any], ...], dict[str, Any]]
) -> dict[str, Any]:
    stage2_kwargs = _stage2_config_kwargs(config)
    cache_key = tuple(sorted(stage2_kwargs.items()))
    if cache_key not in cache:
        cache[cache_key] = run_stage2_research_engine(Stage2Config(**stage2_kwargs))
    return cache[cache_key]


def _trial_accounting(
    family: str, config: dict[str, Any], result: dict[str, Any], *, selected_by: str
) -> dict[str, Any]:
    return {
        "family": family,
        "config_hash": _stable_hash(config),
        "features_hash": _stable_hash(result.get("features_records", [])),
        "label_hash": _stable_hash(result.get("labels_records", [])),
        "model_hash": _stable_hash(
            {"model_family": result.get("config", {}).get("model_family")}
        ),
        "threshold_hash": _stable_hash(
            {"decision_threshold": config.get("decision_threshold")}
        ),
        "translation_policy_hash": _stable_hash(
            {
                "sizing_policy": config.get("sizing_policy"),
                "regime_throttle_policy": config.get("regime_throttle_policy"),
                "regime_size_policy": config.get("regime_size_policy"),
            }
        ),
        "validation_window_hash": _stable_hash(
            {"source_path": config.get("source_path")}
        ),
        "selected_by": selected_by,
    }


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _stage2_config_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "source_path",
        "symbol",
        "timeframe",
        "timezone",
        "earliest_trigger_time",
        "latest_trigger_time",
        "horizon_bars",
        "stop_multiple",
        "target_multiple",
        "break_buffer_points",
        "spec_name",
        "model_family",
        "feature_family",
    }
    return {key: value for key, value in config.items() if key in allowed_keys}


def run_policy_gate_cycle(
    base_config: dict[str, Any], controller: dict[str, Any]
) -> dict[str, Any]:
    result = run_stage2_research_engine(
        Stage2Config(**_stage2_config_kwargs(base_config))
    )
    validation = build_validation_audit(result, {})
    rows: list[dict[str, Any]] = []
    for policy in get_break_quality_policies()[
        : _policy_trial_limit(controller, len(get_break_quality_policies()))
    ]:
        filtered = _apply_benchmark_contract(
            result,
            validation,
            threshold=float(controller.get("frozen_threshold", 0.45) or 0.45),
            gate_policy=policy["name"],
            meta_policy=controller.get("benchmark_meta_policy"),
        )
        execution = run_event_driven_policy_backtest(
            filtered,
            threshold=float(controller.get("frozen_threshold", 0.45) or 0.45),
            sizing_policy=controller.get("benchmark_sizing_policy"),
            regime_throttle_policy=controller.get("benchmark_regime_throttle_policy"),
            regime_size_policy=controller.get("benchmark_regime_size_policy"),
        )
        utility = (
            compute_execution_utility(execution)
            if execution.get("status") == "complete"
            else {"score": None}
        )
        rows.append(
            {
                "trial_id": f"trial-gate-{policy['name']}",
                "family": "policy_gate",
                "overrides": {"policy_gate": policy["name"]},
                "trade_count": int(execution.get("trade_count", 0) or 0),
                "total_pnl_r": float(execution.get("total_pnl_r", 0.0) or 0.0),
                "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
                "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
                "utility_score": utility.get("score"),
                "decision": "accept",
            }
        )
    return _package_policy_cycle("policy_gate", controller, rows)


def run_policy_meta_cycle(
    base_config: dict[str, Any], controller: dict[str, Any]
) -> dict[str, Any]:
    result = run_stage2_research_engine(
        Stage2Config(**_stage2_config_kwargs(base_config))
    )
    validation = build_validation_audit(result, {})
    rows: list[dict[str, Any]] = []
    for policy in get_reclaim_meta_policies()[
        : _policy_trial_limit(controller, len(get_reclaim_meta_policies()))
    ]:
        filtered = _apply_benchmark_contract(
            result,
            validation,
            threshold=float(controller.get("frozen_threshold", 0.45) or 0.45),
            gate_policy=controller.get("benchmark_policy_gate"),
            meta_policy=policy["name"],
        )
        execution = run_event_driven_policy_backtest(
            filtered,
            threshold=float(controller.get("frozen_threshold", 0.45) or 0.45),
            sizing_policy=controller.get("benchmark_sizing_policy"),
            regime_throttle_policy=controller.get("benchmark_regime_throttle_policy"),
            regime_size_policy=controller.get("benchmark_regime_size_policy"),
        )
        utility = (
            compute_execution_utility(execution)
            if execution.get("status") == "complete"
            else {"score": None}
        )
        rows.append(
            {
                "trial_id": f"trial-meta-{policy['name']}",
                "family": "policy_meta",
                "overrides": {"policy_meta": policy["name"]},
                "trade_count": int(execution.get("trade_count", 0) or 0),
                "total_pnl_r": float(execution.get("total_pnl_r", 0.0) or 0.0),
                "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
                "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
                "utility_score": utility.get("score"),
                "decision": "accept",
            }
        )
    return _package_policy_cycle("policy_meta", controller, rows)


def run_translation_policy_cycle(
    base_config: dict[str, Any], controller: dict[str, Any]
) -> dict[str, Any]:
    result = run_stage2_research_engine(
        Stage2Config(**_stage2_config_kwargs(base_config))
    )
    validation = build_validation_audit(result, {})
    search_v1 = build_translation_policy_search_space()
    rows: list[dict[str, Any]] = []
    for (
        decision_threshold,
        sizing_policy,
        regime_throttle_policy,
        regime_size_policy,
    ) in product(
        search_v1["space"]["decision_threshold"],
        search_v1["space"]["sizing_policy"],
        search_v1["space"]["regime_throttle_policy"],
        search_v1["space"]["regime_size_policy"],
    ):
        filtered = _apply_benchmark_contract(
            result,
            validation,
            threshold=float(decision_threshold),
            gate_policy=controller.get("benchmark_policy_gate"),
            meta_policy=controller.get("benchmark_meta_policy"),
        )
        execution = run_event_driven_policy_backtest(
            filtered,
            threshold=float(decision_threshold),
            sizing_policy=str(sizing_policy),
            regime_throttle_policy=str(regime_throttle_policy),
            regime_size_policy=str(regime_size_policy),
        )
        utility = (
            compute_execution_utility(execution)
            if execution.get("status") == "complete"
            else {"score": None}
        )
        rows.append(
            {
                "trial_id": f"trial-translation-{decision_threshold}-{sizing_policy}-{regime_throttle_policy}-{regime_size_policy}",
                "family": "translation_policy",
                "overrides": {
                    "decision_threshold": float(decision_threshold),
                    "sizing_policy": str(sizing_policy),
                    "regime_throttle_policy": str(regime_throttle_policy),
                    "regime_size_policy": str(regime_size_policy),
                },
                "trade_count": int(execution.get("trade_count", 0) or 0),
                "total_pnl_r": float(execution.get("total_pnl_r", 0.0) or 0.0),
                "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
                "avg_size_multiplier": float(
                    execution.get("avg_size_multiplier", 0.0) or 0.0
                ),
                "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
                "utility_score": utility.get("score"),
                "decision": "accept",
            }
        )
    return _package_policy_cycle("translation_policy", controller, rows)


def run_tail_path_cleanup_cycle(
    base_config: dict[str, Any], controller: dict[str, Any]
) -> dict[str, Any]:
    diagnostic = _load_tail_cleanup_diagnostic()
    if diagnostic.get("status") != "complete":
        return {
            "family": "tail_path_cleanup",
            "spec_version": controller.get("spec_version"),
            "controller": controller,
            "trial_count": 0,
            "ranked_trials": [],
            "best_trial": None,
            "accepted_trial": None,
            "batch_decision": "revise",
            "reason": diagnostic.get("reason", "missing_cpcv_failure_attribution"),
        }

    result = run_stage2_research_engine(
        Stage2Config(**_stage2_config_kwargs(base_config))
    )
    validation = build_validation_audit(result, {}, controller)
    threshold = float(controller.get("frozen_threshold", 0.45) or 0.45)
    base_records = _apply_benchmark_contract(
        result,
        validation,
        threshold=threshold,
        gate_policy=controller.get("benchmark_policy_gate"),
        meta_policy=controller.get("benchmark_meta_policy"),
    )
    walk_forward_status = validation.get("walk_forward", {}).get("status")
    source_path = str(base_config.get("source_path", ""))
    no_holdout_access = _boundary_role_for_source(source_path) != "holdout"
    targeted_axes = dict(diagnostic.get("dominant_failure_axes", {}) or {})
    run_id = f"tail-{uuid4().hex[:12]}"
    summary_root = REPORTS_DIR / "runs" / run_id / "tail_path_cleanup"
    summary_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    search_v1 = build_tail_path_cleanup_search_space()
    valid_policies = {policy["name"] for policy in get_tail_path_cleanup_policies()}
    for idx, policy_row in enumerate(search_v1.get("policies", []), start=1):
        policy_name = str(policy_row.get("name"))
        if policy_name not in valid_policies:
            continue
        policy_records = apply_tail_path_cleanup_policy(
            base_records, policy_name=policy_name, threshold=threshold
        )
        binary_execution = run_event_driven_policy_backtest(
            policy_records,
            threshold=threshold,
            sizing_policy="binary_threshold_v1",
            regime_throttle_policy="none",
            regime_size_policy="none",
        )
        sized_execution = run_event_driven_policy_backtest(
            policy_records,
            threshold=threshold,
            sizing_policy=controller.get("benchmark_sizing_policy"),
            regime_throttle_policy=controller.get("benchmark_regime_throttle_policy"),
            regime_size_policy=controller.get("benchmark_regime_size_policy"),
        )
        binary_utility = (
            compute_execution_utility(binary_execution)
            if binary_execution.get("status") == "complete"
            else {"score": None}
        )
        sized_utility = (
            compute_execution_utility(sized_execution)
            if sized_execution.get("status") == "complete"
            else {"score": None}
        )
        utility_gap = None
        if (
            sized_utility.get("score") is not None
            and binary_utility.get("score") is not None
        ):
            utility_gap = float(sized_utility["score"]) - float(binary_utility["score"])
        cpcv_eval = _evaluate_tail_cleanup_cpcv(
            diagnostic=diagnostic,
            policy_name=policy_name,
            threshold=threshold,
            controller=controller,
        )
        deep_retrace_improved = _deep_retrace_improved(diagnostic, cpcv_eval)
        dominant_bucket = _dominant_loss_bucket(cpcv_eval)
        row = {
            "trial_id": f"trial-tail-{idx:03d}",
            "family": "tail_path_cleanup",
            "selected_by": "cpcv_failure_attribution",
            "parent_benchmark_id": controller.get("parent_benchmark_id")
            or controller.get("benchmark_name"),
            "targeted_failure_axes": targeted_axes,
            "policy_delta": {"policy_name": policy_name},
            "overrides": {"tail_cleanup_policy": policy_name},
            "walk_forward_status": walk_forward_status,
            "cpcv_status": cpcv_eval.get("status"),
            "translation_status": (
                "pass" if sized_execution.get("status") == "complete" else "fail"
            ),
            "no_holdout_access": no_holdout_access,
            "binary_utility_score": binary_utility.get("score"),
            "sized_utility_score": sized_utility.get("score"),
            "utility_gap_vs_binary": utility_gap,
            "binary_trade_count": int(binary_execution.get("trade_count", 0) or 0),
            "trade_count": int(sized_execution.get("trade_count", 0) or 0),
            "total_pnl_r": float(sized_execution.get("total_pnl_r", 0.0) or 0.0),
            "avg_trade_r": float(sized_execution.get("avg_trade_r", 0.0) or 0.0),
            "max_drawdown_r": float(sized_execution.get("max_drawdown_r", 0.0) or 0.0),
            "utility_score": sized_utility.get("score"),
            "observed_sharpe": compute_sharpe_ratio(
                [
                    float(item.get("executed_pnl_r", 0.0) or 0.0)
                    for item in sized_execution.get("equity_curve", [])
                ]
            ),
            "n_obs": int(sized_execution.get("trade_count", 0) or 0),
            "deep_retrace_improved": deep_retrace_improved,
            "dominant_loss_bucket": dominant_bucket,
            "high_conf_loss_bucket_cleared": dominant_bucket != "[0.65,1.00]",
            "cpcv_summary": cpcv_eval,
        }
        row["trial_accounting"] = {
            **_trial_accounting(
                "tail_path_cleanup",
                {**base_config, "tail_cleanup_policy": policy_name},
                result,
                selected_by="cpcv_failure_attribution",
            ),
            "targeted_failure_axes": targeted_axes,
            "policy_delta": row["policy_delta"],
            "parent_benchmark_id": row["parent_benchmark_id"],
        }
        rows.append(row)
        _record_tail_cleanup_trial(row, base_config)

    _attach_tail_cleanup_dsr(rows)
    for row in rows:
        row["decision"] = _decide_tail_cleanup_trial(row)

    ranked = sorted(
        rows,
        key=lambda item: (
            float(item.get("utility_score") or float("-inf")),
            float(item.get("total_pnl_r", 0.0) or 0.0),
        ),
        reverse=True,
    )
    accepted = next((row for row in ranked if row.get("decision") == "accept"), None)
    payload = {
        "family": "tail_path_cleanup",
        "spec_version": controller.get("spec_version"),
        "controller": controller,
        "source_diagnostic": build_tail_path_cleanup_search_space().get(
            "diagnostic_artifact"
        ),
        "trial_count": len(rows),
        "ranked_trials": ranked,
        "best_trial": ranked[0] if ranked else None,
        "accepted_trial": accepted,
        "batch_decision": "accept" if accepted is not None else "revise",
    }
    (summary_root / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload


def run_boundary_confirmation_cycle(
    base_config: dict[str, Any], controller: dict[str, Any], *, boundary_role: str
) -> dict[str, Any]:
    from trading_ml.config import load_databento_manifest

    manifest = load_databento_manifest()
    source_path = select_manifest_source_path(
        manifest,
        timeframe=str(base_config.get("timeframe", "30s")),
        boundary_role=boundary_role,
    )
    if not source_path:
        return {
            "family": f"{boundary_role}_confirmation",
            "batch_decision": "revise",
            "trial_count": 0,
            "best_trial": None,
            "accepted_trial": None,
            "reason": f"missing_{boundary_role}_source",
        }
    trial_config = dict(base_config)
    trial_config["source_path"] = source_path
    result = run_stage2_research_engine(
        Stage2Config(**_stage2_config_kwargs(trial_config))
    )
    validation = build_validation_audit(result, {})
    filtered = _apply_benchmark_contract(
        result,
        validation,
        threshold=float(controller.get("frozen_threshold", 0.45) or 0.45),
        gate_policy=controller.get("benchmark_policy_gate"),
        meta_policy=controller.get("benchmark_meta_policy"),
    )
    execution = run_event_driven_policy_backtest(
        filtered,
        threshold=float(controller.get("frozen_threshold", 0.45) or 0.45),
        sizing_policy=controller.get("benchmark_sizing_policy"),
        regime_throttle_policy=controller.get("benchmark_regime_throttle_policy"),
        regime_size_policy=controller.get("benchmark_regime_size_policy"),
    )
    utility = (
        compute_execution_utility(execution)
        if execution.get("status") == "complete"
        else {"score": None}
    )
    translation = build_translation_analysis(
        result,
        filtered,
        sizing_policy=controller.get("benchmark_sizing_policy"),
        regime_throttle_policy=controller.get("benchmark_regime_throttle_policy"),
        regime_size_policy=controller.get("benchmark_regime_size_policy"),
    )
    row = {
        "trial_id": f"{boundary_role}-benchmark",
        "family": f"{boundary_role}_confirmation",
        "overrides": {"boundary_role": boundary_role},
        "trade_count": int(execution.get("trade_count", 0) or 0),
        "total_pnl_r": float(execution.get("total_pnl_r", 0.0) or 0.0),
        "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
        "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
        "utility_score": utility.get("score"),
        "walk_forward_status": validation.get("walk_forward", {}).get("status"),
        "cpcv_status": validation.get("cpcv", {}).get("status"),
        "translation_status": translation.get("status"),
        "decision": "accept",
    }
    if row["walk_forward_status"] != "pass" or row["translation_status"] == "fail":
        row["decision"] = "reject"
    return {
        "family": f"{boundary_role}_confirmation",
        "spec_version": controller.get("spec_version"),
        "controller": controller,
        "trial_count": 1,
        "ranked_trials": [row],
        "best_trial": row,
        "accepted_trial": row if row["decision"] == "accept" else None,
        "batch_decision": "accept" if row["decision"] == "accept" else "revise",
        "validation_audit": validation,
    }


def run_market_state_setup_quality_cycle(
    base_config: dict[str, Any], controller: dict[str, Any]
) -> dict[str, Any]:
    variants = market_state_policy_variant_specs()
    max_trials = _policy_trial_limit(controller, len(variants))
    variants = variants[:max_trials]
    simulation = run_market_state_policy_simulation(
        {"stage2_config": base_config},
        variants=variants,
        max_cpcv_paths=int(controller.get("market_state_max_cpcv_paths", 0) or 0)
        or None,
    )
    if simulation.get("status") != "complete":
        return {
            "family": "market_state_setup_quality",
            "spec_version": controller.get("spec_version"),
            "controller": controller,
            "trial_count": 0,
            "ranked_trials": [],
            "best_trial": None,
            "accepted_trial": None,
            "batch_decision": "revise",
            "reason": simulation.get("reason", "simulation_unavailable"),
            "simulation": simulation,
        }

    baseline = next(
        (row for row in simulation["policy_variants"] if row["variant"] == "baseline"),
        simulation["policy_variants"][0],
    )
    rows = []
    for idx, variant in enumerate(simulation["policy_variants"], start=1):
        row = _market_state_trial_row(
            idx,
            variant,
            baseline,
            base_config,
            controller,
            n_trials=len(simulation["policy_variants"]),
        )
        rows.append(row)
    _attach_market_state_dsr(rows)
    for row in rows:
        row["strict_gates"]["dsr_pass"] = row["deflated_sharpe"]["status"] == "pass"
        row["decision"] = "accept" if all(row["strict_gates"].values()) else "reject"
        _record_market_state_trial(row, base_config, controller)

    ranked = sorted(
        rows,
        key=lambda row: (
            row["decision"] == "accept",
            float(row.get("mean_cpcv_path_pnl_r", float("-inf"))),
            float(row.get("total_pnl_r", float("-inf"))),
        ),
        reverse=True,
    )
    accepted = next((row for row in ranked if row["decision"] == "accept"), None)
    run_id = f"market-state-{uuid4().hex[:12]}"
    summary_root = REPORTS_DIR / "runs" / run_id / "market_state_setup_quality"
    summary_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": "market_state_setup_quality",
        "spec_version": controller.get("spec_version"),
        "controller": controller,
        "space": build_market_state_setup_quality_search_space(),
        "trial_count": len(rows),
        "models_trained": 0,
        "governance": {
            "promotion_blocked": True,
            "models_trained": 0,
            "reason": "tiny_market_state_setup_probe",
        },
        "holdout_status": "locked",
        "ranked_trials": ranked,
        "best_trial": ranked[0] if ranked else None,
        "accepted_trial": accepted,
        "batch_decision": "accept" if accepted is not None else "revise",
        "strict_pass_criteria": _market_state_strict_pass_criteria(baseline),
        "simulation_summary": {
            "candidate_count": simulation.get("diagnostic_summary", {}).get(
                "candidate_count"
            ),
            "state_counts": simulation.get("diagnostic_summary", {}).get(
                "candidate_counts_by_state"
            ),
            "quality_counts": simulation.get("diagnostic_summary", {}).get(
                "candidate_counts_by_quality"
            ),
        },
    }
    (summary_root / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload


def _market_state_trial_row(
    idx: int,
    variant: dict[str, Any],
    baseline: dict[str, Any],
    base_config: dict[str, Any],
    controller: dict[str, Any],
    *,
    n_trials: int,
) -> dict[str, Any]:
    cpcv_paths = list(variant.get("cpcv_paths", []) or [])
    path_pnls = [float(row.get("total_pnl_r", 0.0) or 0.0) for row in cpcv_paths]
    cpcv_status = _market_state_cpcv_status(variant)
    sharpe = compute_sharpe_ratio(path_pnls)
    psr_probability = (
        _psr_probability(sharpe, len(path_pnls)) if sharpe is not None else 0.0
    )
    deflated_sharpe = {
        "status": "pending",
        "probability": 0.0,
        "psr_status": (
            "pass" if psr_probability >= 0.95 and (sharpe or 0.0) > 0 else "fail"
        ),
        "psr_probability": psr_probability,
        "observed_sharpe": sharpe,
        "n_trials": n_trials,
        "n_obs": len(path_pnls),
        "sr_std": 0.0,
        "variance_source": "cross_trial_cpcv_path_sharpe",
    }
    min_trade_count = int(controller.get("market_state_min_trade_count", 350) or 350)
    min_retained_fraction = float(
        controller.get("market_state_min_retained_fraction", 0.70) or 0.70
    )
    worst_path_threshold = float(
        controller.get("market_state_worst_path_threshold_r", -5.0) or -5.0
    )
    baseline_positive_rate = float(baseline.get("positive_path_rate", 0.0) or 0.0)
    retained_fraction = float(variant.get("trade_count", 0) or 0) / max(
        float(variant.get("candidate_count", 0) or 0), 1.0
    )
    strict_gates = {
        "cpcv_pass": cpcv_status == "pass",
        "dsr_pass": deflated_sharpe["status"] == "pass",
        "psr_pass": deflated_sharpe["psr_status"] == "pass",
        "worst_path_breach_below_threshold": float(min(path_pnls) if path_pnls else 0.0)
        > worst_path_threshold,
        "mean_cpcv_positive": float(variant.get("mean_cpcv_path_pnl_r", 0.0) or 0.0)
        > 0,
        "median_cpcv_positive": float(variant.get("median_cpcv_path_pnl_r", 0.0) or 0.0)
        > 0,
        "positive_path_rate_holds": float(variant.get("positive_path_rate", 0.0) or 0.0)
        >= baseline_positive_rate,
        "trade_count_acceptable": int(variant.get("trade_count", 0) or 0)
        >= min_trade_count,
        "not_deleting_too_many_trades": retained_fraction >= min_retained_fraction,
        "holdout_locked": True,
    }
    row = {
        "trial_id": f"trial-market-state-{idx:03d}",
        "family": "market_state_setup_quality",
        "variant": variant["variant"],
        "overrides": {"market_state_policy": variant["variant"]},
        "candidate_count": int(variant.get("candidate_count", 0) or 0),
        "trade_count": int(variant.get("trade_count", 0) or 0),
        "effective_trade_count": float(
            variant.get("effective_trade_count", 0.0) or 0.0
        ),
        "retained_fraction": retained_fraction,
        "total_pnl_r": float(variant.get("total_pnl_r", 0.0) or 0.0),
        "avoided_pnl_r": float(variant.get("avoided_pnl_r", 0.0) or 0.0),
        "mean_cpcv_path_pnl_r": float(variant.get("mean_cpcv_path_pnl_r", 0.0) or 0.0),
        "median_cpcv_path_pnl_r": float(
            variant.get("median_cpcv_path_pnl_r", 0.0) or 0.0
        ),
        "worst_3_cpcv_paths": variant.get("worst_3_cpcv_paths", []),
        "positive_path_rate": float(variant.get("positive_path_rate", 0.0) or 0.0),
        "prior_worst_path_effect": variant.get("prior_worst_path_effect", []),
        "state_contribution_table": variant.get("state_contribution_table", []),
        "cpcv_summary": _market_state_cpcv_summary(variant, cpcv_status),
        "deflated_sharpe": deflated_sharpe,
        "strict_gates": strict_gates,
        "decision": "reject",
        "trial_accounting": {
            "family": "market_state_setup_quality",
            "n_trials": n_trials,
            "selected_by": "governed_tiny_market_state_policy_batch",
            "data_slice": {
                "source_path": base_config.get("source_path"),
                "symbol": base_config.get("symbol"),
                "timeframe": base_config.get("timeframe"),
            },
            "models_trained": 0,
            "holdout_status": "locked",
        },
    }
    return row


def _market_state_cpcv_status(variant: dict[str, Any]) -> str:
    paths = list(variant.get("cpcv_paths", []) or [])
    if not paths:
        return "pending"
    pnls = [float(path.get("total_pnl_r", 0.0) or 0.0) for path in paths]
    pbo = sum(1 for pnl in pnls if pnl <= 0) / len(pnls)
    return (
        "pass"
        if pbo <= 0.25
        and float(variant.get("mean_cpcv_path_pnl_r", 0.0) or 0.0) > 0
        and float(variant.get("median_cpcv_path_pnl_r", 0.0) or 0.0) > 0
        and float(variant.get("positive_path_rate", 0.0) or 0.0) >= 0.60
        and min(pnls) > -5.0
        else "fail"
    )


def _attach_market_state_dsr(rows: list[dict[str, Any]]) -> None:
    observed = [
        float(row.get("deflated_sharpe", {}).get("observed_sharpe"))
        for row in rows
        if row.get("deflated_sharpe", {}).get("observed_sharpe") is not None
    ]
    sr_std = _tail_sample_std(observed)
    n_trials = max(len(rows), 1)
    for row in rows:
        observed_sr = row.get("deflated_sharpe", {}).get("observed_sharpe")
        n_obs = int(row.get("deflated_sharpe", {}).get("n_obs", 0) or 0)
        if observed_sr is None or sr_std <= 0 or n_obs <= 1:
            row["deflated_sharpe"].update(
                {
                    "status": "pending" if observed_sr is None else "fail",
                    "probability": 0.0,
                    "n_trials": n_trials,
                    "sr_std": sr_std,
                }
            )
            continue
        probability = deflated_sharpe_probability(
            observed_sr=float(observed_sr),
            n_trials=n_trials,
            sr_std=sr_std,
            n_obs=n_obs,
            skew=0.0,
            kurtosis=3.0,
        )
        row["deflated_sharpe"].update(
            {
                "status": (
                    "pass" if probability >= 0.95 and float(observed_sr) > 0 else "fail"
                ),
                "probability": probability,
                "n_trials": n_trials,
                "sr_std": sr_std,
            }
        )


def _market_state_cpcv_summary(variant: dict[str, Any], status: str) -> dict[str, Any]:
    paths = list(variant.get("cpcv_paths", []) or [])
    pnls = sorted(float(path.get("total_pnl_r", 0.0) or 0.0) for path in paths)
    pbo = sum(1 for pnl in pnls if pnl <= 0) / len(pnls) if pnls else 1.0
    return {
        "status": status,
        "pbo": pbo,
        "mean_total_pnl_r": float(variant.get("mean_cpcv_path_pnl_r", 0.0) or 0.0),
        "median_total_pnl_r": float(variant.get("median_cpcv_path_pnl_r", 0.0) or 0.0),
        "min_path_pnl_r": min(pnls) if pnls else 0.0,
        "path_positive_rate": float(variant.get("positive_path_rate", 0.0) or 0.0),
        "distribution": _tail_path_distribution(pnls),
        "worst_paths": variant.get("worst_3_cpcv_paths", []),
        "paths": paths,
    }


def _market_state_strict_pass_criteria(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "cpcv_status": "pass",
        "deflated_sharpe_status": "pass",
        "psr_status": "pass",
        "min_path_pnl_r": "> -5.0",
        "mean_cpcv_path_pnl_r": "> 0",
        "median_cpcv_path_pnl_r": "> 0",
        "positive_path_rate": f">= baseline {float(baseline.get('positive_path_rate', 0.0) or 0.0):.3f}",
        "min_trade_count": 350,
        "min_retained_fraction": 0.70,
        "holdout": "locked",
    }


def _psr_probability(observed_sr: float | None, n_obs: int) -> float:
    if observed_sr is None or n_obs <= 1:
        return 0.0
    return deflated_sharpe_probability(
        observed_sr=observed_sr,
        n_trials=1,
        sr_std=1.0,
        n_obs=n_obs,
        skew=0.0,
        kurtosis=3.0,
    )


def _record_market_state_trial(
    row: dict[str, Any], base_config: dict[str, Any], controller: dict[str, Any]
) -> None:
    append_experiment_record(
        ExperimentRecord(
            experiment_id=f"{controller.get('spec_version', 'bnr')}-{row['trial_id']}",
            hypothesis="Tiny governed market-state/setup-quality policy simulation before any model search.",
            config_ref=str(controller.get("spec_version", "unknown")),
            data_slice={
                "source_path": base_config["source_path"],
                "symbol": base_config["symbol"],
                "timeframe": base_config["timeframe"],
            },
            result={
                "variant": row["variant"],
                "trade_count": row["trade_count"],
                "total_pnl_r": row["total_pnl_r"],
                "cpcv_summary": row["cpcv_summary"],
                "deflated_sharpe": row["deflated_sharpe"],
                "strict_gates": row["strict_gates"],
                "trial_accounting": row["trial_accounting"],
            },
            decision=row["decision"],
            phase="exploration",
        )
    )


def _apply_benchmark_contract(
    result: dict[str, Any],
    validation: dict[str, Any],
    *,
    threshold: float,
    gate_policy: str | None,
    meta_policy: str | None,
) -> list[dict[str, Any]]:
    feature_map = {
        str(row["candidate_id"]): row for row in result.get("features_records", [])
    }
    filtered = []
    for row in validation.get("walk_forward", {}).get(
        "stitched_prediction_records", []
    ):
        merged = dict(row)
        merged.update(feature_map.get(str(row.get("candidate_id")), {}))
        filtered.append(merged)
    if gate_policy:
        filtered = apply_break_quality_policy(
            filtered,
            result.get("features_records", []),
            policy_name=gate_policy,
            threshold=threshold,
        )
        threshold = 0.0
    if meta_policy:
        filtered = apply_reclaim_meta_policy(filtered, policy_name=meta_policy)
    if threshold > 0:
        filtered = [
            row
            for row in filtered
            if float(row.get("probability", 0.0) or 0.0) >= threshold
        ]
    return filtered


def _package_policy_cycle(
    family: str, controller: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["utility_score"] or float("-inf")),
            row["total_pnl_r"],
        ),
        reverse=True,
    )
    accepted = ranked[0] if ranked else None
    return {
        "family": family,
        "spec_version": controller.get("spec_version"),
        "controller": controller,
        "trial_count": len(rows),
        "ranked_trials": ranked,
        "best_trial": ranked[0] if ranked else None,
        "accepted_trial": accepted,
        "batch_decision": "accept" if accepted is not None else "revise",
    }


def _policy_trial_limit(controller: dict[str, Any], default: int) -> int:
    limit = controller.get("max_batch_trials", default)
    try:
        return max(0, min(int(limit), int(default)))
    except (TypeError, ValueError):
        return int(default)


def _load_tail_cleanup_diagnostic() -> dict[str, Any]:
    search_v1 = build_tail_path_cleanup_search_space()
    diagnostic_path = Path(
        str(
            search_v1.get(
                "diagnostic_artifact", "reports/cpcv_failure_attribution.json"
            )
        )
    )
    if not diagnostic_path.is_absolute():
        diagnostic_path = REPORTS_DIR.parent / diagnostic_path
    if not diagnostic_path.exists():
        return {
            "status": "missing",
            "reason": "diagnostic_artifact_missing",
            "path": str(diagnostic_path),
        }
    return json.loads(diagnostic_path.read_text(encoding="utf-8"))


def _evaluate_tail_cleanup_cpcv(
    *,
    diagnostic: dict[str, Any],
    policy_name: str,
    threshold: float,
    controller: dict[str, Any],
) -> dict[str, Any]:
    source = Path(str(diagnostic.get("source_diagnostics", "")))
    exploration = (
        json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}
    )
    cpcv = dict(exploration.get("cpcv", {}) or {})
    artifact_root = cpcv.get("artifact_root")
    if not artifact_root:
        return {"status": "pending", "reason": "missing_artifact_root"}
    root = Path(str(artifact_root))
    path_files = sorted(root.glob("path_cpcv_*_rows.json"))
    if not path_files:
        return {"status": "pending", "reason": "missing_path_rows"}

    rows: list[dict[str, Any]] = []
    negative_paths = 0
    path_pnls: list[float] = []
    for path_file in path_files:
        path_rows = json.loads(path_file.read_text(encoding="utf-8"))
        policy_rows = apply_tail_path_cleanup_policy(
            path_rows, policy_name=policy_name, threshold=threshold
        )
        execution = run_event_driven_policy_backtest(
            policy_rows,
            threshold=threshold,
            sizing_policy=controller.get("benchmark_sizing_policy"),
            regime_throttle_policy=controller.get("benchmark_regime_throttle_policy"),
            regime_size_policy=controller.get("benchmark_regime_size_policy"),
        )
        if execution.get("status") != "complete":
            total_pnl_r = 0.0
            sharpe_r = None
            trade_count = 0
            attribution = {
                "subtype_breakdown": [],
                "threshold_distribution": [],
                "largest_loss_cluster_r": 0.0,
            }
        else:
            total_pnl_r = float(execution.get("total_pnl_r", 0.0) or 0.0)
            trade_count = int(execution.get("trade_count", 0) or 0)
            sharpe_r = compute_sharpe_ratio(
                [
                    float(item.get("executed_pnl_r", 0.0) or 0.0)
                    for item in execution.get("equity_curve", [])
                ]
            )
            attribution = _tail_cleanup_path_attribution(policy_rows, execution)
        if total_pnl_r <= 0:
            negative_paths += 1
        path_pnls.append(total_pnl_r)
        rows.append(
            {
                "path_id": path_file.stem.replace("_rows", ""),
                "rows_artifact": str(path_file),
                "trade_count": trade_count,
                "total_pnl_r": total_pnl_r,
                "avg_trade_r": (
                    float(execution.get("avg_trade_r", 0.0) or 0.0)
                    if execution.get("status") == "complete"
                    else 0.0
                ),
                "win_rate": (
                    float(execution.get("win_rate", 0.0) or 0.0)
                    if execution.get("status") == "complete"
                    else 0.0
                ),
                "max_drawdown_r": (
                    float(execution.get("max_drawdown_r", 0.0) or 0.0)
                    if execution.get("status") == "complete"
                    else 0.0
                ),
                "sharpe_r": sharpe_r,
                **attribution,
            }
        )

    sorted_pnls = sorted(path_pnls)
    mean_pnl = sum(path_pnls) / len(path_pnls)
    median_pnl = sorted_pnls[len(sorted_pnls) // 2]
    min_pnl = min(path_pnls)
    positive_rate = sum(1 for value in path_pnls if value > 0) / len(path_pnls)
    pbo = negative_paths / len(path_pnls)
    ranked = sorted(rows, key=lambda row: row["total_pnl_r"])
    status = (
        "pass"
        if pbo <= 0.25
        and mean_pnl > 0
        and median_pnl > 0
        and positive_rate >= 0.60
        and min_pnl > -5.0
        else "fail"
    )
    return {
        "status": status,
        "pbo": pbo,
        "mean_total_pnl_r": mean_pnl,
        "median_total_pnl_r": median_pnl,
        "min_path_pnl_r": min_pnl,
        "path_positive_rate": positive_rate,
        "distribution": _tail_path_distribution(sorted_pnls),
        "worst_paths": ranked[:3],
        "best_paths": list(reversed(ranked[-3:])),
        "paths": rows,
    }


def _tail_cleanup_path_attribution(
    records: list[dict[str, Any]], execution: dict[str, Any]
) -> dict[str, Any]:
    executed = {
        str(row.get("candidate_id")): row for row in execution.get("equity_curve", [])
    }
    merged = []
    for row in records:
        cid = str(row.get("candidate_id"))
        if cid not in executed:
            continue
        joined = dict(row)
        joined["executed_pnl_r"] = float(
            executed[cid].get("executed_pnl_r", 0.0) or 0.0
        )
        merged.append(joined)
    return {
        "largest_loss_cluster_r": _largest_loss_cluster_from_rows(merged),
        "subtype_breakdown": _simple_group_rows(merged, "setup_subtype"),
        "threshold_distribution": _probability_bins_rows(merged),
    }


def _largest_loss_cluster_from_rows(rows: list[dict[str, Any]]) -> float:
    running = 0.0
    worst = 0.0
    for row in rows:
        pnl = float(row.get("executed_pnl_r", 0.0) or 0.0)
        if pnl < 0:
            running += pnl
            worst = min(worst, running)
        else:
            running = 0.0
    return worst


def _simple_group_rows(rows: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(column, "unknown"))
        bucket = groups.setdefault(
            key, {"key": key, "trade_count": 0, "total_pnl_r": 0.0}
        )
        bucket["trade_count"] += 1
        bucket["total_pnl_r"] += float(row.get("executed_pnl_r", 0.0) or 0.0)
    return sorted(groups.values(), key=lambda item: item["trade_count"], reverse=True)


def _probability_bins_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = {
        "[0.45,0.55)": {"trade_count": 0, "total_pnl_r": 0.0},
        "[0.55,0.65)": {"trade_count": 0, "total_pnl_r": 0.0},
        "[0.65,1.00]": {"trade_count": 0, "total_pnl_r": 0.0},
    }
    for row in rows:
        probability = float(row.get("probability", 0.0) or 0.0)
        if probability < 0.55:
            key = "[0.45,0.55)"
        elif probability < 0.65:
            key = "[0.55,0.65)"
        else:
            key = "[0.65,1.00]"
        buckets[key]["trade_count"] += 1
        buckets[key]["total_pnl_r"] += float(row.get("executed_pnl_r", 0.0) or 0.0)
    return [{"bucket": key, **value} for key, value in buckets.items()]


def _tail_path_distribution(sorted_pnls: list[float]) -> dict[str, float]:
    def pct(level: float) -> float:
        if not sorted_pnls:
            return 0.0
        idx = min(
            len(sorted_pnls) - 1, max(0, int(round((len(sorted_pnls) - 1) * level)))
        )
        return float(sorted_pnls[idx])

    negative_tail = [value for value in sorted_pnls if value < 0]
    return {
        "p10_total_pnl_r": pct(0.10),
        "p25_total_pnl_r": pct(0.25),
        "p75_total_pnl_r": pct(0.75),
        "p90_total_pnl_r": pct(0.90),
        "negative_tail_contribution_r": float(sum(negative_tail)),
    }


def _deep_retrace_improved(
    source_diagnostic: dict[str, Any], trial_cpcv: dict[str, Any]
) -> bool:
    source_worst = list(source_diagnostic.get("worst_paths", []) or [])
    baseline_loss = None
    if source_worst:
        for row in source_worst[0].get("subtype_breakdown", []):
            if row.get("key") == "deep_retrace_repair":
                baseline_loss = float(row.get("total_pnl_r", 0.0) or 0.0)
                break
    if baseline_loss is None:
        return False
    trial_worst = list(trial_cpcv.get("worst_paths", []) or [])
    if not trial_worst:
        return False
    trial_loss = 0.0
    for row in trial_worst[0].get("subtype_breakdown", []):
        if row.get("key") == "deep_retrace_repair":
            trial_loss = float(row.get("total_pnl_r", 0.0) or 0.0)
            break
    return trial_loss > baseline_loss


def _dominant_loss_bucket(cpcv_summary: dict[str, Any]) -> str | None:
    worst_paths = list(cpcv_summary.get("worst_paths", []) or [])
    if not worst_paths:
        return None
    candidate = None
    for row in worst_paths[0].get("threshold_distribution", []):
        total_pnl = float(row.get("total_pnl_r", 0.0) or 0.0)
        if total_pnl >= 0:
            continue
        if candidate is None or total_pnl < candidate[1]:
            candidate = (str(row.get("bucket")), total_pnl)
    return candidate[0] if candidate else None


def _attach_tail_cleanup_dsr(rows: list[dict[str, Any]]) -> None:
    observed = [
        float(row.get("observed_sharpe", 0.0) or 0.0)
        for row in rows
        if row.get("observed_sharpe") is not None
    ]
    sr_std = _tail_sample_std(observed) if len(observed) >= 2 else 0.0
    trial_count = max(len(rows), 1)
    for row in rows:
        observed_sr = row.get("observed_sharpe")
        if observed_sr is None or sr_std <= 0 or int(row.get("n_obs", 0) or 0) <= 1:
            row["deflated_sharpe"] = {
                "status": "pending",
                "probability": 0.0,
                "n_trials": trial_count,
            }
            continue
        probability = deflated_sharpe_probability(
            observed_sr=float(observed_sr),
            n_trials=trial_count,
            sr_std=sr_std,
            n_obs=int(row.get("n_obs", 0) or 0),
            skew=0.0,
            kurtosis=3.0,
        )
        row["deflated_sharpe"] = {
            "status": (
                "pass" if probability >= 0.95 and float(observed_sr) > 0 else "fail"
            ),
            "probability": probability,
            "n_trials": trial_count,
        }


def _decide_tail_cleanup_trial(row: dict[str, Any]) -> str:
    gap = row.get("utility_gap_vs_binary")
    binary_competitive = gap is not None and float(gap) <= 0.50
    if row.get("walk_forward_status") != "pass":
        return "reject"
    if row.get("cpcv_status") != "pass":
        return "reject"
    if row.get("translation_status") != "pass":
        return "reject"
    if not row.get("no_holdout_access"):
        return "reject"
    if row.get("deflated_sharpe", {}).get("status") != "pass":
        return "reject"
    if not binary_competitive:
        return "reject"
    if not row.get("deep_retrace_improved"):
        return "reject"
    if not row.get("high_conf_loss_bucket_cleared"):
        return "reject"
    return "accept"


def _record_tail_cleanup_trial(
    row: dict[str, Any], base_config: dict[str, Any]
) -> None:
    append_experiment_record(
        ExperimentRecord(
            experiment_id=f"{row['parent_benchmark_id']}-{row['trial_id']}",
            hypothesis="Target CPCV tail-path failure axes without broad threshold/model search.",
            config_ref=str(row["parent_benchmark_id"]),
            data_slice={
                "source_path": base_config["source_path"],
                "symbol": base_config["symbol"],
                "timeframe": base_config["timeframe"],
            },
            result={
                "overrides": row["overrides"],
                "utility_score": row.get("utility_score"),
                "total_pnl_r": row.get("total_pnl_r"),
                "binary_utility_score": row.get("binary_utility_score"),
                "utility_gap_vs_binary": row.get("utility_gap_vs_binary"),
                "cpcv_summary": row.get("cpcv_summary"),
                "trial_accounting": row.get("trial_accounting"),
            },
            decision="pending",
            phase="exploration",
        )
    )


def _boundary_role_for_source(source_path: str) -> str | None:
    from trading_ml.config import load_databento_manifest

    manifest = load_databento_manifest()
    for entry in manifest.get("files", []):
        if entry.get("source_path") == source_path:
            return entry.get("boundary_role")
    return None


def _tail_sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return variance**0.5
