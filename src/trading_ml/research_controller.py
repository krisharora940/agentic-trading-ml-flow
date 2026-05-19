from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

from trading_ml.break_quality_policy import apply_break_quality_policy, get_break_quality_policies
from trading_ml.config import load_bnr_config, load_global_config
from trading_ml.bnr_subtypes import list_bnr_subtypes
from trading_ml.evidence_sources import select_manifest_source_path
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.feature_families import list_feature_families
from trading_ml.reclaim_meta_policy import apply_reclaim_meta_policy, get_reclaim_meta_policies
from trading_ml.registry import append_experiment_record
from trading_ml.schemas import ExperimentRecord
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.translation_analysis import build_translation_analysis
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


def build_label_search_space() -> dict[str, Any]:
    return load_bnr_config()["label_search_v1"]


def build_subtype_search_space() -> dict[str, Any]:
    return load_bnr_config()["subtype_search_v1"]


def load_controller_config(override: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_bnr_config()
    controller = dict(config.get("controller", {}))
    frozen = dict(config.get("frozen_benchmark", {}))
    controller.setdefault("spec_version", "bnr_spec_vA")
    controller.setdefault("active_family", "setup")
    controller.setdefault("active_model_family", "linear_baseline")
    controller.setdefault("frozen_threshold", config.get("controller", {}).get("frozen_threshold", 0.45))
    controller.setdefault("benchmark_name", config.get("controller", {}).get("benchmark_name", "bnr_hybrid_linear_v1"))
    controller.setdefault("benchmark_policy_gate", frozen.get("policy_gate"))
    controller.setdefault("benchmark_meta_policy", frozen.get("policy_meta"))
    controller.setdefault("min_candidate_ratio_vs_baseline", 0.7)
    controller.setdefault("require_positive_net_delta", True)
    controller.setdefault("min_roc_auc_delta", 0.0)
    if override:
        controller.update(override)
    return controller


def generate_search_trials(base_config: dict[str, Any], family: str | None = None, controller_override: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    active_family = family or load_controller_config(controller_override)["active_family"]
    trials: list[dict[str, Any]] = []
    if active_family == "setup":
        search_v1 = build_search_space()
        space = search_v1["space"]
        ordered_keys = ["earliest_trigger_time", "horizon_bars", "target_multiple", "break_buffer_points"]
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
        for feature_family in search_v1["space"]["feature_family"]:
            if feature_family not in valid_families:
                continue
            trial = dict(base_config)
            trial["feature_family"] = feature_family
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
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    if active_family == "threshold":
        search_v1 = build_threshold_search_space()
        for decision_threshold in search_v1["space"]["decision_threshold"]:
            trial = dict(base_config)
            trial["decision_threshold"] = float(decision_threshold)
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
    if active_family == "subtype":
        search_v1 = build_subtype_search_space()
        valid_subtypes = set(list_bnr_subtypes())
        for setup_subtype in search_v1["space"]["setup_subtype"]:
            if setup_subtype not in valid_subtypes:
                continue
            trial = dict(base_config)
            trial["setup_subtype"] = setup_subtype
            trials.append(trial)
        return trials[: int(search_v1["max_batch_trials"])]
    raise ValueError(f"Unsupported search family: {active_family}")


def run_governed_search(base_config: dict[str, Any], controller_override: dict[str, Any] | None = None) -> dict[str, Any]:
    return run_governed_research_cycle(base_config, controller_override=controller_override)


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
    if active_family == "validation_window":
        return run_boundary_confirmation_cycle(base_config, controller, boundary_role="validation")
    if active_family == "holdout_confirmation":
        return run_boundary_confirmation_cycle(base_config, controller, boundary_role="holdout")
    result_cache: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    baseline_result = _cached_stage2_result(base_config, result_cache)
    baseline = _summarize_baseline(active_family, str(controller["spec_version"]), baseline_result, controller)
    trials = generate_search_trials(base_config, family=active_family, controller_override=controller)
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

    ranked = sorted(results, key=lambda item: (item.net_avg_pnl_r, item.roc_auc or float("-inf")), reverse=True)
    accepted = next((trial for trial in ranked if trial.decision == "accept"), None)
    _record_baseline(active_family, baseline, base_config, baseline_result)
    return {
        "family": active_family,
        "spec_version": controller["spec_version"],
        "space": (
            build_search_space()
            if active_family == "setup"
            else build_model_search_space()
            if active_family == "model"
            else build_feature_search_space()
            if active_family == "feature"
            else build_feature_threshold_search_space()
            if active_family == "feature_threshold"
            else build_threshold_search_space()
            if active_family == "threshold"
            else build_subtype_search_space()
            if active_family == "subtype"
            else build_label_search_space()
        ),
        "controller": controller,
        "baseline": baseline.to_dict(),
        "trial_count": len(results),
        "ranked_trials": [trial.to_dict() for trial in ranked],
        "best_trial": ranked[0].to_dict() if ranked else None,
        "accepted_trial": accepted.to_dict() if accepted else None,
        "batch_decision": "accept" if accepted is not None else "revise",
    }


def _summarize_baseline(family: str, spec_version: str, result: dict[str, Any], controller: dict[str, Any]) -> ControllerTrialSummary:
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
        net_avg_pnl_r=_estimate_net_avg_pnl_r(result, threshold=controller.get("frozen_threshold") if family in {"feature", "model", "threshold"} else None),
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
        if family in {"threshold", "feature_threshold"} and "decision_threshold" in config
        else controller.get("frozen_threshold")
        if family in {"feature", "feature_threshold", "model", "threshold", "label"}
        else None
    )
    net_avg_pnl_r = _estimate_net_avg_pnl_r(result, threshold=threshold)
    roc_auc = float(metrics["roc_auc"]) if "roc_auc" in metrics else None
    roc_delta = None if roc_auc is None or baseline.roc_auc is None else roc_auc - baseline.roc_auc
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
        return {"feature_family": config["feature_family"]}
    if family == "feature_threshold":
        return {
            "feature_family": config["feature_family"],
            "decision_threshold": float(config["decision_threshold"]),
        }
    if family == "threshold":
        return {"decision_threshold": float(config["decision_threshold"])}
    if family == "label":
        return {
            "horizon_bars": config["horizon_bars"],
            "stop_multiple": config["stop_multiple"],
            "target_multiple": config["target_multiple"],
        }
    if family == "subtype":
        return {"setup_subtype": config["setup_subtype"]}
    return {
        "earliest_trigger_time": config["earliest_trigger_time"],
        "horizon_bars": config["horizon_bars"],
        "target_multiple": config["target_multiple"],
        "break_buffer_points": config["break_buffer_points"],
    }


def _decide_trial(
    trial: ControllerTrialSummary,
    baseline: ControllerTrialSummary,
    controller: dict[str, Any],
) -> str:
    if trial.status != "fit":
        return "reject"
    min_candidate_ratio = float(controller["min_candidate_ratio_vs_baseline"])
    if baseline.candidate_count > 0 and (trial.candidate_count / baseline.candidate_count) < min_candidate_ratio:
        return "reject"
    if bool(controller["require_positive_net_delta"]) and trial.net_delta_vs_baseline <= 0:
        return "reject"
    min_roc_auc_delta = float(controller["min_roc_auc_delta"])
    if trial.roc_auc_delta_vs_baseline is not None and trial.roc_auc_delta_vs_baseline < min_roc_auc_delta:
        return "reject"
    return "accept"


def _record_baseline(family: str, baseline: ControllerTrialSummary, config: dict[str, Any], result: dict[str, Any]) -> None:
    append_experiment_record(
        ExperimentRecord(
            experiment_id=f"{baseline.spec_version}-{family}-baseline",
            hypothesis=f"Frozen {family} baseline for {baseline.spec_version}.",
            config_ref=baseline.spec_version,
            data_slice={"source_path": config["source_path"], "symbol": config["symbol"], "timeframe": config["timeframe"]},
            result={
                "overrides": {},
                "candidate_count": result.get("candidate_count", 0),
                "label_summary": result.get("label_summary", {}),
                "model_summary": result.get("model_summary", {}),
                "net_avg_pnl_r": baseline.net_avg_pnl_r,
            },
            decision=baseline.decision,
            phase="exploration",
        )
    )


def _record_trial(family: str, summary: ControllerTrialSummary, config: dict[str, Any], result: dict[str, Any]) -> None:
    append_experiment_record(
        ExperimentRecord(
            experiment_id=f"{summary.spec_version}-{family}-{summary.trial_id}",
            hypothesis=f"Test {family} overrides against frozen baseline.",
            config_ref=summary.spec_version,
            data_slice={"source_path": config["source_path"], "symbol": config["symbol"], "timeframe": config["timeframe"]},
            result={
                "overrides": summary.overrides,
                "candidate_count": result.get("candidate_count", 0),
                "label_summary": result.get("label_summary", {}),
                "model_summary": result.get("model_summary", {}),
                "net_avg_pnl_r": summary.net_avg_pnl_r,
                "net_delta_vs_baseline": summary.net_delta_vs_baseline,
                "roc_auc_delta_vs_baseline": summary.roc_auc_delta_vs_baseline,
            },
            decision=summary.decision,
            phase="exploration",
        )
    )


def _estimate_net_avg_pnl_r(result: dict[str, Any], threshold: float | None = None) -> float:
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


def _estimate_prediction_net_avg_pnl_r(result: dict[str, Any], threshold: float | None) -> float | None:
    if threshold is None:
        return None

    global_config = load_global_config()
    model_summary = dict(result.get("model_summary", {}))
    prediction_records = list(model_summary.get("prediction_records", []))
    avg_risk_points = float(model_summary.get("avg_risk_points", 0.0) or 0.0)
    avg_entry_price = float(model_summary.get("avg_entry_price", 0.0) or 0.0)
    if not prediction_records or avg_risk_points <= 0 or avg_entry_price <= 0:
        return None

    selected = [row for row in prediction_records if float(row.get("probability", 0.0) or 0.0) >= float(threshold)]
    if not selected:
        return 0.0

    avg_pnl_r = sum(float(row.get("pnl_r", 0.0) or 0.0) for row in selected) / len(selected)
    cost_points = _round_trip_cost_points(avg_entry_price, global_config)
    cost_r = cost_points / avg_risk_points
    return avg_pnl_r - cost_r


def _round_trip_cost_points(avg_entry_price: float, global_config: dict[str, Any]) -> float:
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


def _cached_stage2_result(config: dict[str, Any], cache: dict[tuple[tuple[str, Any], ...], dict[str, Any]]) -> dict[str, Any]:
    stage2_kwargs = _stage2_config_kwargs(config)
    cache_key = tuple(sorted(stage2_kwargs.items()))
    if cache_key not in cache:
        cache[cache_key] = run_stage2_research_engine(Stage2Config(**stage2_kwargs))
    return cache[cache_key]


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


def run_policy_gate_cycle(base_config: dict[str, Any], controller: dict[str, Any]) -> dict[str, Any]:
    result = run_stage2_research_engine(Stage2Config(**_stage2_config_kwargs(base_config)))
    validation = build_validation_audit(result, {})
    rows: list[dict[str, Any]] = []
    for policy in get_break_quality_policies():
        filtered = _apply_benchmark_contract(
            result,
            validation,
            threshold=float(controller.get("frozen_threshold", 0.45) or 0.45),
            gate_policy=policy["name"],
            meta_policy=controller.get("benchmark_meta_policy"),
        )
        execution = run_event_driven_policy_backtest(filtered, threshold=0.0)
        utility = compute_execution_utility(execution) if execution.get("status") == "complete" else {"score": None}
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


def run_policy_meta_cycle(base_config: dict[str, Any], controller: dict[str, Any]) -> dict[str, Any]:
    result = run_stage2_research_engine(Stage2Config(**_stage2_config_kwargs(base_config)))
    validation = build_validation_audit(result, {})
    rows: list[dict[str, Any]] = []
    for policy in get_reclaim_meta_policies():
        filtered = _apply_benchmark_contract(
            result,
            validation,
            threshold=float(controller.get("frozen_threshold", 0.45) or 0.45),
            gate_policy=controller.get("benchmark_policy_gate"),
            meta_policy=policy["name"],
        )
        execution = run_event_driven_policy_backtest(filtered, threshold=0.0)
        utility = compute_execution_utility(execution) if execution.get("status") == "complete" else {"score": None}
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


def run_boundary_confirmation_cycle(base_config: dict[str, Any], controller: dict[str, Any], *, boundary_role: str) -> dict[str, Any]:
    from trading_ml.config import load_databento_manifest

    manifest = load_databento_manifest()
    source_path = select_manifest_source_path(manifest, timeframe=str(base_config.get("timeframe", "30s")), boundary_role=boundary_role)
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
    result = run_stage2_research_engine(Stage2Config(**_stage2_config_kwargs(trial_config)))
    validation = build_validation_audit(result, {})
    filtered = _apply_benchmark_contract(
        result,
        validation,
        threshold=float(controller.get("frozen_threshold", 0.45) or 0.45),
        gate_policy=controller.get("benchmark_policy_gate"),
        meta_policy=controller.get("benchmark_meta_policy"),
    )
    execution = run_event_driven_policy_backtest(filtered, threshold=0.0)
    utility = compute_execution_utility(execution) if execution.get("status") == "complete" else {"score": None}
    translation = build_translation_analysis(result, filtered)
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


def _apply_benchmark_contract(
    result: dict[str, Any],
    validation: dict[str, Any],
    *,
    threshold: float,
    gate_policy: str | None,
    meta_policy: str | None,
) -> list[dict[str, Any]]:
    filtered = list(validation.get("walk_forward", {}).get("stitched_prediction_records", []))
    if gate_policy:
        filtered = apply_break_quality_policy(filtered, result.get("features_records", []), policy_name=gate_policy, threshold=threshold)
        threshold = 0.0
    if meta_policy:
        filtered = apply_reclaim_meta_policy(filtered, policy_name=meta_policy)
    if threshold > 0:
        filtered = [row for row in filtered if float(row.get("probability", 0.0) or 0.0) >= threshold]
    return filtered


def _package_policy_cycle(family: str, controller: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: (float(row["utility_score"] or float("-inf")), row["total_pnl_r"]), reverse=True)
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
