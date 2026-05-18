from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

from trading_ml.config import load_bnr_config, load_global_config
from trading_ml.registry import append_experiment_record
from trading_ml.schemas import ExperimentRecord
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine


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


def load_controller_config(override: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_bnr_config()
    controller = dict(config.get("controller", {}))
    controller.setdefault("spec_version", "bnr_spec_vA")
    controller.setdefault("active_family", "setup")
    controller.setdefault("active_model_family", "linear_baseline")
    controller.setdefault("frozen_threshold", config.get("controller", {}).get("frozen_threshold", 0.45))
    controller.setdefault("benchmark_name", config.get("controller", {}).get("benchmark_name", "bnr_hybrid_linear_v1"))
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
    baseline_result = run_stage2_research_engine(Stage2Config(**base_config))
    baseline = _summarize_baseline(active_family, str(controller["spec_version"]), baseline_result)
    trials = generate_search_trials(base_config, family=active_family, controller_override=controller)
    results: list[ControllerTrialSummary] = []
    for idx, trial_config in enumerate(trials, start=1):
        result = run_stage2_research_engine(Stage2Config(**trial_config))
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
        "space": build_search_space() if active_family == "setup" else build_model_search_space(),
        "controller": controller,
        "baseline": baseline.to_dict(),
        "trial_count": len(results),
        "ranked_trials": [trial.to_dict() for trial in ranked],
        "best_trial": ranked[0].to_dict() if ranked else None,
        "accepted_trial": accepted.to_dict() if accepted else None,
        "batch_decision": "accept" if accepted is not None else "revise",
    }


def _summarize_baseline(family: str, spec_version: str, result: dict[str, Any]) -> ControllerTrialSummary:
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
        net_avg_pnl_r=_estimate_net_avg_pnl_r(result),
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
    net_avg_pnl_r = _estimate_net_avg_pnl_r(result)
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


def _estimate_net_avg_pnl_r(result: dict[str, Any]) -> float:
    global_config = load_global_config()
    label_summary = result.get("label_summary", {})
    avg_pnl_r = float(label_summary.get("avg_pnl_r", 0.0) or 0.0)
    model_summary = result.get("model_summary", {})
    avg_risk_points = float(model_summary.get("avg_risk_points", 0.0) or 0.0)
    avg_entry_price = float(model_summary.get("avg_entry_price", 0.0) or 0.0)
    if avg_risk_points <= 0 or avg_entry_price <= 0:
        return avg_pnl_r
    entry_bps = float(global_config["slippage"]["entry_bps"])
    exit_bps = float(global_config["slippage"]["exit_bps"])
    total_bps = entry_bps + exit_bps
    cost_points = avg_entry_price * (total_bps / 10_000.0)
    cost_r = cost_points / avg_risk_points
    return avg_pnl_r - cost_r
