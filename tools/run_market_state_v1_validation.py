from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.config import load_bnr_config
from trading_ml.deflated_sharpe_analysis import compute_sharpe_ratio, deflated_sharpe_probability
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.market_state_quality import (
    _followthrough_gate_decision,
    build_market_state_setup_quality_diagnostic,
)
from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.translation_analysis import build_translation_analysis
from trading_ml.validation_audit import build_validation_audit


FROZEN_VARIANT = "market_state_setup_quality_v1"
N_RECENT_POLICY_TRIALS = 11


def main() -> None:
    state = build_agent_loop_state()
    run_id = "market-state-v1-validation"
    stage2_config = dict(state["stage2_config"])
    diagnostic = build_market_state_setup_quality_diagnostic(state)
    if diagnostic.get("status") != "complete":
        raise RuntimeError(f"diagnostic unavailable: {diagnostic.get('reason')}")

    stage2 = run_stage2_research_engine(Stage2Config(**stage2_config))
    validation = build_validation_audit(stage2, {"trial_count": N_RECENT_POLICY_TRIALS}, artifact_context={"run_id": run_id})
    feature_by_id = {str(row["candidate_id"]): row for row in diagnostic["_labeled_rows"]}
    stitched = list(validation.get("walk_forward", {}).get("stitched_prediction_records", []) or [])
    policy_records = _apply_market_state_v1(stitched, feature_by_id)
    threshold = float(load_bnr_config().get("frozen_benchmark", {}).get("threshold", 0.45) or 0.45)
    execution = run_event_driven_policy_backtest(policy_records, threshold=threshold)
    translation = build_translation_analysis(stage2, policy_records)
    cpcv = _candidate_cpcv(diagnostic)
    dsr_psr = _dsr_psr(execution, diagnostic)
    calibration = _calibration_by_state(policy_records)
    state_contribution = _state_contribution(execution, feature_by_id)
    gates = _pass_fail(validation, cpcv, dsr_psr, execution, translation, diagnostic)

    report = {
        "status": "complete",
        "candidate": FROZEN_VARIANT,
        "definition": {
            "base": "BNR benchmark exploration data only",
            "policy": "exclude weak_or_grindy_continuation + pre-entry breakout quality gate",
            "threshold": threshold,
            "holdout_status": "locked",
            "variants_tested_this_pass": 1,
            "n_trials_for_dsr_psr": N_RECENT_POLICY_TRIALS,
        },
        "validation": {
            "walk_forward": {
                "status": validation.get("walk_forward", {}).get("status"),
                "mean_roc_auc": validation.get("walk_forward", {}).get("mean_roc_auc"),
                "fold_count": validation.get("walk_forward", {}).get("fold_count"),
            },
            "purging": validation.get("purging", {}),
            "cpcv": cpcv,
            "dsr_psr": dsr_psr,
            "translation": translation,
            "execution": execution,
            "calibration_by_state_bucket": calibration,
            "state_contribution": state_contribution,
            "daily_session_pnl": execution.get("session_rows", []),
            "slippage_commission_assumptions": execution.get("fill_assumptions", {}),
            "leakage_audit": {
                "status": "pass"
                if diagnostic.get("leakage_audit", {}).get("status") == "pass"
                and diagnostic.get("followthrough_confirmation_policy_gate", {}).get("leakage_audit", {}).get("status") == "pass"
                else "fail",
                "diagnostic_leakage": diagnostic.get("leakage_audit", {}),
                "policy_gate_leakage": diagnostic.get("followthrough_confirmation_policy_gate", {}).get("leakage_audit", {}),
            },
        },
        "pass_fail": gates,
        "promotion_decision": "advance_to_holdout_request" if all(gates.values()) else "blocked",
    }
    output = REPORTS_DIR / "market_state_setup_quality_v1_validation.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(_console_summary(report, output), indent=2, default=str))


def _apply_market_state_v1(records: list[dict[str, Any]], feature_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = []
    for row in records:
        joined = dict(row)
        joined.update(feature_by_id.get(str(row.get("candidate_id")), {}))
        keep, fill_impact, pit_valid = _followthrough_gate_decision(joined, "breakout_quality")
        if keep <= 0 or not pit_valid:
            continue
        if fill_impact:
            joined["pnl_r"] = float(joined.get("pnl_r", 0.0) or 0.0) + fill_impact
        filtered.append(joined)
    return filtered


def _candidate_cpcv(diagnostic: dict[str, Any]) -> dict[str, Any]:
    variants = list(diagnostic.get("followthrough_confirmation_policy_gate", {}).get("policy_variants", []) or [])
    row = next(item for item in variants if item["variant"] == "exclude_weak_or_grindy_plus_pre_entry_breakout_quality_gate")
    paths = list(row.get("cpcv_paths", []) or [])
    pnls = sorted(float(path.get("total_pnl_r", 0.0) or 0.0) for path in paths)
    pbo = sum(1 for pnl in pnls if pnl <= 0) / len(pnls) if pnls else 1.0
    status = (
        "pass"
        if pbo <= 0.25
        and row["mean_cpcv_path_pnl_r"] > 0
        and row["median_cpcv_path_pnl_r"] > 0
        and row["positive_path_rate"] >= 0.60
        and min(pnls) > -5.0
        else "fail"
    )
    return {
        "status": status,
        "pbo": pbo,
        "mean_total_pnl_r": row["mean_cpcv_path_pnl_r"],
        "median_total_pnl_r": row["median_cpcv_path_pnl_r"],
        "min_path_pnl_r": min(pnls) if pnls else 0.0,
        "path_positive_rate": row["positive_path_rate"],
        "worst_paths": row["worst_3_cpcv_paths"],
        "prior_worst_path_effect": row["prior_worst_path_effect"],
        "paths": paths,
    }


def _dsr_psr(execution: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    returns = [float(row.get("executed_pnl_r", 0.0) or 0.0) for row in execution.get("equity_curve", [])]
    observed = compute_sharpe_ratio(returns)
    trial_srs = []
    for section in ["cheap_state_policy_simulation", "followthrough_confirmation_policy_gate"]:
        for row in diagnostic.get(section, {}).get("policy_variants", []):
            pnls = [float(path.get("total_pnl_r", 0.0) or 0.0) for path in row.get("cpcv_paths", [])]
            sr = compute_sharpe_ratio(pnls)
            if sr is not None:
                trial_srs.append(sr)
    sr_std = _sample_std(trial_srs)
    dsr_probability = (
        deflated_sharpe_probability(observed_sr=observed, n_trials=N_RECENT_POLICY_TRIALS, sr_std=sr_std, n_obs=len(returns))
        if observed is not None and sr_std > 0
        else 0.0
    )
    psr_probability = (
        deflated_sharpe_probability(observed_sr=observed, n_trials=1, sr_std=1.0, n_obs=len(returns))
        if observed is not None
        else 0.0
    )
    return {
        "status": "pass" if dsr_probability >= 0.95 and psr_probability >= 0.95 and (observed or 0.0) > 0 else "fail",
        "dsr_status": "pass" if dsr_probability >= 0.95 and (observed or 0.0) > 0 else "fail",
        "psr_status": "pass" if psr_probability >= 0.95 and (observed or 0.0) > 0 else "fail",
        "dsr_probability": dsr_probability,
        "psr_probability": psr_probability,
        "observed_sharpe": observed,
        "n_trials": N_RECENT_POLICY_TRIALS,
        "n_obs": len(returns),
        "sr_std": sr_std,
    }


def _calibration_by_state(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError:
        return []
    frame = pd.DataFrame(records)
    if frame.empty:
        return []
    rows = []
    for state, group in frame.groupby("market_state", dropna=False):
        rows.append(
            {
                "market_state": str(state),
                "count": int(len(group)),
                "probability_mean": float(group["probability"].mean()),
                "hit_rate": float(group["label"].mean()) if "label" in group else None,
                "avg_pnl_r": float(group["pnl_r"].mean()),
            }
        )
    return sorted(rows, key=lambda item: item["count"], reverse=True)


def _state_contribution(execution: dict[str, Any], feature_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in execution.get("equity_curve", []):
        state = str(feature_by_id.get(str(row.get("candidate_id")), {}).get("market_state", "unknown"))
        bucket = groups.setdefault(state, {"market_state": state, "trade_count": 0, "total_pnl_r": 0.0})
        bucket["trade_count"] += 1
        bucket["total_pnl_r"] += float(row.get("executed_pnl_r", 0.0) or 0.0)
    return sorted(groups.values(), key=lambda item: item["trade_count"], reverse=True)


def _pass_fail(
    validation: dict[str, Any],
    cpcv: dict[str, Any],
    dsr_psr: dict[str, Any],
    execution: dict[str, Any],
    translation: dict[str, Any],
    diagnostic: dict[str, Any],
) -> dict[str, bool]:
    return {
        "walk_forward_pass": validation.get("walk_forward", {}).get("status") == "pass",
        "cpcv_pass": cpcv.get("status") == "pass",
        "worst_path_gt_minus_5r": float(cpcv.get("min_path_pnl_r", 0.0) or 0.0) > -5.0,
        "dsr_psr_pass": dsr_psr.get("status") == "pass",
        "trade_count_acceptable": int(execution.get("trade_count", 0) or 0) >= 100,
        "translation_pass": translation.get("status") == "pass",
        "drawdown_tolerable": float(execution.get("max_drawdown_r", 0.0) or 0.0) > -12.0,
        "no_leakage": diagnostic.get("leakage_audit", {}).get("status") == "pass",
    }


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _console_summary(report: dict[str, Any], output: Path) -> dict[str, Any]:
    cpcv = report["validation"]["cpcv"]
    execution = report["validation"]["execution"]
    return {
        "output": str(output),
        "promotion_decision": report["promotion_decision"],
        "pass_fail": report["pass_fail"],
        "walk_forward": report["validation"]["walk_forward"],
        "cpcv": {
            "status": cpcv["status"],
            "mean": cpcv["mean_total_pnl_r"],
            "median": cpcv["median_total_pnl_r"],
            "min_path": cpcv["min_path_pnl_r"],
            "positive_path_rate": cpcv["path_positive_rate"],
            "pbo": cpcv["pbo"],
            "worst_paths": [(row["path_id"], row["total_pnl_r"]) for row in cpcv["worst_paths"]],
        },
        "dsr_psr": report["validation"]["dsr_psr"],
        "execution": {
            "status": execution.get("status"),
            "trade_count": execution.get("trade_count"),
            "total_pnl_r": execution.get("total_pnl_r"),
            "max_drawdown_r": execution.get("max_drawdown_r"),
            "win_rate": execution.get("win_rate"),
            "session_count": execution.get("session_count"),
        },
    }


if __name__ == "__main__":
    main()
