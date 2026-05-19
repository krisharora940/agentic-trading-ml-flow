from __future__ import annotations

from statistics import NormalDist
from typing import Any

from trading_ml.config import load_bnr_config
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest


def compute_sharpe_ratio(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / max(len(returns) - 1, 1)
    std = variance ** 0.5
    if std <= 0:
        return None
    return (mean / std) * (len(returns) ** 0.5)


def build_deflated_sharpe_audit(
    stage2_result: dict[str, Any],
    walk_forward: dict[str, Any],
    cpcv: dict[str, Any],
    search_results: dict[str, Any],
    *,
    controller_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controller = dict(controller_state or {})
    stitched = list(walk_forward.get("stitched_prediction_records", []))
    if not stitched:
        return {"status": "pending", "reason": "missing_stitched_predictions"}

    feature_map = {str(row["candidate_id"]): row for row in stage2_result.get("features_records", [])}
    merged_records = []
    for row in stitched:
        merged = dict(row)
        merged.update(feature_map.get(str(row.get("candidate_id")), {}))
        merged_records.append(merged)

    bnr_config = load_bnr_config()
    frozen = dict(bnr_config.get("frozen_benchmark", {}))
    accepted = dict(search_results.get("accepted_trial", {}) or {})
    overrides = dict(accepted.get("overrides", {}) or {})
    threshold = float(overrides.get("decision_threshold", controller.get("frozen_threshold", frozen.get("threshold", 0.45))) or 0.45)
    sizing_policy = str(overrides.get("sizing_policy", controller.get("benchmark_sizing_policy", frozen.get("sizing_policy", "binary_threshold_v1"))))
    regime_throttle_policy = str(overrides.get("regime_throttle_policy", controller.get("benchmark_regime_throttle_policy", frozen.get("regime_throttle_policy", "none"))))
    regime_size_policy = str(overrides.get("regime_size_policy", controller.get("benchmark_regime_size_policy", frozen.get("regime_size_policy", "none"))))

    execution = run_event_driven_policy_backtest(
        merged_records,
        threshold=threshold,
        sizing_policy=sizing_policy,
        regime_throttle_policy=regime_throttle_policy,
        regime_size_policy=regime_size_policy,
    )
    if execution.get("status") != "complete":
        return {"status": "pending", "reason": "execution_unavailable", "execution_status": execution.get("status")}

    trade_returns = [float(row.get("executed_pnl_r", 0.0) or 0.0) for row in execution.get("equity_curve", [])]
    observed_sr = compute_sharpe_ratio(trade_returns)
    if observed_sr is None:
        return {"status": "pending", "reason": "insufficient_trade_returns", "trade_count": len(trade_returns)}

    sr_std, variance_source = _sharpe_dispersion(cpcv, search_results)
    if sr_std is None or sr_std <= 0:
        return {"status": "pending", "reason": "insufficient_sharpe_dispersion", "trade_count": len(trade_returns)}

    skew = _sample_skew(trade_returns)
    kurtosis = _sample_kurtosis(trade_returns)
    n_trials = max(int(search_results.get("trial_count", 0) or 0), 1)
    dsr_probability = deflated_sharpe_probability(
        observed_sr=observed_sr,
        n_trials=n_trials,
        sr_std=sr_std,
        n_obs=len(trade_returns),
        skew=skew,
        kurtosis=kurtosis,
    )
    status = "pass" if dsr_probability >= 0.95 and observed_sr > 0 else "fail"
    return {
        "status": status,
        "probability": dsr_probability,
        "observed_sharpe": observed_sr,
        "deflated_sharpe": observed_sr - _expected_max_sr(n_trials, sr_std),
        "n_trials": n_trials,
        "n_obs": len(trade_returns),
        "sr_std": sr_std,
        "variance_source": variance_source,
        "skew": skew,
        "kurtosis": kurtosis,
        "trade_count": int(execution.get("trade_count", 0) or 0),
        "threshold": threshold,
        "sizing_policy": sizing_policy,
        "regime_throttle_policy": regime_throttle_policy,
        "regime_size_policy": regime_size_policy,
    }


def deflated_sharpe_probability(
    *,
    observed_sr: float,
    n_trials: int,
    sr_std: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    if n_trials <= 0 or sr_std <= 0 or n_obs <= 1:
        return 0.0
    expected_max = _expected_max_sr(n_trials, sr_std)
    denominator = ((1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * (observed_sr**2)) / max(n_obs - 1, 1)) ** 0.5
    if denominator <= 0:
        return 0.0
    test_stat = (observed_sr - expected_max) / denominator
    return NormalDist().cdf(test_stat)


def _expected_max_sr(n_trials: int, sr_std: float) -> float:
    euler_mascheroni = 0.5772156649
    normal = NormalDist()
    term_a = (1.0 - euler_mascheroni) * normal.inv_cdf(1.0 - 1.0 / n_trials)
    term_b = euler_mascheroni * normal.inv_cdf(1.0 - 1.0 / (n_trials * 2.718281828459045))
    return sr_std * (term_a + term_b)


def _sharpe_dispersion(cpcv: dict[str, Any], search_results: dict[str, Any]) -> tuple[float | None, str]:
    path_sharpes = [float(row.get("sharpe_r", 0.0) or 0.0) for row in cpcv.get("paths", []) if row.get("sharpe_r") is not None]
    if len(path_sharpes) >= 2:
        return _sample_std(path_sharpes), "cpcv_path_sharpe"
    trial_proxies = [float(row.get("net_avg_pnl_r", 0.0) or 0.0) for row in search_results.get("ranked_trials", [])]
    if len(trial_proxies) >= 2:
        return _sample_std(trial_proxies), "trial_net_avg_pnl_r_proxy"
    return None, "unavailable"


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return variance ** 0.5


def _sample_skew(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    mean = sum(values) / len(values)
    std = _sample_std(values)
    if std <= 0:
        return 0.0
    return sum(((value - mean) / std) ** 3 for value in values) / len(values)


def _sample_kurtosis(values: list[float]) -> float:
    if len(values) < 4:
        return 3.0
    mean = sum(values) / len(values)
    std = _sample_std(values)
    if std <= 0:
        return 3.0
    return sum(((value - mean) / std) ** 4 for value in values) / len(values)
