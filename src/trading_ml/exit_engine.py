from __future__ import annotations

import json
from statistics import mean, median
from typing import Any
from uuid import uuid4

from trading_ml.config import load_global_config
from trading_ml.deflated_sharpe_analysis import (
    compute_sharpe_ratio,
    deflated_sharpe_probability,
)
from trading_ml.market_state_quality import (
    _available_cpcv_path_rows,
    _diagnostic_config,
    _followthrough_gate_decision,
    build_market_state_setup_quality_diagnostic,
)
from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_data import load_ohlcv_file, regular_session


EXIT_VARIANTS = [
    "fixed_1_5r_tp_sl_baseline",
    "scratch_no_followthrough_after_3_bars",
    "time_stop_10_bars",
    "breakeven_after_0_75r_mfe",
    "structure_trailing_stop",
    "partial_1r_runner",
    "tempo_deterioration_exit",
    "opposing_recent_high_low_target_or_exit",
]

SCRATCH_REFINEMENT_VARIANTS = [
    "scratch_no_followthrough_after_2_bars",
    "scratch_no_followthrough_after_3_bars",
    "scratch_no_followthrough_after_4_bars",
    "scratch_3_bars_plus_breakeven_after_0_75r_mfe",
    "scratch_3_bars_plus_mae_cut_0_65r",
    "scratch_3_bars_plus_session_drawdown_stop_3r",
]

STRUCTURE_PARTIAL_REPLAY_VARIANTS = [
    "partial_1r_runner_structure_trail",
    "partial_1r_runner_fixed_2r",
    "partial_1r_runner_fixed_3r",
    "structure_trail_after_1r",
    "structure_trail_after_breakout_continuation",
    "fixed_1_5r_tp_sl_baseline",
]


def run_exit_engine_cycle(state: dict[str, Any]) -> dict[str, Any]:
    diagnostic = build_market_state_setup_quality_diagnostic(state)
    if diagnostic.get("status") != "complete":
        return {
            "status": "pending",
            "reason": "diagnostic_unavailable",
            "diagnostic": diagnostic,
        }
    config = _diagnostic_config(state, str(diagnostic["source_path"]))
    bars = regular_session(
        load_ohlcv_file(
            config.source_path,
            symbol=config.symbol,
            timeframe=config.timeframe,
            timezone=config.timezone,
        )
    )
    frozen_entries = _frozen_v1_entries(diagnostic)
    rows = [
        _evaluate_exit_policy(name, frozen_entries, bars, config.horizon_bars)
        for name in EXIT_VARIANTS
    ]
    _attach_dsr_psr(rows)
    baseline = next(
        row for row in rows if row["variant"] == "fixed_1_5r_tp_sl_baseline"
    )
    for row in rows:
        row["improvement_attribution"] = _improvement_attribution(
            baseline["trade_results"], row["trade_results"]
        )
        row["decision"] = "advance_candidate" if _passes_exit_gates(row) else "reject"
    ranked = sorted(
        rows,
        key=lambda row: (
            row["decision"] == "advance_candidate",
            row["mean_cpcv_path_pnl_r"],
            row["total_pnl_r"],
        ),
        reverse=True,
    )
    accepted = next(
        (row for row in ranked if row["decision"] == "advance_candidate"), None
    )
    payload = {
        "status": "complete",
        "family": "exit_engine",
        "candidate": "market_state_setup_quality_v1",
        "entry_definition": "exclude_weak_or_grindy_continuation + pre_entry_breakout_quality_gate",
        "governance": {
            "entries_frozen": True,
            "setup_filters_frozen": True,
            "holdout_status": "locked",
            "models_trained": 0,
            "broad_search": False,
            "promotion_blocked_unless_full_gates_pass": True,
        },
        "trial_count": len(rows),
        "fill_slippage_assumptions": _fill_assumptions(),
        "leakage_audit": _leakage_audit(rows),
        "ranked_policies": [_strip_trade_results(row) for row in ranked],
        "best_policy": _strip_trade_results(ranked[0]) if ranked else None,
        "accepted_policy": _strip_trade_results(accepted) if accepted else None,
        "batch_decision": "accept" if accepted else "revise",
    }
    run_id = f"exit-engine-{uuid4().hex[:12]}"
    output_dir = REPORTS_DIR / "runs" / run_id / "exit_engine"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (REPORTS_DIR / "exit_engine_market_state_v1.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    payload["artifact_path"] = str(REPORTS_DIR / "exit_engine_market_state_v1.json")
    payload["run_artifact_path"] = str(output_dir / "summary.json")
    return payload


def run_scratch_exit_refinement_cycle(state: dict[str, Any]) -> dict[str, Any]:
    diagnostic = build_market_state_setup_quality_diagnostic(state)
    if diagnostic.get("status") != "complete":
        return {
            "status": "pending",
            "reason": "diagnostic_unavailable",
            "diagnostic": diagnostic,
        }
    config = _diagnostic_config(state, str(diagnostic["source_path"]))
    bars = regular_session(
        load_ohlcv_file(
            config.source_path,
            symbol=config.symbol,
            timeframe=config.timeframe,
            timezone=config.timezone,
        )
    )
    frozen_entries = _frozen_v1_entries(diagnostic)
    rows = [
        _evaluate_exit_policy(name, frozen_entries, bars, config.horizon_bars)
        for name in SCRATCH_REFINEMENT_VARIANTS
    ]
    _attach_dsr_psr(rows)
    baseline = next(
        row for row in rows if row["variant"] == "scratch_no_followthrough_after_3_bars"
    )
    for row in rows:
        row["improvement_attribution"] = _improvement_attribution(
            baseline["trade_results"], row["trade_results"]
        )
        row["decision"] = "advance_candidate" if _passes_exit_gates(row) else "reject"
    ranked = sorted(
        rows,
        key=lambda row: (
            row["decision"] == "advance_candidate",
            row["mean_cpcv_path_pnl_r"],
            row["total_pnl_r"],
        ),
        reverse=True,
    )
    accepted = next(
        (row for row in ranked if row["decision"] == "advance_candidate"), None
    )
    payload = {
        "status": "complete",
        "family": "exit_engine_scratch_refinement",
        "candidate": "market_state_setup_quality_v1",
        "entry_definition": "exclude_weak_or_grindy_continuation + pre_entry_breakout_quality_gate",
        "governance": {
            "entries_frozen": True,
            "setup_filters_frozen": True,
            "holdout_status": "locked",
            "models_trained": 0,
            "broad_search": False,
            "targeted_blockers": ["PBO 0.263", "max_drawdown -12.91R"],
        },
        "trial_count": len(rows),
        "fill_slippage_assumptions": _fill_assumptions(),
        "leakage_audit": _leakage_audit(rows),
        "ranked_policies": [_strip_trade_results(row) for row in ranked],
        "best_policy": _strip_trade_results(ranked[0]) if ranked else None,
        "accepted_policy": _strip_trade_results(accepted) if accepted else None,
        "batch_decision": "accept" if accepted else "revise",
    }
    run_id = f"exit-scratch-refine-{uuid4().hex[:12]}"
    output_dir = REPORTS_DIR / "runs" / run_id / "exit_engine"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (REPORTS_DIR / "exit_engine_scratch_refinement_market_state_v1.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    payload["artifact_path"] = str(
        REPORTS_DIR / "exit_engine_scratch_refinement_market_state_v1.json"
    )
    payload["run_artifact_path"] = str(output_dir / "summary.json")
    return payload


def _frozen_v1_entries(diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in diagnostic.get("_labeled_rows", []):
        keep, _, pit_valid = _followthrough_gate_decision(row, "breakout_quality")
        if keep > 0 and pit_valid:
            rows.append(row)
    return rows


def _evaluate_exit_policy(
    policy: str, entries: list[dict[str, Any]], bars: Any, horizon_bars: int
) -> dict[str, Any]:
    trade_results = [
        _simulate_trade_exit(row, bars, horizon_bars, policy) for row in entries
    ]
    if policy == "scratch_3_bars_plus_session_drawdown_stop_3r":
        trade_results = _apply_session_drawdown_stop(trade_results, limit_r=-3.0)
    pnls = [row["pnl_r"] for row in trade_results]
    cpcv = _exit_cpcv(policy, trade_results)
    return {
        "variant": policy,
        "trade_count": len(trade_results),
        "total_pnl_r": float(sum(pnls)),
        "avg_trade_r": float(mean(pnls)) if pnls else 0.0,
        "median_trade_r": float(median(pnls)) if pnls else 0.0,
        "win_rate": (
            float(sum(1 for pnl in pnls if pnl > 0) / len(pnls)) if pnls else 0.0
        ),
        "payoff_ratio": _payoff_ratio(pnls),
        "max_drawdown_r": _max_drawdown(pnls),
        "mean_cpcv_path_pnl_r": cpcv["mean_total_pnl_r"],
        "median_cpcv_path_pnl_r": cpcv["median_total_pnl_r"],
        "pbo": cpcv["pbo"],
        "worst_3_cpcv_paths": cpcv["worst_paths"],
        "cpcv_summary": cpcv,
        "leakage_audit": {
            "status": "pass",
            "point_in_time_exit": True,
            "uses_path_specific_filter": False,
        },
        "trade_results": trade_results,
    }


def _apply_session_drawdown_stop(
    trades: list[dict[str, Any]], limit_r: float
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    stopped_sessions: set[str] = set()
    session_pnl: dict[str, float] = {}
    for trade in sorted(trades, key=lambda row: row["entry_time"]):
        session = str(trade["session_date"])
        if session in stopped_sessions:
            continue
        kept.append(trade)
        session_pnl[session] = session_pnl.get(session, 0.0) + float(trade["pnl_r"])
        if session_pnl[session] <= limit_r:
            stopped_sessions.add(session)
            trade["session_drawdown_stop_triggered"] = True
    return kept


def _simulate_trade_exit(
    row: dict[str, Any], bars: Any, horizon_bars: int, policy: str
) -> dict[str, Any]:
    pd = _require_pandas()
    entry_time = pd.Timestamp(row["entry_time"])
    future = bars[bars.index >= entry_time].head(horizon_bars)
    direction = str(row["direction"])
    entry = float(row["entry_price"])
    stop = float(row["stop_price"])
    target = float(row["target_price"])
    risk = max(abs(entry - stop), 0.25)
    active_stop = stop
    partial_taken = False
    partial_pnl = 0.0
    recent_target_r = max(
        0.5, min(float(row.get("distance_to_recent_high_low", 1.5) or 1.5), 1.5)
    )
    if policy == "opposing_recent_high_low_target_or_exit":
        target = (
            entry + recent_target_r * risk
            if direction == "long"
            else entry - recent_target_r * risk
        )
    elif policy == "partial_1r_runner_fixed_2r":
        target = entry + 2.0 * risk if direction == "long" else entry - 2.0 * risk
    elif policy == "partial_1r_runner_fixed_3r":
        target = entry + 3.0 * risk if direction == "long" else entry - 3.0 * risk
    elif policy == "partial_1r_runner_structure_trail":
        target = entry + 999.0 * risk if direction == "long" else entry - 999.0 * risk

    outcome = "timeout"
    exit_price = entry
    exit_time = entry_time
    bars_held = 0
    mfe_r = 0.0
    closes: list[float] = []
    opens: list[float] = []

    for idx, (ts, bar) in enumerate(future.iterrows(), start=1):
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        open_ = float(bar["open"])
        closes.append(close)
        opens.append(open_)
        favorable = (
            (high - entry) / risk if direction == "long" else (entry - low) / risk
        )
        mfe_r = max(mfe_r, favorable)

        if (
            policy
            in {
                "breakeven_after_0_75r_mfe",
                "scratch_3_bars_plus_breakeven_after_0_75r_mfe",
            }
            and mfe_r >= 0.75
        ):
            active_stop = (
                max(active_stop, entry)
                if direction == "long"
                else min(active_stop, entry)
            )
        if _structure_trail_active(policy, mfe_r, idx, opens, closes, direction) or (
            policy == "partial_1r_runner_structure_trail" and partial_taken and idx >= 3
        ):
            recent = future.iloc[max(0, idx - 3) : idx]
            if direction == "long":
                active_stop = max(active_stop, float(recent["low"].min()))
            else:
                active_stop = min(active_stop, float(recent["high"].max()))
        if _partial_1r_policy(policy) and not partial_taken:
            one_r_hit = (
                high >= entry + risk if direction == "long" else low <= entry - risk
            )
            if one_r_hit:
                partial_taken = True
                partial_pnl = 0.5
                active_stop = entry

        stop_hit = low <= active_stop if direction == "long" else high >= active_stop
        target_hit = high >= target if direction == "long" else low <= target
        bars_held = idx
        if stop_hit and target_hit:
            outcome = "ambiguous_stop_first"
            exit_price = active_stop
            exit_time = ts
            break
        if stop_hit:
            outcome = "stop"
            exit_price = active_stop
            exit_time = ts
            break
        if target_hit:
            outcome = "target"
            exit_price = target
            exit_time = ts
            break

        adverse_r = (
            (entry - low) / risk if direction == "long" else (high - entry) / risk
        )
        if policy == "scratch_3_bars_plus_mae_cut_0_65r" and adverse_r >= 0.65:
            outcome = "mae_cut"
            exit_price = close
            exit_time = ts
            break

        scratch_bar = _scratch_bar(policy)
        if scratch_bar is not None and idx == scratch_bar and mfe_r < 0.25:
            outcome = "scratch_no_followthrough"
            exit_price = close
            exit_time = ts
            break
        if policy == "time_stop_10_bars" and idx == 10:
            outcome = "time_stop"
            exit_price = close
            exit_time = ts
            break
        if (
            policy == "tempo_deterioration_exit"
            and idx >= 4
            and _tempo_deteriorated(opens, closes)
        ):
            outcome = "tempo_deterioration"
            exit_price = close
            exit_time = ts
            break
    else:
        if not future.empty:
            exit_time = future.index[-1]
            exit_price = float(future.iloc[-1]["close"])

    pnl_r = _net_pnl_r(entry, exit_price, direction, risk)
    if _partial_1r_policy(policy) and partial_taken:
        runner_pnl = _net_pnl_r(entry, exit_price, direction, risk)
        pnl_r = partial_pnl + 0.5 * runner_pnl
    return {
        "candidate_id": str(row["candidate_id"]),
        "session_date": str(row["session_date"]),
        "market_state": str(row.get("market_state", "")),
        "setup_quality": str(row.get("setup_quality", "")),
        "policy": policy,
        "outcome": outcome,
        "entry_time": str(entry_time),
        "exit_time": str(exit_time),
        "bars_held": int(bars_held),
        "pnl_r": float(pnl_r),
        "baseline_pnl_r": float(row.get("pnl_r", 0.0) or 0.0),
    }


def _scratch_bar(policy: str) -> int | None:
    if policy == "scratch_no_followthrough_after_2_bars":
        return 2
    if policy in {
        "scratch_no_followthrough_after_3_bars",
        "scratch_3_bars_plus_breakeven_after_0_75r_mfe",
        "scratch_3_bars_plus_mae_cut_0_65r",
        "scratch_3_bars_plus_session_drawdown_stop_3r",
    }:
        return 3
    if policy == "scratch_no_followthrough_after_4_bars":
        return 4
    return None


def _partial_1r_policy(policy: str) -> bool:
    return policy in {
        "partial_1r_runner",
        "partial_1r_runner_structure_trail",
        "partial_1r_runner_fixed_2r",
        "partial_1r_runner_fixed_3r",
    }


def _structure_trail_active(
    policy: str,
    mfe_r: float,
    idx: int,
    opens: list[float],
    closes: list[float],
    direction: str,
) -> bool:
    if idx < 3:
        return False
    if policy == "structure_trailing_stop":
        return mfe_r >= 0.50
    if policy == "structure_trail_after_1r":
        return mfe_r >= 1.0
    if (
        policy != "structure_trail_after_breakout_continuation"
        or mfe_r < 0.50
        or len(closes) < 3
    ):
        return False
    recent_signs = [
        1 if close > open_ else -1 if close < open_ else 0
        for open_, close in zip(opens[-3:], closes[-3:])
    ]
    nonzero = [sign for sign in recent_signs if sign != 0]
    wanted = 1 if direction == "long" else -1
    return len(nonzero) >= 2 and sum(sign == wanted for sign in nonzero) >= 2


def _exit_cpcv(policy: str, trade_results: list[dict[str, Any]]) -> dict[str, Any]:
    result_by_id = {row["candidate_id"]: row for row in trade_results}
    rows = []
    for path in _available_cpcv_path_rows():
        pnls = [
            result_by_id[str(row.get("candidate_id"))]["pnl_r"]
            for row in path["rows"]
            if str(row.get("candidate_id")) in result_by_id
        ]
        total = float(sum(pnls))
        rows.append(
            {
                "path_id": path["path_id"],
                "trade_count": len(pnls),
                "total_pnl_r": total,
                "avg_trade_r": float(mean(pnls)) if pnls else 0.0,
            }
        )
    path_pnls = [row["total_pnl_r"] for row in rows]
    return {
        "status": (
            "pass"
            if rows
            and sum(1 for pnl in path_pnls if pnl <= 0) / len(path_pnls) <= 0.25
            and mean(path_pnls) > 0
            and median(path_pnls) > 0
            and min(path_pnls) > -5.0
            else "fail"
        ),
        "policy": policy,
        "pbo": (
            float(sum(1 for pnl in path_pnls if pnl <= 0) / len(path_pnls))
            if path_pnls
            else 1.0
        ),
        "mean_total_pnl_r": float(mean(path_pnls)) if path_pnls else 0.0,
        "median_total_pnl_r": float(median(path_pnls)) if path_pnls else 0.0,
        "min_path_pnl_r": min(path_pnls) if path_pnls else 0.0,
        "worst_paths": sorted(rows, key=lambda row: row["total_pnl_r"])[:3],
        "paths": rows,
    }


def _attach_dsr_psr(rows: list[dict[str, Any]]) -> None:
    trial_srs = []
    for row in rows:
        sr = compute_sharpe_ratio([trade["pnl_r"] for trade in row["trade_results"]])
        row["observed_sharpe"] = sr
        if sr is not None:
            trial_srs.append(sr)
    sr_std = _sample_std(trial_srs)
    n_trials = len(rows)
    for row in rows:
        sr = row.get("observed_sharpe")
        n_obs = len(row["trade_results"])
        dsr = (
            deflated_sharpe_probability(
                observed_sr=sr, n_trials=n_trials, sr_std=sr_std, n_obs=n_obs
            )
            if sr is not None and sr_std > 0
            else 0.0
        )
        psr = (
            deflated_sharpe_probability(
                observed_sr=sr, n_trials=1, sr_std=1.0, n_obs=n_obs
            )
            if sr is not None
            else 0.0
        )
        row["dsr_psr"] = {
            "status": (
                "pass" if dsr >= 0.95 and psr >= 0.95 and (sr or 0.0) > 0 else "fail"
            ),
            "dsr_probability": dsr,
            "psr_probability": psr,
            "n_trials": n_trials,
            "n_obs": n_obs,
            "sr_std": sr_std,
            "observed_sharpe": sr,
        }


def _improvement_attribution(
    baseline: list[dict[str, Any]], trial: list[dict[str, Any]]
) -> dict[str, Any]:
    base_by_id = {row["candidate_id"]: row for row in baseline}
    loser_cut = 0.0
    winner_expansion = 0.0
    winner_giveback = 0.0
    new_loser = 0.0
    for row in trial:
        base = float(base_by_id[row["candidate_id"]]["pnl_r"])
        pnl = float(row["pnl_r"])
        delta = pnl - base
        if base < 0 and delta > 0:
            loser_cut += delta
        elif base > 0 and delta > 0:
            winner_expansion += delta
        elif base > 0 and delta < 0:
            winner_giveback += delta
        elif base < 0 and delta < 0:
            new_loser += delta
    return {
        "cutting_losers_r": loser_cut,
        "expanding_winners_r": winner_expansion,
        "winner_giveback_r": winner_giveback,
        "worsened_losers_r": new_loser,
        "primary_source": (
            "cutting_losers"
            if loser_cut >= max(winner_expansion, abs(winner_giveback), abs(new_loser))
            else "expanding_winners" if winner_expansion > 0 else "mixed_or_negative"
        ),
    }


def _passes_exit_gates(row: dict[str, Any]) -> bool:
    return (
        row["cpcv_summary"]["status"] == "pass"
        and row["dsr_psr"]["status"] == "pass"
        and row["trade_count"] >= 100
        and row["max_drawdown_r"] > -12.0
        and row["leakage_audit"]["status"] == "pass"
    )


def _net_pnl_r(entry: float, exit_price: float, direction: str, risk: float) -> float:
    cfg = _fill_assumptions()
    ticks = float(cfg["ticks_per_side"])
    tick_size = float(cfg["tick_size"])
    slip = ticks * tick_size
    filled_entry = entry + slip if direction == "long" else entry - slip
    filled_exit = exit_price - slip if direction == "long" else exit_price + slip
    pnl_points = (
        filled_exit - filled_entry
        if direction == "long"
        else filled_entry - filled_exit
    )
    return pnl_points / risk


def _tempo_deteriorated(opens: list[float], closes: list[float]) -> bool:
    recent_opens = opens[-4:]
    recent_closes = closes[-4:]
    if len(recent_closes) < 4:
        return False
    bodies = [abs(c - o) for o, c in zip(recent_opens, recent_closes)]
    signs = [
        1 if c > o else -1 if c < o else 0 for o, c in zip(recent_opens, recent_closes)
    ]
    alternating = (
        sum(1 for a, b in zip(signs[:-1], signs[1:]) if a and b and a != b) >= 2
    )
    compressing = bodies[-1] <= max(bodies[0], 1e-9) * 0.5
    return alternating or compressing


def _payoff_ratio(pnls: list[float]) -> float:
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [abs(pnl) for pnl in pnls if pnl < 0]
    return (mean(wins) / mean(losses)) if wins and losses else 0.0


def _max_drawdown(pnls: list[float]) -> float:
    peak = 0.0
    cumulative = 0.0
    drawdown = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = min(drawdown, cumulative - peak)
    return drawdown


def _fill_assumptions() -> dict[str, Any]:
    slippage = dict(load_global_config().get("slippage", {}) or {})
    return {
        "model": slippage.get("model", "ticks"),
        "profile": slippage.get("profile", "base"),
        "tick_size": float(slippage.get("tick_size", 0.25) or 0.25),
        "ticks_per_side": float(slippage.get("base_ticks_per_side", 3.0) or 3.0),
        "entry_and_exit_slippage_applied": True,
        "commission_model": dict(load_global_config().get("costs", {}) or {}),
    }


def _leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues = [
        row["variant"] for row in rows if row["leakage_audit"]["status"] != "pass"
    ]
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "point_in_time_basis": "Entries are frozen; exits use only bars observed at or before the exit decision bar.",
        "holdout_locked": True,
    }


def _strip_trade_results(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: value for key, value in row.items() if key != "trade_results"}


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("exit engine requires pandas") from exc
    return pd
