from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
from ml4t.backtest import (
    AssetClass,
    BacktestConfig,
    ContractSpec,
    ExecutionMode,
    Strategy,
    run_backtest,
)
from ml4t.backtest.config import DataFrequency, ExecutionPrice, SlippageType

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.config import load_bnr_config, load_databento_manifest, load_global_config
from trading_ml.evidence_sources import select_manifest_source_path
from trading_ml.exit_engine import _simulate_trade_exit
from trading_ml.market_state_quality import (
    _diagnostic_config,
    _followthrough_gate_decision,
    build_market_state_setup_quality_diagnostic,
)
from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_data import load_ohlcv_file, regular_session
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.translation_analysis import build_translation_analysis
from trading_ml.validation_audit import build_validation_audit


BENCHMARK_NAME = "market_state_setup_quality_v1"
ENTRY_POLICY_NAME = "exclude_weak_or_grindy_continuation + pre_entry_breakout_quality_gate"
EXIT_POLICY_NAME = "scratch_no_followthrough_after_3_bars"
N_RECENT_POLICY_TRIALS = 11
BACKTEST_SYMBOL = "MNQ"
MNQ_MULTIPLIER = 2.0


@dataclass(slots=True)
class MarketStateBacktestBundle:
    report: dict[str, Any]
    output_path: Path
    run_dir: Path


class ScheduledTradeReplayStrategy(Strategy):
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.skipped_overlap_entries = 0
        self.same_timestamp_entry_collisions = 0
        self.close_without_position = 0

    def on_data(self, timestamp, data, context, broker) -> None:
        exit_actions = _decode_actions(context.get("exit_actions"))
        entry_actions = _decode_actions(context.get("entry_actions"))
        position = broker.get_position(self.symbol)

        if exit_actions:
            if position is not None:
                broker.close_position(self.symbol)
                position = None
            else:
                self.close_without_position += len(exit_actions)

        if not entry_actions:
            return

        if len(entry_actions) > 1:
            self.same_timestamp_entry_collisions += len(entry_actions) - 1
        action = sorted(entry_actions, key=lambda item: float(item.get("probability", 0.0)), reverse=True)[0]
        if position is not None:
            self.skipped_overlap_entries += 1
            return
        if action["direction"] == "long":
            broker.buy(self.symbol, contracts=1)
        else:
            broker.sell(self.symbol, contracts=1)


def run_market_state_v1_ml4t_backtest(*, boundary_role: str = "exploration") -> MarketStateBacktestBundle:
    state = build_agent_loop_state()
    source_path = select_manifest_source_path(load_databento_manifest(), timeframe="30s", boundary_role=boundary_role)
    if not source_path:
        raise RuntimeError(f"missing_{boundary_role}_source")
    stage2_config = dict(state.get("stage2_config", {}) or {})
    stage2_config["source_path"] = source_path
    state["stage2_config"] = stage2_config
    state["phase"] = "validation_confirmation" if boundary_role == "validation" else "exploration"
    diagnostic = build_market_state_setup_quality_diagnostic(state)
    if diagnostic.get("status") != "complete":
        raise RuntimeError(f"diagnostic unavailable: {diagnostic.get('reason')}")

    source_path = str(diagnostic["source_path"])
    config = _diagnostic_config(state, source_path)
    bars = regular_session(
        load_ohlcv_file(
            config.source_path,
            symbol=config.symbol,
            timeframe=config.timeframe,
            timezone=config.timezone,
        )
    )

    stage2 = run_stage2_research_engine(Stage2Config(**dict(state["stage2_config"])))
    validation = build_validation_audit(
        stage2,
        {"trial_count": N_RECENT_POLICY_TRIALS},
        artifact_context={"run_id": f"ml4t-bt-{boundary_role}-{uuid4().hex[:12]}"},
    )
    stitched = list(validation.get("walk_forward", {}).get("stitched_prediction_records", []) or [])
    feature_by_id = {str(row["candidate_id"]): row for row in diagnostic.get("_labeled_rows", [])}
    policy_records = _apply_market_state_v1(stitched, feature_by_id)
    planned_trades = _build_planned_trades(policy_records, bars, config.horizon_bars)

    prices_df = _bars_to_polars(bars)
    context_df = _build_context_frame(planned_trades)
    strategy = ScheduledTradeReplayStrategy(BACKTEST_SYMBOL)
    backtest_result = run_backtest(
        prices_df,
        strategy,
        context=context_df,
        config=_backtest_config(),
        contract_specs={
            BACKTEST_SYMBOL: ContractSpec(
                symbol=BACKTEST_SYMBOL,
                asset_class=AssetClass.FUTURE,
                multiplier=MNQ_MULTIPLIER,
                tick_size=_tick_size(),
            )
        },
    )

    translation = build_translation_analysis(stage2, policy_records)
    result_dict = backtest_result.to_dict()
    trade_records = backtest_result.to_trade_records()
    fill_records = _frame_to_records(backtest_result.to_fills_dataframe())
    equity_records = _frame_to_records(backtest_result.to_equity_dataframe())

    run_id = f"ml4t-backtest-{uuid4().hex[:12]}"
    run_dir = REPORTS_DIR / "runs" / run_id / "ml4t_backtest"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "complete",
        "benchmark_name": BENCHMARK_NAME,
        "boundary_role": f"{boundary_role}_walk_forward_oos",
        "source_path": source_path,
        "entry_policy": ENTRY_POLICY_NAME,
        "exit_policy": EXIT_POLICY_NAME,
        "threshold": float(load_bnr_config().get("frozen_benchmark", {}).get("threshold", 0.45) or 0.45),
        "walk_forward": {
            "status": validation.get("walk_forward", {}).get("status"),
            "mean_roc_auc": validation.get("walk_forward", {}).get("mean_roc_auc"),
            "fold_count": validation.get("walk_forward", {}).get("fold_count"),
            "stitched_candidates": len(stitched),
        },
        "gates": {
            "entry_gate": "breakout_quality",
            "scratch_exit": EXIT_POLICY_NAME,
            "holdout_locked": True,
        },
        "planned_trade_stream": {
            "count": len(planned_trades),
            "same_timestamp_entry_collisions": strategy.same_timestamp_entry_collisions,
            "overlap_skips": strategy.skipped_overlap_entries,
            "close_without_position": strategy.close_without_position,
        },
        "backtest": {
            "engine": "ml4t.backtest",
            "execution_mode": "same_bar",
            "execution_price": "close",
            "slippage_type": "fixed",
            "slippage_fixed_points_per_fill": _slippage_points_per_side(),
            "contract_multiplier": MNQ_MULTIPLIER,
            "num_trades": result_dict.get("num_trades"),
            "win_rate": result_dict.get("win_rate"),
            "total_return_pct": result_dict.get("total_return_pct"),
            "max_drawdown_pct": result_dict.get("max_drawdown_pct"),
            "total_gross_pnl": result_dict.get("total_gross_pnl"),
            "total_costs": result_dict.get("total_costs"),
            "sharpe": result_dict.get("sharpe"),
            "sortino": result_dict.get("sortino"),
            "profit_factor": result_dict.get("profit_factor"),
            "avg_trade": result_dict.get("avg_trade"),
            "largest_win": result_dict.get("largest_win"),
            "largest_loss": result_dict.get("largest_loss"),
        },
        "translation": {
            "status": translation.get("status"),
            "binary_utility": translation.get("binary_utility_score"),
            "sized_utility": translation.get("sized_utility_score"),
            "utility_gap": translation.get("sized_vs_binary_gap"),
        },
        "artifacts": {
            "trade_records": str(run_dir / "trade_records.json"),
            "fill_records": str(run_dir / "fill_records.json"),
            "equity_curve": str(run_dir / "equity_curve.json"),
            "planned_trades": str(run_dir / "planned_trades.json"),
        },
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (run_dir / "trade_records.json").write_text(json.dumps(trade_records, indent=2, default=str), encoding="utf-8")
    (run_dir / "fill_records.json").write_text(json.dumps(fill_records, indent=2, default=str), encoding="utf-8")
    (run_dir / "equity_curve.json").write_text(json.dumps(equity_records, indent=2, default=str), encoding="utf-8")
    (run_dir / "planned_trades.json").write_text(json.dumps(planned_trades, indent=2, default=str), encoding="utf-8")

    suffix = "validation" if boundary_role == "validation" else "exploration"
    output_path = REPORTS_DIR / f"market_state_setup_quality_v1_ml4t_backtest_{suffix}.json"
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return MarketStateBacktestBundle(report=summary, output_path=output_path, run_dir=run_dir)


def _apply_market_state_v1(
    stitched_records: list[dict[str, Any]],
    feature_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in stitched_records:
        joined = dict(record)
        joined.update(feature_by_id.get(str(record.get("candidate_id")), {}))
        keep, _, pit_valid = _followthrough_gate_decision(joined, "breakout_quality")
        if keep <= 0 or not pit_valid:
            continue
        filtered.append(joined)
    return filtered


def _build_planned_trades(
    policy_records: list[dict[str, Any]],
    bars: Any,
    horizon_bars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joined in policy_records:
        exit_row = _simulate_trade_exit(joined, bars, horizon_bars, EXIT_POLICY_NAME)
        rows.append(
            {
                "candidate_id": str(joined["candidate_id"]),
                "session_date": str(joined.get("session_date", "")),
                "direction": str(joined["direction"]),
                "entry_time": str(joined["entry_time"]),
                "exit_time": str(exit_row["exit_time"]),
                "probability": float(joined.get("probability", 0.0) or 0.0),
                "label": int(joined.get("label", 0) or 0),
                "market_state": str(joined.get("market_state", "")),
                "setup_quality": str(joined.get("setup_quality", "")),
                "setup_subtype": str(joined.get("setup_subtype", joined.get("subtype", ""))),
                "baseline_pnl_r": float(joined.get("pnl_r", 0.0) or 0.0),
                "scratch_exit_pnl_r": float(exit_row["pnl_r"]),
                "scratch_outcome": str(exit_row["outcome"]),
                "bars_held": int(exit_row["bars_held"]),
            }
        )
    return sorted(rows, key=lambda row: (row["entry_time"], -row["probability"], row["candidate_id"]))


def _bars_to_polars(bars: Any) -> pl.DataFrame:
    frame = bars.reset_index().rename(columns={"ts_event": "timestamp"})
    if "timestamp" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    frame["symbol"] = BACKTEST_SYMBOL
    return pl.from_pandas(frame[["timestamp", "symbol", "open", "high", "low", "close", "volume"]])


def _build_context_frame(planned_trades: list[dict[str, Any]]) -> pl.DataFrame:
    by_ts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in planned_trades:
        entry_bucket = by_ts.setdefault(str(row["entry_time"]), {"entries": [], "exits": []})
        entry_bucket["entries"].append(
            {
                "candidate_id": row["candidate_id"],
                "direction": row["direction"],
                "probability": row["probability"],
                "setup_subtype": row["setup_subtype"],
            }
        )
        exit_bucket = by_ts.setdefault(str(row["exit_time"]), {"entries": [], "exits": []})
        exit_bucket["exits"].append(
            {
                "candidate_id": row["candidate_id"],
                "direction": row["direction"],
                "probability": row["probability"],
                "scratch_outcome": row["scratch_outcome"],
            }
        )
    rows = []
    for timestamp, bucket in sorted(by_ts.items()):
        rows.append(
            {
                "timestamp": _parse_timestamp(timestamp),
                "entry_actions": json.dumps(bucket["entries"]),
                "exit_actions": json.dumps(bucket["exits"]),
            }
        )
    return pl.DataFrame(rows)


def _backtest_config() -> BacktestConfig:
    costs = dict(load_global_config().get("costs", {}) or {})
    commission_per_trade = float(costs.get("commission_per_trade", 0.0) or 0.0)
    return BacktestConfig(
        allow_short_selling=True,
        allow_leverage=True,
        initial_cash=100000.0,
        timezone=str(load_global_config().get("project", {}).get("timezone", "America/New_York")),
        data_frequency=DataFrequency.IRREGULAR,
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.CLOSE,
        slippage_type=SlippageType.FIXED,
        slippage_fixed=_slippage_points_per_side(),
        commission_per_trade=commission_per_trade,
    )


def _slippage_points_per_side() -> float:
    slippage = dict(load_global_config().get("slippage", {}) or {})
    return float(slippage.get("base_ticks_per_side", 3.0) or 3.0) * _tick_size()


def _tick_size() -> float:
    slippage = dict(load_global_config().get("slippage", {}) or {})
    return float(slippage.get("tick_size", 0.25) or 0.25)


def _decode_actions(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _frame_to_records(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if hasattr(frame, "to_dicts"):
        return list(frame.to_dicts())
    if hasattr(frame, "to_dict"):
        try:
            return list(frame.to_dict(orient="records"))
        except TypeError:
            pass
    return []


def _parse_timestamp(value: str) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to parse market-state timestamps") from exc
    return pd.Timestamp(value).to_pydatetime()
