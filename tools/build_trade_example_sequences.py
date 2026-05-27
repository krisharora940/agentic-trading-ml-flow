from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading_ml.stage2_data import load_ohlcv_file, regular_session


TRADES_PATH = Path(
    "/Users/radhikaarora/Documents/Trading ML V4/reports/manual_labels/trades_20260526092550_normalized.csv"
)
BARS_PATH = Path(
    "/Users/radhikaarora/Documents/Trading ML V4/data/cache/mnq_30s_exploration_opening.parquet"
)
OUTPUT_DIR = Path("/Users/radhikaarora/Documents/Trading ML V4/reports/manual_labels")
INDEX_PATH = OUTPUT_DIR / "trades_20260526092550_sequence_index.csv"
SEQUENCES_PATH = OUTPUT_DIR / "trades_20260526092550_sequences.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "trades_20260526092550_sequence_summary.json"

PRE_BARS = 12
POST_BARS = 24


def _parse_trade_timestamp(date_text: str, time_text: str) -> pd.Timestamp:
    clean_time = str(time_text).replace(" EST", "").replace(" EDT", "")
    naive = pd.Timestamp(f"{date_text} {clean_time}")
    return naive.tz_localize("America/New_York")


def _align_to_bar(
    timestamp: pd.Timestamp, session_index: pd.DatetimeIndex
) -> pd.Timestamp | None:
    if session_index.empty:
        return None
    eligible = session_index[session_index <= timestamp]
    if len(eligible) > 0:
        return pd.Timestamp(eligible[-1])
    later = session_index[session_index > timestamp]
    if len(later) > 0:
        return pd.Timestamp(later[0])
    return None


def _quality_label(row: pd.Series) -> str:
    bucket = str(row.get("quality_bucket", "") or "")
    setup_valid = str(row.get("setup_valid_normalized", "") or "")
    if bucket:
        return bucket
    return "avoid" if setup_valid == "no" else "valid"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(TRADES_PATH)
    bars = regular_session(
        load_ohlcv_file(
            BARS_PATH, symbol="MNQ", timeframe="30s", timezone="America/New_York"
        )
    )

    index_rows: list[dict[str, object]] = []
    with SEQUENCES_PATH.open("w", encoding="utf-8") as handle:
        for trade_id, (_, row) in enumerate(trades.iterrows(), start=1):
            open_ts = _parse_trade_timestamp(
                str(row["Open Date"]), str(row["Open Time"])
            )
            close_ts = _parse_trade_timestamp(
                str(row["Open Date"]), str(row["Close Time"])
            )
            session_bars = bars[bars.index.date == open_ts.date()]
            entry_anchor = _align_to_bar(open_ts, session_bars.index)
            exit_anchor = _align_to_bar(close_ts, session_bars.index)
            matched = entry_anchor is not None

            window_rows: list[dict[str, object]] = []
            if matched and entry_anchor is not None:
                anchor_loc = session_bars.index.get_loc(entry_anchor)
                if not isinstance(anchor_loc, int):
                    anchor_loc = int(anchor_loc.start)
                start = max(0, anchor_loc - PRE_BARS)
                stop = min(len(session_bars), anchor_loc + POST_BARS + 1)
                window = session_bars.iloc[start:stop].copy()
                for i, (ts, bar) in enumerate(window.iterrows(), start=start):
                    window_rows.append(
                        {
                            "ts_event": ts.isoformat(),
                            "bar_index": int(i),
                            "bars_from_entry_anchor": int(i - anchor_loc),
                            "open": float(bar["open"]),
                            "high": float(bar["high"]),
                            "low": float(bar["low"]),
                            "close": float(bar["close"]),
                            "volume": float(bar["volume"]),
                            "is_entry_anchor": bool(ts == entry_anchor),
                            "is_exit_anchor": bool(
                                exit_anchor is not None and ts == exit_anchor
                            ),
                        }
                    )

            example_id = f"trade-example-{trade_id:04d}"
            record = {
                "example_id": example_id,
                "open_date": str(row["Open Date"]),
                "open_time": str(row["Open Time"]),
                "close_time": str(row["Close Time"]),
                "side": str(row["Side"]),
                "entry_price": float(row["Entry Price"]),
                "exit_price": float(row["Exit Price"]),
                "net_pnl": float(row["Net P&L"]),
                "duration_seconds": float(row["Duration"]),
                "quantity": float(row["Quantity"]),
                "setup_valid_normalized": str(row.get("setup_valid_normalized", "")),
                "quality_bucket": _quality_label(row),
                "valid_reason_tags": str(row.get("valid_reason_tags", "") or ""),
                "invalid_reason_tags": str(row.get("invalid_reason_tags", "") or ""),
                "primary_valid_reason": str(row.get("primary_valid_reason", "") or ""),
                "primary_invalid_reason": str(
                    row.get("primary_invalid_reason", "") or ""
                ),
                "entry_timestamp_raw": open_ts.isoformat(),
                "exit_timestamp_raw": close_ts.isoformat(),
                "entry_anchor_timestamp": (
                    entry_anchor.isoformat() if entry_anchor is not None else ""
                ),
                "exit_anchor_timestamp": (
                    exit_anchor.isoformat() if exit_anchor is not None else ""
                ),
                "matched_to_bar_data": bool(matched),
                "window_pre_bars": PRE_BARS,
                "window_post_bars": POST_BARS,
                "bars": window_rows,
            }
            handle.write(json.dumps(record) + "\n")
            index_rows.append(
                {
                    "example_id": example_id,
                    "open_date": row["Open Date"],
                    "open_time": row["Open Time"],
                    "close_time": row["Close Time"],
                    "side": row["Side"],
                    "entry_price": row["Entry Price"],
                    "exit_price": row["Exit Price"],
                    "net_pnl": row["Net P&L"],
                    "setup_valid_normalized": row.get("setup_valid_normalized", ""),
                    "quality_bucket": _quality_label(row),
                    "primary_valid_reason": row.get("primary_valid_reason", ""),
                    "primary_invalid_reason": row.get("primary_invalid_reason", ""),
                    "entry_timestamp_raw": open_ts.isoformat(),
                    "entry_anchor_timestamp": (
                        entry_anchor.isoformat() if entry_anchor is not None else ""
                    ),
                    "exit_anchor_timestamp": (
                        exit_anchor.isoformat() if exit_anchor is not None else ""
                    ),
                    "matched_to_bar_data": bool(matched),
                    "bars_in_window": len(window_rows),
                }
            )

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(INDEX_PATH, index=False)
    summary = {
        "trades_path": str(TRADES_PATH),
        "bars_path": str(BARS_PATH),
        "sequence_index_path": str(INDEX_PATH),
        "sequences_path": str(SEQUENCES_PATH),
        "example_count": int(len(index_df)),
        "matched_count": (
            int(index_df["matched_to_bar_data"].sum()) if not index_df.empty else 0
        ),
        "unmatched_count": (
            int((~index_df["matched_to_bar_data"]).sum()) if not index_df.empty else 0
        ),
        "quality_bucket_counts": (
            index_df["quality_bucket"].value_counts(dropna=False).to_dict()
            if not index_df.empty
            else {}
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
