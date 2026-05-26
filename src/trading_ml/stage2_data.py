from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(slots=True)
class DataQualityReport:
    source_path: str
    symbol: str
    timeframe: str
    timezone: str
    rows: int
    date_start: str | None
    date_end: str | None
    sessions: int
    earliest_session_end: str | None
    latest_session_end: str | None
    missing_regular_session_bars: int
    duplicate_timestamps: int
    opening_zone_sessions: int
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Stage 2 requires pandas. Install with `python3 -m pip install pandas pyarrow`."
        ) from exc
    return pd


def _infer_timestamp_column(df: Any) -> str | None:
    candidates = ["ts_event", "timestamp", "datetime", "time", "date"]
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _normalize_columns(df: Any) -> Any:
    renamed = {column: str(column).strip().lower() for column in df.columns}
    df = df.rename(columns=renamed)
    missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return df


def load_ohlcv_file(
    source_path: str | Path,
    *,
    symbol: str = "MNQ",
    timeframe: str = "30s",
    timezone: str = "America/New_York",
    source_timezone: str | None = None,
) -> Any:
    """Load OHLCV data into a canonical timezone-aware DatetimeIndex named ts_event."""
    pd = require_pandas()
    path = Path(source_path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".numbers":
        df = _read_numbers(path)
    else:
        raise ValueError(f"Unsupported OHLCV file format: {suffix}")

    if not isinstance(df.index, pd.DatetimeIndex):
        timestamp_column = _infer_timestamp_column(df)
        if timestamp_column is None:
            raise ValueError(
                "No timestamp column found and index is not datetime-like."
            )
        df[timestamp_column] = _parse_timestamps(
            df[timestamp_column], timezone, source_timezone
        )
        df = df.set_index(timestamp_column)
    else:
        df.index = pd.to_datetime(df.index, errors="raise")

    df.index.name = "ts_event"
    if df.index.tz is None:
        tz = ZoneInfo(source_timezone or timezone)
        df.index = df.index.tz_localize(tz)
    df.index = df.index.tz_convert(ZoneInfo(timezone))

    df = _normalize_columns(df)
    df = df.loc[:, list(OHLCV_COLUMNS)].sort_index()
    df["symbol"] = symbol
    df["source_timeframe"] = timeframe
    df["ts_event_utc"] = df.index.tz_convert("UTC")
    return df


def _parse_timestamps(series: Any, timezone: str, source_timezone: str | None) -> Any:
    pd = require_pandas()
    as_text = series.dropna().astype(str)
    has_explicit_offset = bool(
        as_text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).any()
    )
    if has_explicit_offset:
        return pd.to_datetime(series, errors="raise", utc=True).dt.tz_convert(timezone)
    parsed = pd.to_datetime(series, errors="raise")
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize(source_timezone or timezone)
    return parsed.dt.tz_convert(timezone)


def _read_numbers(path: Path) -> Any:
    pd = require_pandas()
    try:
        from numbers_parser import Document
    except ImportError as exc:
        raise RuntimeError("Reading .numbers files requires `numbers-parser`.") from exc

    document = Document(str(path))
    for sheet in document.sheets:
        for table in sheet.tables:
            rows = table.rows(values_only=True)
            if not rows:
                continue
            header = [
                str(value).strip().lower() if value is not None else ""
                for value in rows[0]
            ]
            if {"open", "high", "low", "close", "volume"}.issubset(set(header)):
                return pd.DataFrame(rows[1:], columns=header)
    raise ValueError(f"No OHLCV-like table found in Numbers file: {path}")


def regular_session(
    df: Any, *, session_start: str = "09:30:00", session_end: str = "16:00:00"
) -> Any:
    return df.between_time(session_start, session_end, inclusive="left")


def build_data_quality_report(
    df: Any,
    *,
    source_path: str | Path,
    symbol: str = "MNQ",
    timeframe: str = "30s",
    timezone: str = "America/New_York",
    session_start: str = "09:30:00",
    session_end: str = "16:00:00",
) -> DataQualityReport:
    pd = require_pandas()
    flags: list[str] = []
    if df.empty:
        flags.append("empty_dataset")
        return DataQualityReport(
            str(source_path),
            symbol,
            timeframe,
            timezone,
            0,
            None,
            None,
            0,
            None,
            None,
            0,
            0,
            0,
            flags,
        )

    duplicates = int(df.index.duplicated().sum())
    if duplicates:
        flags.append("duplicate_timestamps")

    rth = regular_session(df, session_start=session_start, session_end=session_end)
    session_dates = sorted({idx.date() for idx in rth.index})
    expected_freq = "30s" if timeframe == "30s" else "1min"
    missing = 0
    opening_zone_sessions = 0
    session_ends: list[Any] = []
    for session_date in session_dates:
        session_df = rth[rth.index.date == session_date]
        if session_df.empty:
            continue
        session_ends.append(session_df.index.max())
        start = pd.Timestamp.combine(
            session_date, pd.Timestamp(session_start).time()
        ).tz_localize(timezone)
        end = pd.Timestamp.combine(
            session_date, pd.Timestamp(session_end).time()
        ).tz_localize(timezone)
        expected = pd.date_range(
            start=start, end=end, freq=expected_freq, inclusive="left"
        )
        missing += len(expected.difference(session_df.index))
        zone = session_df.between_time("09:30:00", "09:30:59", inclusive="both")
        if len(zone) >= (2 if timeframe == "30s" else 1):
            opening_zone_sessions += 1

    if missing:
        flags.append("missing_regular_session_bars")
    if opening_zone_sessions < len(session_dates):
        flags.append("missing_opening_zone_sessions")

    return DataQualityReport(
        source_path=str(source_path),
        symbol=symbol,
        timeframe=timeframe,
        timezone=timezone,
        rows=int(len(df)),
        date_start=df.index.min().isoformat(),
        date_end=df.index.max().isoformat(),
        sessions=len(session_dates),
        earliest_session_end=min(session_ends).isoformat() if session_ends else None,
        latest_session_end=max(session_ends).isoformat() if session_ends else None,
        missing_regular_session_bars=int(missing),
        duplicate_timestamps=duplicates,
        opening_zone_sessions=opening_zone_sessions,
        quality_flags=flags,
    )
