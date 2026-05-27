from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_bnr import BNRZone, CandidateSetup


LEDGER_SCHEMA_VERSION = 1


def candidate_event_ledger_dir() -> Path:
    root = REPORTS_DIR / "candidate_event_ledger"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_candidate_event_ledger_key(
    *,
    source_path: str,
    symbol: str,
    timeframe: str,
    variant_names: list[str],
    session_dates: list[str],
    generator_version: str,
) -> str:
    payload = {
        "source_path": str(source_path),
        "symbol": str(symbol),
        "timeframe": str(timeframe),
        "variant_names": list(sorted(variant_names)),
        "session_dates": list(session_dates),
        "generator_version": str(generator_version),
        "schema_version": LEDGER_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def candidate_event_ledger_path(ledger_key: str) -> Path:
    return candidate_event_ledger_dir() / f"{ledger_key}.json"


def load_candidate_event_ledger(ledger_key: str) -> dict[str, Any] | None:
    path = candidate_event_ledger_path(ledger_key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_candidate_event_ledger(payload: dict[str, Any]) -> str:
    ledger_key = str(payload["ledger_key"])
    path = candidate_event_ledger_path(ledger_key)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def candidate_to_ledger_record(
    candidate: CandidateSetup, *, variant_name: str
) -> dict[str, Any]:
    record = candidate.to_dict()
    record["ledger_variant"] = str(variant_name)
    return record


def candidate_from_ledger_record(record: dict[str, Any]) -> CandidateSetup:
    zone = BNRZone(**dict(record["zone"]))
    payload = dict(record)
    payload.pop("ledger_variant", None)
    payload["zone"] = zone
    return CandidateSetup(**payload)
