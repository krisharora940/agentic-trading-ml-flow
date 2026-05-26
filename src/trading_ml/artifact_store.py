from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trading_ml.paths import REPORTS_DIR
from trading_ml.schemas import utc_now_iso


def persist_node_artifact(
    *,
    run_id: str,
    node_name: str,
    cycle: int,
    phase: str,
    state: dict[str, Any],
    payload: dict[str, Any],
) -> Path:
    root = REPORTS_DIR / "runs" / run_id / "node_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    created_at = utc_now_iso()
    input_hash = _stable_hash(
        {
            "stage2_config": state.get("stage2_config", {}),
            "controller_state": state.get("controller_state", {}),
            "phase": phase,
            "research_cycle": cycle,
        }
    )
    output_hash = _stable_hash(payload)
    record = {
        "run_id": run_id,
        "node_name": node_name,
        "cycle": cycle,
        "phase": phase,
        "created_at": created_at,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "git_commit": _git_commit(),
        "config_hash": _stable_hash(state.get("bnr_spec", {})),
        "data_manifest_hash": _stable_hash(state.get("data_manifest", {})),
        "payload": payload,
    }
    path = root / f"{cycle:03d}_{node_name}_{created_at.replace(':', '-')}.json"
    path.write_text(
        json.dumps(record, sort_keys=True, indent=2, default=str), encoding="utf-8"
    )
    return path


def _stable_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _git_commit() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip()
        return commit or "unknown"
    except Exception:
        return "unknown"
