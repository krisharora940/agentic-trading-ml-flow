#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

root = Path("reports/runs")
runs = sorted((p for p in root.glob("bnr-*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)

if not runs:
    print("No BNR runs found.")
    raise SystemExit(0)

latest = runs[0]
artifacts = sorted((latest / "node_artifacts").glob("*.json"))

print(f"latest_run: {latest.name}")

if not artifacts:
    print("latest_node: none")
    raise SystemExit(0)

latest_node = artifacts[-1]
print(f"latest_node: {latest_node.name}")

def load_payload(pattern: str) -> dict:
    matches = sorted((latest / "node_artifacts").glob(pattern))
    if not matches:
        return {}
    data = json.loads(matches[-1].read_text())
    return dict(data.get("payload", {}) or {})

program = load_payload("*program_director*.json")
promotion = load_payload("*promotion_decision*.json")
audit = load_payload("*audit_agent*.json")
desk = load_payload("*desk_director*.json")

if desk:
    print(f"desk_selected_node: {desk.get('selected_node')}")

if program:
    next_step_plan = dict(program.get("next_step_plan", {}) or {})
    print(f"governor_selected_family: {next_step_plan.get('selected_family')}")
    print(f"governor_assigned_action: {next_step_plan.get('assigned_research_action')}")
    print(f"governor_hypothesis_id: {next_step_plan.get('hypothesis_id')}")

if promotion:
    print(f"promotion_decision: {promotion.get('decision')}")
    gate = dict(promotion.get("promotion_gate", {}) or {})
    if gate:
        print("promotion_gate:")
        for key in [
            "walk_forward_status",
            "cpcv_status",
            "deflated_sharpe_status",
            "multiple_testing_status",
            "multiple_testing_promotable_method",
            "calibration_status",
            "random_signal_plumbing_status",
            "translation_status",
            "purging_status",
        ]:
            if key in gate:
                print(f"  {key}: {gate.get(key)}")

if audit:
    budget = dict(audit.get("budget_usage", {}) or {})
    if budget:
        print("budget_usage:")
        for key in ["runtime_seconds", "trials", "model_trains", "full_validations", "cpcv_runs"]:
            if key in budget:
                print(f"  {key}: {budget.get(key)}")

print(f"artifacts_dir: {latest / 'node_artifacts'}")
PY
