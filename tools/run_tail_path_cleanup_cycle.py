from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_ml.agent_workflow import build_agent_loop_state  # noqa: E402
from trading_ml.diagnostic_adapter import prepare_diagnostic_runtime  # noqa: E402
from trading_ml.research_controller import run_governed_research_cycle  # noqa: E402


def main() -> None:
    prepare_diagnostic_runtime()
    os.environ.setdefault("TRADING_ML_DISABLE_SHAP", "1")
    state = build_agent_loop_state()
    payload = run_governed_research_cycle(
        dict(state["stage2_config"]),
        family="tail_path_cleanup",
        controller_override=state.get("controller_state", {}),
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
