import json
from tempfile import TemporaryDirectory
import unittest

from trading_ml.logging_utils import JsonlRunLogger, RunLogEvent


class LoggingTests(unittest.TestCase):
    def test_jsonl_logger_writes_one_event(self) -> None:
        with TemporaryDirectory() as tmpdir:
            logger = JsonlRunLogger(run_id="test-run", log_dir=tmpdir)
            path = logger.log(
                RunLogEvent(
                    event_type="agent_action",
                    actor="governor",
                    message="Created initial state",
                    payload={"phase": "foundation"},
                )
            )

            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)

            payload = json.loads(lines[0])
            self.assertEqual(payload["actor"], "governor")
            self.assertEqual(payload["payload"]["phase"], "foundation")


if __name__ == "__main__":
    unittest.main()
