import unittest
from unittest.mock import patch

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.search import generate_search_trials, run_governed_search


class SearchTests(unittest.TestCase):
    def test_generate_search_trials_is_bounded(self) -> None:
        state = build_agent_loop_state()
        trials = generate_search_trials(
            state["stage2_config"],
            family="setup",
            controller_override={"active_family": "setup"},
        )
        self.assertGreater(len(trials), 0)
        self.assertLessEqual(len(trials), 6)

    def test_generate_model_family_trials_is_bounded(self) -> None:
        state = build_agent_loop_state()
        trials = generate_search_trials(
            state["stage2_config"],
            family="model",
            controller_override={"active_family": "model"},
        )
        self.assertEqual(len(trials), 2)
        self.assertEqual(
            {trial["model_family"] for trial in trials}, {"linear_baseline", "gbm"}
        )

    def test_generate_feature_family_trials_is_bounded(self) -> None:
        state = build_agent_loop_state()
        trials = generate_search_trials(
            state["stage2_config"],
            family="feature",
            controller_override={"active_family": "feature"},
        )
        self.assertEqual(len(trials), 3)
        self.assertIn("feature_family", trials[0])

    def test_generate_feature_family_trials_can_focus_on_bnr_state_slice(self) -> None:
        state = build_agent_loop_state()
        trials = generate_search_trials(
            state["stage2_config"],
            family="feature",
            controller_override={
                "active_family": "feature",
                "focus_setup_state": "repair",
                "focus_environment_state": "volatile_chop",
                "focus_path_class": "failure",
            },
        )
        self.assertEqual(len(trials), 3)
        self.assertEqual(trials[0]["feature_family"], "context_plus_regime")
        self.assertEqual(trials[0]["focus_setup_state"], "repair")
        self.assertEqual(trials[0]["focus_environment_state"], "volatile_chop")
        self.assertEqual(trials[0]["focus_path_class"], "failure")

    def test_generate_threshold_trials_is_bounded(self) -> None:
        state = build_agent_loop_state()
        trials = generate_search_trials(
            state["stage2_config"],
            family="threshold",
            controller_override={"active_family": "threshold"},
        )
        self.assertEqual(len(trials), 5)
        self.assertEqual(
            {trial["decision_threshold"] for trial in trials},
            {0.45, 0.5, 0.55, 0.6, 0.65},
        )

    def test_generate_feature_threshold_trials_is_bounded(self) -> None:
        state = build_agent_loop_state()
        trials = generate_search_trials(
            state["stage2_config"],
            family="feature_threshold",
            controller_override={"active_family": "feature_threshold"},
        )
        self.assertEqual(len(trials), 15)
        self.assertIn("feature_family", trials[0])
        self.assertIn("decision_threshold", trials[0])

    def test_generate_label_trials_is_bounded(self) -> None:
        state = build_agent_loop_state()
        trials = generate_search_trials(
            state["stage2_config"],
            family="label",
            controller_override={"active_family": "label"},
        )
        self.assertEqual(len(trials), 8)
        self.assertIn("horizon_bars", trials[0])
        self.assertIn("stop_multiple", trials[0])
        self.assertIn("target_multiple", trials[0])

    def test_run_governed_search_returns_ranked_trials(self) -> None:
        state = build_agent_loop_state()
        with patch("trading_ml.research_controller.append_experiment_record"):
            result = run_governed_search(state["stage2_config"])
        self.assertEqual(result["trial_count"], len(result["ranked_trials"]))
        self.assertIsNotNone(result["best_trial"])
        self.assertIn("net_avg_pnl_r", result["best_trial"])
        self.assertEqual(result["family"], "model")
        self.assertIn("spec_version", result)
        self.assertIn("baseline", result)
        self.assertIn(result["batch_decision"], {"accept", "revise"})


if __name__ == "__main__":
    unittest.main()
