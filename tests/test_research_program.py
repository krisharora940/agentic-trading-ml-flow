import unittest

from trading_ml.research_program import build_program_state, evaluate_program_state


class ResearchProgramTests(unittest.TestCase):
    def test_program_director_prefers_robustness_rebuild_when_cpcv_fails(self) -> None:
        state = {
            "program_state": build_program_state(),
            "stage2_config": {"source_path": "data/cache/exploration.parquet", "feature_family": "bnr_plus_context"},
            "bnr_spec": {
                "frozen_benchmark": {
                    "feature_family": "bnr_plus_context",
                    "model_family": "linear_baseline",
                    "threshold": 0.45,
                    "policy_gate": "balanced_deep_eff_ge_0.20",
                    "policy_meta": "balanced_clean_post_reclaim_ge_0.30",
                }
            },
            "data_manifest_loaded": True,
            "stage2_result": {
                "model_diagnostics": {
                    "shap_analysis": {
                        "top_features": [
                            {"feature": "reclaim_body_strength"},
                            {"feature": "post_reclaim_close_strength"},
                        ]
                    }
                }
            },
            "audit_summary": {
                "walk_forward": {"status": "pass"},
                "cpcv": {"status": "fail"},
                "purging": {"status": "pass"},
                "multiple_testing": {"status": "pass", "promotable_method": False},
                "random_signal_plumbing": "pass",
            },
            "translation_summary": {"status": "pass"},
            "controller_state": {"active_family": "model"},
        }
        program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertEqual(plan["lane"], "robustness_rebuild")
        self.assertEqual(plan["policy_family"], "policy_meta")

    def test_program_director_chooses_translation_policy_when_translation_is_weak(self) -> None:
        state = {
            "program_state": build_program_state(),
            "stage2_config": {"source_path": "data/cache/exploration.parquet", "feature_family": "bnr_plus_context"},
            "bnr_spec": {
                "frozen_benchmark": {
                    "feature_family": "bnr_plus_context",
                    "model_family": "linear_baseline",
                    "threshold": 0.45,
                    "sizing_policy": "confidence_linear_v1",
                    "regime_throttle_policy": "high_vol_or_non_trending_off_v1",
                    "regime_size_policy": "trend_vol_scale_v1",
                }
            },
            "audit_summary": {
                "walk_forward": {"status": "pass"},
                "cpcv": {"status": "pass"},
                "purging": {"status": "pass"},
                "multiple_testing": {"status": "pass", "promotable_method": False},
                "random_signal_plumbing": "pass",
            },
            "translation_summary": {"status": "fail"},
            "controller_state": {"active_family": "model"},
        }
        program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertEqual(plan["lane"], "translation_lab")
        self.assertEqual(plan["controller_override"]["active_family"], "translation_policy")


if __name__ == "__main__":
    unittest.main()
