import unittest
from unittest import mock

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
        with (
            mock.patch("trading_ml.research_program._load_cpcv_failure_attribution", return_value={}),
            mock.patch(
                "trading_ml.research_program._build_multi_cycle_memory",
                return_value={
                    "failure_counts": {},
                    "accepted_families": [],
                    "recent_rejected_families": [],
                    "killed_families": [],
                    "exhausted_families": [],
                    "persistent_tail_failure": {"status": "inactive"},
                },
            ),
        ):
            program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertEqual(plan["lane"], "robustness_rebuild")
        self.assertEqual(plan["selected_family"], "policy_meta")

    def test_program_director_prefers_subtype_when_cpcv_attribution_points_to_structural_failure(self) -> None:
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
            "stage2_result": {"model_diagnostics": {"shap_analysis": {"top_features": []}}},
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
        attribution = {
            "status": "complete",
            "dominant_failure_axes": {
                "subtype": {"key": "deep_retrace_repair", "trade_count": 23, "total_pnl_r": -23.0},
                "time_of_day": {"key": "14:32", "trade_count": 12, "total_pnl_r": -10.0},
                "probability_bucket": {"key": "[0.65,1.00]", "trade_count": 29, "total_pnl_r": -25.0},
            },
            "failure_summary": {"failure_type": "tail_path_fragility"},
        }
        with (
            mock.patch("trading_ml.research_program._load_cpcv_failure_attribution", return_value=attribution),
            mock.patch(
                "trading_ml.research_program._build_multi_cycle_memory",
                return_value={
                    "failure_counts": {},
                    "accepted_families": [],
                    "recent_rejected_families": [],
                    "killed_families": [],
                    "exhausted_families": [],
                },
            ),
        ):
            program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertEqual(plan["selected_family"], "subtype")
        self.assertTrue(plan["rejected_alternatives"])
        self.assertIn("family_scores", plan)
        self.assertIn("falsification_rule", plan)
        self.assertEqual(plan["approval_required"], "search_space_approval")
        self.assertIn("setup_subtype", plan["search_budget"]["allowed_knobs"])
        self.assertIn("threshold", plan["families_rejected"])
        self.assertEqual(plan["evidence_used"]["current_blocker"], "cpcv_tail_path_fragility")

    def test_program_director_blocks_validation_window_and_skips_exhausted_families_while_cpcv_fails(self) -> None:
        state = {
            "program_state": build_program_state(),
            "stage2_config": {"source_path": "data/cache/exploration.parquet", "feature_family": "bnr_plus_context"},
            "bnr_spec": {
                "frozen_benchmark": {
                    "feature_family": "bnr_plus_context",
                    "model_family": "linear_baseline",
                    "threshold": 0.45,
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
        attribution = {
            "status": "complete",
            "dominant_failure_axes": {
                "subtype": {"key": "deep_retrace_repair", "trade_count": 23, "total_pnl_r": -23.0},
                "probability_bucket": {"key": "[0.65,1.00]", "trade_count": 29, "total_pnl_r": -25.0},
            },
            "failure_summary": {"failure_type": "tail_path_fragility"},
        }
        with (
            mock.patch("trading_ml.research_program._load_cpcv_failure_attribution", return_value=attribution),
            mock.patch(
                "trading_ml.research_program._build_multi_cycle_memory",
                return_value={
                    "failure_counts": {"subtype": 3, "model": 4},
                    "accepted_families": [],
                    "recent_rejected_families": ["model", "subtype"],
                    "killed_families": [],
                    "exhausted_families": ["model", "subtype"],
                },
            ),
        ):
            program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertEqual(plan["selected_family"], "policy_meta")
        self.assertNotIn("policy_meta", [row["family"] for row in plan["rejected_alternatives"]])
        self.assertIn("validation_window", plan["families_rejected"])
        validation_score = next(row for row in plan["family_scores"] if row["family"] == "validation_window")
        self.assertLess(validation_score["risk_adjusted_score"], 0)

    def test_program_director_routes_repeated_tail_paths_to_tail_cleanup(self) -> None:
        state = {
            "program_state": build_program_state(),
            "stage2_config": {"source_path": "data/cache/exploration.parquet", "feature_family": "bnr_plus_context"},
            "bnr_spec": {"frozen_benchmark": {"feature_family": "bnr_plus_context", "model_family": "linear_baseline", "threshold": 0.45}},
            "data_manifest_loaded": True,
            "stage2_result": {"model_diagnostics": {"shap_analysis": {"top_features": []}}},
            "audit_summary": {
                "walk_forward": {"status": "pass"},
                "cpcv": {"status": "fail"},
                "purging": {"status": "pass"},
                "multiple_testing": {"status": "pass", "promotable_method": False},
                "random_signal_plumbing": "pass",
            },
            "translation_summary": {"status": "pass"},
            "controller_state": {"active_family": "policy_meta"},
        }
        attribution = {
            "status": "complete",
            "dominant_failure_axes": {
                "subtype": {"key": "deep_retrace_repair", "trade_count": 23, "total_pnl_r": -23.0},
                "probability_bucket": {"key": "[0.65,1.00]", "trade_count": 29, "total_pnl_r": -25.0},
            },
            "failure_summary": {"failure_type": "tail_path_fragility"},
        }
        with (
            mock.patch("trading_ml.research_program._load_cpcv_failure_attribution", return_value=attribution),
            mock.patch(
                "trading_ml.research_program._build_multi_cycle_memory",
                return_value={
                    "failure_counts": {"model": 2, "policy_meta": 1},
                    "accepted_families": [],
                    "recent_rejected_families": ["model", "policy_meta"],
                    "killed_families": [],
                    "exhausted_families": [],
                    "persistent_tail_failure": {
                        "status": "active",
                        "path_ids": ["cpcv_010", "cpcv_003", "cpcv_002"],
                        "families": ["model", "policy_meta"],
                    },
                },
            ),
        ):
            program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertEqual(plan["selected_family"], "tail_path_cleanup")
        self.assertEqual(plan["approval_required"], "search_space_approval")
        self.assertIn("exact failed CPCV paths", " ".join(plan["known_risks"]))

    def test_program_director_moves_to_label_after_tail_cleanup_fails(self) -> None:
        state = {
            "program_state": build_program_state(),
            "stage2_config": {"source_path": "data/cache/exploration.parquet", "feature_family": "bnr_plus_context"},
            "bnr_spec": {
                "frozen_benchmark": {
                    "feature_family": "bnr_plus_context",
                    "model_family": "linear_baseline",
                    "threshold": 0.45,
                }
            },
            "data_manifest_loaded": True,
            "stage2_result": {"model_diagnostics": {"shap_analysis": {"top_features": [{"feature": "reclaim_body_strength"}]}}},
            "audit_summary": {
                "walk_forward": {"status": "pass"},
                "cpcv": {"status": "fail"},
                "purging": {"status": "pass"},
                "multiple_testing": {"status": "pass", "promotable_method": False},
                "random_signal_plumbing": "pass",
            },
            "translation_summary": {"status": "pass"},
            "controller_state": {"active_family": "tail_path_cleanup"},
        }
        attribution = {
            "status": "complete",
            "dominant_failure_axes": {
                "probability_bucket": {"key": "[0.65,1.00]", "trade_count": 29, "total_pnl_r": -25.0},
            },
            "failure_summary": {"failure_type": "tail_path_fragility"},
        }
        with (
            mock.patch("trading_ml.research_program._load_cpcv_failure_attribution", return_value=attribution),
            mock.patch(
                "trading_ml.research_program._build_multi_cycle_memory",
                return_value={
                    "failure_counts": {"label": 4, "tail_path_cleanup": 1},
                    "accepted_families": [],
                    "recent_rejected_families": ["tail_path_cleanup"],
                    "killed_families": [],
                    "exhausted_families": ["label"],
                    "persistent_tail_failure": {
                        "status": "active",
                        "path_ids": ["cpcv_010", "cpcv_003", "cpcv_002"],
                        "families": ["model", "policy_meta", "tail_path_cleanup"],
                        "tail_cleanup_failed": True,
                    },
                },
            ),
        ):
            program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertEqual(plan["selected_family"], "label")
        self.assertIn("Label", plan["family_scores"][0]["family"].title())

    def test_program_director_moves_to_sample_expansion_after_label_geometry_fails(self) -> None:
        state = {
            "program_state": build_program_state(),
            "stage2_config": {"source_path": "data/cache/exploration.parquet", "feature_family": "bnr_plus_context"},
            "bnr_spec": {"frozen_benchmark": {"feature_family": "bnr_plus_context", "model_family": "linear_baseline", "threshold": 0.45}},
            "data_manifest_loaded": True,
            "stage2_result": {"model_diagnostics": {"shap_analysis": {"top_features": []}}},
            "audit_summary": {
                "walk_forward": {"status": "pass"},
                "cpcv": {"status": "fail"},
                "purging": {"status": "pass"},
                "multiple_testing": {"status": "fail", "promotable_method": False},
                "random_signal_plumbing": "pass",
            },
            "translation_summary": {"status": "pass"},
            "controller_state": {"active_family": "label"},
        }
        with (
            mock.patch("trading_ml.research_program._load_cpcv_failure_attribution", return_value={"failure_summary": {"failure_type": "tail_path_fragility"}}),
            mock.patch(
                "trading_ml.research_program._build_multi_cycle_memory",
                return_value={
                    "failure_counts": {"label": 1, "tail_path_cleanup": 1},
                    "accepted_families": [],
                    "recent_rejected_families": ["label", "tail_path_cleanup"],
                    "killed_families": [],
                    "exhausted_families": [],
                    "persistent_tail_failure": {
                        "status": "active",
                        "path_ids": ["cpcv_010", "cpcv_003", "cpcv_002"],
                        "families": ["model", "policy_meta", "tail_path_cleanup", "label"],
                        "tail_cleanup_failed": True,
                        "label_geometry_failed": True,
                    },
                },
            ),
        ):
            program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertEqual(plan["selected_family"], "sample_expansion")
        self.assertIn("latest_trigger_time", plan["search_budget"]["allowed_knobs"])

    def test_program_director_parks_benchmark_when_tail_persists_across_three_families(self) -> None:
        state = {
            "program_state": build_program_state(),
            "stage2_config": {"source_path": "data/cache/exploration.parquet", "feature_family": "bnr_plus_context"},
            "bnr_spec": {"frozen_benchmark": {"feature_family": "bnr_plus_context", "model_family": "linear_baseline", "threshold": 0.45}},
            "data_manifest_loaded": True,
            "stage2_result": {"model_diagnostics": {"shap_analysis": {"top_features": []}}},
            "audit_summary": {
                "walk_forward": {"status": "pass"},
                "cpcv": {"status": "fail"},
                "purging": {"status": "pass"},
                "multiple_testing": {"status": "fail", "promotable_method": False},
                "random_signal_plumbing": "pass",
            },
            "translation_summary": {"status": "pass"},
            "controller_state": {"active_family": "sample_expansion"},
        }
        with (
            mock.patch("trading_ml.research_program._load_cpcv_failure_attribution", return_value={"failure_summary": {"failure_type": "tail_path_fragility"}}),
            mock.patch(
                "trading_ml.research_program._build_multi_cycle_memory",
                return_value={
                    "failure_counts": {},
                    "accepted_families": [],
                    "recent_rejected_families": ["model", "label", "sample_expansion"],
                    "killed_families": [],
                    "exhausted_families": [],
                    "persistent_tail_failure": {
                        "status": "active",
                        "path_ids": ["cpcv_010", "cpcv_003", "cpcv_002"],
                        "families": ["model", "label", "sample_expansion"],
                        "tail_cleanup_failed": True,
                        "label_geometry_failed": True,
                        "sample_expansion_failed": True,
                        "dsr_failed_after_real_trials": True,
                        "any_hard_gate_passed": False,
                    },
                },
            ),
        ):
            program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertEqual(plan["benchmark_status"], "exhausted_or_structurally_fragile")
        self.assertEqual(plan["action"], "park_bnr_benchmark_definition")
        self.assertEqual(plan["search_budget"]["max_trials"], 0)

    def test_program_director_logs_rejected_alternatives_and_kill_criteria(self) -> None:
        state = {
            "program_state": build_program_state(),
            "stage2_config": {"source_path": "data/cache/exploration.parquet", "feature_family": "bnr_plus_context"},
            "bnr_spec": {
                "frozen_benchmark": {
                    "feature_family": "bnr_plus_context",
                    "model_family": "linear_baseline",
                    "threshold": 0.45,
                }
            },
            "data_manifest_loaded": True,
            "stage2_result": {
                "model_diagnostics": {
                    "shap_analysis": {
                        "top_features": [
                            {"feature": "break_efficiency_ratio"},
                            {"feature": "first_break_close_excess_points"},
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
        attribution = {
            "status": "complete",
            "dominant_failure_axes": {
                "subtype": {"key": "clean_break_continuation", "trade_count": 4, "total_pnl_r": -4.0},
                "time_of_day": {"key": "14:32", "trade_count": 12, "total_pnl_r": -10.0},
                "probability_bucket": {"key": "[0.65,1.00]", "trade_count": 29, "total_pnl_r": -25.0},
            },
            "failure_summary": {"failure_type": "tail_path_fragility"},
        }
        with mock.patch("trading_ml.research_program._load_cpcv_failure_attribution", return_value=attribution):
            program = evaluate_program_state(state)
        plan = program["next_step_plan"]
        self.assertIn("why_selected", plan)
        self.assertIn("why_rejected", plan)
        self.assertIn("kill_criteria", plan)
        self.assertIn("evidence_not_used", plan)
        self.assertIn("multi_cycle_memory", plan)
        self.assertIn("threshold", plan["why_rejected"])

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
