import unittest

from trading_ml.research_os import (
    append_failure_memory,
    build_curated_domain_priors,
    build_hypotheses_from_priors,
    build_research_backlog,
    build_research_director_plan,
    build_research_director_summary,
    count_viable_hypotheses,
)


class ResearchOSTests(unittest.TestCase):
    def test_domain_research_priors_start_with_ml4trading(self) -> None:
        priors = build_curated_domain_priors()
        self.assertTrue(priors)
        self.assertEqual(priors[0]["source"], "ml4trading.io")
        self.assertIn("measurable_translation", priors[0])

    def test_backlog_prioritizes_structural_work_after_cpcv_tail_failure(self) -> None:
        hypotheses = build_hypotheses_from_priors(build_curated_domain_priors())
        failure_memory = [
            {
                "family": "policy_meta",
                "failure_type": "cpcv_tail_path_fragility",
                "status": "freeze",
            }
        ]
        backlog = build_research_backlog(hypotheses, failure_memory, stage2_result={"data_quality": {"sessions": 180}})
        top_families = [row["family"] for row in backlog[:3]]
        self.assertIn("candidate_universe_expansion", top_families)
        model_row = next(row for row in backlog if row["family"] == "model") if any(row["family"] == "model" for row in backlog) else None
        self.assertTrue(model_row is None or "cpcv_tail_path_fragility" in model_row["blocked_by"])

    def test_research_director_plan_attaches_hypothesis_to_existing_family_choice(self) -> None:
        priors = build_curated_domain_priors()
        hypotheses = build_hypotheses_from_priors(priors)
        backlog = build_research_backlog(hypotheses, [], stage2_result={"data_quality": {"sessions": 200}})
        summary = build_research_director_summary({"domain_priors": priors, "research_backlog": backlog, "failure_memory": []})
        plan = build_research_director_plan(
            {
                "domain_priors": priors,
                "research_backlog": backlog,
                "research_director_summary": summary,
                "failure_memory": [],
                "blocking_issues": [],
            },
            {
                "selected_family": "candidate_universe_expansion",
                "controller_override": {"active_family": "candidate_universe_expansion"},
            },
        )
        self.assertEqual(plan["next_action"], "run_family_experiment")
        self.assertEqual(plan["hypothesis_id"], next(row["hypothesis_id"] for row in backlog if row["family"] == "candidate_universe_expansion"))
        self.assertEqual(plan["research_director"]["active_hypothesis"]["family"], "candidate_universe_expansion")

    def test_append_failure_memory_deduplicates_same_failure_signature(self) -> None:
        state = {
            "audit_summary": {"cpcv": {"status": "fail"}, "walk_forward": {"status": "pass"}},
            "translation_summary": {"status": "pass"},
            "promotion_decision": "freeze",
            "next_step_plan": {"selected_family": "subtype", "reason": "tail issue"},
            "active_hypothesis": {"hypothesis_id": "H-00007", "family": "subtype"},
            "failure_memory": [],
            "blocking_issues": [],
        }
        first = append_failure_memory(state)
        second = append_failure_memory({**state, "failure_memory": first})
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0]["failure_type"], "cpcv_tail_path_fragility")

    def test_backlog_blocks_exact_failed_hypothesis_and_keeps_other_paths_viable(self) -> None:
        hypotheses = build_hypotheses_from_priors(build_curated_domain_priors())
        backlog = build_research_backlog(
            hypotheses,
            [{"family": "setup", "hypothesis_id": "H-00001", "failure_type": "cpcv_tail_path_fragility", "status": "freeze"}],
            stage2_result={"data_quality": {"sessions": 200}},
        )
        failed = next(row for row in backlog if row["hypothesis_id"] == "H-00001")
        self.assertIn("prior_failed_hypothesis", failed["blocked_by"])
        self.assertGreater(count_viable_hypotheses(backlog), 0)
        self.assertNotEqual(backlog[0]["hypothesis_id"], "H-00001")

    def test_research_director_assigns_cpcv_attribution_after_cpcv_failure(self) -> None:
        priors = build_curated_domain_priors()
        hypotheses = build_hypotheses_from_priors(priors)
        backlog = build_research_backlog(hypotheses, [], stage2_result={"data_quality": {"sessions": 200}})
        summary = build_research_director_summary({"domain_priors": priors, "research_backlog": backlog, "failure_memory": []})
        plan = build_research_director_plan(
            {
                "domain_priors": priors,
                "research_backlog": backlog,
                "research_director_summary": summary,
                "failure_memory": [{"family": "setup", "hypothesis_id": "H-00001", "failure_type": "cpcv_tail_path_fragility", "status": "freeze"}],
                "research_action_history": [],
                "blocking_issues": [],
            },
            {
                "selected_family": "setup",
                "controller_override": {"active_family": "setup"},
            },
        )
        self.assertEqual(plan["assigned_research_action"], "cpcv_attribution")

    def test_research_director_runs_structural_family_directly_after_pivot(self) -> None:
        priors = build_curated_domain_priors()
        hypotheses = build_hypotheses_from_priors(priors)
        backlog = build_research_backlog(
            hypotheses,
            [
                {"family": "setup", "hypothesis_id": "H-00001", "failure_type": "cpcv_tail_path_fragility", "status": "freeze"},
                {"family": "setup", "hypothesis_id": "H-00005", "failure_type": "cpcv_tail_path_fragility", "status": "freeze"},
            ],
            stage2_result={"data_quality": {"sessions": 200}},
        )
        summary = build_research_director_summary({"domain_priors": priors, "research_backlog": backlog, "failure_memory": []})
        plan = build_research_director_plan(
            {
                "domain_priors": priors,
                "research_backlog": backlog,
                "research_director_summary": summary,
                "failure_memory": [
                    {"family": "setup", "hypothesis_id": "H-00001", "failure_type": "cpcv_tail_path_fragility", "status": "freeze"},
                    {"family": "setup", "hypothesis_id": "H-00005", "failure_type": "cpcv_tail_path_fragility", "status": "freeze"},
                ],
                "research_action_history": [{"action_id": "cpcv_attribution", "family": "setup", "hypothesis_id": "H-00001"}],
                "blocking_issues": [],
            },
            {
                "selected_family": "candidate_universe_expansion",
                "controller_override": {"active_family": "candidate_universe_expansion"},
            },
        )
        self.assertEqual(plan["assigned_research_action"], "candidate_universe_expansion")


if __name__ == "__main__":
    unittest.main()
