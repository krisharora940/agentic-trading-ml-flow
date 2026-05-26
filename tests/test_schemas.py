import unittest

from trading_ml.bootstrap import build_initial_project_state
from trading_ml.schemas import (
    ContinuationProfile,
    DeskProposal,
    ExperimentRecord,
    FailureProfile,
    FamilyExhaustionRecord,
    RedTeamReview,
    ResearchActionPlan,
    StateOntology,
    StateTransition,
)


class SchemaTests(unittest.TestCase):
    def test_initial_project_state_starts_in_foundation_phase(self) -> None:
        state = build_initial_project_state()
        self.assertEqual(state.phase, "foundation")
        self.assertIsNotNone(state.bnr_spec)
        self.assertEqual(state.evidence_boundary.mode, "strict")

    def test_experiment_record_serializes_expected_fields(self) -> None:
        record = ExperimentRecord(
            experiment_id="exp-001",
            hypothesis="Opening zone break continuation improves expectancy.",
            config_ref="configs/global.toml",
            data_slice={"start": "2024-01-01", "end": "2024-06-30"},
            result={"sharpe": 0.0},
            decision="queued",
            phase="exploration",
        )
        self.assertEqual(record.experiment_id, "exp-001")
        self.assertEqual(record.phase, "exploration")

    def test_v2_supervisor_schemas_serialize(self) -> None:
        proposal = DeskProposal(
            proposal_id="DPROP-1",
            node="path_modeler",
            family="path_modeling",
            claim="Auction state determines continuation validity.",
            target_market_state="opening_impulse_continuation",
        ).to_dict()
        plan = ResearchActionPlan(
            plan_id="RAP-1",
            proposal_id=proposal["proposal_id"],
            action_id="continuation_policy_search",
            family=proposal["family"],
            objective=proposal["claim"],
            allowed_policy_atoms=["tempo_persistence_gate"],
            search_mechanics=["component_ablation"],
            doctrine={"primary_modeling_target": "auction_state_continuation_validity"},
        ).to_dict()
        review = RedTeamReview(
            proposal_id=proposal["proposal_id"], status="pass"
        ).to_dict()
        self.assertTrue(plan["requires_governor_validation"])
        self.assertEqual(plan["validation_scope"], "governor_only")
        self.assertTrue(plan["allowed_policy_atoms"])
        self.assertTrue(plan["search_mechanics"])
        self.assertEqual(review["status"], "pass")

    def test_state_ontology_and_branch_schemas_serialize(self) -> None:
        transition = StateTransition(
            from_state="opening_impulse",
            to_state="healthy_continuation",
            trigger="bnr_event",
            sample_size=12,
        ).to_dict()
        continuation = ContinuationProfile(
            state="opening_impulse",
            sample_size=12,
            continuation_rate=0.58,
            failure_rate=0.42,
        ).to_dict()
        failure = FailureProfile(
            failure_family="no_follow_through",
            state="volatile_chop",
            sample_size=8,
        ).to_dict()
        ontology = StateOntology(
            ontology_id="STATE-ONT-1",
            version=1,
            primary_modeling_target="auction_state_continuation_validity",
            bnr_role="event_trigger_within_state_machine",
            transitions=[transition],
            continuation_profiles=[continuation],
            failure_profiles=[failure],
        ).to_dict()
        exhaustion = FamilyExhaustionRecord(
            family="feature",
            status="exhausted",
            reason="same_family_cycle_limit_reached",
        ).to_dict()
        self.assertEqual(ontology["bnr_role"], "event_trigger_within_state_machine")
        self.assertEqual(exhaustion["status"], "exhausted")


if __name__ == "__main__":
    unittest.main()
