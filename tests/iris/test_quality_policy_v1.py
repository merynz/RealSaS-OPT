import unittest

from models.iris.vnext.cohort_v8 import vnext_fit8_policy_v5
from models.iris.vnext.quality_policy_v1 import (
    CarrierSealPolicyV1,
    DOWNSTREAM_SPECIALIST_STAGES,
    MetricCourtV1,
    QualityStatus,
    historical_reference_status,
)


class QualityPolicyV1Tests(unittest.TestCase):
    def test_scene_core_is_sealed_downstream_without_court(self):
        policy = CarrierSealPolicyV1()
        with self.assertRaisesRegex(RuntimeError, "REQUIRES_EXPLICIT_COURT"):
            policy.assert_downstream_parameter_allowed(
                stage_name="A1_APPEARANCE_EVIDENCE",
                parameter_name="scene_core.field.weight",
            )

    def test_specialist_head_is_allowed_downstream(self):
        CarrierSealPolicyV1().assert_downstream_parameter_allowed(
            stage_name="R1_ROLE_FREE_RELATIONS",
            parameter_name="relations.front.weight",
        )

    def test_explicit_reopening_court_is_required_and_sufficient_at_contract_level(self):
        CarrierSealPolicyV1().assert_downstream_parameter_allowed(
            stage_name="T1_MECHANICAL_STRUCTURE",
            parameter_name="scene_core.field.weight",
            reopening_court_id="IRIS-CARRIER-FALSIFICATION-001",
        )

    def test_unknown_post_m1_alias_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "UNKNOWN_POST_M1_STAGE_ID"):
            CarrierSealPolicyV1().assert_stage_definition(
                stage_name="A1_APPEARANCE",
                scene_lr_scale=0.0,
            )

    def test_v8_policy_uses_exact_canonical_downstream_ids_and_zero_scene_lr(self):
        policy = vnext_fit8_policy_v5()
        downstream = tuple(stage for stage in policy.stages if stage.stage_id in DOWNSTREAM_SPECIALIST_STAGES)
        self.assertEqual(tuple(stage.stage_id for stage in downstream), DOWNSTREAM_SPECIALIST_STAGES)
        seal = CarrierSealPolicyV1()
        for stage in downstream:
            seal.assert_stage_definition(
                stage_name=stage.stage_id,
                scene_lr_scale=stage.scene_lr_scale,
            )
            self.assertEqual(stage.scene_lr_scale, 0.0)

    def test_optimizer_group_guard_rejects_nonzero_scene_lr(self):
        with self.assertRaisesRegex(RuntimeError, "OPTIMIZER_LR_NONZERO_WITHOUT_COURT"):
            CarrierSealPolicyV1().assert_optimizer_configuration(
                stage_name="J1_JOINT_FINETUNE",
                group_lrs={"scene_core": 1.0e-6, "evidence_heads": 5.0e-5, "structure": 2.5e-5},
            )

    def test_optimizer_group_guard_accepts_zero_scene_lr(self):
        CarrierSealPolicyV1().assert_optimizer_configuration(
            stage_name="A1_APPEARANCE_EVIDENCE",
            group_lrs={"scene_core": 0.0, "evidence_heads": 1.0e-4, "structure": 0.0},
        )

    def test_historical_reference_never_declares_court_pass(self):
        self.assertEqual(
            historical_reference_status(
                value_p95=0.04,
                comparable_normalized_residual=True,
            ),
            QualityStatus.ELIGIBLE_FOR_COURT,
        )
        self.assertEqual(
            historical_reference_status(
                value_p95=0.12,
                comparable_normalized_residual=True,
            ),
            QualityStatus.OPTIMIZATION_DEBT,
        )
        self.assertEqual(
            historical_reference_status(
                value_p95=0.001,
                comparable_normalized_residual=False,
            ),
            QualityStatus.MEASUREMENT_ONLY,
        )

    def test_only_preregistered_authoritative_heldout_metric_can_pass(self):
        court = MetricCourtV1(
            metric_name="example_normalized_p95",
            threshold=0.05,
            threshold_preregistered=True,
            target_authoritative=True,
            heldout=True,
        )
        self.assertEqual(court.status(0.049), QualityStatus.COURT_PASS)
        self.assertEqual(court.status(0.051), QualityStatus.OPTIMIZATION_DEBT)

    def test_missing_threshold_cannot_pass(self):
        court = MetricCourtV1(
            metric_name="unsealed_metric",
            threshold=None,
            threshold_preregistered=False,
            target_authoritative=True,
            heldout=True,
        )
        self.assertEqual(court.status(0.0), QualityStatus.ELIGIBLE_FOR_COURT)

    def test_non_authoritative_target_is_measurement_only(self):
        court = MetricCourtV1(
            metric_name="proxy_owner",
            threshold=0.01,
            threshold_preregistered=True,
            target_authoritative=False,
            heldout=True,
        )
        self.assertEqual(court.status(0.0), QualityStatus.MEASUREMENT_ONLY)


if __name__ == "__main__":
    unittest.main()
