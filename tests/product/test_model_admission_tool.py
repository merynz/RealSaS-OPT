from __future__ import annotations

import unittest

from tools.admit_product_model_release_v1 import FamilySpec, _validate_sha, registration_payload


def _noop(_path):
    return {}


class ProductModelAdmissionToolTest(unittest.TestCase):
    def setUp(self):
        self.spec = FamilySpec(
            family="IRIS",
            architecture_id="RealSaS.IRIS.VNextEvidenceModel.v3",
            artifact_type="RealSaS.IRISVNextEvidenceCheckpoint",
            artifact_schema_version="v1",
            payload_schema="RealSaS.IRISVNextEvidenceCheckpoint.v1",
            validator=_noop,
        )

    def test_registration_is_draft_and_cannot_claim_evaluation_or_promotion(self):
        payload = registration_payload(
            self.spec,
            name="iris-product-deadbeef",
            checkpoint_artifact_id="00000000-0000-4000-8000-000000000001",
            checkpoint_sha256="a" * 64,
            engine_release_id="00000000-0000-4000-8000-000000000002",
            source_code_sha="b" * 40,
            actor="ci",
            validation={
                "schema": self.spec.payload_schema,
                "architecture_id": self.spec.architecture_id,
            },
        )
        self.assertEqual(payload["initial_state"], "DRAFT")
        self.assertEqual(payload["family"], "IRIS")
        self.assertEqual(payload["architecture_id"], self.spec.architecture_id)
        self.assertFalse(payload["metadata"]["evaluation_records_minted"])
        self.assertFalse(payload["metadata"]["promotion_performed"])
        self.assertTrue(payload["metadata"]["validated_by_family_runtime_loader"])
        self.assertNotIn("channel", payload)

    def test_optional_lineage_is_only_added_when_explicit(self):
        base = dict(
            spec=self.spec,
            name="iris-product-deadbeef",
            checkpoint_artifact_id="00000000-0000-4000-8000-000000000001",
            checkpoint_sha256="a" * 64,
            engine_release_id="00000000-0000-4000-8000-000000000002",
            source_code_sha="b" * 40,
            actor="ci",
            validation={},
        )
        payload = registration_payload(**base)
        self.assertNotIn("originating_attempt_id", payload)
        self.assertNotIn("originating_experiment_id", payload)
        self.assertNotIn("supersedes_model_release_id", payload)

        payload = registration_payload(
            **base,
            originating_attempt_id="00000000-0000-4000-8000-000000000003",
            originating_experiment_id="00000000-0000-4000-8000-000000000004",
            supersedes_model_release_id="00000000-0000-4000-8000-000000000005",
        )
        self.assertIn("originating_attempt_id", payload)
        self.assertIn("originating_experiment_id", payload)
        self.assertIn("supersedes_model_release_id", payload)

    def test_source_sha_is_exact_lower_hex(self):
        self.assertEqual(_validate_sha("a" * 40, 40, "BAD"), "a" * 40)
        with self.assertRaisesRegex(RuntimeError, "BAD"):
            _validate_sha("A" * 40, 40, "BAD")
        with self.assertRaisesRegex(RuntimeError, "BAD"):
            _validate_sha("z" * 40, 40, "BAD")
        with self.assertRaisesRegex(RuntimeError, "BAD"):
            _validate_sha("a" * 39, 40, "BAD")


if __name__ == "__main__":
    unittest.main()
