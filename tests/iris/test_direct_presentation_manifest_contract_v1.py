import json
from pathlib import Path
import tempfile
import unittest

from tools.iris_vnext.normalize_direct_presentation_manifest_contract_v1 import normalize


class DirectPresentationManifestContractV1Tests(unittest.TestCase):
    def _manifest(self):
        return {
            "schema": "RealSaS.DirectIRISPresentationMemoryManifest.v1",
            "status": "MATERIALIZED_INFERENCE_ADMISSIBLE_ROLE_FREE_PRESENTATION__NO_TRAINING",
            "execution_source_sha": "0" * 40,
            "teacher_geometry_input": False,
            "teacher_topology_input": False,
            "materialized_v2_truth_input": False,
            "subjects": [{"subject_id": "Snake", "file": "snake.presentation_v1.pt", "sha256": "1" * 64}],
            "optimizer_steps_executed": 0,
        }

    def test_normalization_adds_only_conditioning_aliases(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = root / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
            receipt_path = root / "NORMALIZATION_RECEIPT.json"
            original = self._manifest()
            manifest_path.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            receipt = normalize(manifest_path, receipt_path)
            normalized = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(receipt["status"], "PASS__PROVENANCE_ALIAS_NORMALIZED__NO_EVIDENCE_PAYLOAD_MUTATION")
            self.assertFalse(receipt["presentation_subject_payloads_read"])
            self.assertFalse(receipt["presentation_subject_payloads_mutated"])
            self.assertEqual(normalized["subjects"], original["subjects"])
            self.assertFalse(normalized["teacher_geometry_input"])
            self.assertFalse(normalized["teacher_topology_input"])
            self.assertFalse(normalized["materialized_v2_truth_input"])
            self.assertFalse(normalized["teacher_geometry_in_conditioning"])
            self.assertFalse(normalized["teacher_topology_in_conditioning"])
            self.assertFalse(normalized["materialized_v2_truth_in_conditioning"])
            for key, value in original.items():
                self.assertEqual(normalized[key], value)

    def test_rejects_true_source_firewall_flag(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = root / "manifest.json"
            receipt_path = root / "receipt.json"
            value = self._manifest()
            value["teacher_geometry_input"] = True
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SOURCE_FIREWALL_BROKEN"):
                normalize(manifest_path, receipt_path)

    def test_rejects_true_existing_conditioning_alias(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = root / "manifest.json"
            receipt_path = root / "receipt.json"
            value = self._manifest()
            value["teacher_geometry_in_conditioning"] = True
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CONDITIONING_FIREWALL_BROKEN"):
                normalize(manifest_path, receipt_path)


if __name__ == "__main__":
    unittest.main()
