import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.iris_vnext.validate_structural_evidence_input_binding_v1 import (
    OUTPUT_SCHEMA,
    OUTPUT_STATUS,
    SUBJECTS,
    validate_binding,
    validate_materialized,
)


ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "canonical" / "STRUCTURAL_EVIDENCE_INPUT_BINDING_V1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StructuralEvidenceInputBindingV1Tests(unittest.TestCase):
    def _binding(self):
        return json.loads(BINDING.read_text(encoding="utf-8"))

    def test_repository_binding_is_source_only_and_valid(self):
        value = validate_binding(self._binding())
        self.assertFalse(value["scientific_execution_authorized"])
        self.assertFalse(value["optimizer_or_training_authorized"])
        self.assertFalse(value["notebook_execution_authorized"])
        self.assertEqual(value["required_output"]["subject_order"], list(SUBJECTS))

    def test_pairwise_front_cannot_be_relabelled_global_visibility(self):
        value = self._binding()
        value["authority_law"]["global_visibility"] = "PAIRWISE_FRONT"
        with self.assertRaisesRegex(ValueError, "VISIBILITY_AUTHORITY_DRIFT"):
            validate_binding(value)

    def test_semantic_or_teacher_shortcuts_fail_closed(self):
        value = self._binding()
        value["authority_law"]["named_semantic_taxonomy_as_structural_authority"] = True
        with self.assertRaisesRegex(ValueError, "SEMANTIC_SHORTCUT"):
            validate_binding(value)
        value = self._binding()
        value["authority_law"]["teacher_topology_or_rig_in_inference_forward"] = True
        with self.assertRaisesRegex(ValueError, "TEACHER_INFERENCE_SHORTCUT"):
            validate_binding(value)

    def test_ready_binding_requires_exact_git_sha(self):
        value = self._binding()
        value["status"] = "READY_FOR_ZERO_OPTIMIZER_MATERIALIZATION__NOT_SCIENTIFIC_PASS"
        with self.assertRaisesRegex(ValueError, "SOURCE_SHA_MISSING"):
            validate_binding(value)
        value["implementation_source_sha"] = "a" * 40
        validate_binding(value)

    def test_hashed_materialization_admission(self):
        binding = self._binding()
        binding["status"] = "READY_FOR_ZERO_OPTIMIZER_MATERIALIZATION__NOT_SCIENTIFIC_PASS"
        binding["implementation_source_sha"] = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for subject in SUBJECTS:
                filename = subject.lower().replace("-", "").replace(" ", "") + ".all_hit_relation_evidence_v1.npz"
                path = root / filename
                path.write_bytes((subject + "\n").encode("utf-8"))
                rows.append({
                    "subject_id": subject,
                    "file": filename,
                    "sha256": _sha(path),
                    "views": [{} for _ in range(8)],
                    "normalization": {"authority": "EXACT_LEGACY_IRIS_SHARD"},
                })
            manifest = {
                "schema": OUTPUT_SCHEMA,
                "status": OUTPUT_STATUS,
                "subject_order": list(SUBJECTS),
                "subjects": rows,
                "policy": {
                    "candidate_rays_per_view": 8192,
                    "grid_size": 32,
                    "all_unordered_stack_pairs_emitted_both_orientations": True,
                    "pair_point_coordinate_domain": "EXACT_LEGACY_IRIS_NORMALIZED_SOURCE_WORLD",
                },
                "authority": {
                    "front": "DETERMINISTIC_DERIVED__FULL_RAY_STACK_DEPTH_ORDER",
                    "visibility": "DETERMINISTIC_DERIVED__GLOBAL_STACK_RANK_ZERO_ONLY",
                    "occlusion": "POSITIVE_SUPPORT_ONLY__SAME_RAY_PAIR__NO_BINARY_NEGATIVE_AUTHORITY",
                    "projected_boundary": "NOT_EMITTED__KEEP_SEPARATE_CURRENT_VIEW_RGBA_AUTHORITY",
                    "static_same_cut_attachment": "NOT_EMITTED__NO_NEW_AUTHORITY",
                },
                "training_authorized": False,
                "scientific_fit_started": False,
                "teacher_inputs_in_inference_forward": False,
                "source_bindings": {
                    "stage01_bootstrap_sha256": binding["source_inputs"]["stage01"]["bootstrap_sha256"]
                },
            }
            (root / "ALL_HIT_RELATION_EVIDENCE_V1_MANIFEST.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            result = validate_materialized(binding, root)
            self.assertEqual(result["status"], "ADMITTED_HASHED_EVIDENCE_ONLY__NO_SCIENTIFIC_PASS")
            self.assertFalse(result["training_authorized"])
            self.assertFalse(result["scientific_court_authorized"])

            (root / rows[0]["file"]).write_bytes(b"tamper\n")
            with self.assertRaisesRegex(ValueError, "HASH_MISMATCH"):
                validate_materialized(binding, root)


if __name__ == "__main__":
    unittest.main()
