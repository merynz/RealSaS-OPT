from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "canonical" / "GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json"
MODEL_HOME = ROOT / "models" / "geppetto" / "reference_strength_v1"
EXPERIMENT_HOME = ROOT / "experiments" / "geppetto_reference_strength_fullstack_v1"


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _load_checkpoint_authority():
    path = MODEL_HOME / "checkpoint_authority_v1.py"
    spec = importlib.util.spec_from_file_location("realsas_geppetto_fit1_checkpoint_authority", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load checkpoint authority")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeppettoFIT1EvidenceManifestV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.authority = _load_checkpoint_authority()

    def test_schema_status_and_claim_boundary(self) -> None:
        m = self.manifest
        self.assertEqual(m["schema"], "RealSaS.GeppettoFIT1EvidenceManifest.v1")
        self.assertIn("FIT1_TERMINAL_PASS", m["status"])
        claims = m["claim_boundary"]
        self.assertIs(claims["geppetto_fit1_closed"], True)
        self.assertIs(claims["geppetto_promoted_frozen"], True)
        self.assertIs(claims["unseen_character_generalization"], False)
        self.assertIs(claims["unseen_family_generalization"], False)
        self.assertIs(claims["arachne_a0_closed"], False)
        self.assertIs(claims["arachne_a1_promoted"], False)
        self.assertIs(claims["product_pass"], False)

    def test_checkpoint_authority_matches_manifest(self) -> None:
        m = self.manifest
        source = m["source_authority"]
        result = m["terminal_result"]
        ids = m["content_identities"]
        a = self.authority

        self.assertEqual(source["frozen_optimizer_source_commit"], a.FROZEN_SOURCE_COMMIT)
        self.assertEqual(source["seal_commit"], a.SEAL_COMMIT)
        self.assertEqual(ids["mage_target_sha256"], a.TARGET_CONTENT_SHA256)
        self.assertEqual(ids["qualified_skeleton_ir_sha256"], a.QUALIFIED_SKELETON_IR_SHA256)
        self.assertEqual(ids["final_checkpoint_sha256"], a.FINAL_CHECKPOINT_SHA256)
        self.assertEqual(ids["final_result_json_sha256"], a.FINAL_RESULT_JSON_SHA256)
        self.assertEqual(ids["signed_zero_surface_sha256"], a.SIGNED_ZERO_SURFACE_SHA256)
        self.assertEqual(ids["upstream_iris_checkpoint_sha256"], a.UPSTREAM_IRIS_CHECKPOINT_SHA256)
        self.assertEqual(result["closure_optimizer_step"], a.CLOSURE_STEP)
        self.assertEqual(result["terminal_streak_checks"], a.TERMINAL_STREAK_CHECKS)
        self.assertEqual(result["terminal_stability_optimizer_steps"], a.TERMINAL_STABILITY_OPTIMIZER_STEPS)
        self.assertEqual(result["qualified_control_count"], a.QUALIFIED_CONTROL_COUNT)
        self.assertEqual(tuple(result["diffusion_evaluation_seeds"]), a.DIFFUSION_EVAL_SEEDS)
        self.assertIs(a.GENERALIZATION_CLAIM, False)

    def test_promoted_frozen_source_blob_identities(self) -> None:
        model = self.manifest["promoted_model"]
        pairs = (
            (
                MODEL_HOME / "geppetto_reference_strength_candidate_v1.py",
                model["promoted_candidate_git_blob_sha1"],
            ),
            (
                MODEL_HOME / "rigging_surface_tensorization_v1.py",
                model["promoted_tensorization_git_blob_sha1"],
            ),
            (
                EXPERIMENT_HOME / "geppetto_reference_strength_candidate_v1.py",
                model["sealed_candidate_git_blob_sha1"],
            ),
            (
                EXPERIMENT_HOME / "rigging_surface_tensorization_v1.py",
                model["sealed_tensorization_git_blob_sha1"],
            ),
            (
                EXPERIMENT_HOME / "geppetto_reference_strength_no_learned_slot_v1.py",
                model["sealed_no_learned_slot_git_blob_sha1"],
            ),
            (
                MODEL_HOME / "geppetto_reference_strength_no_learned_slot_v1.py",
                model["promoted_no_learned_slot_wrapper_git_blob_sha1_at_transaction"],
            ),
        )
        for path, expected in pairs:
            self.assertTrue(path.is_file(), str(path))
            self.assertEqual(_git_blob_sha1(path.read_bytes()), expected, str(path))

        self.assertEqual(
            (MODEL_HOME / "geppetto_reference_strength_candidate_v1.py").read_bytes(),
            (EXPERIMENT_HOME / "geppetto_reference_strength_candidate_v1.py").read_bytes(),
        )
        self.assertEqual(
            (MODEL_HOME / "rigging_surface_tensorization_v1.py").read_bytes(),
            (EXPERIMENT_HOME / "rigging_surface_tensorization_v1.py").read_bytes(),
        )

    def test_required_scientific_chain_files_exist(self) -> None:
        for key in ("preregistration", "apparatus_freeze", "loss_freeze", "runner", "closure", "promotion"):
            rel = self.manifest["source_authority"][key]
            self.assertTrue((ROOT / rel).is_file(), rel)

    def test_cross_document_hash_and_scope_bindings(self) -> None:
        m = self.manifest
        expected = (
            m["source_authority"]["frozen_optimizer_source_commit"],
            m["content_identities"]["final_checkpoint_sha256"],
            m["content_identities"]["final_result_json_sha256"],
            m["content_identities"]["qualified_skeleton_ir_sha256"],
        )
        docs = (
            ROOT / "canonical" / "GEPPETTO_REFERENCE_STRENGTH_FIT1_CLOSURE_20260908.md",
            ROOT / "canonical" / "GEPPETTO_REFERENCE_STRENGTH_MAINLINE_PROMOTION_20260909.md",
            ROOT / "canonical" / "FIT1_EVIDENCE_INDEX_20260909.md",
        )
        texts = []
        for path in docs:
            text = path.read_text(encoding="utf-8")
            texts.append(text)
            for token in expected:
                self.assertIn(token, text, f"{path}: missing {token}")
            self.assertRegex(text, r"(?i)generalization")
        joined = "\n".join(texts)
        self.assertIn("PRODUCT_PASS", joined)
        self.assertRegex(joined, r"(?i)(generalization[^\n]*not|not[^\n]*generalization)")

    def test_external_artifact_locators_are_hash_bound(self) -> None:
        ext = self.manifest["external_artifacts"]
        ids = self.manifest["content_identities"]
        self.assertEqual(ext["provider"], "Google Drive")
        for key in ("contract_folder_id", "run_folder_id", "seal_document_id", "package_manifest_id"):
            self.assertTrue(ext[key])
        self.assertEqual(ext["checkpoint"]["sha256"], ids["final_checkpoint_sha256"])
        self.assertEqual(ext["final_result"]["sha256"], ids["final_result_json_sha256"])
        self.assertEqual(ext["qualified_skeleton"]["sha256"], ids["qualified_skeleton_ir_sha256"])
        self.assertGreater(ext["checkpoint"]["bytes"], 1_000_000)
        self.assertGreater(ext["final_result"]["bytes"], 1_000_000)

    def test_upstream_iris_witness_is_explicit_and_nonprivileged_at_inference(self) -> None:
        upstream = self.manifest["upstream_witness"]
        self.assertTrue((ROOT / upstream["iris_promoted_witness"]).is_file())
        self.assertTrue((ROOT / upstream["iris_scene_first_source"]).is_file())
        witness = json.loads((ROOT / upstream["iris_promoted_witness"]).read_text(encoding="utf-8"))
        self.assertEqual(witness["checkpoint_sha256"], upstream["iris_checkpoint_sha256"])
        self.assertEqual(witness["clipped_zero_surface_sha256"], upstream["signed_zero_surface_sha256"])
        self.assertIs(witness["teacher_mesh_used_at_inference"], False)
        self.assertIs(upstream["teacher_mesh_used_at_inference"], False)


if __name__ == "__main__":
    unittest.main()
