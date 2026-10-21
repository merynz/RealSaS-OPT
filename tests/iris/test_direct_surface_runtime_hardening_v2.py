import json
from pathlib import Path
import tempfile
import unittest

from tools.iris_vnext.materialize_fit8_direct_surface_resumable_v2 import (
    DECODER_ID,
    SUBJECT_SCHEMA,
    _valid_completed_subject,
    classify_returncode,
    sha256_file,
)


SOURCE_SHA = "a" * 40
MANIFEST_SHA = "b" * 64
SHARD_SHA = "c" * 64
CHECKPOINT_SHA = "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"
CANDIDATE_SHA = "f" * 64


class DirectSurfaceRuntimeHardeningV2Tests(unittest.TestCase):
    def test_returncode_classifies_sigkill_without_guessing(self):
        row = classify_returncode(-9)
        self.assertEqual(row["returncode"], -9)
        self.assertTrue(row["terminated_by_signal"])
        self.assertEqual(row["signal_number"], 9)
        self.assertEqual(row["signal_name"], "SIGKILL")
        ordinary = classify_returncode(3)
        self.assertFalse(ordinary["terminated_by_signal"])
        self.assertIsNone(ordinary["signal_name"])

    def _write_valid_subject(self, root: Path):
        stage = root / "DIRECT_IRIS_SURFACE.stage12.npz"
        stage.write_bytes(b"stage12")
        stage_sha = sha256_file(stage)
        direct = {
            "schema": "RealSaS.DirectIRISSurfaceReceipt.v1",
            "subject_id": "Snake",
            "source_commit": SOURCE_SHA,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "candidate_binding_hash": CANDIDATE_SHA,
            "stage12_npz_sha256": stage_sha,
            "decoder_id": DECODER_ID,
            "base_cells": 512,
            "fine_cells": 1024,
            "teacher_truth_used": False,
            "compiler_qualified": False,
        }
        direct_path = root / "DIRECT_IRIS_SURFACE_RECEIPT.json"
        direct_path.write_text(json.dumps(direct), encoding="utf-8")
        execution = {
            "schema": SUBJECT_SCHEMA,
            "status": "PASS__EXACT_T512_T1024_SUBJECT_MATERIALIZED__NO_TRAINING",
            "subject_id": "Snake",
            "execution_source_sha": SOURCE_SHA,
            "prepared_manifest_sha256": MANIFEST_SHA,
            "prepared_shard_file": "snake.vnext.pt",
            "prepared_shard_sha256": SHARD_SHA,
            "v8_checkpoint_sha256": CHECKPOINT_SHA,
            "scene_binding_sha256": "d" * 64,
            "scene_planes_sha256": "e" * 64,
            "candidate_binding_hash": CANDIDATE_SHA,
            "stage12_npz_sha256": stage_sha,
            "receipt_sha256": sha256_file(direct_path),
            "vertex_count": 10,
            "face_count": 12,
            "field_query_api": "HISTORICAL_SCENE_CORE_QUERY_V1",
            "decoder_id": DECODER_ID,
            "base_cells": 512,
            "fine_cells": 1024,
            "teacher_truth_used": False,
            "tessa_invoked": False,
            "optimizer_steps_executed": 0,
            "backward_steps_executed": 0,
        }
        (root / "DIRECT_IRIS_SURFACE_SUBJECT_EXECUTION_V2.json").write_text(
            json.dumps(execution), encoding="utf-8"
        )

    def test_completed_subject_reuse_is_hash_and_semantics_bound(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_valid_subject(root)
            row = _valid_completed_subject(
                root,
                subject_id="Snake",
                expected_source_sha=SOURCE_SHA,
                prepared_manifest_sha=MANIFEST_SHA,
                prepared_shard_sha=SHARD_SHA,
                v8_checkpoint_sha=CHECKPOINT_SHA,
            )
            self.assertIsNotNone(row)
            self.assertEqual(row["decoder_id"], DECODER_ID)

            self.assertIsNone(_valid_completed_subject(
                root,
                subject_id="Snake",
                expected_source_sha="9" * 40,
                prepared_manifest_sha=MANIFEST_SHA,
                prepared_shard_sha=SHARD_SHA,
                v8_checkpoint_sha=CHECKPOINT_SHA,
            ))

    def test_completed_subject_reuse_rejects_surface_policy_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_valid_subject(root)
            path = root / "DIRECT_IRIS_SURFACE_SUBJECT_EXECUTION_V2.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["fine_cells"] = 512
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertIsNone(_valid_completed_subject(
                root,
                subject_id="Snake",
                expected_source_sha=SOURCE_SHA,
                prepared_manifest_sha=MANIFEST_SHA,
                prepared_shard_sha=SHARD_SHA,
                v8_checkpoint_sha=CHECKPOINT_SHA,
            ))

    def test_completed_subject_reuse_rejects_inner_receipt_candidate_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_valid_subject(root)
            direct_path = root / "DIRECT_IRIS_SURFACE_RECEIPT.json"
            direct = json.loads(direct_path.read_text(encoding="utf-8"))
            direct["candidate_binding_hash"] = "1" * 64
            direct_path.write_text(json.dumps(direct), encoding="utf-8")
            execution_path = root / "DIRECT_IRIS_SURFACE_SUBJECT_EXECUTION_V2.json"
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            execution["receipt_sha256"] = sha256_file(direct_path)
            execution_path.write_text(json.dumps(execution), encoding="utf-8")
            self.assertIsNone(_valid_completed_subject(
                root,
                subject_id="Snake",
                expected_source_sha=SOURCE_SHA,
                prepared_manifest_sha=MANIFEST_SHA,
                prepared_shard_sha=SHARD_SHA,
                v8_checkpoint_sha=CHECKPOINT_SHA,
            ))

    def test_subject_worker_source_keeps_exact_decoder_and_forbids_training(self):
        source = Path("tools/iris_vnext/materialize_fit8_direct_surface_subject_v2.py").read_text(encoding="utf-8")
        self.assertIn("SparseRegularTetraPolicyV5()", source)
        self.assertIn("decode_vnext_sparse_surface_v3", source)
        self.assertIn('"base_cells": 512', source)
        self.assertIn('"fine_cells": 1024', source)
        self.assertIn('"optimizer_steps_executed": 0', source)
        self.assertIn('"teacher_truth_used": False', source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("loss.backward", source)


if __name__ == "__main__":
    unittest.main()
