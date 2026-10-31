from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from product.intake_worker.worker import Config, IntakeWorkerError, _write_request, materialize_once


class FakeClient:
    def __init__(self, *, attempt: int = 1):
        self.attempt = attempt
        self.completed: list[tuple[str, str, bytes]] = []
        self.failures: list[tuple[str, str, str, bool]] = []

    def claim(self):
        return {
            "compile_id": "cmp_test",
            "lease_token": "lease-token",
            "attempt": self.attempt,
            "views": [{"slot": "front"}],
        }

    def download_view(self, compile_id: str, lease_token: str, slot: str):
        self._assert_identity(compile_id, lease_token)
        self.assert_slot = slot
        return b"\x89PNG\r\n\x1a\n", "image/png", "hero.png"

    def complete(self, compile_id: str, lease_token: str, payload: bytes):
        self._assert_identity(compile_id, lease_token)
        self.completed.append((compile_id, lease_token, payload))
        return {"status": "QUEUED"}

    def fail(self, compile_id: str, lease_token: str, error_code: str, retry: bool):
        self._assert_identity(compile_id, lease_token)
        self.failures.append((compile_id, lease_token, error_code, retry))

    @staticmethod
    def _assert_identity(compile_id: str, lease_token: str):
        if compile_id != "cmp_test" or lease_token != "lease-token":
            raise AssertionError((compile_id, lease_token))


def config(*, max_attempts: int = 3) -> Config:
    return Config(
        platform_url="http://127.0.0.1:8080",
        operator_token="operator-test-token",
        worker_name="worker-test",
        materializer_command=("materializer",),
        poll_seconds=1.0,
        lease_seconds=300,
        max_attempts=max_attempts,
    )


class IntakeWorkerTests(unittest.TestCase):
    def test_config_requires_operator_token_and_materializer(self):
        with patch.dict(os.environ, {
            "REALSAS_OPERATOR_TOKEN": "",
            "REALSAS_SOURCE_MATERIALIZER_COMMAND": "",
        }, clear=False):
            with self.assertRaisesRegex(IntakeWorkerError, "REALSAS_OPERATOR_TOKEN"):
                Config.from_env()

        with patch.dict(os.environ, {
            "REALSAS_OPERATOR_TOKEN": "token",
            "REALSAS_SOURCE_MATERIALIZER_COMMAND": "",
        }, clear=False):
            with self.assertRaisesRegex(IntakeWorkerError, "never fabricates missing views"):
                Config.from_env()

    def test_materialization_request_preserves_raw_evidence_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request_path, output_path = _write_request(
                root,
                "cmp_test",
                {"attempt": 2},
                [{
                    "slot": "front",
                    "path": str(root / "front.png"),
                    "mime_type": "image/png",
                    "original_name": "hero.png",
                }],
            )
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "RealSaS.PublicSourceMaterializationRequest.v1")
            self.assertEqual(payload["required_output"]["schema"], "RealSaS.SourceObservation.v1")
            self.assertEqual(payload["required_output"]["path"], str(output_path))
            self.assertTrue(payload["authority"]["raw_views_are_external_evidence_only"])
            self.assertTrue(payload["authority"]["missing_views_must_not_be_duplicated_or_declared_observed"])
            self.assertEqual(len(payload["views"]), 1)

    def test_missing_materializer_output_never_completes_compile(self):
        client = FakeClient(attempt=1)
        with patch("product.intake_worker.worker.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
            did_work = materialize_once(config(max_attempts=3), client)
        self.assertTrue(did_work)
        self.assertEqual(client.completed, [])
        self.assertEqual(len(client.failures), 1)
        self.assertTrue(client.failures[0][3])
        self.assertIn("MATERIALIZER_OUTPUT_MISSING", client.failures[0][2])

    def test_codec_validation_precedes_platform_completion(self):
        client = FakeClient(attempt=1)

        def fake_run(command, **kwargs):
            request_path = Path(command[-1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            Path(request["required_output"]["path"]).write_bytes(b"codec-output")
            return subprocess.CompletedProcess(command, 0)

        with patch("product.intake_worker.worker.subprocess.run", side_effect=fake_run), patch(
            "product.intake_worker.worker.decode_source_observation_arrays_v1",
            side_effect=ValueError("SOURCE_OBSERVATION_NATIVE_SHAPE_MISMATCH"),
        ):
            did_work = materialize_once(config(max_attempts=3), client)
        self.assertTrue(did_work)
        self.assertEqual(client.completed, [])
        self.assertEqual(len(client.failures), 1)
        self.assertIn("SOURCE_OBSERVATION_NATIVE_SHAPE_MISMATCH", client.failures[0][2])

    def test_valid_codec_output_is_the_only_path_to_completion(self):
        client = FakeClient(attempt=1)

        def fake_run(command, **kwargs):
            request_path = Path(command[-1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            Path(request["required_output"]["path"]).write_bytes(b"codec-output")
            return subprocess.CompletedProcess(command, 0)

        with patch("product.intake_worker.worker.subprocess.run", side_effect=fake_run), patch(
            "product.intake_worker.worker.decode_source_observation_arrays_v1",
            return_value={"validated": True},
        ):
            did_work = materialize_once(config(max_attempts=3), client)
        self.assertTrue(did_work)
        self.assertEqual(client.failures, [])
        self.assertEqual(client.completed, [("cmp_test", "lease-token", b"codec-output")])

    def test_terminal_attempt_records_failure_without_retry(self):
        client = FakeClient(attempt=3)
        with patch("product.intake_worker.worker.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaisesRegex(IntakeWorkerError, "MATERIALIZER_OUTPUT_MISSING"):
                materialize_once(config(max_attempts=3), client)
        self.assertEqual(client.completed, [])
        self.assertEqual(len(client.failures), 1)
        self.assertFalse(client.failures[0][3])


if __name__ == "__main__":
    unittest.main()
