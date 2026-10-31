from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from product.workspace_worker.worker import (
    Config,
    REQUIRED_BUNDLE_FILES,
    REQUIRED_ROLES,
    WorkspaceWorkerError,
    _validate_bundle,
    _write_request,
    materialize_once,
)


class FakeClient:
    def __init__(self, *, attempt: int = 1):
        self.attempt = attempt
        self.completed: list[bytes] = []
        self.failures: list[tuple[str, bool]] = []

    def claim(self):
        return {
            "compile_id": "cmp_workspace",
            "product_revision_id": "11111111-1111-1111-1111-111111111111",
            "lease_token": "lease-token",
            "attempt": self.attempt,
            "artifacts": [{"role": role} for role in sorted(REQUIRED_ROLES)],
        }

    def download_artifact(self, compile_id: str, lease_token: str, role: str):
        self._identity(compile_id, lease_token)
        return role.encode("utf-8"), f"RealSaS.{role}", "v1"

    def complete(self, compile_id: str, lease_token: str, payload: bytes):
        self._identity(compile_id, lease_token)
        self.completed.append(payload)

    def fail(self, compile_id: str, lease_token: str, error_code: str, retry: bool):
        self._identity(compile_id, lease_token)
        self.failures.append((error_code, retry))

    @staticmethod
    def _identity(compile_id: str, lease_token: str):
        if compile_id != "cmp_workspace" or lease_token != "lease-token":
            raise AssertionError((compile_id, lease_token))


def config(*, max_attempts: int = 3) -> Config:
    return Config(
        platform_url="http://127.0.0.1:8080",
        operator_token="operator-token",
        worker_name="workspace-test",
        materializer_command=("workspace-materializer",),
        poll_seconds=1.0,
        lease_seconds=300,
        max_attempts=max_attempts,
    )


def write_valid_bundle(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(REQUIRED_BUNDLE_FILES):
            archive.writestr(name, b"{}")


class WorkspaceWorkerTests(unittest.TestCase):
    def test_request_requires_exact_product_revision_roles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            request, output = _write_request(
                root,
                "cmp_workspace",
                {"product_revision_id": "revision", "attempt": 1},
                [{"role": role, "path": f"/{role}"} for role in sorted(REQUIRED_ROLES)],
            )
            payload = json.loads(request.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["required_roles"]), REQUIRED_ROLES)
            self.assertEqual(payload["required_output"]["path"], str(output))
            self.assertTrue(payload["authority"]["workspace_is_derived_not_product_authority"])
            self.assertTrue(payload["authority"]["proof_must_be_preserved"])

    def test_bundle_validator_rejects_missing_canonical_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bundle.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("bundle_manifest.json", b"{}")
            with self.assertRaisesRegex(WorkspaceWorkerError, "REQUIRED_FILE_MISSING"):
                _validate_bundle(path)

    def test_valid_bundle_is_only_completion_path(self):
        client = FakeClient()

        def fake_run(command, **kwargs):
            request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            write_valid_bundle(Path(request["required_output"]["path"]))
            return subprocess.CompletedProcess(command, 0)

        with patch("product.workspace_worker.worker.subprocess.run", side_effect=fake_run):
            self.assertTrue(materialize_once(config(), client))
        self.assertEqual(client.failures, [])
        self.assertEqual(len(client.completed), 1)

    def test_invalid_bundle_never_completes(self):
        client = FakeClient()

        def fake_run(command, **kwargs):
            request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            Path(request["required_output"]["path"]).write_bytes(b"not-a-zip")
            return subprocess.CompletedProcess(command, 0)

        with patch("product.workspace_worker.worker.subprocess.run", side_effect=fake_run):
            self.assertTrue(materialize_once(config(max_attempts=3), client))
        self.assertEqual(client.completed, [])
        self.assertEqual(len(client.failures), 1)
        self.assertTrue(client.failures[0][1])

    def test_wrong_role_set_is_rejected_before_materializer(self):
        client = FakeClient()
        lease = client.claim()
        lease["artifacts"] = lease["artifacts"][:-1]
        client.claim = lambda: lease
        with patch("product.workspace_worker.worker.subprocess.run") as run:
            with self.assertRaisesRegex(WorkspaceWorkerError, "ROLE_SET_MISMATCH"):
                materialize_once(config(), client)
        run.assert_not_called()
        self.assertEqual(client.completed, [])


if __name__ == "__main__":
    unittest.main()
