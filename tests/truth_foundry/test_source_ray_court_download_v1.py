import base64
import hashlib
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/truth_foundry/download_pinned_appearance_court_v1.js"
ARCHIVE = b"sealed artifact transport test bytes"
DIGEST = "sha256:" + hashlib.sha256(ARCHIVE).hexdigest()


def _download(tmp_path, *, metadata_patch=None, archive=ARCHIVE):
    metadata = {"id": 123, "name": "sealed-court", "workflow_run": {"id": 456},
                "expired": False, "digest": DIGEST}
    metadata.update(metadata_patch or {})
    payload = {"metadata": metadata, "archive": base64.b64encode(archive).decode(),
               "env": {"COURT_ARTIFACT_ID": "123", "COURT_RUN_ID": "456",
                       "COURT_ARTIFACT_NAME": "sealed-court", "COURT_ARTIFACT_DIGEST": DIGEST,
                       "RUNNER_TEMP": str(tmp_path)}}
    harness = r'''
const fs = require("node:fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
let downloads = 0;
const github = {rest: {actions: {
  getArtifact: async () => ({data: input.metadata}),
  downloadArtifact: async () => { downloads++; return {data: Buffer.from(input.archive, "base64")}; },
}}};
require(process.argv[1])({github, context: {repo: {owner: "fixture", repo: "fixture"}}, env: input.env})
  .then(result => process.stdout.write(JSON.stringify({result, downloads})))
  .catch(error => process.stdout.write(JSON.stringify({error: error.message, downloads})));
'''
    process = subprocess.run(["node", "-e", harness, str(MODULE)], input=json.dumps(payload),
                             text=True, capture_output=True, check=True)
    return json.loads(process.stdout)


def test_transport_verifies_metadata_and_exact_archive_without_host_cli(tmp_path):
    result = _download(tmp_path)
    assert result["downloads"] == 1
    assert result["result"]["digest"] == DIGEST
    assert Path(result["result"]["path"]).read_bytes() == ARCHIVE


@pytest.mark.parametrize("patch", [
    {"id": 124}, {"name": "other-court"}, {"workflow_run": {"id": 457}},
    {"expired": True}, {"digest": "sha256:" + "f" * 64},
])
def test_wrong_artifact_identity_is_rejected_before_download(tmp_path, patch):
    result = _download(tmp_path, metadata_patch=patch)
    assert result == {"error": "SOURCE_RAY_COURT_METADATA_DRIFT", "downloads": 0}
    assert not (tmp_path / "source-ray-appearance-243.zip").exists()


def test_wrong_downloaded_bytes_are_rejected_before_publication(tmp_path):
    result = _download(tmp_path, archive=b"substituted archive")
    assert result == {"error": "SOURCE_RAY_COURT_ARCHIVE_BYTES_DRIFT", "downloads": 1}
    assert not (tmp_path / "source-ray-appearance-243.zip").exists()
