from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator.adapters.source import (
    LICENSE_PROVENANCE_PROFILE,
    seal_source_bytes,
    seal_source_license_provenance,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(role: str, path: Path, kind: str) -> dict:
    return {
        "role": role,
        "path": str(path),
        "expected_sha256": _sha(path),
        "expected_size_bytes": path.stat().st_size,
        "source_kind": kind,
    }


def _ctx(tmp_path: Path, run_id: str, manifest: dict) -> dict:
    return {
        "run_id": run_id,
        "run_root": tmp_path / run_id,
        "run_manifest": manifest,
        "ledger": {"stages": []},
        "stage": {"id": "01_SOURCE_BYTES_SEALED"},
    }


def _install(ctx: dict, stage_id: str, result: dict) -> None:
    assert result["status"] == "PASS", result
    ctx["ledger"]["stages"].append(
        {"id": stage_id, "status": "PASS", "outputs": result["outputs"]}
    )


def _payload(result: dict) -> dict:
    return json.loads(Path(result["outputs"][0]["path"]).read_text(encoding="utf-8"))


def test_source_and_license_seals_are_run_portable(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"same exact source bytes")
    license_file = tmp_path / "LICENSE.txt"
    license_file.write_text("CC0 exact bytes\n", encoding="utf-8")
    source_page = tmp_path / "source.html"
    source_page.write_text("<html>exact source page</html>\n", encoding="utf-8")

    manifest = {
        "subject_id": "SUBJECT_FREE_PORTABILITY_TEST",
        "source_files": [
            _row("RAW_SOURCE", source, "PRIMARY"),
            _row("LICENSE_BYTES", license_file, "LICENSE"),
            _row("SOURCE_PAGE_SNAPSHOT", source_page, "PROVENANCE"),
        ],
        "source_license": {
            "profile": LICENSE_PROVENANCE_PROFILE,
            "source_pack": "fixture-pack",
            "license_name": "CC0-1.0",
            "license_ref": "https://creativecommons.org/publicdomain/zero/1.0/",
            "license_file_role": "LICENSE_BYTES",
            "source_page_role": "SOURCE_PAGE_SNAPSHOT",
        },
    }

    left = _ctx(tmp_path, "RUN_A", manifest)
    right = _ctx(tmp_path, "RUN_B", manifest)

    left_stage1 = seal_source_bytes(left)
    right_stage1 = seal_source_bytes(right)
    left_payload = _payload(left_stage1)
    right_payload = _payload(right_stage1)

    assert left_payload == right_payload
    assert left_stage1["outputs"][0]["sha256"] == right_stage1["outputs"][0]["sha256"]
    assert "run_id" not in left_payload
    assert "subject_id" not in left_payload
    assert all("path" not in row for row in left_payload["files"])

    for ctx, stage1 in ((left, left_stage1), (right, right_stage1)):
        _install(ctx, "01_SOURCE_BYTES_SEALED", stage1)
        ctx["stage"] = {"id": "02_SOURCE_LICENSE_PROVENANCE"}

    left_stage2 = seal_source_license_provenance(left)
    right_stage2 = seal_source_license_provenance(right)
    left_license = _payload(left_stage2)
    right_license = _payload(right_stage2)

    assert left_license == right_license
    assert left_stage2["outputs"][0]["sha256"] == right_stage2["outputs"][0]["sha256"]
    assert "run_id" not in left_license
    assert "subject_id" not in left_license
