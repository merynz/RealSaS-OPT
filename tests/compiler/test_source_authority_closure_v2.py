from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator.adapters.source import (
    LICENSE_PROVENANCE_PROFILE,
    SOURCE_CLOSURE_PROFILE,
    seal_source_bytes,
    seal_source_license_provenance,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(role: str, path: Path, kind: str) -> dict:
    return {
        "role": role,
        "path": str(path),
        "expected_sha256": sha(path),
        "expected_size_bytes": path.stat().st_size,
        "source_kind": kind,
    }


def fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    buffer = tmp_path / "mesh.bin"
    buffer.write_bytes(b"exact buffer bytes")
    texture = tmp_path / "albedo.png"
    texture.write_bytes(b"exact texture bytes")
    gltf = tmp_path / "subject.gltf"
    gltf.write_text(json.dumps({
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "mesh.bin", "byteLength": buffer.stat().st_size}],
        "images": [{"uri": "albedo.png"}],
    }, sort_keys=True) + "\n", encoding="utf-8")
    license_file = tmp_path / "License.txt"
    license_file.write_text("CC0 fixture bytes\n", encoding="utf-8")
    source_page = tmp_path / "source.html"
    source_page.write_text("<html>source snapshot</html>\n", encoding="utf-8")
    manifest = {
        "subject_id": "generic_source_fixture",
        "source_files": [
            row("RAW_SOURCE", gltf, "PRIMARY"),
            row("SOURCE_DEP_BUFFER", buffer, "DEPENDENCY"),
            row("SOURCE_DEP_IMAGE", texture, "DEPENDENCY"),
            row("LICENSE_BYTES", license_file, "LICENSE"),
            row("SOURCE_PAGE_SNAPSHOT", source_page, "PROVENANCE"),
        ],
        "source_license": {
            "profile": LICENSE_PROVENANCE_PROFILE,
            "source_pack": "generic-pack",
            "license_name": "CC0-1.0",
            "license_ref": "https://creativecommons.org/publicdomain/zero/1.0/",
            "license_file_role": "LICENSE_BYTES",
            "source_page_role": "SOURCE_PAGE_SNAPSHOT",
        },
    }
    ctx = {
        "run_id": "SOURCE_CLOSURE_V2",
        "run_root": tmp_path / "run",
        "run_manifest": manifest,
        "ledger": {"stages": []},
        "stage": {"id": "01_SOURCE_BYTES_SEALED"},
    }
    return ctx, {"gltf": gltf, "buffer": buffer, "texture": texture,
                 "license": license_file, "page": source_page}


def install(ctx: dict, stage_id: str, result: dict) -> None:
    assert result["status"] == "PASS", result
    ctx["ledger"]["stages"].append({"id": stage_id, "status": "PASS", "outputs": result["outputs"]})


def test_exact_gltf_source_closure_and_license_provenance_pass(tmp_path):
    ctx, _ = fixture(tmp_path)
    stage1 = seal_source_bytes(ctx)
    assert stage1["status"] == "PASS"
    assert stage1["diagnostics"]["source_closure_profile"] == SOURCE_CLOSURE_PROFILE
    assert stage1["diagnostics"]["resolved_dependency_count"] == 2
    install(ctx, "01_SOURCE_BYTES_SEALED", stage1)
    ctx["stage"] = {"id": "02_SOURCE_LICENSE_PROVENANCE"}
    stage2 = seal_source_license_provenance(ctx)
    assert stage2["status"] == "PASS"
    assert stage2["diagnostics"]["provenance_profile"] == LICENSE_PROVENANCE_PROFILE
    assert stage2["diagnostics"]["exact_byte_binding_count"] == 2


def test_gltf_dependency_must_be_declared_and_sealed(tmp_path):
    ctx, _ = fixture(tmp_path)
    ctx["run_manifest"]["source_files"] = [
        item for item in ctx["run_manifest"]["source_files"] if item["role"] != "SOURCE_DEP_IMAGE"
    ]
    result = seal_source_bytes(ctx)
    assert result["status"] == "FAIL"
    assert result["blockers"] == ["SOURCE_GLTF_DEPENDENCY_CLOSURE_INCOMPLETE"]
    assert result["diagnostics"]["missing"]


def test_gltf_remote_dependency_uri_is_forbidden(tmp_path):
    ctx, paths = fixture(tmp_path)
    payload = json.loads(paths["gltf"].read_text())
    payload["images"][0]["uri"] = "https://example.invalid/albedo.png"
    paths["gltf"].write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    primary = ctx["run_manifest"]["source_files"][0]
    primary["expected_sha256"] = sha(paths["gltf"])
    primary["expected_size_bytes"] = paths["gltf"].stat().st_size
    result = seal_source_bytes(ctx)
    assert result["status"] == "FAIL"
    assert result["blockers"] == ["SOURCE_GLTF_EXTERNAL_URI_MUST_BE_LOCAL_FILE"]


def test_exact_license_and_source_page_must_be_stage1_roles(tmp_path):
    ctx, _ = fixture(tmp_path)
    ctx["run_manifest"]["source_files"] = [
        item for item in ctx["run_manifest"]["source_files"] if item["role"] != "SOURCE_PAGE_SNAPSHOT"
    ]
    stage1 = seal_source_bytes(ctx)
    assert stage1["status"] == "PASS"
    install(ctx, "01_SOURCE_BYTES_SEALED", stage1)
    ctx["stage"] = {"id": "02_SOURCE_LICENSE_PROVENANCE"}
    result = seal_source_license_provenance(ctx)
    assert result["status"] == "FAIL"
    assert result["blockers"] == ["SOURCE_LICENSE_STAGE1_ROLE_MISSING:SOURCE_PAGE_SNAPSHOT"]


def test_multiple_primary_sources_fail_closed(tmp_path):
    ctx, paths = fixture(tmp_path)
    ctx["run_manifest"]["source_files"][1]["source_kind"] = "PRIMARY"
    result = seal_source_bytes(ctx)
    assert result["status"] == "FAIL"
    assert result["blockers"] == ["SOURCE_PRIMARY_CARDINALITY_MUST_BE_ONE"]
