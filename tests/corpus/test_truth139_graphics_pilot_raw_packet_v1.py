from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.corpus.build_truth139_graphics_pilot_raw_packet_v1 import (
    REPO_ROOT,
    build,
    checked_out_git_sha,
    sha,
)


def _head() -> str:
    return checked_out_git_sha(REPO_ROOT)


def _fixture(tmp_path: Path, *, external_dependency: bool = False):
    source_root = tmp_path / "source"
    rel = Path("raw/quaternius/pack/glTF/Subject.gltf")
    source = source_root / rel
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"asset":{"version":"2.0"}}')
    source_sha = sha(source)
    deps = [{"uri": "buffer.bin", "sha256": "9" * 64, "byte_size": 1}] if external_dependency else []
    source_row = {
        "corpus_relative_path": rel.as_posix(),
        "sha256": source_sha,
        "byte_size": source.stat().st_size,
        "format": "GLTF",
        "creator": "quaternius",
        "pack_id": "pack",
        "dependencies": deps,
        "dependency_count": len(deps),
        "source_page_snapshot": {"path": "snapshot.html", "sha256": "a" * 64},
    }
    work = {
        "schema": "RealSaS.Truth139MaterializationWorkOrders.v1",
        "work_order_identity_sha256": "b" * 64,
        "subjects": [{"subject_id": "s0", "source": source_row}],
    }
    work_path = tmp_path / "work.json"
    work_path.write_text(json.dumps(work, sort_keys=True))
    pilot = {
        "schema": "RealSaS.Truth139GraphicsMaterializationPilot.v1",
        "status": "READY_FOR_GRAPHICS_MATERIALIZATION_PILOT__NO_TRAINING_AUTHORITY",
        "parent_work_order_identity_sha256": "b" * 64,
        "pilot_identity_sha256": "c" * 64,
        "pilot_size": 1,
        "selected": [{
            "pilot_index": 0,
            "subject_id": "s0",
            "source_sha256": source_sha,
            "corpus_relative_path": rel.as_posix(),
        }],
    }
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps(pilot, sort_keys=True))
    apparatus = {
        "schema": "RealSaS.Truth139ApparatusManifest.v1",
        "status": "READY_FOR_COMMON_SOURCE_IMPORT_RENDER_APPARATUS__NOT_PLATFORM_TRUTH_ADMITTED",
        "parent_work_order_identity_sha256": "b" * 64,
        "parent_pilot_identity_sha256": "c" * 64,
        "apparatus_identity_sha256": "d" * 64,
        "selected": [{"subject_key": "s0", "source": source_row}],
    }
    apparatus_path = tmp_path / "apparatus.json"
    apparatus_path.write_text(json.dumps(apparatus, sort_keys=True))
    closure = {
        "schema": "RealSaS.Truth139StructuralCorpusClosureReceipt.v1",
        "status": "PASS_TRUTH139_STRUCTURAL_CORPUS_CLOSED__RICH_TRUTH_MATERIALIZATION_READY",
        "github_run_id": "1",
        "git_commit_sha": "e" * 40,
        "receipt_identity_sha256": "f" * 64,
        "artifacts": {"TRUTH139_MATERIALIZATION_WORK_ORDERS_V1.json": {"sha256": sha(work_path)}},
    }
    closure_path = tmp_path / "closure.json"
    closure_path.write_text(json.dumps(closure, sort_keys=True))
    return source_root, work_path, pilot_path, apparatus_path, closure_path, rel, source_sha


def test_raw_packet_copies_exact_self_contained_source_and_binds_structural_receipt(tmp_path):
    source_root, work, pilot, apparatus, closure, rel, source_sha = _fixture(tmp_path)
    output = tmp_path / "out"
    result = build(
        source_root=source_root,
        work_orders=work,
        pilot=pilot,
        apparatus=apparatus,
        closure_receipt=closure,
        output_root=output,
        code_commit_sha=_head(),
    )
    copied = output / "source_root" / rel
    assert result["status"] == "PASS_TRUTH139_GRAPHICS_PILOT_RAW_BYTES_PACKED__RENDER_AND_TRUTH_NOT_PERFORMED"
    assert result["subject_count"] == 1
    assert result["code_commit_sha"] == _head()
    assert len(result["packer_source_sha256"]) == 64
    assert sha(copied) == source_sha
    assert result["structural_closure"]["work_order_file_sha256"] == sha(work)
    assert result["policy"]["render_performed"] is False
    assert result["policy"]["derived_truth_materialized"] is False
    assert result["training_performed"] is False
    assert (output / "metadata" / "STRUCTURAL_CORPUS_CLOSURE_RECEIPT.json").is_file()


def test_raw_packet_rejects_external_dependency_until_explicit_packer_v2(tmp_path):
    source_root, work, pilot, apparatus, closure, _rel, _sha = _fixture(tmp_path, external_dependency=True)
    with pytest.raises(ValueError, match="EXTERNAL_DEPENDENCIES_REQUIRE_EXPLICIT_PACKER_V2"):
        build(
            source_root=source_root,
            work_orders=work,
            pilot=pilot,
            apparatus=apparatus,
            closure_receipt=closure,
            output_root=tmp_path / "out",
            code_commit_sha=_head(),
        )


def test_raw_packet_rejects_structural_work_order_file_drift(tmp_path):
    source_root, work, pilot, apparatus, closure, _rel, _sha = _fixture(tmp_path)
    closure_doc = json.loads(closure.read_text())
    closure_doc["artifacts"]["TRUTH139_MATERIALIZATION_WORK_ORDERS_V1.json"]["sha256"] = "0" * 64
    closure.write_text(json.dumps(closure_doc))
    with pytest.raises(ValueError, match="CLOSURE_WORK_ORDER_FILE_DRIFT"):
        build(
            source_root=source_root,
            work_orders=work,
            pilot=pilot,
            apparatus=apparatus,
            closure_receipt=closure,
            output_root=tmp_path / "out",
            code_commit_sha=_head(),
        )


def test_raw_packet_rejects_declared_commit_that_is_not_checked_out(tmp_path):
    source_root, work, pilot, apparatus, closure, _rel, _sha = _fixture(tmp_path)
    wrong = "0" * 40
    if wrong == _head():
        wrong = "f" * 40
    with pytest.raises(ValueError, match="RAW_PACKET_CODE_COMMIT_DRIFT"):
        build(
            source_root=source_root,
            work_orders=work,
            pilot=pilot,
            apparatus=apparatus,
            closure_receipt=closure,
            output_root=tmp_path / "out",
            code_commit_sha=wrong,
        )
