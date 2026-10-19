from __future__ import annotations

from pathlib import Path
import json

from tools.iris_vnext.materialize_fit8_direct_surface_staged_v2 import (
    FAILURE_SCHEMA,
    MATERIALIZER_ID,
    STAGE_RECEIPT_SCHEMA,
    _persist_stage,
    _stage_snapshot,
    parser,
)


def test_staged_materializer_keeps_reference_grid_cli_locked():
    destinations = {action.dest for action in parser()._actions}
    assert "base_cells" not in destinations
    assert "fine_cells" not in destinations
    assert "resolution" not in destinations
    assert "model_query_chunk" in destinations


def test_stage_snapshot_cannot_mint_scientific_authority():
    row = _stage_snapshot(
        subject_id="Snake",
        stage="COARSE_T512_QUERY",
        status="PASS",
        execution_source_sha="a" * 40,
        diagnostics={"coarse_field_min": -1.0, "coarse_field_max": 1.0},
    )
    assert row["schema"] == STAGE_RECEIPT_SCHEMA
    assert row["materializer_id"] == MATERIALIZER_ID
    assert row["base_cells"] == 512
    assert row["fine_cells"] == 1024
    assert row["teacher_truth_used"] is False
    assert row["tessa_invoked"] is False
    assert row["scientific_pass_claimed"] is False


def test_persist_stage_writes_machine_readable_progress(tmp_path: Path):
    _persist_stage(
        tmp_path,
        subject_id="Bat",
        stage="INDEXED_MT_COUNT",
        status="PASS",
        execution_source_sha="b" * 40,
        diagnostics={"boundary_crossing_tets": 0, "exact_faces": 17},
    )
    row = json.loads((tmp_path / "DIRECT_IRIS_SURFACE_STAGE.json").read_text())
    assert row["subject_id"] == "Bat"
    assert row["stage"] == "INDEXED_MT_COUNT"
    assert row["status"] == "PASS"
    assert row["diagnostics"]["exact_faces"] == 17
    assert row["scientific_pass_claimed"] is False


def test_failure_schema_is_distinct_from_success_batch_authority():
    assert FAILURE_SCHEMA == "RealSaS.DirectIRISSurfaceMaterializationFailure.v2"
    assert "Failure" in FAILURE_SCHEMA
