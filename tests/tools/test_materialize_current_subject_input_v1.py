from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.ops.materialize_current_subject_input_v1 import (
    _load_witness_binding,
    _required_sections,
)


def test_required_sections_selects_only_current_read_set() -> None:
    manifest = {
        "run_id": "SEED",
        "source_files": {"rows": []},
        "observation": {"camera_bundle": {}},
        "normalization": {"center_xyz": [0, 0, 0]},
        "unrelated": {"must_not_be_bound": True},
    }
    selected = _required_sections(manifest, {"observation", "source_files", "normalization"})
    assert list(selected) == ["normalization", "observation", "source_files"]
    assert "unrelated" not in selected


def test_required_sections_fails_closed_on_missing_current_read_set() -> None:
    with pytest.raises(RuntimeError, match="CURRENT_SUBJECT_INPUT_SEED_MISSING_READ_SET:normalization"):
        _required_sections({"observation": {}}, {"observation", "normalization"})


def test_witness_binding_separates_platform_subject_from_legacy_evidence_id(tmp_path: Path) -> None:
    inventory = tmp_path / "witness.json"
    inventory.write_text(json.dumps({
        "subject_key": "knight",
        "current_host_lineage_names": {"caa_run": "KNIGHT_LEGACY_RUN"},
    }), encoding="utf-8")
    path, witness, raw, run_id = _load_witness_binding(
        tmp_path,
        "witness.json",
        "current_host_lineage_names.caa_run",
        "knight",
    )
    assert path == inventory.resolve()
    assert witness["subject_key"] == "knight"
    assert raw == inventory.read_bytes()
    assert run_id == "KNIGHT_LEGACY_RUN"


def test_witness_binding_fails_closed_on_platform_subject_mismatch(tmp_path: Path) -> None:
    inventory = tmp_path / "witness.json"
    inventory.write_text(json.dumps({
        "subject_key": "other",
        "current_host_lineage_names": {"caa_run": "KNIGHT_LEGACY_RUN"},
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CURRENT_SUBJECT_INPUT_WITNESS_SUBJECT_MISMATCH:other:knight"):
        _load_witness_binding(
            tmp_path,
            "witness.json",
            "current_host_lineage_names.caa_run",
            "knight",
        )
