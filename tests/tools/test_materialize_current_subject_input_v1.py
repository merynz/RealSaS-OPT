from __future__ import annotations

import pytest

from tools.ops.materialize_current_subject_input_v1 import _required_sections


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
