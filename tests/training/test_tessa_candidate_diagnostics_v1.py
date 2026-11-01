import json
from pathlib import Path

from tools.training.tessa_candidate_diagnostics_v1 import write_candidate_diagnostics


def test_failed_candidates_survive_without_becoming_teacher_passes(tmp_path):
    producer = tmp_path / "producer.py"
    producer.write_text("# version one\n")
    rows = [{"kind": "SOURCE_FALLBACK", "mechanical_pass": False, "self_intersection_pass": False, "strain": float("inf")}]
    out = write_candidate_diagnostics(subject_id="s", output_dir=tmp_path, candidates=rows, parents={}, producer_files=[producer])
    d = json.loads(out.read_text())
    assert d["status"] == "NO_MECHANICALLY_ADMISSIBLE_CANDIDATE"
    assert d["candidates"][0]["self_intersection_pass"] is False
    assert d["candidates"][0]["strain"] is None
    assert d["nonfinite_metric_paths"] == ["candidates[0].strain"]
    assert d["mechanical_pass_count"] == 0
    assert not d["training_authorized"]
    assert rows[0]["strain"] == float("inf")


def test_changed_producer_preserves_previous_diagnostic_identity(tmp_path):
    p = tmp_path / "producer.py"
    p.write_text("# one\n")
    kwargs = dict(subject_id="s", output_dir=tmp_path, candidates=[{"mechanical_pass": False}], parents={}, producer_files=[p])
    first = write_candidate_diagnostics(**kwargs)
    original = first.read_bytes()
    p.write_text("# two\n")
    second = write_candidate_diagnostics(**kwargs)
    assert first != second
    assert first.read_bytes() == original
    assert second.is_file()
