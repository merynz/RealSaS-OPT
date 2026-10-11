import hashlib
import json
import zipfile

import pytest

from tools.ops.export_knight_sealed_inputs_v1 import (
    SealedInputCatalogue, go_context, go_selection_sql, pin_current_visual_inputs,
)


def test_nested_sealed_inputs_keep_exact_bytes_and_distinct_paths(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    first, second = root / "first.bin", root / "second.bin"
    first.write_bytes(b"source bytes")
    second.write_bytes(first.read_bytes())
    sha = hashlib.sha256(first.read_bytes()).hexdigest()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"inputs": [{"path": str(first), "sha256": sha},
                                              {"path": str(second), "sha256": sha}]}))
    catalogue = SealedInputCatalogue([root])
    catalogue.add(manifest, origin="SEALED_ROOT")
    report = catalogue.write(tmp_path / "out", context={})
    assert len(report["files"]) == 3
    with zipfile.ZipFile(tmp_path / "out/SEALED_INPUTS.zip") as archive:
        assert archive.read("objects/" + sha) == first.read_bytes()
        assert len(archive.namelist()) == 3
    assert report["qualification_minted"] is False


def test_changed_explicit_bytes_and_outside_paths_fail_before_export(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "input.bin"
    source.write_bytes(b"changed")
    catalogue = SealedInputCatalogue([root])
    with pytest.raises(RuntimeError, match="BYTES_DRIFT"):
        catalogue.add(source, expected="a" * 64, origin="SEALED_ROOT")
    with pytest.raises(RuntimeError, match="OUTSIDE_DECLARED_ROOTS"):
        catalogue.add(tmp_path / "secret", origin="SEALED_ROOT")


def test_unbound_path_is_observable_not_silently_minted_as_bound_input(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"future_layer_path": str(tmp_path / "missing.bin")}))
    catalogue = SealedInputCatalogue([tmp_path])
    catalogue.add(manifest, origin="SEALED_ROOT")
    assert len(catalogue.rows) == 1
    assert len(catalogue.unbound_references) == 1
    assert catalogue.unbound_references[0]["key"] == "future_layer_path"


def test_source_mutation_between_inspection_and_export_is_rejected(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"original")
    catalogue = SealedInputCatalogue([tmp_path])
    catalogue.add(source, origin="SEALED_ROOT")
    source.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="CHANGED_BEFORE_EXPORT"):
        catalogue.write(tmp_path / "out", context={})


def test_bounded_inspection_cannot_truncate_an_oversized_input(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"12345")
    with pytest.raises(RuntimeError, match="BYTE_BUDGET_EXCEEDED"):
        SealedInputCatalogue([tmp_path], byte_budget=4).add(source, origin="SEALED_ROOT")


def test_failed_reference_preserves_other_verified_bytes_without_admitting_drift(tmp_path):
    bad, good = tmp_path / "old_failure.json", tmp_path / "current_input.bin"
    bad.write_text('{"changed":true}')
    good.write_bytes(b"sealed current input")
    sha = hashlib.sha256(good.read_bytes()).hexdigest()
    manifest = tmp_path / "ledger.json"
    manifest.write_text(json.dumps({"history": {"path": str(bad), "sha256": "a" * 64},
                                   "current": {"path": str(good), "sha256": sha}}))
    catalogue = SealedInputCatalogue([tmp_path], collect_reference_failures=True)
    catalogue.inspect(manifest, origin="EXPLICIT_ROOT")
    report = catalogue.write(tmp_path / "out", context={"go": "read-only"})
    assert report["inspection_complete"] is False
    assert report["all_bound_references_verified"] is False
    assert report["reference_failures"][0]["expected_sha256"] == "a" * 64
    assert report["reference_failures"][0]["actual_sha256"] == hashlib.sha256(bad.read_bytes()).hexdigest()
    assert report["reference_failures"][0]["file_transported_as_verified_input"] is False
    assert str(bad) not in catalogue.rows
    assert report["qualification_minted"] is False
    with zipfile.ZipFile(tmp_path / "out/SEALED_INPUTS.zip") as archive:
        assert archive.read("objects/" + sha) == good.read_bytes()
        assert json.loads(archive.read("SEALED_INPUT_INSPECTION.json"))["inspection_complete"] is False


def test_partial_diagnostics_retain_root_and_budget_guards(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "large.bin"
    source.write_bytes(b"12345")
    catalogue = SealedInputCatalogue([root], byte_budget=4, collect_reference_failures=True)
    catalogue.inspect(source, origin="EXPLICIT_ROOT")
    catalogue.inspect(tmp_path / "outside", origin="EXPLICIT_ROOT")
    assert not catalogue.rows
    assert catalogue.total_bytes == 0
    assert len(catalogue.reference_failures) == 2
    assert "BYTE_BUDGET_EXCEEDED" in catalogue.reference_failures[0]["error"]
    assert "OUTSIDE_DECLARED_ROOTS" in catalogue.reference_failures[1]["error"]


def test_explicit_seeds_get_budget_before_reference_walk(tmp_path):
    history, current = tmp_path / "history.bin", tmp_path / "current.bin"
    history.write_bytes(b"h" * 100)
    current.write_bytes(b"current")
    manifest = tmp_path / "ledger.json"
    manifest.write_text(json.dumps({"path": str(history),
                                   "sha256": hashlib.sha256(history.read_bytes()).hexdigest()}))
    catalogue = SealedInputCatalogue([tmp_path], byte_budget=manifest.stat().st_size + 7,
                                    collect_reference_failures=True, defer_reference_walk=True)
    catalogue.inspect(manifest, origin="EXPLICIT_LEDGER")
    catalogue.inspect(current, origin="PINNED_CURRENT_INPUT")
    report = catalogue.write(tmp_path / "out", context={})
    assert str(current) in catalogue.rows
    assert str(history) not in catalogue.rows
    assert not catalogue.pending_references
    assert report["inspection_complete"] is False
    assert "BYTE_BUDGET_EXCEEDED" in report["reference_failures"][0]["error"]


def current_visual_fixture(tmp_path):
    def ref(name, data):
        p = tmp_path / name
        p.write_bytes(data)
        return {"path": str(p), "sha256": hashlib.sha256(data).hexdigest()}
    rasters = [{"view_index": i, "image": ref(f"V{i}.png", b"rgba")} for i in range(8)]
    masks = [{"view_index": i, "mask": ref(f"V{i}.mask", b"mask")} for i in range(8)]
    history = ref("history.bin", b"x" * 10000)
    sidecar = ref("completion.npz", b"sealed-field")
    asset = ref("asset.json", json.dumps({"metadata": {
        "canonical_completion_field_path": sidecar["path"],
        "canonical_completion_field_sha256": sidecar["sha256"],
        "old_model_path": history["path"], "old_model_sha256": history["sha256"]}}).encode())
    ids = ["05_CAMERA_CONTRACT_SOLVED", "07_OBSERVATION_CONTRACT_QUALIFIED",
           "18_CANONICAL_MESH_ADDRESSING_BUILD", "21_CAA_COMPILE",
           "23_COMPLETE_APPEARANCE_ASSET_BAKED", "24_COMPLETE_APPEARANCE_QUALIFIED"]
    ledger = {"stages": [{"id": sid, "outputs": [asset] if sid.startswith("23_") else []} for sid in ids]}
    manifest = {"observation": {"source_rasters": rasters, "source_foreground_masks": masks}}
    return ledger, manifest, asset, sidecar, history


def test_current_visual_source_and_field_receive_budget_before_old_model_history(tmp_path):
    ledger, manifest, asset, sidecar, history = current_visual_fixture(tmp_path)
    budget = (tmp_path / "asset.json").stat().st_size + 64 + len(b"sealed-field")
    catalogue = SealedInputCatalogue([tmp_path], byte_budget=budget,
        collect_reference_failures=True, defer_reference_walk=True)
    pin_current_visual_inputs(catalogue, ledger, manifest)
    report = catalogue.write(tmp_path / "out", context={})
    assert all(row["image"]["path"] in catalogue.rows for row in manifest["observation"]["source_rasters"])
    assert sidecar["path"] in catalogue.rows
    assert history["path"] not in catalogue.rows
    assert len(report["reference_failures"]) == 1
    assert "BYTE_BUDGET_EXCEEDED" in report["reference_failures"][0]["error"]
    assert report["inspection_complete"] is False
    assert report["qualification_minted"] is False


def test_invalid_source_view_matrix_cannot_be_reported_as_complete(tmp_path):
    ledger, manifest, *_ = current_visual_fixture(tmp_path)
    manifest["observation"]["source_rasters"][7]["view_index"] = 6
    with pytest.raises(RuntimeError, match="SOURCE_VIEW_MATRIX_INVALID"):
        pin_current_visual_inputs(SealedInputCatalogue([tmp_path]), ledger, manifest)


def test_go_subject_selection_uses_exact_bytes_without_accepting_sql_or_unbounded_inputs():
    sql = go_selection_sql(["a" * 64, "a" * 64])
    assert "lower(s.display_name)" in sql
    assert "subject_input_artifacts" in sql and "attempt_artifacts" in sql
    assert "a.content_sha256 IN ('" + "a" * 64 + "')" in sql
    assert "LIMIT 21" in sql
    with pytest.raises(RuntimeError, match="INPUT_HASH_INVALID"):
        go_selection_sql(["'; SELECT secret"])
    with pytest.raises(RuntimeError, match="INPUT_HASH_INVALID"):
        go_selection_sql([None])
    with pytest.raises(RuntimeError, match="QUERY_LIMIT_EXCEEDED"):
        go_selection_sql([f"{i:064x}" for i in range(257)])


def test_go_context_reports_truncation_and_never_mints_a_baseline(monkeypatch, tmp_path):
    from tools import platform_deployment_smoke as api
    seen = []
    monkeypatch.setattr(api, "registry_read", lambda root, sql: "\n".join(
        f"00000000-0000-0000-0000-{i:012d}|subject-{i}" for i in range(21)))
    monkeypatch.setattr(api, "request", lambda path: seen.append(path) or {"active_session": None})
    result = go_context(tmp_path, input_hashes=["b" * 64])
    assert len(result["subjects"]) == len(seen) == 20
    assert result["subject_query_truncated"] is True
    assert result["subject_selection_is_execution_authority"] is False
    assert result["no_matching_subject_is_qualified_baseline"] is False
    assert result["transitive_section_payload_identity_match_proven"] is False
