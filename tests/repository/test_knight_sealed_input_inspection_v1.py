import hashlib
import json
import zipfile

import pytest

from tools.ops.export_knight_sealed_inputs_v1 import SealedInputCatalogue


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
