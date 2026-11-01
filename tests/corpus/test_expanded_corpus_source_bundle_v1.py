import hashlib
import json
import zipfile
import pytest

from tools.corpus.prepare_expanded_corpus_source_bundle_v1 import create


def candidate(tmp_path, document, digest=None):
    base, publishers = tmp_path / "base", tmp_path / "publishers"
    base.mkdir()
    publishers.mkdir()
    source = base / "fixture.gltf"
    source.write_text(json.dumps(document))
    manifest = tmp_path / "candidates.json"
    manifest.write_text(json.dumps({"subjects": [{"subject_id": "fixture", "source_id": "fixture_pack", "origin": "ESTABLISHED_BASE", "source": {
        "path": str(source), "sha256": digest or hashlib.sha256(source.read_bytes()).hexdigest()}}]}))
    return manifest, base, publishers


def test_bundle_reopens_original_bytes_without_promoting_training_authority(tmp_path):
    args = candidate(tmp_path, {"asset": {"version": "2.0"}})
    output = tmp_path / "bundle.zip"
    manifest, archive = create(*args, output)
    assert manifest["subject_count"] == 1
    assert not manifest["training_authorized"]
    with zipfile.ZipFile(output) as z:
        row = manifest["subjects"][0]
        assert hashlib.sha256(z.read(row["source_relative_path"])).hexdigest() == row["source_sha256"]
    assert archive["size_bytes"] == output.stat().st_size


def test_missing_texture_and_external_root_are_not_silently_bundled(tmp_path):
    args = candidate(tmp_path, {"images": [{"uri": "../outside.png"}]})
    (tmp_path / "outside.png").write_bytes(b"outside declared source roots")
    with pytest.raises(ValueError, match="OUTSIDE_DECLARED_ROOTS"):
        create(*args, tmp_path / "bundle.zip")


def test_changed_canonical_source_cannot_be_staged(tmp_path):
    args = candidate(tmp_path, {"asset": {"version": "2.0"}}, digest="0" * 64)
    with pytest.raises(ValueError, match="CANONICAL_BYTES_DRIFT"):
        create(*args, tmp_path / "bundle.zip")
