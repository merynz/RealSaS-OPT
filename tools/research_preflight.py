from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

BAD = {"", "TODO", "TBD", "UNKNOWN", "NONE", None}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need(obj, path):
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise AssertionError(f"missing required field: {path}")
        cur = cur[key]
    if cur in BAD:
        raise AssertionError(f"unresolved required field: {path}={cur!r}")
    return cur


def valid_sha(x, label):
    assert isinstance(x, str) and SHA256_RE.fullmatch(x), f"invalid SHA-256 {label}: {x!r}"


def validate_source(m, source_bundle: Path | None):
    for p in (
        "schema", "experiment_id", "source_revision", "status",
        "source_bundle.filename", "source_bundle.sha256",
        "github.repository", "github.source_authority_commit", "github.source_root",
        "mirrors.drive.file_id", "mirrors.drive.path",
        "mirrors.library.file_id", "mirrors.library.path",
        "immutable_source_inputs.gfdr_v2_source_sha256",
    ):
        need(m, p)
    assert m["status"] == "SOURCE_THREE_STORE_PASS", "source three-store gate is not PASS"
    valid_sha(m["source_bundle"]["sha256"], "source_bundle")
    for name, spec in m.get("first_party_files", {}).items():
        valid_sha(spec.get("sha256"), f"first_party_files.{name}")
        assert spec.get("git_blob_sha1") not in BAD, f"missing git blob identity for {name}"
    assert m.get("first_party_files"), "first_party_files empty"
    tests = m.get("tests", {})
    assert tests, "tests missing"
    for k, v in tests.items():
        if isinstance(v, str):
            assert v == "PASS" or v == "BIT_EXACT" or v.startswith("<="), f"source test not PASS: {k}={v}"
    if source_bundle is not None:
        assert source_bundle.exists(), f"source bundle missing: {source_bundle}"
        got = sha256(source_bundle)
        exp = m["source_bundle"]["sha256"]
        assert got == exp, f"source bundle SHA mismatch: {got} != {exp}"


def _validate_entry(x, label):
    for k in ("relative_path", "size_bytes", "sha256", "access"):
        assert x.get(k) not in BAD, f"{label} unresolved {k}"
    valid_sha(x["sha256"], f"{label}.sha256")
    assert int(x["size_bytes"]) > 0, f"{label} empty file"


def validate_input(m, corpus_root: Path | None):
    for p in ("schema", "experiment_id", "source_revision", "population", "immutable_source_inputs", "corpus", "truth_firewall"):
        need(m, p)
    assert m["population"].get("sealed21") == "CLOSED"
    assert m["population"].get("external10") == "CLOSED"
    assert m["population"].get("training_or_retune") == "FORBIDDEN"
    inputs = m["immutable_source_inputs"]
    assert isinstance(inputs, list) and inputs, "immutable_source_inputs empty"
    for i, x in enumerate(inputs):
        for k in ("id", "role", "sha256", "authority", "locator"):
            assert x.get(k) not in BAD, f"immutable_source_inputs[{i}] unresolved {k}"
        valid_sha(x["sha256"], f"immutable_source_inputs[{i}]")
        for name, h in x.get("members", {}).items():
            valid_sha(h, f"immutable_source_inputs[{i}].members.{name}")
    c = m["corpus"]
    rasters = c.get("observable_rasters", []); sidecars = c.get("evaluator_sidecars", [])
    assert rasters and sidecars, "corpus file manifest incomplete"
    seen = set()
    for i, x in enumerate(rasters):
        _validate_entry(x, f"observable_rasters[{i}]")
        assert x["access"] == "OBSERVABLE_PHASE_ALLOWED", "raster access policy mismatch"
        assert x["relative_path"] not in seen, "duplicate corpus path"; seen.add(x["relative_path"])
    for i, x in enumerate(sidecars):
        _validate_entry(x, f"evaluator_sidecars[{i}]")
        assert x["access"] == "EVALUATOR_ONLY__FORBIDDEN_UNTIL_TRUTH_OPEN", "truth firewall mismatch"
        assert x["relative_path"] not in seen, "duplicate corpus path"; seen.add(x["relative_path"])
    valid_sha(c.get("observable_raster_set_sha256"), "observable_raster_set_sha256")
    valid_sha(c.get("evaluator_sidecar_set_sha256"), "evaluator_sidecar_set_sha256")
    if corpus_root is not None:
        for kind, xs in (("raster", rasters), ("sidecar", sidecars)):
            for x in xs:
                p = corpus_root / x["relative_path"]
                assert p.exists(), f"{kind} missing: {p}"
                assert p.stat().st_size == int(x["size_bytes"]), f"{kind} size mismatch: {p}"
                got = sha256(p)
                assert got == x["sha256"], f"{kind} SHA mismatch: {p}: {got} != {x['sha256']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--source-bundle", type=Path)
    ap.add_argument("--corpus-root", type=Path)
    ap.add_argument("--result-path", type=Path)
    a = ap.parse_args()
    sm = json.loads(a.source_manifest.read_text())
    im = json.loads(a.input_manifest.read_text())
    assert sm.get("experiment_id") == im.get("experiment_id"), "experiment_id mismatch"
    assert sm.get("source_revision") == im.get("source_revision"), "source_revision mismatch"
    validate_source(sm, a.source_bundle)
    validate_input(im, a.corpus_root)
    if a.result_path is not None:
        assert not a.result_path.exists(), f"canonical result already exists: {a.result_path}"
    print(json.dumps({"status": "PASS", "experiment_id": sm["experiment_id"], "source_revision": sm["source_revision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
