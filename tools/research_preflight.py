from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

BAD_STRINGS = {"", "TODO", "TBD", "UNKNOWN", "NONE"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def is_bad(v):
    return v is None or (isinstance(v, str) and v.strip().upper() in BAD_STRINGS)


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
    if is_bad(cur):
        raise AssertionError(f"unresolved required field: {path}={cur!r}")
    return cur


def valid_sha(x, label):
    assert isinstance(x, str) and SHA256_RE.fullmatch(x), f"invalid SHA-256 {label}: {x!r}"


def validate_source(m, bundle: Path | None):
    for p in (
        "schema", "experiment_id", "source_bundle.filename", "source_bundle.sha256",
        "github.repository", "github.commit", "github.path",
        "drive.file_id", "drive.path", "drive.readback_sha256",
        "library.file_id", "library.path", "library.readback_sha256",
        "checkpoint.identity", "checkpoint.sha256", "checkpoint.container_locator", "checkpoint.container_sha256",
        "evaluator.identity", "evaluator.sha256", "tests.status", "three_store_source_gate",
        "sealed21", "external10", "training_or_retune",
    ):
        need(m, p)
    assert m["three_store_source_gate"] == "PASS"
    assert m["tests"]["status"] == "PASS"
    assert m["sealed21"] == "CLOSED" and m["external10"] == "CLOSED"
    assert m["training_or_retune"] == "FORBIDDEN"
    bundle_sha = m["source_bundle"]["sha256"]
    valid_sha(bundle_sha, "source_bundle.sha256")
    for label, x in (
        ("drive.readback_sha256", m["drive"]["readback_sha256"]),
        ("library.readback_sha256", m["library"]["readback_sha256"]),
        ("checkpoint.sha256", m["checkpoint"]["sha256"]),
        ("checkpoint.container_sha256", m["checkpoint"]["container_sha256"]),
        ("evaluator.sha256", m["evaluator"]["sha256"]),
    ):
        valid_sha(x, label)
    assert m["drive"]["readback_sha256"] == bundle_sha
    assert m["library"]["readback_sha256"] == bundle_sha
    if bundle is not None:
        assert bundle.exists(), f"source bundle missing: {bundle}"
        got = sha256(bundle)
        assert got == bundle_sha, f"source bundle SHA mismatch: {got} != {bundle_sha}"


def validate_compact_input(m):
    for p in ("schema", "experiment_id", "detailed_inventory.sha256", "detailed_inventory.drive_file_id",
              "detailed_inventory.library_file_id", "detailed_inventory.verified_files", "inputs",
              "truth_firewall", "sealed21", "external10", "training_or_retune"):
        need(m, p)
    assert m["sealed21"] == "CLOSED" and m["external10"] == "CLOSED"
    assert m["training_or_retune"] == "FORBIDDEN"
    valid_sha(m["detailed_inventory"]["sha256"], "detailed_inventory.sha256")
    assert int(m["detailed_inventory"]["verified_files"]) == 136
    xs = m["inputs"]
    assert isinstance(xs, list) and xs, "inputs empty"
    ids = set()
    for i, x in enumerate(xs):
        for k in ("id", "role", "sha256", "authority", "locator"):
            assert not is_bad(x.get(k)), f"inputs[{i}] unresolved {k}"
        valid_sha(x["sha256"], f"inputs[{i}].sha256")
        assert x["id"] not in ids, f"duplicate input id: {x['id']}"
        ids.add(x["id"])
    required = {"INPUT_INVENTORY_R6", "OBSERVABLE_RASTER_SET", "EVALUATOR_SIDECAR_SET", "N1D_CANONICAL_PROOF_PACK", "HYBRID_V11_STAGEA_FREEZE_BUNDLE"}
    assert required <= ids, f"missing required input ids: {sorted(required - ids)}"
    return {x["id"]: x for x in xs}


def _entry(x, label, expected_access):
    for k in ("relative_path", "size_bytes", "sha256", "access"):
        assert not is_bad(x.get(k)), f"{label} unresolved {k}"
    valid_sha(x["sha256"], f"{label}.sha256")
    assert int(x["size_bytes"]) > 0
    assert x["access"] == expected_access, f"{label} access mismatch"


def validate_inventory(inv, compact, inputs_by_id, corpus_root: Path | None):
    assert inv.get("experiment_id") == compact.get("experiment_id"), "inventory experiment_id mismatch"
    assert inv.get("source_revision") == "R6", f"unexpected inventory revision: {inv.get('source_revision')}"
    c = need(inv, "corpus")
    rasters = c.get("observable_rasters", [])
    sidecars = c.get("evaluator_sidecars", [])
    assert len(rasters) == 128, f"expected 128 rasters, got {len(rasters)}"
    assert len(sidecars) == 8, f"expected 8 sidecars, got {len(sidecars)}"
    seen = set()
    for i, x in enumerate(rasters):
        _entry(x, f"raster[{i}]", "OBSERVABLE_PHASE_ALLOWED")
        assert x["relative_path"] not in seen
        seen.add(x["relative_path"])
    for i, x in enumerate(sidecars):
        _entry(x, f"sidecar[{i}]", "EVALUATOR_ONLY__FORBIDDEN_UNTIL_TRUTH_OPEN")
        assert x["relative_path"] not in seen
        seen.add(x["relative_path"])
    rsha = need(c, "observable_raster_set_sha256")
    ssha = need(c, "evaluator_sidecar_set_sha256")
    valid_sha(rsha, "observable_raster_set_sha256")
    valid_sha(ssha, "evaluator_sidecar_set_sha256")
    assert rsha == inputs_by_id["OBSERVABLE_RASTER_SET"]["sha256"]
    assert ssha == inputs_by_id["EVALUATOR_SIDECAR_SET"]["sha256"]
    if corpus_root is not None:
        for kind, seq in (("raster", rasters), ("sidecar", sidecars)):
            for x in seq:
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
    ap.add_argument("--detailed-inventory", type=Path)
    ap.add_argument("--corpus-root", type=Path)
    ap.add_argument("--result-path", type=Path)
    a = ap.parse_args()
    sm = json.loads(a.source_manifest.read_text())
    im = json.loads(a.input_manifest.read_text())
    assert sm.get("experiment_id") == im.get("experiment_id"), "experiment_id mismatch"
    validate_source(sm, a.source_bundle)
    by_id = validate_compact_input(im)
    if a.detailed_inventory is not None:
        assert a.detailed_inventory.exists(), "detailed inventory missing"
        got = sha256(a.detailed_inventory)
        exp = im["detailed_inventory"]["sha256"]
        assert got == exp, f"detailed inventory SHA mismatch: {got} != {exp}"
        inv = json.loads(a.detailed_inventory.read_text())
        validate_inventory(inv, im, by_id, a.corpus_root)
    if a.result_path is not None:
        assert not a.result_path.exists(), f"canonical result already exists: {a.result_path}"
    print(json.dumps({
        "status": "PASS",
        "experiment_id": sm["experiment_id"],
        "source_bundle_sha256": sm["source_bundle"]["sha256"],
        "input_inventory_sha256": im["detailed_inventory"]["sha256"],
        "detailed_inventory_verified": a.detailed_inventory is not None,
        "corpus_replayed": a.corpus_root is not None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
