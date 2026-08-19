from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

BAD = {"", "TODO", "TBD", "UNKNOWN", "NONE", None}


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


def validate_source(m, root: Path):
    for p in (
        "schema", "experiment_id", "source_bundle.filename", "source_bundle.sha256",
        "github.repository", "github.commit", "github.path",
        "drive.file_id", "drive.path", "library.file_id", "library.path",
        "tests.command", "tests.status", "evaluator.sha256", "evaluator.identity",
    ):
        need(m, p)
    assert m["tests"]["status"] == "PASS", "source tests are not PASS"
    local = m.get("source_bundle", {}).get("local_verify_path")
    if local:
        p = (root / local).resolve() if not Path(local).is_absolute() else Path(local)
        assert p.exists(), f"source bundle missing: {p}"
        got = sha256(p)
        exp = m["source_bundle"]["sha256"]
        assert got == exp, f"source bundle SHA mismatch: {got} != {exp}"


def validate_input(m, root: Path):
    for p in ("schema", "experiment_id", "inputs"):
        need(m, p)
    assert isinstance(m["inputs"], list) and m["inputs"], "inputs must be a non-empty list"
    for i, x in enumerate(m["inputs"]):
        for k in ("id", "role", "sha256", "authority", "locator"):
            if x.get(k) in BAD:
                raise AssertionError(f"input[{i}] unresolved {k}")
        lp = x.get("local_verify_path")
        if lp:
            p = (root / lp).resolve() if not Path(lp).is_absolute() else Path(lp)
            assert p.exists(), f"input[{i}] local file missing: {p}"
            got = sha256(p)
            assert got == x["sha256"], f"input[{i}] SHA mismatch: {got} != {x['sha256']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--result-path", type=Path)
    a = ap.parse_args()
    sm = json.loads(a.source_manifest.read_text())
    im = json.loads(a.input_manifest.read_text())
    assert sm.get("experiment_id") == im.get("experiment_id"), "experiment_id mismatch"
    validate_source(sm, a.root)
    validate_input(im, a.root)
    if a.result_path is not None:
        assert not a.result_path.exists(), f"canonical result already exists: {a.result_path}"
    print(json.dumps({"status": "PASS", "experiment_id": sm["experiment_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
