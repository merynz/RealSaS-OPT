from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "compiler" / "vendor" / "realsas_v05_current_execution_closure.b64"
MANIFEST = BUNDLE / "manifest.json"
TARGET_REL = "realsas_synthesis/canonical_graph_optimizer.py"
VISIBLE_ROOT = ROOT / "compiler" / "vendor"
EXPECTED_SHA256 = "b2fddb64753ca783e298be4f1078c70b9667fa9931c67de738f955976e2587c1"
EXPECTED_BYTES = 53544


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_exact_optimizer_bytes() -> bytes:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("scope") != "CURRENT_IRIS_TO_COMPILER_EXECUTION_CLOSURE":
        raise RuntimeError("vendor scope mismatch")
    records = {str(r["path"]): r for r in manifest.get("records", [])}
    record = records.get(TARGET_REL)
    if record is None:
        raise RuntimeError("canonical optimizer record missing from vendor manifest")
    if int(record.get("bytes", -1)) != EXPECTED_BYTES or str(record.get("sha256")) != EXPECTED_SHA256:
        raise RuntimeError("canonical optimizer manifest authority drift")

    encoded: list[bytes] = []
    for part in manifest.get("part_records", []):
        path = BUNDLE / str(part["name"])
        data = path.read_bytes()
        if len(data) != int(part["chars"]):
            raise RuntimeError(f"vendor part size mismatch:{path.name}")
        if _sha(data) != str(part["sha256"]):
            raise RuntimeError(f"vendor part sha mismatch:{path.name}")
        encoded.append(data)
    raw = base64.b64decode(b"".join(encoded), validate=True)
    if len(raw) != int(manifest["raw_size_bytes"]) or _sha(raw) != str(manifest["raw_sha256"]):
        raise RuntimeError("vendor archive authority mismatch")

    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        for info in zf.infolist():
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError(f"unsafe vendor path:{info.filename}")
        data = zf.read(TARGET_REL)
    if len(data) != EXPECTED_BYTES or _sha(data) != EXPECTED_SHA256:
        raise RuntimeError("materialized canonical optimizer sha/size mismatch")
    return data


def visible_path() -> Path:
    return VISIBLE_ROOT / TARGET_REL


def verify_visible() -> None:
    path = visible_path()
    if not path.is_file():
        raise RuntimeError(f"visible canonical optimizer missing:{path}")
    data = path.read_bytes()
    exact = extract_exact_optimizer_bytes()
    if data != exact:
        raise RuntimeError("visible canonical optimizer is not byte-exact vendor authority")


def materialize() -> Path:
    data = extract_exact_optimizer_bytes()
    path = visible_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError("refusing to overwrite non-identical visible canonical optimizer")
    path.write_bytes(data)
    init = path.parent / "__init__.py"
    if not init.exists():
        init.write_text('"""Visible byte-exact canonical graph optimizer authority."""\n', encoding="utf-8")
    verify_visible()
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_visible()
        print(f"PASS_VISIBLE_CANONICAL_GRAPH_OPTIMIZER {EXPECTED_SHA256} {EXPECTED_BYTES}")
    else:
        path = materialize()
        print(f"MATERIALIZED {path.relative_to(ROOT)} {EXPECTED_SHA256} {EXPECTED_BYTES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
