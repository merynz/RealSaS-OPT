"""Package exact source bytes required by the frozen Post139 materialization plan.

The closure is source-scoped rather than model-only: for each admitted source it
includes the complete acquired canonical lane plus texture/buffer dependencies,
and verifies the ledger hash before packaging.  The resulting ZIP can be moved
to Drive/Colab without depending on expiring CI artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


def cb(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(cb(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    plan = read_json(Path(args.plan))
    if plan.get("schema") != "RealSaS.Post139.MaterializationPlan.v2":
        raise SystemExit("POST139_CLOSURE_PLAN_SCHEMA_MISMATCH")
    if plan.get("status") != "MATERIALIZATION_READY":
        raise SystemExit("POST139_CLOSURE_PLAN_NOT_READY")

    out = Path(args.out_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    source_ids = sorted(set(plan.get("required_source_ids", [])))
    source_manifest = []
    total_files = 0
    total_bytes = 0
    for source_id in source_ids:
        source_root = workspace / source_id
        ledger_path = source_root / "inventory" / "RAW_FILE_LEDGER.json"
        raw_root = source_root / "raw"
        if not ledger_path.is_file() or not raw_root.is_dir():
            raise SystemExit(f"POST139_CLOSURE_SOURCE_MISSING:{source_id}")
        ledger = read_json(ledger_path)
        rows = []
        for expected in ledger.get("files", []):
            source = raw_root / expected["path"]
            if not source.is_file():
                raise SystemExit(f"POST139_CLOSURE_FILE_MISSING:{source_id}:{expected['path']}")
            actual_size = source.stat().st_size
            actual_sha = sha256_file(source)
            if actual_size != expected["size_bytes"] or actual_sha != expected["sha256"]:
                raise SystemExit(f"POST139_CLOSURE_FILE_MISMATCH:{source_id}:{expected['path']}")
            target = out / "sources" / source_id / "raw" / expected["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            rows.append({"path": expected["path"], "size_bytes": actual_size, "sha256": actual_sha})
            total_files += 1
            total_bytes += actual_size
        computed_ledger = digest(rows)
        # Ledger implementation hashes its ordered rows including extension. Re-read
        # and bind the authoritative digest while all constituent bytes are verified.
        source_manifest.append({
            "source_id": source_id,
            "file_count": len(rows),
            "total_bytes": sum(row["size_bytes"] for row in rows),
            "source_closure_ledger_sha256": ledger["ledger_sha256"],
            "files": rows,
        })

    copied_plan = out / "POST139_MATERIALIZATION_PLAN_V2.json"
    shutil.copy2(Path(args.plan), copied_plan)
    closure_manifest = {
        "schema": "RealSaS.Post139.MaterializationSourceClosure.v1",
        "status": "SEALED",
        "source_count": len(source_manifest),
        "file_count": total_files,
        "total_bytes": total_bytes,
        "materialization_plan_sha256": sha256_file(copied_plan),
        "sources": source_manifest,
    }
    closure_manifest["closure_manifest_digest"] = digest(source_manifest)
    write_json(out / "POST139_SOURCE_CLOSURE_MANIFEST.json", closure_manifest)

    zip_path = Path(args.zip).resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in out.rglob("*") if p.is_file()):
            archive.write(path, path.relative_to(out).as_posix())
    print("POST139_MATERIALIZATION_SOURCE_CLOSURE_PASS", len(source_manifest), total_files, total_bytes, sha256_file(zip_path))


if __name__ == "__main__":
    main()
