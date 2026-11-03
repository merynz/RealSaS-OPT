"""Package the exact Post139 source-byte and authority closure.

The closure is source-scoped rather than model-only: for every source required by
the materialization plan it includes the complete acquired canonical lane plus
texture/buffer dependencies and proves the raw-file ledger from those bytes.
When --authority-dir is supplied, the canonical corpus authority ledgers are
embedded as well.  ZIP metadata is normalized so identical inputs yield
identical archive bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
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


def copy_authority(authority_dir: Path, out: Path) -> dict[str, Any]:
    root_path = authority_dir / "POST139_CORPUS_AUTHORITY_V1.json"
    if not root_path.is_file():
        raise SystemExit("POST139_CLOSURE_AUTHORITY_ROOT_MISSING")
    root = read_json(root_path)
    if root.get("schema") != "RealSaS.Post139.CorpusAuthority.v1" or root.get("status") != "SEALED":
        raise SystemExit("POST139_CLOSURE_AUTHORITY_NOT_SEALED")
    expected_digest = digest({key: value for key, value in root.items() if key != "authority_digest"})
    if root.get("authority_digest") != expected_digest:
        raise SystemExit("POST139_CLOSURE_AUTHORITY_ROOT_DIGEST_MISMATCH")
    target = out / "authority"
    target.mkdir(parents=True, exist_ok=True)
    files = []
    required = [root_path]
    for ref in root.get("ledgers", {}).values():
        required.append(authority_dir / ref["file"])
    for source in sorted(required, key=lambda path: path.name):
        if not source.is_file():
            raise SystemExit(f"POST139_CLOSURE_AUTHORITY_LEDGER_MISSING:{source.name}")
        destination = target / source.name
        shutil.copyfile(source, destination)
        files.append({"path": f"authority/{source.name}", "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size})
    return {
        "authority_digest": root["authority_digest"],
        "authority_root_sha256": sha256_file(target / root_path.name),
        "file_count": len(files),
        "files": files,
    }


def deterministic_zip(root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0x800
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--authority-dir")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    plan_path = Path(args.plan).resolve()
    plan = read_json(plan_path)
    if plan.get("schema") != "RealSaS.Post139.MaterializationPlan.v2":
        raise SystemExit("POST139_CLOSURE_PLAN_SCHEMA_MISMATCH")
    if plan.get("status") != "MATERIALIZATION_READY":
        raise SystemExit("POST139_CLOSURE_PLAN_NOT_READY")

    out = Path(args.out_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    source_ids = sorted(set(plan.get("required_source_ids", [])))
    subject_source_ids = sorted({row["source_id"] for row in plan.get("primary_subjects", []) + plan.get("auxiliary_subjects", [])})
    if source_ids != subject_source_ids:
        raise SystemExit(f"POST139_CLOSURE_REQUIRED_SOURCE_SET_MISMATCH:{source_ids}:{subject_source_ids}")

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
        if ledger.get("schema") != "RealSaS.Post139.RawFileLedger.v1" or ledger.get("source_id") != source_id:
            raise SystemExit(f"POST139_CLOSURE_LEDGER_SCHEMA_OR_SOURCE_MISMATCH:{source_id}")
        expected_rows = ledger.get("files", [])
        if digest(expected_rows) != ledger.get("ledger_sha256"):
            raise SystemExit(f"POST139_CLOSURE_LEDGER_DIGEST_INVALID:{source_id}")
        rows = []
        for expected in expected_rows:
            source = raw_root / expected["path"]
            if not source.is_file():
                raise SystemExit(f"POST139_CLOSURE_FILE_MISSING:{source_id}:{expected['path']}")
            actual_size = source.stat().st_size
            actual_sha = sha256_file(source)
            if actual_size != int(expected["size_bytes"]) or actual_sha != expected["sha256"]:
                raise SystemExit(f"POST139_CLOSURE_FILE_MISMATCH:{source_id}:{expected['path']}")
            row = {
                "path": expected["path"],
                "size_bytes": actual_size,
                "sha256": actual_sha,
                "extension": expected["extension"],
            }
            rows.append(row)
            target = out / "sources" / source_id / "raw" / expected["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            total_files += 1
            total_bytes += actual_size
        recomputed = digest(rows)
        if recomputed != ledger["ledger_sha256"]:
            raise SystemExit(f"POST139_CLOSURE_RECOMPUTED_LEDGER_MISMATCH:{source_id}")
        extension_counts = dict(sorted(Counter(row["extension"] for row in rows).items()))
        if int(ledger.get("file_count", -1)) != len(rows) or int(ledger.get("total_bytes", -1)) != sum(row["size_bytes"] for row in rows):
            raise SystemExit(f"POST139_CLOSURE_LEDGER_TOTALS_MISMATCH:{source_id}")
        if ledger.get("extension_counts") != extension_counts:
            raise SystemExit(f"POST139_CLOSURE_LEDGER_EXTENSION_COUNTS_MISMATCH:{source_id}")
        source_manifest.append({
            "source_id": source_id,
            "file_count": len(rows),
            "total_bytes": sum(row["size_bytes"] for row in rows),
            "extension_counts": extension_counts,
            "source_closure_ledger_sha256": recomputed,
            "files": rows,
        })

    copied_plan = out / "POST139_MATERIALIZATION_PLAN_V2.json"
    shutil.copyfile(plan_path, copied_plan)
    authority = copy_authority(Path(args.authority_dir).resolve(), out) if args.authority_dir else None
    closure_core = {
        "sources": source_manifest,
        "materialization_plan_sha256": sha256_file(copied_plan),
        "authority": authority,
    }
    closure_manifest = {
        "schema": "RealSaS.Post139.MaterializationSourceClosure.v1",
        "status": "SEALED",
        "source_count": len(source_manifest),
        "file_count": total_files,
        "total_bytes": total_bytes,
        "materialization_plan_sha256": closure_core["materialization_plan_sha256"],
        "authority": authority,
        "sources": source_manifest,
        "closure_manifest_digest": digest(closure_core),
    }
    write_json(out / "POST139_SOURCE_CLOSURE_MANIFEST.json", closure_manifest)

    zip_path = Path(args.zip).resolve()
    deterministic_zip(out, zip_path)
    print(
        "POST139_MATERIALIZATION_SOURCE_CLOSURE_PASS",
        len(source_manifest),
        total_files,
        total_bytes,
        closure_manifest["closure_manifest_digest"],
        sha256_file(zip_path),
    )


if __name__ == "__main__":
    main()
