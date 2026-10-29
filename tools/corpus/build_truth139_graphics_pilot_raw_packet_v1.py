#!/usr/bin/env python3
"""Build a byte-exact light raw-source packet for the frozen Truth139 Graphics pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

WORK_SCHEMA = "RealSaS.Truth139MaterializationWorkOrders.v1"
PILOT_SCHEMA = "RealSaS.Truth139GraphicsMaterializationPilot.v1"
APPARATUS_SCHEMA = "RealSaS.Truth139ApparatusManifest.v1"
CLOSURE_SCHEMA = "RealSaS.Truth139StructuralCorpusClosureReceipt.v1"
OUTPUT_SCHEMA = "RealSaS.Truth139GraphicsPilotRawPacket.v1"
STATUS = "PASS_TRUTH139_GRAPHICS_PILOT_RAW_BYTES_PACKED__RENDER_AND_TRUTH_NOT_PERFORMED"
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load(path: Path, code: str) -> tuple[Path, dict]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return path, value


def checked_out_git_sha(repo_root: Path = REPO_ROOT) -> str:
    try:
        value = subprocess.run(
            ["git", "-C", str(Path(repo_root).resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("RAW_PACKET_GIT_CHECKOUT_REQUIRED") from exc
    if len(value) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
        raise ValueError("RAW_PACKET_GIT_HEAD_INVALID")
    return value.lower()


def build(
    *,
    source_root: Path,
    work_orders: Path,
    pilot: Path,
    apparatus: Path,
    closure_receipt: Path,
    output_root: Path,
    code_commit_sha: str,
    repo_root: Path = REPO_ROOT,
) -> dict:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    work_path, work = load(work_orders, "RAW_PACKET_WORK")
    pilot_path, pilot_doc = load(pilot, "RAW_PACKET_PILOT")
    apparatus_path, apparatus_doc = load(apparatus, "RAW_PACKET_APPARATUS")
    closure_path, closure = load(closure_receipt, "RAW_PACKET_CLOSURE")

    checkout_sha = checked_out_git_sha(repo_root)
    if str(code_commit_sha).lower() != checkout_sha:
        raise ValueError(f"RAW_PACKET_CODE_COMMIT_DRIFT:{code_commit_sha}!={checkout_sha}")
    if work.get("schema") != WORK_SCHEMA:
        raise ValueError("RAW_PACKET_WORK_SCHEMA")
    if (
        closure.get("schema") != CLOSURE_SCHEMA
        or closure.get("status") != "PASS_TRUTH139_STRUCTURAL_CORPUS_CLOSED__RICH_TRUTH_MATERIALIZATION_READY"
    ):
        raise ValueError("RAW_PACKET_CLOSURE_SCHEMA_STATUS")
    expected_work_file_sha = str(
        ((closure.get("artifacts") or {}).get("TRUTH139_MATERIALIZATION_WORK_ORDERS_V1.json") or {}).get("sha256")
        or ""
    )
    if expected_work_file_sha != sha(work_path):
        raise ValueError("RAW_PACKET_CLOSURE_WORK_ORDER_FILE_DRIFT")
    if (
        pilot_doc.get("schema") != PILOT_SCHEMA
        or pilot_doc.get("status") != "READY_FOR_GRAPHICS_MATERIALIZATION_PILOT__NO_TRAINING_AUTHORITY"
    ):
        raise ValueError("RAW_PACKET_PILOT_SCHEMA_STATUS")
    if (
        apparatus_doc.get("schema") != APPARATUS_SCHEMA
        or apparatus_doc.get("status") != "READY_FOR_COMMON_SOURCE_IMPORT_RENDER_APPARATUS__NOT_PLATFORM_TRUTH_ADMITTED"
    ):
        raise ValueError("RAW_PACKET_APPARATUS_SCHEMA_STATUS")

    work_identity = str(work.get("work_order_identity_sha256") or "")
    if (
        pilot_doc.get("parent_work_order_identity_sha256") != work_identity
        or apparatus_doc.get("parent_work_order_identity_sha256") != work_identity
    ):
        raise ValueError("RAW_PACKET_WORK_IDENTITY_DRIFT")
    if apparatus_doc.get("parent_pilot_identity_sha256") != pilot_doc.get("pilot_identity_sha256"):
        raise ValueError("RAW_PACKET_PILOT_IDENTITY_DRIFT")

    rows = {str(row.get("subject_id") or ""): row for row in work.get("subjects") or ()}
    apparatus_rows = {
        str(row.get("subject_key") or ""): row for row in apparatus_doc.get("selected") or ()
    }
    frozen = list(pilot_doc.get("selected") or ())
    if len(frozen) != int(pilot_doc.get("pilot_size", -1)) or len(apparatus_rows) != len(frozen):
        raise ValueError("RAW_PACKET_CARDINALITY")

    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("RAW_PACKET_OUTPUT_NOT_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)
    files_root = output_root / "source_root"
    files_root.mkdir()

    packed = []
    for frozen_row in frozen:
        subject_id = str(frozen_row.get("subject_id") or "")
        row = rows.get(subject_id)
        apparatus_row = apparatus_rows.get(subject_id)
        if row is None or apparatus_row is None:
            raise ValueError("RAW_PACKET_SUBJECT_BINDING:" + subject_id)
        source = dict(row.get("source") or {})
        if dict(apparatus_row.get("source") or {}) != source:
            raise ValueError("RAW_PACKET_APPARATUS_SOURCE_DRIFT:" + subject_id)
        if (
            source.get("sha256") != frozen_row.get("source_sha256")
            or source.get("corpus_relative_path") != frozen_row.get("corpus_relative_path")
        ):
            raise ValueError("RAW_PACKET_PILOT_SOURCE_DRIFT:" + subject_id)
        dependencies = list(source.get("dependencies") or ())
        if int(source.get("dependency_count", len(dependencies))) != len(dependencies):
            raise ValueError("RAW_PACKET_DEPENDENCY_COUNT:" + subject_id)
        if dependencies:
            raise ValueError("RAW_PACKET_EXTERNAL_DEPENDENCIES_REQUIRE_EXPLICIT_PACKER_V2:" + subject_id)

        relative = Path(str(source.get("corpus_relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("RAW_PACKET_SOURCE_PATH:" + subject_id)
        original = (source_root / relative).resolve()
        if not original.is_relative_to(source_root) or not original.is_file():
            raise FileNotFoundError("RAW_PACKET_SOURCE_MISSING:" + subject_id)
        actual_sha = sha(original)
        expected_sha = str(source.get("sha256") or "")
        if actual_sha != expected_sha:
            raise ValueError(f"RAW_PACKET_SOURCE_SHA:{subject_id}:{actual_sha}!={expected_sha}")

        target = (files_root / relative).resolve()
        if not target.is_relative_to(files_root):
            raise ValueError("RAW_PACKET_TARGET_ESCAPE:" + subject_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if sha(target) != expected_sha:
            raise ValueError("RAW_PACKET_COPY_SHA:" + subject_id)
        packed.append(
            {
                "pilot_index": int(frozen_row.get("pilot_index", -1)),
                "subject_id": subject_id,
                "corpus_relative_path": relative.as_posix(),
                "sha256": expected_sha,
                "byte_size": target.stat().st_size,
                "dependency_count": 0,
            }
        )

    metadata_root = output_root / "metadata"
    metadata_root.mkdir()
    metadata_bindings = []
    for name, path in (
        ("TRUTH139_MATERIALIZATION_WORK_ORDERS_V1.json", work_path),
        ("TRUTH139_GRAPHICS_MATERIALIZATION_PILOT_V1.json", pilot_path),
        ("TRUTH139_GRAPHICS_APPARATUS_MANIFEST_V1.json", apparatus_path),
        ("STRUCTURAL_CORPUS_CLOSURE_RECEIPT.json", closure_path),
    ):
        destination = metadata_root / name
        shutil.copyfile(path, destination)
        metadata_bindings.append({"path": "metadata/" + name, "sha256": sha(destination)})

    payload = {
        "schema": OUTPUT_SCHEMA,
        "date": "2026-10-29",
        "status": STATUS,
        "code_commit_sha": checkout_sha,
        "packer_source_sha256": sha(Path(__file__).resolve()),
        "work_order_identity_sha256": work_identity,
        "pilot_identity_sha256": pilot_doc["pilot_identity_sha256"],
        "apparatus_identity_sha256": apparatus_doc["apparatus_identity_sha256"],
        "subject_count": len(packed),
        "subjects": packed,
        "metadata_files": metadata_bindings,
        "structural_closure": {
            "github_run_id": str(closure.get("github_run_id") or ""),
            "git_commit_sha": str(closure.get("git_commit_sha") or ""),
            "receipt_identity_sha256": str(closure.get("receipt_identity_sha256") or ""),
            "work_order_file_sha256": expected_work_file_sha,
        },
        "policy": {
            "exact_source_bytes_only": True,
            "external_dependencies_supported": False,
            "render_performed": False,
            "derived_truth_materialized": False,
            "training_performed": False,
            "split_created": False,
            "qualification_minted": False,
        },
        "training_performed": False,
        "qualification_minted": False,
    }
    payload["packet_identity_sha256"] = digest_json(payload)
    manifest = output_root / "RAW_PACKET_MANIFEST.json"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--work-orders", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--apparatus", required=True, type=Path)
    parser.add_argument("--closure-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--code-commit-sha", required=True)
    args = parser.parse_args()
    result = build(
        source_root=args.source_root,
        work_orders=args.work_orders,
        pilot=args.pilot,
        apparatus=args.apparatus,
        closure_receipt=args.closure_receipt,
        output_root=args.output_root,
        code_commit_sha=args.code_commit_sha,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "subject_count": result["subject_count"],
                "packet_identity_sha256": result["packet_identity_sha256"],
                "code_commit_sha": result["code_commit_sha"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
