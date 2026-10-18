from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.IRISVNextFit8RemotePreparationAudit.v1"
STATUS_CURRENT = "CURRENT_SOURCE_REMOTE_CANDIDATE"
STATUS_STALE = "STALE_SOURCE_REMOTE"
STATUS_UNBOUND = "UNBOUND_REMOTE_SNAPSHOT"
STATUS_INCOMPLETE = "INCOMPLETE_REMOTE_SNAPSHOT"
STATUS_INVALID = "INVALID_REMOTE_SNAPSHOT"

_REQUIRED = {
    "teacher_batch_receipt": Path("teachers/TEACHER_BATCH_RECEIPT.json"),
    "cohort_manifest": Path("prepared/COHORT_MANIFEST.json"),
    "preparation_receipt": Path("prepared/PREPARATION_RECEIPT.json"),
    "hydration_receipt": Path("receipts/HYDRATION_RECEIPT.json"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"VNEXT_REMOTE_AUDIT_JSON_OBJECT_REQUIRED:{path}")
    return value


def audit_snapshot(snapshot_root: Path, expected_current_source_sha: str) -> dict[str, Any]:
    root = snapshot_root.resolve()
    expected = expected_current_source_sha.strip().lower()
    if len(expected) != 40 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("VNEXT_REMOTE_AUDIT_EXPECTED_SOURCE_SHA_INVALID")

    missing = [label for label, rel in _REQUIRED.items() if not (root / rel).is_file()]
    file_rows = []
    if root.is_dir():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            file_rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "snapshot_root": str(root),
        "expected_current_source_sha": expected,
        "required_missing": missing,
        "file_count": len(file_rows),
        "files": file_rows,
        "remote_source_current": False,
        "remote_consumption_authorized": False,
        "authorization_note": "READ_ONLY_AUDIT_NEVER_AUTHORIZES_CONSUMPTION__REGENERATION_AND_FULL_ROUNDTRIP_REQUIRED",
        "scientific_fit_pass_claimed": False,
    }

    if missing:
        report["status"] = STATUS_INCOMPLETE
        return report

    try:
        teacher = _load_json(root / _REQUIRED["teacher_batch_receipt"])
        manifest = _load_json(root / _REQUIRED["cohort_manifest"])
        preparation = _load_json(root / _REQUIRED["preparation_receipt"])
        hydration = _load_json(root / _REQUIRED["hydration_receipt"])
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        report["status"] = STATUS_INVALID
        report["invalid_reason"] = f"{type(exc).__name__}:{exc}"
        return report

    report.update(
        {
            "teacher_batch_receipt_sha256": sha256_file(root / _REQUIRED["teacher_batch_receipt"]),
            "cohort_manifest_sha256": sha256_file(root / _REQUIRED["cohort_manifest"]),
            "preparation_receipt_sha256": sha256_file(root / _REQUIRED["preparation_receipt"]),
            "hydration_receipt_sha256": sha256_file(root / _REQUIRED["hydration_receipt"]),
            "teacher_batch_schema": teacher.get("schema"),
            "cohort_manifest_schema": manifest.get("schema"),
            "preparation_receipt_schema": preparation.get("schema"),
            "hydration_receipt_schema": hydration.get("schema"),
            "remote_preparation_source_commit_sha": preparation.get("source_commit_sha"),
            "remote_preparation_source_commit_bound": bool(preparation.get("source_commit_bound")),
            "remote_scientific_fit_started": bool(preparation.get("scientific_fit_started")),
            "remote_scientific_fit_pass_claimed": bool(preparation.get("scientific_fit_pass_claimed")),
            "manifest_value_sha256": manifest.get("manifest_value_sha256"),
        }
    )

    source = preparation.get("source_commit_sha")
    if not preparation.get("source_commit_bound") or not isinstance(source, str) or len(source) != 40:
        report["status"] = STATUS_UNBOUND
        return report

    if source.lower() != expected:
        report["status"] = STATUS_STALE
        return report

    report["status"] = STATUS_CURRENT
    report["remote_source_current"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--expected-current-source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-unless-current", action="store_true")
    args = parser.parse_args()

    report = audit_snapshot(args.snapshot_root, args.expected_current_source_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        "VNEXT_FIT8_REMOTE_PREPARATION_AUDIT",
        f"status={report['status']}",
        f"files={report['file_count']}",
        f"expected={report['expected_current_source_sha']}",
        f"remote_source={report.get('remote_preparation_source_commit_sha')}",
        "consumption_authorized=false",
    )
    if args.fail_unless_current and report["status"] != STATUS_CURRENT:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
