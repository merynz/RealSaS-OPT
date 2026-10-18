from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.IRISVNextFit8CandidateRoundTripReceipt.v1"
CANDIDATE_REMOTE_PREFIX = "gdrive:RealSaS_Fit8_IRIS_vNext_candidates/"
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


def _sha40(value: str, label: str) -> str:
    out = value.strip().lower()
    if len(out) != 40 or any(c not in "0123456789abcdef" for c in out):
        raise ValueError(label)
    return out


def _files(root: Path) -> dict[str, dict[str, Any]]:
    base = root.resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"VNEXT_CANDIDATE_ROUNDTRIP_ROOT_MISSING:{base}")
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = path.relative_to(base).as_posix()
        result[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"VNEXT_CANDIDATE_JSON_OBJECT_REQUIRED:{path}")
    return value


def verify_candidate_roundtrip(
    local_root: Path,
    roundtrip_root: Path,
    expected_source_sha: str,
    candidate_remote: str,
) -> dict[str, Any]:
    expected = _sha40(expected_source_sha, "VNEXT_CANDIDATE_EXPECTED_SOURCE_SHA_INVALID")
    required_remote = CANDIDATE_REMOTE_PREFIX + expected
    if candidate_remote.rstrip("/") != required_remote:
        raise ValueError(
            f"VNEXT_CANDIDATE_REMOTE_MUST_BE_EXACT_SHA_SCOPED:{required_remote}"
        )

    local = local_root.resolve()
    roundtrip = roundtrip_root.resolve()
    for label, rel in _REQUIRED.items():
        if not (local / rel).is_file():
            raise FileNotFoundError(f"VNEXT_CANDIDATE_REQUIRED_LOCAL_FILE_MISSING:{label}")
        if not (roundtrip / rel).is_file():
            raise FileNotFoundError(f"VNEXT_CANDIDATE_REQUIRED_ROUNDTRIP_FILE_MISSING:{label}")

    local_files = _files(local)
    remote_files = _files(roundtrip)
    if set(local_files) != set(remote_files):
        missing_remote = sorted(set(local_files) - set(remote_files))
        extra_remote = sorted(set(remote_files) - set(local_files))
        raise ValueError(
            "VNEXT_CANDIDATE_ROUNDTRIP_FILESET_DRIFT:"
            f"missing={missing_remote}:extra={extra_remote}"
        )
    drift = [
        path
        for path in sorted(local_files)
        if local_files[path]["sha256"] != remote_files[path]["sha256"]
        or local_files[path]["bytes"] != remote_files[path]["bytes"]
    ]
    if drift:
        raise ValueError("VNEXT_CANDIDATE_ROUNDTRIP_SHA256_DRIFT:" + ",".join(drift))

    preparation = _load_json(local / _REQUIRED["preparation_receipt"])
    manifest = _load_json(local / _REQUIRED["cohort_manifest"])
    teacher = _load_json(local / _REQUIRED["teacher_batch_receipt"])
    hydration = _load_json(local / _REQUIRED["hydration_receipt"])

    if preparation.get("schema") != "RealSaS.IRISVNextFit8PreparationReceipt.v2":
        raise ValueError("VNEXT_CANDIDATE_PREPARATION_RECEIPT_SCHEMA_INVALID")
    if manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("VNEXT_CANDIDATE_COHORT_MANIFEST_SCHEMA_INVALID")
    if preparation.get("source_commit_bound") is not True:
        raise ValueError("VNEXT_CANDIDATE_SOURCE_NOT_BOUND")
    source = str(preparation.get("source_commit_sha") or "").lower()
    if source != expected:
        raise ValueError(
            f"VNEXT_CANDIDATE_SOURCE_SHA_DRIFT:expected={expected}:actual={source}"
        )
    if preparation.get("scientific_fit_started") is not False:
        raise ValueError("VNEXT_CANDIDATE_SCIENTIFIC_FIT_ALREADY_STARTED")
    if preparation.get("scientific_fit_pass_claimed") is not False:
        raise ValueError("VNEXT_CANDIDATE_SCIENTIFIC_FIT_PASS_OVERCLAIM")

    report = {
        "schema": SCHEMA,
        "status": "ROUNDTRIP_VERIFIED_CANDIDATE",
        "expected_source_sha": expected,
        "candidate_remote": required_remote,
        "file_count": len(local_files),
        "files": [
            {"path": path, **local_files[path]} for path in sorted(local_files)
        ],
        "teacher_batch_receipt_sha256": sha256_file(
            local / _REQUIRED["teacher_batch_receipt"]
        ),
        "cohort_manifest_sha256": sha256_file(local / _REQUIRED["cohort_manifest"]),
        "cohort_manifest_value_sha256": manifest.get("manifest_value_sha256"),
        "preparation_receipt_sha256": sha256_file(
            local / _REQUIRED["preparation_receipt"]
        ),
        "hydration_receipt_sha256": sha256_file(
            local / _REQUIRED["hydration_receipt"]
        ),
        "teacher_batch_schema": teacher.get("schema"),
        "hydration_receipt_schema": hydration.get("schema"),
        "roundtrip_sha256_equal": True,
        "source_commit_bound": True,
        "scientific_fit_started": False,
        "scientific_fit_pass_claimed": False,
        "candidate_authorized_for_training": False,
        "canonical_remote_promoted": False,
        "authority_note": (
            "ROUNDTRIP_VERIFICATION_ESTABLISHES_BYTE_IDENTITY_ONLY__"
            "EXPERIMENT_ADMISSION_AND_PREREGISTRATION_REMAIN_SEPARATE"
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--roundtrip-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--candidate-remote", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = verify_candidate_roundtrip(
        args.local_root,
        args.roundtrip_root,
        args.expected_source_sha,
        args.candidate_remote,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "VNEXT_FIT8_CANDIDATE_ROUNDTRIP_PASS",
        f"files={report['file_count']}",
        f"source={report['expected_source_sha']}",
        "training_authorized=false",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
