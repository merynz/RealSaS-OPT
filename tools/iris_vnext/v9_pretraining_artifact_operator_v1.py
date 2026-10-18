#!/usr/bin/env python3
from __future__ import annotations

"""Build an exact V9 pre-training candidate without external-storage writes.

The operator is deliberately typed and fail-closed. It checks out one allowlisted
review branch at one exact SHA, hydrates only descriptor-pinned Drive objects,
audits the quarantined remote read-only, runs the reviewed preparation/truth/
migration tools, and emits a local candidate for GitHub Actions artifact transport.
It does not persist the candidate to Drive, does not claim a remote round-trip,
does not seal final pre-training materialization, and never imports or invokes a
trainer/optimizer.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "RealSaS.IRISVNextV9PretrainingArtifactRequest.v1"
RESULT_SCHEMA = "RealSaS.IRISVNextV9PretrainingArtifactBuild.v1"
LOCAL_RECEIPT_SCHEMA = "RealSaS.IRISVNextV9LocalCandidateBuildReceipt.v1"
ALLOWED_BRANCH = "research/iris-vnext-post-v8-quality-plan-20261018"
QUARANTINED_REMOTE = "gdrive:RealSaS_Fit8_IRIS_vNext_20261016"
CANDIDATE_REMOTE_PREFIX = "gdrive:RealSaS_Fit8_IRIS_vNext_candidates/"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SUBJECT_FILES = {
    "fit8_snake_v1.pt",
    "fit8_bat_v1.pt",
    "fit8_trex_v1.pt",
    "fit8_horse_v1.pt",
    "fit8_spider_v1.pt",
    "fit8_greenblob_v1.pt",
    "fit8_shark_v1.pt",
    "fit8_george_v1.pt",
}


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(
        list(map(str, args)), cwd=cwd, check=check, text=True,
        env=env, capture_output=capture,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"VNEXT_V9_OPERATOR_JSON_OBJECT_REQUIRED:{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def require_sha64(value: str, code: str) -> str:
    out = str(value or "").lower()
    if not SHA64.fullmatch(out):
        raise RuntimeError(code)
    return out


def copyid(file_id: str, destination: Path) -> None:
    if not file_id or "/" in file_id or " " in file_id:
        raise RuntimeError("VNEXT_V9_OPERATOR_DRIVE_FILE_ID_INVALID")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["rclone", "backend", "copyid", "gdrive:", file_id, str(destination)])


def verify_file(path: Path, expected_sha: str, code: str) -> None:
    if not path.is_file():
        raise RuntimeError(code + ":MISSING")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"{code}:SHA_DRIFT:{expected_sha}:{actual}")


def file_census(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return rows


def validate_request(request: dict[str, Any]) -> str:
    if request.get("schema") != REQUEST_SCHEMA:
        raise RuntimeError("VNEXT_V9_OPERATOR_REQUEST_SCHEMA_INVALID")
    if request.get("target_branch") != ALLOWED_BRANCH:
        raise RuntimeError("VNEXT_V9_OPERATOR_TARGET_BRANCH_INVALID")
    target_sha = str(request.get("target_source_sha") or "").lower()
    if not SHA40.fullmatch(target_sha):
        raise RuntimeError("VNEXT_V9_OPERATOR_TARGET_SHA_INVALID")
    if request.get("training_allowed") is not False:
        raise RuntimeError("VNEXT_V9_OPERATOR_TRAINING_MUST_BE_FALSE")
    if request.get("external_storage_write_allowed") is not False:
        raise RuntimeError("VNEXT_V9_OPERATOR_EXTERNAL_WRITE_MUST_BE_FALSE")
    if request.get("quarantined_remote") != QUARANTINED_REMOTE:
        raise RuntimeError("VNEXT_V9_OPERATOR_QUARANTINE_REMOTE_DRIFT")

    inputs = request.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("VNEXT_V9_OPERATOR_INPUTS_REQUIRED")
    stage = inputs.get("stage01") or {}
    require_sha64(stage.get("sha256", ""), "VNEXT_V9_OPERATOR_STAGE01_SHA_INVALID")
    legacy = inputs.get("legacy") or {}
    require_sha64(legacy.get("manifest_sha256", ""), "VNEXT_V9_OPERATOR_LEGACY_MANIFEST_SHA_INVALID")
    shards = legacy.get("shards")
    if not isinstance(shards, list) or {str(row.get("file")) for row in shards} != EXPECTED_SUBJECT_FILES:
        raise RuntimeError("VNEXT_V9_OPERATOR_LEGACY_SHARD_SET_INVALID")
    for row in shards:
        require_sha64(row.get("sha256", ""), "VNEXT_V9_OPERATOR_LEGACY_SHARD_SHA_INVALID")
        if not row.get("file_id"):
            raise RuntimeError("VNEXT_V9_OPERATOR_LEGACY_SHARD_FILE_ID_MISSING")

    v8 = inputs.get("v8") or {}
    for key in ("checkpoint", "latest", "completion", "scientific_start"):
        row = v8.get(key) or {}
        if not row.get("file_id"):
            raise RuntimeError("VNEXT_V9_OPERATOR_V8_FILE_ID_MISSING:" + key)
        require_sha64(row.get("sha256", ""), "VNEXT_V9_OPERATOR_V8_SHA_INVALID:" + key)
    return target_sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    request_path = args.request.resolve()
    request = load(request_path)
    target_sha = validate_request(request)
    repo = args.repo.resolve()
    authority = args.authority_root.resolve()
    out = args.out_dir.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    worktree = work / "source"

    # Same-repository exact-ref check; no arbitrary URL/ref execution.
    run(["git", "fetch", "--no-tags", "origin", f"refs/heads/{ALLOWED_BRANCH}:refs/remotes/origin/{ALLOWED_BRANCH}"], cwd=repo)
    actual = subprocess.check_output(
        ["git", "rev-parse", f"refs/remotes/origin/{ALLOWED_BRANCH}"], cwd=repo, text=True
    ).strip()
    if actual != target_sha:
        raise RuntimeError(f"VNEXT_V9_OPERATOR_SOURCE_DRIFT:{actual}:{target_sha}")
    if worktree.exists():
        shutil.rmtree(worktree)
    run(["git", "worktree", "add", "--detach", str(worktree), target_sha], cwd=repo)

    try:
        source_env = dict(os.environ)
        source_env["GITHUB_SHA"] = target_sha
        source_env["PYTHONHASHSEED"] = "0"
        source_env["PYTHONPATH"] = str(worktree)

        # 1. Quarantined remote: read-only forensic snapshot.
        audit_root = work / "quarantine_snapshot"
        for rel in ("teachers", "prepared", "receipts"):
            (audit_root / rel).mkdir(parents=True, exist_ok=True)
        run(["rclone", "copy", f"{QUARANTINED_REMOTE}/teachers", str(audit_root / "teachers"), "--checksum"], check=False)
        run(["rclone", "copy", f"{QUARANTINED_REMOTE}/prepared", str(audit_root / "prepared"), "--checksum"], check=False)
        run(["rclone", "copy", f"{QUARANTINED_REMOTE}/receipts/HYDRATION_RECEIPT.json", str(audit_root / "receipts"), "--checksum"], check=False)
        audit_report = out / "evidence/QUARANTINE_AUDIT.json"
        run([
            "python", "tools/iris_vnext/audit_fit8_preparation_snapshot_v1.py",
            "--snapshot-root", str(audit_root),
            "--expected-current-source-sha", target_sha,
            "--output", str(audit_report),
        ], cwd=worktree, env=source_env)
        audit = load(audit_report)
        if audit.get("remote_consumption_authorized") is not False:
            raise RuntimeError("VNEXT_V9_OPERATOR_QUARANTINE_AUTHORITY_VIOLATION")
        if audit.get("status") not in {
            "CURRENT_SOURCE_REMOTE_CANDIDATE", "STALE_SOURCE_REMOTE",
            "UNBOUND_REMOTE_SNAPSHOT", "INCOMPLETE_REMOTE_SNAPSHOT",
            "INVALID_REMOTE_SNAPSHOT",
        }:
            raise RuntimeError("VNEXT_V9_OPERATOR_QUARANTINE_STATUS_INVALID")

        # 2. Exact admitted inputs from immutable Drive IDs, hashes checked again
        # against the research source binding before any teacher/build step.
        inputs = request["inputs"]
        stage_spec = inputs["stage01"]
        stage_zip = work / str(stage_spec["file"])
        copyid(str(stage_spec["file_id"]), stage_zip)
        verify_file(stage_zip, str(stage_spec["sha256"]), "VNEXT_V9_OPERATOR_STAGE01")
        stage_root = work / "stage01"
        shutil.unpack_archive(str(stage_zip), str(stage_root))
        boots = list(stage_root.rglob("BOOTSTRAP_MANIFEST.json"))
        if len(boots) != 1:
            raise RuntimeError(f"VNEXT_V9_OPERATOR_STAGE01_BOOTSTRAP_CARDINALITY:{len(boots)}")
        stage_root = boots[0].parent

        legacy_root = work / "legacy_bound"
        legacy_root.mkdir(parents=True, exist_ok=True)
        legacy = inputs["legacy"]
        legacy_manifest = legacy_root / str(legacy["manifest_file"])
        copyid(str(legacy["manifest_file_id"]), legacy_manifest)
        verify_file(legacy_manifest, str(legacy["manifest_sha256"]), "VNEXT_V9_OPERATOR_LEGACY_MANIFEST")
        for row in legacy["shards"]:
            dst = legacy_root / str(row["file"])
            copyid(str(row["file_id"]), dst)
            verify_file(dst, str(row["sha256"]), "VNEXT_V9_OPERATOR_LEGACY_SHARD:" + dst.name)

        binding = load(worktree / "canonical/VNEXT_FIT8_INPUT_BINDING_V1.json")
        if binding.get("stage01_bundle", {}).get("sha256") != stage_spec["sha256"]:
            raise RuntimeError("VNEXT_V9_OPERATOR_STAGE01_BINDING_DRIFT")
        if binding.get("legacy_iris_cohort_manifest", {}).get("sha256") != legacy["manifest_sha256"]:
            raise RuntimeError("VNEXT_V9_OPERATOR_LEGACY_MANIFEST_BINDING_DRIFT")
        expected_shards = {row["legacy_shard_file"]: row["legacy_shard_sha256"] for row in binding["subjects"]}
        actual_shards = {row["file"]: row["sha256"] for row in legacy["shards"]}
        if expected_shards != actual_shards:
            raise RuntimeError("VNEXT_V9_OPERATOR_LEGACY_BINDING_DRIFT")

        hydration = {
            "schema": "RealSaS.VNextFit8LegacyHydration.v1",
            "subjects": [
                {
                    "subject_id": next(b["subject_id"] for b in binding["subjects"] if b["legacy_shard_file"] == row["file"]),
                    "file": row["file"],
                    "sha256": row["sha256"],
                    "exact_copy_count": 1,
                    "transport_file_id": row["file_id"],
                }
                for row in legacy["shards"]
            ],
            "cohort_manifest_sha256": legacy["manifest_sha256"],
            "cohort_manifest_exact_copy_count": 1,
            "legacy_optimizer_state_reused": False,
            "transport_note": "DIRECT_PINNED_FILE_IDS__HASH_IS_AUTHORITY__COPY_COUNT_NOT_A_SCIENTIFIC_CLAIM",
        }
        write_json(legacy_root / "HYDRATION_RECEIPT.json", hydration)

        # 3. Exact authored-motion teacher export. Runner cache must contain the
        # same pinned Blender version used by the reviewed workflow.
        blender = Path.home() / ".cache/realsas/blender/blender-4.5.14-linux-x64/blender"
        if not blender.is_file():
            raise RuntimeError("VNEXT_V9_OPERATOR_PINNED_BLENDER_4_5_14_MISSING")
        teachers = work / "teachers"
        run([
            "python", "tools/iris_vnext/export_fit8_authored_teachers_v1.py",
            "--blender", str(blender),
            "--binding", "canonical/VNEXT_FIT8_INPUT_BINDING_V1.json",
            "--stage01-bundle", str(stage_zip),
            "--stage01-root", str(stage_root),
            "--legacy-shard-dir", str(legacy_root),
            "--legacy-cohort-manifest", str(legacy_manifest),
            "--output-dir", str(teachers),
            "--max-frames-per-clip", "16",
        ], cwd=worktree, env=source_env)
        if not (teachers / "TEACHER_BATCH_RECEIPT.json").is_file():
            raise RuntimeError("VNEXT_V9_OPERATOR_TEACHER_BATCH_MISSING")

        # 4. Topology-safe V4 cohort.
        prepared = work / "prepared"
        run([
            "python", "tools/iris_vnext/prepare_fit8_vnext_cohort_v1.py",
            "--binding", "canonical/VNEXT_FIT8_INPUT_BINDING_V1.json",
            "--stage01-bundle", str(stage_zip),
            "--stage01-root", str(stage_root),
            "--legacy-shard-dir", str(legacy_root),
            "--legacy-cohort-manifest", str(legacy_manifest),
            "--teacher-dir", str(teachers),
            "--output-dir", str(prepared),
        ], cwd=worktree, env=source_env)
        prep_receipt = load(prepared / "PREPARATION_RECEIPT.json")
        if prep_receipt.get("source_commit_sha") != target_sha or prep_receipt.get("source_commit_bound") is not True:
            raise RuntimeError("VNEXT_V9_OPERATOR_PREPARATION_SOURCE_NOT_BOUND")

        # 5. Relation Truth V2 + Boundary Causality V1.
        relation = work / "truth/relation_v2"
        boundary = work / "truth/boundary_v1"
        run([
            "python", "tools/iris_vnext/build_fit8_relation_truth_v2.py",
            "--legacy-shard-dir", str(legacy_root),
            "--teacher-dir", str(teachers),
            "--stage01-root", str(stage_root),
            "--output-dir", str(relation),
        ], cwd=worktree, env=source_env)
        run([
            "python", "tools/iris_vnext/build_fit8_boundary_causality_v1.py",
            "--teacher-dir", str(teachers),
            "--stage01-root", str(stage_root),
            "--relation-truth-dir", str(relation),
            "--output-dir", str(boundary),
        ], cwd=worktree, env=source_env)
        relation_manifest = relation / "RELATION_TRUTH_V2_MANIFEST.json"
        boundary_manifest = boundary / "BOUNDARY_CAUSALITY_V1_MANIFEST.json"
        if not relation_manifest.is_file() or not boundary_manifest.is_file():
            raise RuntimeError("VNEXT_V9_OPERATOR_TRUTH_MANIFEST_MISSING")

        # 6. Exact completed V8 authority and deterministic zero-step migration.
        v8_root = work / "v8"
        v8_root.mkdir(parents=True, exist_ok=True)
        v8_spec = inputs["v8"]
        v8_paths: dict[str, Path] = {}
        for key, filename in (
            ("checkpoint", str(v8_spec["checkpoint"]["file"])),
            ("latest", "LATEST.json"),
            ("completion", "TRAINING_COMPLETE_V8_M1_SEALED.json"),
            ("scientific_start", "SCIENTIFIC_START_V8_M1_SEALED.json"),
        ):
            spec = v8_spec[key]
            path = v8_root / filename
            copyid(str(spec["file_id"]), path)
            verify_file(path, str(spec["sha256"]), "VNEXT_V9_OPERATOR_V8:" + key)
            v8_paths[key] = path
        migration = work / "migration"
        run([
            "python", "tools/iris_vnext/migrate_v8_relation_head_v2.py",
            "--checkpoint", str(v8_paths["checkpoint"]),
            "--latest-pointer", str(v8_paths["latest"]),
            "--completion-receipt", str(v8_paths["completion"]),
            "--scientific-start", str(v8_paths["scientific_start"]),
            "--output-dir", str(migration),
        ], cwd=worktree, env=source_env)
        migration_receipt = migration / "V8_RELATION_HEAD_TO_V2_MIGRATION_RECEIPT.json"
        migration_value = load(migration_receipt)
        if migration_value.get("status") != "PASS" or int(migration_value.get("training_steps_performed", -1)) != 0:
            raise RuntimeError("VNEXT_V9_OPERATOR_MIGRATION_NOT_ZERO_STEP_PASS")

        # 7. Assemble local candidate. No remote-roundtrip or materialization seal
        # is minted here; those belong to the later connector persistence court.
        candidate = out / "candidate"
        candidate.mkdir(parents=True, exist_ok=True)
        for source, relative in (
            (teachers, "teachers"),
            (prepared, "prepared"),
            (relation, "truth/relation_v2"),
            (boundary, "truth/boundary_v1"),
            (migration, "migration"),
        ):
            shutil.copytree(source, candidate / relative)
        (candidate / "receipts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_root / "HYDRATION_RECEIPT.json", candidate / "receipts/HYDRATION_RECEIPT.json")
        candidate_remote = CANDIDATE_REMOTE_PREFIX + target_sha
        write_json(candidate / "CANDIDATE.json", {
            "schema": "RealSaS.IRISVNextFit8PretrainingCandidate.v1",
            "source_commit_sha": target_sha,
            "candidate_remote": candidate_remote,
            "quarantined_remote_consumed": False,
            "remote_persistence_completed": False,
            "remote_roundtrip_verified": False,
            "scientific_fit_started": False,
            "scientific_fit_pass_claimed": False,
            "training_authorized": False,
        })

        census = file_census(candidate)
        local_receipt = {
            "schema": LOCAL_RECEIPT_SCHEMA,
            "status": "LOCAL_CANDIDATE_BUILT__REMOTE_ROUNDTRIP_PENDING",
            "request_sha256": sha256_file(request_path),
            "target_source_sha": target_sha,
            "target_branch": ALLOWED_BRANCH,
            "candidate_remote_planned": candidate_remote,
            "candidate_file_count": len(census),
            "candidate_files": census,
            "teacher_batch_receipt_sha256": sha256_file(teachers / "TEACHER_BATCH_RECEIPT.json"),
            "cohort_manifest_sha256": sha256_file(prepared / "COHORT_MANIFEST.json"),
            "preparation_receipt_sha256": sha256_file(prepared / "PREPARATION_RECEIPT.json"),
            "relation_truth_v2_manifest_sha256": sha256_file(relation_manifest),
            "boundary_causality_v1_manifest_sha256": sha256_file(boundary_manifest),
            "migration_receipt_sha256": sha256_file(migration_receipt),
            "quarantine_audit_sha256": sha256_file(audit_report),
            "external_storage_written": False,
            "remote_roundtrip_verified": False,
            "final_materialization_receipt_minted": False,
            "training_steps_executed": 0,
            "scientific_fit_started": False,
            "training_authorized": False,
        }
        local_receipt_path = out / "evidence/LOCAL_CANDIDATE_BUILD_RECEIPT.json"
        write_json(local_receipt_path, local_receipt)

        result = {
            "schema": RESULT_SCHEMA,
            "status": "LOCAL_CANDIDATE_ARTIFACT_READY__DRIVE_PERSISTENCE_PENDING",
            "target_source_sha": target_sha,
            "target_branch": ALLOWED_BRANCH,
            "quarantine_status": audit["status"],
            "candidate_remote_planned": candidate_remote,
            "local_build_receipt_sha256": sha256_file(local_receipt_path),
            "external_storage_written": False,
            "remote_roundtrip_verified": False,
            "training_steps_executed": 0,
            "scientific_fit_started": False,
            "training_authorized": False,
        }
        write_json(out / "OPERATOR_RESULT.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        # Source bytes remain represented by target SHA; remove worktree so the
        # artifact contains only candidate/evidence, not a mutable source clone.
        run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
