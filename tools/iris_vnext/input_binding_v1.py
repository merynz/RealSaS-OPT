from __future__ import annotations

"""Fail-closed input binding for IRIS vNext Fit8 preparation.

This module is intentionally independent of training code.  It proves that the
Stage01 bundle/extraction and the frozen legacy IRIS evidence shards are exactly
the inputs named by canonical/VNEXT_FIT8_INPUT_BINDING_V1.json.  It never admits
legacy optimizer/checkpoint state.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


SUBJECT_IDS = ("Snake", "Bat", "T-Rex", "Horse", "Spider", "GreenBlob", "Shark", "George")


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_binding_v1(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "RealSaS.VNextFit8InputBinding.v1":
        raise ValueError("VNEXT_INPUT_BINDING_SCHEMA_INVALID")
    if value.get("legacy_optimizer_state_reused") is not False:
        raise ValueError("VNEXT_INPUT_BINDING_LEGACY_OPTIMIZER_FORBIDDEN")
    subjects = tuple(value.get("subjects") or ())
    if tuple(str(row.get("subject_id")) for row in subjects) != SUBJECT_IDS:
        raise ValueError("VNEXT_INPUT_BINDING_SUBJECT_ORDER_DRIFT")
    if len({str(row.get("legacy_subject_id")) for row in subjects}) != 8:
        raise ValueError("VNEXT_INPUT_BINDING_LEGACY_SUBJECT_ID_DRIFT")
    for row in subjects:
        for key in (
            "legacy_shard_sha256",
            "raw_source_sha256",
            "observation_admission_sha256",
            "camera_bundle_sha256",
            "source_rest_candidate_sha256",
        ):
            digest = str(row.get(key) or "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"VNEXT_INPUT_BINDING_HASH_INVALID:{row['subject_id']}:{key}")
    return value


def _require_hash(path: Path, expected: str, code: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{code}_MISSING:{path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{code}_HASH_DRIFT:{path}:{actual}")
    return actual


def verify_stage01_v1(
    binding: dict[str, Any],
    *,
    stage01_bundle: str | Path,
    stage01_root: str | Path,
) -> dict[str, Any]:
    bundle = Path(stage01_bundle).resolve()
    root = Path(stage01_root).resolve()
    bundle_expected = str(binding["stage01_bundle"]["sha256"])
    bundle_hash = _require_hash(bundle, bundle_expected, "VNEXT_STAGE01_BUNDLE")
    bootstrap_path = root / "BOOTSTRAP_MANIFEST.json"
    if not bootstrap_path.is_file():
        raise FileNotFoundError("VNEXT_STAGE01_BOOTSTRAP_MISSING")
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap_rows = {str(row["subject_id"]): row for row in bootstrap.get("subjects") or ()}

    subject_receipts = []
    for row in binding["subjects"]:
        legacy_id = str(row["legacy_subject_id"])
        if legacy_id not in bootstrap_rows:
            raise ValueError(f"VNEXT_STAGE01_SUBJECT_MISSING:{legacy_id}")
        boot = bootstrap_rows[legacy_id]
        raw = root / str(row["raw_source_path_in_stage01"])
        admission = root / str(row["observation_admission_path_in_stage01"])
        camera = root / str(row["camera_bundle_path_in_stage01"])
        rest = root / str(row["source_rest_candidate_path_in_stage01"])
        checks = {
            "raw_source": _require_hash(raw, row["raw_source_sha256"], "VNEXT_STAGE01_RAW_SOURCE"),
            "observation_admission": _require_hash(
                admission, row["observation_admission_sha256"], "VNEXT_STAGE01_ADMISSION"
            ),
            "camera_bundle": _require_hash(camera, row["camera_bundle_sha256"], "VNEXT_STAGE01_CAMERA"),
            "source_rest_candidate": _require_hash(
                rest, row["source_rest_candidate_sha256"], "VNEXT_STAGE01_REST_EVIDENCE"
            ),
        }
        if str(boot.get("raw_source", {}).get("sha256")) != row["raw_source_sha256"]:
            raise ValueError(f"VNEXT_STAGE01_BOOTSTRAP_RAW_HASH_DRIFT:{legacy_id}")
        if str(boot.get("display_name")) != row["subject_id"]:
            raise ValueError(f"VNEXT_STAGE01_DISPLAY_NAME_DRIFT:{legacy_id}")
        subject_receipts.append(
            {"subject_id": row["subject_id"], "legacy_subject_id": legacy_id, **checks}
        )
    return {
        "stage01_bundle_sha256": bundle_hash,
        "stage01_bootstrap_sha256": sha256_file(bootstrap_path),
        "subjects": subject_receipts,
    }


def verify_legacy_evidence_v1(
    binding: dict[str, Any],
    *,
    legacy_shard_dir: str | Path,
    legacy_cohort_manifest: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(legacy_shard_dir).resolve()
    manifest_hash = None
    if legacy_cohort_manifest is not None:
        manifest = Path(legacy_cohort_manifest).resolve()
        manifest_hash = _require_hash(
            manifest,
            str(binding["legacy_iris_cohort_manifest"]["sha256"]),
            "VNEXT_LEGACY_COHORT_MANIFEST",
        )
    rows = []
    for row in binding["subjects"]:
        path = root / str(row["legacy_shard_file"])
        digest = _require_hash(path, row["legacy_shard_sha256"], "VNEXT_LEGACY_SHARD")
        rows.append(
            {
                "subject_id": row["subject_id"],
                "legacy_subject_id": row["legacy_subject_id"],
                "shard_file": path.name,
                "sha256": digest,
            }
        )
    return {"legacy_cohort_manifest_sha256": manifest_hash, "subjects": rows}


def build_input_receipt_v1(
    *,
    binding_path: str | Path,
    stage01_bundle: str | Path,
    stage01_root: str | Path,
    legacy_shard_dir: str | Path,
    legacy_cohort_manifest: str | Path | None = None,
) -> dict[str, Any]:
    binding_path = Path(binding_path).resolve()
    binding = load_binding_v1(binding_path)
    return {
        "schema": "RealSaS.VNextFit8VerifiedInputs.v1",
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "stage01": verify_stage01_v1(
            binding, stage01_bundle=stage01_bundle, stage01_root=stage01_root
        ),
        "legacy_evidence": verify_legacy_evidence_v1(
            binding,
            legacy_shard_dir=legacy_shard_dir,
            legacy_cohort_manifest=legacy_cohort_manifest,
        ),
        "legacy_optimizer_state_reused": False,
        "legacy_5400_checkpoint_reused": False,
        "teacher_inputs_in_inference_forward": False,
    }
