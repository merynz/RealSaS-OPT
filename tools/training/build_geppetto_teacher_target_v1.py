from __future__ import annotations

"""Build an anonymous Geppetto mechanical-core FIT target from sealed FBX teacher arrays."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.geppetto_reference_strength_fullstack_v1.mechanical_core_target_v1 import (
    build_mechanical_core_target_v1,
    target_content_sha256_v1,
    world_heads_from_rest_world_source_v1,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-source-npz", required=True)
    parser.add_argument("--teacher-source-audit", required=True)
    parser.add_argument("--normalization-domain", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    source_npz = Path(args.teacher_source_npz).expanduser().resolve()
    source_audit_path = Path(args.teacher_source_audit).expanduser().resolve()
    normalization_path = Path(args.normalization_domain).expanduser().resolve()
    out_npz = Path(args.output_npz).expanduser().resolve()
    out_report = Path(args.output_report).expanduser().resolve()

    source_audit = _load_json(source_audit_path)
    if source_audit.get("schema") != "RealSaS.KnightGeppettoTeacherSourceExtraction.v1":
        raise RuntimeError("GEPPETTO_TEACHER_SOURCE_AUDIT_SCHEMA_DRIFT")
    if source_audit.get("status") != "PASS":
        raise RuntimeError("GEPPETTO_TEACHER_SOURCE_AUDIT_NOT_PASS")
    if source_audit.get("product_authority_claimed") is not False:
        raise RuntimeError("GEPPETTO_TEACHER_SOURCE_PRODUCT_AUTHORITY_FORBIDDEN")
    if str(source_audit["output_npz"]["sha256"]) != _sha256(source_npz):
        raise RuntimeError("GEPPETTO_TEACHER_SOURCE_NPZ_SHA_DRIFT")

    norm = _load_json(normalization_path)
    if str(norm.get("schema_version") or norm.get("schema") or "") != "RealSaS.NormalizationDomainIR.v1":
        raise RuntimeError("GEPPETTO_TEACHER_NORMALIZATION_SCHEMA_DRIFT")
    center = np.asarray(norm["center_xyz"], dtype=np.float64)
    half = float(norm["half_extent"])
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(half) or half <= 0:
        raise RuntimeError("GEPPETTO_TEACHER_NORMALIZATION_INVALID")

    with np.load(source_npz, allow_pickle=False) as data:
        required = {"parents", "deform_mask", "skin", "rest_world_source"}
        if not required.issubset(set(data.files)):
            raise RuntimeError(
                "GEPPETTO_TEACHER_SOURCE_ARRAYS_MISSING:"
                + ",".join(sorted(required - set(data.files)))
            )
        parents = np.asarray(data["parents"], dtype=np.int64)
        deform = np.asarray(data["deform_mask"], dtype=np.uint8).astype(bool)
        skin = np.asarray(data["skin"], dtype=np.float32)
        rest = np.asarray(data["rest_world_source"], dtype=np.float64)

    heads_world = world_heads_from_rest_world_source_v1(rest)
    target = build_mechanical_core_target_v1(
        parents=parents,
        deform_mask=deform,
        skin=skin,
        bone_heads_world=heads_world,
    )
    target_hash = target_content_sha256_v1(target)
    normalized = (
        (np.asarray(target.positions_world, dtype=np.float64) - center[None, :]) / half
    ).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise RuntimeError("GEPPETTO_TEACHER_TARGET_NORMALIZED_NONFINITE")
    maximum_abs = float(np.max(np.abs(normalized)))
    if maximum_abs > 1.0 + 1e-6:
        raise RuntimeError(
            f"GEPPETTO_TEACHER_TARGET_OUTSIDE_FROZEN_NORMALIZATION:{maximum_abs}"
        )

    selected_source = np.asarray(target.source_indices_provenance_only, dtype=np.int64)
    selected_deform = deform[selected_source]
    selected_skin_mass = np.asarray(target.skin_mass, dtype=np.float64)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        positions_world=np.asarray(target.positions_world, dtype=np.float32),
        positions_normalized=normalized,
        parent_indices=np.asarray(target.parent_indices, dtype=np.int64),
        root_mask=np.asarray(target.root_mask, dtype=np.uint8),
        source_indices_provenance_only=selected_source,
        skin_mass=selected_skin_mass,
    )
    out_sha = _sha256(out_npz)

    report = {
        "schema": "RealSaS.KnightGeppettoMechanicalCoreTarget.v1",
        "status": "PASS",
        "authority_class": "TRAINING_EVALUATION_ONLY",
        "teacher_source_npz_sha256": _sha256(source_npz),
        "teacher_source_audit_sha256": _sha256(source_audit_path),
        "source_fbx_sha256": str(source_audit["source_fbx"]["sha256"]),
        "normalization_domain": {
            "path": str(normalization_path),
            "sha256": _sha256(normalization_path),
            "normalization_hash": str(norm["normalization_hash"]),
            "center_xyz": center.tolist(),
            "half_extent": half,
        },
        "mechanical_core_rule": str(target.rule),
        "position_authority": str(target.position_authority),
        "target_content_sha256": target_hash,
        "target_count": int(target.count),
        "root_count": int(np.count_nonzero(target.root_mask)),
        "selected_skin_supported_count": int(
            np.count_nonzero(selected_skin_mass > 1e-8)
        ),
        "selected_non_deform_bridge_count": int(
            np.count_nonzero(~selected_deform)
        ),
        "max_abs_normalized_target_coordinate": maximum_abs,
        "output_npz": {
            "path": str(out_npz),
            "sha256": out_sha,
        },
        "teacher_supervision_during_training_allowed": True,
        "teacher_feedback_during_free_running_inference_allowed": False,
        "source_indices_are_provenance_only": True,
        "source_names_exported": False,
        "canonical_joint_ids_created": False,
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("GEPPETTO_KNIGHT_MECHANICAL_CORE_TARGET_PASS")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
