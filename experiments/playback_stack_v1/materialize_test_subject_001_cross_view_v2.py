from __future__ import annotations

"""Materialize TEST_SUBJECT_001 with deterministic cross-view source donors.

This wrapper deliberately leaves the sealed SAME_VIEW v1 fixture untouched. It
reuses the exact v1 geometry/motion/texture materializer and replaces only the
appearance-authority loader/binder with the v2 donor contract.
"""

import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    FaceAppearanceAuthorityV3,
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import AppearanceProvenance
from compiler.realsas_compiler_core.types import QualificationError
from experiments.playback_stack_v1 import materialize_test_subject_001_v1 as base
from experiments.playback_stack_v1.build_test_subject_001_cross_view_appearance_authority_v2 import (
    APPEARANCE_AUTHORITY_SCHEMA,
    APPEARANCE_POLICY,
)


SCHEMA = "RealSaS.TestSubject001PlaybackV3Materialization.CrossViewDonor.v2"
SCOPE = "BODY_ONLY__FULL_DENSE_ZERO_SURFACE__D1_3D_MOTION__CROSS_VIEW_SOURCE_APPEARANCE"


def _load_cross_view_appearance_authority(
    path: Path,
    *,
    face_count: int,
    source_texture_rows: dict[str, dict],
):
    raw = base._load_json(path)
    if raw.get("schema") != APPEARANCE_AUTHORITY_SCHEMA:
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_AUTHORITY_SCHEMA_DRIFT")
    if raw.get("status") != "PASS":
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_AUTHORITY_NOT_PASS")
    if raw.get("policy") != APPEARANCE_POLICY:
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_AUTHORITY_POLICY_DRIFT")
    if raw.get("source_zero_surface_sha256") != base.EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_ZERO_SURFACE_DRIFT")
    if int(raw.get("face_count", -1)) != int(face_count):
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_FACE_COUNT_DRIFT")
    if bool(raw.get("completion_used")):
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_COMPLETION_FORBIDDEN")

    views = raw.get("views")
    if not isinstance(views, dict) or set(views) != set(base.VIEW_IDS):
        raise RuntimeError("TEST_SUBJECT_001_APPEARANCE_VIEW_SET_DRIFT")

    parsed: dict[str, dict] = {}
    direct_sets: dict[str, set[int]] = {}
    for view_id in base.VIEW_IDS:
        row = views[view_id]
        if not isinstance(row, dict):
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_VIEW_ROW_INVALID:{view_id}")
        if row.get("source_texture_sha256") != source_texture_rows[view_id]["sha256"]:
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_TEXTURE_SHA_DRIFT:{view_id}")
        direct = tuple(int(x) for x in (row.get("direct_source_face_indices") or ()))
        if tuple(sorted(set(direct))) != direct:
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_DIRECT_SET_NOT_SORTED_UNIQUE:{view_id}")
        if direct and (direct[0] < 0 or direct[-1] >= face_count):
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_FACE_OUT_OF_RANGE:{view_id}")
        if not direct:
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_DIRECT_SOURCE_EMPTY:{view_id}")
        donor_groups_raw = row.get("other_view_source_face_indices_by_donor") or {}
        if not isinstance(donor_groups_raw, dict):
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_DONOR_GROUP_INVALID:{view_id}")
        donor_groups: dict[str, tuple[int, ...]] = {}
        assigned = set(direct)
        for donor_view_id, values_raw in donor_groups_raw.items():
            donor_view_id = str(donor_view_id)
            if donor_view_id not in base.VIEW_IDS or donor_view_id == view_id:
                raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_DONOR_VIEW_INVALID:{view_id}:{donor_view_id}")
            values = tuple(int(x) for x in values_raw)
            if tuple(sorted(set(values))) != values:
                raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_DONOR_SET_NOT_SORTED_UNIQUE:{view_id}:{donor_view_id}")
            if values and (values[0] < 0 or values[-1] >= face_count):
                raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_DONOR_FACE_OUT_OF_RANGE:{view_id}:{donor_view_id}")
            overlap = assigned.intersection(values)
            if overlap:
                raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_PARTITION_OVERLAP:{view_id}:{min(overlap)}")
            assigned.update(values)
            donor_groups[donor_view_id] = values
        unseen = tuple(int(x) for x in (row.get("unseen_face_indices") or ()))
        if tuple(sorted(set(unseen))) != unseen:
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_UNSEEN_SET_NOT_SORTED_UNIQUE:{view_id}")
        if assigned.intersection(unseen):
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_UNSEEN_OVERLAP:{view_id}")
        if assigned.union(unseen) != set(range(face_count)):
            raise RuntimeError(f"TEST_SUBJECT_001_APPEARANCE_PARTITION_INCOMPLETE:{view_id}")
        direct_sets[view_id] = set(direct)
        parsed[view_id] = {"direct": direct, "donor_groups": donor_groups, "unseen": unseen}

    # The donor view must independently satisfy the frozen v1 source-evidence rule.
    for target_view_id in base.VIEW_IDS:
        for donor_view_id, values in parsed[target_view_id]["donor_groups"].items():
            missing = [face_index for face_index in values if face_index not in direct_sets[donor_view_id]]
            if missing:
                raise RuntimeError(
                    f"TEST_SUBJECT_001_DONOR_NOT_SOURCE_QUALIFIED:{target_view_id}:{donor_view_id}:{missing[0]}"
                )
    return parsed, base._sha(path)


def _build_uv_and_face_authorities_v2(dense_vertices, dense_faces, cameras, authority_by_target):
    face_count = len(dense_faces)
    raw_uv_by_view: dict[str, np.ndarray] = {}
    stored_uv_by_view: dict[str, np.ndarray] = {}

    for view_index, view_id in enumerate(base.VIEW_IDS):
        camera = qualify_camera_v3(cameras[view_id], view_id=view_id, view_index=view_index)
        xyz = project_points_xyz_v3(dense_vertices, camera)
        raw_uv, stored_uv = base.source_pixel_center_uv_v1(xyz, resolution=camera.resolution)
        raw_uv_by_view[view_id] = raw_uv
        stored_uv_by_view[view_id] = np.ascontiguousarray(stored_uv, dtype=np.float64)

    out_authority: dict[str, tuple[FaceAppearanceAuthorityV3, ...]] = {}
    counts: dict[str, dict] = {}
    eps = 1.0e-9
    for target_index, target_view_id in enumerate(base.VIEW_IDS):
        row = authority_by_target[target_view_id]
        direct_set = set(row["direct"])
        donor_by_face: dict[int, str] = {}
        for donor_view_id, face_indices in row["donor_groups"].items():
            for face_index in face_indices:
                donor_by_face[int(face_index)] = donor_view_id

        authorities = []
        for face_index in range(face_count):
            if face_index in direct_set:
                donor_view_id = target_view_id
                provenance = AppearanceProvenance.DIRECT_SOURCE
                donor_index = target_index
            elif face_index in donor_by_face:
                donor_view_id = donor_by_face[face_index]
                provenance = AppearanceProvenance.OTHER_VIEW_SOURCE
                donor_index = base.VIEW_IDS.index(donor_view_id)
            else:
                authorities.append(FaceAppearanceAuthorityV3(AppearanceProvenance.UNSEEN, None, None))
                continue

            donor_vertices = np.asarray(dense_faces[face_index], dtype=np.int64)
            uv = raw_uv_by_view[donor_view_id][donor_vertices]
            if np.any(uv < -eps) or np.any(uv > 1.0 + eps):
                raise RuntimeError(
                    f"TEST_SUBJECT_001_SOURCEBOUND_UV_OUT_OF_BOUNDS:{target_view_id}:{face_index}:{donor_view_id}"
                )
            authorities.append(
                FaceAppearanceAuthorityV3(
                    provenance,
                    donor_index,
                    f"SOURCE_TEXTURE:{donor_view_id}",
                )
            )

        out_authority[target_view_id] = tuple(authorities)
        other_count = len(donor_by_face)
        counts[target_view_id] = {
            "direct_source_faces": len(direct_set),
            "other_view_source_faces": other_count,
            "source_bound_faces": len(direct_set) + other_count,
            "unseen_faces": face_count - len(direct_set) - other_count,
            "other_view_source_faces_by_donor": {
                donor_view_id: len(face_indices)
                for donor_view_id, face_indices in row["donor_groups"].items()
            },
        }
    return stored_uv_by_view, out_authority, counts


def materialize(args) -> dict:
    # Patch only the v1 fixture extension points for the duration of this call.
    old_loader = base._load_same_view_appearance_authority
    old_builder = base._build_uv_and_face_authorities
    old_schema = base.SCHEMA
    old_authority_schema = base.APPEARANCE_AUTHORITY_SCHEMA
    old_policy = base.APPEARANCE_POLICY
    try:
        base._load_same_view_appearance_authority = _load_cross_view_appearance_authority
        base._build_uv_and_face_authorities = _build_uv_and_face_authorities_v2
        base.SCHEMA = SCHEMA
        base.APPEARANCE_AUTHORITY_SCHEMA = APPEARANCE_AUTHORITY_SCHEMA
        base.APPEARANCE_POLICY = APPEARANCE_POLICY
        report = base.materialize(args)
    finally:
        base._load_same_view_appearance_authority = old_loader
        base._build_uv_and_face_authorities = old_builder
        base.SCHEMA = old_schema
        base.APPEARANCE_AUTHORITY_SCHEMA = old_authority_schema
        base.APPEARANCE_POLICY = old_policy

    report["scope"] = SCOPE
    report["appearance_policy"] = APPEARANCE_POLICY
    report["other_view_donor_used"] = any(
        int(row.get("other_view_source_faces", 0)) > 0
        for row in report["appearance_counts_by_view"].values()
    )
    report_path = Path(report["report_path"])
    base._write_json(report_path, {k: v for k, v in report.items() if k != "report_path"})
    return report


def main() -> None:
    report = materialize(base._parser().parse_args())
    print("TEST_SUBJECT_001_RUNTIME_V3_CROSS_VIEW_V2_MATERIALIZED")
    print(json.dumps({
        "archive_path": report["package"]["archive_path"],
        "archive_sha256": report["package"]["archive_sha256"],
        "report_path": report["report_path"],
        "playback_contract_hash": report["playback_contract_hash"],
        "appearance_policy": report["appearance_policy"],
        "other_view_donor_used": report["other_view_donor_used"],
        "runtime_qualified": report["runtime_qualified"],
        "founder_visual_pass_claimed": report["founder_visual_pass_claimed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
