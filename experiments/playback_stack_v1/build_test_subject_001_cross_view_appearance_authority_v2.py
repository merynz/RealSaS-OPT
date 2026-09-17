from __future__ import annotations

"""Build deterministic cross-view source appearance authority for TEST_SUBJECT_001.

V2 preserves the sealed v1 source-evidence rule exactly. A face is source-qualified
for a donor view only when the v1 first-hit/body-owner/alpha rule admits it there.
For each target view:
- use DIRECT_SOURCE when the target view itself qualifies the face;
- otherwise use OTHER_VIEW_SOURCE from the nearest camera-direction donor that
  independently qualifies the face;
- otherwise remain UNSEEN.

No completion, inpainting, learned synthesis, geometry deletion, or motion result is
allowed to participate in donor selection.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.playback_full_surface_v3 import qualify_camera_v3
from compiler.realsas_compiler_core.types import QualificationError
from experiments.playback_stack_v1 import build_test_subject_001_same_view_appearance_authority_v1 as v1
from experiments.playback_stack_v1.materialize_test_subject_001_v1 import VIEW_IDS


SCHEMA = "RealSaS.TestSubject001CrossViewAppearanceAuthorityBuilder.v2"
APPEARANCE_AUTHORITY_SCHEMA = "RealSaS.TestSubject001CrossViewAppearanceAuthority.v2"
APPEARANCE_POLICY = "TARGET_DIRECT_ELSE_NEAREST_QUALIFIED_OTHER_VIEW_SOURCE_ELSE_UNSEEN"
DONOR_RULE = "MAX_CAMERA_FORWARD_DOT_THEN_VIEW_INDEX__SOURCE_MUST_PASS_V1_DIRECT_RULE"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _camera_forward_by_view(cameras: dict[str, dict]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for view_index, view_id in enumerate(VIEW_IDS):
        camera = qualify_camera_v3(cameras[view_id], view_id=view_id, view_index=view_index)
        forward = np.asarray(camera.forward, dtype=np.float64)
        norm = float(np.linalg.norm(forward))
        if not np.isfinite(forward).all() or norm <= 1.0e-12:
            raise QualificationError(f"R2_APPEARANCE_DONOR_CAMERA_FORWARD_INVALID:{view_id}")
        out[view_id] = forward / norm
    return out


def _donor_order(target_view_id: str, forward_by_view: dict[str, np.ndarray]) -> tuple[str, ...]:
    target = forward_by_view[target_view_id]
    rows = []
    for donor_index, donor_view_id in enumerate(VIEW_IDS):
        if donor_view_id == target_view_id:
            continue
        score = float(np.dot(target, forward_by_view[donor_view_id]))
        rows.append((-score, donor_index, donor_view_id))
    rows.sort()
    return tuple(row[2] for row in rows)


def assign_cross_view_authority_v2(
    direct_faces_by_view: dict[str, tuple[int, ...]],
    cameras: dict[str, dict],
    *,
    face_count: int,
) -> tuple[dict[str, dict], dict]:
    """Pure deterministic donor assignment from already-qualified v1 evidence."""

    face_count = int(face_count)
    if face_count <= 0:
        raise QualificationError("R2_APPEARANCE_DONOR_FACE_COUNT_INVALID")
    if set(direct_faces_by_view) != set(VIEW_IDS):
        raise QualificationError("R2_APPEARANCE_DONOR_DIRECT_VIEW_SET_MISMATCH")

    direct_sets: dict[str, set[int]] = {}
    for view_id in VIEW_IDS:
        values = tuple(int(x) for x in direct_faces_by_view[view_id])
        if tuple(sorted(set(values))) != values:
            raise QualificationError(f"R2_APPEARANCE_DONOR_DIRECT_SET_NOT_SORTED_UNIQUE:{view_id}")
        if values and (values[0] < 0 or values[-1] >= face_count):
            raise QualificationError(f"R2_APPEARANCE_DONOR_DIRECT_FACE_OUT_OF_RANGE:{view_id}")
        direct_sets[view_id] = set(values)

    forward_by_view = _camera_forward_by_view(cameras)
    out: dict[str, dict] = {}
    total_other = 0
    total_unseen = 0
    for target_view_id in VIEW_IDS:
        direct = direct_sets[target_view_id]
        donor_groups: dict[str, list[int]] = {view_id: [] for view_id in VIEW_IDS if view_id != target_view_id}
        unseen: list[int] = []
        donor_order = _donor_order(target_view_id, forward_by_view)

        for face_index in range(face_count):
            if face_index in direct:
                continue
            selected = None
            for donor_view_id in donor_order:
                if face_index in direct_sets[donor_view_id]:
                    selected = donor_view_id
                    break
            if selected is None:
                unseen.append(face_index)
            else:
                donor_groups[selected].append(face_index)

        donor_groups = {k: v for k, v in donor_groups.items() if v}
        other_count = sum(len(v) for v in donor_groups.values())
        if len(direct) + other_count + len(unseen) != face_count:
            raise QualificationError(f"R2_APPEARANCE_DONOR_PARTITION_INCOMPLETE:{target_view_id}")
        total_other += other_count
        total_unseen += len(unseen)
        out[target_view_id] = {
            "direct_source_face_indices": sorted(direct),
            "other_view_source_face_indices_by_donor": donor_groups,
            "unseen_face_indices": unseen,
            "donor_priority": list(donor_order),
            "stats": {
                "direct_source_face_count": len(direct),
                "other_view_source_face_count": other_count,
                "unseen_face_count": len(unseen),
                "covered_source_face_count": len(direct) + other_count,
            },
        }

    summary = {
        "other_view_source_face_assignments_total": total_other,
        "unseen_face_assignments_total": total_unseen,
    }
    return out, summary


def build_authority(args) -> dict:
    zero_path = Path(args.zero_surface).resolve()
    camera_paths = tuple(Path(x).resolve() for x in args.cameras)
    source_paths = tuple(Path(x).resolve() for x in args.source_textures)
    owner_manifest = Path(args.owner_manifest).resolve()
    owner_paths = tuple(Path(x).resolve() for x in args.owner_rasters)
    out_path = Path(args.out).resolve()
    if str(args.nvdiffrast_commit) != v1.EXPECTED_NVDIFFRAST_COMMIT:
        raise RuntimeError("R2_APPEARANCE_NVDIFFRAST_PIN_DRIFT")

    vertices, faces = v1._load_zero_surface(zero_path)
    cameras = v1._load_cameras(camera_paths)
    alpha_masks, texture_hashes = v1._load_source_alpha(source_paths)
    owners, owner_hashes = v1._load_owner_rasters(owner_manifest, owner_paths)
    first_hit_by_view, calibration = v1.render_first_hit_face_ids_v1(vertices, faces, cameras)

    direct_faces_by_view: dict[str, tuple[int, ...]] = {}
    v1_stats: dict[str, dict] = {}
    for view_id in VIEW_IDS:
        direct, stats = v1.classify_direct_source_faces_v1(
            first_hit_by_view[view_id],
            owners[view_id],
            alpha_masks[view_id],
            face_count=len(faces),
        )
        if not direct:
            raise RuntimeError(f"R2_APPEARANCE_DIRECT_SOURCE_EMPTY:{view_id}")
        direct_faces_by_view[view_id] = direct
        v1_stats[view_id] = stats

    assigned, summary = assign_cross_view_authority_v2(
        direct_faces_by_view,
        cameras,
        face_count=len(faces),
    )

    views = {}
    for view_id in VIEW_IDS:
        row = assigned[view_id]
        row["source_texture_sha256"] = texture_hashes[view_id]
        row["owner_raster_sha256"] = owner_hashes[view_id]
        row["v1_source_evidence_stats"] = v1_stats[view_id]
        views[view_id] = row

    payload = {
        "schema": APPEARANCE_AUTHORITY_SCHEMA,
        "status": "PASS",
        "policy": APPEARANCE_POLICY,
        "builder_schema": SCHEMA,
        "source_evidence_rule": v1.DIRECT_RULE,
        "source_alpha_threshold": v1.SOURCE_ALPHA_THRESHOLD,
        "donor_rule": DONOR_RULE,
        "source_zero_surface_sha256": v1.EXPECTED_ZERO_SURFACE_SHA256,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "owner_manifest_sha256": _sha(owner_manifest),
        "nvdiffrast_commit": v1.EXPECTED_NVDIFFRAST_COMMIT,
        "rasterizer_calibration": calibration,
        "geometry_policy": "FULL_ZERO_SURFACE_PRESERVED__FIRST_HIT_USED_ONLY_FOR_APPEARANCE_PROVENANCE",
        "completion_used": False,
        "other_view_donor_used": bool(summary["other_view_source_face_assignments_total"]),
        "summary": summary,
        "views": views,
    }
    _write_json(out_path, payload)
    return {**payload, "out_path": str(out_path), "out_sha256": _sha(out_path)}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--source-textures", nargs=8, required=True)
    p.add_argument("--owner-manifest", required=True)
    p.add_argument("--owner-rasters", nargs=8, required=True)
    p.add_argument("--nvdiffrast-commit", default=v1.EXPECTED_NVDIFFRAST_COMMIT)
    p.add_argument("--out", required=True)
    return p


def main() -> None:
    report = build_authority(_parser().parse_args())
    print("TEST_SUBJECT_001_CROSS_VIEW_APPEARANCE_AUTHORITY_V2_PASS")
    print(json.dumps({
        "out_path": report["out_path"],
        "out_sha256": report["out_sha256"],
        "face_count": report["face_count"],
        "policy": report["policy"],
        "summary": report["summary"],
        "counts_by_view": {
            view_id: report["views"][view_id]["stats"]
            for view_id in VIEW_IDS
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
