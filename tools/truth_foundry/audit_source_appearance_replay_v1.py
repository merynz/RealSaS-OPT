#!/usr/bin/env python3
from __future__ import annotations

"""Audit a source-appearance policy replay against the sealed observation court.

The earlier revision used the valid but conservative Hausdorff triangle bound
``d(replay, admitted) + d(admitted, geometry)``.  That upper bound can exceed the
sealed 2 px policy even when the replay itself is within 2 px of source geometry.

This revision re-materializes the *same observation import lane* with
``blender_source_parity_export_v1.py`` and evaluates the replay directly with the
original admission's exact ``project_mask`` implementation and exact symmetric
Euclidean distance-transform metric.  The sealed 2 px policy is unchanged.
Replay-to-admitted distance and the old triangle bound remain diagnostics only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from tools.corpus.iris_observation_admission_v1 import project_mask

SCHEMA = "RealSaS.SourceAppearanceReplayEquivalenceCourt.v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mask(path: Path, resolution: int) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), np.uint8)
    if raw.size != resolution * resolution:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_MASK_LAYOUT:" + path.name)
    value = raw.reshape(resolution, resolution) > 0
    if not value.any():
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_EMPTY_MASK:" + path.name)
    return value


def _symmetric_gap(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or not a.any() or not b.any():
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_GAP_DOMAIN")
    return float(
        max(
            distance_transform_edt(~b)[a].max(),
            distance_transform_edt(~a)[b].max(),
        )
    )


def _observation_import_geometry(imported_path: Path, *, subject: str, admission: dict, candidate: dict):
    imported = _load(imported_path)
    receipt = dict(imported.get("import") or {})
    source_sha = str(admission.get("source_sha256") or "")
    if not source_sha or str(candidate.get("source_sha256") or "") != source_sha:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_SOURCE_LINEAGE_DRIFT:" + subject)
    if str(receipt.get("source_sha256") or "") != source_sha:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_IMPORTED_SOURCE_HASH_DRIFT:" + subject)

    sealed_import = dict(candidate.get("import") or {})
    for key in (
        "operation",
        "importer_options",
        "pose_policy",
        "raw_source_mutated",
        "excluded_import_controls",
    ):
        if receipt.get(key) != sealed_import.get(key):
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_IMPORT_POLICY_DRIFT:" + subject + ":" + key)

    vertices = np.asarray(imported.get("vertices"), dtype=np.float64)
    faces = np.asarray(imported.get("faces"), dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_IMPORTED_GEOMETRY_INVALID:" + subject)
    if int(admission.get("imported_vertex_count", -1)) != len(vertices):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_IMPORTED_VERTEX_COUNT_DRIFT:" + subject)
    if int(imported.get("bone_count", -1)) != int(admission.get("bone_count", -2)):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_IMPORTED_BONE_COUNT_DRIFT:" + subject)
    return vertices, faces, receipt


def audit(
    admission_path: Path,
    candidate_path: Path,
    camera_path: Path,
    imported_geometry_path: Path,
    replay_dir: Path,
) -> dict:
    admission = _load(admission_path)
    candidate = _load(candidate_path)
    camera_bundle = _load(camera_path)
    replay_path = replay_dir / "SOURCE_APPEARANCE_POLICY_REPLAY_V1.json"
    replay = _load(replay_path)
    if admission.get("schema") != "RealSaS.IrisBindObservationAdmission.v1" or admission.get("status") != "PASS_IRIS_BIND_OBSERVATIONS":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_ADMISSION_STATUS")
    if candidate.get("schema") != "RealSaS.Controlled8ViewSourceRestCandidate.v2":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CANDIDATE_SCHEMA")
    if camera_bundle.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CAMERA_SCHEMA")
    if replay.get("schema") != "RealSaS.SourceAppearancePolicyReplay.v1":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_RECEIPT_SCHEMA")
    subject = str(admission["subject_id"])
    if candidate.get("subject_id") != subject or replay.get("subject_id") != subject:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_SUBJECT_DRIFT")
    if _sha(candidate_path) != str(admission.get("candidate_sha256")) or _sha(camera_path) != str(admission.get("camera_sha256")):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_ADMISSION_LINEAGE_DRIFT")
    if replay.get("candidate_sha256") != _sha(candidate_path) or replay.get("camera_bundle_sha256") != _sha(camera_path):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_REPLAY_LINEAGE_DRIFT")
    if _sha(camera_path) != str(candidate.get("camera_bundle", {}).get("sha256")):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CANDIDATE_CAMERA_DRIFT")

    policy = dict(admission.get("policy") or {})
    if float(policy.get("maximum_mask_gap_px", -1)) != 2.0 or str(policy.get("raster_filter") or "") != "BOX_1PX":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_POLICY_DRIFT")
    resolution = int(policy.get("resolution", 0))
    if resolution <= 0 or int(candidate.get("resolution", 0)) != resolution:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_RESOLUTION_DRIFT")
    if candidate.get("source_lit_appearance_parity_status") != "PENDING_INDEPENDENT_REFERENCE":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_LIT_PARITY_BOUNDARY_DRIFT")

    vertices, faces, import_receipt = _observation_import_geometry(
        imported_geometry_path,
        subject=subject,
        admission=admission,
        candidate=candidate,
    )

    admitted_views = {int(v["view_index"]): v for v in admission.get("views") or ()}
    candidate_views = {int(v["view_index"]): v for v in candidate.get("views") or ()}
    replay_views = {int(v["view_index"]): v for v in replay.get("views") or ()}
    cameras = {int(v["view_index"]): v for v in camera_bundle.get("cameras") or ()}
    expected = set(range(8))
    if set(admitted_views) != expected or set(candidate_views) != expected or set(replay_views) != expected or set(cameras) != expected:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_VIEW_SET")

    root = admission_path.parent
    rows = []
    all_pass = True
    exact_count = 0
    max_replay_gap = 0.0
    max_direct_geometry_gap = 0.0
    max_admitted_import_geometry_gap = 0.0
    max_triangle_bound = 0.0
    xor_total = 0
    for index in range(8):
        av = admitted_views[index]
        cv = candidate_views[index]
        rv = replay_views[index]
        camera = cameras[index]
        admitted_mask_path = root / f"V{index}.mask.bin"
        admitted_png_path = root / f"V{index}.png"
        generated_mask_path = replay_dir / f"V{index}.mask.bin"
        generated_png_path = replay_dir / f"V{index}.png"
        if _sha(admitted_mask_path) != str(av["mask_sha256"]) or _sha(admitted_mask_path) != str(cv["foreground_mask"]["sha256"]):
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_ADMITTED_MASK_HASH_DRIFT")
        if _sha(admitted_png_path) != str(av["raster_sha256"]) or _sha(admitted_png_path) != str(cv["raster"]["sha256"]):
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_ADMITTED_RASTER_HASH_DRIFT")
        if _sha(generated_mask_path) != str(rv["mask_sha256"]) or _sha(generated_png_path) != str(rv["raster_sha256"]):
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_GENERATED_HASH_DRIFT")

        admitted = _mask(admitted_mask_path, resolution)
        generated = _mask(generated_mask_path, resolution)
        generated_png_alpha = np.asarray(Image.open(generated_png_path).convert("RGBA"))[:, :, 3] > 0
        if not np.array_equal(generated_png_alpha, generated):
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_GENERATED_ALPHA_MASK_DRIFT")

        # Re-run the exact admission projection on the same rest-import geometry
        # used by the observation lane.  This is the authority metric.
        reference = project_mask(vertices, faces, camera)
        if reference.shape != generated.shape or not reference.any():
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_REFERENCE_MASK_INVALID")
        direct_geometry_gap = _symmetric_gap(generated, reference)
        admitted_import_geometry_gap = _symmetric_gap(admitted, reference)
        passed = direct_geometry_gap <= float(policy["maximum_mask_gap_px"]) + 1.0e-12

        # Diagnostics only.  The old triangle upper bound is deliberately not
        # used as the gate now that direct geometry measurement is available.
        replay_gap = _symmetric_gap(generated, admitted)
        sealed_raw_geometry_gap = float(av["maximum_mask_gap_px"])
        triangle_bound = replay_gap + sealed_raw_geometry_gap
        exact = bool(np.array_equal(generated, admitted))
        xor = int(np.count_nonzero(generated ^ admitted))

        all_pass = all_pass and passed
        exact_count += int(exact)
        xor_total += xor
        max_replay_gap = max(max_replay_gap, replay_gap)
        max_direct_geometry_gap = max(max_direct_geometry_gap, direct_geometry_gap)
        max_admitted_import_geometry_gap = max(max_admitted_import_geometry_gap, admitted_import_geometry_gap)
        max_triangle_bound = max(max_triangle_bound, triangle_bound)
        rows.append(
            {
                "view_index": index,
                "admitted_foreground_pixel_count": int(admitted.sum()),
                "replay_foreground_pixel_count": int(generated.sum()),
                "import_geometry_reference_pixel_count": int(reference.sum()),
                "exact_mask_equal": exact,
                "xor_pixel_count": xor,
                "replay_to_admitted_hausdorff_px": replay_gap,
                "sealed_admitted_to_raw_geometry_hausdorff_px": sealed_raw_geometry_gap,
                "diagnostic_triangle_inequality_upper_bound_px": triangle_bound,
                "admitted_to_observation_import_geometry_hausdorff_px": admitted_import_geometry_gap,
                "direct_replay_to_observation_import_geometry_hausdorff_px": direct_geometry_gap,
                "sealed_maximum_mask_gap_px": float(policy["maximum_mask_gap_px"]),
                "passes_sealed_direct_geometry_court": passed,
            }
        )

    for key in (
        "source_lit_appearance_parity_claimed",
        "rgb_equivalence_claimed",
        "sampler_qualified",
        "qualification_minted",
        "student_target_allowed",
        "training_allowed",
    ):
        if replay.get(key) is not False:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_AUTHORITY_LEAK:" + key)
    return {
        "schema": SCHEMA,
        "subject_id": subject,
        "admission_sha256": _sha(admission_path),
        "candidate_sha256": _sha(candidate_path),
        "camera_bundle_sha256": _sha(camera_path),
        "observation_import_geometry_sha256": _sha(imported_geometry_path),
        "observation_import_operation": import_receipt.get("operation"),
        "replay_receipt_sha256": _sha(replay_path),
        "view_count": 8,
        "exact_mask_equal_view_count": exact_count,
        "xor_pixel_count_total": xor_total,
        "maximum_replay_to_admitted_hausdorff_px": max_replay_gap,
        "maximum_admitted_to_observation_import_geometry_hausdorff_px": max_admitted_import_geometry_gap,
        "maximum_direct_replay_to_observation_import_geometry_hausdorff_px": max_direct_geometry_gap,
        "maximum_diagnostic_triangle_upper_bound_px": max_triangle_bound,
        "sealed_maximum_mask_gap_px": float(policy["maximum_mask_gap_px"]),
        "authority_metric": "DIRECT_REPLAY_TO_OBSERVATION_IMPORT_GEOMETRY__ORIGINAL_PROJECT_MASK_AND_SYMMETRIC_EDT",
        "triangle_upper_bound_is_authority": False,
        "source_opacity_policy_replay_equivalent": bool(all_pass),
        "source_lit_appearance_parity_claimed": False,
        "rgb_equivalence_claimed": False,
        "effective_alpha_value_equivalence_claimed": False,
        "sampler_qualified": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "court_status": "PASS_SOURCE_APPEARANCE_MASK_EQUIVALENCE" if all_pass else "FAIL_SOURCE_APPEARANCE_MASK_EQUIVALENCE",
        "views": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--admission", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--camera-bundle", required=True, type=Path)
    p.add_argument("--imported-geometry", required=True, type=Path)
    p.add_argument("--replay-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = audit(a.admission, a.candidate, a.camera_bundle, a.imported_geometry, a.replay_dir)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "views"}, sort_keys=True))
    if not result["source_opacity_policy_replay_equivalent"]:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_EQUIVALENCE_FAILED")


if __name__ == "__main__":
    main()
