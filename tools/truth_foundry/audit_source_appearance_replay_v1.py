#!/usr/bin/env python3
from __future__ import annotations

"""Audit sealed source-opacity policy transport onto the packed teacher surface.

Two different invariants must not be conflated:

1. The admitted source-observation lane already proves that its rendered
   foreground support agrees with independently measured raw-source geometry
   within the sealed FIT8 mask-gap policy.
2. This court asks whether replaying that exact sealed material/opacity policy on
   the packed Truth Foundry evaluator preserves the same foreground-support
   property on the *packed teacher geometry actually being evaluated*.

Packed-teacher geometry need not equal the observation-import geometry for this
appearance-policy transport claim.  That cross-lane geometry relation remains an
explicit diagnostic and cannot be silently upgraded by this court.

The court still claims no source-lit RGB parity, no effective alpha-value parity,
no qualified sampler, no student target, no training authority and no product
activation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from tools.corpus.iris_observation_admission_v1 import project_mask

SCHEMA = "RealSaS.SourceAppearancePolicyTransportCourt.v1"


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


def _geometry(value: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(value.get("vertices"), dtype=np.float64)
    faces = np.asarray(value.get("faces"), dtype=np.int64)
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
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_GEOMETRY_INVALID:" + label)
    return vertices, faces


def _observation_import_geometry(
    imported_path: Path, *, subject: str, admission: dict, candidate: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
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

    vertices, faces = _geometry(imported, "OBSERVATION_IMPORT:" + subject)
    if int(admission.get("imported_vertex_count", -1)) != len(vertices):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_IMPORTED_VERTEX_COUNT_DRIFT:" + subject)
    if int(imported.get("bone_count", -1)) != int(admission.get("bone_count", -2)):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_IMPORTED_BONE_COUNT_DRIFT:" + subject)
    return vertices, faces, receipt


def _packed_geometry(
    packed_path: Path, *, subject: str, replay: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    packed = _load(packed_path)
    if packed.get("schema") != "RealSaS.PackedRestGeometryDiagnostic.v1":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_PACKED_GEOMETRY_SCHEMA:" + subject)
    if packed.get("subject_id") != subject:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_PACKED_GEOMETRY_SUBJECT:" + subject)
    if packed.get("mesh_selection") != "NATIVE_TEACHER_RENDERABLE_SOURCE_MESHES__CUSTOM_SHAPE_HELPERS_EXCLUDED":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_PACKED_GEOMETRY_SELECTION:" + subject)
    if packed.get("packed_source_scene_blend_sha256") != replay.get("packed_source_scene_blend_sha256"):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_PACKED_GEOMETRY_BLEND_DRIFT:" + subject)
    if (
        packed.get("qualification_minted") is not False
        or packed.get("training_allowed") is not False
        or packed.get("product_pass_claimed") is not False
    ):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_PACKED_GEOMETRY_AUTHORITY_LEAK:" + subject)
    vertices, faces = _geometry(packed, "PACKED_TEACHER:" + subject)
    return vertices, faces, packed


def _validate_material_policy_transport(candidate: dict, replay: dict, subject: str) -> None:
    expected_rows = list(candidate.get("material_transform") or ())
    actual_rows = list(replay.get("material_replay") or ())
    expected = {str(row.get("material") or ""): row for row in expected_rows}
    actual = {str(row.get("material") or ""): row for row in actual_rows}
    if not expected or "" in expected or len(expected) != len(expected_rows):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_SEALED_MATERIAL_SET_INVALID:" + subject)
    if "" in actual or len(actual) != len(actual_rows) or set(actual) != set(expected):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_MATERIAL_SET_DRIFT:" + subject)
    for name, sealed in expected.items():
        transported = actual[name]
        if transported.get("operation") != sealed.get("operation") or transported.get("replayed") is not True:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_MATERIAL_OPERATION_DRIFT:" + subject + ":" + name)
        if sealed.get("operation") == "BSDF_LIGHT_RESPONSE_TO_EMISSION__ALPHA_GRAPH_RETAINED":
            if transported.get("alpha_policy") != sealed.get("alpha_policy"):
                raise RuntimeError("SOURCE_APPEARANCE_REPLAY_ALPHA_POLICY_DRIFT:" + subject + ":" + name)
        elif sealed.get("operation") != "NATIVE_EMISSION_GRAPH_PRESERVED":
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_MATERIAL_OPERATION_UNSUPPORTED:" + subject + ":" + name)


def audit(
    admission_path: Path,
    candidate_path: Path,
    camera_path: Path,
    imported_geometry_path: Path,
    packed_geometry_path: Path,
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
    _validate_material_policy_transport(candidate, replay, subject)

    policy = dict(admission.get("policy") or {})
    maximum_gap = float(policy.get("maximum_mask_gap_px", -1))
    if maximum_gap != 2.0 or str(policy.get("raster_filter") or "") != "BOX_1PX":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_POLICY_DRIFT")
    resolution = int(policy.get("resolution", 0))
    if resolution <= 0 or int(candidate.get("resolution", 0)) != resolution:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_RESOLUTION_DRIFT")
    if candidate.get("source_lit_appearance_parity_status") != "PENDING_INDEPENDENT_REFERENCE":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_LIT_PARITY_BOUNDARY_DRIFT")

    observation_vertices, observation_faces, import_receipt = _observation_import_geometry(
        imported_geometry_path,
        subject=subject,
        admission=admission,
        candidate=candidate,
    )
    packed_vertices, packed_faces, packed_receipt = _packed_geometry(
        packed_geometry_path,
        subject=subject,
        replay=replay,
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
    xor_total = 0
    max_replay_admitted_gap = 0.0
    max_replay_packed_gap = 0.0
    max_packed_observation_gap = 0.0
    max_admitted_observation_gap = 0.0
    max_direct_observation_gap = 0.0
    max_triangle_bound = 0.0
    geometry_lane_equivalent = True
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

        observation_reference = project_mask(observation_vertices, observation_faces, camera)
        packed_reference = project_mask(packed_vertices, packed_faces, camera)
        if (
            observation_reference.shape != generated.shape
            or packed_reference.shape != generated.shape
            or not observation_reference.any()
            or not packed_reference.any()
        ):
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_REFERENCE_MASK_INVALID")

        sealed_source_gap = float(av["maximum_mask_gap_px"])
        if not np.isfinite(sealed_source_gap) or sealed_source_gap > maximum_gap + 1.0e-12:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_SOURCE_SUPPORT_ADMISSION_DRIFT")
        replay_packed_gap = _symmetric_gap(generated, packed_reference)
        passed = replay_packed_gap <= maximum_gap + 1.0e-12

        replay_admitted_gap = _symmetric_gap(generated, admitted)
        packed_observation_gap = _symmetric_gap(packed_reference, observation_reference)
        admitted_observation_gap = _symmetric_gap(admitted, observation_reference)
        direct_observation_gap = _symmetric_gap(generated, observation_reference)
        triangle_bound = replay_admitted_gap + sealed_source_gap
        exact = bool(np.array_equal(generated, admitted))
        xor = int(np.count_nonzero(generated ^ admitted))
        view_geometry_lane_equivalent = packed_observation_gap <= maximum_gap + 1.0e-12

        all_pass = all_pass and passed
        geometry_lane_equivalent = geometry_lane_equivalent and view_geometry_lane_equivalent
        exact_count += int(exact)
        xor_total += xor
        max_replay_admitted_gap = max(max_replay_admitted_gap, replay_admitted_gap)
        max_replay_packed_gap = max(max_replay_packed_gap, replay_packed_gap)
        max_packed_observation_gap = max(max_packed_observation_gap, packed_observation_gap)
        max_admitted_observation_gap = max(max_admitted_observation_gap, admitted_observation_gap)
        max_direct_observation_gap = max(max_direct_observation_gap, direct_observation_gap)
        max_triangle_bound = max(max_triangle_bound, triangle_bound)
        rows.append(
            {
                "view_index": index,
                "admitted_foreground_pixel_count": int(admitted.sum()),
                "replay_foreground_pixel_count": int(generated.sum()),
                "observation_import_geometry_reference_pixel_count": int(observation_reference.sum()),
                "packed_teacher_geometry_reference_pixel_count": int(packed_reference.sum()),
                "exact_mask_equal": exact,
                "xor_pixel_count": xor,
                "sealed_admitted_source_mask_to_raw_geometry_hausdorff_px": sealed_source_gap,
                "replay_to_packed_teacher_geometry_hausdorff_px": replay_packed_gap,
                "passes_sealed_policy_transport_support_court": passed,
                "replay_to_admitted_hausdorff_px": replay_admitted_gap,
                "packed_to_observation_import_geometry_hausdorff_px": packed_observation_gap,
                "packed_to_observation_import_geometry_within_sealed_gap": view_geometry_lane_equivalent,
                "admitted_to_observation_import_geometry_hausdorff_px": admitted_observation_gap,
                "direct_replay_to_observation_import_geometry_hausdorff_px": direct_observation_gap,
                "diagnostic_triangle_inequality_upper_bound_px": triangle_bound,
                "sealed_maximum_mask_gap_px": maximum_gap,
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
    if replay.get("authority") != "SOURCE_APPEARANCE_MASK_REPLAY_CANDIDATE_ONLY":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_RECEIPT_AUTHORITY_DRIFT")

    return {
        "schema": SCHEMA,
        "subject_id": subject,
        "admission_sha256": _sha(admission_path),
        "candidate_sha256": _sha(candidate_path),
        "camera_bundle_sha256": _sha(camera_path),
        "observation_import_geometry_sha256": _sha(imported_geometry_path),
        "packed_teacher_geometry_sha256": _sha(packed_geometry_path),
        "packed_source_scene_blend_sha256": packed_receipt["packed_source_scene_blend_sha256"],
        "observation_import_operation": import_receipt.get("operation"),
        "replay_receipt_sha256": _sha(replay_path),
        "view_count": 8,
        "exact_mask_equal_view_count": exact_count,
        "xor_pixel_count_total": xor_total,
        "maximum_replay_to_admitted_hausdorff_px": max_replay_admitted_gap,
        "maximum_replay_to_packed_teacher_geometry_hausdorff_px": max_replay_packed_gap,
        "maximum_packed_to_observation_import_geometry_hausdorff_px": max_packed_observation_gap,
        "maximum_admitted_to_observation_import_geometry_hausdorff_px": max_admitted_observation_gap,
        "maximum_direct_replay_to_observation_import_geometry_hausdorff_px": max_direct_observation_gap,
        "maximum_diagnostic_triangle_upper_bound_px": max_triangle_bound,
        "sealed_maximum_mask_gap_px": maximum_gap,
        "authority_metric": "PAIRED_SUPPORT_PRESERVATION__ADMITTED_SOURCE_MASK_TO_RAW_GEOMETRY__REPLAY_MASK_TO_PACKED_TEACHER_GEOMETRY",
        "triangle_upper_bound_is_authority": False,
        "geometry_lane_equivalence_required_for_policy_transport": False,
        "packed_to_observation_import_geometry_equivalent_within_sealed_gap": bool(geometry_lane_equivalent),
        "packed_geometry_source_observation_equivalence_claimed": False,
        "source_opacity_policy_transport_equivalent": bool(all_pass),
        "source_lit_appearance_parity_claimed": False,
        "rgb_equivalence_claimed": False,
        "effective_alpha_value_equivalence_claimed": False,
        "sampler_qualified": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "court_status": "PASS_SOURCE_APPEARANCE_POLICY_TRANSPORT" if all_pass else "FAIL_SOURCE_APPEARANCE_POLICY_TRANSPORT",
        "views": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--admission", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--camera-bundle", required=True, type=Path)
    p.add_argument("--imported-geometry", required=True, type=Path)
    p.add_argument("--packed-geometry", required=True, type=Path)
    p.add_argument("--replay-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = audit(
        a.admission,
        a.candidate,
        a.camera_bundle,
        a.imported_geometry,
        a.packed_geometry,
        a.replay_dir,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "views"}, sort_keys=True))
    if not result["source_opacity_policy_transport_equivalent"]:
        raise RuntimeError("SOURCE_APPEARANCE_POLICY_TRANSPORT_FAILED")


if __name__ == "__main__":
    main()
