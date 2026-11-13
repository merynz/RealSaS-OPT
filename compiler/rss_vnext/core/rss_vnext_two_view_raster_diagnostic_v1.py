from __future__ import annotations

"""Source alpha vs base/rest TESSA silhouette diagnostic (NOT source fidelity).

Use the existing RealSaS V3 camera projector and half-integer top-left raster
fill. This is a source-locked piece of RealSaS-style multiview supervision, not
8-view camera synthesis, per-pixel 3D depth, surface RGB ownership, pose-
conditioned animation (ArtMesh/AniGen), or editable drawable authority.
All original RGBA and neural fields remain available in independent CAS.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from .camera_geometry_v2 import CameraProjectionV3, project_points_xyz_v3
from .hashing import content_sha256
from .mesh.product_coverage_v1 import coverage_metrics, rasterize_triangles_half_integer_top_left
from .rss_vnext_inference_evidence_v1 import RSSvNextInferenceEvidenceIRV1, validate_inference_evidence_v1
from .rss_vnext_source_evidence_binding_v1 import (
    RSSvNextSourceEvidenceBindingCourtV1,
    validate_rss_vnext_source_evidence_binding_receipt_v1,
)
from .rss_vnext_source_observation_v1 import RSSvNextTwoViewSourceObservationV1
from .silhouette_metrics_v2 import silhouette_distance_metrics
from .types import QualificationError

SCHEMA = "RealSaS.RSSvNext.TwoViewRasterDiagnostic.v1"
_MAX_SOURCE_PIXELS_PER_VIEW = 1024 * 1024
_MAX_FACES = 20000
_MAX_RASTER_BBOX_TESTS = 2_000_000
_AUTHORITY = "OBSERVED_SOURCE_ALPHA_VS_LEARNED_PROJECTED_GEOMETRY_DIAGNOSTIC_ONLY"
_STATUS = "MEASURED_TWO_VIEW_SILHOUETTE_NOT_SOURCE_FIDELITY"


def _fail(code: str) -> None:
    raise QualificationError("RSS_VNEXT_TWO_VIEW_RASTER_" + code)


@dataclass(frozen=True)
class RSSvNextTwoViewRasterDiagnosticV1:
    source_binding_sha256: str
    inference_evidence_hash: str
    checkpoint_sha256: str
    original_mesh_sha256: str | None
    source_evidence_binding_court_hash: str
    state: str
    views: tuple[Mapping[str, Any], ...]
    diagnostic_hash: str
    schema: str = SCHEMA
    authority: str = _AUTHORITY
    source_fidelity_qualified: bool = False
    depth_or_first_hit_qualified: bool = False
    source_color_owner_qualified: bool = False
    semantic_part_or_drawable_qualified: bool = False
    canonical_mesh_authority: bool = False
    animation_ready_authority: bool = False


def _seal(value: RSSvNextTwoViewRasterDiagnosticV1) -> RSSvNextTwoViewRasterDiagnosticV1:
    payload = asdict(value)
    payload.pop("diagnostic_hash")
    return replace(value, diagnostic_hash=content_sha256(payload))


def validate_rss_vnext_two_view_raster_diagnostic_v1(
    value: RSSvNextTwoViewRasterDiagnosticV1,
) -> None:
    if (
        type(value) is not RSSvNextTwoViewRasterDiagnosticV1
        or value.schema != SCHEMA
        or value.authority != _AUTHORITY
        or value.state not in (
            _STATUS, "ABSTAIN_NO_TOPOLOGY", "ABSTAIN_RESOURCE_BUDGET",
            "ABSTAIN_EMPTY_RASTER",
        )
        or any((
            value.source_fidelity_qualified, value.depth_or_first_hit_qualified,
            value.source_color_owner_qualified,
            value.semantic_part_or_drawable_qualified,
            value.canonical_mesh_authority, value.animation_ready_authority,
        ))
    ):
        _fail("RECEIPT_PREMATURE_AUTHORITY")
    payload = asdict(value)
    digest = payload.pop("diagnostic_hash")
    if content_sha256(payload) != digest:
        _fail("RECEIPT_CONTENT_DRIFT")


def _project(
    xyz: np.ndarray, camera_values: np.ndarray, view_index: int, *,
    width: int, height: int,
) -> np.ndarray:
    # Existing V3 projector takes one resolution: x=width, y scaled to
    # height. This equals the fixed RSS source-ray align_corners=False
    # convention: original camera 'up' projects to decreasing image y.
    cam = CameraProjectionV3(
        view_id=("FRONT", "BACK")[view_index], view_index=view_index,
        origin=tuple(map(float, camera_values[0:3])),
        right=tuple(map(float, camera_values[3:6])),
        screen_up=tuple(map(float, camera_values[6:9])),
        forward=tuple(map(float, camera_values[9:12])),
        half_extent=float(camera_values[12]), resolution=width,
    )
    projected = project_points_xyz_v3(xyz, cam)
    projected[:, 1] *= float(height) / float(width)
    return projected


def _metrics(
    source_alpha: np.ndarray, predicted: bytes, *, width: int, height: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    raster = np.frombuffer(predicted, dtype=np.uint8).reshape(height, width).astype(bool)
    # Preserve uncertainty: both alpha masks are *diagnostic choices*, not a
    # single preselected source foreground truth or a transparent-material.
    for name, threshold in (("alpha_nonzero", 1), ("alpha_ge_128", 128)):
        source_mask = np.asarray(source_alpha >= threshold, dtype=np.uint8)
        source_bytes = source_mask.tobytes(order="C")
        metrics = (
            coverage_metrics(source_bytes, predicted, width=width, height=height)
            if bool(np.any(source_mask)) else None
        )
        if metrics is not None and bool(np.any(raster)):
            mean, p95, maximum = silhouette_distance_metrics(
                source_mask.astype(bool), raster,
            )
            metrics["silhouette_edge_mean_px"] = mean
            metrics["silhouette_edge_p95_px"] = p95
            metrics["silhouette_edge_max_px"] = maximum
        result[name] = {
            "threshold_uint8_inclusive": threshold,
            "source_foreground_mask_sha256": sha256(source_bytes).hexdigest(),
            "metrics": metrics,
            "source_threshold_has_foreground": bool(np.any(source_mask)),
        }
    return result




def _two_view_depth_nullspace_witness(
    source: RSSvNextTwoViewSourceObservationV1,
    xyz: np.ndarray,
    *,
    width: int, height: int,
) -> dict[str, Any]:
    """Prove what the two *observed* orthographic camera planes cannot see.

    SVD of the four image-coordinate Jacobian rows exposes any shared depth
    nullspace. With exact front/back orthographic cameras both projections are
    unchanged by moving 3D mesh vertices along that nullspace. The witness
    perturbs alternating vertices to show even a *different surface shape*
    can have identical 2D silhouettes and source-ray samples. This is an
    identifiability check, NOT an invented depth teacher or model output.

    With nonparallel cameras a full-rank Jacobian does not prove triangulation:
    actual source point correspondences and depth remain unobserved.
    """
    cameras = np.asarray(source.camera_features, dtype=np.float64)
    rows = np.stack((
        cameras[0, 3:6] / cameras[0, 12],
        cameras[0, 6:9] / cameras[0, 12],
        cameras[1, 3:6] / cameras[1, 12],
        cameras[1, 6:9] / cameras[1, 12],
    ), axis=0)
    _, sigma, vh = np.linalg.svd(rows, full_matrices=True)
    if (
        sigma.shape != (3,)
        or not np.isfinite(sigma).all()
        or float(sigma[0]) <= 0.0
    ):
        _fail("SOURCE_CAMERA_PROJECTION_SVD_INVALID")
    tolerance = float(sigma[0]) * 1.0e-5
    effective_rank = int(np.count_nonzero(sigma > tolerance))
    axis = np.array(vh[-1], dtype=np.float64)
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0:
        axis *= -1.0
    displacement = float(np.min(cameras[:, 12])) * .2
    result: dict[str, Any] = {
        "method": "TWO_ORIGINAL_ORTHOGRAPHIC_CAMERA_IMAGE_JACOBIAN_SVD",
        "effective_rank": effective_rank,
        "singular_values": [float(v) for v in sigma],
        "rank_tolerance": tolerance,
        "weakest_projection_axis_xyz": [float(v) for v in axis],
        "original_camera_projection_matrix_sha256": content_sha256(rows.tolist()),
        "source_point_correspondence_authority": "UNKNOWN_NOT_SOURCE_OBSERVED",
        "source_depth_or_surface_completion_authority": False,
        "canonical_mesh_authority": False,
        "counterfactual_status": "NOT_EVALUATED_NO_COMMON_NULLSPACE",
        "counterfactual_3d_vertex_displacement_world": None,
        "counterfactual_max_two_view_pixel_drift": None,
        "counterfactual_mesh_sha256": None,
    }
    if effective_rank != 2:
        return result
    counterfactual = np.array(xyz, dtype=np.float64, copy=True)
    signs = np.where(
        np.arange(len(counterfactual)) % 2 == 0, 1.0, -1.0
    )
    counterfactual += signs[:, None] * displacement * axis[None, :]
    maximum = 0.0
    for view_index in (0, 1):
        actual = _project(
            xyz, source.camera_features[view_index], view_index,
            width=width, height=height,
        )
        changed = _project(
            counterfactual, source.camera_features[view_index], view_index,
            width=width, height=height,
        )
        maximum = max(
            maximum, float(np.max(np.abs(changed[:, :2] - actual[:, :2])))
        )
    result.update({
        "counterfactual_status": (
            "DISTINCT_3D_GEOMETRY_SAME_TWO_SOURCE_PROJECTIONS"
            if maximum <= 1.0e-4
            else "NUMERICAL_NULLSPACE_WITNESS_UNSTABLE"
        ),
        "counterfactual_3d_vertex_displacement_world": displacement,
        "counterfactual_max_two_view_pixel_drift": maximum,
        "counterfactual_mesh_sha256": content_sha256(counterfactual.tolist()),
    })
    return result


def compile_rss_vnext_two_view_raster_diagnostic_v1(
    *,
    source: RSSvNextTwoViewSourceObservationV1,
    evidence: RSSvNextInferenceEvidenceIRV1,
    source_evidence_binding_court: RSSvNextSourceEvidenceBindingCourtV1,
) -> RSSvNextTwoViewRasterDiagnosticV1:
    """Do not qualify; explicit ABSTAIN avoids unbounded CPU raster work."""
    if type(source) is not RSSvNextTwoViewSourceObservationV1:
        _fail("CANONICAL_SOURCE_REQUIRED")
    validate_inference_evidence_v1(evidence)
    validate_rss_vnext_source_evidence_binding_receipt_v1(source_evidence_binding_court)
    if (
        source_evidence_binding_court.source_binding_sha256 != source.source_binding_sha256
        or source_evidence_binding_court.inference_evidence_hash != evidence.evidence_hash
        or source_evidence_binding_court.checkpoint_sha256 != evidence.checkpoint_sha256
    ):
        _fail("SOURCE_EVIDENCE_COURT_PARENT_DRIFT")

    raw_mesh = evidence.mesh
    mesh_hash = content_sha256(raw_mesh) if raw_mesh is not None else None
    parent = dict(
        source_binding_sha256=source.source_binding_sha256,
        inference_evidence_hash=evidence.evidence_hash,
        checkpoint_sha256=evidence.checkpoint_sha256,
        original_mesh_sha256=mesh_hash,
        source_evidence_binding_court_hash=source_evidence_binding_court.court_hash,
    )
    if raw_mesh is None:
        return _seal(RSSvNextTwoViewRasterDiagnosticV1(
            **parent, state="ABSTAIN_NO_TOPOLOGY", views=(), diagnostic_hash="",
        ))
    nface = len(raw_mesh["faces"])
    rgba = source.native_rgba
    h, w = map(int, rgba.shape[-2:])
    if w * h > _MAX_SOURCE_PIXELS_PER_VIEW or nface > _MAX_FACES:
        return _seal(RSSvNextTwoViewRasterDiagnosticV1(
            **parent, state="ABSTAIN_RESOURCE_BUDGET", views=(), diagnostic_hash="",
        ))
    faces = np.asarray(raw_mesh["faces"], dtype=np.int64)
    # This independently derived evidence prevents a 100% two-view silhouette
    # match being misread as uniqueness of the 3D canonical surface.
    depth_witness = _two_view_depth_nullspace_witness(
        source, np.asarray(raw_mesh["rest_xyz"], dtype=np.float64),
        width=w, height=h,
    )
    rows = []
    raster_empty = False
    for view in range(2):
        row: dict[str, Any] = {
            "view_id": ("FRONT", "BACK")[view],
            "view_index": view,
            "original_source_rgba_sha256": source.view_pixel_sha256[view],
            "width": w, "height": h,
            "geometry": {},
            "source_depth_authority": "UNKNOWN_NO_SOURCE_DEPTH",
            "source_color_owner_authority": "UNKNOWN_NO_SOURCE_POINT_CORRESPONDENCE",
            "raster_rule": "EXISTING_HALF_INTEGER_TOP_LEFT_BINARY_SILHOUETTE",
            "two_view_source_depth_observability": depth_witness,
        }
        for candidate_name, field in (("base", "base_xyz"), ("rest", "rest_xyz")):
            xyz = np.asarray(raw_mesh[field], dtype=np.float64)
            projected = _project(xyz, source.camera_features[view], view, width=w, height=h)
            # Scalar raster requires an explicit bound on the SUM of clipped
            # triangle bounding-box tests. Pixel count * face count is not a
            # reliable cost predictor. Do NOT silently downsample source art.
            xy = projected[faces, :2]
            lower = np.floor(np.min(xy, axis=1) - .5)
            upper = np.ceil(np.max(xy, axis=1) - .5)
            lo = np.maximum(lower, [0., 0.])
            hi = np.minimum(upper, [float(w - 1), float(h - 1)])
            box = np.maximum(hi - lo + 1, 0)
            bbox_work = int(np.sum(box[:, 0] * box[:, 1], dtype=np.float64))
            if bbox_work > _MAX_RASTER_BBOX_TESTS:
                return _seal(RSSvNextTwoViewRasterDiagnosticV1(
                    **parent, state="ABSTAIN_RESOURCE_BUDGET",
                    views=(), diagnostic_hash="",
                ))
            triangles = [
                tuple((float(projected[int(i), 0]), float(projected[int(i), 1])) for i in tri)
                for tri in faces
            ]
            mask_bytes = rasterize_triangles_half_integer_top_left(
                triangles, width=w, height=h,
            )
            has_mask = any(mask_bytes)
            if not has_mask:
                raster_empty = True
            row["geometry"][candidate_name] = {
                "binary_projected_raster_sha256": sha256(mask_bytes).hexdigest(),
                "raster_bbox_test_upper_bound": bbox_work,
                "predicted_foreground_pixel_count": int(sum(mask_bytes)),
                "alpha_comparison": _metrics(rgba[view, 3], mask_bytes, width=w, height=h),
                "candidate_3d_surface_fidelity_qualified": False,
            }
        rows.append(row)
    return _seal(RSSvNextTwoViewRasterDiagnosticV1(
        **parent, state="ABSTAIN_EMPTY_RASTER" if raster_empty else _STATUS,
        views=tuple(rows), diagnostic_hash="",
    ))


__all__ = [
    "SCHEMA", "RSSvNextTwoViewRasterDiagnosticV1",
    "compile_rss_vnext_two_view_raster_diagnostic_v1",
    "validate_rss_vnext_two_view_raster_diagnostic_v1",
]
