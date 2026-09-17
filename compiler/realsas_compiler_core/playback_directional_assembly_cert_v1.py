from __future__ import annotations

"""Render-free certificate for a complete directional Runtime-v4 assembly.

The certificate proves the failure class that produced Mage pepper/black holes is
absent from the admitted runtime representation:

1. every active drawable face is source-backed (DIRECT in its owner view);
2. BODY_UNDERLAY is active and submitted before every foreground slot at all frames;
3. product continuity-underlay authority is present and explicitly source-only;
4. every active 2D attachment remains a continuous piecewise-linear embedding for the
   exact Runtime-v4 interpolation intervals (no triangle collapse/inversion and no
   nonadjacent boundary crossing);
5. BODY setup coverage retains the qualified source-alpha evidence.

This does not claim subjective animation quality. It is a pre-render topological and
source-authority admission certificate.
"""

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_directional_motion_cert_v1 import (
    _boundary_edges,
    _boundary_interval_nonintersection,
    _interval_signed_area2_min,
    _signed_area2,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import project_points_xyz_v3
from compiler.realsas_compiler_core.playback_runtime_v3 import AppearanceProvenance
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    DEFAULT_VIEWS,
    provenance_from_code,
    validate_playback_runtime_v4_contract,
    validate_runtime_v4_clip,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.export.current_v4_directional_runtime_v4 import (
    BODY_COMPONENT_ID,
    RuntimeV4DirectionalAssemblyProjectionV1,
)

DIRECTIONAL_ASSEMBLY_CERTIFICATE_SCHEMA = "RealSaS.DirectionalAssemblyMotionCertificate.v1"


@dataclass(frozen=True)
class DirectionalAssemblyMotionCertificateV1:
    clip_id: str
    source_backed_active_faces_certified: bool
    continuity_underlay_certified: bool
    continuous_embedding_certified: bool
    body_underlay_always_active_first: bool
    body_source_alpha_recall_floor: float
    body_source_precision_floor: float
    min_signed_area2_margin: float
    min_boundary_distance_at_critical_times: float
    active_asset_interval_count: int
    certificate_hash: str
    schema_version: str = DIRECTIONAL_ASSEMBLY_CERTIFICATE_SCHEMA


def _body_coverage(
    product,
    view_ids: tuple[str, ...],
    body_component_id: str,
) -> tuple[float, float]:
    by_view = {
        int(direction.view_index): {
            str(component.component_id): component
            for component in direction.components
        }
        for direction in product.directional_renderables.directions
    }
    if set(by_view) != set(range(len(view_ids))):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_PRODUCT_VIEW_SET_MISMATCH")
    recall = []
    precision = []
    for i in range(len(view_ids)):
        body = by_view[i].get(str(body_component_id))
        if body is None:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_BODY_COMPONENT_MISSING")
        report = dict(body.mesh.qualification_report or {})
        residual = dict(report.get("candidate_residual_report") or {})
        r = float(report.get("source_alpha_recall", residual.get("source_alpha_recall", -1.0)))
        p = float(report.get("precision_inside_alpha", residual.get("precision_inside_alpha", -1.0)))
        if not math.isfinite(r) or not math.isfinite(p) or r < 0.0 or p < 0.0:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_BODY_COVERAGE_EVIDENCE_MISSING")
        recall.append(r)
        precision.append(p)
    return float(min(recall)), float(min(precision))


def certify_directional_runtime_v4_assembly_v1(
    *,
    product,
    projection: RuntimeV4DirectionalAssemblyProjectionV1,
    clip_id: str,
    min_body_source_alpha_recall: float = 0.97,
    min_body_source_precision: float = 0.995,
    min_signed_area2: float = 1.0e-4,
    required_view_ids: Sequence[str] = DEFAULT_VIEWS,
    body_component_id: str | None = None,
) -> DirectionalAssemblyMotionCertificateV1:
    view_ids = tuple(map(str, required_view_ids))
    body_component_id = str(body_component_id or projection.body_component_id or BODY_COMPONENT_ID)
    if body_component_id != str(projection.body_component_id):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_BODY_COMPONENT_ID_DRIFT")
    contract = projection.contract
    validate_playback_runtime_v4_contract(contract, required_view_ids=view_ids)
    clips = {clip.clip_id: clip for clip in projection.clips}
    clip = clips.get(str(clip_id))
    if clip is None:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_CLIP_MISSING")
    validate_runtime_v4_clip(contract, clip, required_view_ids=view_ids)

    metadata = dict(product.directional_renderables.metadata or {})
    if metadata.get("continuity_underlay_qualified") is not True:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_CONTINUITY_UNDERLAY_REQUIRED")
    underlay_hash = str(metadata.get("continuity_underlay_set_hash") or "")
    if len(underlay_hash) != 64:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_CONTINUITY_UNDERLAY_HASH_INVALID")
    underlay_payload = metadata.get("continuity_underlay")
    if not isinstance(underlay_payload, dict):
        # Current continuity module stores the typed payload under a private metadata
        # key; locate by schema rather than guessing its private key name.
        candidates = [
            value for value in metadata.values()
            if isinstance(value, dict)
            and str(value.get("schema_version", "")).startswith("RealSaS.QualifiedContinuityUnderlaySet")
        ]
        if len(candidates) != 1:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_CONTINUITY_UNDERLAY_PAYLOAD_MISSING")
        underlay_payload = candidates[0]
    views_payload = tuple(underlay_payload.get("views") or ())
    if len(views_payload) != len(view_ids):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_CONTINUITY_VIEW_CARDINALITY")
    for row in views_payload:
        if bool(row.get("new_pixels_generated", True)):
            raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_UNDERLAY_NEW_PIXELS_FORBIDDEN")
        if str(row.get("substrate_component_id")) != body_component_id:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_UNDERLAY_SUBSTRATE_DRIFT")

    body_recall, body_precision = _body_coverage(product, view_ids, body_component_id)
    if body_recall < float(min_body_source_alpha_recall):
        raise QualificationError(f"DIRECTIONAL_ASSEMBLY_CERT_BODY_RECALL_FAIL:{body_recall}")
    if body_precision < float(min_body_source_precision):
        raise QualificationError(f"DIRECTIONAL_ASSEMBLY_CERT_BODY_PRECISION_FAIL:{body_precision}")

    slot_index = {slot.slot_id: i for i, slot in enumerate(contract.slots)}
    if body_component_id not in slot_index:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_BODY_SLOT_MISSING")
    body_slot = body_component_id

    # Active owner-view assets must be direct source only. Other-view overlays exist
    # solely to make source authority total for inactive assets and may never be the
    # active target-view drawable.
    asset_index = {asset.asset_id: i for i, asset in enumerate(contract.assets)}
    asset_by_attachment = {asset.attachment_id: asset for asset in contract.assets}
    overlay_by_view_asset = {
        view.view_id: {row.asset_id: row for row in view.assets}
        for view in contract.views
    }
    owner = dict(projection.owner_view_by_asset)

    min_area = float("inf")
    min_boundary = float("inf")
    interval_count = 0
    rest_sign_by_asset_view = {}

    for frame_index, frame in enumerate(clip.frames):
        for view_index, view_id in enumerate(view_ids):
            composition = frame.composition_by_view[view_id]
            if not composition.draw_order_slot_ids or composition.draw_order_slot_ids[0] != body_slot:
                raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_BODY_NOT_FIRST")
            active_body = composition.active_attachment_by_slot.get(body_slot)
            if not active_body:
                raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_BODY_NOT_ACTIVE")
            for slot_id in composition.draw_order_slot_ids:
                attachment = composition.active_attachment_by_slot.get(slot_id)
                if not attachment:
                    raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_ACTIVE_SLOT_EMPTY")
                asset = asset_by_attachment.get(attachment)
                if asset is None:
                    raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_ACTIVE_ASSET_MISSING")
                if int(owner.get(asset.asset_id, -1)) != view_index:
                    raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_ACTIVE_ASSET_NOT_OWNER_VIEW")
                overlay = overlay_by_view_asset[view_id][asset.asset_id]
                if np.any([
                    provenance_from_code(int(code)) != AppearanceProvenance.DIRECT_SOURCE
                    for code in overlay.provenance_codes
                ]):
                    raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_ACTIVE_FACE_NOT_DIRECT_SOURCE")
                if np.any(np.asarray(overlay.donor_view_indices, dtype=np.int64) != view_index):
                    raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_ACTIVE_DONOR_NOT_TARGET_VIEW")

    # Prove exact Runtime-v4 linear interpolation of every active owner-view asset.
    for view_index, view_id in enumerate(view_ids):
        camera = contract.views[view_index].camera
        owner_assets = [
            asset for asset in contract.assets
            if int(owner.get(asset.asset_id, -1)) == view_index
        ]
        if len(owner_assets) != len(contract.slots):
            raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_OWNER_ASSET_COUNT_DRIFT")
        for asset in owner_assets:
            triangles = np.asarray(asset.triangles, dtype=np.int64)
            boundary = _boundary_edges(triangles)
            projected = [
                project_points_xyz_v3(
                    np.asarray(frame.canonical_posed_xyz_by_asset[asset.asset_id], dtype=np.float64),
                    camera,
                )[:, :2]
                for frame in clip.frames
            ]
            rest = project_points_xyz_v3(
                np.asarray(asset.rest_xyz, dtype=np.float64),
                camera,
            )[:, :2]
            rest_area = np.asarray(
                [_signed_area2(rest, tri) for tri in triangles],
                dtype=np.float64,
            )
            if np.any(np.abs(rest_area) <= float(min_signed_area2)):
                raise QualificationError(
                    f"DIRECTIONAL_ASSEMBLY_CERT_REST_DEGENERATE:{view_id}:{asset.asset_id}"
                )
            signs = np.where(rest_area > 0.0, 1.0, -1.0)
            rest_sign_by_asset_view[(view_id, asset.asset_id)] = signs

            for fi in range(len(projected) - 1):
                p0, p1 = projected[fi], projected[fi + 1]
                interval_count += 1
                for ti, tri in enumerate(triangles):
                    margin = _interval_signed_area2_min(
                        p0, p1, tri, float(signs[ti])
                    )
                    min_area = min(min_area, margin)
                    if margin <= float(min_signed_area2):
                        raise QualificationError(
                            "DIRECTIONAL_ASSEMBLY_CERT_INTERVAL_COLLAPSE_OR_INVERSION:"
                            f"{view_id}:{asset.asset_id}:{fi}:{ti}:{margin}"
                        )
                boundary_distance = _boundary_interval_nonintersection(p0, p1, boundary)
                min_boundary = min(min_boundary, boundary_distance)

    if interval_count <= 0:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_CERT_NO_INTERVALS")

    payload = {
        "schema": DIRECTIONAL_ASSEMBLY_CERTIFICATE_SCHEMA,
        "clip_id": clip.clip_id,
        "source_product_state_hash": product.product_state_hash,
        "projection_hash": projection.projection_hash,
        "continuity_underlay_set_hash": underlay_hash,
        "body_component_id": body_component_id,
        "source_backed_active_faces_certified": True,
        "continuity_underlay_certified": True,
        "continuous_embedding_certified": True,
        "body_underlay_always_active_first": True,
        "body_source_alpha_recall_floor": body_recall,
        "body_source_precision_floor": body_precision,
        "min_signed_area2_margin": min_area,
        "min_boundary_distance_at_critical_times": min_boundary,
        "active_asset_interval_count": interval_count,
        "completion_used": False,
        "runtime_interpolation": "LINEAR_VIEW_LOCAL_EQUAL_DEPTH_DIRECTIONAL_ASSETS",
    }
    return DirectionalAssemblyMotionCertificateV1(
        clip_id=clip.clip_id,
        source_backed_active_faces_certified=True,
        continuity_underlay_certified=True,
        continuous_embedding_certified=True,
        body_underlay_always_active_first=True,
        body_source_alpha_recall_floor=body_recall,
        body_source_precision_floor=body_precision,
        min_signed_area2_margin=float(min_area),
        min_boundary_distance_at_critical_times=float(min_boundary),
        active_asset_interval_count=int(interval_count),
        certificate_hash=content_sha256(payload),
    )


__all__ = [
    "DIRECTIONAL_ASSEMBLY_CERTIFICATE_SCHEMA",
    "DirectionalAssemblyMotionCertificateV1",
    "certify_directional_runtime_v4_assembly_v1",
]
