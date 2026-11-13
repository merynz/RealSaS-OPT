from __future__ import annotations

"""Real-source 2D silhouette qualification for uncalibrated RSS two-image Mesh.

This creates a CanonicalMeshIR **only** when an exact TESSA token decode and
the two *original* alpha silhouettes support the proposed surface. FRONT/BACK
orientations are USER CLAIMS. The fitted orthographic view transforms are
camera HYPOTHESES, never original measured camera rays, source depth, texture
ownership, hidden geometry truth or mechanical qualification.
"""

from dataclasses import asdict, replace
from hashlib import sha256
import io
import math
from typing import Any

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import mint_canonical_mesh_v1
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics, rasterize_triangles_half_integer_top_left,
    source_connected_component_recall_metrics,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR, CanonicalMeshVertexCandidateIR,
    ComponentCarrierDecisionIR, ComponentRegionIR, RiggingSurfaceIR,
    build_component_carrier_policy,
    build_mechanical_partition, canonical_mesh_candidate_lineage_hash,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR, build_surface_addressing,
    static_mesh_qualification_hash,
)
from compiler.realsas_compiler_core.types import SurfaceNode, SurfaceSupportBinding
from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    RSSvNextUncalibratedTwoImageSourceV1,
)
from compiler.realsas_compiler_core.silhouette_metrics_v2 import (
    silhouette_distance_metrics,
)

_MAX_VIEW_SIDE = 384
_MAX_FACE_RASTER_VISITS = 3_000_000
_MAX_VERTICES = 32768
_MAX_FACES = 32768
_THRESHOLDS = {
    "min_recall": .90, "min_precision": .90,
    "min_component_recall": .85, "max_edge_p95": 2.,
    "max_hole": .10, "max_interior_uncovered": .10,
}


def _replay_tessa(scene: dict[str, np.ndarray], receipt: dict[str, Any]) -> None:
    from dataclasses import fields
    from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
    from models.tessa.v2.court_v2 import _codec_config, evaluate_decoded_topology_v2
    from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1

    raw = receipt.get("tessa_decoder_config")
    allowed = {f.name for f in fields(TESSAStructuralizerConfigV3)}
    if type(raw) is not dict or set(raw) != allowed:
        raise RuntimeError("RSS_VNEXT_RAW_MESH_TESSA_CONFIG_NOT_PINNED")
    cfg = TESSAStructuralizerConfigV3(**raw)
    cfg.validate()
    token_rows = scene["tessa_topology_token_ids"]
    if (
        token_rows.dtype.kind not in "iu" or token_rows.shape[0] != 1
        or token_rows.shape[1] > 131072
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_TESSA_TOKEN_DOMAIN")
    tokens = tuple(map(int, token_rows[0]))
    court = evaluate_decoded_topology_v2(tokens, config=cfg)
    if (
        not court.grammar_valid or court.zero_area_face_count
        or court.nonmanifold_edge_count or not court.finite_geometry
        or not court.within_scale_policy
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_TESSA_TOPOLOGY_COURT_FAILED")
    decoded = decode_tessa_asset_sequence_v1(tokens, cfg=_codec_config(cfg))
    base = scene["tessa_candidate_base_vertices"][0]
    faces = scene["tessa_candidate_faces"]
    components = scene["tessa_candidate_vertex_components"]
    face_components = scene["tessa_candidate_face_components"]
    if (
        not np.array_equal(faces, decoded.faces)
        or not np.array_equal(components, decoded.vertex_component_indices)
        or not np.array_equal(face_components, decoded.face_component_indices)
        or base.shape != decoded.vertices_normalized.shape
        or not np.allclose(
            base, decoded.vertices_normalized * 2.,
            rtol=0., atol=2.e-6,
        )
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_TESSA_TOKEN_SURFACE_MISMATCH")


def _observed_alpha(source: RSSvNextUncalibratedTwoImageSourceV1, index: int):
    if source.view_claims != ("FRONT", "BACK"):
        return None, "FRONT_AND_BACK_VIEW_CLAIMS_REQUIRED"
    if not source.source_images[index]["source_alpha_present"]:
        return None, "SOURCE_HAS_NO_OBSERVED_ALPHA"
    with Image.open(io.BytesIO(source.original_files[index])) as image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        factor = min(1., _MAX_VIEW_SIDE / max(width, height))
        new_size = (max(16, round(width * factor)), max(16, round(height * factor)))
        if new_size != rgba.size:
            rgba = rgba.resize(new_size, resample=Image.Resampling.NEAREST)
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
        mask = alpha >= 128
    count = int(np.count_nonzero(mask))
    if count < 20 or count > mask.size * .95:
        return None, "SOURCE_ALPHA_IS_EMPTY_OR_FULL_FRAME"
    return mask, None


def _project_fitted(
    xyz: np.ndarray, source_mask: np.ndarray, view_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = source_mask.shape
    pixels_y, pixels_x = np.where(source_mask)
    # User FRONT/BACK is an orientation hypothesis, not a calibration record.
    uv = xyz[:, :2].copy()
    uv[:, 0] *= (1. if view_index == 0 else -1.)
    uv[:, 1] *= -1.
    low = uv.min(axis=0)
    high = uv.max(axis=0)
    span = high - low
    observed_low = np.array([pixels_x.min(), pixels_y.min()], dtype=np.float64)
    observed_high = np.array([pixels_x.max() + 1, pixels_y.max() + 1], dtype=np.float64)
    image_span = observed_high - observed_low
    if min(span) < 1e-5 or min(image_span) < 2:
        raise RuntimeError("RSS_VNEXT_RAW_MESH_FITTED_PROJECTION_DEGENERATE")
    scale = float(min(image_span / span))
    if scale < 1e-7 or not math.isfinite(scale):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_CAMERA_SCALE_INVALID")
    offset = .5 * (observed_low + observed_high) - scale * (.5 * (low + high))
    projected = uv * scale + offset
    return projected, {
        "camera_authority": "FITTED_UNCALIBRATED_ORTHOGRAPHIC_HYPOTHESIS",
        "view_claim": ("FRONT", "BACK")[view_index],
        "screen_scale_px_per_model_unit": scale,
        "screen_offset_xy_px": [float(x) for x in offset],
        "source_camera_was_observed": False,
    }


def _measure_source_views(
    source: RSSvNextUncalibratedTwoImageSourceV1,
    xyz: np.ndarray, faces: np.ndarray,
) -> tuple[bool, list[dict[str, Any]]]:
    views = []
    for i in range(2):
        source_mask, blocker = _observed_alpha(source, i)
        if blocker is not None:
            views.append({"view_index": i, "accepted": False, "blocker": blocker,
                          "original_bytes_sha256": sha256(source.original_files[i]).hexdigest()})
            continue
        height, width = source_mask.shape
        points, hypothesis = _project_fitted(xyz, source_mask, i)
        # Bound worst-case raster work BEFORE entering the pure Python rasterizer.
        visits = 0
        triangles = []
        for tri in faces:
            p = points[tri]
            width_bbox = min(width, max(0, math.ceil(float(np.ptp(p[:, 0])) + 2)))
            height_bbox = min(height, max(0, math.ceil(float(np.ptp(p[:, 1])) + 2)))
            visits += width_bbox * height_bbox
            if visits > _MAX_FACE_RASTER_VISITS:
                raise RuntimeError("RSS_VNEXT_RAW_MESH_RASTER_RESOURCE_LIMIT")
            triangles.append(tuple(tuple(float(x) for x in xy) for xy in p))
        rendered = rasterize_triangles_half_integer_top_left(
            triangles, width=width, height=height,
        )
        foreground = source_mask.astype(np.uint8).tobytes(order="C")
        stats = coverage_metrics(foreground, rendered, width=width, height=height)
        components = source_connected_component_recall_metrics(
            foreground, rendered, width=width, height=height,
            minimum_foreground_fraction=.005,
        )
        raster = np.frombuffer(rendered, dtype=np.uint8).reshape(height, width).astype(bool)
        if not raster.any():
            edge_p95 = float("inf")
        else:
            _, edge_p95, _ = silhouette_distance_metrics(source_mask, raster)
        accepted = (
            stats["recall"] >= _THRESHOLDS["min_recall"]
            and stats["precision"] >= _THRESHOLDS["min_precision"]
            and stats["largest_coherent_hole_fraction"] <= _THRESHOLDS["max_hole"]
            and stats["interior_uncovered_fraction"] <= _THRESHOLDS["max_interior_uncovered"]
            and components["eligible_component_count"] > 0
            and components["minimum_eligible_component_recall"] >= _THRESHOLDS["min_component_recall"]
            and edge_p95 <= _THRESHOLDS["max_edge_p95"]
        )
        views.append({
            "view_index": i,
            "accepted": bool(accepted),
            "original_bytes_sha256": sha256(source.original_files[i]).hexdigest(),
            "raster_source_alpha_sha256": sha256(foreground).hexdigest(),
            "predicted_raster_sha256": sha256(rendered).hexdigest(),
            "raster_resolution": [width, height],
            "measured": stats,
            "component_recall": components,
            "silhouette_edge_p95_px": edge_p95,
            "camera_hypothesis": hypothesis,
            "source_depth_observed": False,
            "source_rgb_surface_ownership_qualified": False,
        })
    return all(row["accepted"] for row in views), views


def _candidate_from_tessa(
    *, xyz: np.ndarray, faces: np.ndarray, component_ids: np.ndarray,
    receipt: dict[str, Any], scene_digest: str,
) -> tuple[Any, Any, Any, Any]:
    # The source of this geometric mesh is the learned TESSA proposal, not a
    # source-ray reconstruction. The screen-space court alone decides admission.
    from compiler.realsas_compiler_core.product_authority_v1 import (
        canonical_mesh_candidate_lineage_hash as candidate_hash,
    )
    identity = content_sha256({
        "schema": "RealSaS.RSSvNext.RawMeshCandidateSource.v1",
        "source_binding": receipt["source_binding_sha256"],
        "scene_digest": scene_digest,
        "checkpoint": receipt["checkpoint_sha256"],
    })
    n = xyz.shape[0]
    vertex_ids = tuple("RSSV:%08d" % i for i in range(n))
    surface_ids = tuple("RSSS:%s:%08d" % (identity[:16], i) for i in range(n))
    component_names = {
        int(comp): "RSSC:%s:%d" % (identity[:16], int(comp))
        for comp in np.unique(component_ids)
    }
    source = RiggingSurfaceIR(
        surface_nodes=tuple(
            SurfaceNode(
                surface_id=surface_ids[i],
                P=tuple(float(x) for x in xyz[i]),
                support_views=(),
                provenance_refs=(scene_digest,),
                source_observation_ids=(),
                metadata={"neural_geometry_not_source_3d_truth": True},
            )
            for i in range(n)
        ),
        geometry_lineage_hash=identity,
        builder_id="RSS_VNEXT_TESSA_LEARNED_PROPOSAL_BASIS",
        metadata={"source_surface_qualified": False},
    )
    partition = build_mechanical_partition(
        surface=source,
        components=tuple(
            ComponentRegionIR(
                component_id=component_names[comp],
                surface_ids=tuple(
                    surface_ids[i] for i in range(n) if int(component_ids[i]) == comp
                ),
                evidence_refs=(scene_digest,),
                metadata={"mechanical_separability": "UNKNOWN"},
            ) for comp in sorted(component_names)
        ),
        boundary_constraints=(),
        metadata={"component_ids_from_tessa": True},
    )
    carrier_policy = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(
                component_id=component_names[comp], carrier_class="MESH",
                evidence_refs=(scene_digest,),
                metadata={"source_3d_truth": False},
            ) for comp in sorted(component_names)
        ),
    )
    surface_vertices = tuple(
        CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=vertex_ids[i],
            support_binding=SurfaceSupportBinding(
                mode="IDENTITY_SURFACE_NODE",
                coefficients=((surface_ids[i], 1.),),
                metadata={"support": "TESSA_PREDICTED_3D_POSITION"},
            ),
            component_id=component_names[int(component_ids[i])],
            P=tuple(float(v) for v in xyz[i]),
            metadata={"vertex_index": i, "model_generated_geometry": True},
        )
        for i in range(n)
    )
    face_ids = tuple(tuple(vertex_ids[int(index)] for index in face) for face in faces)
    edge_ids = tuple(sorted({
        tuple(sorted((vertex_ids[int(a)], vertex_ids[int(b)])))
        for face in faces
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    }))
    candidate = CanonicalMeshCandidateIR(
        vertices=surface_vertices, faces=face_ids, edges=edge_ids,
        surface_binding_hash=source.geometry_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        producer_id="RSS_VNEXT_RAW_TWO_SOURCE_TESSA",
        producer_policy_hash=content_sha256({"schema": "RSSvNext.RawTwoViewMeshPolicy.v1"}),
        candidate_lineage_hash="",
        metadata={"source_3d_independently_observed": False,
                  "original_scene_cas_sha256": scene_digest},
    )
    candidate = replace(
        candidate, candidate_lineage_hash=candidate_hash(candidate),
    )
    validate_canonical_mesh_candidate(
        candidate, surface=source, partition=partition,
        carrier_policy=carrier_policy,
    )
    return candidate, source, partition, carrier_policy


def admit_raw_two_view_mesh_v1(
    *,
    source: RSSvNextUncalibratedTwoImageSourceV1,
    scene: dict[str, np.ndarray],
    receipt: dict[str, Any],
    scene_digest: str,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    """Return only an independently measured canonical static carrier or FAIL.

    A learned checkpoint may propose any topology. It cannot label its own
    normals/silhouettes as source observations; this court recomputes actual
    original-alpha comparisons from immutable input file bytes.
    """
    if scene["tessa_candidate_rest_vertices"].shape[1] > _MAX_VERTICES or (
        scene["tessa_candidate_faces"].shape[0] > _MAX_FACES
    ):
        raise RuntimeError("RSS_VNEXT_RAW_MESH_STATIC_RESOURCE_LIMIT")
    _replay_tessa(scene, receipt)
    xyz = scene["tessa_candidate_rest_vertices"][0].astype(np.float64)
    faces = scene["tessa_candidate_faces"].astype(np.int64)
    components = scene["tessa_candidate_vertex_components"].astype(np.int64)
    qualified, views = _measure_source_views(source, xyz, faces)
    witness = {
        "schema": "RealSaS.RSSvNext.RawTwoObservedViewStaticAdmission.v1",
        "source_binding_sha256": source.source_binding_sha256,
        "scene_content_sha256": scene_digest,
        "unified_checkpoint_sha256": receipt["checkpoint_sha256"],
        "original_camera_calibrated": False,
        "camera_hypotheses_are_observed": False,
        "original_alpha_only_observation": True,
        "source_3d_depth_observed": False,
        "source_rgb_surface_ownership_qualified": False,
        "hidden_geometry_qualified": False,
        "view_results": views,
        "source_static_silhouette_qualified": bool(qualified),
        "product_revision_minted": False,
    }
    if not qualified:
        return None, None, witness
    candidate, _, _, _ = _candidate_from_tessa(
        xyz=xyz, faces=faces, component_ids=components,
        receipt=receipt, scene_digest=scene_digest,
    )
    addresses = build_surface_addressing(candidate)
    static = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash=addresses.addressing_hash,
        appearance_domain_binding_hash="",
        geometry_gate_binding_hash=content_sha256(witness),
        partition_binding_hash=candidate.partition_binding_hash,
        qualification_report={
            "status": "PASS_STATIC_CANONICAL_CARRIER",
            "static_quality_policy_passed": True,
            "source_fidelity_qualification_passed": True,
            "source_fidelity_scope": "TWO_ORIGINAL_ALPHA_MASKS_WITH_FITTED_CAMERA_HYPOTHESES",
            "source_3d_geometry_qualified": False,
            "source_original_camera_qualified": False,
            "source_rgb_surface_ownership_qualified": False,
            "source_hidden_surface_truth_claimed": False,
            "demo_geometry_lineage": False,
        },
        qualification_hash="",
        metadata={
            "source_geometry_unobserved": True,
            "two_original_source_silhouettes_recomputed": True,
            "camera_hypotheses_not_calibrated": True,
        },
    )
    static = replace(static, qualification_hash=static_mesh_qualification_hash(static))
    carrier = build_mechanical_carrier_evidence_v1(
        candidate, static_qualification=static, surface_addressing=addresses,
    )
    mesh = mint_canonical_mesh_v1(
        candidate, static_qualification=static, carrier_evidence=carrier,
        proposal_evidence_refs=(
            receipt["checkpoint_sha256"],
            scene_digest,
            content_sha256(witness),
        ),
    )
    return mesh, static, witness


__all__ = ["admit_raw_two_view_mesh_v1"]
