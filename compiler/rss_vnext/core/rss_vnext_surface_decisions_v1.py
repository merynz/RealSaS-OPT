from __future__ import annotations

"""RSS unified TESSA/Geometry -> Compiler addressed face evidence.

Exactly replay TESSA's frozen token codec, check face/component addresses,
calculate refined-geometry areas and normals, and expose per-face full
Semantic/Presentation/Graphics/Mechanical proposals plus original two-source
rays. This is a *derived view* of the unchanged source-bound shared carrier,
NOT another neural scene, canonical mesh, texture, drawable or rig.

Topology ABSTAIN never blocks independent semantic/graphics/affordance reads.
"""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    candidate_face_region_readout_v1,
)
from compiler.realsas_compiler_core.rss_vnext_expert_fanout_v1 import (
    RSSvNextExpertEvidenceFanoutV1,
)
from compiler.realsas_compiler_core.types import QualificationError

SCHEMA = "RealSaS.RSSvNext.ComputedCandidateSurface.v1"


@dataclass(frozen=True)
class RSSvNextComputedCandidateFaceV1:
    candidate_face_index: int
    candidate_vertex_indices: tuple[int, int, int]
    candidate_component_index: int
    area: float
    unit_normal_xyz: tuple[float, float, float]
    region_probabilities: tuple[float, ...]  # UNKNOWN final, not a drawable
    presentation_role_probabilities: tuple[float, ...]
    graphics_logits_by_corner: tuple[tuple[float, ...], ...]
    affordance_logits_by_corner: tuple[tuple[float, ...], ...]
    original_rgba_by_corner: tuple[tuple[tuple[float, ...], ...], ...]
    original_projection_valid_by_corner: tuple[tuple[bool, bool], ...]
    source_visibility_qualified: bool = False
    source_appearance_texture_qualified: bool = False
    independent_drawable_qualified: bool = False
    mechanical_anchor_qualified: bool = False


@dataclass(frozen=True)
class RSSvNextComputedCandidateSurfaceV1:
    source_inference_evidence_hash: str
    neural_scene_carrier_sha256: str
    unified_model_config_sha256: str
    candidate_mesh_sha256: str | None
    topology_token_sequence_sha256: str | None
    state: str
    reason: str
    faces: tuple[RSSvNextComputedCandidateFaceV1, ...]
    schema: str = SCHEMA
    canonical_mesh_authority: bool = False
    signed_solid_authority: bool = False
    source_visibility_authority: bool = False
    independent_editability_authority: bool = False


def _fail(code: str) -> None:
    raise QualificationError(code)


def _softmax(values: Any) -> tuple[float, ...]:
    values = tuple(float(x) for x in values)
    high = max(values)
    z = [math.exp(x - high) for x in values]
    norm = sum(z)
    if not math.isfinite(norm) or norm <= 0.:
        _fail("RSS_VNEXT_SURFACE_SOFTMAX_NONFINITE")
    return tuple(x / norm for x in z)


def _replay_topology(evidence: Any, cfg: Any) -> None:
    """No short-circuit on a self-consistent but rehashed mesh proposal."""
    from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
    from models.tessa.v2.court_v2 import (
        _codec_config, evaluate_decoded_topology_v2,
    )

    mesh = evidence.mesh
    tokens = mesh["topology_tokens"]
    court = evaluate_decoded_topology_v2(tokens, config=cfg.tessa)
    if not court.grammar_valid:
        _fail("RSS_VNEXT_SURFACE_TESSA_TOKEN_GRAMMAR_INVALID")
    if (court.zero_area_face_count or court.nonmanifold_edge_count
        or not court.finite_geometry or not court.within_scale_policy):
        _fail("RSS_VNEXT_SURFACE_TESSA_TOPOLOGY_INVALID")
    decoded = decode_tessa_asset_sequence_v1(tokens, cfg=_codec_config(cfg.tessa))
    actual_faces = np.asarray(mesh["faces"], dtype=np.int64)
    decoded_faces = np.asarray(decoded.faces, dtype=np.int64)
    actual_base = np.asarray(mesh["base_xyz"], dtype=np.float64)
    decoded_base = np.asarray(decoded.vertices_normalized, dtype=np.float64) * 2.
    if (
        not np.array_equal(actual_faces, decoded_faces)
        or actual_base.shape != decoded_base.shape
        or not np.allclose(actual_base, decoded_base, rtol=0, atol=2.e-6)
        or tuple(mesh["vertex_components"]) != tuple(
            int(x) for x in decoded.vertex_component_indices
        )
        or tuple(mesh["face_components"]) != tuple(
            int(x) for x in decoded.face_component_indices
        )
    ):
        _fail("RSS_VNEXT_SURFACE_TESSA_TOKEN_MESH_DECODE_MISMATCH")
    # TESSA's decoded index topology is a proposal. It does NOT prove
    # source SDF sign, appendage independence, physical joint ownership,
    # multi-layer compositing or PartVerse segmentation correspondence.


def _computed_candidate_surface_normals(
    mesh: dict[str, Any],
) -> tuple[np.ndarray, tuple[tuple[float, float, float], ...], tuple[float, ...]]:
    vertices = np.asarray(mesh["rest_xyz"], dtype=np.float64)
    faces = np.asarray(mesh["faces"], dtype=np.int64)
    if (
        not np.isfinite(vertices).all()
        or vertices.ndim != 2 or vertices.shape[1] != 3
        or np.any(np.abs(vertices) > 1.000002)
    ):
        _fail("RSS_VNEXT_SURFACE_REFINED_COORDINATE_DOMAIN_INVALID")
    cross = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    length = np.linalg.norm(cross, axis=-1)
    if not bool(np.all(np.isfinite(length))) or bool(np.any(length <= 1.e-10)):
        _fail("RSS_VNEXT_SURFACE_REFINED_TRIANGLE_DEGENERATE")
    area = tuple(float(x) * .5 for x in length)
    tri_normals = tuple(tuple(float(v) for v in row / scale)
                        for row, scale in zip(cross, length))
    indexed_normals = np.zeros_like(vertices)
    for j in range(3):
        np.add.at(indexed_normals, faces[:, j], cross)
    norm = np.linalg.norm(indexed_normals, axis=-1, keepdims=True)
    expected = indexed_normals / np.maximum(norm, 1.e-8)
    observed = np.asarray(mesh["normals_xyz"], dtype=np.float64)
    if not np.allclose(expected, observed, rtol=0., atol=2.e-4):
        _fail("RSS_VNEXT_SURFACE_NEURAL_NORMALS_NOT_FROM_INDEXED_GEOMETRY")
    return vertices, tri_normals, area


def compile_rss_vnext_surface_candidates_v1(
    fanout: RSSvNextExpertEvidenceFanoutV1,
) -> RSSvNextComputedCandidateSurfaceV1:
    """Compiled COMPUTATIONS, not an authority escalation or representation loss."""
    if type(fanout) is not RSSvNextExpertEvidenceFanoutV1:
        _fail("RSS_VNEXT_SURFACE_EXPECTS_QUALIFIED_SHARED_FANOUT")
    evidence = fanout.evidence
    if evidence.mesh_status == "ABSTAIN":
        return RSSvNextComputedCandidateSurfaceV1(
            source_inference_evidence_hash=evidence.evidence_hash,
            neural_scene_carrier_sha256=fanout.neural_carrier_sha256,
            unified_model_config_sha256=fanout.model_config_sha256,
            candidate_mesh_sha256=None, topology_token_sequence_sha256=None,
            state="ABSTAIN_NO_VALID_TOPOLOGY",
            reason=evidence.reason, faces=(),
        )
    _replay_topology(evidence, fanout.model_config)
    mesh = evidence.mesh
    vertices, unit_normals, areas = _computed_candidate_surface_normals(mesh)
    region = candidate_face_region_readout_v1(evidence)
    if len(region) != len(mesh["faces"]):
        _fail("RSS_VNEXT_SURFACE_FACE_REGION_ADDRESS_DRIFT")
    faces = []
    for i, indices in enumerate(mesh["faces"]):
        idx = tuple(indices)
        row = region[i]
        if tuple(row["candidate_vertex_indices"]) != idx or row["candidate_face_index"] != i:
            _fail("RSS_VNEXT_SURFACE_FACE_REGION_ADDRESS_DRIFT")
        role = [0.] * len(mesh["presentation_role_logits"][0])
        for vi in idx:
            p = _softmax(mesh["presentation_role_logits"][vi])
            for j, value in enumerate(p):
                role[j] += value / 3.
        rays = mesh["source_rays"]
        faces.append(RSSvNextComputedCandidateFaceV1(
            candidate_face_index=i,
            candidate_vertex_indices=idx,
            candidate_component_index=int(mesh["face_components"][i]),
            area=areas[i],
            unit_normal_xyz=unit_normals[i],
            region_probabilities=tuple(row["candidate_region_probabilities"])
                                 + (float(row["unknown_probability"]),),
            presentation_role_probabilities=tuple(role),
            # Keep raw additive logits. The view-conditioned Graphics renderer
            # resolves their meaning; sigmoid(logit) is not source RGB.
            graphics_logits_by_corner=tuple(
                tuple(float(x) for x in mesh["graphics_logits"][vi]) for vi in idx
            ),
            affordance_logits_by_corner=tuple(
                tuple(float(x) for x in mesh["affordance_logits"][vi]) for vi in idx
            ),
            original_rgba_by_corner=tuple(
                tuple(tuple(float(x) for x in rays["rgba"][vi][source])
                      for source in range(2))
                for vi in idx
            ),
            original_projection_valid_by_corner=tuple(
                tuple(rays["valid"][vi]) for vi in idx
            ),
        ))
    return RSSvNextComputedCandidateSurfaceV1(
        source_inference_evidence_hash=evidence.evidence_hash,
        neural_scene_carrier_sha256=fanout.neural_carrier_sha256,
        unified_model_config_sha256=fanout.model_config_sha256,
        candidate_mesh_sha256=content_sha256(mesh),
        topology_token_sequence_sha256=content_sha256(mesh["topology_tokens"]),
        state="UNQUALIFIED_COMPUTED_SURFACE_CANDIDATE",
        reason="",
        faces=tuple(faces),
    )


__all__ = [
    "RSSvNextComputedCandidateFaceV1",
    "RSSvNextComputedCandidateSurfaceV1",
    "compile_rss_vnext_surface_candidates_v1",
]
