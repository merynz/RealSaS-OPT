from __future__ import annotations

"""Sparse, non-authoritative RSS 3D face-to-face relation computation.

The expert predicts directed R×R region relationships, NOT F×F source face
labels. Compiler marginalizes anonymous slot membership at requested
candidate face pairs; it never enumerates a dense quadratic face-pair graph,
guesses anchors, discards UNKNOWN mass, or elevates learned affinities to
editable drawables. Production contact/visibility has separate Source Truth
courts and must not be inferred from categorical slot names.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_expert_fanout_v1 import (
    RSSvNextExpertEvidenceFanoutV1,
)
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    candidate_face_region_readout_v1,
)
from compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1 import (
    RSSvNextComputedCandidateSurfaceV1,
)
from compiler.realsas_compiler_core.types import QualificationError

SCHEMA = "RealSaS.RSSvNext.SparseCandidateFaceRelations.v1"
_MIN_KNOWN_CROSS_SLOT = 1.e-6


@dataclass(frozen=True)
class RSSvNextSparseFaceRelationV1:
    from_face_index: int
    to_face_index: int
    from_unknown_probability: float
    to_unknown_probability: float
    known_cross_slot_support: float
    same_drawable_affinity: float | None
    cut_affinity: float | None
    directed_attachment_affinity: float | None
    state: str
    # Strictly candidates: geometry may be compiled, mechanics not.
    source_drawable_qualified: bool = False
    source_separability_qualified: bool = False
    source_contact_qualified: bool = False
    mechanical_parent_anchor_qualified: bool = False
    source_visibility_qualified: bool = False


@dataclass(frozen=True)
class RSSvNextSparseFaceRelationScheduleV1:
    inference_evidence_hash: str
    unified_model_config_sha256: str
    neural_scene_carrier_sha256: str
    candidate_mesh_sha256: str | None
    face_relations: tuple[RSSvNextSparseFaceRelationV1, ...]
    schema: str = SCHEMA
    independent_drawable_authority: bool = False
    attachment_authority: bool = False
    pose_conditioned_presentation_authority: bool = False


def compile_rss_vnext_sparse_face_relations_v1(
    fanout: RSSvNextExpertEvidenceFanoutV1,
    surface: RSSvNextComputedCandidateSurfaceV1,
    *,
    requested_face_pairs: Iterable[tuple[int, int]],
) -> RSSvNextSparseFaceRelationScheduleV1:
    if (
        type(fanout) is not RSSvNextExpertEvidenceFanoutV1
        or type(surface) is not RSSvNextComputedCandidateSurfaceV1
        or surface.source_inference_evidence_hash != fanout.evidence.evidence_hash
        or surface.neural_scene_carrier_sha256 != fanout.neural_carrier_sha256
        or surface.unified_model_config_sha256 != fanout.model_config_sha256
    ):
        raise QualificationError("RSS_VNEXT_FACE_RELATION_PARENT_BINDING_DRIFT")
    # A finite, explicit candidate pair list is a *caller choice*, not a
    # learned full-coverage graph. Avoid F×F auto expansion and hidden
    # random cutoffs; GPU model proposals may have millions of face pairs.
    if type(requested_face_pairs) not in (tuple, list):
        raise QualificationError("RSS_VNEXT_FACE_RELATION_PAIR_LIST_REQUIRED")
    if len(requested_face_pairs) > fanout.model_config.max_pair_queries:
        raise QualificationError("RSS_VNEXT_FACE_RELATION_RESOURCE_BUDGET")
    if fanout.evidence.mesh_status == "ABSTAIN":
        if requested_face_pairs or surface.state != "ABSTAIN_NO_VALID_TOPOLOGY":
            raise QualificationError("RSS_VNEXT_FACE_RELATION_ABSTAIN_REQUIRES_NO_FACES")
        return RSSvNextSparseFaceRelationScheduleV1(
            inference_evidence_hash=fanout.evidence.evidence_hash,
            unified_model_config_sha256=fanout.model_config_sha256,
            neural_scene_carrier_sha256=fanout.neural_carrier_sha256,
            candidate_mesh_sha256=None,
            face_relations=(),
        )
    if (
        surface.state != "UNQUALIFIED_COMPUTED_SURFACE_CANDIDATE"
        or surface.candidate_mesh_sha256 != content_sha256(fanout.evidence.mesh)
    ):
        raise QualificationError("RSS_VNEXT_FACE_RELATION_UNQUALIFIED_MESH_BINDING")
    faces = surface.faces
    nfaces = len(faces)
    if not nfaces or nfaces != len(fanout.evidence.mesh["faces"]):
        raise QualificationError("RSS_VNEXT_FACE_RELATION_FACE_CARDINALITY_DRIFT")
    # Reject a manually re-sealed candidate surface with fabricated region
    # probabilities rather than accepting its convenient scalar summary.
    truth = candidate_face_region_readout_v1(fanout.evidence)
    for i, face in enumerate(faces):
        expected = tuple(truth[i]["candidate_region_probabilities"]) + (
            truth[i]["unknown_probability"],
        )
        if (
            face.candidate_face_index != i
            or face.candidate_vertex_indices != tuple(fanout.evidence.mesh["faces"][i])
            or face.candidate_component_index != fanout.evidence.mesh["face_components"][i]
            or len(face.region_probabilities) != len(expected)
            or not np.allclose(face.region_probabilities, expected, rtol=0., atol=1.e-7)
        ):
            raise QualificationError("RSS_VNEXT_FACE_RELATION_SOURCE_FACE_MEMBERSHIP_DRIFT")
    unique = set()
    for pair in requested_face_pairs:
        if (
            type(pair) not in (tuple, list) or len(pair) != 2
            or any(type(v) is not int or not 0 <= v < nfaces for v in pair)
            or pair[0] == pair[1]
        ):
            raise QualificationError("RSS_VNEXT_FACE_RELATION_PAIR_DOMAIN_INVALID")
        key = tuple(pair)
        if key in unique:
            raise QualificationError("RSS_VNEXT_FACE_RELATION_PAIR_DUPLICATE")
        unique.add(key)

    # Only R×R typed relation tensor is copied once; NEVER the full neural H,
    # scene triplanes, render rays, or N×N/Nfaces² score matrices.
    region = fanout.region_relation_matrix_v1()
    raw = region.detach().float().cpu().numpy()
    if raw.shape != (fanout.region_slot_count, fanout.region_slot_count, 3):
        raise QualificationError("RSS_VNEXT_FACE_RELATION_REGION_GRAPH_DRIFT")
    p = 1. / (1. + np.exp(-np.clip(raw, -64., 64.)))
    diag = np.eye(fanout.region_slot_count, dtype=np.bool_)
    p[diag] = 0.
    results = []
    for fa, fb in requested_face_pairs:
        a = faces[fa].region_probabilities
        b = faces[fb].region_probabilities
        joint = np.outer(a[:-1], b[:-1])
        joint[diag] = 0.
        support = float(joint.sum())
        if support < _MIN_KNOWN_CROSS_SLOT:
            scores = (None, None, None)
            status = "ABSTAIN_NO_CROSS_SLOT_KNOWN_MASS"
        else:
            scores = tuple(float(v) for v in (
                joint[:, :, None] * p
            ).sum(axis=(0, 1)) / support)
            status = "UNQUALIFIED_SPARSE_LEARNED_PAIR"
        results.append(RSSvNextSparseFaceRelationV1(
            from_face_index=fa,
            to_face_index=fb,
            from_unknown_probability=float(a[-1]),
            to_unknown_probability=float(b[-1]),
            known_cross_slot_support=support,
            same_drawable_affinity=scores[0],
            cut_affinity=scores[1],
            directed_attachment_affinity=scores[2],
            state=status,
        ))
    return RSSvNextSparseFaceRelationScheduleV1(
        inference_evidence_hash=fanout.evidence.evidence_hash,
        unified_model_config_sha256=fanout.model_config_sha256,
        neural_scene_carrier_sha256=fanout.neural_carrier_sha256,
        candidate_mesh_sha256=surface.candidate_mesh_sha256,
        face_relations=tuple(results),
    )


__all__ = [
    "RSSvNextSparseFaceRelationV1",
    "RSSvNextSparseFaceRelationScheduleV1",
    "compile_rss_vnext_sparse_face_relations_v1",
]
