from __future__ import annotations

"""Bidirectional IRIS V3 relation evidence and fail-closed presentation compile.

IRIS V3 uses an ordered-pair parameterization. The architecture can represent
asymmetric outputs even for mathematical relations that should be symmetric.
Shipping therefore stores both directions explicitly and Compiler only acts when
both directions support the same legal decision. No neural confidence scalar is
minted or inferred.

This module intentionally keeps runtime-local interleaving out of V1. If opposite
precedence survives across qualified overlap samples for one fragment pair, the
compile fails closed instead of deleting a cycle or using face index as order.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping

from .graphics_v1 import (
    CompiledGraphicsV1,
    GraphicsAttachmentV1,
    GraphicsFragmentV1,
    GraphicsOrderEdgeV1,
    GraphicsViewStateV1,
    _UF,
    _adj,
    _prob,
    compiled_graphics_hash_v1,
)
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class IRISGraphicsStaticPairEvidenceV1:
    face_a: int
    face_b: int
    same_drawable_ab: float
    same_drawable_ba: float
    cut_ab: float
    cut_ba: float
    attachment_ab: float
    attachment_ba: float
    evidence_ref: str = ""
    schema_version: str = "RealSaS.IRISGraphicsStaticPairEvidence.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class IRISGraphicsViewPairEvidenceV1:
    view_index: int
    face_a: int
    face_b: int
    front_ab: float
    front_ba: float
    occlusion_ab: float
    occlusion_ba: float
    projected_alpha_boundary_ab: float
    projected_alpha_boundary_ba: float
    projected_appearance_boundary_ab: float
    projected_appearance_boundary_ba: float
    support_ref: str
    evidence_ref: str = ""
    schema_version: str = "RealSaS.IRISGraphicsViewPairEvidence.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class GraphicsFaceVisibilityEvidenceV1:
    view_index: int
    face_index: int
    visible: bool
    support_ref: str
    authority: str
    schema_version: str = "RealSaS.GraphicsFaceVisibilityEvidence.v1"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class IRISGraphicsEvidenceV1:
    mesh_binding_hash: str
    shared_h_transport_hash: str
    model_release_binding: str
    static_pairs: tuple[IRISGraphicsStaticPairEvidenceV1, ...]
    view_pairs: tuple[IRISGraphicsViewPairEvidenceV1, ...]
    visibility: tuple[GraphicsFaceVisibilityEvidenceV1, ...]
    evidence_hash: str
    schema_version: str = "RealSaS.IRISGraphicsEvidence.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


def iris_graphics_evidence_hash_v1(value: IRISGraphicsEvidenceV1) -> str:
    payload = value.to_dict()
    payload.pop("evidence_hash", None)
    return content_sha256(payload)


def build_iris_graphics_evidence_v1(
    *,
    mesh_binding_hash: str,
    shared_h_transport_hash: str,
    model_release_binding: str,
    static_pairs: Iterable[IRISGraphicsStaticPairEvidenceV1],
    view_pairs: Iterable[IRISGraphicsViewPairEvidenceV1],
    visibility: Iterable[GraphicsFaceVisibilityEvidenceV1],
    metadata: Mapping[str, Any] | None = None,
) -> IRISGraphicsEvidenceV1:
    if not mesh_binding_hash or not shared_h_transport_hash or not model_release_binding:
        raise QualificationError("IRIS_GRAPHICS_REQUIRED_BINDING_EMPTY")
    value = IRISGraphicsEvidenceV1(
        mesh_binding_hash=str(mesh_binding_hash),
        shared_h_transport_hash=str(shared_h_transport_hash),
        model_release_binding=str(model_release_binding),
        static_pairs=tuple(static_pairs),
        view_pairs=tuple(view_pairs),
        visibility=tuple(visibility),
        evidence_hash="",
        metadata={
            "authority_class": "IRIS_LEARNED_GRAPHICS_EVIDENCE_ONLY",
            "relation_head": "RealSaS.IRIS.GraphicsRelationHead.v3",
            "ordered_pair_outputs_preserved_bidirectionally": True,
            "neural_confidence_output": False,
            "pairwise_visibility_output": False,
            "categorical_semantics_used": False,
            "product_graphics_authority_claimed": False,
            **dict(metadata or {}),
        },
    )
    return replace(value, evidence_hash=iris_graphics_evidence_hash_v1(value))


@dataclass(frozen=True)
class GraphicsRelationCompilePolicyV1:
    same_drawable_min: float = 0.65
    cut_max: float = 0.35
    boundary_max: float = 0.50
    front_probability_min: float = 0.65
    occlusion_probability_min: float = 0.50
    attachment_affinity_min: float = 0.70
    mechanical_contact_min: float = 0.75
    support_band_alpha_boundary_min: float = 0.65
    maximum_support_band_pixels: int = 2

    def validate(self) -> None:
        values = (
            self.same_drawable_min,
            self.cut_max,
            self.boundary_max,
            self.front_probability_min,
            self.occlusion_probability_min,
            self.attachment_affinity_min,
            self.mechanical_contact_min,
            self.support_band_alpha_boundary_min,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in values):
            raise QualificationError("IRIS_GRAPHICS_POLICY_INVALID")
        if not 0 <= int(self.maximum_support_band_pixels) <= 4:
            raise QualificationError("IRIS_GRAPHICS_POLICY_INVALID")


def _check_evidence(value: IRISGraphicsEvidenceV1, face_count: int) -> None:
    if value.evidence_hash != iris_graphics_evidence_hash_v1(value):
        raise QualificationError("IRIS_GRAPHICS_EVIDENCE_HASH_DRIFT")
    md = dict(value.metadata or {})
    if md.get("ordered_pair_outputs_preserved_bidirectionally") is not True:
        raise QualificationError("IRIS_GRAPHICS_BIDIRECTIONAL_EVIDENCE_REQUIRED")
    if md.get("neural_confidence_output") is not False:
        raise QualificationError("IRIS_GRAPHICS_NEURAL_CONFIDENCE_FORBIDDEN")
    if md.get("pairwise_visibility_output") is not False:
        raise QualificationError("IRIS_GRAPHICS_PAIRWISE_VISIBILITY_FORBIDDEN")
    if md.get("product_graphics_authority_claimed") is not False:
        raise QualificationError("IRIS_GRAPHICS_PREMATURE_AUTHORITY")
    for row in value.static_pairs:
        if row.face_a == row.face_b or min(row.face_a, row.face_b) < 0 or max(row.face_a, row.face_b) >= face_count:
            raise QualificationError("IRIS_GRAPHICS_STATIC_FACE_INVALID")
        for number in (
            row.same_drawable_ab, row.same_drawable_ba, row.cut_ab, row.cut_ba,
            row.attachment_ab, row.attachment_ba,
        ):
            _prob(number, "static")
    for row in value.view_pairs:
        if row.face_a == row.face_b or min(row.face_a, row.face_b) < 0 or max(row.face_a, row.face_b) >= face_count:
            raise QualificationError("IRIS_GRAPHICS_VIEW_FACE_INVALID")
        if row.view_index < 0 or not row.support_ref:
            raise QualificationError("IRIS_GRAPHICS_VIEW_SUPPORT_INVALID")
        for number in (
            row.front_ab, row.front_ba, row.occlusion_ab, row.occlusion_ba,
            row.projected_alpha_boundary_ab, row.projected_alpha_boundary_ba,
            row.projected_appearance_boundary_ab, row.projected_appearance_boundary_ba,
        ):
            _prob(number, "view")
    for row in value.visibility:
        if row.view_index < 0 or row.face_index < 0 or row.face_index >= face_count:
            raise QualificationError("IRIS_GRAPHICS_VISIBILITY_FACE_INVALID")
        if type(row.visible) is not bool or not row.support_ref or not row.authority:
            raise QualificationError("IRIS_GRAPHICS_VISIBILITY_SUPPORT_INVALID")


def compile_iris_graphics_v1(
    *,
    canonical_mesh_binding_hash: str,
    canonical_mesh_skin_binding_hash: str,
    source_graphics_binding_hash: str,
    faces: Iterable[Iterable[int]],
    evidence: IRISGraphicsEvidenceV1,
    mechanical_attachment_evidence: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    policy: GraphicsRelationCompilePolicyV1 = GraphicsRelationCompilePolicyV1(),
) -> CompiledGraphicsV1:
    policy.validate()
    triangles = tuple(tuple(map(int, face)) for face in faces)
    if not triangles or any(len(face) != 3 for face in triangles):
        raise QualificationError("IRIS_GRAPHICS_CANONICAL_MESH_INVALID")
    if evidence.mesh_binding_hash != canonical_mesh_binding_hash:
        raise QualificationError("IRIS_GRAPHICS_CANONICAL_MESH_BINDING_DRIFT")
    if not canonical_mesh_skin_binding_hash or not source_graphics_binding_hash:
        raise QualificationError("IRIS_GRAPHICS_REQUIRED_BINDING_EMPTY")
    _check_evidence(evidence, len(triangles))

    adjacency = _adj(triangles)
    boundary_by_pair: dict[tuple[int, int], float] = {}
    for row in evidence.view_pairs:
        pair = tuple(sorted((row.face_a, row.face_b)))
        boundary_by_pair[pair] = max(
            boundary_by_pair.get(pair, 0.0),
            row.projected_alpha_boundary_ab,
            row.projected_alpha_boundary_ba,
            row.projected_appearance_boundary_ab,
            row.projected_appearance_boundary_ba,
        )

    union = _UF(len(triangles))
    static_by_pair: dict[tuple[int, int], IRISGraphicsStaticPairEvidenceV1] = {}
    for row in evidence.static_pairs:
        pair = tuple(sorted((row.face_a, row.face_b)))
        if pair in static_by_pair:
            raise QualificationError("IRIS_GRAPHICS_STATIC_PAIR_DUPLICATE")
        static_by_pair[pair] = row
        if pair not in adjacency:
            continue
        if (
            min(row.same_drawable_ab, row.same_drawable_ba) >= policy.same_drawable_min
            and max(row.cut_ab, row.cut_ba) <= policy.cut_max
            and boundary_by_pair.get(pair, 0.0) <= policy.boundary_max
        ):
            union.union(*pair)

    groups: dict[int, list[int]] = {}
    for face_index in range(len(triangles)):
        groups.setdefault(union.find(face_index), []).append(face_index)
    fragments: list[GraphicsFragmentV1] = []
    face_fragment: dict[int, str] = {}
    for rows in sorted(groups.values(), key=min):
        face_indices = tuple(sorted(rows))
        fragment_id = "GFX:" + content_sha256(
            {"mesh": canonical_mesh_binding_hash, "faces": face_indices}
        )[:24]
        max_boundary = max(
            [boundary_by_pair.get(pair, 0.0) for pair in boundary_by_pair if pair[0] in face_indices or pair[1] in face_indices]
            or [0.0]
        )
        support_band = policy.maximum_support_band_pixels if max_boundary >= policy.support_band_alpha_boundary_min else 0
        fragments.append(
            GraphicsFragmentV1(
                fragment_id=fragment_id,
                face_indices=face_indices,
                support_band_pixels=int(support_band),
                metadata={
                    "canonical_mesh_only": True,
                    "categorical_identity": None,
                    "support_band_changes_mechanical_geometry": False,
                },
            )
        )
        for face_index in face_indices:
            face_fragment[face_index] = fragment_id

    # Visibility is an independent typed channel. Missing rows never imply visible.
    visible_by_view: dict[int, set[str]] = {}
    for row in evidence.visibility:
        if row.visible:
            visible_by_view.setdefault(row.view_index, set()).add(face_fragment[row.face_index])
        else:
            visible_by_view.setdefault(row.view_index, set())

    # Keep one global order per view only when all qualified local supports agree.
    directed_support: dict[tuple[int, str, str], list[float]] = {}
    conflicting: set[tuple[int, frozenset[str]]] = set()
    for row in evidence.view_pairs:
        fa, fb = face_fragment[row.face_a], face_fragment[row.face_b]
        if fa == fb:
            continue
        occlusion_supported = min(row.occlusion_ab, row.occlusion_ba) >= policy.occlusion_probability_min
        if not occlusion_supported:
            continue
        ab_supported = (
            row.front_ab >= policy.front_probability_min
            and row.front_ba <= 1.0 - policy.front_probability_min
        )
        ba_supported = (
            row.front_ba >= policy.front_probability_min
            and row.front_ab <= 1.0 - policy.front_probability_min
        )
        unordered = frozenset((fa, fb))
        if ab_supported and ba_supported:
            raise QualificationError("IRIS_GRAPHICS_FRONT_ALGEBRA_CONTRADICTION")
        if not ab_supported and not ba_supported:
            continue
        front, back = (fa, fb) if ab_supported else (fb, fa)
        reverse_key = (row.view_index, back, front)
        if reverse_key in directed_support:
            conflicting.add((row.view_index, unordered))
        support = min(
            row.occlusion_ab,
            row.occlusion_ba,
            row.front_ab if ab_supported else row.front_ba,
            1.0 - (row.front_ba if ab_supported else row.front_ab),
        )
        directed_support.setdefault((row.view_index, front, back), []).append(float(support))

    if conflicting:
        raise QualificationError("IRIS_GRAPHICS_LOCAL_INTERLEAVING_RUNTIME_SUPPORT_REQUIRED")

    views = sorted(set(visible_by_view) | {row.view_index for row in evidence.view_pairs})
    view_states: list[GraphicsViewStateV1] = []
    for view_index in views:
        edges = []
        for (candidate_view, front, back), values in sorted(directed_support.items()):
            if candidate_view != view_index:
                continue
            # Historical field name only. This is a Compiler conservative support
            # score, not a neural confidence output.
            support = min(values)
            edges.append(
                GraphicsOrderEdgeV1(
                    view_index=view_index,
                    front_fragment_id=front,
                    back_fragment_id=back,
                    confidence=float(support),
                    evidence_refs=(),
                )
            )
        # A cycle means some local supports cannot be represented by V1's global
        # DAG. Do not delete the weakest edge; V2 executable support is required.
        graph: dict[str, set[str]] = {}
        indegree: dict[str, int] = {}
        for edge in edges:
            graph.setdefault(edge.front_fragment_id, set()).add(edge.back_fragment_id)
            indegree.setdefault(edge.front_fragment_id, 0)
            indegree[edge.back_fragment_id] = indegree.get(edge.back_fragment_id, 0) + 1
        ready = [node for node, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for other in graph.get(node, ()):
                indegree[other] -= 1
                if indegree[other] == 0:
                    ready.append(other)
        if indegree and visited != len(indegree):
            raise QualificationError("IRIS_GRAPHICS_LOCAL_INTERLEAVING_RUNTIME_SUPPORT_REQUIRED")
        view_states.append(
            GraphicsViewStateV1(
                view_index=view_index,
                visible_fragment_ids=tuple(sorted(visible_by_view.get(view_index, set()))),
                order_edges=tuple(edges),
                metadata={
                    "pairwise_visibility_used": False,
                    "missing_visibility_implies_visible": False,
                    "local_interleaving_policy": "FAIL_CLOSED_UNTIL_EXECUTABLE_V2_SUPPORT",
                },
            )
        )

    attachments: list[GraphicsAttachmentV1] = []
    mechanical = dict(mechanical_attachment_evidence or {})
    for pair, row in static_by_pair.items():
        affinity = min(row.attachment_ab, row.attachment_ba)
        if affinity < policy.attachment_affinity_min:
            continue
        mechanical_row = dict(mechanical.get(pair) or {})
        contact = float(mechanical_row.get("persistent_contact", 0.0)) if mechanical_row else 0.0
        if contact < policy.mechanical_contact_min:
            continue
        fa, fb = face_fragment[pair[0]], face_fragment[pair[1]]
        if fa == fb:
            continue
        barycentric = tuple(map(float, mechanical_row.get("barycentric_anchor") or ()))
        anchor_face = int(mechanical_row.get("anchor_face_index", -1))
        parent = str(mechanical_row.get("parent_mechanical_domain_id") or "")
        if len(barycentric) != 3 or abs(sum(barycentric) - 1.0) > 1.0e-6 or any(value < 0.0 for value in barycentric):
            raise QualificationError("IRIS_GRAPHICS_ATTACHMENT_BARYCENTRIC_INVALID")
        if not 0 <= anchor_face < len(triangles) or not parent:
            raise QualificationError("IRIS_GRAPHICS_ATTACHMENT_BINDING_INVALID")
        attachments.append(
            GraphicsAttachmentV1(
                fragment_id=fa,
                parent_mechanical_domain_id=parent,
                anchor_face_index=anchor_face,
                barycentric_anchor=barycentric,
                confidence=float(min(affinity, contact)),
                evidence_refs=tuple(
                    value for value in (row.evidence_ref, str(mechanical_row.get("evidence_ref") or "")) if value
                ),
            )
        )

    value = CompiledGraphicsV1(
        canonical_mesh_binding_hash=str(canonical_mesh_binding_hash),
        canonical_mesh_skin_binding_hash=str(canonical_mesh_skin_binding_hash),
        source_graphics_binding_hash=str(source_graphics_binding_hash),
        graphics_evidence_binding_hash=evidence.evidence_hash,
        fragments=tuple(fragments),
        views=tuple(view_states),
        attachments=tuple(attachments),
        graph_hash="",
        metadata={
            "authority_class": "COMPILED_GRAPHICS",
            "evidence_schema": "RealSaS.IRISGraphicsEvidence.v1",
            "relation_head": "RealSaS.IRIS.GraphicsRelationHead.v3",
            "single_canonical_mesh_authority": True,
            "independent_visual_mesh_allowed": False,
            "categorical_semantics_used": False,
            "source_pixels_win_where_observed": True,
            "runtime_consumption_required": True,
            "triangle_array_order_is_presentation_authority": False,
            "silhouette_support_is_2d_only": True,
            "neural_confidence_used": False,
            "pairwise_visibility_used": False,
            "historical_confidence_fields_semantics": "COMPILER_CONSERVATIVE_SUPPORT_SCORE",
            "local_interleaving_without_executable_support": "FAIL_CLOSED",
        },
    )
    return replace(value, graph_hash=compiled_graphics_hash_v1(value))


__all__ = [
    "GraphicsFaceVisibilityEvidenceV1",
    "GraphicsRelationCompilePolicyV1",
    "IRISGraphicsEvidenceV1",
    "IRISGraphicsStaticPairEvidenceV1",
    "IRISGraphicsViewPairEvidenceV1",
    "build_iris_graphics_evidence_v1",
    "compile_iris_graphics_v1",
    "iris_graphics_evidence_hash_v1",
]
