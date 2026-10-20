from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping

from .graphics_v1 import (
    GraphicsAttachmentV1,
    GraphicsFragmentV1,
    GraphicsOrderEdgeV1,
    GraphicsViewStateV1,
    _UF,
    _adj,
    _dag,
)
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class GraphicsCompilePolicyV2:
    same_drawable_min: float = 0.65
    cut_max: float = 0.35
    boundary_max: float = 0.50
    front_probability_min: float = 0.65
    occlusion_probability_min: float = 0.50
    visibility_probability_min: float = 0.50
    attachment_affinity_min: float = 0.70
    mechanical_contact_min: float = 0.75
    support_band_alpha_boundary_min: float = 0.65
    maximum_support_band_pixels: int = 2

    def validate(self) -> None:
        vals = (
            self.same_drawable_min,
            self.cut_max,
            self.boundary_max,
            self.front_probability_min,
            self.occlusion_probability_min,
            self.visibility_probability_min,
            self.attachment_affinity_min,
            self.mechanical_contact_min,
            self.support_band_alpha_boundary_min,
        )
        if any(not 0.0 <= float(v) <= 1.0 for v in vals):
            raise QualificationError("GRAPHICS_V2_POLICY_INVALID")
        if not 0 <= int(self.maximum_support_band_pixels) <= 4:
            raise QualificationError("GRAPHICS_V2_POLICY_INVALID")


@dataclass(frozen=True)
class GraphicsStaticRelationEvidenceV2:
    face_a: int
    face_b: int
    same_drawable_affinity: float = 0.0
    same_drawable_valid: bool = False
    cut_probability: float = 0.0
    cut_valid: bool = False
    attachment_affinity: float = 0.0
    attachment_valid: bool = False
    alpha_boundary_probability: float = 0.0
    alpha_boundary_valid: bool = False
    appearance_boundary_probability: float = 0.0
    appearance_boundary_valid: bool = False
    evidence_ref: str = ""
    schema_version: str = "RealSaS.GraphicsStaticRelationEvidence.v2"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class GraphicsViewPairEvidenceV2:
    view_index: int
    face_a: int
    face_b: int
    front_probability: float = 0.0
    front_valid: bool = False
    occlusion_probability: float = 0.0
    occlusion_valid: bool = False
    evidence_ref: str = ""
    schema_version: str = "RealSaS.GraphicsViewPairEvidence.v2"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class GraphicsGlobalVisibilityEvidenceV2:
    """Per-face visibility produced from a complete ray/view authority.

    `valid=False` is an abstention, not an invisible label.  Pairwise visibility
    is intentionally absent from V2.
    """

    view_index: int
    face_index: int
    visibility_probability: float = 0.0
    valid: bool = False
    complete_stack: bool = False
    evidence_ref: str = ""
    schema_version: str = "RealSaS.GraphicsGlobalVisibilityEvidence.v2"

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class GraphicsEvidenceV2:
    mesh_binding_hash: str
    static_relations: tuple[GraphicsStaticRelationEvidenceV2, ...]
    view_pair_relations: tuple[GraphicsViewPairEvidenceV2, ...]
    global_visibility: tuple[GraphicsGlobalVisibilityEvidenceV2, ...]
    evidence_hash: str
    schema_version: str = "RealSaS.GraphicsEvidence.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


@dataclass(frozen=True)
class CompiledGraphicsV2:
    canonical_mesh_binding_hash: str
    canonical_mesh_skin_binding_hash: str
    source_graphics_binding_hash: str
    graphics_evidence_binding_hash: str
    fragments: tuple[GraphicsFragmentV1, ...]
    views: tuple[GraphicsViewStateV1, ...]
    attachments: tuple[GraphicsAttachmentV1, ...]
    graph_hash: str
    schema_version: str = "RealSaS.CompiledGraphics.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> Json:
        return asdict(self)


def _hash(payload: Json, key: str) -> str:
    body = dict(payload)
    body.pop(key, None)
    return content_sha256(body)


def graphics_evidence_hash_v2(value: GraphicsEvidenceV2) -> str:
    return _hash(value.to_dict(), "evidence_hash")


def compiled_graphics_hash_v2(value: CompiledGraphicsV2) -> str:
    return _hash(value.to_dict(), "graph_hash")


def _prob(value: float, label: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise QualificationError("GRAPHICS_V2_PROBABILITY_INVALID:" + label)
    return value


def build_graphics_evidence_v2(
    *,
    mesh_binding_hash: str,
    static_relations: Iterable[GraphicsStaticRelationEvidenceV2] = (),
    view_pair_relations: Iterable[GraphicsViewPairEvidenceV2] = (),
    global_visibility: Iterable[GraphicsGlobalVisibilityEvidenceV2] = (),
    metadata: Mapping[str, Any] | None = None,
) -> GraphicsEvidenceV2:
    static = tuple(static_relations)
    pair = tuple(view_pair_relations)
    visibility = tuple(global_visibility)
    if not mesh_binding_hash or not (static or pair or visibility):
        raise QualificationError("GRAPHICS_V2_EVIDENCE_INCOMPLETE")
    value = GraphicsEvidenceV2(
        mesh_binding_hash=str(mesh_binding_hash),
        static_relations=static,
        view_pair_relations=pair,
        global_visibility=visibility,
        evidence_hash="",
        metadata={
            "categorical_semantics_used": False,
            "pairwise_visibility_used": False,
            "global_visibility_requires_complete_stack": True,
            "uncalibrated_neural_confidence_used": False,
            "unknown_is_abstain": True,
            **dict(metadata or {}),
        },
    )
    return replace(value, evidence_hash=graphics_evidence_hash_v2(value))


def compile_graphics_v2(
    *,
    canonical_mesh_binding_hash: str,
    canonical_mesh_skin_binding_hash: str,
    source_graphics_binding_hash: str,
    faces: Iterable[Iterable[int]],
    evidence: GraphicsEvidenceV2,
    mechanical_attachment_evidence: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    policy: GraphicsCompilePolicyV2 = GraphicsCompilePolicyV2(),
) -> CompiledGraphicsV2:
    """Compile role-free evidence while keeping pair order and visibility separate."""

    policy.validate()
    if evidence.evidence_hash != graphics_evidence_hash_v2(evidence):
        raise QualificationError("GRAPHICS_V2_EVIDENCE_HASH_DRIFT")
    if evidence.mesh_binding_hash != canonical_mesh_binding_hash:
        raise QualificationError("GRAPHICS_V2_CANONICAL_MESH_BINDING_DRIFT")
    if not canonical_mesh_skin_binding_hash or not source_graphics_binding_hash:
        raise QualificationError("GRAPHICS_V2_REQUIRED_BINDING_EMPTY")

    tri = tuple(tuple(map(int, face)) for face in faces)
    n = len(tri)
    if not n or any(len(face) != 3 for face in tri):
        raise QualificationError("GRAPHICS_V2_CANONICAL_MESH_INVALID")

    # Static evidence may merge adjacent canonical faces into persistent
    # presentation fragments.  Missing evidence abstains and therefore cannot
    # fabricate a merge/cut decision.
    uf = _UF(n)
    adjacency = _adj(tri)
    static_by_pair: dict[tuple[int, int], GraphicsStaticRelationEvidenceV2] = {}
    alpha = [0.0] * n
    for row in evidence.static_relations:
        a, b = int(row.face_a), int(row.face_b)
        if a == b or min(a, b) < 0 or max(a, b) >= n:
            raise QualificationError("GRAPHICS_V2_STATIC_RELATION_FACE_INVALID")
        pair = tuple(sorted((a, b)))
        if pair in static_by_pair:
            raise QualificationError("GRAPHICS_V2_STATIC_RELATION_DUPLICATE")
        static_by_pair[pair] = row

        if row.alpha_boundary_valid:
            boundary = _prob(row.alpha_boundary_probability, "alpha")
            for face_index in pair:
                alpha[face_index] = max(alpha[face_index], boundary)

        merge_valid = (
            row.same_drawable_valid
            and row.cut_valid
            and row.alpha_boundary_valid
            and row.appearance_boundary_valid
        )
        if not merge_valid or pair not in adjacency:
            continue
        same = _prob(row.same_drawable_affinity, "same")
        cut = _prob(row.cut_probability, "cut")
        alpha_p = _prob(row.alpha_boundary_probability, "alpha")
        appearance_p = _prob(row.appearance_boundary_probability, "appearance")
        if (
            same >= policy.same_drawable_min
            and cut <= policy.cut_max
            and max(alpha_p, appearance_p) <= policy.boundary_max
        ):
            uf.union(a, b)

    groups: dict[int, list[int]] = {}
    for face_index in range(n):
        groups.setdefault(uf.find(face_index), []).append(face_index)

    fragments: list[GraphicsFragmentV1] = []
    face_fragment: dict[int, str] = {}
    for rows in sorted(groups.values(), key=min):
        indices = tuple(sorted(rows))
        fragment_id = "GFX:" + content_sha256(
            {"mesh": canonical_mesh_binding_hash, "faces": indices}
        )[:24]
        support = (
            policy.maximum_support_band_pixels
            if max(alpha[index] for index in indices) >= policy.support_band_alpha_boundary_min
            else 0
        )
        fragment = GraphicsFragmentV1(
            fragment_id,
            indices,
            int(support),
            metadata={
                "canonical_mesh_only": True,
                "categorical_identity": None,
                "support_band_changes_mechanical_geometry": False,
                "static_unknown_does_not_merge": True,
            },
        )
        fragments.append(fragment)
        for index in indices:
            face_fragment[index] = fragment_id

    # Pair order evidence is view-conditioned but never supplies visibility.
    pair_rows: dict[tuple[int, str, str], list[tuple[float, float, str]]] = {}
    views_present: set[int] = set()
    for row in evidence.view_pair_relations:
        view_index = int(row.view_index)
        a, b = int(row.face_a), int(row.face_b)
        views_present.add(view_index)
        if view_index < 0 or a == b or min(a, b) < 0 or max(a, b) >= n:
            raise QualificationError("GRAPHICS_V2_VIEW_PAIR_INVALID")
        if not (row.front_valid and row.occlusion_valid):
            continue
        front = _prob(row.front_probability, "front")
        occlusion = _prob(row.occlusion_probability, "occlusion")
        if occlusion < policy.occlusion_probability_min:
            continue
        fragment_a, fragment_b = face_fragment[a], face_fragment[b]
        if fragment_a == fragment_b:
            continue
        low, high = sorted((fragment_a, fragment_b))
        score = front if fragment_a == low else 1.0 - front
        # No free neural confidence exists.  The edge strength is derived only
        # from admitted probabilities used by this decision.
        strength = min(max(score, 1.0 - score), occlusion)
        pair_rows.setdefault((view_index, low, high), []).append(
            (score, strength, row.evidence_ref)
        )

    # Global visibility comes only from complete-stack evidence.  Invalid rows
    # abstain.  If no valid visibility is available for a fragment, it remains
    # visible by default rather than converting unknown into hidden.
    visibility_by_face: dict[tuple[int, int], list[float]] = {}
    for row in evidence.global_visibility:
        view_index = int(row.view_index)
        face_index = int(row.face_index)
        views_present.add(view_index)
        if view_index < 0 or face_index < 0 or face_index >= n:
            raise QualificationError("GRAPHICS_V2_VISIBILITY_FACE_INVALID")
        if row.valid and not row.complete_stack:
            raise QualificationError("GRAPHICS_V2_VISIBILITY_REQUIRES_COMPLETE_STACK")
        if not row.valid:
            continue
        visibility_by_face.setdefault((view_index, face_index), []).append(
            _prob(row.visibility_probability, "global_visibility")
        )

    view_states: list[GraphicsViewStateV1] = []
    for view_index in sorted(views_present):
        raw_edges: list[GraphicsOrderEdgeV1] = []
        for (row_view, low, high), rows in sorted(pair_rows.items()):
            if row_view != view_index:
                continue
            weight = sum(item[1] for item in rows)
            if weight <= 0.0:
                continue
            score = sum(item[0] * item[1] for item in rows) / weight
            if score >= policy.front_probability_min:
                front_id, back_id, p = low, high, score
            elif score <= 1.0 - policy.front_probability_min:
                front_id, back_id, p = high, low, 1.0 - score
            else:
                continue
            raw_edges.append(
                GraphicsOrderEdgeV1(
                    view_index,
                    front_id,
                    back_id,
                    float(p),
                    tuple(sorted({item[2] for item in rows if item[2]})),
                )
            )
        kept, removed = _dag(raw_edges)

        visible: list[str] = []
        unknown_visibility: list[str] = []
        for fragment in fragments:
            admitted: list[float] = []
            for face_index in fragment.face_indices:
                admitted.extend(visibility_by_face.get((view_index, face_index), ()))
            if not admitted:
                visible.append(fragment.fragment_id)
                unknown_visibility.append(fragment.fragment_id)
                continue
            # Any visible canonical face makes the presentation fragment visible.
            if max(admitted) >= policy.visibility_probability_min:
                visible.append(fragment.fragment_id)

        view_states.append(
            GraphicsViewStateV1(
                view_index,
                tuple(sorted(visible)),
                kept,
                metadata={
                    "cycle_edges_removed": [edge.to_dict() for edge in removed],
                    "triangle_array_order_is_presentation_authority": False,
                    "pairwise_visibility_used": False,
                    "global_visibility_authority": "COMPLETE_RAY_STACK_OR_ABSTAIN",
                    "unknown_visibility_defaults_visible": True,
                    "unknown_visibility_fragment_ids": tuple(sorted(unknown_visibility)),
                },
            )
        )

    attachments: list[GraphicsAttachmentV1] = []
    mechanical = dict(mechanical_attachment_evidence or {})
    for pair, row in static_by_pair.items():
        if not row.attachment_valid:
            continue
        affinity = _prob(row.attachment_affinity, "attachment")
        if affinity < policy.attachment_affinity_min:
            continue
        mech = dict(mechanical.get(pair) or {})
        if not mech or float(mech.get("persistent_contact", 0.0)) < policy.mechanical_contact_min:
            continue
        fragment_a, fragment_b = face_fragment[pair[0]], face_fragment[pair[1]]
        if fragment_a == fragment_b:
            continue
        bary = tuple(map(float, mech.get("barycentric_anchor") or ()))
        anchor = int(mech.get("anchor_face_index", -1))
        parent = str(mech.get("parent_mechanical_domain_id") or "")
        if len(bary) != 3 or any(value < -1.0e-9 for value in bary) or abs(sum(bary) - 1.0) > 1.0e-6:
            raise QualificationError("GRAPHICS_V2_ATTACHMENT_BARYCENTRIC_INVALID")
        if not 0 <= anchor < n or not parent:
            raise QualificationError("GRAPHICS_V2_ATTACHMENT_BINDING_INVALID")
        contact = float(mech["persistent_contact"])
        attachments.append(
            GraphicsAttachmentV1(
                fragment_a,
                parent,
                anchor,
                bary,
                min(affinity, contact),
                tuple(
                    value
                    for value in (row.evidence_ref, str(mech.get("evidence_ref") or ""))
                    if value
                ),
            )
        )

    value = CompiledGraphicsV2(
        str(canonical_mesh_binding_hash),
        str(canonical_mesh_skin_binding_hash),
        str(source_graphics_binding_hash),
        evidence.evidence_hash,
        tuple(fragments),
        tuple(view_states),
        tuple(attachments),
        "",
        metadata={
            "authority_class": "COMPILED_GRAPHICS",
            "single_canonical_mesh_authority": True,
            "independent_visual_mesh_allowed": False,
            "categorical_semantics_used": False,
            "source_pixels_win_where_observed": True,
            "runtime_consumption_required": True,
            "triangle_array_order_is_presentation_authority": False,
            "pairwise_visibility_used": False,
            "uncalibrated_neural_confidence_used": False,
            "global_visibility_requires_complete_stack": True,
        },
    )
    return replace(value, graph_hash=compiled_graphics_hash_v2(value))
