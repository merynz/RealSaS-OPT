from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


Json = dict[str, Any]


class TESSAActionKindV1(IntEnum):
    """Structured autoregressive mesh grammar.

    The grammar is clean-room inspired by adjacent-face mesh generation, but it
    keeps RealSaS component/chart boundaries explicit and never grants model
    output canonical authority.
    """

    BOS = 0
    COMPONENT_BEGIN = 1
    CHART_BEGIN = 2
    FACE_SEED = 3
    FACE_ADJACENT = 4
    CHART_END = 5
    COMPONENT_END = 6
    EOS = 7


@dataclass(frozen=True)
class TESSAScalingPolicyV1:
    """Resource/sequence contract for product-scale TESSA generation."""

    max_faces: int = 32768
    max_vertices: int = 32768
    max_components: int = 128
    max_charts_per_component: int = 512
    local_attention_window: int = 2048
    query_chunk_size: int = 256
    surface_latent_count: int = 384
    coordinate_bins: int = 1024

    def validate(self) -> None:
        if self.max_faces < 6952:
            raise ValueError("TESSA_SCALE_POLICY_MUST_ADMIT_KNIGHT_6952_FACES")
        if self.max_vertices < 3665:
            raise ValueError("TESSA_SCALE_POLICY_MUST_ADMIT_KNIGHT_3665_VERTICES")
        if self.local_attention_window < 128:
            raise ValueError("TESSA_LOCAL_ATTENTION_WINDOW_TOO_SMALL")
        if self.query_chunk_size < 1 or self.query_chunk_size > self.local_attention_window:
            raise ValueError("TESSA_QUERY_CHUNK_INVALID")
        if self.surface_latent_count < 64:
            raise ValueError("TESSA_SURFACE_LATENT_COUNT_TOO_SMALL")
        if self.coordinate_bins < 128:
            raise ValueError("TESSA_COORDINATE_BINS_TOO_SMALL")

    def admits(self, *, vertex_count: int, face_count: int) -> bool:
        self.validate()
        return vertex_count <= self.max_vertices and face_count <= self.max_faces


@dataclass(frozen=True)
class TESSAVertexProposalV1:
    """Learned vertex proposal before Compiler support qualification.

    `primary_surface_id` is a learned/pointer anchor only.  It is not a
    SurfaceSupportBinding and cannot be promoted without deterministic Compiler
    support resolution and source-fidelity qualification.
    """

    proposal_vertex_id: str
    P: tuple[float, float, float]
    primary_surface_id: str
    component_id: str
    confidence: float = 1.0
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class TESSAMeshProposalV1:
    """TESSA learned output. Authority class is PROPOSAL only."""

    vertices: tuple[TESSAVertexProposalV1, ...]
    faces: tuple[tuple[str, str, str], ...]
    source_geometry_lineage_hash: str
    model_provenance: str
    topology_sequence_hash: str
    metadata: Json = field(default_factory=dict)
    schema_version: str = "RealSaS.TESSAMeshProposal.v1"

    def validate(self, policy: TESSAScalingPolicyV1 | None = None) -> None:
        policy = policy or TESSAScalingPolicyV1()
        policy.validate()
        if not self.source_geometry_lineage_hash:
            raise ValueError("TESSA_SOURCE_GEOMETRY_LINEAGE_MISSING")
        ids = {v.proposal_vertex_id for v in self.vertices}
        if len(ids) != len(self.vertices):
            raise ValueError("TESSA_PROPOSAL_VERTEX_ID_DUPLICATE")
        if not policy.admits(vertex_count=len(self.vertices), face_count=len(self.faces)):
            raise ValueError("TESSA_PROPOSAL_EXCEEDS_SCALE_POLICY")
        for face in self.faces:
            if len(face) != 3 or len(set(face)) != 3 or any(v not in ids for v in face):
                raise ValueError("TESSA_PROPOSAL_FACE_INVALID")
        if any(not v.primary_surface_id for v in self.vertices):
            raise ValueError("TESSA_PRIMARY_SURFACE_ANCHOR_MISSING")


@dataclass(frozen=True)
class TESSAMechanicalRewardPolicyV1:
    """Training-time consequence weights; never Compiler qualification policy."""

    surface: float = 1.0
    boundary: float = 0.5
    component: float = 1.0
    flip: float = 2.0
    stretch: float = 1.0
    area: float = 1.0
    condition: float = 1.0
    jacobian: float = 2.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0.0:
                raise ValueError(f"TESSA_REWARD_WEIGHT_NEGATIVE:{name}")
