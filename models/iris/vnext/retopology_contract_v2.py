from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

SCHEMA_VERSION = "RealSaS.TRetopologyProposal.v2"


@dataclass(frozen=True)
class FSurfaceVertexBindingV2:
    f_face_id: int
    barycentric: tuple[float, float, float]
    residual_xyz: tuple[float, float, float]
    residual_limit: float
    confidence: float
    provenance: str

    def validate(self) -> None:
        if self.f_face_id < 0:
            raise ValueError("T_RETOPOLOGY_F_FACE_ID_INVALID")
        if len(self.barycentric) != 3 or not all(isfinite(v) for v in self.barycentric):
            raise ValueError("T_RETOPOLOGY_BARYCENTRIC_INVALID")
        if any(v < -1e-7 or v > 1.0 + 1e-7 for v in self.barycentric):
            raise ValueError("T_RETOPOLOGY_BARYCENTRIC_OUT_OF_RANGE")
        if abs(sum(self.barycentric) - 1.0) > 1e-6:
            raise ValueError("T_RETOPOLOGY_BARYCENTRIC_SUM_INVALID")
        if len(self.residual_xyz) != 3 or not all(isfinite(v) for v in self.residual_xyz):
            raise ValueError("T_RETOPOLOGY_RESIDUAL_INVALID")
        if not isfinite(self.residual_limit) or self.residual_limit < 0.0:
            raise ValueError("T_RETOPOLOGY_RESIDUAL_LIMIT_INVALID")
        norm = sqrt(sum(float(v) * float(v) for v in self.residual_xyz))
        if norm > self.residual_limit + 1e-9:
            raise ValueError("T_RETOPOLOGY_RESIDUAL_EXCEEDS_LIMIT")
        if not isfinite(self.confidence) or self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("T_RETOPOLOGY_CONFIDENCE_INVALID")
        if not self.provenance:
            raise ValueError("T_RETOPOLOGY_BINDING_PROVENANCE_MISSING")


@dataclass(frozen=True)
class TRetopologyProposalV2:
    source_f_surface_sha256: str
    vertex_bindings: tuple[FSurfaceVertexBindingV2, ...]
    faces: tuple[tuple[int, int, int], ...]
    decoder_family: str
    proposal_provenance: str
    free_xyz_authority: bool = False
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("T_RETOPOLOGY_SCHEMA_INVALID")
        if len(self.source_f_surface_sha256) != 64:
            raise ValueError("T_RETOPOLOGY_F_SURFACE_SHA_INVALID")
        if self.free_xyz_authority:
            raise ValueError("T_RETOPOLOGY_FREE_XYZ_AUTHORITY_FORBIDDEN")
        if not self.vertex_bindings:
            raise ValueError("T_RETOPOLOGY_VERTEX_BINDINGS_EMPTY")
        for binding in self.vertex_bindings:
            binding.validate()
        if not self.faces:
            raise ValueError("T_RETOPOLOGY_FACES_EMPTY")
        vertex_count = len(self.vertex_bindings)
        for face in self.faces:
            if len(face) != 3:
                raise ValueError("T_RETOPOLOGY_FACE_ARITY_INVALID")
            if len(set(face)) != 3:
                raise ValueError("T_RETOPOLOGY_FACE_DEGENERATE_INDEX")
            if min(face) < 0 or max(face) >= vertex_count:
                raise ValueError("T_RETOPOLOGY_FACE_INDEX_OOB")
        if not self.decoder_family:
            raise ValueError("T_RETOPOLOGY_DECODER_FAMILY_MISSING")
        if not self.proposal_provenance:
            raise ValueError("T_RETOPOLOGY_PROVENANCE_MISSING")


@dataclass(frozen=True)
class TCompilerPromotionInputV2:
    proposal: TRetopologyProposalV2
    f_surface_vertex_count: int
    f_surface_face_count: int
    mechanical_probe_set_sha256: str

    def validate(self) -> None:
        self.proposal.validate()
        if self.f_surface_vertex_count <= 0 or self.f_surface_face_count <= 0:
            raise ValueError("T_RETOPOLOGY_F_SURFACE_SIZE_INVALID")
        if len(self.mechanical_probe_set_sha256) != 64:
            raise ValueError("T_RETOPOLOGY_MECHANICAL_PROBE_SHA_INVALID")
        for binding in self.proposal.vertex_bindings:
            if binding.f_face_id >= self.f_surface_face_count:
                raise ValueError("T_RETOPOLOGY_BINDING_FACE_OOB")


def validate_retopology_batch_v2(values: Sequence[TCompilerPromotionInputV2]) -> None:
    if not values:
        raise ValueError("T_RETOPOLOGY_BATCH_EMPTY")
    f_hashes = {value.proposal.source_f_surface_sha256 for value in values}
    probe_hashes = {value.mechanical_probe_set_sha256 for value in values}
    if len(f_hashes) != 1:
        raise ValueError("T_RETOPOLOGY_BATCH_F_SURFACE_DRIFT")
    if len(probe_hashes) != 1:
        raise ValueError("T_RETOPOLOGY_BATCH_PROBE_SET_DRIFT")
    for value in values:
        value.validate()


__all__ = [
    "FSurfaceVertexBindingV2",
    "SCHEMA_VERSION",
    "TCompilerPromotionInputV2",
    "TRetopologyProposalV2",
    "validate_retopology_batch_v2",
]
