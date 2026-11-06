"""Executable CPU reference puppet for Operational Representation Synthesis v0.

Reuses RealSaS's actual canonical visibility/CAA compositing pipeline. This
does not qualify source-original Cycles parity or promise a shipping renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .appearance_render_v2 import render_caa_reference
from .operational_representation_synthesis_v0 import (
    RepresentationCandidateV0, SourceGraphicsEvidenceV0,
    validate_candidate_v0, _require,
)


@dataclass(frozen=True)
class PuppetReferenceFrameV0:
    premultiplied_rgba: np.ndarray
    straight_rgba_u8: np.ndarray
    owner_face_index: np.ndarray
    packet_owner_index: np.ndarray
    layer_overflow: np.ndarray
    exact_depth_ambiguity: np.ndarray
    edit_state: tuple[int, ...]
    candidate_id: str
    qualification: str = "REFERENCE_EXECUTED__SOURCE_PARITY_UNKNOWN"
    schema_version: str = "RealSaS.PuppetReferenceFrame.v0"


class OperationalPuppetRuntimeV0:
    """An editable source-addressed mesh puppet with pose-aware CAA projection.

    The canonical mesh remains the only geometry authority. "Packets" group
    faces for possible export/draw scheduling; this reference executor
    preserves one global per-fragment geometry-depth composition instead of
    incorrectly relying on a fixed packet Z-order.
    """

    def __init__(self, *, evidence: SourceGraphicsEvidenceV0,
                 candidate: RepresentationCandidateV0, mesh, face_uv,
                 texture_rgba_u8, edit_rectangles):
        evidence.validate()
        _require((candidate.mesh_binding_hash == evidence.mesh_binding_hash
                  and candidate.skin_binding_hash == evidence.skin_binding_hash
                  and candidate.source_graphics_binding_hash
                  == evidence.source_graphics_binding_hash),
                 "ORS_RUNTIME_BINDING_DRIFT")
        validate_candidate_v0(
            candidate, face_count=evidence.face_count,
            address_count=len(candidate.edit_control_address_indices),
        )
        _require(len(mesh.faces) == evidence.face_count,
                 "ORS_RUNTIME_MESH_FACE_COUNT_DRIFT")
        self.mesh = mesh
        self.evidence = evidence
        self.candidate = candidate
        self.face_uv = np.asarray(face_uv, dtype=np.float64).copy()
        self.base_texture = np.asarray(texture_rgba_u8, dtype=np.uint8).copy()
        _require(self.face_uv.shape == (evidence.face_count, 3, 2)
                 and np.isfinite(self.face_uv).all()
                 and np.all((self.face_uv >= 0) & (self.face_uv <= 1)),
                 "ORS_RUNTIME_FACE_UV_INVALID")
        texture = self.base_texture
        _require(texture.ndim == 3 and texture.shape[2] == 4
                 and texture.shape[0] >= 2 and texture.shape[1] >= 2,
                 "ORS_RUNTIME_TEXTURE_INVALID")
        rects = tuple(tuple(r) for r in edit_rectangles)
        _require(len(rects) == len(candidate.edit_control_address_indices),
                 "ORS_RUNTIME_EDIT_ADDRESS_COUNT_DRIFT")
        occupied = np.zeros(texture.shape[:2], dtype=np.uint8)
        for x0, y0, x1, y1 in rects:
            _require(all(type(v) is int for v in (x0, y0, x1, y1))
                     and 0 <= x0 < x1 <= texture.shape[1]
                     and 0 <= y0 < y1 <= texture.shape[0],
                     "ORS_RUNTIME_EDIT_RECT_INVALID")
            _require(not np.any(occupied[y0:y1, x0:x1]),
                     "ORS_RUNTIME_EDIT_ADDRESS_OVERLAP_UNQUALIFIED")
            occupied[y0:y1, x0:x1] = 1
        self.edit_rectangles = rects
        packet_owner = np.full(evidence.face_count, -1, dtype=np.int32)
        for i, packet in enumerate(candidate.packets):
            packet_owner[list(packet.face_indices)] = i
        _require(np.all(packet_owner >= 0), "ORS_RUNTIME_PACKET_OWNER_MISSING")
        self.packet_owner_by_face = packet_owner
        # No source-original source pixels or artist ownership are ingested.

    def edited_texture(self, control_bits):
        bits = tuple(control_bits)
        _require(len(bits) == len(self.edit_rectangles)
                 and all(type(b) is int and b in (0, 1) for b in bits),
                 "ORS_RUNTIME_CONTROL_BITS_INVALID")
        texture = self.base_texture.copy()
        for bit, (x0, y0, x1, y1) in zip(bits, self.edit_rectangles):
            if bit:
                # Original Cat V2 edit rule; exactly preserves straight alpha.
                texture[y0:y1, x0:x1, :3] = (
                    255 - self.base_texture[y0:y1, x0:x1, :3])
        _require(np.array_equal(texture[..., 3], self.base_texture[..., 3]),
                 "ORS_RUNTIME_ALPHA_DRIFT")
        return texture

    def render(self, *, camera, posed_positions=None, control_bits=None):
        if control_bits is None:
            control_bits = (0,) * len(self.edit_rectangles)
        bits = tuple(control_bits)
        texture = self.edited_texture(bits)
        positions = None
        if posed_positions is not None:
            positions = np.asarray(posed_positions, dtype=np.float64)
            _require(positions.shape == (len(self.mesh.vertices), 3)
                     and np.isfinite(positions).all(),
                     "ORS_RUNTIME_POSE_POSITIONS_INVALID")
        # Provenance 0 is source-direct for the reference CAA provenance
        # vocabulary. This is NOT an independent source authority admission.
        provenance = np.zeros(texture.shape[:2], dtype=np.uint8)
        rendered = render_caa_reference(
            mesh=self.mesh, camera=camera, face_uv=self.face_uv,
            texture_rgba_u8=texture, provenance_atlas=provenance,
            positions=positions,
        )
        owner = rendered.owner_face_index
        packet_owner = np.full(owner.shape, -1, dtype=np.int32)
        visible = owner >= 0
        packet_owner[visible] = self.packet_owner_by_face[owner[visible]]
        reason = ("REFERENCE_UNKNOWN_LAYER_OVERFLOW" if np.any(rendered.layer_overflow)
                  else "REFERENCE_UNKNOWN_EXACT_DEPTH_TIE"
                  if np.any(rendered.exact_depth_ambiguity)
                  else "REFERENCE_EXECUTED__SOURCE_PARITY_UNKNOWN")
        return PuppetReferenceFrameV0(
            premultiplied_rgba=rendered.premultiplied_rgba,
            straight_rgba_u8=rendered.straight_rgba_u8,
            owner_face_index=owner,
            packet_owner_index=packet_owner,
            layer_overflow=rendered.layer_overflow,
            exact_depth_ambiguity=rendered.exact_depth_ambiguity,
            edit_state=bits, candidate_id=self.candidate.candidate_id,
            qualification=reason,
        )
