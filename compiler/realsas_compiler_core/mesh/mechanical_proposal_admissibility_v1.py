from __future__ import annotations

"""Precomputed local mechanical admissibility guard for static mesh proposals.

This is a fast proposal-level companion to the global G3B court.  It freezes the
candidate's qualified skin transfer and G3B probe poses, then evaluates only the
old/new local triangle patch supplied by a static topology operator.

The guard is deliberately lexicographic/hard:
- unsafe local face count may not increase;
- worst normalized mechanical severity may not increase.

It does not mint product authority.  Full G3B/G3/actual-motion courts remain
mandatory after batches and at final seal.
"""

from dataclasses import dataclass
import math
import numpy as np

from .deformation_stress_v1 import _candidate_skin_matrix
from .deformation_stress_v2 import _pose_skin_matrices, _triangle_metrics_batch_exact
from .skin_topology_compatibility_v1 import _stress_angle, DEFAULT_MAX_EDGE_RATIO
from ..joint_frames_v1 import derive_joint_frames_from_skeleton
from ..motion_3d_v1 import apply_lbs_matrix_v1
from ..product_authority_v1 import (
    validate_deformation_capability_envelope,
    validate_mesh_qualification_policy,
)
from ..types import QualificationError


@dataclass(frozen=True)
class LocalMechanicalSignatureV1:
    face_count: int
    unsafe_face_count: int
    maximum_severity: float
    minimum_area_ratio: float
    maximum_area_ratio: float
    maximum_condition_number: float
    maximum_edge_ratio: float


class MechanicalProposalAdmissibilityGuardV1:
    """Callable proposal predicate for flip/collapse/cavity operators."""

    def __init__(
        self,
        candidate,
        *,
        surface,
        skeleton,
        skin,
        envelope,
        cameras,
        policy,
        max_edge_ratio: float = DEFAULT_MAX_EDGE_RATIO,
        tolerance: float = 1e-9,
    ):
        validate_mesh_qualification_policy(policy)
        validate_deformation_capability_envelope(
            envelope, known_joint_ids={j.canonical_joint_id for j in skeleton.joints}
        )
        if envelope.skeleton_lineage_hash != skeleton.skeleton_lineage_hash:
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_ENVELOPE_SKELETON_DRIFT")
        if not math.isfinite(float(max_edge_ratio)) or float(max_edge_ratio) <= 1.0:
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_EDGE_LIMIT_INVALID")
        if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_TOLERANCE_INVALID")

        rest, weights, _ = _candidate_skin_matrix(
            candidate, surface=surface, skeleton=skeleton, skin=skin
        )
        self.rest = np.asarray(rest, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.vertex_index = {
            str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)
        }
        if len(self.vertex_index) != len(candidate.vertices):
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_VERTEX_ID_DUPLICATE")
        self.policy = policy
        self.max_edge_ratio = float(max_edge_ratio)
        self.tolerance = float(tolerance)

        frames = derive_joint_frames_from_skeleton(skeleton, cameras=cameras)
        joint_ids = tuple(j.canonical_joint_id for j in skeleton.joints)
        stress_angle = float(_stress_angle(envelope))
        probes = [("REST", None, None, 0.0)]
        for jid in sorted(joint_ids):
            for axis_index, axis_name in enumerate(("X", "Y", "Z")):
                for sign in (-1.0, 1.0):
                    deg = sign * stress_angle
                    probes.append((f"{jid}:LOCAL_{axis_name}:{deg:+g}", jid, axis_index, deg))

        posed_rows = []
        probe_ids = []
        for probe_id, jid, axis_index, degrees in probes:
            skin_by_id = _pose_skin_matrices(
                skeleton,
                frames,
                joint_id=jid,
                local_axis_index=axis_index,
                degrees=degrees,
            )
            matrices = np.stack([skin_by_id[x] for x in joint_ids], axis=0)
            posed_rows.append(
                np.asarray(
                    apply_lbs_matrix_v1(self.rest, self.weights, matrices),
                    dtype=np.float64,
                )
            )
            probe_ids.append(probe_id)
        self.posed = tuple(posed_rows)
        self.probe_ids = tuple(probe_ids)

    def _face_indices(self, faces):
        out = []
        for face in tuple(faces):
            row = tuple(map(str, face))
            if len(row) != 3 or any(v not in self.vertex_index for v in row):
                raise QualificationError("MECHANICAL_PROPOSAL_GUARD_FACE_UNKNOWN_VERTEX")
            out.append(tuple(self.vertex_index[v] for v in row))
        if not out:
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_EMPTY_PATCH")
        return np.asarray(out, dtype=np.int64)

    def signature(self, faces) -> LocalMechanicalSignatureV1:
        fi = self._face_indices(faces)
        min_area = np.ones(len(fi), dtype=np.float64)
        max_area = np.ones(len(fi), dtype=np.float64)
        max_cond = np.ones(len(fi), dtype=np.float64)
        max_edge = np.ones(len(fi), dtype=np.float64)

        for posed in self.posed:
            area, cond, _emin, emax, _smin = _triangle_metrics_batch_exact(
                self.rest, posed, fi
            )
            if not (
                np.isfinite(area).all()
                and np.isfinite(cond).all()
                and np.isfinite(emax).all()
            ):
                return LocalMechanicalSignatureV1(
                    len(fi), len(fi), float("inf"), 0.0, float("inf"),
                    float("inf"), float("inf")
                )
            min_area = np.minimum(min_area, area)
            max_area = np.maximum(max_area, area)
            max_cond = np.maximum(max_cond, cond)
            max_edge = np.maximum(max_edge, emax)

        severity = np.maximum.reduce(
            (
                max_cond / max(float(self.policy.g3_max_dynamic_condition_number), 1e-12),
                max_edge / max(float(self.max_edge_ratio), 1e-12),
                max_area / max(float(self.policy.g3_max_dynamic_area_ratio), 1e-12),
                float(self.policy.g3_min_dynamic_area_ratio) / np.maximum(min_area, 1e-12),
            )
        )
        unsafe = severity > (1.0 + self.tolerance)
        return LocalMechanicalSignatureV1(
            face_count=len(fi),
            unsafe_face_count=int(np.count_nonzero(unsafe)),
            maximum_severity=float(np.max(severity, initial=0.0)),
            minimum_area_ratio=float(np.min(min_area, initial=1.0)),
            maximum_area_ratio=float(np.max(max_area, initial=1.0)),
            maximum_condition_number=float(np.max(max_cond, initial=1.0)),
            maximum_edge_ratio=float(np.max(max_edge, initial=1.0)),
        )

    def __call__(self, proposal) -> bool:
        old_faces = tuple(proposal.get("old_faces") or ())
        new_faces = tuple(proposal.get("new_faces") or ())
        if not old_faces or not new_faces:
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_PATCH_MISSING")
        old = self.signature(old_faces)
        new = self.signature(new_faces)
        return bool(
            new.unsafe_face_count <= old.unsafe_face_count
            and new.maximum_severity <= old.maximum_severity * (1.0 + self.tolerance)
        )


__all__ = [
    "LocalMechanicalSignatureV1",
    "MechanicalProposalAdmissibilityGuardV1",
]
