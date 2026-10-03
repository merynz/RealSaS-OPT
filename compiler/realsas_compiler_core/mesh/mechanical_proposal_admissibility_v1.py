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
        matrix_rows = []
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
            matrix_rows.append(np.asarray(matrices,dtype=np.float64))
            probe_ids.append(probe_id)
        self.posed = np.stack(posed_rows, axis=0)
        self.probe_matrices = np.stack(matrix_rows, axis=0)
        self.probe_ids = tuple(probe_ids)
        self._signature_cache = {}

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

    def _signature_from_face_state(self, r, p) -> LocalMechanicalSignatureV1:
        r=np.asarray(r,dtype=np.float64)
        p=np.asarray(p,dtype=np.float64)
        if r.ndim!=3 or r.shape[1:]!=(3,3):
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_REST_PATCH_SHAPE_INVALID")
        if p.ndim!=4 or p.shape[1:]!=(len(r),3,3):
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_POSED_PATCH_SHAPE_INVALID")

        r1=r[:,1]-r[:,0]; r2=r[:,2]-r[:,0]
        l1=np.linalg.norm(r1,axis=1)
        if np.any(l1<=1e-12):
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_REST_EDGE_DEGENERATE")
        u=r1/l1[:,None]
        x2=np.sum(r2*u,axis=1)
        perp=r2-x2[:,None]*u
        y2=np.linalg.norm(perp,axis=1)
        if np.any(y2<=1e-12):
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_REST_TRIANGLE_DEGENERATE")

        inv=np.zeros((len(r),2,2),dtype=np.float64)
        inv[:,0,0]=1.0/l1
        inv[:,0,1]=-x2/(l1*y2)
        inv[:,1,1]=1.0/y2

        pedges=np.stack((p[:,:,1]-p[:,:,0],p[:,:,2]-p[:,:,0]),axis=3)
        F=np.einsum("pfci,fij->pfcj",pedges,inv,optimize=True)
        singular=np.linalg.svd(F,compute_uv=False)
        smax=singular[...,0]
        smin=singular[...,1]
        area=smax*smin
        cond=smax/np.maximum(smin,1e-15)

        rest_edges=np.stack((
            np.linalg.norm(r[:,1]-r[:,0],axis=1),
            np.linalg.norm(r[:,2]-r[:,1],axis=1),
            np.linalg.norm(r[:,0]-r[:,2],axis=1),
        ),axis=1)
        posed_edges=np.stack((
            np.linalg.norm(p[:,:,1]-p[:,:,0],axis=2),
            np.linalg.norm(p[:,:,2]-p[:,:,1],axis=2),
            np.linalg.norm(p[:,:,0]-p[:,:,2],axis=2),
        ),axis=2)
        edge_max=(posed_edges/rest_edges[None]).max(axis=2)

        finite=(
            np.isfinite(area).all()
            and np.isfinite(cond).all()
            and np.isfinite(edge_max).all()
        )
        if not finite:
            return LocalMechanicalSignatureV1(
                len(r),len(r),float("inf"),0.0,float("inf"),float("inf"),float("inf")
            )

        min_area=area.min(axis=0)
        max_area=area.max(axis=0)
        max_cond=cond.max(axis=0)
        max_edge=edge_max.max(axis=0)
        severity=np.maximum.reduce((
            max_cond/max(float(self.policy.g3_max_dynamic_condition_number),1e-12),
            max_edge/max(float(self.max_edge_ratio),1e-12),
            max_area/max(float(self.policy.g3_max_dynamic_area_ratio),1e-12),
            float(self.policy.g3_min_dynamic_area_ratio)/np.maximum(min_area,1e-12),
        ))
        unsafe=severity>(1.0+self.tolerance)
        return LocalMechanicalSignatureV1(
            face_count=len(r),
            unsafe_face_count=int(np.count_nonzero(unsafe)),
            maximum_severity=float(np.max(severity,initial=0.0)),
            minimum_area_ratio=float(np.min(min_area,initial=1.0)),
            maximum_area_ratio=float(np.max(max_area,initial=1.0)),
            maximum_condition_number=float(np.max(max_cond,initial=1.0)),
            maximum_edge_ratio=float(np.max(max_edge,initial=1.0)),
        )

    def signature(self, faces) -> LocalMechanicalSignatureV1:
        face_rows=tuple(tuple(map(str,face)) for face in tuple(faces))
        key=tuple(sorted(face_rows))
        cached=self._signature_cache.get(key)
        if cached is not None:
            return cached
        fi=self._face_indices(face_rows)
        out=self._signature_from_face_state(self.rest[fi],self.posed[:,fi])
        self._signature_cache[key]=out
        return out

    def _temporary_split_state(self, spec):
        nid=str(spec.get("id") or "")
        edge=tuple(map(str,spec.get("edge") or ()))
        t=float(spec.get("fraction",0.5))
        if not nid or len(edge)!=2 or edge[0] not in self.vertex_index or edge[1] not in self.vertex_index:
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_SPLIT_SPEC_INVALID")
        if not math.isfinite(t) or t<=0.0 or t>=1.0:
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_SPLIT_FRACTION_INVALID")
        ia,ib=(self.vertex_index[edge[0]],self.vertex_index[edge[1]])
        rest=(1.0-t)*self.rest[ia]+t*self.rest[ib]
        weights=(1.0-t)*self.weights[ia]+t*self.weights[ib]
        weights=weights/np.maximum(float(weights.sum()),1e-15)
        hom=np.concatenate([rest,np.ones((1,),dtype=np.float64)],axis=0)
        per=np.einsum("pjac,c->pja",self.probe_matrices,hom,optimize=True)[...,:3]
        posed=np.einsum("j,pja->pa",weights,per,optimize=True)
        return nid,rest,posed

    def signature_with_temporary_split(self, faces, spec) -> LocalMechanicalSignatureV1:
        nid,temp_rest,temp_posed=self._temporary_split_state(spec)
        face_rows=tuple(tuple(map(str,face)) for face in tuple(faces))
        if not face_rows:
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_EMPTY_PATCH")
        r=[]; p=[]
        for face in face_rows:
            if len(face)!=3:
                raise QualificationError("MECHANICAL_PROPOSAL_GUARD_FACE_INVALID")
            rr=[]; pp=[]
            for vid in face:
                if vid==nid:
                    rr.append(temp_rest); pp.append(temp_posed)
                elif vid in self.vertex_index:
                    i=self.vertex_index[vid]
                    rr.append(self.rest[i]); pp.append(self.posed[:,i])
                else:
                    raise QualificationError("MECHANICAL_PROPOSAL_GUARD_FACE_UNKNOWN_VERTEX")
            r.append(np.stack(rr,axis=0))
            p.append(np.stack(pp,axis=1))
        return self._signature_from_face_state(
            np.stack(r,axis=0),
            np.stack(p,axis=1),
        )

    def __call__(self, proposal) -> bool:
        old_faces = tuple(proposal.get("old_faces") or ())
        new_faces = tuple(proposal.get("new_faces") or ())
        if not old_faces or not new_faces:
            raise QualificationError("MECHANICAL_PROPOSAL_GUARD_PATCH_MISSING")
        old = self.signature(old_faces)
        split_spec=proposal.get("temporary_split_vertex")
        new = (
            self.signature_with_temporary_split(new_faces,split_spec)
            if split_spec is not None
            else self.signature(new_faces)
        )
        return bool(
            new.unsafe_face_count <= old.unsafe_face_count
            and new.maximum_severity <= old.maximum_severity * (1.0 + self.tolerance)
        )




__all__ = [
    "LocalMechanicalSignatureV1",
    "MechanicalProposalAdmissibilityGuardV1",
]
