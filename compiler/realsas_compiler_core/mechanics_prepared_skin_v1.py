from __future__ import annotations

"""Bounded, prepared sparse LBS research kernel for interactive Mechanics.

Prepare once per immutable CanonicalMesh/Rig/Skin triple. At each frame, only
pose matrices and translations change. This kernel is independent of graphics
and does not learn or synthesize skin, controls or rig topology.

Research opt-in, not a shipping 60 FPS certification. The original V1 LBS
remains the regression oracle and default runtime.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .mechanics_control_solver_v1 import DeformedMeshIRV1, MechanicsPoseIRV1
from .types import QualificationError


@dataclass(frozen=True)
class PreparedMechanicsSkinDeformerV1:
    canonical_mesh_vertex_ids: tuple[str, ...]
    canonical_joint_ids: tuple[str, ...]
    influence_vertex_indices: np.ndarray
    influence_joint_indices: np.ndarray
    influence_weights: np.ndarray
    rest_offset_vectors: np.ndarray
    rest_vertex_positions: np.ndarray
    vertex_count: int
    influence_count: int
    mesh_lineage_hash: str
    rig_lineage_hash: str
    control_program_hash: str
    deform_structure_hash: str
    skin_lineage_hash: str
    metadata: dict = field(default_factory=dict)


def _readonly(values: np.ndarray) -> np.ndarray:
    # bytes-backed NumPy views cannot be made writable again by a consumer.
    # This protects the once-prepared carrier against accidental control-frame
    # mutation (ordinary setflags(write=False) can be reversed on owners).
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def prepare_mechanics_skin_deformer_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    *,
    maximum_vertices: int = 250000,
    maximum_joints: int = 512,
    maximum_influences: int = 2000000,
) -> PreparedMechanicsSkinDeformerV1:
    if (mesh.mesh_lineage_hash != rig.mesh_lineage_hash
            or mesh.mesh_lineage_hash != skin.mesh_lineage_hash
            or rig.deform_structure_hash != skin.deform_structure_hash):
        raise QualificationError("MECHANICS_PREPARED_LBS_LINEAGE_DRIFT")
    if min(maximum_vertices,maximum_joints,maximum_influences)<1:
        raise QualificationError("MECHANICS_PREPARED_LBS_RESOURCE_POLICY_INVALID")
    vertex_ids=tuple(str(v.canonical_mesh_vertex_id) for v in mesh.vertices)
    joint_ids=tuple(str(j.canonical_joint_id) for j in rig.joints)
    n,j=len(vertex_ids),len(joint_ids)
    if (not n or not j or n>maximum_vertices or j>maximum_joints
            or len(set(vertex_ids))!=n or len(set(joint_ids))!=j):
        raise QualificationError("MECHANICS_PREPARED_LBS_MESH_OR_JOINT_DOMAIN_INVALID")
    try:
        vertices=np.asarray([v.P for v in mesh.vertices],dtype=np.float64)
        joints=np.asarray([row.P for row in rig.joints],dtype=np.float64)
    except (ValueError,TypeError) as exc:
        raise QualificationError("MECHANICS_PREPARED_LBS_REST_POSITION_INVALID") from exc
    if (vertices.shape!=(n,3) or joints.shape!=(j,3)
            or not np.isfinite(vertices).all() or not np.isfinite(joints).all()):
        raise QualificationError("MECHANICS_PREPARED_LBS_REST_POSITION_INVALID")
    by_vertex={row.canonical_mesh_vertex_id:row for row in skin.rows}
    if len(by_vertex)!=n or len(skin.rows)!=n or set(by_vertex)!=set(vertex_ids):
        raise QualificationError("MECHANICS_PREPARED_LBS_SKIN_COVERAGE_DRIFT")
    joint_index={name:i for i,name in enumerate(joint_ids)}
    vi:list[int]=[]
    ji:list[int]=[]
    weights:list[float]=[]
    for row_id,vid in enumerate(vertex_ids):
        row=by_vertex[vid]
        if not row.influences:
            raise QualificationError("MECHANICS_PREPARED_LBS_EMPTY_SKIN_ROW")
        seen:set[str]=set()
        total=0.0
        for joint_id,raw_w in row.influences:
            if joint_id not in joint_index or joint_id in seen:
                raise QualificationError("MECHANICS_PREPARED_LBS_SKIN_JOINT_INVALID")
            seen.add(joint_id)
            try:
                w=float(raw_w)
            except (ValueError,TypeError) as exc:
                raise QualificationError("MECHANICS_PREPARED_LBS_SKIN_WEIGHT_INVALID") from exc
            if not math.isfinite(w) or w<0.0:
                raise QualificationError("MECHANICS_PREPARED_LBS_SKIN_WEIGHT_INVALID")
            vi.append(row_id)
            ji.append(joint_index[joint_id])
            weights.append(w)
            total+=w
            if len(vi)>maximum_influences:
                raise QualificationError("MECHANICS_PREPARED_LBS_INFLUENCE_RESOURCE_BOUND")
        if abs(total-1.)>1e-6:
            raise QualificationError("MECHANICS_PREPARED_LBS_SKIN_SIMPLEX_INVALID")
    vis=np.asarray(vi,dtype=np.int32)
    jis=np.asarray(ji,dtype=np.int32)
    ws=np.asarray(weights,dtype=np.float64)
    offsets=vertices[vis]-joints[jis]
    return PreparedMechanicsSkinDeformerV1(
        canonical_mesh_vertex_ids=vertex_ids,
        canonical_joint_ids=joint_ids,
        influence_vertex_indices=_readonly(vis),
        influence_joint_indices=_readonly(jis),
        influence_weights=_readonly(ws),
        rest_offset_vectors=_readonly(offsets),
        rest_vertex_positions=_readonly(vertices),
        vertex_count=n,
        influence_count=len(weights),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        deform_structure_hash=rig.deform_structure_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        metadata={
            "authority":"MECHANICS_PREPARED_SKIN_RESEARCH_ONLY",
            "skin_and_rest_read_only":True,
            "control_or_skin_synthesis_performed":False,
            "graphics_authority_claimed":False,
            "training_authorized":False,
        },
    )


def deform_prepared_mechanics_skin_v1(
    prepared: PreparedMechanicsSkinDeformerV1,
    pose: MechanicsPoseIRV1,
    *,
    expected_mesh_lineage_hash: str | None = None,
    expected_skin_lineage_hash: str | None = None,
    expected_deform_structure_hash: str | None = None,
) -> DeformedMeshIRV1:
    if (pose.rig_lineage_hash!=prepared.rig_lineage_hash
            or pose.control_program_hash!=prepared.control_program_hash
            or (expected_mesh_lineage_hash is not None
                and expected_mesh_lineage_hash!=prepared.mesh_lineage_hash)
            or (expected_skin_lineage_hash is not None
                and expected_skin_lineage_hash!=prepared.skin_lineage_hash)
            or (expected_deform_structure_hash is not None
                and expected_deform_structure_hash!=prepared.deform_structure_hash)):
        raise QualificationError("MECHANICS_PREPARED_LBS_POSE_OR_BINDING_DRIFT")
    by_joint={row.canonical_joint_id:row for row in pose.joints}
    if (len(by_joint)!=len(prepared.canonical_joint_ids)
            or len(pose.joints)!=len(by_joint)
            or set(by_joint)!=set(prepared.canonical_joint_ids)):
        raise QualificationError("MECHANICS_PREPARED_LBS_POSE_JOINT_COVERAGE_DRIFT")
    try:
        positions=np.asarray([by_joint[jid].solved_P for jid in prepared.canonical_joint_ids],
                             dtype=np.float64)
        rotations=np.asarray([by_joint[jid].rotation_basis for jid in prepared.canonical_joint_ids],
                             dtype=np.float64)
    except (ValueError,TypeError) as exc:
        raise QualificationError("MECHANICS_PREPARED_LBS_POSE_NUMERIC_INVALID") from exc
    if (positions.shape!=(len(prepared.canonical_joint_ids),3)
            or rotations.shape!=(len(prepared.canonical_joint_ids),3,3)
            or not np.isfinite(positions).all() or not np.isfinite(rotations).all()):
        raise QualificationError("MECHANICS_PREPARED_LBS_POSE_NUMERIC_INVALID")
    ii=prepared.influence_joint_indices
    vv=prepared.influence_vertex_indices
    rotated=np.einsum("mij,mj->mi",rotations[ii],prepared.rest_offset_vectors)
    contribution=positions[ii]+rotated
    if not np.isfinite(contribution).all():
        raise QualificationError("MECHANICS_PREPARED_LBS_CONTRIBUTION_NONFINITE")
    weighted=prepared.influence_weights[:,None]*contribution
    out=np.stack([
        np.bincount(vv,weights=weighted[:,k],minlength=prepared.vertex_count)
        for k in range(3)
    ],axis=1)
    if not np.isfinite(out).all():
        raise QualificationError("MECHANICS_PREPARED_LBS_RESULT_NONFINITE")
    return DeformedMeshIRV1(
        canonical_mesh_vertex_ids=prepared.canonical_mesh_vertex_ids,
        positions=tuple(tuple(map(float,xyz)) for xyz in out),
        mesh_lineage_hash=prepared.mesh_lineage_hash,
        skin_lineage_hash=prepared.skin_lineage_hash,
        rig_lineage_hash=prepared.rig_lineage_hash,
        metadata={
            "deformation_method":"LINEAR_BLEND_SKINNING",
            "mechanical_mesh_authority_preserved":True,
            "visual_rebind_performed":False,
            "execution":"PREPARED_SPARSE_NUMPY_RESEARCH_ONLY",
            "control_or_skin_synthesis_performed":False,
        },
    )


__all__=[
    "PreparedMechanicsSkinDeformerV1",
    "prepare_mechanics_skin_deformer_v1",
    "deform_prepared_mechanics_skin_v1",
]
