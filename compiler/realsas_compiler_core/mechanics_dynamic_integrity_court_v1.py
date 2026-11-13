from __future__ import annotations

"""Measured non-rest LBS and collision regression over actual canonical faces.

A rig can pass a rest-pose solver check while its moving arm intersects the
torso, its skin mapping loses vertices, or a socket detaches. This court
executes multiple bounded interventions and replays the canonical solver,
LBS, attachment transport and triangle intersection census.

This proves ONLY the tested geometric behavior. In particular, neither
source-observed contact nor learned secondary motion can be inferred from
correct rig mathematics. Incomplete coverage must never mean zero collision.
"""

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .canonical_mesh_ir_v1 import (
    CanonicalMeshIR, validate_canonical_mesh_domain_v1,
)
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .dynamic_geometry_integrity_v2 import unexpected_intersection_pairs
from .hashing import content_sha256
from .mechanics_attachment_socket_runtime_v1 import solve_attachment_socket_world_v1
from .mechanics_control_solver_v1 import (
    ControlTransformIRV1, deform_mesh_lbs_v1, solve_control_pose_v1,
)
from .types import QualificationError


SCHEMA = "RealSaS.MechanicsDynamicIntegrityCourt.v1"


@dataclass(frozen=True)
class DynamicIntegrityProbeV1:
    control_id: str
    intervention_scale: float
    posed_mesh_sha256: str
    pose_hash: str
    socket_hash: str
    new_intersection_count: int
    missing_rest_intersection_count: int
    total_posed_intersection_count: int
    max_constraint_residual: float
    max_vertex_displacement: float
    all_vertices_finite: bool
    all_socket_positions_finite: bool


@dataclass(frozen=True)
class MechanicsDynamicIntegrityCourtV1:
    canonical_mesh_lineage_hash: str
    canonical_rig_lineage_hash: str
    canonical_skin_lineage_hash: str
    canonical_mechanics_lineage_hash: str
    examined_control_count: int
    examined_face_count: int
    total_face_count: int
    intervention_steps: tuple[float, ...]
    probes: tuple[DynamicIntegrityProbeV1, ...]
    rest_intersection_count: int
    maximum_new_intersection_count: int
    complete_face_and_control_coverage: bool
    no_new_self_intersections: bool
    all_probe_vertices_finite: bool
    source_contact_qualified: bool = False
    learned_secondary_motion_qualified: bool = False
    source_pose_quality_qualified: bool = False
    native_render_parity_qualified: bool = False
    product_revision_minted: bool = False
    status: str = "INCOMPLETE_OR_UNQUALIFIED"
    court_hash: str = ""
    schema: str = SCHEMA

    def to_dict(self):
        return asdict(self)


def compile_mechanics_dynamic_integrity_court_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    *,
    max_faces: int = 1024,
    max_controls: int = 128,
    intervention_steps: tuple[float, ...] = (.25, .65, 1.),
    relative_translation: float = .02,
    max_constraint_residual: float = .001,
) -> MechanicsDynamicIntegrityCourtV1:
    if (
        type(mesh) is not CanonicalMeshIR
        or type(rig) is not CanonicalRigIRV2
        or type(skin) is not CanonicalSkinIRV2
        or type(mechanics) is not CanonicalMechanicsIRV1
        or rig.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.deform_structure_hash != rig.deform_structure_hash
        or mechanics.mesh_lineage_hash != mesh.mesh_lineage_hash
        or mechanics.rig_lineage_hash != rig.rig_lineage_hash
        or mechanics.skin_lineage_hash != skin.skin_lineage_hash
    ):
        raise QualificationError("DYNAMIC_INTEGRITY_CANONICAL_PARENT_DRIFT")
    if (
        type(max_faces) is not int or not 1 <= max_faces <= 4096
        or type(max_controls) is not int or not 1 <= max_controls <= 128
        or type(intervention_steps) is not tuple
        or not 1 <= len(intervention_steps) <= 4
        or any(type(v) not in (float, int) or not math.isfinite(v)
               or not 0 < v <= 1 for v in intervention_steps)
        or len(set(intervention_steps)) != len(intervention_steps)
        or tuple(sorted(intervention_steps)) != intervention_steps
        or type(relative_translation) not in (float, int)
        or not math.isfinite(relative_translation) or not 0 < relative_translation <= .1
        or type(max_constraint_residual) not in (float, int)
        or not math.isfinite(max_constraint_residual)
        or not 0 <= max_constraint_residual <= 1.
    ):
        raise QualificationError("DYNAMIC_INTEGRITY_POLICY_INVALID")
    validate_canonical_mesh_domain_v1(mesh)
    # Too-large meshes are explicitly censored, not "sampled and passed".
    # Full-collision qualification needs every canonical face and control.
    if (
        len(mesh.faces) > max_faces or len(rig.controls) > max_controls
        or len(mesh.faces) * max(1, len(rig.controls)) * len(intervention_steps)
        > 32_768
    ):
        result = MechanicsDynamicIntegrityCourtV1(
            canonical_mesh_lineage_hash=mesh.mesh_lineage_hash,
            canonical_rig_lineage_hash=rig.rig_lineage_hash,
            canonical_skin_lineage_hash=skin.skin_lineage_hash,
            canonical_mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
            examined_control_count=0,
            examined_face_count=0,
            total_face_count=len(mesh.faces),
            intervention_steps=tuple(map(float, intervention_steps)),
            probes=(), rest_intersection_count=-1,
            maximum_new_intersection_count=-1,
            complete_face_and_control_coverage=False,
            no_new_self_intersections=False,
            all_probe_vertices_finite=False,
            status="INCOMPLETE_RESOURCE_BUDGET_NOT_COLLISION_PASS",
        )
        return replace(result, court_hash=content_sha256({
            k: v for k, v in result.to_dict().items() if k != "court_hash"
        }))
    if not rig.controls:
        raise QualificationError("DYNAMIC_INTEGRITY_NO_CONTROL_TO_INTERVENE")
    id_to_index = {
        row.canonical_mesh_vertex_id: i
        for i, row in enumerate(mesh.vertices)
    }
    faces = np.asarray([
        [id_to_index[vertex_id] for vertex_id in face]
        for face in mesh.faces
    ], dtype=np.int64)
    rest = np.asarray([row.P for row in mesh.vertices], dtype=np.float64)
    scale = float(np.max(np.ptp(rest, axis=0)))
    if not math.isfinite(scale) or scale <= 1e-9:
        raise QualificationError("DYNAMIC_INTEGRITY_MESH_SCALE_INVALID")
    rest_pose = solve_control_pose_v1(rig, ())
    rest_deformed = deform_mesh_lbs_v1(mesh, rig, skin, rest_pose)
    rest_positions = np.asarray(rest_deformed.positions, dtype=np.float64)
    rest_hits = set(unexpected_intersection_pairs(
        vertices=rest_positions, faces=faces,
    ))
    rest_sockets = solve_attachment_socket_world_v1(rig, rest_pose, ())
    probes: list[DynamicIntegrityProbeV1] = []
    maximum_new = 0
    for control in rig.controls:
        for fraction in intervention_steps:
            shift = (
                scale * relative_translation * float(fraction),
                scale * relative_translation * float(fraction) * .4,
                0.,
            )
            action = (ControlTransformIRV1(
                canonical_control_id=control.canonical_control_id,
                translation=shift,
            ),)
            pose = solve_control_pose_v1(rig, action)
            residual = max((float(v) for _, v in pose.constraint_residuals), default=0.)
            if not math.isfinite(residual) or residual > max_constraint_residual:
                raise QualificationError("DYNAMIC_INTEGRITY_CONSTRAINT_RESIDUAL_EXCEEDED")
            deformed = deform_mesh_lbs_v1(mesh, rig, skin, pose)
            if (
                deformed.canonical_mesh_vertex_ids
                != rest_deformed.canonical_mesh_vertex_ids
                or deformed.skin_lineage_hash != skin.skin_lineage_hash
                or deformed.mesh_lineage_hash != mesh.mesh_lineage_hash
            ):
                raise QualificationError("DYNAMIC_INTEGRITY_VERTEX_OR_SKIN_ORDER_DRIFT")
            xyz = np.asarray(deformed.positions, dtype=np.float64)
            if xyz.shape != rest_positions.shape or not np.isfinite(xyz).all():
                raise QualificationError("DYNAMIC_INTEGRITY_POSED_VERTEX_INVALID")
            sockets = solve_attachment_socket_world_v1(rig, pose, action)
            if set(sockets) != set(rest_sockets) or any(
                len(point) != 3 or not np.isfinite(point).all()
                for point in sockets.values()
            ):
                raise QualificationError("DYNAMIC_INTEGRITY_SOCKET_ID_OR_POSITION_DRIFT")
            hits = set(unexpected_intersection_pairs(
                vertices=xyz, faces=faces,
            ))
            added = hits - rest_hits
            maximum_new = max(maximum_new, len(added))
            probes.append(DynamicIntegrityProbeV1(
                control_id=control.canonical_control_id,
                intervention_scale=float(fraction),
                posed_mesh_sha256=content_sha256(deformed.to_dict()),
                pose_hash=content_sha256(pose.to_dict()),
                socket_hash=content_sha256(sockets),
                new_intersection_count=len(added),
                missing_rest_intersection_count=len(rest_hits - hits),
                total_posed_intersection_count=len(hits),
                max_constraint_residual=residual,
                max_vertex_displacement=float(
                    np.max(np.linalg.norm(xyz - rest_positions, axis=1))
                ),
                all_vertices_finite=True,
                all_socket_positions_finite=True,
            ))
    collision_pass = maximum_new == 0
    result = MechanicsDynamicIntegrityCourtV1(
        canonical_mesh_lineage_hash=mesh.mesh_lineage_hash,
        canonical_rig_lineage_hash=rig.rig_lineage_hash,
        canonical_skin_lineage_hash=skin.skin_lineage_hash,
        canonical_mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
        examined_control_count=len(rig.controls),
        examined_face_count=len(mesh.faces),
        total_face_count=len(mesh.faces),
        intervention_steps=tuple(map(float, intervention_steps)),
        probes=tuple(probes),
        rest_intersection_count=len(rest_hits),
        maximum_new_intersection_count=maximum_new,
        complete_face_and_control_coverage=True,
        no_new_self_intersections=collision_pass,
        all_probe_vertices_finite=True,
        status=(
            "PASS_BOUNDED_FULL_CONTROL_DYNAMIC_GEOMETRY"
            if collision_pass else "FAIL_NEW_SELF_INTERSECTION"
        ),
    )
    return replace(result, court_hash=content_sha256({
        k: v for k, v in result.to_dict().items() if k != "court_hash"
    }))


def validate_mechanics_dynamic_integrity_court_v1(
    receipt: dict, *,
    mesh: CanonicalMeshIR, rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2, mechanics: CanonicalMechanicsIRV1,
) -> MechanicsDynamicIntegrityCourtV1:
    """Independent Proof consumer reruns actual interventions, not flags.

    The court is canonical only with the currently compiled algorithm and
    exactly the same parent IDs and policy. A caller cannot rehash a forged
    zero-intersections list into proof.
    """
    if (
        type(receipt) is not dict
        or receipt.get("schema") != SCHEMA
        or receipt.get("canonical_mesh_lineage_hash") != mesh.mesh_lineage_hash
        or receipt.get("canonical_rig_lineage_hash") != rig.rig_lineage_hash
        or receipt.get("canonical_skin_lineage_hash") != skin.skin_lineage_hash
        or receipt.get("canonical_mechanics_lineage_hash") != mechanics.mechanics_lineage_hash
    ):
        raise QualificationError("DYNAMIC_INTEGRITY_REPLAY_PARENT_OR_SCHEMA_DRIFT")
    if receipt.get("source_contact_qualified") is not False or receipt.get(
        "learned_secondary_motion_qualified"
    ) is not False or receipt.get("product_revision_minted") is not False:
        raise QualificationError("DYNAMIC_INTEGRITY_PREMATURE_AUTHORITY")
    try:
        replayed = compile_mechanics_dynamic_integrity_court_v1(
            mesh, rig, skin, mechanics,
            max_faces=1024, max_controls=128,
            intervention_steps=(.25, .65, 1.),
            relative_translation=.02,
            max_constraint_residual=.001,
        )
    except (ValueError, TypeError) as exc:
        raise QualificationError("DYNAMIC_INTEGRITY_REPLAY_FAILED") from exc
    if receipt != replayed.to_dict():
        raise QualificationError("DYNAMIC_INTEGRITY_REPLAY_BYTES_OR_METRICS_DRIFT")
    return replayed


__all__ = [
    "SCHEMA", "DynamicIntegrityProbeV1", "MechanicsDynamicIntegrityCourtV1",
    "compile_mechanics_dynamic_integrity_court_v1",
    "validate_mechanics_dynamic_integrity_court_v1",
]
