from __future__ import annotations

"""Explicit canonical surface-particle XPBD, distinct from secondary bone XPBD.

Authority: canonical TESSA mesh + primary MIRA-skinned vertex positions.
An editor/Compiler-qualified *face selection* and pinned vertex IDs choose the
simulation domain. Source segmentation, proximity, and PartVerse labels do NOT
authorize dynamics or reveal physical material parameters.

The solver applies XPBD stretch/shear (triangle edges), opposite-vertex bend,
and triangle-area constraints with per-substep Lagrange multipliers. Plane and static sphere
contacts are explicit world runtime policies. Joint-attached spheres are
explicit canonical-joint contact proxies, interpolated across fixed steps. The solver modifies ONLY
vertex positions of the existing DeformedMeshIRV1 and retains the original
mesh/face addresses and the primary Mechanics pose.

Optional discrete point-triangle and edge-edge self-contact are resolved by
the same canonical surface particles. This is not swept CCD and it does not
resolve face-face coplanar overlaps, tearing, friction, dihedral calibrated
bending or source material coefficients.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from collections import deque
from typing import Any, Mapping

import numpy as np

from .canonical_mesh_ir_v1 import (
    CanonicalMeshIR, validate_canonical_mesh_domain_v1,
)
from .canonical_rig_ir_v2 import (
    CanonicalRigIRV2, validate_canonical_rig_domain_v2,
)
from .canonical_skin_ir_v2 import (
    CanonicalSkinIRV2, validate_canonical_skin_domain_v2,
)
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .hashing import content_sha256
from .mechanics_control_solver_v1 import DeformedMeshIRV1, MechanicsPoseIRV1
from .mechanics_xpbd_secondary_v1 import (
    XPBDPlaneColliderV1, XPBDSphereColliderV1,
)
from .types import QualificationError
from .mechanics_xpbd_surface_self_contact_v1 import (
    prepare_surface_self_contact_topology_v1,
    evaluate_surface_self_contacts_v1,
)

Vec3 = tuple[float, float, float]
Face3 = tuple[str, str, str]


@dataclass(frozen=True)
class DeclaredXPBDSurfaceRequestV1:
    """A selected topological patch; no semantic inference or auto-pinning."""

    selected_faces: tuple[Face3, ...]
    pinned_vertex_ids: tuple[str, ...]
    activation_authority: str = "EXPLICIT_RUNTIME_POLICY"
    max_vertices: int = 4096
    max_faces: int = 8192


@dataclass(frozen=True)
class XPBDSurfaceDomainV1:
    mesh_lineage_hash: str
    rig_lineage_hash: str
    skin_lineage_hash: str
    mechanics_lineage_hash: str
    deform_structure_hash: str
    control_program_hash: str
    selected_faces: tuple[Face3, ...]
    vertex_ids: tuple[str, ...]
    pinned_vertex_ids: tuple[str, ...]
    dynamic_vertex_ids: tuple[str, ...]
    stretch_edges: tuple[tuple[str, str, float], ...]
    bend_edges: tuple[tuple[str, str, float], ...]
    triangle_rest_areas: tuple[float, ...]
    domain_hash: str
    schema_version: str = "RealSaS.XPBD.SurfaceDomain.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class XPBDSurfaceJointSphereColliderV1:
    """Explicit proxy attached to a canonical joint, not inferred cloth truth."""

    collider_id: str
    canonical_joint_id: str
    radius: float
    margin: float = 0.0
    excluded_vertex_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class XPBDSurfacePolicyV1:
    domain_hash: str
    inverse_mass: float = 1.0
    stretch_compliance: float = 0.0
    bend_compliance: float = 1e-4
    area_compliance: float = 1e-6
    follow_compliance: float = 0.015
    gravity_xyz: Vec3 = (0.0, -9.81, 0.0)
    damping_per_second: float = 1.5
    planes: tuple[XPBDPlaneColliderV1, ...] = ()
    spheres: tuple[XPBDSphereColliderV1, ...] = ()
    joint_spheres: tuple[XPBDSurfaceJointSphereColliderV1, ...] = field(default=(), kw_only=True)
    self_collision_thickness: float = field(default=0.0, kw_only=True)
    self_collision_compliance: float = field(default=0.0, kw_only=True)
    max_self_collision_pairs: int = field(default=16384, kw_only=True)
    max_self_collision_cells: int = field(default=200000, kw_only=True)
    collision_margin: float = 0.0
    fixed_step_seconds: float = 1.0/120.0
    maximum_dt_seconds: float = 0.25
    maximum_substeps: int = 32
    iterations: int = 20
    max_stretch_relative_error: float = 0.025
    max_area_relative_error: float = 0.08
    max_contact_penetration: float = 1e-4
    schema_version: str = "RealSaS.XPBD.SurfacePolicy.v1"


@dataclass(frozen=True)
class XPBDSurfaceParticleStateV1:
    canonical_mesh_vertex_id: str
    world_P: Vec3
    world_velocity: Vec3


@dataclass(frozen=True)
class XPBDSurfaceStateV1:
    domain_hash: str
    policy_hash: str
    primary_deformation_hash: str
    particles: tuple[XPBDSurfaceParticleStateV1, ...]
    pinned_positions: tuple[tuple[str, Vec3], ...]
    ticks: int
    frames: int
    state_hash: str
    schema_version: str = "RealSaS.XPBD.SurfaceState.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class XPBDSurfaceResultV1:
    deformation: DeformedMeshIRV1
    state: XPBDSurfaceStateV1
    status: str
    max_stretch_relative_error: float
    max_area_relative_error: float
    max_contact_penetration: float
    contact_projections: int
    substeps: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _xyz(value: object, code: str) -> np.ndarray:
    try:
        a=np.asarray(value,dtype=np.float64)
    except (TypeError,ValueError) as exc:
        raise QualificationError(code) from exc
    if a.shape!=(3,) or not np.isfinite(a).all():
        raise QualificationError(code)
    return a


def _scalar(value: object, code: str, *, lo=0., hi=1e5,
            positive=False) -> float:
    try:
        v=float(value)
    except (TypeError,ValueError) as exc:
        raise QualificationError(code) from exc
    if not math.isfinite(v) or v>hi or (v<=lo if positive else v<lo):
        raise QualificationError(code)
    return v


def _triangle_area(a: np.ndarray,b: np.ndarray,c: np.ndarray) -> float:
    return 0.5*float(np.linalg.norm(np.cross(b-a,c-a)))


def prepare_declared_xpbd_surface_domain_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    request: DeclaredXPBDSurfaceRequestV1,
) -> XPBDSurfaceDomainV1:
    if (
        mesh.mesh_lineage_hash != rig.mesh_lineage_hash
        or mesh.mesh_lineage_hash != skin.mesh_lineage_hash
        or mesh.mesh_lineage_hash != mechanics.mesh_lineage_hash
        or rig.rig_lineage_hash != mechanics.rig_lineage_hash
        or rig.control_program_hash != mechanics.control_program_hash
        or rig.deform_structure_hash != mechanics.deform_structure_hash
        or rig.deform_structure_hash != skin.deform_structure_hash
        or skin.skin_lineage_hash != mechanics.skin_lineage_hash
    ):
        raise QualificationError("XPBD_SURFACE_PARENT_LINEAGE_DRIFT")
    validate_canonical_mesh_domain_v1(mesh)
    validate_canonical_rig_domain_v2(rig)
    validate_canonical_skin_domain_v2(skin)
    if (
        mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH"
        or rig.metadata.get("authority") != "COMPILER_CANONICAL_RIG"
        or skin.metadata.get("authority") != "COMPILER_CANONICAL_SKIN"
        or mechanics.metadata.get("authority") != "COMPILER_CANONICAL_MECHANICS"
    ):
        raise QualificationError("XPBD_SURFACE_PARENT_AUTHORITY_UNQUALIFIED")
    if (
        type(request) is not DeclaredXPBDSurfaceRequestV1
        or request.activation_authority != "EXPLICIT_RUNTIME_POLICY"
        or not isinstance(request.selected_faces,tuple)
        or not isinstance(request.pinned_vertex_ids,tuple)
        or not request.selected_faces or not request.pinned_vertex_ids
        or len(request.selected_faces)>8192
        or type(request.max_faces) is not int
        or not 1 <= request.max_faces <= 8192
        or len(request.selected_faces)>request.max_faces
        or type(request.max_vertices) is not int
        or not 3 <= request.max_vertices <= 4096
    ):
        raise QualificationError("XPBD_SURFACE_EXPLICIT_SELECTION_REQUIRED")
    mesh_verts={v.canonical_mesh_vertex_id:v for v in mesh.vertices}
    if len(mesh_verts)!=len(mesh.vertices):
        raise QualificationError("XPBD_SURFACE_MESH_ADDRESS_DUPLICATE")
    faces=set(mesh.faces)
    if len(faces)!=len(mesh.faces):
        raise QualificationError("XPBD_SURFACE_SOURCE_FACE_DUPLICATE")
    selected=request.selected_faces
    if len(set(selected))!=len(selected) or any(
        face not in faces for face in selected
    ):
        raise QualificationError("XPBD_SURFACE_SELECTED_FACE_NOT_SOURCE")
    selected_vertex_ids=set(v for face in selected for v in face)
    if len(selected_vertex_ids)>request.max_vertices:
        raise QualificationError("XPBD_SURFACE_VERTEX_RESOURCE_BOUND")
    pinned=set(request.pinned_vertex_ids)
    if len(pinned)!=len(request.pinned_vertex_ids) or not pinned <= selected_vertex_ids:
        raise QualificationError("XPBD_SURFACE_PINS_NOT_SELECTED_VERTICES")
    dynamic=selected_vertex_ids-pinned
    if not dynamic:
        raise QualificationError("XPBD_SURFACE_NO_DYNAMIC_VERTICES")
    # Prevent hidden seams: every selected vertex touching an unselected
    # triangle is a fixed boundary. Internal selected edges can move freely.
    for face in mesh.faces:
        if face not in set(selected) and any(x in selected_vertex_ids for x in face):
            if any(x in dynamic for x in face):
                raise QualificationError("XPBD_SURFACE_CUT_BOUNDARY_MUST_BE_PINNED")
    components={mesh_verts[v].component_id for v in selected_vertex_ids}
    if len(components)!=1:
        raise QualificationError("XPBD_SURFACE_CROSS_COMPONENT_SELECTION")
    rest={v: _xyz(mesh_verts[v].P,"XPBD_SURFACE_REST_VERTEX_INVALID")
          for v in selected_vertex_ids}
    edge_faces:dict[tuple[str,str],list[str]]={}
    neighbors={v:set() for v in selected_vertex_ids}
    for face in selected:
        a,b,c=face
        if _triangle_area(rest[a],rest[b],rest[c])<=1e-10:
            raise QualificationError("XPBD_SURFACE_DEGENERATE_TRIANGLE")
        for x,y,opp in ((a,b,c),(b,c,a),(c,a,b)):
            edge=tuple(sorted((x,y)))
            edge_faces.setdefault(edge,[]).append(opp)
            neighbors[x].add(y);neighbors[y].add(x)
    if any(len(x)>2 for x in edge_faces.values()):
        raise QualificationError("XPBD_SURFACE_NONMANIFOLD_EDGE")
    mesh_edges={tuple(sorted(e)) for e in mesh.edges}
    if set(edge_faces)-mesh_edges:
        raise QualificationError("XPBD_SURFACE_MISSING_CANONICAL_EDGE")
    # Every connected selected patch must have at least one anchor.
    seen=set(pinned)
    queue=deque(pinned)
    while queue:
        v=queue.popleft()
        for w in neighbors[v]-seen:
            seen.add(w);queue.append(w)
    if seen!=selected_vertex_ids:
        raise QualificationError("XPBD_SURFACE_UNPINNED_COMPONENT")
    skin_ids={row.canonical_mesh_vertex_id for row in skin.rows}
    if skin_ids!={v.canonical_mesh_vertex_id for v in mesh.vertices}:
        raise QualificationError("XPBD_SURFACE_SKIN_VERTEX_COVERAGE_DRIFT")
    stretches=tuple((a,b,float(np.linalg.norm(rest[a]-rest[b])))
                    for a,b in sorted(edge_faces))
    bends=tuple(sorted((
        tuple(sorted(opps))+(float(np.linalg.norm(rest[opps[0]]-rest[opps[1]])),)
        for edge,opps in edge_faces.items() if len(opps)==2
    )))
    areas=tuple(_triangle_area(*(rest[x] for x in f)) for f in selected)
    result=XPBDSurfaceDomainV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        selected_faces=selected,
        vertex_ids=tuple(sorted(selected_vertex_ids)),
        pinned_vertex_ids=tuple(sorted(pinned)),
        dynamic_vertex_ids=tuple(sorted(dynamic)),
        stretch_edges=stretches,
        bend_edges=bends,
        triangle_rest_areas=areas,
        domain_hash="",
        metadata={
            "authority":"EXPLICIT_RUNTIME_POLICY",
            "partverse_label_is_not_motion_truth":True,
            "source_material_parameters_qualified":False,
            "surface_solver_region":"CANONICAL_TESSA_FACE_PATCH",
        },
    )
    return replace(result,domain_hash=content_sha256({
        k:v for k,v in result.to_dict().items() if k!="domain_hash"
    }))


def requalify_declared_xpbd_surface_domain_v1(
    mesh: CanonicalMeshIR, rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2, mechanics: CanonicalMechanicsIRV1,
    domain: XPBDSurfaceDomainV1,
) -> None:
    if type(domain) is not XPBDSurfaceDomainV1:
        raise QualificationError("XPBD_SURFACE_DOMAIN_TYPE_INVALID")
    if (
        domain.mesh_lineage_hash != mesh.mesh_lineage_hash
        or domain.rig_lineage_hash != rig.rig_lineage_hash
        or domain.skin_lineage_hash != skin.skin_lineage_hash
        or domain.mechanics_lineage_hash != mechanics.mechanics_lineage_hash
        or domain.deform_structure_hash != rig.deform_structure_hash
        or domain.control_program_hash != rig.control_program_hash
    ):
        raise QualificationError("XPBD_SURFACE_PARENT_LINEAGE_DRIFT")
    expected=prepare_declared_xpbd_surface_domain_v1(
        mesh,rig,skin,mechanics,
        DeclaredXPBDSurfaceRequestV1(
            selected_faces=domain.selected_faces,
            pinned_vertex_ids=domain.pinned_vertex_ids,
        ),
    )
    if expected.to_dict()!=domain.to_dict():
        raise QualificationError("XPBD_SURFACE_DOMAIN_REQUALIFICATION_DRIFT")


def _policy(domain: XPBDSurfaceDomainV1, policy: XPBDSurfacePolicyV1) -> str:
    if (
        domain.schema_version != "RealSaS.XPBD.SurfaceDomain.v1"
        or domain.domain_hash != content_sha256({
            k:v for k,v in domain.to_dict().items() if k!="domain_hash"
        })
        or policy.domain_hash!=domain.domain_hash
        or policy.schema_version != "RealSaS.XPBD.SurfacePolicy.v1"
    ):
        raise QualificationError("XPBD_SURFACE_POLICY_DOMAIN_DRIFT")
    for field_name in ("inverse_mass","stretch_compliance","bend_compliance",
                       "area_compliance","follow_compliance","collision_margin",
                       "damping_per_second"):
        val=getattr(policy,field_name)
        _scalar(val,"XPBD_SURFACE_PARAMETER_INVALID",
                hi=1e4,positive=field_name in ("inverse_mass","follow_compliance"))
    _xyz(policy.gravity_xyz,"XPBD_SURFACE_GRAVITY_INVALID")
    h=_scalar(policy.fixed_step_seconds,"XPBD_SURFACE_STEP_INVALID",
              positive=True,hi=.1)
    _scalar(policy.maximum_dt_seconds,"XPBD_SURFACE_MAX_DT_INVALID",lo=h,hi=.25)
    if (type(policy.iterations) is not int or not 1<=policy.iterations<=64
            or type(policy.maximum_substeps) is not int
            or not 1<=policy.maximum_substeps<=64):
        raise QualificationError("XPBD_SURFACE_ITERATION_RESOURCE_BOUND")
    for key in ("max_stretch_relative_error","max_area_relative_error",
                "max_contact_penetration"):
        _scalar(getattr(policy,key),"XPBD_SURFACE_RESIDUAL_POLICY_INVALID",hi=0.5)
    _scalar(policy.self_collision_thickness,"XPBD_SELF_CONTACT_THICKNESS_INVALID",hi=100.)
    _scalar(policy.self_collision_compliance,"XPBD_SELF_CONTACT_COMPLIANCE_INVALID",hi=10.)
    if (type(policy.max_self_collision_pairs) is not int
            or not 1<=policy.max_self_collision_pairs<=32768
            or type(policy.max_self_collision_cells) is not int
            or not 1<=policy.max_self_collision_cells<=200000):
        raise QualificationError("XPBD_SELF_CONTACT_RESOURCE_POLICY_INVALID")
    if (policy.self_collision_thickness == 0.
            and policy.self_collision_compliance != 0.):
        raise QualificationError("XPBD_SELF_CONTACT_INACTIVE_COMPLIANCE")
    seen=set()
    for collider in policy.planes:
        if not collider.collider_id or collider.collider_id in seen:
            raise QualificationError("XPBD_SURFACE_COLLIDER_DUPLICATE")
        seen.add(collider.collider_id)
        _xyz(collider.point,"XPBD_SURFACE_PLANE_POINT_INVALID")
        n=_xyz(collider.normal,"XPBD_SURFACE_PLANE_NORMAL_INVALID")
        if abs(float(np.linalg.norm(n))-1)>1e-6:
            raise QualificationError("XPBD_SURFACE_PLANE_NORMAL_NOT_UNIT")
        _scalar(collider.margin,"XPBD_SURFACE_COLLIDER_MARGIN_INVALID",hi=1000.)
    for collider in policy.spheres:
        if not collider.collider_id or collider.collider_id in seen:
            raise QualificationError("XPBD_SURFACE_COLLIDER_DUPLICATE")
        seen.add(collider.collider_id)
        _xyz(collider.center,"XPBD_SURFACE_SPHERE_CENTER_INVALID")
        _scalar(collider.radius,"XPBD_SURFACE_SPHERE_RADIUS_INVALID",
                positive=True,hi=1000.)
        _scalar(collider.margin,"XPBD_SURFACE_COLLIDER_MARGIN_INVALID",hi=1000.)
    for collider in policy.joint_spheres:
        if (type(collider) is not XPBDSurfaceJointSphereColliderV1
                or not isinstance(collider.collider_id, str)
                or not isinstance(collider.canonical_joint_id, str)
                or not collider.collider_id or not collider.canonical_joint_id
                or collider.collider_id in seen):
            raise QualificationError("XPBD_SURFACE_JOINT_SPHERE_INVALID")
        seen.add(collider.collider_id)
        _scalar(collider.radius,"XPBD_SURFACE_JOINT_SPHERE_RADIUS_INVALID",
                positive=True,hi=1000.)
        _scalar(collider.margin,"XPBD_SURFACE_COLLIDER_MARGIN_INVALID",hi=1000.)
        if (type(collider.excluded_vertex_ids) is not tuple
                or len(set(collider.excluded_vertex_ids))!=len(collider.excluded_vertex_ids)
                or not set(collider.excluded_vertex_ids)<=set(domain.dynamic_vertex_ids)):
            raise QualificationError("XPBD_SURFACE_JOINT_SPHERE_EXCLUSION_INVALID")
    if len(seen)>16 or len(domain.dynamic_vertex_ids)>4096:
        raise QualificationError("XPBD_SURFACE_CONTACT_RESOURCE_BOUND")
    return content_sha256(asdict(policy))


def resolve_declared_xpbd_surface_joint_positions_v1(
    pose: MechanicsPoseIRV1, policy: XPBDSurfacePolicyV1,
) -> dict[str, Vec3]:
    """Resolve joint collider centers from the authoritative solved pose.

    This is the single bridge used by both live Mechanics and the editor;
    callers cannot substitute a label-derived or source-rest joint location.
    """
    required={row.canonical_joint_id for row in policy.joint_spheres}
    if not required:
        return {}
    if not isinstance(pose, MechanicsPoseIRV1):
        raise QualificationError("XPBD_SURFACE_SOLVED_POSE_REQUIRED")
    by_id={row.canonical_joint_id:row.solved_P for row in pose.joints}
    if len(by_id)!=len(pose.joints) or not required<=set(by_id):
        raise QualificationError(
            "XPBD_SURFACE_COLLIDER_JOINT_NOT_IN_SOLVED_POSE"
        )
    return {
        jid:tuple(map(float,_xyz(
            by_id[jid],"XPBD_SURFACE_KINEMATIC_JOINT_POSITION_INVALID",
        ))) for jid in sorted(required)
    }


def _state_hash(state: XPBDSurfaceStateV1) -> str:
    return content_sha256({
        k:v for k,v in state.to_dict().items() if k!="state_hash"
    })


def xpbd_surface_state_from_dict_v1(payload: Mapping[str, Any]) -> XPBDSurfaceStateV1:
    if not isinstance(payload,Mapping):
        raise QualificationError("XPBD_SURFACE_STATE_WIRE_INVALID")
    try:
        state=XPBDSurfaceStateV1(
            domain_hash=str(payload["domain_hash"]),
            policy_hash=str(payload["policy_hash"]),
            primary_deformation_hash=str(payload["primary_deformation_hash"]),
            particles=tuple(XPBDSurfaceParticleStateV1(
                str(p["canonical_mesh_vertex_id"]),
                tuple(float(x) for x in p["world_P"]),
                tuple(float(x) for x in p["world_velocity"])
            ) for p in payload["particles"]),
            pinned_positions=tuple(
                (str(i),tuple(float(x) for x in xyz))
                for i,xyz in payload["pinned_positions"]),
            ticks=payload["ticks"],frames=payload["frames"],
            state_hash=str(payload["state_hash"]),
            schema_version=str(payload["schema_version"]),
            metadata=dict(payload["metadata"]),
        )
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        raise QualificationError("XPBD_SURFACE_STATE_WIRE_INVALID") from exc
    if (
        state.schema_version!="RealSaS.XPBD.SurfaceState.v1"
        or type(state.ticks) is not int or state.ticks<0
        or type(state.frames) is not int or state.frames<0
    ):
        raise QualificationError("XPBD_SURFACE_STATE_WIRE_INVALID")
    for p in state.particles:
        if not p.canonical_mesh_vertex_id:
            raise QualificationError("XPBD_SURFACE_STATE_PARTICLE_ID_INVALID")
        _xyz(p.world_P,"XPBD_SURFACE_STATE_NONFINITE")
        _xyz(p.world_velocity,"XPBD_SURFACE_STATE_NONFINITE")
    for jid,xyz in state.pinned_positions:
        if not jid: raise QualificationError("XPBD_SURFACE_STATE_PIN_ID_INVALID")
        _xyz(xyz,"XPBD_SURFACE_STATE_NONFINITE")
    if state.state_hash != _state_hash(state):
        raise QualificationError("XPBD_SURFACE_STATE_HASH_DRIFT")
    return state


def _distance_pair(pos: dict[str,np.ndarray], a: str,b: str,
                   rest: float,lam:float,compliance:float,
                   h:float,inv_mass:dict[str,float]) -> float:
    d=pos[a]-pos[b]
    length=float(np.linalg.norm(d))
    if length<=1e-12:
        raise QualificationError("XPBD_SURFACE_DISTANCE_COLLAPSED")
    n=d/length
    wa,wb=inv_mass[a],inv_mass[b]
    alpha=compliance/(h*h)
    denominator=wa+wb+alpha
    if denominator<=1e-18:
        # Two fixed vertices with zero compliance have no movable DOF.
        # If they already satisfy their constraint, it is a no-op;
        # otherwise this frame's authored pins are geometrically incompatible.
        if abs(length-rest)>1e-8:
            raise QualificationError("XPBD_SURFACE_FIXED_CONSTRAINT_CONFLICT")
        return lam
    dl=-(length-rest+alpha*lam)/denominator
    pos[a]=pos[a]+wa*dl*n
    pos[b]=pos[b]-wb*dl*n
    return lam+dl


def _triangle_xpbd(
    pos: dict[str,np.ndarray],face:Face3,rest:float,
    lam:float,compliance:float,h:float,inv_mass:dict[str,float],
) -> float:
    a,b,c=face
    cross=np.cross(pos[b]-pos[a],pos[c]-pos[a])
    length=float(np.linalg.norm(cross))
    if length<=1e-12:
        raise QualificationError("XPBD_SURFACE_AREA_COLLAPSED")
    normal=cross/length
    area=.5*length
    grads=(
        .5*np.cross(pos[b]-pos[c],normal),
        .5*np.cross(pos[c]-pos[a],normal),
        .5*np.cross(pos[a]-pos[b],normal),
    )
    den=sum(inv_mass[j]*float(g@g) for j,g in zip(face,grads))
    alpha=compliance/(h*h)
    if den+alpha<=1e-18:
        if abs(area-rest)>1e-8:
            raise QualificationError("XPBD_SURFACE_FIXED_AREA_CONFLICT")
        return lam
    dl=-(area-rest+alpha*lam)/(den+alpha)
    for jid,g in zip(face,grads):
        pos[jid]=pos[jid]+inv_mass[jid]*dl*g
    return lam+dl


def step_declared_xpbd_surface_v1(
    mesh: CanonicalMeshIR, primary_deformation: DeformedMeshIRV1,
    domain: XPBDSurfaceDomainV1, policy: XPBDSurfacePolicyV1,
    *, previous_state: XPBDSurfaceStateV1 | None = None,
    dt_seconds: float | None = None,
    joint_world_positions: Mapping[str, Vec3] | None = None,
) -> XPBDSurfaceResultV1:
    signature=_policy(domain,policy)
    required_joint_ids={c.canonical_joint_id for c in policy.joint_spheres}
    if (not isinstance(joint_world_positions, Mapping)
            and joint_world_positions is not None):
        raise QualificationError("XPBD_SURFACE_KINEMATIC_JOINT_COVERAGE_DRIFT")
    if set(joint_world_positions or {})!=required_joint_ids:
        raise QualificationError("XPBD_SURFACE_KINEMATIC_JOINT_COVERAGE_DRIFT")
    joint_world={
        jid:_xyz(joint_world_positions[jid],
                 "XPBD_SURFACE_KINEMATIC_JOINT_POSITION_INVALID")
        for jid in sorted(required_joint_ids)
    }
    joint_centers={
        c.collider_id:joint_world[c.canonical_joint_id]
        for c in policy.joint_spheres
    }
    if (
        primary_deformation.mesh_lineage_hash!=domain.mesh_lineage_hash
        or primary_deformation.skin_lineage_hash!=domain.skin_lineage_hash
        or primary_deformation.rig_lineage_hash!=domain.rig_lineage_hash
        or mesh.mesh_lineage_hash!=domain.mesh_lineage_hash
    ):
        raise QualificationError("XPBD_SURFACE_PRIMARY_LINEAGE_DRIFT")
    ids=primary_deformation.canonical_mesh_vertex_ids
    if (len(ids)!=len(mesh.vertices) or len(ids)!=len(set(ids))
            or set(ids)!={v.canonical_mesh_vertex_id for v in mesh.vertices}
            or len(primary_deformation.positions)!=len(ids)):
        raise QualificationError("XPBD_SURFACE_PRIMARY_VERTEX_COVERAGE_DRIFT")
    primary={vid:_xyz(p,"XPBD_SURFACE_PRIMARY_POSITION_INVALID")
             for vid,p in zip(ids,primary_deformation.positions)}
    primary_hash=content_sha256({
        "mesh":domain.mesh_lineage_hash,
        "skin":domain.skin_lineage_hash,
        "positions":[[vid,tuple(map(float,primary[vid]))] for vid in ids],
    })
    dynamic=domain.dynamic_vertex_ids
    pins=domain.pinned_vertex_ids
    if (not dynamic or not pins or set(dynamic)&set(pins)
            or set(dynamic)|set(pins)!=set(domain.vertex_ids)):
        raise QualificationError("XPBD_SURFACE_PARTICLE_DOMAIN_INVALID")
    pos={jid: primary[jid].copy() for jid in domain.vertex_ids}
    velocity={jid:np.zeros(3,dtype=np.float64) for jid in dynamic}
    if previous_state is None:
        if dt_seconds is not None:
            raise QualificationError("XPBD_SURFACE_INITIALIZE_WITHOUT_DT")
        ticks,frames,steps=0,0,0
        prior_pins={jid:primary[jid] for jid in pins}
        prior_joint_centers={k:v.copy() for k,v in joint_centers.items()}
    else:
        if previous_state.state_hash != _state_hash(previous_state):
            raise QualificationError("XPBD_SURFACE_STATE_HASH_DRIFT")
        if (
            previous_state.schema_version!="RealSaS.XPBD.SurfaceState.v1"
            or previous_state.domain_hash!=domain.domain_hash
            or previous_state.policy_hash!=signature
            or type(previous_state.ticks) is not int
            or type(previous_state.frames) is not int
            or previous_state.ticks<0 or previous_state.frames<0
        ):
            raise QualificationError("XPBD_SURFACE_STATE_POLICY_OR_DOMAIN_DRIFT")
        rows={r.canonical_mesh_vertex_id:r for r in previous_state.particles}
        prior_pins={jid:_xyz(p,"XPBD_SURFACE_PRIOR_PIN_INVALID")
                    for jid,p in previous_state.pinned_positions}
        if (len(rows)!=len(previous_state.particles)
                or set(rows)!=set(dynamic)
                or len(prior_pins)!=len(previous_state.pinned_positions)
                or set(prior_pins)!=set(pins)):
            raise QualificationError("XPBD_SURFACE_STATE_COVERAGE_DRIFT")
        for jid in dynamic:
            pos[jid]=_xyz(rows[jid].world_P,"XPBD_SURFACE_PRIOR_POSITION_INVALID")
            velocity[jid]=_xyz(rows[jid].world_velocity,"XPBD_SURFACE_PRIOR_VELOCITY_INVALID")
        old_centers=previous_state.metadata.get(
            "kinematic_joint_sphere_centers", []
        )
        if not isinstance(old_centers,list) or any(
            not isinstance(row,(tuple,list)) or len(row)!=2
            for row in old_centers
        ):
            raise QualificationError("XPBD_SURFACE_PRIOR_KINEMATIC_CENTERS_INVALID")
        try:
            prior_joint_centers={
                cid:_xyz(p,"XPBD_SURFACE_PRIOR_KINEMATIC_CENTER_INVALID")
                for cid,p in old_centers
            }
        except (ValueError,TypeError) as exc:
            raise QualificationError(
                "XPBD_SURFACE_PRIOR_KINEMATIC_CENTERS_INVALID"
            ) from exc
        if (len(prior_joint_centers)!=len(old_centers)
                or set(prior_joint_centers)!=set(joint_centers)):
            raise QualificationError("XPBD_SURFACE_PRIOR_KINEMATIC_CENTERS_DRIFT")
        if dt_seconds is None:
            if any(not np.array_equal(prior_joint_centers[cid],center)
                   for cid,center in joint_centers.items()):
                raise QualificationError(
                    "XPBD_SURFACE_CHANGED_KINEMATIC_COLLIDER_REQUIRES_DT"
                )
            if primary_hash!=previous_state.primary_deformation_hash:
                raise QualificationError("XPBD_SURFACE_CHANGED_PRIMARY_REQUIRES_DT")
            steps=0
        else:
            dt=_scalar(dt_seconds,"XPBD_SURFACE_DT_INVALID",
                       positive=True,hi=policy.maximum_dt_seconds)
            exact=dt/policy.fixed_step_seconds
            steps=int(round(exact))
            if (not 1<=steps<=policy.maximum_substeps
                    or abs(exact-steps)>1e-7):
                raise QualificationError("XPBD_SURFACE_FIXED_STEP_PARTITION_DRIFT")
        ticks,frames=previous_state.ticks,previous_state.frames

    h=policy.fixed_step_seconds
    gravity=_xyz(policy.gravity_xyz,"XPBD_SURFACE_GRAVITY_INVALID")
    damping=math.exp(-policy.damping_per_second*h)
    mass={jid:(0. if jid in pins else policy.inverse_mass)
          for jid in domain.vertex_ids}
    # The inner XPBD constraint loop may run many times per frame.
    # A tuple search for every particle/collider pair scales quadratically
    # with exclusion-mask size. Reuse immutable O(1) membership instead.
    excluded_by_collider={
        c.collider_id:frozenset(c.excluded_vertex_ids)
        for c in policy.joint_spheres
    }
    self_topology=(
        prepare_surface_self_contact_topology_v1(
            domain.selected_faces,domain.stretch_edges,
        ) if policy.self_collision_thickness>0. else None
    )
    self_cell_size=max(
        2.*policy.self_collision_thickness,
        float(np.median([e[2] for e in domain.stretch_edges])),
    ) if self_topology is not None else 0.
    touch_count=0
    self_contact_projections=0
    self_contact_max_candidates=0
    for sub in range(steps):
        fraction=(sub+1)/steps
        for jid in pins:
            pos[jid]=prior_pins[jid]+fraction*(primary[jid]-prior_pins[jid])
        substep_joint_centers={
            cid:prior_joint_centers[cid]+fraction*(
                joint_centers[cid]-prior_joint_centers[cid])
            for cid in joint_centers
        }
        before={jid:pos[jid].copy() for jid in dynamic}
        for jid in dynamic:
            pos[jid]=pos[jid]+h*damping*velocity[jid]+h*h*gravity
        lambda_stretch=np.zeros(len(domain.stretch_edges),dtype=np.float64)
        lambda_bend=np.zeros(len(domain.bend_edges),dtype=np.float64)
        lambda_area=np.zeros(len(domain.selected_faces),dtype=np.float64)
        lambda_follow=np.zeros((len(dynamic),3),dtype=np.float64)
        lambda_self:dict[tuple, float]={}
        for _ in range(policy.iterations):
            for index,(a,b,rest) in enumerate(domain.stretch_edges):
                lambda_stretch[index]=_distance_pair(
                    pos,a,b,rest,lambda_stretch[index],
                    policy.stretch_compliance,h,mass,
                )
            for index,(a,b,rest) in enumerate(domain.bend_edges):
                lambda_bend[index]=_distance_pair(
                    pos,a,b,rest,lambda_bend[index],
                    policy.bend_compliance,h,mass,
                )
            for index,face in enumerate(domain.selected_faces):
                lambda_area[index]=_triangle_xpbd(
                    pos,face,domain.triangle_rest_areas[index],
                    lambda_area[index],policy.area_compliance,h,mass,
                )
            alpha=policy.follow_compliance/(h*h)
            for i,jid in enumerate(dynamic):
                w=mass[jid]
                dl=-(pos[jid]-primary[jid]+alpha*lambda_follow[i])/(w+alpha)
                pos[jid]=pos[jid]+w*dl
                lambda_follow[i]+=dl
            # Explicit world contacts are hard unilateral constraints. Since
            # contact is last, *final* residuals still decide acceptance.
            for jid in dynamic:
                for collider in policy.planes:
                    n=np.asarray(collider.normal,dtype=np.float64)
                    C=float(np.dot(pos[jid]-collider.point,n)
                            -collider.margin-policy.collision_margin)
                    if C<0.:
                        pos[jid]=pos[jid]-C*n
                        touch_count+=1
                for collider in policy.spheres:
                    diff=pos[jid]-collider.center
                    length=float(np.linalg.norm(diff))
                    radius=collider.radius+collider.margin+policy.collision_margin
                    if length<radius:
                        n=diff/length if length>1e-12 else np.array((1.,0.,0.))
                        pos[jid]=pos[jid]+(radius-length)*n
                        touch_count+=1
                for collider in policy.joint_spheres:
                    if jid in excluded_by_collider[collider.collider_id]:
                        continue
                    diff=pos[jid]-substep_joint_centers[collider.collider_id]
                    length=float(np.linalg.norm(diff))
                    radius=collider.radius+collider.margin+policy.collision_margin
                    if length<radius:
                        n=diff/length if length>1e-12 else np.array((1.,0.,0.))
                        pos[jid]=pos[jid]+(radius-length)*n
                        touch_count+=1
            if self_topology is not None:
                self_result=evaluate_surface_self_contacts_v1(
                    pos,mass,self_topology,
                    thickness=policy.self_collision_thickness,
                    cell_size=self_cell_size,
                    compliance=policy.self_collision_compliance,
                    step_seconds=h,
                    max_pairs=policy.max_self_collision_pairs,
                    max_cells=policy.max_self_collision_cells,
                    lambdas=lambda_self,project=True,
                )
                self_contact_projections+=self_result.projections
                self_contact_max_candidates=max(
                    self_contact_max_candidates,self_result.candidate_pairs,
                )
        for jid in dynamic:
            velocity[jid]=(pos[jid]-before[jid])/h
            if (not np.isfinite(pos[jid]).all()
                    or not np.isfinite(velocity[jid]).all()):
                raise QualificationError("XPBD_SURFACE_NONFINITE_STEP")
        ticks+=1

    max_stretch=max((
        abs(float(np.linalg.norm(pos[a]-pos[b]))-rest)/rest
        for a,b,rest in domain.stretch_edges
    ),default=0.)
    max_area=max((
        abs(_triangle_area(*(pos[v] for v in f))-A)/A
        for f,A in zip(domain.selected_faces,domain.triangle_rest_areas)
    ),default=0.)
    max_penetration=0.
    for jid in dynamic:
        for collider in policy.planes:
            max_penetration=max(max_penetration,max(0.,
                collider.margin+policy.collision_margin
                -float(np.dot(pos[jid]-collider.point,collider.normal))))
        for collider in policy.spheres:
            max_penetration=max(max_penetration,max(0.,
                collider.radius+collider.margin+policy.collision_margin
                -float(np.linalg.norm(pos[jid]-collider.center))))
        for collider in policy.joint_spheres:
            if jid in excluded_by_collider[collider.collider_id]:
                continue
            max_penetration=max(max_penetration,max(0.,
                collider.radius+collider.margin+policy.collision_margin
                -float(np.linalg.norm(pos[jid]-joint_centers[collider.collider_id]))))
    self_max_penetration=0.
    if self_topology is not None and steps>0:
        self_diagnostic=evaluate_surface_self_contacts_v1(
            pos,mass,self_topology,
            thickness=policy.self_collision_thickness,
            cell_size=self_cell_size,
            max_pairs=policy.max_self_collision_pairs,
            max_cells=policy.max_self_collision_cells,
            project=False,
        )
        self_max_penetration=self_diagnostic.maximum_penetration
        self_contact_max_candidates=max(
            self_contact_max_candidates,self_diagnostic.candidate_pairs,
        )
    state=XPBDSurfaceStateV1(
        domain_hash=domain.domain_hash,
        policy_hash=signature,
        primary_deformation_hash=primary_hash,
        particles=tuple(XPBDSurfaceParticleStateV1(
            jid,tuple(map(float,pos[jid])),tuple(map(float,velocity[jid]))
        ) for jid in dynamic),
        pinned_positions=tuple((jid,tuple(map(float,primary[jid]))) for jid in pins),
        ticks=ticks,frames=frames+int(steps>0),
        state_hash="",
        metadata={
            "authority":"EXPLICIT_XPBD_SURFACE_RUNTIME_STATE",
            "source_material_parameters_qualified":False,
            "training_authorized":False,
            "kinematic_joint_sphere_centers":[
                [cid,list(map(float,joint_centers[cid]))]
                for cid in sorted(joint_centers)
            ],
        },
    )
    state=replace(state,state_hash=_state_hash(state))
    new_positions=tuple(
        tuple(map(float,pos[jid])) if jid in pos
        else tuple(map(float,primary[jid]))
        for jid in ids
    )
    deformation=replace(
        primary_deformation,
        positions=new_positions,
        metadata={
            **primary_deformation.metadata,
            "secondary_surface":"XPBD_CANONICAL_VERTEX_DEFORMATION_V1",
            "xpbd_surface_domain_hash":domain.domain_hash,
            "canonical_face_topology_unchanged":True,
            "source_material_parameters_qualified":False,
        },
    )
    status=("INITIALIZED_XPBD_SURFACE_UNSIMULATED" if previous_state is None
            else "SNAPSHOT_XPBD_SURFACE_NO_STEP" if steps==0
            else "PASS_XPBD_SURFACE_RUNTIME" if (
                max_stretch<=policy.max_stretch_relative_error
                and max_area<=policy.max_area_relative_error
                and max_penetration<=policy.max_contact_penetration
                and self_max_penetration<=policy.max_contact_penetration)
            else "FAIL_XPBD_SURFACE_CONSTRAINT_RESIDUAL")
    return XPBDSurfaceResultV1(
        deformation=deformation,state=state,status=status,
        max_stretch_relative_error=max_stretch,
        max_area_relative_error=max_area,
        max_contact_penetration=max_penetration,
        contact_projections=touch_count,substeps=steps,
        metadata={
            "solver_family":"XPBD_COMPLIANT_SURFACE_PARTICLES",
            "source_physics_truth_claimed":False,
            "same_canonical_mesh_and_face_addresses":True,
            "surface_self_triangle_collision_qualified":bool(self_topology is not None and steps>0 and status=="PASS_XPBD_SURFACE_RUNTIME"),
            "self_collision_discrete_only":True,
            "self_contact_vertex_triangle_and_edge_edge_enabled":bool(self_topology),
            "self_contact_projections":self_contact_projections,
            "self_contact_max_candidate_pairs":self_contact_max_candidates,
            "self_contact_max_penetration":self_max_penetration,
            "explicit_joint_sphere_contact_enabled":bool(policy.joint_spheres),
            "friction_qualified":False,
            "dihedral_material_bending_qualified":False,
        },
    )
