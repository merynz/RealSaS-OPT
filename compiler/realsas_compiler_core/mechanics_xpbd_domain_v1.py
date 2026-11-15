from __future__ import annotations

"""Canonical-mesh-bound admission for *explicitly selected* XPBD secondary regions.

This module solves the historical XPBD's missing ownership problem, not the
numerical XPBD solver. It never converts part labels, static proximity or
semantic segmentation into source-observed motion, contact or stiffness.

The historical solver source is deliberately not on current main. Execution
must remain unavailable until its exact bytes, API and contact/constraint
parity are independently checked and promoted.
"""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from inspect import getsourcefile
from pathlib import Path
import math
from typing import Any

from .canonical_mesh_ir_v1 import CanonicalMeshIR, validate_canonical_mesh_domain_v1
from .canonical_rig_ir_v2 import CanonicalRigIRV2, validate_canonical_rig_domain_v2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2, validate_canonical_skin_domain_v2
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .hashing import content_sha256
from .solver_registry import SOLVER_AUTHORITY, resolve_promoted_solver
from .types import QualificationError


HISTORICAL_XPBD_SOURCE_SHA256 = (
    "a21982e7a9724e65ca89d375a20f79aae38967c1a663cb068d2dcd26d5ae7539"
)
XPBD_SECONDARY_DOMAIN_VERSION = "RealSaS.XPBD.SecondaryDomain.v1"


@dataclass(frozen=True)
class DeclaredXPBDSecondaryRequestV1:
    """Only explicit user/editor policy can enable this unlearned behavior."""

    canonical_chain_ids: tuple[str, ...]
    activation_authority: str = "EXPLICIT_RUNTIME_POLICY"
    skin_support_threshold: float = 0.10
    max_joint_count: int = 128
    max_surface_vertices: int = 100_000


@dataclass(frozen=True)
class XPBDChainDomainV1:
    canonical_chain_id: str
    pinned_root_joint_id: str
    dynamic_joint_ids: tuple[str, ...]
    segment_rest_lengths: tuple[float, ...]
    skinned_vertex_ids: tuple[str, ...]
    skinned_edge_ids: tuple[tuple[str, str], ...]
    skin_support_threshold: float


@dataclass(frozen=True)
class XPBDSecondaryDomainV1:
    mesh_lineage_hash: str
    rig_lineage_hash: str
    skin_lineage_hash: str
    mechanics_lineage_hash: str
    deform_structure_hash: str
    control_program_hash: str
    regions: tuple[XPBDChainDomainV1, ...]
    domain_hash: str
    schema_version: str = XPBD_SECONDARY_DOMAIN_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_declared_xpbd_secondary_domain_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    request: DeclaredXPBDSecondaryRequestV1,
) -> XPBDSecondaryDomainV1:
    """Build typed joint/surface ownership, not an inferred physics program."""

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
        raise QualificationError("XPBD_SECONDARY_CANONICAL_LINEAGE_DRIFT")
    if (
        type(request) is not DeclaredXPBDSecondaryRequestV1
        or request.activation_authority != "EXPLICIT_RUNTIME_POLICY"
        or not isinstance(request.canonical_chain_ids, tuple)
        or not request.canonical_chain_ids
        or len(request.canonical_chain_ids) > 32
        or len(set(request.canonical_chain_ids)) != len(request.canonical_chain_ids)
        or any(not isinstance(cid, str) or not cid for cid in request.canonical_chain_ids)
    ):
        raise QualificationError("XPBD_SECONDARY_EXPLICIT_SELECTION_REQUIRED")
    if (
        not isinstance(request.skin_support_threshold, (int, float))
        or not math.isfinite(request.skin_support_threshold)
        or not 0 < request.skin_support_threshold <= 1.
        or type(request.max_joint_count) is not int
        or not 2 <= request.max_joint_count <= 128
        or type(request.max_surface_vertices) is not int
        or not 1 <= request.max_surface_vertices <= 100_000
    ):
        raise QualificationError("XPBD_SECONDARY_RESOURCE_POLICY_INVALID")
    validate_canonical_mesh_domain_v1(mesh)
    validate_canonical_rig_domain_v2(rig)
    validate_canonical_skin_domain_v2(skin)
    if (
        mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH"
        or rig.metadata.get("authority") != "COMPILER_CANONICAL_RIG"
        or skin.metadata.get("authority") != "COMPILER_CANONICAL_SKIN"
        or mechanics.metadata.get("authority") != "COMPILER_CANONICAL_MECHANICS"
    ):
        raise QualificationError("XPBD_SECONDARY_CANONICAL_AUTHORITY_REQUIRED")
    if len(set(v.canonical_mesh_vertex_id for v in mesh.vertices)) != len(mesh.vertices):
        raise QualificationError("XPBD_SECONDARY_DUPLICATE_MESH_VERTEX")
    mesh_vertex_ids = {v.canonical_mesh_vertex_id for v in mesh.vertices}
    if (
        len(skin.rows) != len(mesh.vertices)
        or {r.canonical_mesh_vertex_id for r in skin.rows} != mesh_vertex_ids
    ):
        raise QualificationError("XPBD_SECONDARY_SKIN_VERTEX_COVERAGE_DRIFT")
    joints = {j.canonical_joint_id: j for j in rig.joints}
    chains = {c.canonical_chain_id: c for c in rig.chains}
    controls = {c.canonical_control_id: c for c in rig.controls}
    if (
        len(joints) != len(rig.joints) or len(chains) != len(rig.chains)
        or len(controls) != len(rig.controls)
    ):
        raise QualificationError("XPBD_SECONDARY_RIG_DUPLICATE_ADDRESS")
    owned: set[str] = set()
    declared: list[XPBDChainDomainV1] = []
    for chain_id in request.canonical_chain_ids:
        chain = chains.get(chain_id)
        if chain is None:
            raise QualificationError("XPBD_SECONDARY_CHAIN_UNDECLARED")
        ids = chain.canonical_joint_ids
        if (
            len(ids) < 2 or len(ids) > request.max_joint_count
            or len(set(ids)) != len(ids)
            or any(jid not in joints for jid in ids)
        ):
            raise QualificationError("XPBD_SECONDARY_CHAIN_DOMAIN_INVALID")
        for parent, child in zip(ids, ids[1:]):
            if joints[child].parent_canonical_joint_id != parent:
                raise QualificationError("XPBD_SECONDARY_CHAIN_PARENT_MISMATCH")
        dynamic = set(ids[1:])
        if dynamic & owned:
            raise QualificationError("XPBD_SECONDARY_JOINT_MULTIPLE_OWNERS")
        # Other primary branches would be left with stale FK if a secondary
        # joint were moved independently. Refuse until a coupled graph exists.
        for j in rig.joints:
            if j.parent_canonical_joint_id in dynamic and j.canonical_joint_id not in dynamic:
                raise QualificationError("XPBD_SECONDARY_PRIMARY_DESCENDANT_CONFLICT")
        # A secondary chain cannot overwrite an authored IK/control output,
        # declared constraint chain, or an attached prop's owner.
        for constraint in rig.constraints:
            if (
                constraint.canonical_chain_id == chain_id
                or dynamic.intersection(constraint.output_canonical_joint_ids)
            ):
                raise QualificationError("XPBD_SECONDARY_PRIMARY_CONSTRAINT_CONFLICT")
        for control in rig.controls:
            if dynamic.intersection(control.target_canonical_joint_ids):
                raise QualificationError("XPBD_SECONDARY_PRIMARY_CONTROL_CONFLICT")
        for anchor in rig.attachment_anchors:
            if anchor.parent_canonical_joint_id in dynamic:
                raise QualificationError("XPBD_SECONDARY_ATTACHMENT_OWNER_CONFLICT")
            if anchor.parent_canonical_control_id is not None:
                owner = controls.get(anchor.parent_canonical_control_id)
                if owner is None:
                    raise QualificationError("XPBD_SECONDARY_ANCHOR_CONTROL_UNKNOWN")
                if dynamic.intersection(owner.target_canonical_joint_ids):
                    raise QualificationError("XPBD_SECONDARY_ATTACHMENT_OWNER_CONFLICT")
        rest: list[float] = []
        for parent, child in zip(ids, ids[1:]):
            length = math.dist(joints[parent].P, joints[child].P)
            if not math.isfinite(length) or length <= 1e-8:
                raise QualificationError("XPBD_SECONDARY_REST_SEGMENT_INVALID")
            rest.append(length)
        influenced: list[str] = []
        for row in skin.rows:
            score = sum(
                float(weight) for jid, weight in row.influences if jid in dynamic
            )
            if not math.isfinite(score):
                raise QualificationError("XPBD_SECONDARY_SKIN_WEIGHT_NONFINITE")
            if score >= request.skin_support_threshold:
                influenced.append(row.canonical_mesh_vertex_id)
        support = set(influenced)
        if not support:
            raise QualificationError("XPBD_SECONDARY_NO_SKINNED_SURFACE")
        if len(support) > request.max_surface_vertices:
            raise QualificationError("XPBD_SECONDARY_SURFACE_BUDGET")
        edges = tuple(sorted({
            tuple(sorted((a, b))) for a, b in mesh.edges
            if a in support and b in support
        }))
        declared.append(XPBDChainDomainV1(
            canonical_chain_id=chain_id,
            pinned_root_joint_id=ids[0],
            dynamic_joint_ids=tuple(ids[1:]),
            segment_rest_lengths=tuple(rest),
            skinned_vertex_ids=tuple(sorted(support)),
            skinned_edge_ids=edges,
            skin_support_threshold=float(request.skin_support_threshold),
        ))
        owned.update(dynamic)
    result = XPBDSecondaryDomainV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        regions=tuple(declared),
        domain_hash="",
        metadata={
            "domain_authority": "EXPLICIT_RUNTIME_POLICY",
            "partverse_semantics_authorized_for_activation": False,
            "physical_parameters_source_qualified": False,
            "source_contact_qualified": False,
            "solver_executed": False,
            "historical_xpbd_source_sha256_expected": HISTORICAL_XPBD_SOURCE_SHA256,
        },
    )
    return XPBDSecondaryDomainV1(
        **{**result.to_dict(), "regions": result.regions,
           "domain_hash": content_sha256({
               key: value for key, value in result.to_dict().items()
               if key != "domain_hash"
           })}
    )


def require_promoted_xpbd_solver_v1(
    domain: XPBDSecondaryDomainV1, *, source_module_path: str | Path
):
    """Verify actual historical bytes AND the selected executable's file.

    A caller-provided SHA string is never accepted as a source byte witness.
    This is a promotion gate, not a general source-code loader or a new
    numerical implementation.
    """
    payload = domain.to_dict()
    content = {k: v for k, v in payload.items() if k != "domain_hash"}
    if (
        domain.schema_version != XPBD_SECONDARY_DOMAIN_VERSION
        or not domain.regions
        or domain.metadata.get("domain_authority") != "EXPLICIT_RUNTIME_POLICY"
        or domain.domain_hash != content_sha256(content)
    ):
        raise QualificationError("XPBD_SECONDARY_DOMAIN_UNQUALIFIED")
    try:
        path = Path(source_module_path).resolve(strict=True)
        raw = path.read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        raise QualificationError("XPBD_SECONDARY_HISTORICAL_SOURCE_UNAVAILABLE") from exc
    if sha256(raw).hexdigest() != HISTORICAL_XPBD_SOURCE_SHA256:
        raise QualificationError("XPBD_SECONDARY_HISTORICAL_SOURCE_HASH_MISMATCH")
    if (
        SOLVER_AUTHORITY["XPBD"].current_main_status
        != "CANONICAL_MAINLINE_EXECUTABLE"
    ):
        raise QualificationError("XPBD_SECONDARY_SOLVER_NOT_PROMOTED")
    solver = resolve_promoted_solver("XPBD")
    origin = getsourcefile(solver)
    if origin is None or Path(origin).resolve() != path:
        raise QualificationError("XPBD_SECONDARY_EXECUTABLE_SOURCE_DRIFT")
    return solver


__all__ = [
    "HISTORICAL_XPBD_SOURCE_SHA256",
    "DeclaredXPBDSecondaryRequestV1",
    "XPBDChainDomainV1",
    "XPBDSecondaryDomainV1",
    "prepare_declared_xpbd_secondary_domain_v1",
    "require_promoted_xpbd_solver_v1",
]
