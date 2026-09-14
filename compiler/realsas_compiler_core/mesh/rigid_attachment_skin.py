from __future__ import annotations

"""Compile an explicit rigid-bone attachment into a runtime mesh-skin carrier.

This is not learned skin truth and does not consume source skin rows.  A previously
qualified rigid attachment supplies the parent joint and bind authority; the Compiler
emits an exact one-hot carrier only so the existing directional deformation/runtime
machinery can move the foreground mesh as a rigid child of that joint.
"""

from dataclasses import replace
from math import isfinite

from ..mesh_binding import mesh_lineage_hash, mesh_skin_lineage_hash
from ..types import QualifiedMeshSkinIR, QualifiedMeshSkinRow, QualificationError


RIGID_ATTACHMENT_TRANSFER_METHOD = "RIGID_BONE_ATTACHMENT_RUNTIME_CARRY_V1"


def _required_hash(name: str, value: str) -> str:
    value = str(value or "").strip()
    if not value or value.lower() in {"latest", "current", "newest"}:
        raise QualificationError(f"RIGID_ATTACHMENT_SKIN_INVALID_{name}")
    return value


def qualify_rigid_attachment_mesh_skin(
    mesh,
    mechanical,
    *,
    parent_joint_id: str,
    component_assembly_hash: str,
    component_lineage_hash: str,
    bind_state_authority_hash: str,
) -> QualifiedMeshSkinIR:
    if mesh.mesh_lineage_hash != mesh_lineage_hash(mesh):
        raise QualificationError("RIGID_ATTACHMENT_SKIN_MESH_LINEAGE_DRIFT")
    parent_joint_id = str(parent_joint_id)
    legal = {str(j.canonical_joint_id) for j in mechanical.skeleton.joints}
    if parent_joint_id not in legal:
        raise QualificationError("RIGID_ATTACHMENT_SKIN_PARENT_NOT_CANONICAL")
    assembly_hash = _required_hash("COMPONENT_ASSEMBLY_HASH", component_assembly_hash)
    component_hash = _required_hash("COMPONENT_LINEAGE_HASH", component_lineage_hash)
    bind_hash = _required_hash("BIND_STATE_AUTHORITY_HASH", bind_state_authority_hash)

    rows = []
    seen = set()
    for vertex in mesh.vertices:
        vid = str(vertex.canonical_mesh_vertex_id)
        if not vid or vid in seen:
            raise QualificationError("RIGID_ATTACHMENT_SKIN_DUPLICATE_VERTEX")
        seen.add(vid)
        coeffs = tuple((str(sid), float(weight)) for sid, weight in vertex.support_binding.coefficients)
        if not coeffs or any((not sid) or (not isfinite(weight)) or weight < 0.0 for sid, weight in coeffs):
            raise QualificationError("RIGID_ATTACHMENT_SKIN_INVALID_SUPPORT")
        if abs(sum(weight for _sid, weight in coeffs) - 1.0) > 1e-8:
            raise QualificationError("RIGID_ATTACHMENT_SKIN_SUPPORT_SIMPLEX")
        rows.append(QualifiedMeshSkinRow(vid, ((parent_joint_id, 1.0),), coeffs, 0.0, 0.0))
    if not rows:
        raise QualificationError("RIGID_ATTACHMENT_SKIN_EMPTY_MESH")

    report = {
        "status": "PASS_RIGID_BONE_ATTACHMENT_RUNTIME_CARRY",
        "transfer_method": RIGID_ATTACHMENT_TRANSFER_METHOD,
        "runtime_representation_only": True,
        "attachment_truth_source": "QUALIFIED_COMPONENT_ATTACHMENT",
        "parent_joint_id": parent_joint_id,
        "component_assembly_hash": assembly_hash,
        "component_lineage_hash": component_hash,
        "bind_state_authority_hash": bind_hash,
        "row_count": len(rows),
        "source_skin_rows_consumed": False,
        "direct_model_query": False,
        "historical_weight_transfer_used": False,
        "one_hot_carry_compiled_from_attachment": True,
    }
    value = QualifiedMeshSkinIR(
        tuple(rows),
        str(mesh.surface_binding_hash),
        str(mechanical.skeleton.skeleton_lineage_hash),
        str(mechanical.skin.skin_lineage_hash),
        str(mesh.mesh_lineage_hash),
        RIGID_ATTACHMENT_TRANSFER_METHOD,
        report,
        "",
        metadata={
            "runtime_representation_only": True,
            "attachment_truth_source": "QUALIFIED_COMPONENT_ATTACHMENT",
            "parent_joint_id": parent_joint_id,
            "component_assembly_hash": assembly_hash,
            "component_lineage_hash": component_hash,
            "bind_state_authority_hash": bind_hash,
            "source_skin_rows_consumed": False,
            "direct_model_query": False,
            "historical_weight_transfer_used": False,
            "one_hot_carry_compiled_from_attachment": True,
        },
    )
    return replace(value, mesh_skin_lineage_hash=mesh_skin_lineage_hash(value))


__all__ = ["RIGID_ATTACHMENT_TRANSFER_METHOD", "qualify_rigid_attachment_mesh_skin"]
