from __future__ import annotations

"""Deterministic QualifiedMeshIR <- QualifiedSkinIR transfer for product geometry."""

import math

from .hashing import content_sha256
from .product_authority_v1 import validate_qualified_mesh
from .types import QualifiedMeshSkinIR, QualifiedMeshSkinRow, QualificationError

_TRANSFER_EPS=1e-9


def product_mesh_skin_lineage_hash(value:QualifiedMeshSkinIR)->str:
    payload=value.to_dict()
    payload.pop("mesh_skin_lineage_hash",None)
    return content_sha256(payload)


def validate_product_mesh_skin(
    value:QualifiedMeshSkinIR,
    *,
    surface,
    skeleton,
    skin,
    mesh,
    partition,
    carrier_policy,
    envelope,
    policy,
)->None:
    validate_qualified_mesh(
        mesh,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
    )
    if value.surface_binding_hash!=surface.geometry_lineage_hash:
        raise QualificationError("PRODUCT_MESH_SKIN_SURFACE_LINEAGE_MISMATCH")
    if value.skeleton_binding_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("PRODUCT_MESH_SKIN_SKELETON_LINEAGE_MISMATCH")
    if value.skin_binding_hash!=skin.skin_lineage_hash:
        raise QualificationError("PRODUCT_MESH_SKIN_SKIN_LINEAGE_MISMATCH")
    if value.mesh_binding_hash!=mesh.mesh_lineage_hash:
        raise QualificationError("PRODUCT_MESH_SKIN_MESH_LINEAGE_MISMATCH")
    if value.transfer_method!="SURFACE_SUPPORT_CONVEX_TRANSFER_V1":
        raise QualificationError("PRODUCT_MESH_SKIN_TRANSFER_METHOD_DRIFT")

    joint_ids={j.canonical_joint_id for j in skeleton.joints}
    mesh_by_id={v.canonical_mesh_vertex_id:v for v in mesh.vertices}
    rows={}
    for row in value.rows:
        vid=row.canonical_mesh_vertex_id
        if vid in rows or vid not in mesh_by_id:
            raise QualificationError("PRODUCT_MESH_SKIN_ROW_ID_INVALID")
        rows[vid]=row
        if tuple(row.source_support_coefficients)!=tuple(mesh_by_id[vid].support_binding.coefficients):
            raise QualificationError("PRODUCT_MESH_SKIN_SUPPORT_PROVENANCE_DRIFT")
        total=0.0
        seen=set()
        for jid,weight in row.influences:
            if jid in seen or jid not in joint_ids:
                raise QualificationError("PRODUCT_MESH_SKIN_JOINT_INVALID")
            seen.add(jid)
            w=float(weight)
            if not math.isfinite(w) or w<0.0:
                raise QualificationError("PRODUCT_MESH_SKIN_WEIGHT_INVALID")
            total+=w
        if abs(total-1.0)>_TRANSFER_EPS:
            raise QualificationError("PRODUCT_MESH_SKIN_SIMPLEX_INVALID")
        if not math.isfinite(float(row.simplex_residual_before)) or row.simplex_residual_before<0.0:
            raise QualificationError("PRODUCT_MESH_SKIN_RESIDUAL_INVALID")
        if not math.isfinite(float(row.correction_l1)) or row.correction_l1<0.0:
            raise QualificationError("PRODUCT_MESH_SKIN_CORRECTION_INVALID")
    if set(rows)!=set(mesh_by_id):
        raise QualificationError("PRODUCT_MESH_SKIN_INCOMPLETE_VERTEX_ACCOUNTING")
    if value.mesh_skin_lineage_hash!=product_mesh_skin_lineage_hash(value):
        raise QualificationError("PRODUCT_MESH_SKIN_LINEAGE_HASH_MISMATCH")


def bind_product_mesh_skin(
    *,
    surface,
    skeleton,
    skin,
    mesh,
    partition,
    carrier_policy,
    envelope,
    policy,
    max_transfer_repair_l1:float=_TRANSFER_EPS,
    max_total_transfer_correction_l1:float=_TRANSFER_EPS,
)->QualifiedMeshSkinIR:
    if max_transfer_repair_l1<0.0 or max_total_transfer_correction_l1<0.0:
        raise ValueError("product mesh-skin correction budgets must be nonnegative")
    validate_qualified_mesh(
        mesh,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
    )
    if skin.surface_binding_hash!=surface.geometry_lineage_hash:
        raise QualificationError("PRODUCT_MESH_SKIN_SOURCE_SURFACE_LINEAGE_MISMATCH")
    if skin.skeleton_binding_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("PRODUCT_MESH_SKIN_SOURCE_SKELETON_LINEAGE_MISMATCH")

    source={row.surface_id:row for row in skin.rows}
    if set(source)!={node.surface_id for node in surface.surface_nodes}:
        raise QualificationError("PRODUCT_MESH_SKIN_SOURCE_ROWS_INCOMPLETE")
    joint_ids={j.canonical_joint_id for j in skeleton.joints}

    rows=[]
    total_correction=0.0
    max_residual=0.0
    for vertex in sorted(mesh.vertices,key=lambda x:x.canonical_mesh_vertex_id):
        accum={}
        provenance=[]
        support_total=0.0
        for sid,coeff in vertex.support_binding.coefficients:
            if sid not in source:
                raise QualificationError(f"PRODUCT_MESH_SKIN_UNSUPPORTED_SURFACE:{sid}")
            c=float(coeff)
            if not math.isfinite(c) or c<0.0:
                raise QualificationError("PRODUCT_MESH_SKIN_SUPPORT_COEFFICIENT_INVALID")
            support_total+=c
            provenance.append((sid,c))
            for jid,weight in source[sid].influences:
                if jid not in joint_ids:
                    raise QualificationError("PRODUCT_MESH_SKIN_SOURCE_UNKNOWN_JOINT")
                w=float(weight)
                if not math.isfinite(w) or w<0.0:
                    raise QualificationError("PRODUCT_MESH_SKIN_SOURCE_WEIGHT_INVALID")
                accum[jid]=accum.get(jid,0.0)+c*w
        if abs(support_total-1.0)>_TRANSFER_EPS:
            raise QualificationError("PRODUCT_MESH_SKIN_SUPPORT_SIMPLEX_INVALID")
        total=sum(accum.values())
        if total<=0.0:
            raise QualificationError("PRODUCT_MESH_SKIN_ZERO_MASS")
        residual=abs(total-1.0)
        if residual>max_transfer_repair_l1+1e-15:
            raise QualificationError(f"PRODUCT_MESH_SKIN_REPAIR_BUDGET_EXCEEDED:{residual}")
        normalized=tuple(sorted((jid,w/total) for jid,w in accum.items() if w>0.0))
        final=dict(normalized)
        correction=sum(abs(final.get(jid,0.0)-accum.get(jid,0.0)) for jid in set(accum)|set(final))
        if correction>max_transfer_repair_l1+1e-15:
            raise QualificationError(f"PRODUCT_MESH_SKIN_ROW_CORRECTION_BUDGET_EXCEEDED:{correction}")
        total_correction+=correction
        if total_correction>max_total_transfer_correction_l1+1e-15:
            raise QualificationError(f"PRODUCT_MESH_SKIN_TOTAL_CORRECTION_BUDGET_EXCEEDED:{total_correction}")
        max_residual=max(max_residual,residual)
        rows.append(QualifiedMeshSkinRow(
            vertex.canonical_mesh_vertex_id,
            normalized,
            tuple(provenance),
            residual,
            correction,
        ))

    report={
        "status":"PASS_PRODUCT_MESH_CONVEX_SKIN_TRANSFER",
        "row_count":len(rows),
        "transfer_method":"SURFACE_SUPPORT_CONVEX_TRANSFER_V1",
        "semantic_skin_synthesis":False,
        "max_simplex_residual_before":max_residual,
        "total_correction_l1":total_correction,
        "bounded_transfer_repair_l1_per_row":max_transfer_repair_l1,
        "bounded_transfer_correction_l1_total":max_total_transfer_correction_l1,
        "qualified_mesh_is_exact_source":True,
    }
    value=QualifiedMeshSkinIR(
        tuple(rows),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        skin.skin_lineage_hash,
        mesh.mesh_lineage_hash,
        "SURFACE_SUPPORT_CONVEX_TRANSFER_V1",
        report,
        "",
        metadata={"source_mesh_used":False,"second_mesh_truth_created":False},
    )
    value=QualifiedMeshSkinIR(**{
        **value.__dict__,
        "mesh_skin_lineage_hash":product_mesh_skin_lineage_hash(value),
    })
    validate_product_mesh_skin(
        value,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        mesh=mesh,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
    )
    return value
