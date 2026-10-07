from __future__ import annotations

"""Identity re-key of W_M from Stage19 carrier ids to exact Stage35 canonical ids.

This is not skin transfer. Stage35 is required to preserve the exact Stage19
candidate vertex basis. If positions/topology or source candidate identity drift,
this path fails and MIRA must run again on the new M.
"""

import math
from dataclasses import replace

from .hashing import content_sha256
from .product_authority_v1 import validate_qualified_mesh
from .types import QualifiedMeshSkinIR, QualifiedMeshSkinRow, QualificationError


def carrier_mesh_skin_lineage_hash(value:QualifiedMeshSkinIR)->str:
    payload=value.to_dict(); payload.pop("mesh_skin_lineage_hash",None)
    return content_sha256(payload)


def bind_carrier_native_mesh_skin_v1(
    *, carrier, skeleton, carrier_skin, mesh, surface, partition, carrier_policy, envelope, policy
):
    validate_qualified_mesh(
        mesh,surface=surface,partition=partition,carrier_policy=carrier_policy,envelope=envelope,policy=policy
    )
    if carrier_skin.carrier_evidence_hash!=carrier.carrier_evidence_hash or carrier_skin.carrier_topology_hash!=carrier.topology_hash or carrier_skin.carrier_geometry_hash!=carrier.geometry_hash:
        raise QualificationError("CARRIER_MESH_SKIN_CARRIER_BINDING_DRIFT")
    if carrier_skin.skeleton_binding_hash!=skeleton.skeleton_lineage_hash:
        raise QualificationError("CARRIER_MESH_SKIN_SKELETON_DRIFT")

    source_rows={str(r.carrier_vertex_id):r for r in carrier_skin.rows}
    if len(source_rows)!=len(carrier_skin.rows) or set(source_rows)!=set(map(str,carrier.ordered_vertex_ids)):
        raise QualificationError("CARRIER_MESH_SKIN_SOURCE_ROWS_DRIFT")

    mesh_by_source={str(v.source_candidate_vertex_id):v for v in mesh.vertices}
    if len(mesh_by_source)!=len(mesh.vertices):
        raise QualificationError("CARRIER_MESH_SKIN_SOURCE_CANDIDATE_ID_DUPLICATE")
    if set(mesh_by_source)!=set(map(str,carrier.ordered_vertex_ids)):
        raise QualificationError("CARRIER_MESH_SKIN_EXACT_VERTEX_BASIS_DRIFT")
    carrier_index={str(vid):i for i,vid in enumerate(carrier.ordered_vertex_ids)}
    for source_id,vertex in mesh_by_source.items():
        ci=carrier_index[source_id]
        if tuple(map(float,vertex.P))!=tuple(map(float,carrier.positions[ci])):
            raise QualificationError("CARRIER_MESH_SKIN_POSITION_DRIFT__RERUN_MIRA")

    rows=[]
    for source_id in map(str,carrier.ordered_vertex_ids):
        vertex=mesh_by_source[source_id]
        src=source_rows[source_id]
        total=sum(float(w) for _,w in src.influences)
        if abs(total-1.0)>1e-9:
            raise QualificationError("CARRIER_MESH_SKIN_SIMPLEX_DRIFT")
        rows.append(QualifiedMeshSkinRow(
            canonical_mesh_vertex_id=str(vertex.canonical_mesh_vertex_id),
            influences=tuple((str(j),float(w)) for j,w in src.influences),
            source_support_coefficients=((source_id,1.0),),
            simplex_residual_before=float(src.simplex_residual_before),
            correction_l1=float(src.correction_l1),
        ))
    report={
        "status":"PASS_CARRIER_NATIVE_IDENTITY_REKEY",
        "row_count":len(rows),
        "transfer_method":"CARRIER_IDENTITY_REKEY_ONLY_V1",
        "semantic_skin_transfer":False,
        "weight_interpolation":False,
        "nearest_vertex_matching":False,
        "carrier_evidence_hash":carrier.carrier_evidence_hash,
        "carrier_topology_hash":carrier.topology_hash,
        "carrier_geometry_hash":carrier.geometry_hash,
        "mesh_mutation_invalidates_skin":True,
    }
    value=QualifiedMeshSkinIR(
        rows=tuple(rows),
        surface_binding_hash=surface.geometry_lineage_hash,
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        skin_binding_hash=carrier_skin.skin_lineage_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        transfer_method="CARRIER_IDENTITY_REKEY_ONLY_V1",
        qualification_report=report,
        mesh_skin_lineage_hash="",
        metadata={
            "source_skin_schema":"RealSaS.QualifiedCarrierSkinIR.v1",
            "carrier_native":True,"second_mesh_truth_created":False,
            "semantic_skin_transfer_performed":False,
        },
    )
    value=replace(value,mesh_skin_lineage_hash=carrier_mesh_skin_lineage_hash(value))
    return value


def validate_carrier_native_mesh_skin_v1(value,*,carrier,skeleton,carrier_skin,mesh):
    if value.transfer_method!="CARRIER_IDENTITY_REKEY_ONLY_V1":
        raise QualificationError("CARRIER_MESH_SKIN_METHOD_DRIFT")
    if value.skin_binding_hash!=carrier_skin.skin_lineage_hash or value.skeleton_binding_hash!=skeleton.skeleton_lineage_hash or value.mesh_binding_hash!=mesh.mesh_lineage_hash:
        raise QualificationError("CARRIER_MESH_SKIN_LINEAGE_DRIFT")
    mesh_ids={str(v.canonical_mesh_vertex_id) for v in mesh.vertices}
    rows={str(r.canonical_mesh_vertex_id):r for r in value.rows}
    if set(rows)!=mesh_ids or len(rows)!=len(value.rows):
        raise QualificationError("CARRIER_MESH_SKIN_ROW_ACCOUNTING_DRIFT")
    for row in rows.values():
        if abs(sum(float(w) for _,w in row.influences)-1.0)>1e-9:
            raise QualificationError("CARRIER_MESH_SKIN_SIMPLEX_INVALID")
        if any((not math.isfinite(float(w)) or float(w)<0.0) for _,w in row.influences):
            raise QualificationError("CARRIER_MESH_SKIN_WEIGHT_INVALID")
    if value.mesh_skin_lineage_hash!=carrier_mesh_skin_lineage_hash(value):
        raise QualificationError("CARRIER_MESH_SKIN_HASH_DRIFT")

__all__=["bind_carrier_native_mesh_skin_v1","validate_carrier_native_mesh_skin_v1","carrier_mesh_skin_lineage_hash"]
