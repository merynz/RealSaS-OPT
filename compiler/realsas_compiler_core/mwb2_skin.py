from __future__ import annotations

from .hashing import content_sha256
from .mesh_binding import validate_qualified_mesh, validate_qualified_mesh_skin
from .types import QualifiedMeshSkinIR, QualifiedMeshSkinRow, QualificationError


def bind_mwb2_mesh_skin(surface, skeleton, skin, mesh) -> QualifiedMeshSkinIR:
    validate_qualified_mesh(mesh, surface)
    if skin.surface_binding_hash != surface.geometry_lineage_hash: raise QualificationError("MESH_WEIGHT_SKIN_LINEAGE_MISMATCH: surface")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash: raise QualificationError("MESH_WEIGHT_SKIN_LINEAGE_MISMATCH: skeleton")
    source = {r.surface_id: r for r in skin.rows}; rows=[]
    for vertex in sorted(mesh.vertices, key=lambda v: v.canonical_mesh_vertex_id):
        accum: dict[str,float] = {}; provenance=[]
        for sid, coeff in vertex.support_binding.coefficients:
            if sid not in source: raise QualificationError(f"MESH_WEIGHT_UNSUPPORTED_SURFACE:{sid}")
            c=float(coeff); provenance.append((sid,c))
            for jid,w in source[sid].influences: accum[jid]=accum.get(jid,0.0)+c*float(w)
        total=sum(accum.values())
        if total <= 0.0: raise QualificationError("MESH_WEIGHT_ZERO_MASS")
        normalized=tuple(sorted((jid,w/total) for jid,w in accum.items() if w>0.0)); correction=abs(total-1.0)
        rows.append(QualifiedMeshSkinRow(vertex.canonical_mesh_vertex_id,normalized,tuple(provenance),correction,correction))
    report={"status":"PASS_MWB2_CONVEX_SKIN_TRANSFER","row_count":len(rows),"transfer_method":"SURFACE_SUPPORT_CONVEX_TRANSFER_V1","semantic_skin_synthesis":False}
    value=QualifiedMeshSkinIR(tuple(rows),surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,skin.skin_lineage_hash,mesh.mesh_lineage_hash,"SURFACE_SUPPORT_CONVEX_TRANSFER_V1",report,"",metadata={"source_mesh_used":False})
    payload=value.to_dict(); payload.pop("mesh_skin_lineage_hash",None)
    value=QualifiedMeshSkinIR(**{**value.__dict__,"mesh_skin_lineage_hash":content_sha256(payload)})
    validate_qualified_mesh_skin(value,surface=surface,skeleton=skeleton,skin=skin,mesh=mesh)
    return value
