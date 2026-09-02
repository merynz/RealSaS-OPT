from __future__ import annotations
from collections import defaultdict
from math import sqrt
from .types import ObservationEvidenceIR, PersistenceGroup, RiggingSurfaceIR, SurfaceNode, QualificationError
from .hashing import content_sha256

def _unit(v):
    n=sqrt(sum(float(x)*float(x) for x in v))
    if n<=1e-12: raise QualificationError("zero camera forward vector")
    return tuple(float(x)/n for x in v)

def _point(sample):
    f=_unit(sample.ray_forward)
    return tuple(float(sample.ray_origin[i])+float(sample.depth)*f[i] for i in range(3))

def build_surface_from_persistence(evidence:ObservationEvidenceIR, groups:tuple[PersistenceGroup,...], *, derive_normals:bool=False)->RiggingSurfaceIR:
    """Compile ray/depth evidence into P and fuse only explicitly supported members.

    support=False observations may be present in evidence or a diagnostic persistence
    group, but they are forbidden from contributing to fused P, raster bindings,
    support/provenance, or SurfaceNode source-observation lineage.
    """
    if derive_normals:
        raise QualificationError("normal derivation belongs to a qualified deterministic geometry operator, not the evidence head")
    by_id={s.observation_id:s for s in evidence.samples}
    if len(by_id)!=len(evidence.samples): raise QualificationError("duplicate observation_id")
    used=set(); nodes=[]
    for g in sorted(groups,key=lambda x:x.group_id):
        if not g.observation_ids: raise QualificationError(f"empty persistence group:{g.group_id}")
        samples=[]
        for oid in g.observation_ids:
            if oid in used: raise QualificationError(f"observation appears in multiple persistence groups:{oid}")
            if oid not in by_id: raise QualificationError(f"persistence references missing observation:{oid}")
            used.add(oid); samples.append(by_id[oid])
        admitted=[s for s in samples if bool(s.support)]
        if not admitted:
            raise QualificationError(f"persistence group has no supported observations:{g.group_id}")
        P=[_point(s) for s in admitted]
        mean=tuple(sum(p[i] for p in P)/len(P) for i in range(3))
        admitted_ids=tuple(sorted(s.observation_id for s in admitted))
        excluded_ids=tuple(sorted(s.observation_id for s in samples if not s.support))
        sid="S:"+content_sha256({"group":g.group_id,"obs":admitted_ids,"P":mean})[:20]
        nodes.append(SurfaceNode(
            surface_id=sid,P=mean,
            support_views=tuple(sorted({int(s.view_index) for s in admitted})),
            provenance_refs=tuple(sorted({s.provenance_ref for s in admitted if s.provenance_ref})),
            source_observation_ids=admitted_ids,
            raster_bindings=tuple(sorted((int(s.view_index),tuple(map(float,s.raster_xy))) for s in admitted)),
            persistence_group_id=g.group_id,
            validity_flags=tuple(sorted({flag for s in admitted for flag in s.validity_flags})),
            metadata={
                "persistence_method":g.method,
                "persistence_diagnostics":g.diagnostics,
                "support_false_excluded_observation_ids":excluded_ids,
                "support_admission_policy":"SUPPORT_TRUE_ONLY_V1",
            },
        ))
    lineage=content_sha256({
        "schema":"RealSaS.RiggingSurfaceIR.v1",
        "support_admission_policy":"SUPPORT_TRUE_ONLY_V1",
        "evidence":evidence.to_dict(),
        "groups":[g.to_dict() for g in groups],
        "nodes":[n.to_dict() for n in nodes],
    })
    return RiggingSurfaceIR(tuple(nodes), geometry_lineage_hash=lineage, metadata={
        "P_authority":"ANALYTIC_FROM_DEPTH_AND_KNOWN_RAY",
        "N_required":False,
        "support_admission_policy":"SUPPORT_TRUE_ONLY_V1",
    })

def rigging_surface_from_d2_arrays(P, support, raster_xy, *, authority_label:str, persistence_label:str="MUTUAL_P003")->RiggingSurfaceIR:
    """Narrow adapter for sealed E0/D2-style 512-anchor carriers."""
    import numpy as np
    P=np.asarray(P,float); support=np.asarray(support); raster_xy=np.asarray(raster_xy,float)
    if P.ndim!=2 or P.shape[1]!=3: raise QualificationError("P must be [N,3]")
    if support.shape!=(len(P),8): raise QualificationError("support must be [N,8]")
    if raster_xy.shape not in {(len(P),16),(len(P),8,2)}: raise QualificationError("raster_xy must be [N,16] or [N,8,2]")
    r=raster_xy.reshape(len(P),8,2)
    nodes=[]
    for i,p in enumerate(P):
        views=tuple(int(v) for v in range(8) if bool(support[i,v]))
        binds=tuple((v,(float(r[i,v,0]),float(r[i,v,1]))) for v in views)
        sid=f"S:{i:04d}:{content_sha256({'p':p.tolist(),'views':views,'authority':authority_label})[:12]}"
        nodes.append(SurfaceNode(sid,tuple(map(float,p)),views,(authority_label,),(),binds,f"{persistence_label}:{i:04d}"))
    lineage=content_sha256({"authority":authority_label,"persistence":persistence_label,"nodes":[n.to_dict() for n in nodes]})
    return RiggingSurfaceIR(tuple(nodes),geometry_lineage_hash=lineage,metadata={"adapter":"SEALED_D2_TYPED_ADAPTER","N_required":False})
