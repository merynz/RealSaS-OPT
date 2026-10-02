from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    normalization_domain_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import signed_zero_surface_from_dict
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse


def load(path: Path):
    return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--chunk-faces",type=int,default=250000)
    a=ap.parse_args()
    rr=a.run_root.expanduser().resolve()

    zero=signed_zero_surface_from_dict(load(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(load(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    surface=rigging_surface_from_dict(load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    prov=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/compacted_dense_face_provenance.json")

    # Fresh Stage15 nonmanifold census.
    edge_faces=defaultdict(list)
    compact_faces=[tuple(map(str,f)) for f in prov["compact_faces"]]
    for fi,face in enumerate(compact_faces):
        for i,j in ((0,1),(1,2),(2,0)):
            e=tuple(sorted((face[i],face[j])))
            edge_faces[e].append(fi)
    bad={e:rows for e,rows in edge_faces.items() if len(rows)>2}
    if not bad:
        raise RuntimeError("NO_STAGE15_NONMANIFOLD_EDGES")

    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],dtype=np.float64)
        df=np.asarray(z["faces"],dtype=np.int64)
    world=np.asarray(norm.center_xyz,dtype=np.float64)[None,:]+vn*float(norm.half_extent)

    md=dict(surface.metadata or {})
    divisions=int(md["compact_voxel_divisions"])
    component_aware=bool(md.get("component_aware_compaction",False))
    inv=compact_inverse(world,df,divisions=divisions,component_aware=component_aware)

    if int(inv.max())+1 != len(surface.surface_nodes):
        raise RuntimeError("COMPACT_INVERSE_NODE_COUNT_DRIFT")

    sid_to_compact={str(n.surface_id):i for i,n in enumerate(surface.surface_nodes)}
    if len(sid_to_compact)!=len(surface.surface_nodes):
        raise RuntimeError("SURFACE_ID_DUPLICATE")

    ncomp=len(surface.surface_nodes)
    ndense=len(world)
    def ckey(a0,b0):
        x,y=(a0,b0) if a0<b0 else (b0,a0)
        return int(x)*ncomp+int(y)
    def dkey(a0,b0):
        x,y=(a0,b0) if a0<b0 else (b0,a0)
        return int(x)*ndense+int(y)

    bad_code_to_edge={}
    bad_codes=[]
    stage15_incidence={}
    for e,rows in bad.items():
        x=sid_to_compact[e[0]]; y=sid_to_compact[e[1]]
        code=ckey(x,y)
        bad_codes.append(code)
        bad_code_to_edge[code]=e
        stage15_incidence[code]=len(rows)
    bad_codes_arr=np.asarray(sorted(bad_codes),dtype=np.int64)

    # Count original dense-edge incidence only where the dense edge maps onto one
    # of the Stage15 nonmanifold compact edges, and only for noncollapsed mapped faces.
    orig_counter=Counter()
    mapped_face_sets=defaultdict(set)
    matched_dense_face_count=Counter()
    chunk=max(10000,int(a.chunk_faces))
    for start in range(0,len(df),chunk):
        raw=np.asarray(df[start:start+chunk],dtype=np.int64)
        mapped=inv[raw]
        valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
        if not np.any(valid):
            continue
        raw=raw[valid]
        mapped=mapped[valid]
        mapped_sorted=np.sort(mapped,axis=1)
        mapped_face_code=(
            mapped_sorted[:,0].astype(object)*ncomp*ncomp
            +mapped_sorted[:,1].astype(object)*ncomp
            +mapped_sorted[:,2].astype(object)
        )
        for ia,ib in ((0,1),(1,2),(2,0)):
            ca=np.minimum(mapped[:,ia],mapped[:,ib]).astype(np.int64)
            cb=np.maximum(mapped[:,ia],mapped[:,ib]).astype(np.int64)
            codes=ca*ncomp+cb
            take=np.isin(codes,bad_codes_arr,assume_unique=False)
            idx=np.nonzero(take)[0]
            for k in idx.tolist():
                code=int(codes[k])
                oa=int(raw[k,ia]); ob=int(raw[k,ib])
                orig_counter[(code,dkey(oa,ob))]+=1
                mapped_face_sets[code].add(int(mapped_face_code[k]))
                matched_dense_face_count[code]+=1

    rows=[]
    source_dense_nonmanifold=0
    compaction_created=0
    mixed=0
    unresolved=0
    for code in sorted(bad_codes):
        orig=[(dk,c) for (cc,dk),c in orig_counter.items() if cc==code]
        max_inc=max((c for _dk,c in orig),default=0)
        dense_bad=sum(1 for _dk,c in orig if c>2)
        distinct_orig=len(orig)
        distinct_mapped=len(mapped_face_sets.get(code,set()))
        if dense_bad>0:
            cls="DENSE_SOURCE_NONMANIFOLD_PRESENT"
            source_dense_nonmanifold+=1
            if distinct_orig>1:
                mixed+=1
        elif distinct_orig>1 and distinct_mapped>2:
            cls="COMPACTION_COLLAPSE_CREATED_NONMANIFOLD"
            compaction_created+=1
        else:
            cls="UNRESOLVED"
            unresolved+=1
        rows.append({
            "compact_edge_surface_ids":list(bad_code_to_edge[code]),
            "stage15_unique_face_incidence":int(stage15_incidence[code]),
            "matched_dense_face_edge_occurrences":int(matched_dense_face_count.get(code,0)),
            "distinct_original_dense_edges_collapsed_here":int(distinct_orig),
            "max_original_dense_edge_face_incidence":int(max_inc),
            "original_dense_nonmanifold_edge_count":int(dense_bad),
            "distinct_noncollapsed_mapped_faces_observed":int(distinct_mapped),
            "classification":cls,
        })

    cls_counts=Counter(r["classification"] for r in rows)
    report={
        "schema":"RealSaS.KnightCompactionNonmanifoldCausality.v1",
        "status":"MEASURED",
        "run_root":str(rr),
        "source_dense_vertex_count":int(len(world)),
        "source_dense_face_count":int(len(df)),
        "compact_node_count":int(ncomp),
        "compact_face_count":int(len(compact_faces)),
        "compact_voxel_divisions":int(divisions),
        "component_aware_compaction":component_aware,
        "stage15_nonmanifold_edge_count":int(len(bad)),
        "classification_counts":dict(cls_counts),
        "dense_source_nonmanifold_compact_edge_count":int(source_dense_nonmanifold),
        "compaction_created_nonmanifold_edge_count":int(compaction_created),
        "mixed_dense_nonmanifold_and_collapse_edge_count":int(mixed),
        "unresolved_edge_count":int(unresolved),
        "rows":rows,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_COMPACTION_NONMANIFOLD_CAUSALITY="+json.dumps({
        "stage15_nonmanifold_edge_count":len(bad),
        "classification_counts":dict(cls_counts),
        "dense_source_nonmanifold_compact_edge_count":source_dense_nonmanifold,
        "compaction_created_nonmanifold_edge_count":compaction_created,
        "mixed_dense_nonmanifold_and_collapse_edge_count":mixed,
        "unresolved_edge_count":unresolved,
        "sample":rows[:20],
    },sort_keys=True))


if __name__=="__main__":
    main()
