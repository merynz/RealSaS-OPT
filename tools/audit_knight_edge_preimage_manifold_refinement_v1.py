from __future__ import annotations

import argparse, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,
    signed_zero_surface_from_dict,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse
from tools.audit_knight_topology_local_voxel_compaction_v1 import (
    topology_local_refine,
    compact_metrics,
)


def loadj(p:Path): return json.loads(p.read_text())


def bad_edge_codes(faces: np.ndarray, inv: np.ndarray) -> tuple[np.ndarray, dict[int,int], int]:
    mapped=inv[faces]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    canon=np.sort(mapped[valid],axis=1)
    uniq=np.unique(canon,axis=0)
    n=int(inv.max())+1
    edge=np.concatenate((uniq[:,[0,1]],uniq[:,[1,2]],uniq[:,[2,0]]),axis=0)
    edge=np.sort(edge,axis=1)
    code=edge[:,0].astype(np.int64)*np.int64(n)+edge[:,1].astype(np.int64)
    uk,cnt=np.unique(code,return_counts=True)
    bad=cnt>2
    return uk[bad],{int(k):int(v) for k,v in zip(uk[bad],cnt[bad])},n


def collect_bad_preimage_edges(
    faces: np.ndarray,
    inv: np.ndarray,
    bad_codes: np.ndarray,
    nclusters: int,
    *,
    chunk_faces: int=300_000,
) -> dict[int,list[tuple[int,int]]]:
    out=defaultdict(list)
    f=np.asarray(faces,dtype=np.int64)
    for start in range(0,len(f),chunk_faces):
        raw=f[start:start+chunk_faces]
        mapped=inv[raw]
        valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
        if not np.any(valid): continue
        raw=raw[valid]; mapped=mapped[valid]
        for ia,ib in ((0,1),(1,2),(2,0)):
            ca=np.minimum(mapped[:,ia],mapped[:,ib]).astype(np.int64)
            cb=np.maximum(mapped[:,ia],mapped[:,ib]).astype(np.int64)
            codes=ca*np.int64(nclusters)+cb
            take=np.isin(codes,bad_codes,assume_unique=False)
            for k in np.nonzero(take)[0].tolist():
                code=int(codes[k])
                u,v=map(int,(raw[k,ia],raw[k,ib]))
                # Preserve cluster side orientation in tuple: endpoint in lower
                # quotient cluster first, upper quotient cluster second.
                if int(mapped[k,ia])<=int(mapped[k,ib]):
                    out[code].append((u,v))
                else:
                    out[code].append((v,u))
    return out


def split_by_edge_preimage_components(
    faces: np.ndarray,
    inv: np.ndarray,
) -> tuple[np.ndarray,dict]:
    bad_codes,bad_inc,n=bad_edge_codes(faces,inv)
    if len(bad_codes)==0:
        return inv.copy(),{
            "bad_edge_count_before":0,
            "disconnected_bad_edge_count":0,
            "tokenized_vertex_count":0,
            "signature_count":0,
        }
    pre=collect_bad_preimage_edges(faces,inv,bad_codes,n)
    tokens=defaultdict(set)
    disconnected=0

    for code in map(int,bad_codes.tolist()):
        pairs=pre.get(code,[])
        if not pairs: continue
        # Deduplicate original dense edges.
        pairs=sorted(set((int(u),int(v)) for u,v in pairs))
        graph=defaultdict(set)
        for u,v in pairs:
            graph[("L",u)].add(("R",v))
            graph[("R",v)].add(("L",u))
        seen=set(); comps=[]
        for node in sorted(graph,key=lambda x:(x[0],x[1])):
            if node in seen: continue
            stack=[node]; seen.add(node); members=[]
            while stack:
                x=stack.pop(); members.append(x)
                for y in graph[x]:
                    if y not in seen:
                        seen.add(y); stack.append(y)
            comps.append(members)
        if len(comps)<=1:
            continue
        disconnected+=1
        # Component token is deterministic from minimum dense vertex on each side.
        comps.sort(key=lambda mem:min((x[1] for x in mem),default=-1))
        for ci,mem in enumerate(comps):
            tok=(code,ci)
            for side,vid in mem:
                tokens[int(vid)].add(tok)

    if not tokens:
        return inv.copy(),{
            "bad_edge_count_before":int(len(bad_codes)),
            "disconnected_bad_edge_count":int(disconnected),
            "tokenized_vertex_count":0,
            "signature_count":0,
        }

    # Refine each existing quotient vertex by exact incident bad-edge component
    # signature. Uninvolved vertices remain one additional signature class.
    sig_ids={}
    rows=[]
    for vid in range(len(inv)):
        sig=tuple(sorted(tokens.get(vid,())))
        key=(int(inv[vid]),sig)
        sid=sig_ids.setdefault(key,len(sig_ids))
        rows.append(sid)
    refined=np.asarray(rows,dtype=np.int64)

    # Re-apply dense-edge connectivity refinement inside the new classes.
    refined,_=topology_local_refine(faces,refined)
    return refined,{
        "bad_edge_count_before":int(len(bad_codes)),
        "disconnected_bad_edge_count":int(disconnected),
        "tokenized_vertex_count":int(len(tokens)),
        "signature_count":int(len(sig_ids)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--iterations",type=int,default=6)
    a=ap.parse_args()
    rr=a.run_root.expanduser().resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    surface=rigging_surface_from_dict(loadj(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],dtype=np.float64)
        faces=np.asarray(z["faces"],dtype=np.int64)
    world=np.asarray(norm.center_xyz,dtype=np.float64)[None,:]+vn*float(norm.half_extent)
    md=dict(surface.metadata or {})
    base=compact_inverse(world,faces,divisions=int(md["compact_voxel_divisions"]),component_aware=bool(md.get("component_aware_compaction",False)))

    inv,first=topology_local_refine(faces,base)
    history=[{
        "iteration":0,
        "phase":"TOPOLOGY_LOCAL_CONNECTIVITY",
        "metrics":compact_metrics(world,faces,inv),
        "refinement":first,
    }]

    for it in range(1,int(a.iterations)+1):
        before=compact_metrics(world,faces,inv)
        if before["nonmanifold_edge_count"]==0: break
        nxt,meta=split_by_edge_preimage_components(faces,inv)
        after=compact_metrics(world,faces,nxt)
        history.append({
            "iteration":it,
            "phase":"EDGE_PREIMAGE_COMPONENT_SPLIT",
            "meta":meta,
            "metrics":after,
        })
        inv=nxt
        if after["nonmanifold_edge_count"]>=before["nonmanifold_edge_count"] and meta["disconnected_bad_edge_count"]==0:
            break

    final=compact_metrics(world,faces,inv)
    # Verify strict refinement of original compaction: each final cluster maps to one base cluster.
    pairs=np.unique(np.column_stack((base,inv)),axis=0)
    seen={}
    cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b): cross+=1

    report={
        "schema":"RealSaS.KnightEdgePreimageManifoldRefinementCourt.v1",
        "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
        "operator":{
            "id":"TOPOLOGY_LOCAL_PLUS_EDGE_PREIMAGE_REFINEMENT_V1",
            "generic_invariants":[
                "NO_CROSS_BASE_CLUSTER_MERGE",
                "DENSE_EDGE_CONNECTED_VERTEX_PREIMAGE",
                "QUOTIENT_EDGE_PREIMAGE_COMPONENTS_MUST_NOT_ALIAS",
                "FACE_DELETION_FORBIDDEN",
                "TEACHER_OR_SUBJECT_LABELS_FORBIDDEN",
            ],
        },
        "base":compact_metrics(world,faces,base),
        "history":history,
        "final":final,
        "cross_base_merge_count":int(cross),
        "success":bool(final["nonmanifold_edge_count"]==0 and cross==0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_EDGE_PREIMAGE_MANIFOLD_REFINEMENT="+json.dumps({
        "base_nonmanifold":report["base"]["nonmanifold_edge_count"],
        "history":[[x["iteration"],x["metrics"]["nonmanifold_edge_count"],x["metrics"]["compact_node_count"]] for x in history],
        "final_nonmanifold":final["nonmanifold_edge_count"],
        "final_nodes":final["compact_node_count"],
        "cross_base_merge_count":cross,
        "success":report["success"],
    },sort_keys=True))
    if not report["success"]:
        raise SystemExit(2)

if __name__=="__main__": main()
