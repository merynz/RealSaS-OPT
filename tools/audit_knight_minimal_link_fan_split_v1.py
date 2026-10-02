from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import normalization_domain_from_dict,signed_zero_surface_from_dict
from compiler.realsas_compiler_core.substrate.scene_first_signed import _adaptive_voxel_compact,mesh_connected_component_labels_v1
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine,compact_metrics
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import split_by_edge_preimage_components
from tools.audit_knight_link_safe_local_decompaction_v1 import bad_edges_and_links

def loadj(p): return json.loads(Path(p).read_text())

def compact_faces_with_dense(faces,inv):
    mapped=inv[faces]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    return faces[valid],mapped[valid]

def bad_edge_rows(faces,inv):
    raw,mapped=compact_faces_with_dense(faces,inv)
    canon=np.sort(mapped,axis=1)
    uniq,first=np.unique(canon,axis=0,return_index=True)
    edge_rows=defaultdict(list)
    for fi,f in enumerate(uniq):
        a,b,c=map(int,f)
        for u,v in ((a,b),(b,c),(c,a)):
            e=(u,v) if u<v else (v,u)
            edge_rows[e].append(fi)
    return uniq,{e:r for e,r in edge_rows.items() if len(r)>2}

def split_offender_vertex_fans(faces,inv):
    inv=np.asarray(inv,dtype=np.int64)
    uniq,bad=bad_edge_rows(faces,inv)
    if not bad:
        return inv.copy(),{"bad_edges_before":0,"split_clusters":0,"added_nodes":0}

    offenders=sorted({x for e in bad for x in e})
    next_id=int(inv.max())+1
    out=inv.copy()
    split_clusters=0

    # For each offending quotient vertex, split its dense preimage by connected
    # components of the induced triangle fan around that vertex. Two dense
    # vertices are connected only when a dense face containing both maps to
    # triangles incident to the same quotient vertex. This is a pure refinement.
    for qv in offenders:
        dense=np.where(inv==qv)[0]
        if len(dense)<=1:
            continue
        dense_set=set(map(int,dense))
        g=defaultdict(set)
        # Dense edge adjacency inside this quotient preimage.
        for f in np.asarray(faces,dtype=np.int64):
            members=[int(x) for x in f if int(x) in dense_set]
            for i in range(len(members)):
                for j in range(i+1,len(members)):
                    a,b=members[i],members[j]
                    g[a].add(b);g[b].add(a)
        for d in dense:
            g[int(d)]
        comps=[];seen=set()
        for d in sorted(g):
            if d in seen: continue
            st=[d];seen.add(d);cc=[]
            while st:
                x=st.pop();cc.append(x)
                for y in g[x]:
                    if y not in seen:
                        seen.add(y);st.append(y)
            comps.append(sorted(cc))
        if len(comps)<=1:
            continue

        # Keep first component on old id, mint one id per additional fan.
        split_clusters+=1
        for cc in comps[1:]:
            nid=next_id;next_id+=1
            out[np.asarray(cc,dtype=np.int64)]=nid

    _,out=np.unique(out,return_inverse=True)
    out=np.asarray(out,dtype=np.int64)
    return out,{
      "bad_edges_before":len(bad),
      "split_clusters":split_clusters,
      "added_nodes":int(out.max())+1-(int(inv.max())+1),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64);faces=np.asarray(z["faces"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)
    normals=np.zeros_like(world);normals[:,2]=1.0
    labels=mesh_connected_component_labels_v1(len(world),faces)

    _cp,_cn,_edges,divisions,base=_adaptive_voxel_compact(
        world,faces,normals,target_nodes=11520,
        preserve_connected_components=True,
        precomputed_component_labels=labels,
    )
    inv,_=topology_local_refine(faces,np.asarray(base,dtype=np.int64))
    history=[]
    for i in range(32):
        m=compact_metrics(world,faces,inv)
        history.append({"phase":"EDGE_PREIMAGE","iteration":i,"nodes":m["compact_node_count"],"nonmanifold":m["nonmanifold_edge_count"]})
        if m["nonmanifold_edge_count"]==0: break
        nxt,_meta=split_by_edge_preimage_components(faces,inv)
        nm=compact_metrics(world,faces,nxt)["nonmanifold_edge_count"]
        inv=nxt
        if nm==m["nonmanifold_edge_count"]:
            break

    before=compact_metrics(world,faces,inv)
    before_rows=bad_edges_and_links(faces,inv)
    inv2,fan_meta=split_offender_vertex_fans(faces,inv)
    after=compact_metrics(world,faces,inv2)
    after_rows=bad_edges_and_links(faces,inv2)

    # Strict refinement of original voxel compaction.
    pairs=np.unique(np.column_stack((np.asarray(base,dtype=np.int64),inv2)),axis=0)
    seen={};cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b): cross+=1

    report={
      "schema":"RealSaS.KnightMinimalLinkFanSplitCourt.v1",
      "status":"MEASURED_AUDIT_ONLY",
      "target_nodes":11520,
      "voxel_divisions":int(divisions),
      "history":history,
      "before":{"metrics":before,"bad_edges":before_rows},
      "fan_split":fan_meta,
      "after":{"metrics":after,"bad_edges":after_rows},
      "cross_base_merge_count":cross,
      "within_12288":bool(after["compact_node_count"]<=12288),
      "success":bool(after["nonmanifold_edge_count"]==0 and after["compact_node_count"]<=12288 and cross==0),
      "generic_invariants":[
        "PURE_REFINEMENT_OF_BASE_VOXEL_EQUIVALENCE",
        "OFFENDING_VERTEX_LINK_FANS_MAY_NOT_ALIAS",
        "FACE_DELETION_FORBIDDEN",
        "TEACHER_OR_SUBJECT_LABELS_FORBIDDEN",
        "FINAL_NODE_COUNT_MUST_NOT_EXCEED_12288",
      ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_MINIMAL_LINK_FAN_SPLIT="+json.dumps({
      "division":int(divisions),
      "before_nodes":before["compact_node_count"],
      "before_nonmanifold":before["nonmanifold_edge_count"],
      "before_bad_edges":before_rows,
      "added_nodes":fan_meta["added_nodes"],
      "after_nodes":after["compact_node_count"],
      "after_nonmanifold":after["nonmanifold_edge_count"],
      "after_bad_edges":after_rows,
      "cross_base_merge_count":cross,
      "within_12288":report["within_12288"],
      "success":report["success"],
    },sort_keys=True))
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__": main()
