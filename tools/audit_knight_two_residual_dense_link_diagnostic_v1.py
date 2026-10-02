from __future__ import annotations
import argparse,json
from collections import defaultdict,deque
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import normalization_domain_from_dict,signed_zero_surface_from_dict
from compiler.realsas_compiler_core.substrate.scene_first_signed import _adaptive_voxel_compact,mesh_connected_component_labels_v1
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine,compact_metrics
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import split_by_edge_preimage_components
from tools.audit_knight_link_safe_local_decompaction_v1 import bad_edges_and_links

def loadj(p): return json.loads(Path(p).read_text())

def reconstruct(world,faces):
    normals=np.zeros_like(world);normals[:,2]=1.0
    labels=mesh_connected_component_labels_v1(len(world),faces)
    _cp,_cn,_edges,div,base=_adaptive_voxel_compact(
        world,faces,normals,target_nodes=11520,
        preserve_connected_components=True,precomputed_component_labels=labels)
    inv,_=topology_local_refine(faces,np.asarray(base,dtype=np.int64))
    hist=[]
    for i in range(32):
        m=compact_metrics(world,faces,inv)
        hist.append({"i":i,"nodes":m["compact_node_count"],"nonmanifold":m["nonmanifold_edge_count"]})
        if m["nonmanifold_edge_count"]==0: break
        nxt,_=split_by_edge_preimage_components(faces,inv)
        nm=compact_metrics(world,faces,nxt)["nonmanifold_edge_count"]
        inv=nxt
        if nm==m["nonmanifold_edge_count"]: break
    return np.asarray(base,dtype=np.int64),inv,div,hist

def edge_dense_preimage(faces,inv,e):
    u,v=e
    rows=[]
    for fi,f in enumerate(np.asarray(faces,dtype=np.int64)):
        mf=inv[f]
        for ia,ib,io in ((0,1,2),(1,2,0),(2,0,1)):
            a,b,o=int(mf[ia]),int(mf[ib]),int(mf[io])
            if {a,b}!={u,v}: continue
            da,db,do=map(int,(f[ia],f[ib],f[io]))
            if a==u and b==v: du,dv=da,db
            else: du,dv=db,da
            rows.append({"face_index":fi,"dense_u":du,"dense_v":dv,"dense_opp":do,"compact_opp":o})
    uniq={}
    for r in rows: uniq[(r["dense_u"],r["dense_v"])]=r
    return list(uniq.values())

def components_on_dense_edges(rows):
    # Two edge-preimage rows are connected when they share a dense endpoint.
    n=len(rows);g=[set() for _ in range(n)]
    for i in range(n):
        ai={rows[i]["dense_u"],rows[i]["dense_v"]}
        for j in range(i+1,n):
            if ai & {rows[j]["dense_u"],rows[j]["dense_v"]}:
                g[i].add(j);g[j].add(i)
    seen=set();comps=[]
    for i in range(n):
        if i in seen: continue
        q=[i];seen.add(i);cc=[]
        while q:
            x=q.pop();cc.append(x)
            for y in g[x]:
                if y not in seen:seen.add(y);q.append(y)
        comps.append(cc)
    return comps

def local_star_stats(faces,inv,qv):
    dense=np.where(inv==qv)[0]
    dset=set(map(int,dense))
    face_rows=[]
    for fi,f in enumerate(np.asarray(faces,dtype=np.int64)):
        if any(int(x) in dset for x in f):
            face_rows.append(fi)
    return {"dense_vertex_count":len(dense),"incident_dense_face_count":len(face_rows)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64);faces=np.asarray(z["faces"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)
    base,inv,div,hist=reconstruct(world,faces)
    bad=bad_edges_and_links(faces,inv)
    rows=[]
    for b in bad:
        e=tuple(map(int,b["edge"]))
        pre=edge_dense_preimage(faces,inv,e)
        comps=components_on_dense_edges(pre)
        rows.append({
          "edge":list(e),"incidence":b["incidence"],"opposite":b["opposite"],"common_neighbors":b["common_neighbors"],
          "u_star":local_star_stats(faces,inv,e[0]),"v_star":local_star_stats(faces,inv,e[1]),
          "dense_edge_preimage_count":len(pre),
          "dense_edge_preimage_component_count":len(comps),
          "dense_edge_preimages":pre,
          "components":comps,
        })
    report={"schema":"RealSaS.KnightTwoResidualDenseLinkDiagnostic.v1","target":11520,"divisions":int(div),
            "history":hist,"fixed_point_nodes":int(inv.max())+1,"residual_count":len(bad),"rows":rows}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TWO_RESIDUAL_DENSE_LINK="+json.dumps({
      "fixed_point_nodes":report["fixed_point_nodes"],"residual_count":len(bad),
      "rows":[{"edge":r["edge"],"incidence":r["incidence"],"pre":r["dense_edge_preimage_count"],
               "pre_components":r["dense_edge_preimage_component_count"],
               "u_dense":r["u_star"]["dense_vertex_count"],"v_dense":r["v_star"]["dense_vertex_count"]} for r in rows]
    },sort_keys=True))
if __name__=="__main__":main()
