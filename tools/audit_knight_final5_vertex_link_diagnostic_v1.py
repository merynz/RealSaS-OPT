from __future__ import annotations
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,signed_zero_surface_from_dict,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine,compact_metrics
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import (
    split_by_edge_preimage_components,bad_edge_codes,
)
from tools.audit_knight_bad_edge_opposite_fan_refinement_v1 import (
    collect_bad_edge_opposite_tokens,refine_by_tokens,
)

def loadj(p): return json.loads(Path(p).read_text())

def quotient_faces(faces,inv):
    mapped=inv[np.asarray(faces,dtype=np.int64)]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    return np.unique(np.sort(mapped[valid],axis=1),axis=0)

def bad_edges(qf):
    edge_rows=defaultdict(list)
    for fi,f in enumerate(qf):
        a,b,c=map(int,f)
        for u,v in ((a,b),(b,c),(c,a)):
            e=(u,v) if u<v else (v,u)
            edge_rows[e].append(fi)
    return {e:r for e,r in edge_rows.items() if len(r)>2}

def link_graph(qf,vid):
    g=defaultdict(set)
    incident=[]
    for fi,f in enumerate(qf):
        if int(vid) not in set(map(int,f)): continue
        incident.append(fi)
        o=[int(x) for x in f if int(x)!=int(vid)]
        if len(o)==2:
            a,b=o;g[a].add(b);g[b].add(a)
    nodes=sorted(g)
    seen=set();comps=[]
    for n in nodes:
        if n in seen: continue
        st=[n];seen.add(n);cc=[]
        while st:
            x=st.pop();cc.append(x)
            for y in g[x]:
                if y not in seen:seen.add(y);st.append(y)
        comps.append(sorted(cc))
    deg={int(n):len(g[n]) for n in nodes}
    branch=[n for n,d in deg.items() if d>2]
    endpoints=[n for n,d in deg.items() if d==1]
    topology=("CYCLE" if len(comps)==1 and nodes and all(d==2 for d in deg.values())
              else "PATH" if len(comps)==1 and len(endpoints)==2 and all(d<=2 for d in deg.values())
              else "DISCONNECTED_OR_BRANCHED")
    return {
      "incident_face_count":len(incident),
      "component_count":len(comps),
      "component_sizes":[len(x) for x in comps],
      "degree_histogram":dict(sorted(Counter(deg.values()).items())),
      "branch_nodes":branch,
      "endpoint_nodes":endpoints,
      "topology":topology,
      "neighbors":nodes,
    }

def dense_preimage_components(faces,inv,e):
    u,v=map(int,e)
    pairs=set()
    rows=[]
    f=np.asarray(faces,dtype=np.int64)
    for raw in f:
        mf=inv[raw]
        if len(set(map(int,mf)))<3: continue
        for ia,ib,io in ((0,1,2),(1,2,0),(2,0,1)):
            qa,qb,qo=map(int,(mf[ia],mf[ib],mf[io]))
            if {qa,qb}!={u,v}: continue
            da,db,do=map(int,(raw[ia],raw[ib],raw[io]))
            if qa==u: du,dv=da,db
            else: du,dv=db,da
            pairs.add((du,dv))
            rows.append((du,dv,do,qo))
    graph=defaultdict(set)
    for du,dv in pairs:
        graph[("U",du)].add(("V",dv));graph[("V",dv)].add(("U",du))
    seen=set();comps=[]
    for n in sorted(graph,key=lambda x:(x[0],x[1])):
        if n in seen:continue
        st=[n];seen.add(n);cc=[]
        while st:
            x=st.pop();cc.append(x)
            for y in graph[x]:
                if y not in seen:seen.add(y);st.append(y)
        comps.append(cc)
    return {
      "valid_dense_edge_count":len(pairs),
      "component_count":len(comps),
      "component_sizes":[len(x) for x in comps],
      "valid_face_witness_count":len(rows),
      "quotient_opposites":sorted({int(r[3]) for r in rows}),
    }

def reconstruct(world,faces):
    base=compact_inverse(world,faces,divisions=56,component_aware=True)
    inv,_=topology_local_refine(faces,base)
    history=[]
    for it in range(1,20):
        m=compact_metrics(world,faces,inv)
        history.append(["EDGE_PREIMAGE",it-1,m["nonmanifold_edge_count"],m["compact_node_count"]])
        if m["nonmanifold_edge_count"]<=7:break
        inv,_=split_by_edge_preimage_components(faces,inv)
    for it in range(1,3):
        m=compact_metrics(world,faces,inv)
        bad_codes,_bad_inc,n=bad_edge_codes(faces,inv)
        tokens,_w=collect_bad_edge_opposite_tokens(faces,inv,bad_codes,n)
        inv,_=refine_by_tokens(faces,inv,tokens)
        mm=compact_metrics(world,faces,inv)
        history.append(["OPPOSITE_FAN",it,mm["nonmanifold_edge_count"],mm["compact_node_count"]])
    return base,inv,history

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64);faces=np.asarray(z["faces"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)
    base,inv,hist=reconstruct(world,faces)
    qf=quotient_faces(faces,inv);bad=bad_edges(qf)
    rows=[]
    for e,fis in sorted(bad.items()):
        u,v=e
        lu=link_graph(qf,u);lv=link_graph(qf,v)
        common=sorted(set(lu["neighbors"])&set(lv["neighbors"]))
        opp=sorted({int(x) for fi in fis for x in qf[fi] if int(x) not in e})
        pre=dense_preimage_components(faces,inv,e)
        rows.append({
          "edge":[u,v],"incidence":len(fis),"opposites":opp,"common_neighbors":common,
          "link_condition_exact":bool(set(common)==set(opp) and len(opp)<=2),
          "u_link":lu,"v_link":lv,"dense_preimage":pre,
          "u_dense_preimage_size":int(np.sum(inv==u)),
          "v_dense_preimage_size":int(np.sum(inv==v)),
        })
    final=compact_metrics(world,faces,inv)
    report={"schema":"RealSaS.KnightFinal5VertexLinkDiagnostic.v1","status":"MEASURED_AUDIT_ONLY",
            "history":hist,"final":final,"rows":rows}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_FINAL5_VERTEX_LINK="+json.dumps({
      "final_nodes":final["compact_node_count"],"final_nonmanifold":final["nonmanifold_edge_count"],
      "rows":[{"edge":r["edge"],"inc":r["incidence"],"opp":len(r["opposites"]),"common":len(r["common_neighbors"]),
               "u_top":r["u_link"]["topology"],"v_top":r["v_link"]["topology"],
               "pre_components":r["dense_preimage"]["component_count"],
               "pre_edges":r["dense_preimage"]["valid_dense_edge_count"]} for r in rows]
    },sort_keys=True))
if __name__=="__main__":main()
