from __future__ import annotations
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import signed_zero_surface_from_dict

def loadj(p): return json.loads(Path(p).read_text())

def compact_faces(faces,inv):
    mapped=np.asarray(inv,dtype=np.int64)[np.asarray(faces,dtype=np.int64)]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    return np.unique(np.sort(mapped[valid],axis=1),axis=0)

def edge_incidence(faces):
    inc=defaultdict(int)
    for a,b,c in np.asarray(faces,dtype=np.int64):
        for u,v in ((a,b),(b,c),(c,a)):
            if u>v:u,v=v,u
            inc[(int(u),int(v))]+=1
    return inc

def vertex_link_rows(faces,node_count):
    links=[defaultdict(set) for _ in range(node_count)]
    incident=np.zeros(node_count,dtype=np.int64)
    for a,b,c in np.asarray(faces,dtype=np.int64):
        a=int(a);b=int(b);c=int(c)
        incident[a]+=1;incident[b]+=1;incident[c]+=1
        links[a][b].add(c);links[a][c].add(b)
        links[b][a].add(c);links[b][c].add(a)
        links[c][a].add(b);links[c][b].add(a)
    rows=[]
    bad=[]
    for v,g in enumerate(links):
        if incident[v]==0:
            continue
        nodes=sorted(g)
        deg={n:len(g[n]) for n in nodes}
        seen=set(); comps=[]
        for n in nodes:
            if n in seen:continue
            st=[n];seen.add(n);cc=[]
            while st:
                x=st.pop();cc.append(x)
                for y in g[x]:
                    if y not in seen:
                        seen.add(y);st.append(y)
            comps.append(cc)
        degree_hist=dict(Counter(deg.values()))
        d1=sum(d==1 for d in deg.values())
        all2=all(d==2 for d in deg.values())
        path=(len(comps)==1 and d1==2 and all(d in (1,2) for d in deg.values()))
        cycle=(len(comps)==1 and all2)
        legal=bool(path or cycle)
        row={
          "vertex":v,"incident_faces":int(incident[v]),"link_node_count":len(nodes),
          "link_component_count":len(comps),"link_component_sizes":sorted(len(x) for x in comps),
          "degree_histogram":{str(k):int(val) for k,val in sorted(degree_hist.items())},
          "boundary_path":path,"interior_cycle":cycle,"legal":legal,
        }
        rows.append(row)
        if not legal:bad.append(row)
    return rows,bad

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        faces=np.asarray(z["faces"],dtype=np.int64)
    with np.load(a.inverse_npz,allow_pickle=False) as z:
        inv=np.asarray(z["final_inverse"],dtype=np.int64)
    cf=compact_faces(faces,inv)
    inc=edge_incidence(cf)
    nonmanifold_edges=[(e,n) for e,n in inc.items() if n>2]
    rows,bad=vertex_link_rows(cf,int(inv.max())+1)
    report={
      "schema":"RealSaS.KnightRepairedVertexManifoldCourt.v1",
      "status":"MEASURED_AUDIT_ONLY",
      "node_count":int(inv.max())+1,"face_count":len(cf),
      "edge_nonmanifold_count":len(nonmanifold_edges),
      "illegal_vertex_link_count":len(bad),
      "illegal_vertex_link_examples":bad[:100],
      "illegal_link_component_count_histogram":dict(Counter(r["link_component_count"] for r in bad)),
      "all_vertex_links_manifold":len(bad)==0,
      "success":bool(len(nonmanifold_edges)==0 and len(bad)==0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_REPAIRED_VERTEX_MANIFOLD="+json.dumps({
      "nodes":report["node_count"],"faces":report["face_count"],
      "nonmanifold_edges":report["edge_nonmanifold_count"],
      "illegal_vertex_links":report["illegal_vertex_link_count"],
      "link_component_hist":report["illegal_link_component_count_histogram"],
      "success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__":main()
