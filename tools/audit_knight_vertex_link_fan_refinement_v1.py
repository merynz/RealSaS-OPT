from __future__ import annotations
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import signed_zero_surface_from_dict
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine,compact_metrics

def loadj(p): return json.loads(Path(p).read_text())

def compact_faces(faces,inv):
    mapped=np.asarray(inv,dtype=np.int64)[np.asarray(faces,dtype=np.int64)]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    return np.unique(np.sort(mapped[valid],axis=1),axis=0)

def link_components(faces,node_count):
    links=[defaultdict(set) for _ in range(node_count)]
    incident=np.zeros(node_count,dtype=np.int64)
    for a,b,c in np.asarray(faces,dtype=np.int64):
        a=int(a);b=int(b);c=int(c)
        incident[a]+=1;incident[b]+=1;incident[c]+=1
        links[a][b].add(c);links[a][c].add(b)
        links[b][a].add(c);links[b][c].add(a)
        links[c][a].add(b);links[c][b].add(a)
    bad={}
    for v,g in enumerate(links):
        if incident[v]==0: continue
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
            comps.append(sorted(cc))
        d1=sum(d==1 for d in deg.values())
        legal=(len(comps)==1 and (all(d==2 for d in deg.values()) or
               (d1==2 and all(d in (1,2) for d in deg.values()))))
        if not legal:
            bad[v]={"components":comps,"degree_histogram":dict(Counter(deg.values()))}
    return bad

def refine_bad_vertex_links(faces,inv):
    inv=np.asarray(inv,dtype=np.int64)
    cf=compact_faces(faces,inv)
    bad=link_components(cf,int(inv.max())+1)
    if not bad:
        return inv.copy(),{"bad_vertex_count_before":0,"multi_fan_dense_vertex_count":0,"unassigned_dense_vertex_count":0}

    neighbor_comp={}
    for q,row in bad.items():
        for ci,comp in enumerate(sorted(row["components"],key=lambda x:(len(x),x))):
            for nb in comp:
                neighbor_comp[(int(q),int(nb))]=int(ci)

    bad_ids=np.asarray(sorted(bad),dtype=np.int64)
    tokens=defaultdict(set)
    f=np.asarray(faces,dtype=np.int64)
    chunk=300_000
    for start in range(0,len(f),chunk):
        raw=f[start:start+chunk]
        mapped=inv[raw]
        valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
        if not np.any(valid):continue
        raw=raw[valid];mapped=mapped[valid]
        for i,j,k in ((0,1,2),(1,0,2),(2,0,1)):
            q=mapped[:,i]
            take=np.isin(q,bad_ids,assume_unique=False)
            for rowidx in np.nonzero(take)[0].tolist():
                qq=int(q[rowidx]); n1=int(mapped[rowidx,j]); n2=int(mapped[rowidx,k])
                c1=neighbor_comp.get((qq,n1));c2=neighbor_comp.get((qq,n2))
                if c1 is None or c2 is None or c1!=c2:
                    raise RuntimeError(f"VERTEX_LINK_FACE_COMPONENT_INCONSISTENT:{qq}:{n1}:{n2}:{c1}:{c2}")
                tokens[int(raw[rowidx,i])].add((qq,int(c1)))

    rows=np.empty(len(inv),dtype=np.int64)
    sig_ids={}
    multi=0;unassigned=0
    for vid in range(len(inv)):
        q=int(inv[vid])
        if q not in bad:
            key=(q,("KEEP",))
        else:
            tt=tuple(sorted(tokens.get(vid,())))
            if len(tt)>1: multi+=1
            if not tt: unassigned+=1
            # Empty evidence gets an isolated residual child rather than being
            # allowed to bridge two proven fan classes.
            key=(q,tt if tt else (("UNASSIGNED",),))
        rows[vid]=sig_ids.setdefault(key,len(sig_ids))
    refined=np.asarray(rows,dtype=np.int64)
    refined,_=topology_local_refine(faces,refined)
    return refined,{
      "bad_vertex_count_before":len(bad),
      "bad_vertices":sorted(int(x) for x in bad),
      "multi_fan_dense_vertex_count":int(multi),
      "unassigned_dense_vertex_count":int(unassigned),
      "signature_count":len(sig_ids),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        faces=np.asarray(z["faces"],dtype=np.int64)
        verts=np.asarray(z["vertices_normalized"],dtype=np.float64)
    with np.load(a.inverse_npz,allow_pickle=False) as z:
        inv=np.asarray(z["final_inverse"],dtype=np.int64)
        base=np.asarray(z["base_inverse"],dtype=np.int64)

    before_cf=compact_faces(faces,inv)
    before_bad=link_components(before_cf,int(inv.max())+1)
    refined,meta=refine_bad_vertex_links(faces,inv)
    after_cf=compact_faces(faces,refined)
    after_bad=link_components(after_cf,int(refined.max())+1)
    metrics=compact_metrics(verts,faces,refined)

    pairs=np.unique(np.column_stack((base,refined)),axis=0)
    seen={};cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b):cross+=1

    report={
      "schema":"RealSaS.KnightVertexLinkFanRefinementCourt.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
      "before":{"nodes":int(inv.max())+1,"illegal_vertex_links":len(before_bad)},
      "meta":meta,
      "after":{"nodes":int(refined.max())+1,"illegal_vertex_links":len(after_bad),
               "nonmanifold_edges":metrics["nonmanifold_edge_count"]},
      "cross_base_merge_count":cross,
      "pure_refinement":cross==0,
      "success":bool(len(after_bad)==0 and metrics["nonmanifold_edge_count"]==0 and cross==0 and meta["multi_fan_dense_vertex_count"]==0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_VERTEX_LINK_FAN_REFINEMENT="+json.dumps(report,sort_keys=True),flush=True)
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__":main()
