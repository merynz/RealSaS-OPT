from __future__ import annotations
import argparse,json
from collections import defaultdict,Counter,deque
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

    # Propagate each proven fan label through the dense-edge graph inside the
    # original quotient vertex.  Face-silent dense vertices are not a third fan:
    # they inherit the nearest proven fan in dense-edge distance.  Components
    # with no fan seed remain isolated ORPHAN_COMPONENT classes and therefore
    # cannot bridge two proven fans.
    propagated={}
    orphan_component_count=0
    mixed_seed_component_count=0
    tie_count=0

    farr=np.asarray(faces,dtype=np.int64)
    for q in sorted(bad):
        dense=np.nonzero(inv==int(q))[0]
        dset=set(map(int,dense.tolist()))
        adj={int(v):set() for v in dense.tolist()}
        for start in range(0,len(farr),300_000):
            chunk=farr[start:start+300_000]
            for ia,ib in ((0,1),(1,2),(2,0)):
                a=chunk[:,ia]; b=chunk[:,ib]
                mask=np.isin(a,dense,assume_unique=False)&np.isin(b,dense,assume_unique=False)
                for u,v in zip(a[mask].tolist(),b[mask].tolist()):
                    u=int(u);v=int(v)
                    adj[u].add(v);adj[v].add(u)

        # Dense-edge connected components.
        seen=set()
        components=[]
        for root in sorted(adj):
            if root in seen: continue
            dq=[root];seen.add(root);cc=[]
            while dq:
                x=dq.pop();cc.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y);dq.append(y)
            components.append(sorted(cc))

        for ci,cc in enumerate(components):
            seed_rows=[]
            seed_labels=set()
            for v in cc:
                tt=tuple(sorted(tokens.get(v,())))
                if len(tt)>1:
                    raise RuntimeError(f"MULTI_FAN_DENSE_VERTEX:{q}:{v}:{tt}")
                if tt:
                    label=int(tt[0][1])
                    seed_rows.append((v,label))
                    seed_labels.add(label)
            if not seed_rows:
                orphan_component_count+=1
                label=("ORPHAN_COMPONENT",int(q),int(ci))
                for v in cc:
                    propagated[v]=label
                continue
            if len(seed_labels)>1:
                mixed_seed_component_count+=1

            # Multi-source shortest-path on dense edges, deterministic tie-break
            # by fan label then seed vertex id.
            best={}
            dq=deque()
            for v,label in sorted(seed_rows,key=lambda x:(x[1],x[0])):
                cand=(0,int(label),int(v))
                old=best.get(v)
                if old is None or cand<old:
                    best[v]=cand; dq.append(v)
            while dq:
                x=dq.popleft()
                dist,label,seed=best[x]
                for y in sorted(adj[x]):
                    cand=(dist+1,label,seed)
                    old=best.get(y)
                    if old is None or cand<old:
                        if old is not None and cand[0]==old[0] and cand[1]!=old[1]:
                            tie_count+=1
                        best[y]=cand; dq.append(y)
            for v in cc:
                if v not in best:
                    raise RuntimeError(f"PROPAGATION_UNREACHED:{q}:{v}")
                propagated[v]=("FAN",int(q),int(best[v][1]))

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
            key=(q,propagated[int(vid)])
        rows[vid]=sig_ids.setdefault(key,len(sig_ids))
    refined=np.asarray(rows,dtype=np.int64)
    refined,_=topology_local_refine(faces,refined)
    return refined,{
      "bad_vertex_count_before":len(bad),
      "bad_vertices":sorted(int(x) for x in bad),
      "multi_fan_dense_vertex_count":int(multi),
      "unassigned_dense_vertex_count":int(unassigned),
      "orphan_dense_component_count":int(orphan_component_count),
      "mixed_seed_dense_component_count":int(mixed_seed_component_count),
      "fan_propagation_tie_count":int(tie_count),
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

    # Diagnose how each original illegal quotient vertex distributed into refined children.
    child_diagnostics={}
    for q in sorted(before_bad):
        dense_ids=np.nonzero(inv==int(q))[0]
        children=np.unique(refined[dense_ids])
        rows=[]
        for child in children.tolist():
            members=dense_ids[refined[dense_ids]==int(child)]
            token_hist=Counter()
            for vid in members.tolist():
                # Reconstruct fan token from incident nondegenerate dense faces lazily below
                pass
            rows.append({"child":int(child),"dense_vertex_count":int(len(members)),
                         "illegal_after":bool(int(child) in after_bad)})
        child_diagnostics[str(int(q))]={
            "dense_preimage_count":int(len(dense_ids)),
            "child_count":int(len(children)),
            "children":rows,
        }

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
      "after_illegal_vertex_ids":sorted(int(x) for x in after_bad),
      "after_illegal_vertex_rows":{
        str(int(k)):{
          "component_sizes":[len(c) for c in v["components"]],
          "degree_histogram":{str(kk):int(vv) for kk,vv in v["degree_histogram"].items()},
        } for k,v in after_bad.items()
      },
      "child_diagnostics":child_diagnostics,
      "success":bool(len(after_bad)==0 and metrics["nonmanifold_edge_count"]==0 and cross==0 and meta["multi_fan_dense_vertex_count"]==0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_VERTEX_LINK_FAN_REFINEMENT="+json.dumps(report,sort_keys=True),flush=True)
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__":main()