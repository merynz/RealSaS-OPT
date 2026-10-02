from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,
    signed_zero_surface_from_dict,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import (
    split_by_edge_preimage_components,
)


def loadj(p:Path): return json.loads(p.read_text())


def classify(faces: np.ndarray, inv: np.ndarray) -> dict:
    mapped=inv[faces]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    raw=faces[valid]
    mapped=mapped[valid]
    canon=np.sort(mapped,axis=1)
    unique_face, first=np.unique(canon,axis=0,return_index=True)
    n=int(inv.max())+1

    edge_rows=defaultdict(list)
    for fi,f in enumerate(unique_face):
        for ia,ib in ((0,1),(1,2),(2,0)):
            x,y=map(int,(f[ia],f[ib])); e=(x,y) if x<y else (y,x)
            edge_rows[e].append(fi)
    bad={e:r for e,r in edge_rows.items() if len(r)>2}

    rows=[]; cc=Counter()
    for edge,fis in sorted(bad.items()):
        A,B=edge
        pre=Counter()
        for rf,mf in zip(raw,mapped):
            for ia,ib in ((0,1),(1,2),(2,0)):
                ca,cb=int(mf[ia]),int(mf[ib])
                if {ca,cb}!={A,B}: continue
                u,v=map(int,(rf[ia],rf[ib]))
                if ca==A and cb==B:
                    pe=(u,v)
                elif ca==B and cb==A:
                    pe=(v,u)
                else:
                    continue
                pre[pe]+=1

        graph=defaultdict(set)
        for (u,v),_c in pre.items():
            graph[("A",u)].add(("B",v))
            graph[("B",v)].add(("A",u))
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

        opposite=[]
        for fi in fis:
            f=unique_face[fi]
            opposite.extend([int(x) for x in f if int(x) not in edge])
        opp=sorted(set(opposite))

        # Link condition for manifold-preserving edge contraction:
        # link(A) ∩ link(B) should equal link(edge) (two opposite vertices for
        # an interior edge, one for a boundary edge). More than two distinct
        # opposite quotient vertices on an edge is already non-manifold.
        if len(comps)>1:
            cls="DISCONNECTED_EDGE_PREIMAGE_REMAINS"
        elif len(opp)>2:
            cls="CONNECTED_PREIMAGE_LINK_CONDITION_VIOLATION"
        else:
            cls="OTHER"
        cc[cls]+=1
        rows.append({
            "edge":[A,B],
            "incidence":len(fis),
            "opposite_vertices":opp,
            "edge_preimage_component_count":len(comps),
            "original_dense_edge_count":len(pre),
            "max_original_dense_edge_incidence":max(pre.values(),default=0),
            "classification":cls,
        })
    return {"count":len(bad),"classes":dict(cc),"rows":rows}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
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
    inv,_=topology_local_refine(faces,base)
    hist=[classify(faces,inv)["count"]]
    for _ in range(6):
        if hist[-1]==0: break
        inv,_=split_by_edge_preimage_components(faces,inv)
        hist.append(classify(faces,inv)["count"])
    final=classify(faces,inv)
    report={
      "schema":"RealSaS.KnightFinalManifoldResidual.v1",
      "status":"MEASURED",
      "history":hist,
      "final_node_count":int(inv.max())+1,
      "final":final,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_FINAL_MANIFOLD_RESIDUAL="+json.dumps({
      "history":hist,
      "final_node_count":int(inv.max())+1,
      "final_count":final["count"],
      "classes":final["classes"],
      "rows":final["rows"],
    },sort_keys=True))

if __name__=="__main__": main()
