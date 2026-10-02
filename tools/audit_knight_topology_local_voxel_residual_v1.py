from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import normalization_domain_from_dict, signed_zero_surface_from_dict
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine


def loadj(p:Path): return json.loads(p.read_text())

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
    ref,_=topology_local_refine(faces,base)

    mapped=ref[faces]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    raw=faces[valid]
    mapped=mapped[valid]
    canon=np.sort(mapped,axis=1)
    # dedupe mapped faces but retain one raw witness index for structural census
    unique_face, first=np.unique(canon,axis=0,return_index=True)
    raw_witness=raw[first]
    n=int(ref.max())+1

    edge_rows=defaultdict(list)
    for fi,f in enumerate(unique_face):
        for ia,ib in ((0,1),(1,2),(2,0)):
            x,y=map(int,(f[ia],f[ib])); e=(x,y) if x<y else (y,x)
            edge_rows[e].append(fi)
    bad={e:r for e,r in edge_rows.items() if len(r)>2}

    # For every bad quotient edge A-B, inspect the original dense edge preimage.
    rows=[]
    class_count=Counter()
    for edge, fis in sorted(bad.items()):
        A,B=edge
        # collect all original dense edges from noncollapsed dense faces that map to A-B
        pre_edges=Counter()
        pre_face_ids=[]
        for di,(rf,mf) in enumerate(zip(raw,mapped)):
            hit=False
            for ia,ib in ((0,1),(1,2),(2,0)):
                ca,cb=int(mf[ia]),int(mf[ib])
                if {ca,cb}=={A,B}:
                    u,v=map(int,(rf[ia],rf[ib]))
                    pe=(u,v) if u<v else (v,u)
                    pre_edges[pe]+=1
                    hit=True
            if hit: pre_face_ids.append(di)

        dense_nonmanifold=sum(1 for c in pre_edges.values() if c>2)

        # Build bipartite connectivity between original A-side and B-side vertices
        # using only original dense edges that collapse to quotient A-B.
        graph=defaultdict(set)
        aside=set(); bside=set()
        for (u,v),c in pre_edges.items():
            ru,rv=int(ref[u]),int(ref[v])
            if ru==A and rv==B:
                au,bv=u,v
            elif ru==B and rv==A:
                au,bv=v,u
            else:
                continue
            aside.add(au); bside.add(bv)
            graph[("A",au)].add(("B",bv)); graph[("B",bv)].add(("A",au))

        seen=set(); comps=[]
        for node in list(graph):
            if node in seen: continue
            stack=[node]; seen.add(node); na=nb=edges=0
            while stack:
                x=stack.pop()
                if x[0]=="A": na+=1
                else: nb+=1
                edges+=len(graph[x])
                for y in graph[x]:
                    if y not in seen:
                        seen.add(y); stack.append(y)
            comps.append((na,nb,edges//2))

        # Count distinct opposite compact vertices from incident quotient faces.
        opposite=[]
        for fi in fis:
            f=unique_face[fi]
            opp=[int(x) for x in f if int(x) not in edge]
            opposite.extend(opp)
        opposite_unique=sorted(set(opposite))

        if dense_nonmanifold:
            cls="DENSE_SOURCE_NONMANIFOLD"
        elif len(comps)>1:
            cls="MULTIPLE_DISCONNECTED_EDGE_PREIMAGE_COMPONENTS"
        elif len(opposite_unique)>2:
            cls="CONNECTED_EDGE_PREIMAGE_BUT_LINK_CONDITION_VIOLATION"
        else:
            cls="OTHER"
        class_count[cls]+=1
        rows.append({
            "quotient_edge":[A,B],
            "quotient_incidence":len(fis),
            "distinct_opposite_quotient_vertices":len(opposite_unique),
            "opposite_quotient_vertices":opposite_unique[:16],
            "original_dense_edge_count":len(pre_edges),
            "max_original_dense_edge_incidence":max(pre_edges.values(),default=0),
            "dense_source_nonmanifold_edge_count":dense_nonmanifold,
            "edge_preimage_component_count":len(comps),
            "edge_preimage_components":[list(x) for x in sorted(comps,reverse=True)[:16]],
            "a_side_dense_vertex_count":len(aside),
            "b_side_dense_vertex_count":len(bside),
            "classification":cls,
        })

    report={
      "schema":"RealSaS.KnightTopologyLocalVoxelResidual.v1",
      "status":"MEASURED",
      "refined_node_count":int(n),
      "residual_nonmanifold_edge_count":len(bad),
      "classification_counts":dict(class_count),
      "rows":rows,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TOPOLOGY_LOCAL_RESIDUAL="+json.dumps({
      "residual":len(bad),
      "classes":dict(class_count),
      "sample":rows[:12],
    },sort_keys=True))

if __name__=="__main__": main()
