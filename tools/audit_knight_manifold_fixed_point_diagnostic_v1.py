from __future__ import annotations
import argparse,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import normalization_domain_from_dict,signed_zero_surface_from_dict
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine,compact_metrics
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import split_by_edge_preimage_components

def loadj(p): return json.loads(Path(p).read_text())

def bad_edges(faces,inv):
    mapped=inv[faces]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    canon=np.sort(mapped[valid],axis=1)
    uniq=np.unique(canon,axis=0)
    edge_rows=defaultdict(list)
    for fi,f in enumerate(uniq):
        for ia,ib in ((0,1),(1,2),(2,0)):
            a,b=sorted((int(f[ia]),int(f[ib])))
            edge_rows[(a,b)].append(fi)
    return uniq,{e:r for e,r in edge_rows.items() if len(r)>2}

def vertex_link_components(uniq_faces,vid):
    nbr_graph=defaultdict(set)
    incident=[]
    for fi,f in enumerate(uniq_faces):
        if vid not in f: continue
        incident.append(fi)
        others=[int(x) for x in f if int(x)!=vid]
        if len(others)==2:
            a,b=others
            nbr_graph[a].add(b); nbr_graph[b].add(a)
    seen=set(); comps=[]
    for n in sorted(nbr_graph):
        if n in seen: continue
        st=[n];seen.add(n);nodes=[]
        while st:
            x=st.pop();nodes.append(x)
            for y in nbr_graph[x]:
                if y not in seen:
                    seen.add(y);st.append(y)
        comps.append(sorted(nodes))
    return incident,comps

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    surface=rigging_surface_from_dict(loadj(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64);faces=np.asarray(z["faces"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)
    md=dict(surface.metadata or {})
    base=compact_inverse(world,faces,divisions=int(md["compact_voxel_divisions"]),component_aware=bool(md.get("component_aware_compaction",False)))
    inv,_=topology_local_refine(faces,base)
    hist=[]
    for i in range(16):
        m=compact_metrics(world,faces,inv);hist.append(m["nonmanifold_edge_count"])
        nxt,meta=split_by_edge_preimage_components(faces,inv)
        nm=compact_metrics(world,faces,nxt)["nonmanifold_edge_count"]
        if nm==m["nonmanifold_edge_count"]:
            fixed_meta=meta;break
        inv=nxt
    uniq,bad=bad_edges(faces,inv)
    rows=[]
    for e,fis in sorted(bad.items()):
        u,v=e
        iu,cu=vertex_link_components(uniq,u);iv,cv=vertex_link_components(uniq,v)
        common=set(cu[0] if len(cu)==1 else []) & set(cv[0] if len(cv)==1 else [])
        opp=sorted({int(x) for fi in fis for x in uniq[fi] if int(x) not in e})
        rows.append({"edge":[u,v],"incidence":len(fis),"opposite":opp,
                     "u_link_component_count":len(cu),"v_link_component_count":len(cv),
                     "u_link_sizes":[len(x) for x in cu],"v_link_sizes":[len(x) for x in cv],
                     "single_link_common_neighbor_count":len(common),
                     "link_edge_opposite_count":len(opp)})
    report={"schema":"RealSaS.KnightManifoldFixedPointDiagnostic.v1","status":"MEASURED","history":hist,
            "fixed_meta":fixed_meta,"final_nonmanifold":len(bad),"rows":rows}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_MANIFOLD_FIXED_POINT="+json.dumps({"history":hist,"fixed_meta":fixed_meta,"final":len(bad),"rows":rows},sort_keys=True))
if __name__=="__main__":main()
