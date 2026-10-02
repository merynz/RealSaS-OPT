from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict, signed_zero_surface_from_dict,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import compact_inverse
from tools.audit_knight_topology_local_voxel_compaction_v1 import topology_local_refine
from tools.audit_knight_edge_preimage_manifold_refinement_v1 import split_by_edge_preimage_components
from tools.audit_knight_link_safe_local_decompaction_v1 import (
    bad_edges_and_links, locally_decompact_offenders,
)

def loadj(p:Path): return json.loads(p.read_text())

def compact_points(points, inv):
    inv=np.asarray(inv,dtype=np.int64)
    n=int(inv.max())+1
    c=np.bincount(inv,minlength=n).astype(np.float64)
    out=np.zeros((n,3),dtype=np.float64)
    np.add.at(out,inv,points)
    out/=c[:,None]
    return out

def compact_faces(faces, inv):
    mapped=np.asarray(inv,dtype=np.int64)[np.asarray(faces,dtype=np.int64)]
    valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
    return np.unique(np.sort(mapped[valid],axis=1),axis=0)

def bad_edges(faces, inv):
    cf=compact_faces(faces,inv)
    rows=defaultdict(list)
    for fi,f in enumerate(cf):
        a,b,c=map(int,f)
        for u,v in ((a,b),(b,c),(c,a)):
            e=(u,v) if u<v else (v,u)
            rows[e].append(fi)
    return cf,{e:ids for e,ids in rows.items() if len(ids)>2}

def pca2(points):
    q=np.asarray(points,dtype=np.float64)
    mu=q.mean(axis=0)
    u,s,vt=np.linalg.svd(q-mu,full_matrices=False)
    return (q-mu)@vt[:2].T,mu,vt[:2]

def project(points,mu,basis):
    return (np.asarray(points)-mu)@basis.T

def refine_to_fixed_point(world,faces,base):
    inv,_=topology_local_refine(faces,base)
    for _ in range(32):
        before=len(bad_edges(faces,inv)[1])
        if before==0: break
        nxt,_=split_by_edge_preimage_components(faces,inv)
        after=len(bad_edges(faces,nxt)[1])
        inv=nxt
        if after==before:
            break
    link=bad_edges_and_links(faces,inv)
    if link:
        offenders=sorted({x for row in link for x in row["edge"]})
        inv,_=locally_decompact_offenders(faces,base,inv,offenders)
    return inv

def representative_preimage(faces,base,edge):
    A,B=edge
    pairs=[]
    for f in np.asarray(faces,dtype=np.int64):
        mf=base[f]
        for ia,ib in ((0,1),(1,2),(2,0)):
            ca,cb=int(mf[ia]),int(mf[ib])
            if {ca,cb}!={A,B}: continue
            u,v=map(int,(f[ia],f[ib]))
            if ca==A and cb==B:
                pairs.append((u,v))
            elif ca==B and cb==A:
                pairs.append((v,u))
    return sorted(set(pairs))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    rr=a.run_root.resolve()
    out=a.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)

    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    surface=rigging_surface_from_dict(loadj(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],dtype=np.float64)
        faces=np.asarray(z["faces"],dtype=np.int64)
    world=np.asarray(norm.center_xyz,dtype=np.float64)[None,:]+vn*float(norm.half_extent)
    md=dict(surface.metadata or {})
    base=compact_inverse(world,faces,divisions=int(md["compact_voxel_divisions"]),component_aware=bool(md.get("component_aware_compaction",False)))
    repaired=refine_to_fixed_point(world,faces,base)

    base_cp=compact_points(world,base)
    rep_cp=compact_points(world,repaired)
    _bcf,bbad=bad_edges(faces,base)
    _rcf,rbad=bad_edges(faces,repaired)

    # Global PCA view from original compact points.
    base2,mu,basis=pca2(base_cp)
    rep2=project(rep_cp,mu,basis)
    bad_mid=np.asarray([(base_cp[u]+base_cp[v])*0.5 for u,v in bbad],dtype=np.float64)
    bad2=project(bad_mid,mu,basis) if len(bad_mid) else np.zeros((0,2))
    rmid=np.asarray([(rep_cp[u]+rep_cp[v])*0.5 for u,v in rbad],dtype=np.float64)
    rbad2=project(rmid,mu,basis) if len(rmid) else np.zeros((0,2))

    fig,axs=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    axs[0].scatter(base2[:,0],base2[:,1],s=2,alpha=0.25)
    if len(bad2):
        axs[0].scatter(bad2[:,0],bad2[:,1],s=16,marker="x")
    axs[0].set_title(f"Before: {len(bbad)} non-manifold compact edges")
    axs[0].set_aspect("equal",adjustable="box")
    axs[0].axis("off")
    axs[1].scatter(rep2[:,0],rep2[:,1],s=2,alpha=0.25)
    if len(rbad2):
        axs[1].scatter(rbad2[:,0],rbad2[:,1],s=16,marker="x")
    axs[1].set_title(f"After: {len(rbad)} non-manifold compact edges")
    axs[1].set_aspect("equal",adjustable="box")
    axs[1].axis("off")
    fig.suptitle("Knight Stage14/15 compaction topology repair — actual compact geometry")
    fig.savefig(out/"global_before_after.png",dpi=180)
    plt.close(fig)

    # Pick representative incidence-4 bad edge with >=4 disconnected original dense edges.
    edge=None; pre=None
    for e,ids in sorted(bbad.items(), key=lambda kv:(-len(kv[1]),kv[0])):
        pairs=representative_preimage(faces,base,e)
        if len(pairs)>=4:
            edge=e;pre=pairs
            break
    if edge is None:
        raise RuntimeError("NO_REPRESENTATIVE_BAD_EDGE")

    vids=sorted({x for p in pre for x in p})
    local=world[vids]
    local2,lmu,lbasis=pca2(local)
    loc={vid:local2[i] for i,vid in enumerate(vids)}
    A,B=edge
    beforeA=project(base_cp[[A]],lmu,lbasis)[0]
    beforeB=project(base_cp[[B]],lmu,lbasis)[0]

    fig,axs=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    ax=axs[0]
    for i,(u,v) in enumerate(pre[:12]):
        pu,pv=loc[u],loc[v]
        ax.plot([pu[0],pv[0]],[pu[1],pv[1]],linewidth=2,alpha=0.8)
        ax.scatter([pu[0],pv[0]],[pu[1],pv[1]],s=25)
    ax.plot([beforeA[0],beforeB[0]],[beforeA[1],beforeB[1]],linewidth=5,alpha=0.9)
    ax.scatter([beforeA[0],beforeB[0]],[beforeA[1],beforeB[1]],s=80)
    ax.set_title(f"Before: {len(pre)} distinct dense edges collapse to ONE compact edge")
    ax.set_aspect("equal",adjustable="box"); ax.axis("off")

    ax=axs[1]
    # Show same dense edges, but their repaired compact endpoint IDs/positions.
    seen=set()
    for u,v in pre[:12]:
        ru,rv=int(repaired[u]),int(repaired[v])
        pu,pv=project(rep_cp[[ru,rv]],lmu,lbasis)
        ax.plot([pu[0],pv[0]],[pu[1],pv[1]],linewidth=2,alpha=0.8)
        ax.scatter([pu[0],pv[0]],[pu[1],pv[1]],s=35)
        seen.add((min(ru,rv),max(ru,rv)))
    ax.set_title(f"After: same dense evidence maps to {len(seen)} separate compact edges")
    ax.set_aspect("equal",adjustable="box"); ax.axis("off")
    fig.suptitle("Representative compaction-collapse repair — actual Knight dense-edge preimage")
    fig.savefig(out/"representative_collapse.png",dpi=220)
    plt.close(fig)

    report={
      "schema":"RealSaS.KnightCompactionTopologyRepairVisualization.v1",
      "base_nonmanifold_edges":len(bbad),
      "repaired_nonmanifold_edges":len(rbad),
      "base_nodes":int(base.max())+1,
      "repaired_nodes":int(repaired.max())+1,
      "representative_base_edge":list(map(int,edge)),
      "representative_dense_preimage_edge_count":len(pre),
      "representative_repaired_compact_edge_count":len(seen),
      "outputs":["global_before_after.png","representative_collapse.png"],
    }
    (out/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TOPOLOGY_REPAIR_VIZ="+json.dumps(report,sort_keys=True))

if __name__=="__main__":
    main()
