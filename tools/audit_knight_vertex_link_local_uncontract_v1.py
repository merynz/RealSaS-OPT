from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import signed_zero_surface_from_dict
from tools.audit_knight_topology_local_voxel_compaction_v1 import compact_metrics
from tools.audit_knight_vertex_link_fan_refinement_v1 import (
    loadj,compact_faces,link_components,
)

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
    bad=link_components(before_cf,int(inv.max())+1)
    out=inv.copy()
    next_id=int(out.max())+1
    released=[]
    for q in sorted(bad):
        dense=np.nonzero(inv==int(q))[0]
        for vid in dense.tolist():
            out[int(vid)]=next_id
            next_id+=1
            released.append(int(vid))
    _,out=np.unique(out,return_inverse=True)
    out=np.asarray(out,dtype=np.int64)

    after_cf=compact_faces(faces,out)
    after_bad=link_components(after_cf,int(out.max())+1)
    metrics=compact_metrics(verts,faces,out)
    pairs=np.unique(np.column_stack((base,out)),axis=0)
    seen={};cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b):cross+=1

    report={
      "schema":"RealSaS.KnightVertexLinkLocalUncontractCourt.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
      "bad_vertex_count_before":len(bad),
      "bad_vertices_before":sorted(int(x) for x in bad),
      "nodes_before":int(inv.max())+1,
      "released_dense_vertex_count":len(released),
      "nodes_after":int(out.max())+1,
      "illegal_vertex_links_after":len(after_bad),
      "nonmanifold_edges_after":metrics["nonmanifold_edge_count"],
      "cross_base_merge_count":cross,
      "face_deletion":False,
      "teacher_or_subject_labels_used":False,
      "success":bool(len(after_bad)==0 and metrics["nonmanifold_edge_count"]==0 and cross==0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_VERTEX_LINK_LOCAL_UNCONTRACT="+json.dumps(report,sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
