from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.preproduct_authority_v1 import signed_zero_surface_from_dict
from tools.audit_knight_topology_local_voxel_compaction_v1 import compact_metrics
from tools.audit_knight_vertex_link_fan_refinement_v1 import (
    loadj,compact_faces,link_components,refine_bad_vertex_links,
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--max-iterations",type=int,default=12)
    a=ap.parse_args();rr=a.run_root.resolve()

    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        faces=np.asarray(z["faces"],dtype=np.int64)
        verts=np.asarray(z["vertices_normalized"],dtype=np.float64)
    with np.load(a.inverse_npz,allow_pickle=False) as z:
        inv=np.asarray(z["final_inverse"],dtype=np.int64)
        base=np.asarray(z["base_inverse"],dtype=np.int64)

    current=inv.copy()
    history=[]
    for it in range(int(a.max_iterations)):
        before_cf=compact_faces(faces,current)
        before_bad=link_components(before_cf,int(current.max())+1)
        before_metrics=compact_metrics(verts,faces,current)
        if not before_bad:
            break
        nxt,meta=refine_bad_vertex_links(faces,current)
        after_cf=compact_faces(faces,nxt)
        after_bad=link_components(after_cf,int(nxt.max())+1)
        after_metrics=compact_metrics(verts,faces,nxt)
        history.append({
          "iteration":it,
          "before_nodes":int(current.max())+1,
          "before_illegal_vertex_links":len(before_bad),
          "before_nonmanifold_edges":before_metrics["nonmanifold_edge_count"],
          "after_nodes":int(nxt.max())+1,
          "after_illegal_vertex_links":len(after_bad),
          "after_nonmanifold_edges":after_metrics["nonmanifold_edge_count"],
          "meta":meta,
        })
        if meta["multi_fan_dense_vertex_count"]!=0:
            current=nxt
            break
        if np.array_equal(nxt,current):
            current=nxt
            break
        current=nxt

    final_cf=compact_faces(faces,current)
    final_bad=link_components(final_cf,int(current.max())+1)
    final_metrics=compact_metrics(verts,faces,current)

    pairs=np.unique(np.column_stack((base,current)),axis=0)
    seen={};cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b):cross+=1

    report={
      "schema":"RealSaS.KnightIterativeVertexLinkRefinementCourt.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
      "initial_nodes":int(inv.max())+1,
      "history":history,
      "final_nodes":int(current.max())+1,
      "final_illegal_vertex_links":len(final_bad),
      "final_nonmanifold_edges":final_metrics["nonmanifold_edge_count"],
      "cross_base_merge_count":cross,
      "pure_refinement":cross==0,
      "multi_fan_dense_vertex_count_max":max(
        [int(row["meta"]["multi_fan_dense_vertex_count"]) for row in history] or [0]
      ),
      "success":bool(
        len(final_bad)==0 and final_metrics["nonmanifold_edge_count"]==0
        and cross==0
        and max([int(row["meta"]["multi_fan_dense_vertex_count"]) for row in history] or [0])==0
      ),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_ITERATIVE_VERTEX_LINK_REFINEMENT="+json.dumps(report,sort_keys=True),flush=True)
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__":main()
