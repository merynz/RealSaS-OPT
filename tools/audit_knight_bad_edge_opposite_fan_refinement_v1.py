from __future__ import annotations
import argparse,json
from collections import defaultdict
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

def loadj(p): return json.loads(Path(p).read_text())

def collect_bad_edge_opposite_tokens(faces,inv,bad_codes,nclusters,chunk_faces=250_000):
    tokens=defaultdict(set)
    witness=defaultdict(list)
    f=np.asarray(faces,dtype=np.int64)
    inv=np.asarray(inv,dtype=np.int64)
    badset=np.asarray(bad_codes,dtype=np.int64)
    for start in range(0,len(f),chunk_faces):
        raw=f[start:start+chunk_faces]
        mapped=inv[raw]
        valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,1]!=mapped[:,2])&(mapped[:,2]!=mapped[:,0])
        if not np.any(valid):
            continue
        raw=raw[valid];mapped=mapped[valid]
        for ia,ib,io in ((0,1,2),(1,2,0),(2,0,1)):
            qa=mapped[:,ia];qb=mapped[:,ib];qo=mapped[:,io]
            lo=np.minimum(qa,qb).astype(np.int64)
            hi=np.maximum(qa,qb).astype(np.int64)
            codes=lo*np.int64(nclusters)+hi
            take=np.isin(codes,badset,assume_unique=False)
            rows=np.nonzero(take)[0]
            for k in rows.tolist():
                code=int(codes[k]);a=int(raw[k,ia]);b=int(raw[k,ib]);opp=int(qo[k])
                if int(qa[k])<=int(qb[k]):
                    left,right=a,b
                else:
                    left,right=b,a
                # Stronger generic invariant than disconnected edge-preimage alone:
                # endpoint equivalence may not alias distinct quotient face fans
                # around an over-incident quotient edge.
                tokens[left].add((code,"L",opp))
                tokens[right].add((code,"R",opp))
                witness[code].append((left,right,opp))
    return tokens,witness

def refine_by_tokens(faces,inv,tokens):
    sig_ids={}
    rows=np.empty(len(inv),dtype=np.int64)
    for vid in range(len(inv)):
        sig=tuple(sorted(tokens.get(int(vid),())))
        key=(int(inv[vid]),sig)
        rows[vid]=sig_ids.setdefault(key,len(sig_ids))
    rows,_=topology_local_refine(faces,rows)
    return np.asarray(rows,dtype=np.int64),{
        "tokenized_dense_vertex_count":int(len(tokens)),
        "signature_count":int(len(sig_ids)),
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    rr=a.run_root.resolve()
    zero=signed_zero_surface_from_dict(loadj(rr/"artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(loadj(rr/"artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json"))
    with np.load(Path(zero.npz_path),allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],np.float64);faces=np.asarray(z["faces"],np.int64)
    world=np.asarray(norm.center_xyz,np.float64)[None,:]+vn*float(norm.half_extent)

    # Exact smear-base compaction: division 56, component aware.
    base=compact_inverse(world,faces,divisions=56,component_aware=True)
    inv,first=topology_local_refine(faces,base)
    history=[{"phase":"TOPOLOGY_LOCAL","iteration":0,"metrics":compact_metrics(world,faces,inv)}]

    # Reproduce the existing best pure-refinement chain first.
    for it in range(1,20):
        before=compact_metrics(world,faces,inv)
        if before["nonmanifold_edge_count"]<=7:
            break
        nxt,meta=split_by_edge_preimage_components(faces,inv)
        after=compact_metrics(world,faces,nxt)
        history.append({"phase":"EDGE_PREIMAGE","iteration":it,"metrics":after,"meta":meta})
        inv=nxt
        if after["nonmanifold_edge_count"]>=before["nonmanifold_edge_count"] and meta["disconnected_bad_edge_count"]==0:
            break

    # New stronger generic fan refinement.
    for it in range(1,9):
        before=compact_metrics(world,faces,inv)
        if before["nonmanifold_edge_count"]==0:
            break
        bad_codes,bad_inc,n=bad_edge_codes(faces,inv)
        tokens,witness=collect_bad_edge_opposite_tokens(faces,inv,bad_codes,n)
        nxt,meta=refine_by_tokens(faces,inv,tokens)
        after=compact_metrics(world,faces,nxt)
        history.append({
            "phase":"BAD_EDGE_OPPOSITE_FAN_SIGNATURE",
            "iteration":it,
            "metrics":after,
            "meta":{
                **meta,
                "bad_edge_count_before":int(len(bad_codes)),
                "witness_rows":int(sum(len(v) for v in witness.values())),
                "bad_incidences":sorted(int(v) for v in bad_inc.values()),
            },
        })
        inv=nxt
        if after["nonmanifold_edge_count"]>=before["nonmanifold_edge_count"]:
            break

    final=compact_metrics(world,faces,inv)
    pairs=np.unique(np.column_stack((base,inv)),axis=0)
    seen={};cross=0
    for b,r in pairs:
        old=seen.setdefault(int(r),int(b))
        if old!=int(b):cross+=1

    report={
      "schema":"RealSaS.KnightBadEdgeOppositeFanRefinementCourt.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
      "base":compact_metrics(world,faces,base),
      "history":history,
      "final":final,
      "cross_base_merge_count":int(cross),
      "operator":{
        "id":"BAD_EDGE_OPPOSITE_FAN_SIGNATURE_REFINEMENT_V1",
        "generic_invariants":[
          "PURE_REFINEMENT_OF_EXISTING_COMPACTION",
          "QUOTIENT_EDGE_INCIDENCE_MUST_NOT_ALIAS_DISTINCT_LOCAL_FACE_FANS",
          "FACE_DELETION_FORBIDDEN",
          "TEACHER_OR_SUBJECT_LABELS_FORBIDDEN",
        ],
      },
      "success":bool(final["nonmanifold_edge_count"]==0 and cross==0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_BAD_EDGE_OPPOSITE_FAN_REFINEMENT="+json.dumps({
      "history":[[x["phase"],x["iteration"],x["metrics"]["nonmanifold_edge_count"],x["metrics"]["compact_node_count"]] for x in history],
      "final_nonmanifold":final["nonmanifold_edge_count"],
      "final_nodes":final["compact_node_count"],
      "cross_base_merge_count":cross,
      "success":report["success"],
    },sort_keys=True))
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__":main()
