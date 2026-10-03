"""Audit-only minimum-correction skin/carrier compatibility projection sweep."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, qualified_skeleton_from_dict,
    qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients
from compiler.realsas_compiler_core.types import QualifiedSkinIR, QualifiedSkinRow


LAMBDAS=(0.0,0.03,0.1,0.3,1.0,3.0,10.0)

def read(path): return json.loads(Path(path).read_text())
def write(path,payload):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")

def unique_edges(candidate, vertex_index):
    edges=set()
    for face in candidate.faces:
        a,b,c=(vertex_index[x] for x in face)
        edges.add(tuple(sorted((a,b)))); edges.add(tuple(sorted((b,c)))); edges.add(tuple(sorted((c,a))))
    return np.asarray(sorted(edges),dtype=np.int64)

def q(a,p): return float(np.quantile(np.asarray(a,dtype=np.float64),p))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--surface-json",type=Path,required=True)
    ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--fresh-inference-dir",type=Path,required=True)
    ap.add_argument("--out-root",type=Path,required=True)
    a=ap.parse_args()
    surface=rigging_surface_from_dict(read(a.surface_json))
    candidate=canonical_mesh_candidate_from_dict(read(a.candidate_json))
    skeleton=qualified_skeleton_from_dict(read(a.fresh_inference_dir/"fresh_qualified_skeleton.json"))
    skin=qualified_skin_from_dict(read(a.fresh_inference_dir/"fresh_qualified_skin.json"))

    sids=tuple(str(n.surface_id) for n in surface.surface_nodes); si={sid:i for i,sid in enumerate(sids)}
    if len(si)!=len(sids): raise RuntimeError("PROJECTION_SURFACE_ID_DUPLICATE")
    jids=tuple(j.canonical_joint_id for j in skeleton.joints); ji={jid:i for i,jid in enumerate(jids)}
    source={row.surface_id:row for row in skin.rows}
    if set(source)!=set(sids): raise RuntimeError("PROJECTION_SKIN_SURFACE_ACCOUNTING_MISMATCH")
    W0=np.zeros((len(sids),len(jids)),dtype=np.float64)
    for r,sid in enumerate(sids):
        for jid,w in source[sid].influences:
            if jid not in ji: raise RuntimeError("PROJECTION_UNKNOWN_JOINT")
            W0[r,ji[jid]]=float(w)
    if not np.allclose(W0.sum(1),1.0,atol=1e-8,rtol=0): raise RuntimeError("PROJECTION_SOURCE_NOT_SIMPLEX")

    vertices=tuple(candidate.vertices); vi={v.candidate_vertex_id:i for i,v in enumerate(vertices)}
    prow=[]; pcol=[]; pdata=[]
    for r,v in enumerate(vertices):
        total=0.0
        for sid,c in _skin_support_coefficients(v):
            if sid not in si: raise RuntimeError("PROJECTION_VERTEX_SUPPORT_UNKNOWN")
            c=float(c); total+=c; prow.append(r); pcol.append(si[sid]); pdata.append(c)
        if abs(total-1.0)>1e-9: raise RuntimeError("PROJECTION_VERTEX_SUPPORT_NOT_SIMPLEX")
    P=sparse.csr_matrix((pdata,(prow,pcol)),shape=(len(vertices),len(sids)),dtype=np.float64)
    edges=unique_edges(candidate,vi)
    rest=np.asarray([v.P for v in vertices],dtype=np.float64)
    span=float(np.max(rest.max(0)-rest.min(0))); lengths=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    if span<=1e-12 or np.any(lengths<=1e-12): raise RuntimeError("PROJECTION_INVALID_CARRIER_GEOMETRY")
    lnorm=lengths/span; median=float(np.median(lnorm))
    edge_weight=median/lnorm

    er=np.repeat(np.arange(len(edges)),2)
    ec=edges.reshape(-1)
    ed=np.tile(np.asarray([1.0,-1.0]),len(edges))
    C=sparse.csr_matrix((ed,(er,ec)),shape=(len(edges),len(vertices)),dtype=np.float64)
    Q=C@P
    D=sparse.diags(edge_weight,format="csr")
    K=(Q.T@D@Q).tocsr()
    I=sparse.eye(len(sids),format="csr",dtype=np.float64)

    base_vertex=P@W0
    base_delta=np.abs(base_vertex[edges[:,0]]-base_vertex[edges[:,1]]).sum(1)
    manifest={"schema":"RealSaS.SkinCarrierProjectionSweep.v1","status":"DIAGNOSTIC_ONLY",
        "product_authority_minted":False,"training_used":False,
        "objective":"L2_TO_MODEL_PLUS_INVERSE_LENGTH_WEIGHTED_CARRIER_EDGE_DIRICHLET",
        "lambda_values":list(LAMBDAS),"surface_node_count":len(sids),"carrier_vertex_count":len(vertices),
        "carrier_edge_count":len(edges),"body_span":span,"median_normalized_edge_length":median,
        "source_skin_lineage_hash":skin.skin_lineage_hash,"source_gradient_p99":q(base_delta/lnorm,.99),
        "rows":[]}
    a.out_root.mkdir(parents=True,exist_ok=True)
    for lam in LAMBDAS:
        if lam==0.0:
            W=W0.copy()
        else:
            W=np.asarray(spsolve((I+float(lam)*K).tocsc(),W0),dtype=np.float64)
        if W.shape!=W0.shape or not np.isfinite(W).all(): raise RuntimeError(f"PROJECTION_SOLVE_INVALID:{lam}")
        negative_mass=float(-np.minimum(W,0.0).sum())
        W=np.maximum(W,0.0)
        row_sum=W.sum(1,keepdims=True)
        if np.any(row_sum<=1e-12): raise RuntimeError(f"PROJECTION_ZERO_ROW:{lam}")
        W/=row_sum
        correction=np.abs(W-W0).sum(1)
        vertex=P@W
        delta=np.abs(vertex[edges[:,0]]-vertex[edges[:,1]]).sum(1)
        grad=delta/lnorm
        tag=("0" if lam==0 else str(lam).replace(".","p"))
        out=a.out_root/f"lambda_{tag}"; out.mkdir(parents=True,exist_ok=True)

        qrows=tuple(QualifiedSkinRow(
            surface_id=sid,
            influences=tuple((jid,float(W[r,c])) for c,jid in enumerate(jids)),
            simplex_residual_before=0.0,
            correction_l1=float(correction[r]),
        ) for r,sid in enumerate(sids))
        report={"status":"DIAGNOSTIC_CARRIER_COMPATIBILITY_PROJECTION","product_authority_minted":False,
            "source_skin_lineage_hash":skin.skin_lineage_hash,"lambda":float(lam),
            "objective":"L2_TO_MODEL_PLUS_INVERSE_LENGTH_WEIGHTED_CARRIER_EDGE_DIRICHLET",
            "negative_mass_clipped":negative_mass,"mean_row_correction_l1":float(np.mean(correction)),
            "p95_row_correction_l1":q(correction,.95),"max_row_correction_l1":float(np.max(correction)),
            "carrier_gradient_p95":q(grad,.95),"carrier_gradient_p99":q(grad,.99),
            "carrier_gradient_p999":q(grad,.999),"carrier_gradient_max":float(np.max(grad))}
        provisional=QualifiedSkinIR(qrows,surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,report,"")
        payload=provisional.to_dict(); payload.pop("skin_lineage_hash",None)
        qskin=QualifiedSkinIR(qrows,surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,report,content_sha256(payload))
        write(out/"fresh_qualified_skin.json",qskin.to_dict())
        shutil.copy2(a.fresh_inference_dir/"fresh_qualified_skeleton.json",out/"fresh_qualified_skeleton.json")
        shutil.copy2(a.fresh_inference_dir/"rig_REPORT.json",out/"rig_REPORT.json")
        write(out/"skin_REPORT.json",report)
        write(out/"PROJECTION_STATS.json",{**report,"skin_lineage_hash":qskin.skin_lineage_hash})
        manifest["rows"].append({"lambda":float(lam),**report,"skin_lineage_hash":qskin.skin_lineage_hash})
        print("PROJECTION_PREP="+json.dumps({"lambda":lam,"p95_corr":report["p95_row_correction_l1"],
            "max_corr":report["max_row_correction_l1"],"grad_p99":report["carrier_gradient_p99"],
            "grad_max":report["carrier_gradient_max"]},sort_keys=True),flush=True)
    write(a.out_root/"PROJECTION_MANIFEST.json",manifest)

if __name__=="__main__": main()
