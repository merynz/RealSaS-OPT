from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.motion_compile_v2 import _tree
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)


def _ctx(authority_root:Path,run_id:str)->dict:
    root=(authority_root/"runs"/run_id).resolve()
    return {
        "repo_root":Path(".").resolve(),
        "authority_root":authority_root.resolve(),
        "run_root":root,
        "run_id":run_id,
        "run_manifest_path":root/"run_manifest.json",
        "run_manifest":json.loads((root/"run_manifest.json").read_text()),
        "stage":{"id":"MOTION_PAIRWISE_RETARGET_DIAGNOSTIC"},
        "ledger":json.loads((root/"ACTIVE_RUN_V2.json").read_text()),
    }


def _unary_cost(tf,sf,*,target_root:bool,source_root:bool)->float:
    c=2.0*float(np.linalg.norm(tf[:3]-sf[:3]))
    c+=0.45*abs(float(tf[3]-sf[3]))
    c+=0.35*abs(float(tf[4]-sf[4]))
    c+=0.35*abs(float(tf[5]-sf[5]))
    if target_root!=source_root:
        c+=1000.0
    if (
        abs(float(tf[0]))>0.05
        and abs(float(sf[0]))>0.05
        and math.copysign(1.0,float(tf[0]))
        !=math.copysign(1.0,float(sf[0]))
    ):
        c+=4.0
    return c


def _edge_cost(tv:np.ndarray,sv:np.ndarray,tscale:float,sscale:float)->tuple[float,float,float]:
    tn=float(np.linalg.norm(tv)); sn=float(np.linalg.norm(sv))
    if tn<=1e-9 or sn<=1e-9:
        return 8.0,-1.0,8.0
    cosine=float(np.clip(np.dot(tv,sv)/(tn*sn),-1.0,1.0))
    tnorm=max(tn/tscale,1e-9)
    snorm=max(sn/sscale,1e-9)
    log_ratio=abs(math.log(snorm/tnorm))
    # Direction dominates; normalized length rejects distant cross-body matches.
    pair=(1.0-cosine)+0.35*min(log_ratio,4.0)
    return pair,cosine,log_ratio


def _solve(
    *,
    tids:list[str],
    tpar:dict[str,str|None],
    tpos:dict[str,np.ndarray],
    tfeat:dict[str,np.ndarray],
    troot:str,
    sids:list[str],
    spos:dict[str,np.ndarray],
    sfeat:dict[str,np.ndarray],
    sroot:str,
    top_k:int,
    pair_weight:float,
):
    ti={jid:i for i,jid in enumerate(tids)}
    si={jid:i for i,jid in enumerate(sids)}
    tscale=max(float(np.linalg.norm(p-tpos[troot])) for p in tpos.values())
    sscale=max(float(np.linalg.norm(p-spos[sroot])) for p in spos.values())
    unary=np.zeros((len(tids),len(sids)),dtype=np.float64)
    for i,tid in enumerate(tids):
        for j,sid in enumerate(sids):
            unary[i,j]=_unary_cost(
                tfeat[tid],sfeat[sid],
                target_root=tid==troot,
                source_root=sid==sroot,
            )

    allowed:dict[int,list[int]]={}
    for i,tid in enumerate(tids):
        if tid==troot:
            allowed[i]=[si[sroot]]
            continue
        order=np.argsort(unary[i],kind="stable")
        allowed[i]=[int(j) for j in order[:min(top_k,len(order))] if sids[int(j)]!=sroot]
        if not allowed[i]:
            raise RuntimeError("PAIRWISE_RETARGET_ALLOWED_EMPTY")

    # x variables only for allowed target/source candidates.
    x_index={}
    var_count=0
    for i in range(len(tids)):
        for j in allowed[i]:
            x_index[(i,j)]=var_count
            var_count+=1

    edges=[(ti[p],ti[c]) for c,p in tpar.items() if p is not None]
    y_index={}
    pair_meta={}
    for e,(pi,ci) in enumerate(edges):
        tv=tpos[tids[ci]]-tpos[tids[pi]]
        for sj in allowed[pi]:
            for sk in allowed[ci]:
                if sj==sk:
                    continue
                sv=spos[sids[sk]]-spos[sids[sj]]
                pc,cos,lr=_edge_cost(tv,sv,tscale,sscale)
                y_index[(e,sj,sk)]=var_count
                pair_meta[(e,sj,sk)]=(pc,cos,lr)
                var_count+=1

    objective=np.zeros(var_count,dtype=np.float64)
    for (i,j),v in x_index.items():
        objective[v]=float(unary[i,j])
    for key,v in y_index.items():
        objective[v]=float(pair_weight)*float(pair_meta[key][0])
    objective+=1e-11*np.arange(var_count,dtype=np.float64)

    rows=[]; cols=[]; data=[]; lo=[]; hi=[]; r=0
    def add(coeffs,lower,upper):
        nonlocal r
        for c,val in coeffs:
            rows.append(r); cols.append(c); data.append(float(val))
        lo.append(float(lower)); hi.append(float(upper)); r+=1

    # Every target gets one source.
    for i in range(len(tids)):
        add([(x_index[(i,j)],1.0) for j in allowed[i]],1.0,1.0)
    # Injective source use.
    for j in range(len(sids)):
        vars_=[x_index[(i,j)] for i in range(len(tids)) if (i,j) in x_index]
        if vars_:
            add([(v,1.0) for v in vars_],-np.inf,1.0)

    # Linearize y = x_parent_j AND x_child_k for every target edge.
    for e,(pi,ci) in enumerate(edges):
        for sj in allowed[pi]:
            for sk in allowed[ci]:
                if sj==sk:
                    continue
                y=y_index[(e,sj,sk)]
                xp=x_index[(pi,sj)]
                xc=x_index[(ci,sk)]
                add([(y,1.0),(xp,-1.0)],-np.inf,0.0)
                add([(y,1.0),(xc,-1.0)],-np.inf,0.0)
                add([(y,1.0),(xp,-1.0),(xc,-1.0)],-1.0,np.inf)

    A=coo_matrix(
        (np.asarray(data,dtype=np.float64),(rows,cols)),
        shape=(r,var_count),
    ).tocsr()
    result=milp(
        c=objective,
        integrality=np.ones(var_count,dtype=np.int8),
        bounds=Bounds(np.zeros(var_count),np.ones(var_count)),
        constraints=LinearConstraint(
            A,np.asarray(lo,dtype=np.float64),np.asarray(hi,dtype=np.float64)
        ),
        options={"presolve":True,"mip_rel_gap":0.0},
    )
    if not bool(result.success) or result.x is None:
        raise RuntimeError("PAIRWISE_RETARGET_MILP_FAIL")
    x=np.asarray(result.x,dtype=np.float64)
    mapping={}
    for i,tid in enumerate(tids):
        chosen=[j for j in allowed[i] if x[x_index[(i,j)]]>0.5]
        if len(chosen)!=1:
            raise RuntimeError("PAIRWISE_RETARGET_NONINTEGRAL")
        mapping[tid]=sids[chosen[0]]

    edge_rows=[]
    cosines=[]
    for e,(pi,ci) in enumerate(edges):
        ps=mapping[tids[pi]]; cs=mapping[tids[ci]]
        tv=tpos[tids[ci]]-tpos[tids[pi]]
        sv=spos[cs]-spos[ps]
        pc,cos,lr=_edge_cost(tv,sv,tscale,sscale)
        cosines.append(cos)
        edge_rows.append({
            "target_parent":tids[pi],
            "target_child":tids[ci],
            "source_parent_correspondence":ps,
            "source_child_correspondence":cs,
            "direction_cosine":cos,
            "normalized_length_log_ratio":lr,
            "pair_cost":pc,
        })
    unary_total=sum(unary[ti[tid],si[sid]] for tid,sid in mapping.items())
    pair_total=sum(row["pair_cost"] for row in edge_rows)
    return {
        "mapping":mapping,
        "objective":float(result.fun),
        "unary_total":float(unary_total),
        "pair_total_unweighted":float(pair_total),
        "edge_rows":edge_rows,
        "mean_direction_cosine":float(np.mean(cosines)),
        "p10_direction_cosine":float(np.quantile(cosines,0.10)),
        "negative_direction_edge_count":int(sum(c<0 for c in cosines)),
        "top_k":int(top_k),
        "pair_weight":float(pair_weight),
        "variable_count":int(var_count),
        "constraint_count":int(r),
    }


def diagnose(*,repo_root:Path,authority_root:Path,run_id:str,clip_id:str,bone_role_report:Path)->dict:
    ctx=_ctx(authority_root,run_id); ctx["repo_root"]=repo_root.resolve()
    skeleton=qualified_skeleton_from_dict(
        stage_output_payload(ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1")
    )
    clip=json.loads(
        (ctx["run_root"]/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json").read_text()
    )
    source_rows=tuple(clip.get("source_skeleton") or ())
    all_sids,spar,schildren,spos,sfeat,sroot=_tree(
        source_rows,id_key="source_joint_id",parent_key="parent_source_joint_id",pos_key="rest_position"
    )
    target_rows=tuple({
        "joint_id":j.canonical_joint_id,
        "parent_id":j.parent_canonical_id,
        "position":j.position,
    } for j in skeleton.joints)
    tids,tpar,tchildren,tpos,tfeat,troot=_tree(
        target_rows,id_key="joint_id",parent_key="parent_id",pos_key="position"
    )
    roles=json.loads(bone_role_report.read_text())
    role_by={str(r["source_joint_id"]):r for r in roles.get("bone_roles") or ()}
    sids=[
        sid for sid in all_sids
        if sid==sroot or bool((role_by.get(sid) or {}).get("has_mesh_influence"))
    ]
    excluded=[sid for sid in all_sids if sid not in sids]
    if len(sids)<len(tids):
        raise RuntimeError("PAIRWISE_RETARGET_CANDIDATE_COUNT_TOO_SMALL")

    configs=[]
    for top_k in (8,12):
        for pair_weight in (0.5,1.0,2.0):
            row=_solve(
                tids=tids,tpar=tpar,tpos=tpos,tfeat=tfeat,troot=troot,
                sids=sids,spos=spos,sfeat=sfeat,sroot=sroot,
                top_k=top_k,pair_weight=pair_weight,
            )
            configs.append(row)
    configs.sort(
        key=lambda x:(
            x["negative_direction_edge_count"],
            -x["p10_direction_cosine"],
            -x["mean_direction_cosine"],
            x["unary_total"],
        )
    )
    best=configs[0]
    return {
        "schema":"RealSaS.MotionPairwiseRestGeometryRetargetDiagnostic.v1",
        "status":"MEASURED",
        "run_id":run_id,
        "clip_id":clip_id,
        "candidate_rule":"MESH_INFLUENCING_OR_UNIQUE_SOURCE_ROOT",
        "excluded_source_joint_ids":excluded,
        "target_joint_count":len(tids),
        "source_candidate_count":len(sids),
        "selection_rule":"MIN_NEGATIVE_EDGES__MAX_P10_COS__MAX_MEAN_COS__MIN_UNARY",
        "best_configuration":best,
        "configurations":[{
            k:v for k,v in row.items() if k not in {"mapping","edge_rows"}
        } for row in configs],
        "diagnostic_only":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",default=".")
    p.add_argument("--authority-root",required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--clip-id",default="demo_idle_v1")
    p.add_argument("--bone-role-report",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    report=diagnose(
        repo_root=Path(a.repo_root),
        authority_root=Path(a.authority_root),
        run_id=a.run_id,
        clip_id=a.clip_id,
        bone_role_report=Path(a.bone_role_report),
    )
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    b=report["best_configuration"]
    print("PAIRWISE_RETARGET_DIAGNOSTIC",json.dumps({
        "excluded_source_joint_ids":report["excluded_source_joint_ids"],
        "top_k":b["top_k"],
        "pair_weight":b["pair_weight"],
        "negative_direction_edge_count":b["negative_direction_edge_count"],
        "p10_direction_cosine":b["p10_direction_cosine"],
        "mean_direction_cosine":b["mean_direction_cosine"],
        "unary_total":b["unary_total"],
        "pair_total_unweighted":b["pair_total_unweighted"],
        "variable_count":b["variable_count"],
        "constraint_count":b["constraint_count"],
    },sort_keys=True))


if __name__=="__main__":
    main()
