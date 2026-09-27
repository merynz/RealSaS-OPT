from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.motion_compile_v2 import _tree
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)


def _context(authority_root: Path, run_id: str) -> dict:
    run_root=(authority_root/"runs"/run_id).resolve()
    return {
        "repo_root":Path(".").resolve(),
        "authority_root":authority_root.resolve(),
        "run_root":run_root,
        "run_id":run_id,
        "run_manifest_path":run_root/"run_manifest.json",
        "run_manifest":json.loads((run_root/"run_manifest.json").read_text()),
        "stage":{"id":"MOTION_DEFORM_CANDIDATE_RETARGET_DIAGNOSTIC"},
        "ledger":json.loads((run_root/"ACTIVE_RUN_V2.json").read_text()),
    }


def _cost_matrix(tids,tfeat,troot,sids,sfeat,sroot):
    cost=np.zeros((len(tids),len(sids)),dtype=np.float64)
    for i,tid in enumerate(tids):
        tf=tfeat[tid]
        for j,sid in enumerate(sids):
            sf=sfeat[sid]
            c=2.0*float(np.linalg.norm(tf[:3]-sf[:3]))
            c+=0.45*abs(float(tf[3]-sf[3]))
            c+=0.35*abs(float(tf[4]-sf[4]))
            c+=0.35*abs(float(tf[5]-sf[5]))
            if (tid==troot)!=(sid==sroot):
                c+=1000.0
            if (
                abs(float(tf[0]))>0.05
                and abs(float(sf[0]))>0.05
                and math.copysign(1.0,float(tf[0]))
                !=math.copysign(1.0,float(sf[0]))
            ):
                c+=4.0
            cost[i,j]=c
    return cost


def diagnose(
    *,
    repo_root:Path,
    authority_root:Path,
    run_id:str,
    clip_id:str,
    bone_role_report:Path,
)->dict:
    ctx=_context(authority_root,run_id)
    ctx["repo_root"]=repo_root.resolve()
    skeleton=qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    payload=json.loads(
        (
            ctx["run_root"]
            /"inputs"/"motion"/"quaternius_knight_v1"
            /f"{clip_id}.motion.json"
        ).read_text()
    )
    source_rows=tuple(payload.get("source_skeleton") or ())
    sids,spar,schildren,spos,sfeat,sroot=_tree(
        source_rows,
        id_key="source_joint_id",
        parent_key="parent_source_joint_id",
        pos_key="rest_position",
    )
    target_rows=tuple({
        "joint_id":j.canonical_joint_id,
        "parent_id":j.parent_canonical_id,
        "position":j.position,
    } for j in skeleton.joints)
    tids,tpar,tchildren,tpos,tfeat,troot=_tree(
        target_rows,
        id_key="joint_id",
        parent_key="parent_id",
        pos_key="position",
    )

    roles=json.loads(bone_role_report.read_text())
    role_by={
        str(row["source_joint_id"]):row
        for row in roles.get("bone_roles") or ()
    }
    candidates=[
        sid for sid in sids
        if sid==sroot or bool((role_by.get(sid) or {}).get("has_mesh_influence"))
    ]
    excluded=[sid for sid in sids if sid not in candidates]
    if sroot not in candidates or len(candidates)<len(tids):
        raise RuntimeError("MOTION_DEFORM_CANDIDATE_SET_TOO_SMALL")

    all_cost=_cost_matrix(tids,tfeat,troot,sids,sfeat,sroot)
    candidate_indices=[sids.index(sid) for sid in candidates]
    candidate_cost=all_cost[:,candidate_indices]
    tr,sc=linear_sum_assignment(candidate_cost)
    mapping={
        tids[int(ti)]:candidates[int(si)]
        for ti,si in zip(tr,sc)
    }
    if len(mapping)!=len(tids) or mapping.get(troot)!=sroot:
        raise RuntimeError("MOTION_DEFORM_CANDIDATE_ASSIGNMENT_INVALID")

    edge_rows=[]
    cosine_values=[]
    for tid in tids:
        parent=tpar[tid]
        if parent is None:
            continue
        sv_child=mapping[tid]
        sv_parent=mapping[parent]
        tv=np.asarray(tpos[tid])-np.asarray(tpos[parent])
        sv=np.asarray(spos[sv_child])-np.asarray(spos[sv_parent])
        tn=float(np.linalg.norm(tv)); sn=float(np.linalg.norm(sv))
        cosine=None
        if tn>1e-9 and sn>1e-9:
            cosine=float(np.dot(tv,sv)/(tn*sn))
            cosine_values.append(cosine)
        edge_rows.append({
            "target_parent":parent,
            "target_child":tid,
            "source_parent":sv_parent,
            "source_child":sv_child,
            "target_edge_length":tn,
            "source_correspondence_edge_length":sn,
            "direction_cosine":cosine,
            "raw_source_parent_of_child":spar[sv_child],
            "raw_source_parent_matches_correspondence":spar[sv_child]==sv_parent,
        })

    assignments=[]
    for tid in tids:
        i=tids.index(tid)
        row=candidate_cost[i]
        order=np.argsort(row,kind="stable")[:6]
        assignments.append({
            "target_joint_id":tid,
            "target_parent_id":tpar[tid],
            "target_position":list(map(float,tpos[tid])),
            "source_joint_id":mapping[tid],
            "source_position":list(map(float,spos[mapping[tid]])),
            "cost":float(row[candidates.index(mapping[tid])]),
            "six_lowest_candidate_sources":[
                {
                    "source_joint_id":candidates[int(k)],
                    "cost":float(row[int(k)]),
                }
                for k in order
            ],
        })

    return {
        "schema":"RealSaS.MotionDeformCandidateRetargetDiagnostic.v1",
        "status":"MEASURED",
        "run_id":run_id,
        "clip_id":clip_id,
        "target_joint_count":len(tids),
        "source_joint_count":len(sids),
        "candidate_source_joint_count":len(candidates),
        "candidate_rule":"MESH_INFLUENCING_OR_UNIQUE_SOURCE_ROOT",
        "candidate_source_joint_ids":candidates,
        "excluded_source_joint_ids":excluded,
        "source_root":sroot,
        "target_root":troot,
        "total_assignment_cost":float(sum(
            candidate_cost[int(i),int(j)] for i,j in zip(tr,sc)
        )),
        "mapping":assignments,
        "correspondence_edge_metrics":{
            "edge_count":len(edge_rows),
            "mean_direction_cosine":(
                float(np.mean(cosine_values)) if cosine_values else None
            ),
            "p10_direction_cosine":(
                float(np.quantile(cosine_values,0.10))
                if cosine_values else None
            ),
            "negative_direction_edge_count":int(
                sum(v<0 for v in cosine_values)
            ),
            "raw_parent_match_fraction":float(np.mean([
                row["raw_source_parent_matches_correspondence"]
                for row in edge_rows
            ])),
        },
        "edge_rows":edge_rows,
        "diagnostic_only":True,
    }


def main()->None:
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
    print("MOTION_DEFORM_CANDIDATE_RETARGET_DIAGNOSTIC",json.dumps({
        "candidate_source_joint_count":report["candidate_source_joint_count"],
        "excluded_source_joint_ids":report["excluded_source_joint_ids"],
        "total_assignment_cost":report["total_assignment_cost"],
        "correspondence_edge_metrics":report["correspondence_edge_metrics"],
    },sort_keys=True))


if __name__=="__main__":
    main()
