#!/usr/bin/env python3
"""SHA-locked private V11 heldout source-face query planner, no Blender/labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
from tools.corpus.corpus370_adaptive_edit_probe_plan_v11 import build_adaptive_plan

SCHEMA="RealSaS.Corpus370HeldoutSourceBranchProbePrivateRunner.v11"
V9_SEAL="RealSaS.Corpus370NativeAnimatedPartitionRealSourceResearchResult.v9"
V10_SEAL="RealSaS.Corpus370NativeEditResponseEquivalenceRealResult.v10"
JOIN_SHA="e72f3380c2f34cbf3e088218ebfcd4be0f4aa71c9138d4b3d1865a4f4074439a"
GROUP_SHA="1d8b8e2e8018d2cd3b94188f2683424659a8b0735f43c43eb381b6b252a3b8a1"
RAYS_SHA="0eeb24089322d0802567ba9736ada1de03e234bbf133bc91164132a80bec3c70"

def require(ok,reason):
    if not ok:raise ValueError(reason)
def sha(p):
    with Path(p).open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()
def checked(p,digest,code):
    p=Path(p).resolve()
    require(p.is_file() and sha(p)==digest,code)
    return json.loads(p.read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n")

def run(args):
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    result={
        "schema":SCHEMA,"status":"FAIL_HELDOUT_SOURCE_PROBE_NOT_CLOSED",
        "worker":args.subject_ordinal,
        "graphics_owner_authorized":False,"training_authorized":False
    }
    try:
        v9=checked(args.public_v9,args.public_v9_sha256,"V11_REAL_V9_PUBLIC_SHA")
        v10=checked(args.public_v10,args.public_v10_sha256,"V11_REAL_V10_PUBLIC_SHA")
        require(v9.get("schema")==V9_SEAL and v10.get("schema")==V10_SEAL
                and v9.get("authority",{}).get("training_authorized") is False
                and v10.get("interpretation",{}).get("training_authorized") is False
                and v10.get("v9_public_seal_sha256")==sha(args.public_v9)
                and v9["hosted"]["jobs_succeeded"]==4
                and v10["github_actions_jobs_succeeded"]==4
                and v10["independent_court_json_sha256_verified_count"]==3
                and len(v9["per_subject"])==len(v10["per_subject"])==3,
                "V11_NO_CORRUPTED_OR_FAKE_SOURCE_V9_V10_SEALS")
        labels=["CAT","TRICERATOPS","COWFISH"]
        sid=labels[args.subject_ordinal]
        v9row=next(x for x in v9["per_subject"] if x["label"]==sid)
        v10row=next(x for x in v10["per_subject"] if x["source_label"]==sid)
        require(v9row["subject_id"]==v10row["source_id"]
                and v9row["source_sha256"]==v10row["source_sha256"],
                "V11_REAL_SOURCE_SUBJECT_JOIN")
        subject=v9row["subject_id"]
        join=checked(Path(args.original_join_root)/
            "CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json",
            JOIN_SHA,"V11_ORIGINAL_NATIVE_SOURCE_HIT_JOIN")
        groups=checked(Path(args.allview_groups_root)/
            "CORPUS370_ALLVIEW_GEOMETRY_GROUPS_PRIVATE_BATCH_V3.json",
            GROUP_SHA,"V11_ORIGINAL_SOURCE_GEOMETRY_PROPOSAL_BATCH")
        rays=checked(args.rays_index,RAYS_SHA,
                     "V11_ORIGINAL_REST_RAYS_INDEX")
        require(subject in join["subjects"]
                and subject in groups["subjects"]
                and subject in rays["subject_results"]
                and join["subjects"][subject]["source_sha256"]==
                     v9row["source_sha256"]==
                     groups["subjects"][subject]["source_sha256"],
                "V11_SOURCE_8_VIEW_GEOMETRY_COHORT")
        pair_ref=join["subjects"][subject]["pair_diagnostic_receipt"]
        pairs=checked(pair_ref["path"],pair_ref["sha256"],
                      "V11_ORIGINAL_SOURCE_HIT_PAIR_SHA")
        group_proposal_path=Path(args.allview_groups_root)/"subjects"/subject/(
            "ALLVIEW_ANONYMOUS_GROUP_CANDIDATES_V3.json")
        proposal=checked(group_proposal_path,
            groups["subjects"][subject]["proposal_sha256"],
            "V11_ORIGINAL_GROUP_HYPOTHESIS_SHA")
        ray_ref=rays["subject_results"][subject]["receipt"]
        ray=checked(ray_ref["path"],ray_ref["sha256"],
                    "V11_ORIGINAL_RAY_SOURCE_COURT_SHA")
        rest_ref=ray["source_rest_geometry"]
        require(rest_ref["sha256"]==pairs["parents"]["rest_geometry_sha256"],
                "V11_SOURCE_PAIRED_ORIGINAL_REST_PROOF_SHA")
        rest_path=Path(rest_ref["path"]).resolve()
        require(rest_path.is_file() and sha(rest_path)==rest_ref["sha256"],
                "V11_SOURCE_REST_ORIGINAL_NPZ_SHA")
        v9path=Path(args.v9_private_root)/str(args.subject_ordinal)/(
            "PRIVATE_ANIMATED_SOURCE_RENDER_MANIFEST_V9.json")
        v10path=Path(args.v10_private_root)/str(args.subject_ordinal)/(
            "PRIVATE_NATIVE_EDIT_EQUIVALENCE_COURT_V10.json")
        old=checked(v9path,v9row["render_manifest_sha256"],
                    "V11_ORIGINAL_REAL_NATIVE_SOURCE_MANIFEST_SHA")
        v10court=checked(v10path,v10row["court_sha256"],
                         "V11_INDEPENDENT_REAL_EDIT_RESPONSE_COURT_SHA")
        require(v10court["parents"]["private_v9_render_manifest_sha256"]==
                    sha(v9path)
                and v10court["parents"]["public_v9_result_sha256"]==
                    sha(args.public_v9)
                and v10court["source_sha256"]==proposal["source_sha256"]
                and v10court["graphics_owner_authorized"] is False
                and v10court["training_authorized"] is False,
                "V11_SOURCE_ORIGINAL_V9_V10_PRIVATE_PARENT_LINKS")
        with np.load(rest_path,allow_pickle=False) as npz:
            require({"rest_vertices_source_world","faces"}<=set(npz.files),
                    "V11_ORIGINAL_REST_NPZ_KEYS")
            rest={k:np.array(npz[k]) for k in
                  ("rest_vertices_source_world","faces")}
        plan=build_adaptive_plan(proposal,pairs,old,v10court,rest,
            source_rest_sha256=sha(rest_path),max_queries=args.max_queries)
        plan["parents"]={
            "original_source_sha256":v9row["source_sha256"],
            "v9_public_seal_sha256":sha(args.public_v9),
            "v10_public_seal_sha256":sha(args.public_v10),
            "original_geometry_proposal_sha256":sha(group_proposal_path),
            "original_rest_npz_sha256":sha(rest_path),
            "original_source_hit_pair_sha256":sha(pair_ref["path"]),
            "original_native_v9_manifest_sha256":sha(v9path),
            "independent_native_v10_court_sha256":sha(v10path),
        }
        receipt=out/"PRIVATE_HELDOUT_BRANCH_PROBE_PLAN_V11.json"
        save(receipt,plan)
        require(plan["new_source_face_intervention_query_count"]>=4
                and plan["real_rgba_consequences_measured_for_heldout_queries"] is False
                and plan["same_drawable_positive_pairs"]==0
                and plan["training_authorized"] is False,
                "V11_HELDOUT_QUERY_AUTHORITY_FAIL")
        result.update({
            "status":"REAL_ORIGINAL_SOURCE_HELDOUT_NONSEMANTIC_PROBE_PLAN_SEALED",
            "subject_label":sid,"subject_id":subject,
            "source_sha256":v9row["source_sha256"],
            "plan_sha256":sha(receipt),
            "new_query_count":plan["new_source_face_intervention_query_count"],
            "finest_candidate_count":plan["source_finest_groups_count"],
            "prior_finest_groups_excluded":plan["prior_finest_group_ids_excluded_count"],
            "new_split_parent_events_covered":plan["selected_splittable_parent_events"],
            "possible_split_parent_events":plan["potential_splittable_parent_events"],
            "unresolved_existing_candidate_size_pairs":
                plan["prior_edit_indistinguishable_candidate_size_pairs"],
            "model_training_authorized":False,
            "graphics_owner_authorized":False,
            "training_authorized":False,
        })
    except (Exception,KeyboardInterrupt) as err:
        result["failure_type"]=type(err).__name__
        result["failure_reason"]=str(err)
        save(out/"PRIVATE_HELDOUT_BRANCH_PROBE_FAILURE_V11.json",{
            "error_type":type(err).__name__,
            "reason":str(err),"traceback":traceback.format_exc(),
            "graphics_owner_authorized":False,
            "training_authorized":False,
        })
    save(out/"PRIVATE_HELDOUT_BRANCH_PROBE_BATCH_V11.json",result)
    print(json.dumps(result,sort_keys=True),flush=True)
    require(result["status"].startswith("REAL_ORIGINAL_"),
            "V11_HELDOUT_SOURCE_QUERY_DESIGN_REQUIRES_REPAIR")
    return result

def main():
    q=argparse.ArgumentParser()
    for k in ("public_v9","public_v9_sha256","public_v10","public_v10_sha256",
              "original_join_root","allview_groups_root","rays_index",
              "v9_private_root","v10_private_root","output"):
        q.add_argument("--"+k.replace("_","-"),required=True,
                       type=str if k.endswith("sha256") else Path)
    q.add_argument("--subject-ordinal",type=int,required=True)
    q.add_argument("--max-queries",type=int,default=12)
    run(q.parse_args())

if __name__=="__main__":
    main()
