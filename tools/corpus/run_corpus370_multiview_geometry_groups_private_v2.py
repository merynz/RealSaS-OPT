#!/usr/bin/env python3
"""Three actual source-exact subjects; all-view GEOMETRY proposal, view4 holdout.

Three prior immutable sources are required: source-hit geometric 352 index,
previous 3-subject native/canonical/renderer independent court, and prior
anonymous layer proposal. The heldout full renderer/edit evidence is never
presented to the producer, only the independent court after candidate seal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from tools.corpus.compile_corpus370_multiview_geometry_groups_v2 import build
from tools.corpus.court_corpus370_multiview_geometry_groups_v2 import qualify

SCHEMA="RealSaS.Corpus370MultiviewGeometricLayerPrivateBatch.v2"
JOIN_SHA="e72f3380c2f34cbf3e088218ebfcd4be0f4aa71c9138d4b3d1865a4f4074439a"
ANON_SHA="47b3f3beb7a3625504ccf8488d6fe0cd8fbe0014421c5e4c77748f829c9f01ca"
RAY_INDEX_SHA="0eeb24089322d0802567ba9736ada1de03e234bbf133bc91164132a80bec3c70"


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh,"sha256").hexdigest()


def load(path,expected,code):
    file=Path(path).resolve()
    require(file.is_file() and sha(file)==expected,code)
    return json.loads(file.read_text())


def seal(path,doc):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def run(join_root,anon_root,ray_root,output_root):
    join_root=Path(join_root).resolve()
    anon_root=Path(anon_root).resolve()
    ray_root=Path(ray_root).resolve()
    out=Path(output_root).resolve()
    join=load(join_root/"CORPUS370_NATIVE_VISIBILITY_PAIR_JOIN_V1.json",
              JOIN_SHA,"MULTIVIEW_SOURCE_JOIN_SHA")
    prior=load(anon_root/"CORPUS370_ANON_LAYER_PRIVATE_BATCH_V1.json",
               ANON_SHA,"MULTIVIEW_OLD_ANON_SUMMARY_SHA")
    rays=load(ray_root/"SOURCE_LOCUS_GEOMETRIC_RAYS_PILOT_INDEX_V1.json",
              RAY_INDEX_SHA,"MULTIVIEW_352_RAY_INDEX_SHA")
    ids=sorted(join["subjects"])
    require(len(ids)==3 and set(ids)==set(prior["subjects"])
            and join.get("training_authorized") is False
            and join.get("graphic_owner_identity_minted") is False
            and prior.get("training_authorized") is False
            and prior.get("graphics_owner_truth_minted") is False
            and rays.get("training_authorized") is False
            and rays.get("graphics_owner_truth_minted") is False
            and len(rays["subject_results"])==352
            and set(ids)<=set(rays["subject_results"]),
            "MULTIVIEW_SOURCE_SCOPE_AND_NONTRAINING")
    report={
        "schema":SCHEMA,
        "status":"PARTIAL_MULTIVIEW_GEOMETRIC_GROUPS__EDIT_FAMILY_NOT_AUTHORIZED",
        "parents":{"joint_source_native_visibility_sha256":JOIN_SHA,
                   "prior_anonymous_layer_batch_sha256":ANON_SHA,
                   "all352_source_geometry_rays_sha256":RAY_INDEX_SHA},
        "source_subject_count":3,
        "subjects":{},
        "source_names_used_as_labels":False,
        "graphics_owner_authorized":False,
        "model_training_authorized":False,
        "product_compilation_authorized":False,
    }
    output=out/"CORPUS370_MULTIVIEW_GEOMETRY_GROUPS_PRIVATE_BATCH_V2.json"
    for sid in ids:
        target=out/"subjects"/sid
        try:
            row=join["subjects"][sid]; old=prior["subjects"][sid]
            require(row["source_sha256"]==old["source_sha256"]
                    and old["training_authorized"] is False
                    and row["training_authorized"] is False,
                    "MULTIVIEW_PRIOR_SUBJECT_SOURCE")
            root=anon_root/"subjects"/sid
            anonymous=load(root/"ANON_LAYER_CANDIDATES_V1.json",
                           old["proposal_sha256"],"MULTIVIEW_OLD_PROPOSAL_BYTES")
            old_court=load(root/"ANON_LAYER_EDIT_PROXY_COURT_V1.json",
                           old["consumer_sha256"],"MULTIVIEW_OLD_COURT_BYTES")
            pair=load(row["pair_diagnostic_receipt"]["path"],
                      row["pair_diagnostic_receipt"]["sha256"],
                      "MULTIVIEW_OLD_RENDER_PAIRS_BYTES")
            canon=load(row["canonical_candidate_receipt"]["path"],
                       row["canonical_candidate_receipt"]["sha256"],
                       "MULTIVIEW_CANONICAL_RECEIPT_BYTES")
            rt=rays["subject_results"][sid]
            require(rt["status"]=="MEASURED_GEOMETRIC_RAY_CANDIDATE"
                    and rt["source_sha256"]==row["source_sha256"]
                    and rt["training_authorized"] is False,
                    "MULTIVIEW_RAY_SOURCE_SCOPE")
            rayfile=ray_root/"subjects"/sid/"SOURCE_LOCUS_GEOMETRIC_RAYS_V1.json"
            geom=load(rayfile,rt["receipt"]["sha256"],
                      "MULTIVIEW_SOURCE_GEOMETRY_BLOB_SHA")
            require(geom["source_rest_geometry"]["sha256"]==
                        pair["parents"]["rest_geometry_sha256"]
                    and geom["source_sha256"]==row["source_sha256"]
                    and canon["canonical_arrays"]["sha256"]==
                        anonymous["parents"]["canonical_arrays_sha256"],
                    "MULTIVIEW_REST_SOURCE_CANONICAL_BINDING")
            npz=Path(canon["canonical_arrays"]["path"]).resolve()
            require(npz.is_file()
                    and sha(npz)==canon["canonical_arrays"]["sha256"],
                    "MULTIVIEW_CANONICAL_SOURCE_ARRAYS_SHA")
            with np.load(npz,allow_pickle=False) as data:
                require({"rest_vertices_canonical","posed_vertices_canonical",
                         "faces"}<=set(data.files),"MULTIVIEW_CANONICAL_ARRAYS")
                candidate=build(data["rest_vertices_canonical"],data["faces"],
                                data["posed_vertices_canonical"],
                                anonymous,old_court,geom)
            proposed=target/"MULTIVIEW_ANONYMOUS_GROUP_CANDIDATES_V2.json"
            seal(proposed,candidate)
            # Only now does the independent consumer access the original
            # rendered edit pairs from the heldout view 4.
            result=qualify(candidate,geom,pair,anonymous)
            result["parents"]={
                "proposal_sha256":sha(proposed),
                "geometric_rays_sha256":sha(rayfile),
                "independent_graphics_pairs_sha256":sha(
                    Path(row["pair_diagnostic_receipt"]["path"])),
            }
            courtfile=target/"MULTIVIEW_ANONYMOUS_EDIT_COURT_V2.json"
            seal(courtfile,result)
            report["subjects"][sid]={
                "status":"PASS_MULTIVIEW_GEOMETRY_GROUP_CANDIDATES_ONLY",
                "source_sha256":row["source_sha256"],
                "baseline_selected_group_count":candidate["old_source_atomic_group_count"],
                "refined_patch_count":candidate["refined_patch_count"],
                "geometric_cannot_link_candidate_count":candidate[
                    "geometry_candidate_constraint_count"],
                "geometry_only_refinement_count":candidate[
                    "minimal_face_refinements_due_to_geometry_count"],
                "selected_group_count":result["selected_group_count"],
                "heldout_render_edit_proxy_fraction":result[
                    "heldout_expressibility_diagnostic_only"],
                "candidate_group_count":candidate["candidate_count"],
                "proposal_sha256":sha(proposed),
                "court_sha256":sha(courtfile),
                "graphics_owner_authorized":False,
                "product_compilation_authorized":False,
                "training_authorized":False,
            }
        except (KeyboardInterrupt,MemoryError):
            raise
        except Exception as exc:
            report["subjects"][sid]={
                "status":"FAIL_MULTIVIEW_GEOMETRIC_SOURCE_GROUPS",
                "exception_type":type(exc).__name__,
                "reason":str(exc),"training_authorized":False,
            }
            seal(target/"MULTIVIEW_GEOMETRIC_GROUP_FAILURE_V2.json",{
                "source_subject_id":sid,
                "type":type(exc).__name__,
                "reason":str(exc),
                "traceback":traceback.format_exc(),
                "training_authorized":False,
            })
        finally:
            seal(output,report)
    status=Counter(x["status"] for x in report["subjects"].values())
    report["status"]="THREE_MULTIVIEW_GEOMETRIC_SOURCE_GROUPS_ACCOUNTED__NO_TRAINING"
    report["candidate_qualified_count"]=int(status[
        "PASS_MULTIVIEW_GEOMETRY_GROUP_CANDIDATES_ONLY"])
    report["failure_count"]=int(status["FAIL_MULTIVIEW_GEOMETRIC_SOURCE_GROUPS"])
    report["selected_group_counts"]={sid:int(x["selected_group_count"])
        for sid,x in report["subjects"].items()
        if x["status"].startswith("PASS_")}
    report["heldout_proxy_revealed_count"]=int(sum(
        int(x.get("heldout_render_edit_proxy_fraction")==1.)
        for x in report["subjects"].values()))
    seal(output,report)
    print(json.dumps({
        "status":report["status"],"qualified_count":report["candidate_qualified_count"],
        "failure_count":report["failure_count"],
        "selected_groups":sorted(report["selected_group_counts"].values()),
        "heldout_reveal_count":report["heldout_proxy_revealed_count"],
        "graphics_owner_authorized":False,"training_authorized":False,
    },sort_keys=True),flush=True)
    require(report["candidate_qualified_count"]==3
            and report["failure_count"]==0,
            "MULTIVIEW_REAL_SOURCE_FAILURES_REQUIRE_REPAIR")
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--native-visibility-join-root",type=Path,required=True)
    parser.add_argument("--prior-anonymous-root",type=Path,required=True)
    parser.add_argument("--source-geometry-rays-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args()
    run(args.native_visibility_join_root,args.prior_anonymous_root,
        args.source_geometry_rays_root,args.output_root)


if __name__=="__main__":main()
