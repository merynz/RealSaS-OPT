#!/usr/bin/env python3
"""Independent byte-exact private eight-camera Blender court replay, 3 subjects.

Reads already finished original-source Blender V4 outputs, not merely their
SUCCESS flags. Reloads SHA-verified RGBA arrays, calls the independent pixel
court and compares its entire result with the original sealed decision.
All raw art, image pixels and teacher source-face/slot data remain private.

Only emits a bounded operational edit-evidence receipt. Never promotes
artist drawable owner truth, materialization, full native CAA or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from tools.corpus.court_corpus370_persistent_eight_view_edit_v4 import evaluate

SCHEMA="RealSaS.Corpus370PersistentEightViewBatchIndependentReplay.v4"
RUNNER="RealSaS.Corpus370PersistentEightViewFullFramePrivateRunner.v4"
COURT="RealSaS.Corpus370PersistentEightViewFullFrameEditCourt.v4"
RENDER="RealSaS.Corpus370AnonymousFullFrameEditorRender.v4"
EXPECTED_SUBJECT_SHA=(
    "1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850",
    "ad66b51cc5fc44c9a3429a5b1b98a61621af08d58d102d8a817499bd6a7f5b5a",
    "e6898505d56d8057c975b6376f123876dd957347282413cf3fb825754841b989",
)
ALLVIEW_GROUP_PARENT="1d8b8e2e8018d2cd3b94188f2683424659a8b0735f43c43eb381b6b252a3b8a1"
GEOMETRY_JOIN_PARENT="e72f3380c2f34cbf3e088218ebfcd4be0f4aa71c9138d4b3d1865a4f4074439a"


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(path,expected_sha=None):
    p=Path(path)
    require(p.is_file(),"V4_REPLAY_PARENT_FILE_MISSING")
    if expected_sha is not None:
        require(sha(p)==expected_sha,"V4_REPLAY_SHA_DRIFT")
    return json.loads(p.read_text())


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".partial")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    tmp.replace(p)


def independent_replay(input_root,output):
    root=Path(input_root).resolve()
    report={
        "schema":SCHEMA,
        "status":"PARTIAL_THREE_SOURCE_RGBA_INDEPENDENT_REPLAY__NO_OWNER_AUTHORITY",
        "original_hosted_run_id":None,
        "expected_source_subjects":3,
        "source_allview_group_batch_sha256":ALLVIEW_GROUP_PARENT,
        "source_native_visibility_join_sha256":GEOMETRY_JOIN_PARENT,
        "subjects":{},
        "graphics_owner_authorized":False,
        "tessa_teacher_authorized":False,
        "product_compilation_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }
    for ordinal,source_sha in enumerate(EXPECTED_SUBJECT_SHA):
        folder=root/str(ordinal)
        try:
            receipt_path=folder/"PRIVATE_FULL_FRAME_LAYER_BATCH_RESULT_V4.json"
            receipt=read(receipt_path)
            require(receipt.get("schema")==RUNNER
                    and receipt.get("status")==
                        "COMPLETE_FULL_FRAME_ORIGINAL_SOURCE_AND_ANON_RENDER_COURT__FAMILY_SEPARATE"
                    and receipt.get("subject_ordinal")==ordinal
                    and receipt.get("source_sha256")==source_sha
                    and receipt.get("geometry_only_allview_groups_sha256")==
                        ALLVIEW_GROUP_PARENT
                    and receipt.get("original_rest_native_visibility_join_sha256")==
                        GEOMETRY_JOIN_PARENT
                    and receipt.get("model_training_authorized") is False
                    and receipt.get("graphics_owner_authorized") is False
                    and receipt.get("training_authorized") is False,
                    "V4_REPLAY_SOURCE_RECEIPT_AUTHORITY")
            manifest_path=folder/"PRIVATE_FULL_FRAME_LAYER_RENDER_V4.json"
            image_path=folder/"PRIVATE_FULL_FRAME_LAYER_RENDER_V4.npz"
            court_path=folder/"FULL_FRAME_EDITOR_INDEPENDENT_COURT_V4.json"
            manifest=read(manifest_path,receipt["render_manifest_sha256"])
            old_court=read(court_path,receipt["independent_court_sha256"])
            require(manifest.get("schema")==RENDER
                    and manifest.get("source_sha256")==source_sha
                    and manifest.get("array_sha256")==receipt["render_arrays_sha256"]
                    and manifest.get("persistent_pair_anchor_view")==4
                    and manifest.get("views")==list(range(8))
                    and manifest.get("sample_count")==32
                    and manifest.get("full_frame_resolution")==192
                    and manifest.get("teacher_only") is True
                    and manifest.get("training_authorized") is False
                    and manifest.get("graphics_owner_authorized") is False,
                    "V4_REPLAY_RENDER_MANIFEST_SCOPE")
            require(image_path.is_file()
                    and sha(image_path)==receipt["render_arrays_sha256"],
                    "V4_REPLAY_RGBA_SOURCE_BYTES_SHA")
            with np.load(image_path,allow_pickle=False) as array:
                require(set(array.files)=={"source_original_vs_anonymous_group_rgba"},
                        "V4_REPLAY_RGBA_ARRAY_SCHEMA")
                independent=evaluate(array["source_original_vs_anonymous_group_rgba"],manifest)
            require(old_court.get("schema")==COURT
                    and old_court.get("training_authorized") is False
                    and old_court.get("graphics_owner_family_authorized") is False
                    and old_court.get("parents",{}).get("render_array_sha256")==
                        receipt["render_arrays_sha256"],
                    "V4_REPLAY_OLD_COURT_SCOPE")
            without_parent=dict(old_court)
            without_parent.pop("parents",None)
            require(without_parent==independent,
                    "V4_REPLAY_INDEPENDENT_PIXEL_COURT_DISAGREES")
            require(independent["original_full_fov_pixels_measured"]==8*192*192
                    and independent["bounded_parity_and_separable_views"]==8
                    and independent["persistent_group_pair_across_all_views"] is True
                    and independent["independently_measured_view4_group_b_reveal"]>0
                    and independent["views_with_visible_a_edit"]>=2
                    and independent["views_with_visible_b_after_a"]>=2
                    and all(x["source_to_soup_bounded_parity_pass"] is True
                            and x["no_op_rerender_parity_pass"] is True
                            for x in independent["per_view"]),
                    "V4_REPLAY_REAL_EIGHT_VIEW_PERFORMANCE_FAIL")
            report["subjects"][str(ordinal)]={
                "status":"PASS_BOUNDED_REAL_EIGHT_CAMERA_PERSISTENT_OP_EDIT_REPLAY",
                "source_sha256":source_sha,
                "source_original_rgba_bytes_sha256":receipt["render_arrays_sha256"],
                "render_manifest_sha256":receipt["render_manifest_sha256"],
                "independent_old_court_sha256":receipt["independent_court_sha256"],
                "independent_replay_matches_old_court":True,
                "full_fov_pixels_replayed":independent["original_full_fov_pixels_measured"],
                "views_with_visible_a_edit":independent["views_with_visible_a_edit"],
                "views_with_visible_b_after_a":independent["views_with_visible_b_after_a"],
                "heldout_view4_conditional_hidden_effect_pixels":
                    independent["independently_measured_view4_group_b_reveal"],
                "source_native_appearance_caa_authorized":False,
                "human_drawable_owner_qualified":False,
                "training_authorized":False,
            }
        except (KeyboardInterrupt,MemoryError):
            raise
        except Exception as ex:
            report["subjects"][str(ordinal)]={
                "status":"FAIL_INDEPENDENT_RGBA_BYTE_REPLAY_OR_EDIT_CONSEQUENCE",
                "error_type":type(ex).__name__,
                "reason":str(ex),
                "training_authorized":False,
            }
            save(Path(output).parent/(f"V4_REPLAY_FAILURE_{ordinal}.json"),{
                "status":"FAIL_INDEPENDENT_RGBA_BYTE_REPLAY_OR_EDIT_CONSEQUENCE",
                "ordinal":ordinal,
                "reason":str(ex),
                "traceback":traceback.format_exc(),
                "training_authorized":False,
            })
        finally:
            save(output,report)
    good=sum(x["status"]=="PASS_BOUNDED_REAL_EIGHT_CAMERA_PERSISTENT_OP_EDIT_REPLAY"
             for x in report["subjects"].values())
    report["measured_independent_pass_subjects"]=int(good)
    report["failure_count"]=3-good
    report["status"]=(
        "THREE_REAL_SOURCE_EIGHT_CAMERA_RGBA_REPLAY_COMPLETE__NO_OWNER_AUTHORITY"
        if good==3 else
        "FAIL_NOT_ALL_SOURCE_RGBA_COURTS_PASS"
    )
    report["replayed_full_fov_pixel_count"]=sum(
        x.get("full_fov_pixels_replayed",0) for x in report["subjects"].values())
    save(output,report)
    print(json.dumps({
        "status":report["status"],
        "independent_subjects":report["measured_independent_pass_subjects"],
        "failure_count":report["failure_count"],
        "replayed_pixels":report["replayed_full_fov_pixel_count"],
        "training_authorized":False,
    },sort_keys=True),flush=True)
    require(good==3,"V4_REPLAY_FAILURES_REQUIRE_REPAIR")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    independent_replay(a.input_root,a.output)


if __name__=="__main__":main()
