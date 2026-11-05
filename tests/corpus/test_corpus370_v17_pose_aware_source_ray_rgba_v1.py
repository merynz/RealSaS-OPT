"""V17 pose-aware source geometric ray and RGBA readset falsification.

All synthetic geometry is test-only; original artist drawable and CAA unknown.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.corpus370_v17_pose_aware_source_ray_rgba_join_v1 import (
    RES,VIEW_INDICES,SIGN_KEYS,CANDIDATES_PER_POSE_CAMERA,
    PRED_SCHEMA,COURT_SCHEMA,projection_camera,source_geometry,
    sample_geometry,ray_contract,predict,sha
)


def geometry():
    rest=np.asarray([
        [-.7,-.6,0.],[.8,-.6,0.],[.8,.7,0.],[-.7,.7,0.]
    ],np.float64)
    faces=np.asarray([[0,1,2],[0,2,3]],np.int64)
    original=np.stack([rest+np.array([p*.09,0.,0.]) for p in range(3)])
    joints=np.zeros((3,4,4,len(rest),3),np.float64)
    for p in range(3):
        for vi in range(4):
            for s in range(4):
                joints[p,vi,s]=original[p]+[0.,s*.013,0.]
    cameras={"schema":"RealSaS.CameraProjectionBundle.v1",
        "cameras":[{
            "view_index":i,"origin":[0.,0.,5.],
            "forward":[0.,0.,-1.],"right":[1.,0.,0.],
            "screen_up":[0.,1.,0.],
            "half_extent":2.,"resolution":512,"clip_start":.01,
            "clip_end":100.
        } for i in range(8)]}
    return rest,faces,original,joints,cameras


def test_pose_aware_geometric_centroid_ray_proposals_without_any_pixels():
    _,faces,original,joints,cameras=geometry()
    rows,summary=sample_geometry(faces,original,joints,cameras)
    assert len(rows)==3*4*2
    assert summary["source_pixel_proposal_count"]==len(rows)
    assert summary["ray_geometry_state_cases"]==len(rows)*5
    assert summary["ray_complete_untied_hit_cases"]>0
    assert summary["original_geometric_hit_count_over_all_states"]>0
    assert {x["source_pose_ordinal"] for x in rows}=={0,1,2}
    assert {x["source_camera_view"] for x in rows}==set(VIEW_INDICES)
    for row in rows:
        assert row["geometry_selection_read_rgba"] is False
        assert row["original_artist_drawable_id"]=="UNKNOWN"
        assert set(row["joint_sign_pair_rays"])==set(SIGN_KEYS)
        assert row["baseline_pose_ray"]["hits"]
        assert all(0<=h["face_index_teacher_only"]<len(faces)
                   for h in row["baseline_pose_ray"]["hits"])


def test_source_ray_camera_raster_binds_192_not_source_capture_pixel_count():
    *_,cameras=geometry()
    cam=projection_camera(cameras,0)
    assert cam.resolution==RES==192
    assert cam.half_extent==2.
    assert cam.origin==(0.,0.,5.)
    assert cam.forward==(0.,0.,-1.)


def test_geometry_only_candidate_is_deterministic_in_all_camera_pose_conditions():
    _,f,p,j,c=geometry()
    a,ma=sample_geometry(f,p,j,c)
    b,mb=sample_geometry(f,p,j,c)
    assert a==b and ma==mb
    assert len(a)<=3*4*CANDIDATES_PER_POSE_CAMERA
    assert any(x["joint_sign_pair_rays"]["PLUS_PLUS"]["hits"] for x in a)


def test_camera_wrong_up_basis_fail_closed():
    *_,cams=geometry()
    cams["cameras"][0]["screen_up"]=[1.,0.,0.]
    with pytest.raises(ValueError):
        projection_camera(cams,0)


def test_camera_malformed_eight_view_inventory_fail_closed():
    *_,cams=geometry()
    cams["cameras"].pop()
    with pytest.raises(ValueError):
        projection_camera(cams,0)


def test_camera_schema_forgery_fails():
    *_,cams=geometry()
    cams["schema"]="RealSaS.ForgedPixelCamera"
    with pytest.raises(ValueError):
        projection_camera(cams,0)


def test_tied_source_sheets_abstain_not_reassign_owner():
    rest,f,orig,j,cams=geometry()
    faces=np.concatenate((f,f),axis=0)
    rows,stats=sample_geometry(faces,orig,j,cams)
    assert stats["ray_incomplete_tied_or_no_hit_abstain_cases"]>0
    assert stats["ray_complete_untied_hit_cases"]==0
    assert all(x["original_artist_drawable_id"]=="UNKNOWN" for x in rows)


def _fixture(tmp_path):
    rest,faces,orig,joints,cams=geometry()
    root=tmp_path
    paths={
        "rest_npz":root/"rest.npz",
        "v13b_npz":root/"v13b.npz",
        "v16_npz":root/"joint.npz",
        "camera_bundle":root/"camera.json",
        "v14_job":root/"job.json",
        "v14_manifest":root/"v14.json",
        "v16_manifest":root/"v16.json",
        "prediction":root/"predict.json"
    }
    np.savez_compressed(paths["rest_npz"],faces=faces,
                        rest_vertices_source_world=rest)
    np.savez_compressed(paths["v13b_npz"],baseline_vertices=orig)
    np.savez_compressed(
        paths["v16_npz"],
        original_native_dual_control_world_vertices=joints,
        original_native_dual_control_rgba=np.zeros(
            (3,4,14,192,192,4),np.float16
        )
    )
    paths["camera_bundle"].write_text(json.dumps(cams))
    source_sha="a"*64
    job={
        "schema":"RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14",
        "source":{"sha256":source_sha},
        "subject_id":"synthetic",
        "rest_geometry":{"sha256":sha(paths["rest_npz"])},
        "camera":{"sha256":sha(paths["camera_bundle"])},
        "v13b_array":{"sha256":sha(paths["v13b_npz"])},
        "artist_drawable_identity_inferred":False,
        "training_authorized":False
    }
    paths["v14_job"].write_text(json.dumps(job))
    v14={
        "schema":"RealSaS.Corpus370SourceNativeMechanicalRasterJointRender.v14",
        "source_sha256":source_sha,
        "subject_id":"synthetic",
        "private_job_sha256":sha(paths["v14_job"]),
        "source_native_perturbed_vertices_verified_before_and_after_render":True,
        "graphics_owner_authorized":False
    }
    paths["v14_manifest"].write_text(json.dumps(v14))
    v16={
        "schema":"RealSaS.Corpus370OriginalDualNativeControlCausalInteractionRender.v16",
        "source_sha256":source_sha,
        "subject_id":"synthetic",
        "parent_v14_job_sha256":sha(paths["v14_job"]),
        "parent_v14_render_manifest_sha256":sha(paths["v14_manifest"]),
        "original_dual_control_rgba_npz_sha256":sha(paths["v16_npz"]),
        "original_v13b_control_vertex_array_sha256":sha(paths["v13b_npz"]),
        "joint_original_source_to_soup_parity_rechecked":True,
        "private_original_joint_world_vertices_for_independent_court":True,
        "graphics_owner_authorized":False,
        "training_authorized":False
    }
    paths["v16_manifest"].write_text(json.dumps(v16))
    return argparse.Namespace(**paths),source_sha


def test_predict_only_original_3d_world_vertices_camera_and_faces(tmp_path):
    args,_=_fixture(tmp_path)
    predict(args)
    p=json.loads(args.prediction.read_text())
    assert p["schema"]==PRED_SCHEMA
    assert len(p["source_hit_ray_rows_teacher_only"])==24
    assert p["summary"]["ray_complete_untied_hit_cases"]>0
    assert p["image_source_pixels_read_for_proposal"] is False
    assert p["source_original_author_drawable_labels_used"] is False
    assert p["graphics_owner_authorized"] is False
    assert p["training_authorized"] is False
    assert p["original_artist_same_drawable_labels_qualified"]==0


@pytest.mark.parametrize("mutation",[
    "source","false_owner","false_training","false_geometry_witness",
    "false_joint_parity","corrupt_v13b","camera_pixel_drift",
    "false_original_source_sha","v16_manifest_sha"
])
def test_v17_geometry_blind_precommit_rejects_source_and_auth_forgery(tmp_path,mutation):
    args,_=_fixture(tmp_path)
    if mutation in ("source","false_training"):
        path=args.v14_job
        d=json.loads(path.read_text())
        if mutation=="source":d["source"]["sha256"]="b"*64
        else:d["training_authorized"]=True
    elif mutation in ("false_owner","false_geometry_witness"):
        path=args.v14_manifest
        d=json.loads(path.read_text())
        if mutation=="false_owner":d["graphics_owner_authorized"]=True
        else:d["source_native_perturbed_vertices_verified_before_and_after_render"]=False
    elif mutation in ("false_joint_parity","false_original_source_sha","v16_manifest_sha"):
        path=args.v16_manifest
        d=json.loads(path.read_text())
        if mutation=="false_joint_parity":d["joint_original_source_to_soup_parity_rechecked"]=False
        elif mutation=="false_original_source_sha":d["source_sha256"]="0"*64
        else:d["original_dual_control_rgba_npz_sha256"]="0"*64
    elif mutation=="corrupt_v13b":
        args.v13b_npz.write_bytes(args.v13b_npz.read_bytes()+b"unexpected")
        with pytest.raises(ValueError):
            source_geometry(args)
        return
    elif mutation=="camera_pixel_drift":
        args.camera_bundle.write_text(args.camera_bundle.read_text()+" ")
        with pytest.raises(ValueError):
            source_geometry(args)
        return
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        source_geometry(args)


def test_v17_no_rays_or_image_response_mint_original_drawable_truth(tmp_path):
    args,_=_fixture(tmp_path)
    predict(args)
    p=json.loads(args.prediction.read_text())
    assert p["alpha_compositing_proven"] is False
    assert p["graphics_owner_authorized"] is False
    assert p["materialization_authorized"] is False
    assert p["original_artist_different_drawable_labels_qualified"]==0
    assert "artist" not in str(p["summary"]).lower()
    assert all("author" not in list(x["baseline_pose_ray"].keys())
               for x in p["source_hit_ray_rows_teacher_only"])


def test_v17_contract_result_is_never_artist_layer_equivalence():
    assert COURT_SCHEMA=="RealSaS.Corpus370PoseAwareSourceRayOriginalRGBAIndependentCourt.v17"
    assert VIEW_INDICES==(0,2,4,6)
    assert SIGN_KEYS==("PLUS_PLUS","PLUS_MINUS","MINUS_PLUS","MINUS_MINUS")
