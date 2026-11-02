from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from tools.corpus.build_source_observation_authority_v1 import build
from tools.corpus.validate_source_observation_authority_v1 import validate


def sha(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path:Path,value): path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def packet_fixture(root:Path):
    packet=root/"packet"; packet.mkdir(); source=root/"actor.glb"; source.write_bytes(b"immutable-source")
    camera=packet/"camera_bundle.json"
    write_json(camera,{"schema":"RealSaS.CameraProjectionBundle.v1","per_view_recenter":False,"per_view_pose_change":False,"common_half_extent":1.0,"common_center_xyz":[0.,0.,0.],"cameras":[{"view_index":i} for i in range(8)]})
    snapshot=packet/"SOURCE_SCENE_SNAPSHOT.json"
    identity=np.eye(4).tolist()
    write_json(snapshot,{"schema":"RealSaS.ImportedSourceSceneSnapshot.v2","source_sha256":sha(source),"source_world_to_canonical":identity,"canonical_to_source_world":identity,"nodes":[{"node_id":"root","parent_node_id":None,"node_to_source_world":identity,"effective_node_to_parent":identity,"node_to_canonical":identity,"active_action_count":0,"active_nla_track_count":0,"driver_count":0}],"bones":[]})
    views=[]; proof_views=[]
    for i in range(8):
        raster=packet/f"V{i}.png"; mask=packet/f"V{i}.mask.bin"
        rgba=np.zeros((1024,1024,4),np.uint8); rgba[100:900,100:900,:3]=[25+i,50,75]; rgba[100:900,100:900,3]=255
        Image.fromarray(rgba,"RGBA").save(raster)
        m=np.zeros((1024,1024),np.uint8); m[100:900,100:900]=1; mask.write_bytes(m.tobytes())
        views.append({"view_index":i,"raster":{"sha256":sha(raster)},"foreground_mask":{"sha256":sha(mask)}})
        proof_views.append({"view_index":i,"raster_sha256":sha(raster),"mask_sha256":sha(mask),"maximum_mask_gap_px":0.0,"border_margin_px":100})
    candidate=packet/"SOURCE_REST_RENDER_CANDIDATE.json"
    write_json(candidate,{"schema":"RealSaS.Controlled8ViewSourceRestCandidate.v2","subject_id":"subject","source_sha256":sha(source),"raw_source_mutated":False,"pose_address":"SOURCE_BIND_REST","native_frame0_is_rest_assumed":False,"training_admission_claimed":False,"product_pass_claimed":False,"internal_consistency":{"animation_free_rest":True,"normalization_uniform":True},"source_scene_snapshot":{"sha256":sha(snapshot)},"camera_bundle":{"sha256":sha(camera)},"views":views})
    proof=packet/"IRIS_OBSERVATION_ADMISSION.json"
    policy={"scope":"IMAGE_DERIVED_FSTAR_BIND_OBSERVATIONS__NOT_FULL_MECHANICAL_TRUTH","resolution":1024,"view_count":8,"minimum_border_margin_px":8,"maximum_relative_vertex_error":1e-5,"maximum_mask_gap_px":2.0,"reference_coverage":"TRIANGLE_UNIT_PIXEL_SQUARE_INTERSECTION","pose":"SOURCE_BIND_REST","native_frame0_is_bind_assumed":False,"native_lit_material_parity_claimed":False,"full_native_motion_truth_admitted":False,"go_stage_qualification_claimed":False}
    write_json(proof,{"schema":"RealSaS.IrisBindObservationAdmission.v1","status":"PASS_IRIS_BIND_OBSERVATIONS","scope":"IMAGE_DERIVED_FSTAR_BIND_OBSERVATIONS__NOT_FULL_MECHANICAL_TRUTH","subject_id":"subject","source_sha256":sha(source),"candidate_sha256":sha(candidate),"camera_sha256":sha(camera),"relative_vertex_error":0.0,"policy":policy,"views":proof_views,"raw_vertex_count":4,"imported_vertex_count":4,"bone_count":0,"full_native_motion_truth_admitted":False,"native_lit_parity_claimed":False,"go_stage_qualification_claimed":False,"fit8_pass_claimed":False})
    return source,packet


def test_source_observation_producer_to_independent_court(tmp_path:Path):
    source,packet=packet_fixture(tmp_path); out=tmp_path/"out"
    manifest=build("subject",source,packet,out); manifest_path=out/"SOURCE_OBSERVATION_AUTHORITY_MANIFEST_V1.json"
    court=validate(manifest_path)
    assert manifest["family_candidate_outcomes"]=={"SOURCE_IMPORT":"QUALIFIED_CORE_CANDIDATE","CAMERA_OBSERVATION":"QUALIFIED_CORE_CANDIDATE"}
    assert court["status"]=="PASS_SOURCE_OBSERVATION_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"
    assert court["family_candidate_outcomes"]["SOURCE_IMPORT"]=="QUALIFIED_CORE_CANDIDATE"
    assert court["authority"]["training_authorized"] is False


def test_source_observation_court_rejects_raster_byte_drift(tmp_path:Path):
    source,packet=packet_fixture(tmp_path); out=tmp_path/"out"; build("subject",source,packet,out); manifest_path=out/"SOURCE_OBSERVATION_AUTHORITY_MANIFEST_V1.json"
    raster=packet/"V3.png"; raster.write_bytes(raster.read_bytes()+b"drift")
    with pytest.raises(ValueError,match="SOURCE_OBSERVATION_COURT_VIEW_HASH"):
        validate(manifest_path)


def test_source_observation_court_rejects_live_rest_driver(tmp_path:Path):
    source,packet=packet_fixture(tmp_path); out=tmp_path/"out"; build("subject",source,packet,out); manifest_path=out/"SOURCE_OBSERVATION_AUTHORITY_MANIFEST_V1.json"
    snapshot=packet/"SOURCE_SCENE_SNAPSHOT.json"; value=json.loads(snapshot.read_text()); value["nodes"][0]["driver_count"]=1; write_json(snapshot,value)
    manifest=json.loads(manifest_path.read_text()); manifest["controls"]["snapshot"]["sha256"]=sha(snapshot); manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    candidate=packet/"SOURCE_REST_RENDER_CANDIDATE.json"; c=json.loads(candidate.read_text()); c["source_scene_snapshot"]["sha256"]=sha(snapshot); write_json(candidate,c)
    manifest=json.loads(manifest_path.read_text()); manifest["controls"]["candidate"]["sha256"]=sha(candidate); manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    proof=packet/"IRIS_OBSERVATION_ADMISSION.json"; p=json.loads(proof.read_text()); p["candidate_sha256"]=sha(candidate); write_json(proof,p)
    manifest=json.loads(manifest_path.read_text()); manifest["controls"]["proof"]["sha256"]=sha(proof); manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    with pytest.raises(ValueError,match="SOURCE_OBSERVATION_COURT_LIVE_REST"):
        validate(manifest_path)
