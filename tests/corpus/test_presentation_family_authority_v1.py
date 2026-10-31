from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
import pytest

from tools.corpus.build_presentation_family_authority_v1 import build
from tools.corpus.validate_presentation_family_authority_v1 import validate


def sha(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path:Path,value): path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def ray(index,view,x,y,hits,alpha,rgb):
    a=np.asarray(alpha,float); trans=np.ones(len(a));
    for i in range(1,len(a)): trans[i]=trans[i-1]*(1-a[i-1])
    contribution=trans*a; rgb=np.asarray(rgb,float); full_alpha=float(contribution.sum()); full_rgb=(rgb*contribution[:,None]).sum(0)
    return {"ray_index":index,"view_index":view,"pixel_x":x,"pixel_y":y,"canonical_hits":hits,"measurement":{"status":"PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION","valid":True,"effective_alpha":a.tolist(),"straight_linear_rgb":rgb.tolist(),"incoming_transmittance":trans.tolist(),"contribution":contribution.tolist(),"predicted_full_alpha":full_alpha,"predicted_full_premultiplied_linear_rgb":full_rgb.tolist()}}


def fixture(root:Path):
    hits0=[{"face_index":0,"depth":1.0,"barycentric":[.2,.3,.5]}]
    hits1=[{"face_index":1,"depth":1.0,"barycentric":[.2,.3,.5]},{"face_index":2,"depth":2.0,"barycentric":[.1,.2,.7]}]
    rays=[ray(0,0,10,20,hits0,[.8],[[.25,.5,.75]]),ray(1,1,30,40,hits1,[.5,.4],[[1.,0.,0.],[0.,1.,0.]])]
    packed={"schema":"RealSaS.SourceRayAppearanceMeasurementCandidate.v1","subject_id":"s","teacher_scene_sha256":"a"*64,"geometry_npz_sha256":"b"*64,"candidate_sha256":"c"*64,"camera_bundle_sha256":"d"*64,"packed_source_scene_blend_sha256":"e"*64,"sampling_contract":{"x":1},"rgb_carrier_calibration":{"x":1},"material_policy_replay":{"x":1},"selected_ray_count":2,"pass_ray_count":2,"fail_ray_count":0,"abstain_ray_count":0,"rays":rays,"authority":"UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS","court_status":"CANDIDATE"}
    pa=root/"PACKED_A.json"; pb=root/"PACKED_B.json"; write(pa,packed); write(pb,packed)
    source_hits=[]
    for r in rays:
        m=r["measurement"]
        for i,h in enumerate(r["canonical_hits"]):
            a=m["effective_alpha"][i]; rgb=np.asarray(m["straight_linear_rgb"][i])*a
            source_hits.append({"ray_index":r["ray_index"],"view_index":r["view_index"],"pixel_x":r["pixel_x"],"pixel_y":r["pixel_y"],"hit_index":i,"face_index":h["face_index"],"effective_alpha":a,"premultiplied_linear_rgb":rgb.tolist(),"footprint_convergence_delta":.001})
    source={"schema":"RealSaS.SourceRayAppearanceReference.v1","subject_id":"s","source_sha256":"f"*64,"candidate_sha256":"c"*64,"camera_bundle_sha256":"d"*64,"appearance_sha256":"1"*64,"geometry_npz_sha256":"b"*64,"source_material_transform":{"x":1},"rgb_carrier_calibration":{"x":1},"equivalence_scope":"TEST","hit_count":len(source_hits),"hits":source_hits,"authority":"SOURCE_OBSERVATION_APPEARANCE_REFERENCE_ONLY"}
    sa=root/"SOURCE_A.json"; sb=root/"SOURCE_B.json"; write(sa,source); write(sb,source)
    arrays=root/"arrays.npz"; np.savez_compressed(arrays,dummy=np.asarray([1],np.int32))
    summary={"schema":"RealSaS.PresentationResponseTruthCandidate.v1","subject_id":"s","array_sha256":sha(arrays),"policy":{"student_observable_contribution_floor":1e-6},"renderer_subject_qualification":{"status":"PASS_PRESENTATION_RAY_SUBJECT_ORACLE_CANDIDATE","source_equivalent":True,"packed_replay_exact":True,"source_reference_replay_exact":True}}
    sp=root/"summary.json"; write(sp,summary)
    return sp,arrays,pa,pb,sa,sb


def test_graphics_domains_are_split_and_courted(tmp_path:Path):
    sp,arrays,pa,pb,sa,sb=fixture(tmp_path); out=tmp_path/"out"
    manifest=build("s",sp,arrays,pa,pb,sa,sb,out); mp=out/"PRESENTATION_FAMILY_AUTHORITY_MANIFEST_V1.json"
    assert manifest["counts"]["directly_identifiable_appearance_hit_count"]==1
    assert manifest["counts"]["student_observable_pair_direction_count"]==2
    assert manifest["counts"]["oracle_ray_count"]==2
    with np.load(out/"PRESENTATION_FAMILY_AUTHORITY_V1.npz",allow_pickle=False) as z:
        assert z["appearance_face"].tolist()==[0]
        assert z["visibility_pair_face_a"].tolist()==[1,2]
        assert z["visibility_pair_face_b"].tolist()==[2,1]
        assert np.allclose(z["visibility_pair_occlusion_probability"],[.5,0.])
    court=validate(mp)
    assert court["status"]=="PASS_PRESENTATION_FAMILY_INDEPENDENT_COURT__STUDENT_ADMISSION_SEPARATE"
    assert court["counts"]["appearance_valid_rows"]==1
    assert court["authority"]["student_supervision_authorized"] is False


def test_hidden_overlap_does_not_become_deterministic_appearance(tmp_path:Path):
    sp,arrays,pa,pb,sa,sb=fixture(tmp_path); out=tmp_path/"out"; build("s",sp,arrays,pa,pb,sa,sb,out)
    with np.load(out/"PRESENTATION_FAMILY_AUTHORITY_V1.npz",allow_pickle=False) as z:
        # The two-hit ray contributes visibility/compositing evidence, but neither
        # per-hit colour is a deterministic appearance target.
        assert z["appearance_face"].tolist()==[0]
        assert len(z["visibility_hit_face"])==3


def test_independent_court_rejects_extracted_numeric_drift(tmp_path:Path):
    sp,arrays,pa,pb,sa,sb=fixture(tmp_path); out=tmp_path/"out"; build("s",sp,arrays,pa,pb,sa,sb,out); mp=out/"PRESENTATION_FAMILY_AUTHORITY_MANIFEST_V1.json"; npz=out/"PRESENTATION_FAMILY_AUTHORITY_V1.npz"
    with np.load(npz,allow_pickle=False) as z: values={k:z[k].copy() for k in z.files}
    values["visibility_hit_contribution"][0]+=.01; np.savez_compressed(npz,**values)
    m=json.loads(mp.read_text()); m["output"]["sha256"]=sha(npz); write(mp,m)
    with pytest.raises(ValueError,match="PRESENTATION_COURT_VALUE:visibility_hit_contribution"): validate(mp)
