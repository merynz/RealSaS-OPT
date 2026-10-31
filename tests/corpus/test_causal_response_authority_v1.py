from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from tools.corpus.build_mechanical_surface_topology_manifest_v1 import build as build_mechanical
from tools.corpus.validate_mechanical_surface_topology_authority_v1 import validate as validate_mechanical
from tools.corpus.build_causal_response_authority_v1 import build as build_response
from tools.corpus.validate_causal_response_authority_v1 import validate as validate_response


def sha(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fixture(root:Path):
    v=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,0.]],np.float64)
    f=np.asarray([[0,1,2],[3,4,5]],np.int32); owner=np.asarray([0,1],np.int32)
    plus=v.copy(); plus[:,2]+=0.2
    minus=v.copy(); minus[:,2]-=0.2
    posed=np.stack([plus,minus],axis=0)
    source=root/"geometry_and_response.npz"
    np.savez_compressed(source,rest_vertices_source_world=v,faces=f,face_owner_privileged=owner,isolated_position=posed)
    return source


def build_stack(root:Path):
    source=source_fixture(root)
    mech_root=root/"mechanical"; build_mechanical("subject",source,mech_root)
    mm_path=mech_root/"MECHANICAL_SURFACE_TOPOLOGY_MANIFEST_V1.json"
    mc=validate_mechanical(mm_path); mc_path=mech_root/"MECHANICAL_SURFACE_TOPOLOGY_COURT_V1.json"; mc_path.write_text(json.dumps(mc,sort_keys=True,indent=2)+"\n")
    scene=root/"teacher_scene.json"
    scene.write_text(json.dumps({"schema":"RealSaS.TeacherSceneIR.v2","subject_key":"subject","geometry_payloads":[{"path":str(source),"sha256":sha(source)}],"calibration":{"actions":[{"control_index_privileged":0,"amplitude":1.0,"sign":1},{"control_index_privileged":0,"amplitude":1.0,"sign":-1}]}},sort_keys=True)+"\n")
    response_root=root/"response"; manifest=build_response("subject",mm_path,mc_path,scene,response_root)
    return manifest,response_root/"CAUSAL_RESPONSE_AUTHORITY_MANIFEST_V1.json",response_root


def test_cross_owner_qualified_seam_enters_response_adjacency(tmp_path:Path):
    manifest,manifest_path,response_root=build_stack(tmp_path)
    assert manifest["counts"]["qualified_cross_owner_pair_count"]==1
    assert manifest["counts"]["qualified_pair_count"]==1
    with np.load(response_root/"CAUSAL_RESPONSE_AUTHORITY_V1.npz",allow_pickle=False) as z:
        assert z["pair_face"].tolist()==[[0,1]]
        assert np.allclose(z["pair_coherence"],[1.0])
        assert np.allclose(z["pair_jump"],[0.0])
        assert z["face_owner"].tolist()==[0,1]
    court=validate_response(manifest_path)
    assert court["status"]=="PASS_CAUSAL_RESPONSE_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"
    assert court["counts"]["qualified_cross_owner_pair_count"]==1
    assert court["family_candidate_outcomes"]["CAUSAL_RESPONSE"]=="QUALIFIED_CORE_CANDIDATE"
    assert court["authority"]["training_authorized"] is False


def test_response_court_rejects_numeric_mutation(tmp_path:Path):
    _,manifest_path,response_root=build_stack(tmp_path)
    array_path=response_root/"CAUSAL_RESPONSE_AUTHORITY_V1.npz"
    with np.load(array_path,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
    arrays["response_energy"][0]+=0.01
    np.savez_compressed(array_path,**arrays)
    manifest=json.loads(manifest_path.read_text()); manifest["output"]["sha256"]=sha(array_path); manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    with pytest.raises(ValueError,match="CAUSAL_COURT_NUMERIC:response_energy"):
        validate_response(manifest_path)


def test_response_stage47_parent_is_hash_bound(tmp_path:Path):
    _,manifest_path,_=build_stack(tmp_path)
    manifest=json.loads(manifest_path.read_text())
    manifest["parents"]["stage47_geometry_response"]["sha256"]="0"*64
    manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")
    with pytest.raises(ValueError,match="CAUSAL_COURT_STAGE47_DRIFT"):
        validate_response(manifest_path)
