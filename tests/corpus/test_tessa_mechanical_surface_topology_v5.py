from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from tools.training.validate_corpus_mechanical_tessa_teacher_v5 import validate


def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root: Path):
    topo=root/"mechanical.npz"
    v=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]],np.float32)
    f=np.asarray([[0,1,2],[2,1,3]],np.int32)
    np.savez_compressed(topo,mechanical_vertices_source_world=v,mechanical_faces=f,face_valid=np.asarray([True,True]),surface_component_id=np.asarray([0,0],np.int32))
    mm=root/"mechanical_manifest.json"; mm.write_text(json.dumps({"schema":"RealSaS.MechanicalSurfaceTopologyAuthority.v1","status":"PASS_MECHANICAL_SURFACE_TOPOLOGY_CANDIDATE__INDEPENDENT_COURT_REQUIRED","subject_id":"s","output":{"path":str(topo),"sha256":sha(topo)},"counts":{"qualified_cross_owner_seam_edge_count":1}},sort_keys=True)+"\n")
    mc=root/"mechanical_court.json"; mc.write_text(json.dumps({"schema":"RealSaS.MechanicalSurfaceTopologyAuthorityCourt.v1","status":"PASS_MECHANICAL_SURFACE_TOPOLOGY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE","subject_id":"s","authority":{"independent_court_passed":True}},sort_keys=True)+"\n")
    teacher=root/"teacher.npz"; np.savez_compressed(teacher,vertices_source_world=v,faces=f,source_triangle_id=np.asarray([0,0,0,1],np.int32),source_barycentric=np.asarray([[1,0,0],[0,1,0],[0,0,1],[0,0,1]],np.float32))
    selected={"ratio":1.0,"kind":"SOURCE_FALLBACK","vertex_count":4,"face_count":2,"mechanical_pass":True}
    return {"schema":"RealSaS.CorpusMechanicalTESSATeacher.v5","status":"PASS_CORPUS_MECHANICAL_TESSA_TEACHER_V5_CANDIDATE__INDEPENDENT_COURT_REQUIRED","subject_id":"s","parents":{"mechanical_surface_manifest":{"path":str(mm),"sha256":sha(mm)},"mechanical_surface_court":{"path":str(mc),"sha256":sha(mc)}},"source_topology":{"surface_component_count":1,"qualified_cross_owner_seam_edge_count":1,"component_relation":"QUALIFIED_MECHANICAL_EDGE_CONNECTED_SURFACE","sdf_sign_topology_consumed":False,"appearance_addressing_consumed":False,"render_vertex_identity_used_as_topology":False},"candidates":[selected],"selected":selected,"teacher_mesh":{"path":str(teacher),"sha256":sha(teacher)},"family_candidate_outcomes":{"SURFACE_TOPOLOGY":"QUALIFIED_CONSUMER_CANDIDATE","CAUSAL_RESPONSE":"EXISTING_EVIDENCE_CONSUMED__NEW_FAMILY_COURT_STILL_REQUIRED"}}


def test_tessa_v5_court_accepts_qualified_mechanical_surface(tmp_path):
    result=validate(fixture(tmp_path))
    assert result["status"]=="PASS_CORPUS_MECHANICAL_TESSA_V5_CONSUMER_COURT__CAUSAL_RESPONSE_AUTHORITY_SEPARATE"
    assert result["qualified_cross_owner_seam_edge_count"]==1
    assert result["authority"]["training_authorized"] is False


def test_tessa_v5_rejects_sdf_sign_topology_dependency(tmp_path):
    m=fixture(tmp_path); m["source_topology"]["sdf_sign_topology_consumed"]=True
    with pytest.raises(ValueError,match="TESSA_V5_COURT_AUTHORITY_CROSS_CONTAMINATION"): validate(m)


def test_tessa_v5_rejects_seam_count_drift(tmp_path):
    m=fixture(tmp_path); m["source_topology"]["qualified_cross_owner_seam_edge_count"]=2
    with pytest.raises(ValueError,match="TESSA_V5_COURT_SEAM_COUNT"): validate(m)
