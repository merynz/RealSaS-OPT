from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
import pytest

from tests.corpus.test_gltf_animation_authority_v1 import write_source
from tools.corpus.build_gltf_animation_authority_v1 import build as build_animation
from tools.corpus.validate_gltf_animation_authority_v1 import validate as validate_animation
from tools.corpus.build_gltf_rig_skin_authority_v1 import build as build_rig
from tools.corpus.validate_gltf_rig_skin_authority_v1 import validate as validate_rig
from tools.corpus.validate_gltf_imported_rig_skin_equivalence_v1 import validate as validate_import
from tools.corpus.build_gltf_animation_canonical_replay_v1 import build as build_replay
from tools.corpus.validate_gltf_animation_canonical_replay_v1 import validate as validate_replay


def sha(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()

def imported_fixture(root:Path,source:Path):
    doc=json.loads(source.read_text()); import base64
    # The animation test source has one triangle and one source joint (Root).
    positions=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float64)
    imported=positions[:,[0,2,1]].copy();imported[:,1]*=-1
    geometry=root/'geometry_and_response.npz'
    np.savez_compressed(geometry,rest_vertices_source_world=imported,faces=np.asarray([[0,1,2]],np.int64),source_weight_offsets=np.asarray([0,1,2,3],np.int64),source_weight_bone_indices_privileged=np.asarray([0,0,0],np.int64),source_weight_values=np.asarray([1.,1.,1.],np.float64),source_weight_coverage=np.asarray([True,True,True]))
    teacher=root/'TEACHER_SCENE.json';teacher.write_text(json.dumps({'schema':'RealSaS.TeacherSceneIR.v2','subject_key':'synthetic-animation','authority':'UNQUALIFIED_DERIVED_MEASUREMENTS','geometry_payloads':[{'path':str(geometry.resolve()),'sha256':sha(geometry),'size_bytes':geometry.stat().st_size}],'calibration':{'native_control_frames_privileged':[{'armature':'Rig','name':'Root','parent':None,'rest_matrix_world':np.eye(4).tolist()}]}},sort_keys=True)+'\n')
    return teacher,geometry

def build_stack(root:Path, version=1):
    source=root/'animated.gltf';write_source(source)
    if version==2:
        from tools.corpus.build_gltf_rig_skin_authority_v2 import build as rig_producer
        from tools.corpus.validate_gltf_rig_skin_authority_v2 import validate as rig_court
    else:rig_producer,rig_court=build_rig,validate_rig
    rig_dir=root/'rig';rm=rig_producer(source,rig_dir);rc=rig_court(rm);rc_path=rig_dir/f'GLTF_RIG_SKIN_AUTHORITY_COURT_V{version}.json';rc_path.write_text(json.dumps(rc,sort_keys=True,indent=2)+'\n')
    rig_manifest=rig_dir/f'GLTF_RIG_SKIN_AUTHORITY_MANIFEST_V{version}.json'
    teacher,geometry=imported_fixture(root,source);ri=validate_import(rig_manifest,rc_path,teacher,geometry);ri_path=rig_dir/'GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE_COURT_V1.json';ri_path.write_text(json.dumps(ri,sort_keys=True,indent=2)+'\n')
    anim_dir=root/'animation';am=build_animation(source,anim_dir);ac=validate_animation(am);ac_path=anim_dir/'GLTF_ANIMATION_KINEMATICS_COURT_V1.json';ac_path.write_text(json.dumps(ac,sort_keys=True,indent=2)+'\n')
    replay_path=root/'ANIMATION_CANONICAL_REPLAY_V1.json';build_replay(anim_dir/'GLTF_ANIMATION_KINEMATICS_MANIFEST_V1.json',ac_path,rig_manifest,ri_path,teacher,replay_path)
    return replay_path

@pytest.mark.parametrize('version',[1,2])
def test_animation_canonical_replay_roundtrip(tmp_path:Path,version):
    replay=build_stack(tmp_path,version);court=validate_replay(replay)
    assert court['status']=='PASS_GLTF_ANIMATION_CANONICAL_REPLAY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE'
    assert court['family_candidate_outcomes']['ANIMATION_KINEMATICS']=='QUALIFIED_CORE_CANDIDATE'
    assert court['authority']['canonical_replay_court_passed'] is True
    assert court['authority']['training_authorized'] is False

def test_animation_replay_digest_mutation_fails_closed(tmp_path:Path):
    replay=build_stack(tmp_path);value=json.loads(replay.read_text());value['animations'][0]['replay_digest_sha256']='0'*64;replay.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    with pytest.raises(ValueError,match='ANIM_REPLAY_COURT_REPLAY_DRIFT'):validate_replay(replay)

def test_animation_replay_requires_imported_rig_equivalence(tmp_path:Path):
    replay=build_stack(tmp_path);value=json.loads(replay.read_text());ri=Path(value['inputs']['rig_importer_court']['path']);payload=json.loads(ri.read_text());payload['status']='FAIL';ri.write_text(json.dumps(payload,sort_keys=True)+'\n');value['inputs']['rig_importer_court']['sha256']=sha(ri);replay.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    with pytest.raises(ValueError,match='ANIM_REPLAY_COURT_RIG_IMPORT_NOT_PASS'):validate_replay(replay)
