"""Synthetic original-source chain tests; NEVER interpreted as real owner proof."""
import json
import numpy as np
import pytest
from tools.corpus.measure_corpus370_operational_native_support_v1 import (
    compile_native_fit,verify_original,sha)
from tools.corpus.compile_corpus370_operational_edit_algebra_v1 import (
    compile_evidence,canonical_bytes)
from tools.corpus.court_corpus370_operational_native_support_v1 import independent_real_court

def synthetic(tmp_path):
    faces=np.array([[0,1,2],[1,3,2],[2,3,4],[3,5,4]],np.int64)
    points=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,2,0],[1,2,0]],float)
    base=np.stack((points,points,points))
    controls=np.broadcast_to(base[:,None,None,None,:,:],(3,1,3,2,6,3)).copy()
    controls[0,0,0,:,0]+=[.2,0,0]
    controls[1,0,0,:,0]+=[.2,0,0]
    controls[2,0,0,:,0]+=[.2,0,0]
    controls[0,0,1,:,1]+=[0,.2,0]
    controls[2,0,1,:,1]+=[0,.2,0]
    paths={k:tmp_path/(k+'.json' if not k.endswith('npz') else k+'.npz')
        for k in ('v18a_manifest','v18a_npz','v13b_manifest',
                  'v13b_npz','v13b_court')}
    np.savez_compressed(paths['v18a_npz'],faces=faces,
        labels_0p045=[0,0,0,0],labels_0p1=[0,0,0,0],
        labels_0p18=[0,0,0,0])
    np.savez_compressed(paths['v13b_npz'],baseline_vertices=base,
        native_control_perturbed_vertices=controls)
    ident='9'*64
    m={'schema':'RealSaS.Corpus370NativeCarrierActuatorInterventionRender.v13b',
       'source_pose_articulation_array_sha256':sha(paths['v13b_npz']),
       'rest_source_sha256':'8'*64,'source_sha256':ident,
       'subject_id':'SYNTHETIC','source_face_count':len(faces),
       'original_source_teacher_pose_count':3}
    paths['v13b_manifest'].write_text(json.dumps(m))
    mc={'schema':'RealSaS.Corpus370AnonymousNativeMixedCarrierResponseCourt.v13b',
        'source_sha256':ident,'subject_id':'SYNTHETIC',
        'parents':{'native_anonymous_intervention_manifest_sha256':sha(paths['v13b_manifest']),
        'native_anonymous_intervention_vertex_array_sha256':sha(paths['v13b_npz'])},
        'graphics_owner_authorized':False}
    paths['v13b_court'].write_text(json.dumps(mc))
    a={'schema':'RealSaS.Corpus370NativeReactiveStainCandidate.v18a',
       'source_sha256':ident,'subject_id':'SYNTHETIC',
       'source_face_count':len(faces),
       'private_candidate_npz_sha256':sha(paths['v18a_npz']),
       'parent_source_immutable_sha256':{
         'v13b_source_intervention':sha(paths['v13b_npz']),
         'source_rest_geometry':m['rest_source_sha256']},
       'fit_source_authored_poses':[0,1],'heldout_source_authored_pose':2,
       'iris_training_authorized':False}
    paths['v18a_manifest'].write_text(json.dumps(a))
    return a,m,mc,paths

def pipeline(tmp_path):
    a,m,mc,paths=synthetic(tmp_path)
    with np.load(paths['v18a_npz'],allow_pickle=False) as az, \
         np.load(paths['v13b_npz'],allow_pickle=False) as mz:
        faces,base,controls=verify_original(a,az,m,mz,mc,paths)
        ev,manifest=compile_native_fit(faces,base,controls,a,m,paths)
        decision=compile_evidence(ev)
        paths['decision']=tmp_path/'decision.json'
        paths['source_report']=tmp_path/'report.json'
        paths['decision'].write_bytes(canonical_bytes(decision))
        paths['source_report'].write_bytes(canonical_bytes(manifest))
        result=independent_real_court(a,m,mc,az,mz,paths,ev,decision,manifest)
    return ev,decision,result,paths

def test_source_chain_and_independent_reconstruction(tmp_path):
    ev,decision,result,_=pipeline(tmp_path)
    assert ev['source_face_count']==4
    assert len(ev['operations'])==3
    assert result['face_task_fit_unknown_count']>0
    assert result['graphics_owner_authorized'] is False
    assert result['partition_comparisons']['OPERATIONAL_FULL_WORKLOAD']['sum_minimum_observed_face_mismatches_per_task']==0

def test_v18a_single_group_cannot_express_measured_region_selector(tmp_path):
    _,_,result,_=pipeline(tmp_path)
    assert result['partition_comparisons']['V18A_THRESHOLD_0.1']['group_task_conflict_count']>=1
    assert result['source_palette_shaders_verified'] is False

def test_reject_forged_original_identity(tmp_path):
    ev,decision,_,paths=pipeline(tmp_path)
    with np.load(paths['v18a_npz'],allow_pickle=False) as az, \
         np.load(paths['v13b_npz'],allow_pickle=False) as mz:
        a=json.loads(paths['v18a_manifest'].read_text())
        m=json.loads(paths['v13b_manifest'].read_text())
        mc=json.loads(paths['v13b_court'].read_text())
        manifest=json.loads(paths['source_report'].read_text())
        a['source_sha256']='a'*64
        with pytest.raises(ValueError,match='SOURCE_IDENTITY'):
            independent_real_court(a,m,mc,az,mz,paths,ev,decision,manifest)

def test_action_fit_masks_do_not_learn_heldout(tmp_path):
    ev,decision,result,_=pipeline(tmp_path)
    assert ev['operations'][1]['validity'].count(False)>0
    assert result['source_editor_edits_verified'] is False
    assert result['training_authorized'] is False
