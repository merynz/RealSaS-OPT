"""Synthetic adversarial checks for joint native+source shader palette workload."""
import copy

import pytest
from tools.corpus.compile_corpus370_operational_edit_algebra_v1 import INPUT_SCHEMA
from tools.corpus.compile_corpus370_operational_palette_joint_v1 import compile_joint

SHA="a"*64
PAL_SHA="b"*64
NATIVE_SHA="c"*64

def samples():
    native={"schema":INPUT_SCHEMA,
        "source_original_sha256":SHA,
        "source_rest_sha256":"d"*64,
        "source_face_order_sha256":"e"*64,
        "source_face_count":5,"workload_policy":"SYNTHETIC_NATIVE_TEST",
        "operations":[{"operation_id":"NATIVE_X",
            "support":[True,True,True,False,False],
            "validity":[True]*5,"evidence_sha256":"f"*64}]}
    palette={"schema":"RealSaS.Corpus370OriginalShaderConstantPaletteFacts.v1",
        "original_source_sha256":SHA,
        "source_face_order_sha256":"e"*64,
        "source_face_count":5,
        "face_palette_validity":[True,True,True,True,False],
        "face_palette_color_key_index_or_null":[0,1,0,1,None],
        "constant_palette_color_keys_hex_rgba":[
            [float(0.25).hex()]*3+[float(1).hex()],
            [float(0.8).hex()]*3+[float(1).hex()]],
        "palette_operation_count":2,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False}
    court={"schema":"RealSaS.Corpus370IndependentOriginalShaderPaletteCourt.v1",
        "status":"INDEPENDENT_SOURCE_SHADER_ADDRESS_VERIFIED__NO_OWNER_AUTHORITY",
        "palette_producer_sha256":PAL_SHA,
        "original_source_sha256":SHA,
        "original_face_count":5,
        "graphics_owner_authorized":False,
        "training_authorized":False}
    return native,palette,court

def test_palette_refines_native_and_unknown_remains_unknown():
    n,p,c=samples()
    joint,dec,report=compile_joint(n,p,c,PAL_SHA,NATIVE_SHA)
    assert len(joint['operations'])==3
    assert report['joint_candidate_atom_count']==4
    assert report['joint_unknown_placeholder_count']==1
    assert report['source_shader_constant_qualified_faces']==4
    assert report['source_shader_unsupported_faces']==1
    assert dec['iris_training_authorized'] is False
    assert report['actual_blender_source_edit_court']=='NOT_EVALUATED'

def test_no_constant_palette_does_not_become_negative():
    n,p,c=samples()
    p['face_palette_validity']=[False]*5
    p['face_palette_color_key_index_or_null']=[None]*5
    p['constant_palette_color_keys_hex_rgba']=[]
    p['palette_operation_count']=0
    joint,dec,report=compile_joint(n,p,c,PAL_SHA,NATIVE_SHA)
    assert len(joint['operations'])==1
    assert report['palette_task_count']==0
    assert report['joint_candidate_atom_count']==2

def test_unknown_face_cannot_silently_become_palette_negative():
    n,p,c=samples()
    p['face_palette_color_key_index_or_null'][-1]=0
    with pytest.raises(ValueError,match='PALETTE_DOMAIN'):
        compile_joint(n,p,c,PAL_SHA,NATIVE_SHA)

def test_independent_court_receipt_required():
    n,p,c=samples()
    c['status']='NOT_VERIFIED'
    with pytest.raises(ValueError,match='INDEPENDENT_COURT'):
        compile_joint(n,p,c,PAL_SHA,NATIVE_SHA)

def test_forged_source_sha_rejected():
    n,p,c=samples()
    p['original_source_sha256']="2"*64
    with pytest.raises(ValueError,match='SOURCE_SHA'):
        compile_joint(n,p,c,PAL_SHA,NATIVE_SHA)

def test_identical_palette_follows_explicit_shared_control_policy():
    n,p,c=samples()
    p['face_palette_color_key_index_or_null']=[0,0,0,0,None]
    p['constant_palette_color_keys_hex_rgba']=[p['constant_palette_color_keys_hex_rgba'][0]]
    p['palette_operation_count']=1
    joint,dec,report=compile_joint(n,p,c,PAL_SHA,NATIVE_SHA)
    assert report['palette_task_count']==1
    assert report['joint_candidate_atom_count']==3
    assert report['graphics_owner_authorized'] is False
