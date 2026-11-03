import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from tools.corpus.source_rest_geometry_v1 import build,decode_import
from tools.corpus.validate_source_rest_geometry_v1 import validate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    eye=np.eye(4).tolist();recipe={'source_sha256':'a'*64,'operation':'test_bind_import','importer_options':{},
        'pose_policy':'SOURCE_BIND_REST__NO_ANIMATION','raw_source_mutated':False,'excluded_import_controls':[]}
    vertices=[[3.,0.,0.],[4.,0.,0.],[3.,1.,0.],[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]
    objects=[{'name':name,'vertex_count':3,'world':eye} for name in ('B','A')]
    ip=root/'import.json';ip.write_text(json.dumps({'vertices':vertices,'faces':[[0,1,2],[3,4,5]],'objects':objects,'bone_count':0,'import':recipe}))
    nodes=[{'node_id':'node:'+name,'source_import_name':name,'type':'MESH','parent_node_id':None,'parent_type':'OBJECT','parent_bone':None,
            'node_to_source_world':eye,'effective_node_to_parent':eye,'authored_import_matrix_basis':eye,
            'matrix_parent_inverse':eye,'node_to_canonical':eye,'active_action_count':0,'active_nla_track_count':0,'driver_count':0} for name in ('A','B')]
    sp=root/'snapshot.json';sp.write_text(json.dumps({'schema':'RealSaS.ImportedSourceSceneSnapshot.v2','source_sha256':'a'*64,
        'source_world_to_canonical':eye,'canonical_to_source_world':eye,'nodes':nodes,'bones':[]}))
    cp=root/'candidate.json';cp.write_text(json.dumps({'schema':'RealSaS.Controlled8ViewSourceRestCandidate.v2','source_sha256':'a'*64,
        'subject_id':'subject','source_scene_snapshot':{'sha256':sha(sp)},'import':recipe}))
    ap=root/'admission.json';ap.write_text(json.dumps({'schema':'RealSaS.IrisBindObservationAdmission.v1','status':'PASS_IRIS_BIND_OBSERVATIONS',
        'subject_id':'subject','source_sha256':'a'*64,'candidate_sha256':sha(cp),'imported_geometry_sha256':sha(ip),
        'imported_vertex_count':6,'bone_count':0}))
    return ip,ap,cp,sp


def test_zero_bones_and_no_textures_do_not_block_surface_acquisition(tmp_path):
    files=fixture(tmp_path);m=build('subject','a'*64,*files,tmp_path/'out')
    path=tmp_path/'out/SOURCE_REST_GEOMETRY_MANIFEST_V1.json';court=validate(path)
    assert court['authority']['independent_source_rest_reconstruction_passed'] is True
    assert m['authority']['kinematic_response_measured'] is False and m['authority']['training_authorized'] is False
    assert m['owner_order']==['A','B']
    with np.load(tmp_path/'out/SOURCE_REST_GEOMETRY_V1.npz') as z:
        assert set(z.files)=={'rest_vertices_source_world','faces','face_owner_privileged'}
        assert z['rest_vertices_source_world'][0].tolist()==[0.,0.,0.]
        assert z['face_owner_privileged'].tolist()==[0,1]


@pytest.mark.parametrize('mutation',['vertex','winding','extra_response'])
def test_independent_court_rejects_resigned_payload_mutation(tmp_path,mutation):
    files=fixture(tmp_path);build('subject','a'*64,*files,tmp_path/'out')
    payload=tmp_path/'out/SOURCE_REST_GEOMETRY_V1.npz'
    with np.load(payload) as z:a={k:z[k].copy() for k in z.files}
    if mutation=='vertex':a['rest_vertices_source_world'][0,0]+=.1
    elif mutation=='winding':a['faces'][0]=a['faces'][0,::-1]
    else:a['isolated_position']=np.zeros((1,6,3))
    np.savez_compressed(payload,**a)
    path=tmp_path/'out/SOURCE_REST_GEOMETRY_MANIFEST_V1.json';m=json.loads(path.read_text());m['output']['sha256']=sha(payload);path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='SOURCE_REST_COURT_'):validate(path)


def test_admitted_import_bytes_guard_remains_exact(tmp_path):
    files=fixture(tmp_path);ip=files[0];ip.write_text(ip.read_text()+' ')
    with pytest.raises(ValueError,match='EXACT_IMPORT_BYTES_DRIFT'):build('subject','a'*64,*files,tmp_path/'out')


def test_cross_object_triangles_cannot_acquire_an_owner():
    value={'vertices':[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],'faces':[[0,1,3]],
        'objects':[{'name':'A','vertex_count':3},{'name':'B','vertex_count':1}]}
    with pytest.raises(ValueError,match='CROSS_OBJECT_FACE'):decode_import(value)


def test_independent_geometry_and_normals_court_works_without_teacher_scene(tmp_path):
    from tools.corpus.build_component_geometry_authority_v1 import build as geometry
    from tools.corpus.validate_component_geometry_authority_v1 import validate as geometry_court
    files=fixture(tmp_path);build('subject','a'*64,*files,tmp_path/'rest')
    camera=tmp_path/'camera.json';camera.write_text(json.dumps({'schema':'Synthetic.Camera.v1'}))
    m=geometry('subject',tmp_path/'rest/SOURCE_REST_GEOMETRY_V1.npz',files[3],camera,tmp_path/'geometry',surface_samples=32,query_samples=32)
    result=geometry_court(m)
    assert result['family_candidate_outcomes']['SURFACE_GEOMETRY']=='QUALIFIED_CORE_CANDIDATE'
    assert result['family_candidate_outcomes']['ORIENTED_NORMALS']=='QUALIFIED_CORE_CANDIDATE'
    assert result['authority']['training_authorized'] is False
