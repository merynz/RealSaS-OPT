import hashlib
import json
import numpy as np
import pytest

from tools.corpus import causal_surface_response_v2 as measure
from tools.corpus import build_causal_response_authority_v2 as producer
from tools.corpus import validate_causal_response_authority_v2 as court
from tools.corpus.build_mechanical_surface_topology_manifest_v1 import build as mechanical
from tools.corpus.validate_mechanical_surface_topology_authority_v1 import validate as mechanical_court


def actions():
    return [{'control_index_privileged':i//2,'amplitude':.1,'sign':1 if i%2 else -1} for i in range(4)]


def compare_independent(rest, faces, posed, *, scale=4.):
    a=measure.build_surface_response_arrays_v2(rest,faces,posed,actions(),source_scale=scale,face_chunk_size=2)
    b=court.deformation_measure(rest,faces,posed,court.control_weights(actions()),scale)
    names=['surface_face_valid','surface_stretch_major_mean','surface_stretch_minor_mean',
           'surface_area_ratio_mean','surface_anisotropy_mean','surface_tangent_valid',
           'surface_tangent_direction_2theta','surface_nonrigid_response']
    for name,expected in zip(names,b):
        if expected.dtype==bool: np.testing.assert_array_equal(a[name],expected)
        else: np.testing.assert_allclose(a[name],expected,atol=2e-9,rtol=2e-9)
    return a


def test_independent_world_frame_agrees_and_ignores_cyclic_face_address():
    rng=np.random.default_rng(3411)
    rest=rng.normal(size=(24,3)); faces=np.arange(24).reshape(-1,3)
    posed=np.stack([rest*np.array([1.+.1*i,1.-.03*i,1.+.02*i])+[0.,0.,.1*i] for i in range(1,5)])
    a=compare_independent(rest,faces,posed)
    b=compare_independent(rest,faces[:,[1,2,0]],posed)
    for name in a:
        np.testing.assert_allclose(a[name],b[name],atol=2e-9,rtol=2e-9)
    assert a['surface_tangent_valid'].all()


def test_conditioning_and_collapsed_actions_remain_absent_targets():
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[.3,1e-6,0.], [0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
    faces=np.array([[0,1,2],[3,4,5]])
    posed=np.repeat(rest[None],4,axis=0); posed[:,5]=posed[:,4]
    a=compare_independent(rest,faces,posed,scale=2.)
    assert not a['surface_face_valid'].any() and not a['surface_tangent_valid'].any()


def stack(root):
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    faces=np.array([[0,1,2],[2,1,3]])
    posed=np.stack([rest*np.array([1.+.1*i,1.,1.])+[0.,0.,.02*i] for i in range(1,5)])
    source=root/'geometry_and_response.npz'
    np.savez_compressed(source,rest_vertices_source_world=rest,faces=faces,face_owner_privileged=np.zeros(2,int),isolated_position=posed)
    mechanical('s',source,root/'mechanical')
    mp=root/'mechanical/MECHANICAL_SURFACE_TOPOLOGY_MANIFEST_V1.json'
    cp=root/'mechanical/court.json'; cp.write_text(json.dumps(mechanical_court(mp)))
    scene=root/'scene.json'; scene.write_text(json.dumps({'schema':'RealSaS.TeacherSceneIR.v2','subject_key':'s',
        'geometry_payloads':[{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}],
        'calibration':{'actions':actions()}}))
    producer.build('s',mp,cp,scene,root/'response')
    return root/'response/CAUSAL_RESPONSE_AUTHORITY_MANIFEST_V2.json'


def test_hash_bound_v2_chain_has_separate_schema_and_no_training_authority(tmp_path):
    p=stack(tmp_path); result=court.validate(p)
    assert result['schema']=='RealSaS.CausalResponseAuthorityCourt.v2'
    assert result['counts']['qualified_face_count']==2
    assert result['authority']['training_authorized'] is False
    from tools.corpus.validate_causal_response_authority_v1 import validate
    with pytest.raises(ValueError,match='CAUSAL_COURT_PARENT'): validate(p)


@pytest.mark.parametrize('mutation',['policy','scalar','shape','tangent'])
def test_court_rejects_resigned_numeric_and_contract_mutations(tmp_path,mutation):
    p=stack(tmp_path); m=json.loads(p.read_text())
    if mutation=='policy': m['measurement_policy']['tangent_frame']='LEGACY_UNDECLARED_QR'
    else:
        target=tmp_path/'response/CAUSAL_RESPONSE_AUTHORITY_V2.npz'
        with np.load(target) as z: arrays={k:z[k].copy() for k in z.files}
        if mutation=='scalar': arrays['stretch_major'][0]+=.01
        elif mutation=='shape': arrays['stretch_major']=arrays['stretch_major'][:,None]
        else: arrays['tangent_direction_2theta'][0]*=-1
        np.savez_compressed(target,**arrays); m['output']['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='CAUSAL_'): court.validate(p)


def test_trajectory_drift_guard_is_not_relaxed():
    raw=np.array([[[0.,0.,0.],[1e-7,0.,0.]]])
    with pytest.raises(ValueError,match='TRAJECTORY_DRIFT'):
        producer.collapse_trajectory(raw,np.array([0,0]),1,1.)
