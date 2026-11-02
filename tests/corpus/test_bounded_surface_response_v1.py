import numpy as np
import pytest
from models.iris.truth import presentation_response_v1 as reference
from tools.training.bounded_surface_response_v1 import build_surface_response_arrays_v1
from tools.training import build_truth139_v32_mechanical_tessa_teacher_v1 as tessa


def apparatus():
    n=10
    v=np.array([[x,y,.01*x*y] for y in range(n) for x in range(n)],float)
    f=np.array([t for y in range(n-1) for x in range(n-1)
                for t in ((y*n+x,y*n+x+1,(y+1)*n+x),
                          (y*n+x+1,(y+1)*n+x+1,(y+1)*n+x))])
    posed=np.stack([v+[0,0,.1],v*np.array([1.1,.95,1.]),v*np.array([.9,1.05,1.])])
    actions=[{'control_index_privileged':0,'amplitude':.1,'sign':s} for s in (1,-1,1)]
    return v,f,posed,actions


@pytest.mark.parametrize('chunk',[1,7,128,10000])
def test_partition_keeps_all_faces_actions_and_cross_partition_adjacency(chunk):
    v,f,posed,actions=apparatus()
    expected=reference.build_surface_response_arrays_v1(v,f,posed,actions)
    actual=build_surface_response_arrays_v1(v,f,posed,actions,face_chunk_size=chunk)
    assert actual.keys()==expected.keys()
    for key in expected:
        if expected[key].dtype.kind in 'biu':np.testing.assert_array_equal(actual[key],expected[key])
        else:np.testing.assert_allclose(actual[key],expected[key],rtol=1e-13,atol=1e-13)


def test_working_deformation_tensor_has_bounded_face_dimension(monkeypatch):
    v,f,posed,actions=apparatus();shapes=[];original=reference.triangle_metric_strain
    def measured(rest,posed,**kw):
        shapes.append(posed.shape)
        return original(rest,posed,**kw)
    monkeypatch.setattr(reference,'triangle_metric_strain',measured)
    build_surface_response_arrays_v1(v,f,posed,actions,face_chunk_size=7)
    assert sum(s[1] for s in shapes)==len(f)
    assert all(s[0]==len(actions) and s[1]<=7 for s in shapes)


def test_mapping_transfer_and_global_strain_quantiles_keep_original_values():
    v,f,posed,_=apparatus()
    queries=np.concatenate([v, v+.001])
    expected=tessa.nearest_mapping(queries,v,f,query_chunk_size=10000)
    for chunk in (1,7,64):
        actual=tessa.nearest_mapping(queries,v,f,query_chunk_size=chunk)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    face_id,bar,_=expected
    original=np.sum(posed[:,f[face_id],:]*bar[None,:,:,None],axis=2)
    np.testing.assert_array_equal(tessa.transfer_positions(face_id,bar,f,posed),original)
    rest=v[f];pt=posed[:,f,:];rr=np.broadcast_to(rest[None],pt.shape)
    scale=max(float(np.linalg.norm(v.max(0)-v.min(0))),1e-9)
    strain,area,valid=reference.triangle_metric_strain(rr,pt,degeneracy_tolerance=1e-10*scale*scale)
    eig=np.clip(np.linalg.eigvalsh(2*strain+np.eye(2)),1e-12,None)
    dist=np.max(np.abs(np.log(np.sqrt(eig))),axis=-1)
    wanted=(bool(valid.all()),float(np.quantile(dist[valid],.99)),float(np.quantile(np.abs(np.log(np.clip(area,1e-12,None)))[valid],.99)))
    np.testing.assert_allclose(tessa.strain_tail(v,f,posed),wanted,rtol=1e-13,atol=1e-13)
