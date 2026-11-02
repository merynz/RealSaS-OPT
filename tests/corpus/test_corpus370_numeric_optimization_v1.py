import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.validate_causal_response_authority_v1 import deformation_measure
from tools.training import build_truth139_v32_mechanical_tessa_teacher_v1 as tessa


def test_action_batching_preserves_every_frozen_scalar_output_bit():
    path=Path('tests/fixtures/corpus370_numeric_optimization_v1/response_reference.json')
    reference=json.loads(path.read_text())
    assert reference['baseline_main_sha']=='c16d9a17d73a308322ead20e065fa54d3f2be019'
    args=[np.array(reference[k]) for k in ('rest','faces','posed','weights')]+[reference['scale']]
    result=deformation_measure(*args)
    for actual,expected in zip(result,reference['expected']):
        np.testing.assert_array_equal(actual,np.asarray(expected))


def test_source_ladder_cache_keeps_all_candidate_metrics_and_bindings_exact():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype=np.int64)
    poses=np.stack([vertices+np.array([0.,0.,.1]),vertices*np.array([1.01,1.,1.])])
    actions=[{'control_index_privileged':0,'amplitude':1.,'sign':i} for i in (1,-1)]
    cameras=[{'origin':[0.,0.,0.],'right':[1.,0.,0.],'screen_up':[0.,1.,0.],'half_extent':2.}]*8
    components=np.zeros(len(faces),dtype=np.int32)
    topology=tessa.topology_signature(len(vertices),faces,components)
    owners=np.zeros(len(faces),dtype=np.int32)
    context=tessa.source_evaluation_context(vertices,faces,poses,actions,cameras)
    source_args=(vertices,faces,components,topology,owners,set(),poses,actions,cameras)
    for candidate in (vertices.copy(),vertices*.995):
        baseline=tessa.evaluate_candidate(candidate,faces,*source_args,1.,'SOURCE_FALLBACK')
        cached=tessa.evaluate_candidate(candidate,faces,*source_args,1.,'SOURCE_FALLBACK',source_context=context)
        assert baseline[0]==cached[0]
        np.testing.assert_array_equal(baseline[1],cached[1])
        np.testing.assert_array_equal(baseline[2],cached[2])
    changed=poses.copy();changed[0,0,0]+=.001
    with pytest.raises(ValueError,match='CONTEXT_DRIFT'):
        tessa.evaluate_candidate(vertices,faces,vertices,faces,components,topology,owners,set(),changed,actions,cameras,1.,'SOURCE_FALLBACK',source_context=context)


@pytest.mark.parametrize('kind',['SIMPLIFIED','SOURCE_FALLBACK'])
def test_source_cache_never_overrides_a_measured_candidate_failure(monkeypatch,kind):
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    poses=(vertices+[.01,0.,0.])[None]
    actions=[{'control_index_privileged':0,'amplitude':.01,'sign':1}]
    _,components=tessa.topology_components(len(vertices),faces)
    topology=tessa.topology_signature(len(vertices),faces,components)
    context=tessa.source_evaluation_context(vertices,faces,poses,actions,[])
    monkeypatch.setattr(tessa,'has_self_intersection',lambda *_:True)
    metrics,*_=tessa.evaluate_candidate(vertices,faces,vertices,faces,components,topology,
        np.zeros(len(faces),np.int32),set(),poses,actions,[],1.,kind,source_context=context)
    assert metrics['self_intersection_pass'] is False and metrics['mechanical_pass'] is False
