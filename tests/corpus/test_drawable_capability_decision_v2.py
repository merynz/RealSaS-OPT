import itertools
import math
import numpy as np
import pytest

from tools.corpus.drawable_capability_decision_v2 import (
    partitions, reachable_states, compile_edit, compatible_pairs,
    feasible_total_orders, decide)
from tools.corpus.measure_drawable_capability_decision_v2 import depth_plan, ordered_over
from tools.corpus.measure_drawable_capability_decision_v2 import representations
from tools.corpus.drawable_capability_decision_v2 import AUTHORITY
from tools.corpus.court_drawable_capability_decision_v2 import validate_decision_tables,check_measurement


def test_exhaustive_boolean_control_compiler_against_scalar_oracle():
    for n in range(1,6):
        for groups in partitions(n):
            legal=[]
            for bits in itertools.product((0,1),repeat=n):
                good=all(groups[a]!=groups[b] or bits[a]==bits[b]
                         for a in range(n) for b in range(n))
                result=compile_edit(groups,bits)
                assert (result['status']=='SATISFIED')==good
                if good:legal.append(bits)
                else:
                    a,b=result['conflicting_addresses']
                    assert groups[a]==groups[b] and bits[a]!=bits[b]
            assert tuple(legal)==reachable_states(groups)


def test_partitions_bell_counts():
    assert [len(list(partitions(n))) for n in range(1,6)]==[1,2,5,15,52]


def test_label_permutation_does_not_change_control_capability():
    assert reachable_states([0,1,0])==reachable_states([8,3,8])


def test_approximate_compatibility_is_not_an_equivalence_class():
    r=compatible_pairs([[0,.02,.04],[.02,0,.02],[.04,.02,0]],.025)
    assert [e['status'] for e in r]==['COMPATIBLE','DISTINGUISHED','COMPATIBLE']


def test_unknown_comparison_is_not_equivalence():
    assert compatible_pairs([[0,None],[None,0]],.025)[0]['status']=='UNKNOWN'


def test_cycle_rejects_flat_order_not_drawable_ownership():
    assert feasible_total_orders([(0,1),(1,0)],3)==()
    assert feasible_total_orders([(0,1),(1,2),(2,0)],3)==()
    assert len(feasible_total_orders([(0,1)],3))==3


def test_precedence_solver_exhaustively_matches_permutation_oracle():
    edges=list(itertools.permutations(range(3),2))
    for use in itertools.product((False,True),repeat=6):
        selected=[e for e,on in zip(edges,use) if on]
        expected=tuple(p for p in itertools.permutations(range(3)) if
            all(p.index(a)<p.index(b) for a,b in selected))
        assert feasible_total_orders(selected,3)==expected


def test_synthetic_crossing_layers_require_local_order():
    # Two samples: red is in front at x0, green at x1. Same two drawables.
    color=np.zeros((1,2,1,2,4));color[:,0,...,0]=1;color[:,1,...,1]=1
    color[...,3]=1
    depth=np.zeros_like(color);depth[...,3]=1
    depth[:,0,...,0]=[.2,.8];depth[:,1,...,0]=[.8,.2]
    plan,q,_=depth_plan(depth,color)
    target=np.array([[[[1,0,0,1],[0,1,0,1]]]])
    assert q.all() and np.array_equal(ordered_over(color,plan),target)
    for p in [(0,1),(1,0)]:
        assert np.any(ordered_over(color,p)!=target)


def test_depth_tie_is_unknown():
    rgba=np.ones((1,2,1,1,4));depth=rgba.copy();depth[...,:3]=.5
    _,known,_=depth_plan(depth,rgba)
    assert not known.any()


def test_no_color_without_geometry():
    rgba=np.ones((1,2,1,1,4));depth=np.zeros_like(rgba)
    with pytest.raises(ValueError,match='WITHOUT_GEOMETRY'):depth_plan(depth,rgba)


def test_nonbinary_alpha_requires_another_contract():
    rgba=np.ones((1,2,1,1,4));rgba[...,3]=.5
    with pytest.raises(ValueError,match='NONBINARY'):depth_plan(rgba,rgba)


@pytest.mark.parametrize('missing',['pose','attachment','secondary_motion','reveal'])
def test_full_contract_abstains_on_unmeasured_capability(missing):
    r=decide([{'id':'one','capabilities':{'rgb':'SATISFIED'}}],['rgb',missing])
    assert r['candidate_decisions'][0]['status']=='UNKNOWN'
    assert r['policy_pareto_frontier']==[] and r['satisfying_alternatives']==[]
    assert r['same_drawable_pairs']==r['different_drawable_pairs']==[]


def test_failure_dominates_missing_evidence():
    r=decide([{'id':'bad','capabilities':{'rgb':'VIOLATED'}}],['rgb','attachment'])
    assert r['candidate_decisions'][0]['status']=='VIOLATED'


def test_cost_policy_changes_recommendation_not_source_facts():
    rows=[{'id':'mask','capabilities':{'edit':'SATISFIED'},'costs':{'K':1,'mask':3}},
          {'id':'split','capabilities':{'edit':'SATISFIED'},'costs':{'K':3,'mask':0}},
          {'id':'unknown','capabilities':{},'costs':{'K':0,'mask':0}}]
    r=decide(rows,['edit'],['K','mask'])
    assert r['policy_pareto_frontier']==['mask','split']
    one=decide(rows,['edit'],['K'])
    assert one['policy_pareto_frontier']==['mask']
    assert one['satisfying_alternatives']==r['satisfying_alternatives']
    assert not one['policy_is_artist_ownership']


def test_equal_cost_ties_preserved():
    rows=[{'id':str(i),'capabilities':{'x':'SATISFIED'},'costs':{'k':1}} for i in range(3)]
    assert decide(rows,['x'],['k'])['policy_pareto_frontier']==['0','1','2']


@pytest.mark.parametrize('cost',[float('nan'),-1,float('inf'),True])
def test_invalid_cost_fails_closed(cost):
    with pytest.raises(ValueError):
        decide([{'id':'x','capabilities':{},'costs':{'k':cost}}],['x'],['k'])


@pytest.mark.parametrize('groups',[[],[-1],[True],[1.5]])
def test_malformed_controls_fail_closed(groups):
    with pytest.raises(ValueError):reachable_states(groups)


def test_duplicate_candidate_rejected():
    with pytest.raises(ValueError):decide([{'id':'a'},{'id':'a'}],['x'])


def test_unknown_runtime_cost_cannot_be_invented():
    with pytest.raises(ValueError):decide([{'id':'a'}],['x'],['runtime_ms'])


def fixture_decisions():
    cat={'domains':{'all_four_views':{'candidate_rows':[
        {'tied_address_groups':list(g),'control_count':len(set(g)),
         'source_address_status':'SATISFIED' if len(set(g))==3 else 'VIOLATED'}
        for g in partitions(3)]}}}
    tric={'global_order_status':'VIOLATED','camera_specific_order_status':'VIOLATED',
          'depth_local_status':'SATISFIED'}
    return {'decisions':representations(cat,tric),'triceratops':tric,
            'same_drawable_pairs':[],'different_drawable_pairs':[],**AUTHORITY}


def test_independent_decision_table_court_accepts_consistent_fixture():
    validate_decision_tables(fixture_decisions())


@pytest.mark.parametrize('attack',[
    'authority','owner','pose','policy','cost','frontier','drop_alternative','fake_status','compositor'])
def test_independent_decision_court_rejects_attacks(attack):
    r=fixture_decisions();d=r['decisions']
    if attack=='authority':r['training_authorized']=True
    elif attack=='owner':r['same_drawable_pairs']=[[0,1]]
    elif attack=='pose':d['cat_full_puppet_contract']['candidate_decisions'][4]['capabilities']['pose_deformation']='SATISFIED'
    elif attack=='policy':d['structural_policy_example']['policy_cost_axes']=['logical_drawables']
    elif attack=='cost':d['cat_edit_contract']['candidate_decisions'][4]['costs']['source_edit_addresses']=0
    elif attack=='frontier':d['structural_policy_example']['policy_pareto_frontier']=[]
    elif attack=='drop_alternative':d['cat_edit_contract']['satisfying_alternatives'].pop()
    elif attack=='fake_status':d['cat_edit_contract']['candidate_decisions'][0]['status']='SATISFIED'
    elif attack=='compositor':d['tric_compositing_contract']['candidate_decisions'][0]['capabilities']['source_compositing']='SATISFIED'
    with pytest.raises(ValueError):validate_decision_tables(r)


def test_physical_court_rejects_false_error_count():
    target=np.ones((1,1,1,4));pred=np.zeros_like(target);valid=np.ones((1,1,1),bool)
    with pytest.raises(ValueError,match='ERROR_COUNT'):
        check_measurement(pred,target,valid,{'mismatched_sample_pixels':0,'max_error':1,
                                            'first_failure_s_y_x':None})


def test_physical_court_rejects_false_counterexample():
    target=np.ones((1,1,1,4));pred=np.zeros_like(target);valid=np.ones((1,1,1),bool)
    with pytest.raises(ValueError,match='COUNTEREXAMPLE_ADDRESS'):
        check_measurement(pred,target,valid,{'mismatched_sample_pixels':1,'max_error':1,
                                            'first_failure_s_y_x':[0,1,0]})
