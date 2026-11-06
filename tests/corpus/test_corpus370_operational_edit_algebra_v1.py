"""Contract tests for finite Boolean edit algebra. SYNTHETIC ONLY."""
import copy
import random

import pytest

from tools.corpus.compile_corpus370_operational_edit_algebra_v1 import (
    compile_evidence, classify_pair,
)
from tools.corpus.court_corpus370_operational_edit_algebra_v1 import court

H1 = '1' * 64
H2 = '2' * 64
H3 = '3' * 64
H4 = '4' * 64


def make(support, validity=None):
    n = len(next(iter(support.values())))
    tasks = []
    for name, values in support.items():
        tasks.append({'operation_id': name,
                      'support': list(values),
                      'validity': list((validity or {}).get(name, [True] * n)),
                      'evidence_sha256': H4})
    return {'schema': 'RealSaS.OperationalEditWorkloadEvidence.v1',
            'source_original_sha256': H1, 'source_rest_sha256': H2,
            'source_face_order_sha256': H3,
            'source_face_count': n, 'workload_policy': 'SYNTHETIC_TEST_ONLY',
            'operations': tasks}


def blocks(report):
    return {frozenset(x['face_indices']) for x in report['atoms']}


def test_coarsest_not_renderer_partition_and_independent_court():
    evidence = make({'tint-metal': [True, True, False, False, False, True],
                     'tint-affected': [True, True, True, False, False, True]})
    result = compile_evidence(evidence)
    assert blocks(result) == {frozenset([0, 1, 5]), frozenset([2]), frozenset([3, 4])}
    assert result['audit']['atom_count'] == 3
    assert court(evidence, result)['exhaustive_oracle_minimum_K'] == 3
    assert result['artist_drawable_owner_authorized'] is False
    assert result['iris_training_authorized'] is False


def test_unknown_is_not_negative_and_not_inferred_same_drawable():
    evidence = make({'a': [True, True, False], 'b': [False, False, False]},
                    {'a': [True, True, True], 'b': [True, False, True]})
    result = compile_evidence(evidence)
    assert blocks(result) == {frozenset([0]), frozenset([1]), frozenset([2])}
    assert result['audit']['unknown_face_count'] == 1
    tasks = sorted(evidence['operations'], key=lambda x: x['operation_id'])
    assert classify_pair(tasks, 0, 1)['relation'] == 'UNKNOWN'
    assert classify_pair(tasks, 1, 2)['relation'] == 'MUST_SEPARATE_FOR_WORKLOAD'
    assert court(evidence, result)['unknown_faces'] == 1


def test_incomplete_compatibility_not_transitive():
    evidence = make({'a': [True, True, False], 'b': [False, True, True]},
                    {'a': [True, False, True], 'b': [False, False, False]})
    tasks = sorted(evidence['operations'], key=lambda t: t['operation_id'])
    assert classify_pair(tasks, 0, 1)['relation'] == 'UNKNOWN'
    assert classify_pair(tasks, 1, 2)['relation'] == 'UNKNOWN'
    assert classify_pair(tasks, 0, 2)['relation'] == 'MUST_SEPARATE_FOR_WORKLOAD'
    assert court(evidence, compile_evidence(evidence))['status'].startswith('PASS_ALGEBRA')


def test_different_task_order_preserves_atoms():
    evidence = make({'z': [True, False, True], 'a': [False, False, True]})
    a = compile_evidence(evidence)
    evidence['operations'].reverse()
    b = compile_evidence(evidence)
    assert blocks(a) == blocks(b)
    assert [x['atom_id'] for x in a['atoms']] == [x['atom_id'] for x in b['atoms']]
    assert court(evidence, b)['atom_count'] == 3


def test_task_family_refines_edit_algebra():
    a = make({'palette': [True, True, False, False]})
    b = make({'palette': [True, True, False, False],
              'affected_region': [True, False, True, False]})
    pa, pb = compile_evidence(a), compile_evidence(b)
    assert len(pa['atoms']) == 2
    assert len(pb['atoms']) == 4
    assert court(a, pa)['atom_count'] == 2
    assert court(b, pb)['atom_count'] == 4


def test_uniform_full_character_hide_does_not_prove_owner():
    evidence = make({'full_character_hide': [True] * 6})
    result = compile_evidence(evidence)
    assert result['audit']['atom_count'] == 1
    assert result['audit']['same_drawable_training_pairs_qualified'] == 0
    assert result['audit']['original_artist_owner'] == 'UNKNOWN'
    assert court(evidence, result)['status'].startswith('PASS_ALGEBRA_ONLY')


def test_court_rejects_forged_merge_and_promotion():
    evidence = make({'a': [True, False, True], 'b': [False, False, False]})
    result = compile_evidence(evidence)
    bad = copy.deepcopy(result)
    bad['atoms'][0]['face_indices'].append(1)
    with pytest.raises(ValueError):
        court(evidence, bad)
    bad = copy.deepcopy(result)
    bad['iris_training_authorized'] = True
    from tools.corpus.court_corpus370_operational_edit_algebra_v1 import sha_obj
    bad.pop('report_sha256')
    bad['report_sha256'] = sha_obj(bad)
    with pytest.raises(ValueError, match='FALSE_AUTHORITY'):
        court(evidence, bad)


def test_court_rejects_source_or_measurement_drift():
    evidence = make({'a': [True, False, True]})
    result = compile_evidence(evidence)
    drift = copy.deepcopy(evidence)
    drift['source_original_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='INPUT_HASH'):
        court(drift, result)
    drift = copy.deepcopy(evidence)
    drift['operations'][0]['support'][0] = False
    with pytest.raises(ValueError, match='INPUT_HASH'):
        court(drift, result)


@pytest.mark.parametrize('seed', range(20))
def test_random_complete_small_cases_match_independent_exhaustive_oracle(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 7)
    t = rng.randint(1, 4)
    evidence = make({f'op-{j}': [rng.choice([True, False]) for _ in range(n)]
                     for j in range(t)})
    result = compile_evidence(evidence)
    receipt = court(evidence, result)
    assert receipt['exhaustive_oracle_minimum_K'] == len(result['atoms'])


@pytest.mark.parametrize('seed', range(10))
def test_random_partial_workloads_never_join_unknown(seed):
    rng = random.Random(seed + 100)
    n = rng.randint(2, 20)
    evidence = make({f'op-{j}': [rng.choice([True, False]) for _ in range(n)]
                     for j in range(3)},
                    {f'op-{j}': [rng.random() > 0.25 for _ in range(n)]
                     for j in range(3)})
    result = compile_evidence(evidence)
    receipt = court(evidence, result)
    assert receipt['unknown_faces'] == result['audit']['unknown_face_count']


def test_boolean_validation_rejects_int_masks():
    e = make({'a': [True, False]})
    e['operations'][0]['support'][1] = 0
    with pytest.raises(ValueError, match='BOOLEAN'):
        compile_evidence(e)
