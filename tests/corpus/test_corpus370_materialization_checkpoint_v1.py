import copy
import hashlib
import json

import pytest

from tools.corpus.build_corpus370_materialization_checkpoint_v1 import build, check_split, render
from tools.corpus.build_corpus370_subject_family_authority_matrix_v1 import build as matrix_build


def test_partial_evidence_is_visible_but_cannot_authorize_a_notebook(tmp_path):
    candidate = tmp_path / 'candidate.json'
    candidate.write_text(json.dumps({'subjects': [{'subject_id': str(i), 'source_sha256': hashlib.sha256(str(i).encode()).hexdigest()}
                                                  for i in range(370)], 'subject_count': 370}))
    matrix = matrix_build(__import__('pathlib').Path('canonical/CORPUS370_SUBJECT_FAMILY_AUTHORITY_POLICY_V1.json'), candidate, [])
    matrix['readback_scope'] = 'PARTIAL_DIAGNOSTIC_PROJECTION__NOT_FINAL_ADMISSION_OR_READINESS'
    path = tmp_path / 'matrix.json'
    path.write_text(json.dumps(matrix))
    checkpoint = build(path)
    assert checkpoint['status'] == 'BLOCKED_PRE_MATERIALIZATION'
    assert checkpoint['counts_are_final'] is False and checkpoint['materialization_authorized'] is False
    assert checkpoint['models']['IRIS']['eligible_subject_count'] == 0
    assert all(s['family_status']['GLOBAL_SOLID_SDF'] == 'UNKNOWN' for s in checkpoint['models']['IRIS']['subjects'])
    assert checkpoint['models']['IRIS']['head_subject_coverage']['global_sdf_sign']['all_eligible_subjects'] == 0
    assert 'PENDING' in render(checkpoint)


def test_cross_model_global_unseen_cannot_leak_into_train():
    roster = {'models': {m: {'eligible_subjects': [{'subject_id': x} for x in ('a', 'b', 'c')]} for m in ('IRIS', 'TESSA')}}
    split = {'schema': 'RealSaS.Corpus370LineageSplit.v1', 'status': 'PASS_GLOBAL_PRODUCT_UNSEEN_AND_MODEL_TRAIN_VAL_SPLIT_SEALED',
             'split_authorized': True, 'training_authorized': False, 'global_unseen': {'subject_ids': ['c']},
             'models': {m: {'train_subject_ids': ['a'], 'val_subject_ids': ['b'], 'global_unseen_subject_ids': ['c']} for m in roster['models']}}
    assert check_split(split, roster, {'a', 'b', 'c'})['IRIS']['global_unseen'] == ['c']
    forged = copy.deepcopy(split)
    forged['models']['TESSA']['train_subject_ids'] = ['a', 'c']
    forged['models']['TESSA']['global_unseen_subject_ids'] = []
    with pytest.raises(ValueError, match='GLOBAL_UNSEEN_LEAKAGE'):
        check_split(forged, roster, {'a', 'b', 'c'})
