import json

import pytest

from models.materialization_contract_v1 import sha256
from tools.notebooks.build_corpus370_materialization_v1 import build


def test_notebook_cannot_be_built_from_an_open_checkpoint(tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'schema': 'RealSaS.Corpus370MaterializationCheckpoint.v1',
                                     'status': 'BLOCKED_PRE_MATERIALIZATION', 'training_authorized': False}))
    seal = tmp_path / 'seal.json'
    seal.write_text('{}')
    with pytest.raises(ValueError, match='CHECKPOINT_NOT_READY'):
        build(checkpoint, seal, zip_sha256='1' * 64, drive_bundle_path='corpus.zip', code_sha='2' * 40)


def test_generated_cells_compile_and_pin_exact_artifacts_without_training(tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'schema': 'RealSaS.Corpus370MaterializationCheckpoint.v1',
        'status': 'PASS_PRE_MATERIALIZATION_CHECKPOINT', 'counts_are_final': True, 'materialization_authorized': True,
        'training_authorized': False, 'models': {'IRIS': {'eligible_subject_count': 1,
         'partial_supervision_subject_count': 1, 'split_counts': {'train': 1, 'val': 0, 'global_unseen': 0}}}}))
    seal = tmp_path / 'seal.json'
    value = {'schema': 'RealSaS.Corpus370MaterializationBundle.v1', 'status': 'PASS_SEALED_STATIC_CORPUS_BUNDLE',
             'training_authorized': False, 'checkpoint': {'sha256': sha256(checkpoint)}}
    seal.write_text(json.dumps(value))
    notebook = build(checkpoint, seal, zip_sha256='1' * 64, drive_bundle_path='corpus/bundle.zip', code_sha='2' * 40)
    for c in notebook['cells']:
        if c['cell_type'] == 'code': compile(''.join(c['source']), '<materialization-cell>', 'exec')
    assert notebook['metadata']['realsas']['seal_sha256'] == sha256(seal)
    assert len(notebook['cells']) == 4
    value['checkpoint']['sha256'] = '0' * 64
    seal.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='BUNDLE_CHECKPOINT_BINDING'):
        build(checkpoint, seal, zip_sha256='1' * 64, drive_bundle_path='corpus.zip', code_sha='2' * 40)
