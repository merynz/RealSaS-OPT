import json

import pytest

from models.materialization_contract_v1 import sha256
from tools.corpus.seal_corpus370_materialization_bundle_v1 import build
from test_corpus370_sealed_materialization_v1 import bundle


def test_export_preserves_original_bytes_and_full_subject_coverage(tmp_path, monkeypatch):
    _, source, _, manifest = bundle(tmp_path)
    checkpoint_path = source / 'checkpoint.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    parent = source / 'parent.json'
    parent.write_text('{}')
    checkpoint['parents'] = {key: {'path': str(parent), 'sha256': sha256(parent)} for key in ('authority_matrix', 'lineage_split')}
    checkpoint_path.write_text(json.dumps(checkpoint))
    # The exporter boundary receives an already independently approved checkpoint.
    import tools.corpus.build_corpus370_materialization_checkpoint_v1 as approval
    monkeypatch.setattr(approval, 'build', lambda *_: checkpoint)
    index = source / 'index.json'
    sample = {'model': 'IRIS', 'subject_id': manifest['subject_id'],
              'manifest': {'path': str(source / 'original.json'), 'sha256': sha256(source / 'original.json')}}
    index.write_text(json.dumps({'schema': 'RealSaS.Corpus370ProjectedModelSubjects.v1', 'training_authorized': False, 'subjects': [sample]}))
    output = tmp_path / 'exported'
    seal = build(checkpoint_path, index, output)
    assert len(seal['model_subjects']) == 1 and seal['training_authorized'] is False
    for row in seal['files']:
        assert sha256(output / row['relative_path']) == row['sha256']
    original = seal['model_subjects'][0]['manifest_relative_path']
    assert (output / original).read_bytes() == (source / 'original.json').read_bytes()
    index.write_text(json.dumps({'schema': 'RealSaS.Corpus370ProjectedModelSubjects.v1', 'training_authorized': False, 'subjects': []}))
    with pytest.raises(ValueError, match='MISSING_ELIGIBLE_SUBJECTS'):
        build(checkpoint_path, index, output)


def test_unresolved_corpus_is_not_exportable(tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'status': 'BLOCKED_PRE_MATERIALIZATION'}))
    with pytest.raises(ValueError, match='CHECKPOINT_NOT_READY'):
        build(checkpoint, tmp_path / 'unused.json', tmp_path / 'output')
