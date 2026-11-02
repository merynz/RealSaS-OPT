import json
from pathlib import Path

import pytest

from models.materialization_contract_v1 import sha256
from tools.corpus.materialize_corpus370_sealed_bundle_v1 import inside, materialize
from test_partial_materialization_contract_v1 import iris_subject


def bundle(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    manifest = iris_subject(source, 'FAIL')
    originals = []
    def collect(value):
        if isinstance(value, list):
            for v in value: collect(v)
        elif isinstance(value, dict):
            if 'path' in value and 'sha256' in value: originals.append(value)
            for v in value.values(): collect(v)
    collect(manifest)
    path = source / 'original.json'
    path.write_text(json.dumps(manifest))
    files = [{'relative_path': p.name, 'sha256': sha256(p), 'size_bytes': p.stat().st_size,
              'original_paths': [str(p)]} for p in source.iterdir()]
    checkpoint = {'schema': 'RealSaS.Corpus370MaterializationCheckpoint.v1', 'status': 'PASS_PRE_MATERIALIZATION_CHECKPOINT',
        'counts_are_final': True, 'materialization_authorized': True, 'training_authorized': False,
        'models': {'IRIS': {'subjects': [{'subject_id': manifest['subject_id'], 'source_sha256': manifest['source_sha256'],
           'split': 'TRAIN', 'family_status': manifest['family_status'], 'disposition': 'ADMIT_PARTIAL'}]}}}
    checkpath = source / 'checkpoint.json'
    checkpath.write_text(json.dumps(checkpoint))
    files.append({'relative_path': checkpath.name, 'sha256': sha256(checkpath), 'size_bytes': checkpath.stat().st_size})
    seal = {'schema': 'RealSaS.Corpus370MaterializationBundle.v1', 'status': 'PASS_SEALED_STATIC_CORPUS_BUNDLE',
        'training_authorized': False, 'files': files, 'checkpoint': {'relative_path': checkpath.name, 'sha256': sha256(checkpath)},
        'model_subjects': [{'model': 'IRIS', 'subject_id': manifest['subject_id'], 'manifest_relative_path': path.name, 'manifest_sha256': sha256(path)}]}
    sealpath = source / 'seal.json'
    sealpath.write_text(json.dumps(seal))
    return sealpath, source, tmp_path / 'output', manifest


def test_copy_resume_relocation_and_numeric_consumer_validation(tmp_path):
    seal, source, output, manifest = bundle(tmp_path)
    before = (source / 'original.json').read_bytes()
    result = materialize(seal, sha256(seal), source, output)
    assert result['status'] == 'PASS_STATIC_CORPUS_MATERIALIZED_AND_CONSUMER_VERIFIED'
    assert result['subjects'][0]['consumer_validation']['surface_query_count'] == 2
    assert result['subjects'][0]['consumer_validation']['global_sdf_valid_count'] == 0
    assert (output / 'original.json').read_bytes() == before
    local = json.loads(Path(result['subjects'][0]['local_manifest']).read_text())
    assert local['family_status'] == manifest['family_status']
    assert Path(local['surface_geometry']['path']).is_relative_to(output)
    assert materialize(seal, sha256(seal), source, output)['verified_file_count'] == result['verified_file_count']
    (output / 'surface.npz').write_bytes(b'corrupt-resumed-copy')
    with pytest.raises(ValueError, match='EXISTING_BYTES_DRIFT'):
        materialize(seal, sha256(seal), source, output)
    assert json.loads((output / 'CORPUS370_MATERIALIZATION_RECEIPT_V1.json').read_text())['status'].startswith('IN_PROGRESS')


def test_exact_coverage_rejects_an_omitted_eligible_subject(tmp_path):
    seal, source, output, _ = bundle(tmp_path)
    value = json.loads(seal.read_text())
    value['model_subjects'] = []
    seal.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='MISSING_ELIGIBLE_SUBJECTS'):
        materialize(seal, sha256(seal), source, output)


@pytest.mark.parametrize('relative', ['../outside', '/absolute', 'local/../../outside', 'windows\\escape'])
def test_bundle_paths_cannot_escape_destination(tmp_path, relative):
    with pytest.raises(ValueError, match='PATH|RELATIVE'):
        inside(tmp_path, relative)
