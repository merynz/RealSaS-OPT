#!/usr/bin/env python3
"""Export the full exact-byte closure of an approved static corpus checkpoint.

The input index contains already projected IRIS/TESSA subject manifests. Their
families and splits must exactly match the pre-materialization checkpoint. Every
bound payload/court/parent is exported unchanged; relocation is recorded by hash.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import zipfile

from models.materialization_contract_v1 import require, sha256
from models.iris.v32.training_input_contract_v1 import validate_manifest as validate_iris
from models.tessa.v32.training_input_contract_v1 import validate_manifest as validate_tessa


def build(checkpoint_path, subject_index_path, output):
    from tools.corpus.build_corpus370_materialization_checkpoint_v1 import build as replay_checkpoint
    checkpoint = json.loads(checkpoint_path.read_text())
    require(checkpoint.get('status') == 'PASS_PRE_MATERIALIZATION_CHECKPOINT'
            and checkpoint.get('materialization_authorized') is True and checkpoint.get('training_authorized') is False,
            'BUNDLE_CHECKPOINT_NOT_READY')
    parents = checkpoint['parents']
    for parent in parents.values():
        require(parent and Path(parent['path']).is_file() and sha256(Path(parent['path'])) == parent['sha256'],
                'BUNDLE_CHECKPOINT_PARENT_DRIFT')
    replay = replay_checkpoint(Path(parents['authority_matrix']['path']), Path(parents['lineage_split']['path']))
    require(replay == checkpoint, 'BUNDLE_CHECKPOINT_REPLAY_DRIFT')
    index = json.loads(subject_index_path.read_text())
    require(index.get('schema') == 'RealSaS.Corpus370ProjectedModelSubjects.v1' and index.get('training_authorized') is False,
            'BUNDLE_SUBJECT_INDEX_CONTRACT')
    expected = {(model, s['subject_id']): s for model, row in checkpoint['models'].items() for s in row['subjects']
                if s['disposition'] in {'ADMIT_FULL', 'ADMIT_PARTIAL'}}
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    visited = set()
    def add(ref):
        path = Path(ref['path']).resolve()
        expected_sha = ref['sha256']
        require(path.is_file() and sha256(path) == expected_sha, 'BUNDLE_BOUND_FILE_DRIFT:' + str(path))
        relative = 'files/' + expected_sha + '/' + path.name
        row = files.setdefault(relative, {'relative_path': relative, 'sha256': expected_sha,
                                         'size_bytes': path.stat().st_size, 'original_paths': []})
        if str(path) not in row['original_paths']: row['original_paths'].append(str(path))
        destination = output / relative
        if destination.exists(): require(sha256(destination) == expected_sha, 'BUNDLE_EXISTING_BYTES_DRIFT')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.with_suffix(destination.suffix + '.partial')
            shutil.copyfile(path, staged)
            require(sha256(staged) == expected_sha, 'BUNDLE_COPY_BYTES_DRIFT')
            staged.replace(destination)
        key = (str(path), expected_sha)
        if key not in visited:
            visited.add(key)
            if path.suffix.lower() == '.json': walk(json.loads(path.read_text()))
        return relative
    def walk(value):
        if isinstance(value, list):
            for item in value: walk(item)
        elif isinstance(value, dict):
            if value.get('path') and re.fullmatch('[0-9a-f]{64}', str(value.get('sha256') or '')):
                add(value)
            for item in value.values(): walk(item)
    checkpoint_relative = add({'path': str(checkpoint_path.resolve()), 'sha256': sha256(checkpoint_path)})
    seen = set()
    subjects = []
    validators = {'IRIS': validate_iris, 'TESSA': validate_tessa}
    for item in index['subjects']:
        key = (item['model'], item['subject_id'])
        require(key in expected and key not in seen, 'BUNDLE_SUBJECT_COVERAGE_DRIFT')
        seen.add(key)
        ref = item['manifest']
        path = Path(ref['path']).resolve()
        require(sha256(path) == ref['sha256'], 'BUNDLE_MODEL_MANIFEST_DRIFT')
        manifest = json.loads(path.read_text())
        decision = expected[key]
        require(all(manifest[k] == decision[k] for k in ('subject_id', 'source_sha256', 'split', 'family_status')),
                'BUNDLE_MODEL_AUTHORITY_DRIFT')
        result = validators[item['model']](path)
        relative = add(ref)
        subjects.append({'model': item['model'], 'subject_id': item['subject_id'],
                         'manifest_relative_path': relative, 'manifest_sha256': ref['sha256'], 'consumer_validation': result})
    require(seen == set(expected), 'BUNDLE_MISSING_ELIGIBLE_SUBJECTS')
    seal = {'schema': 'RealSaS.Corpus370MaterializationBundle.v1', 'status': 'PASS_SEALED_STATIC_CORPUS_BUNDLE',
            'checkpoint': {'relative_path': checkpoint_relative, 'sha256': sha256(checkpoint_path)},
            'files': sorted(files.values(), key=lambda r: r['relative_path']), 'model_subjects': subjects,
            'eligibility_computed_by_notebook': False, 'training_authorized': False}
    path = output / 'CORPUS370_MATERIALIZATION_BUNDLE_V1.json'
    path.write_text(json.dumps(seal, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return seal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--subject-index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--zip', type=Path, required=True)
    a = p.parse_args()
    seal = build(a.checkpoint, a.subject_index, a.output)
    with zipfile.ZipFile(a.zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for row in seal['files']: archive.write(a.output / row['relative_path'], row['relative_path'])
        archive.write(a.output / 'CORPUS370_MATERIALIZATION_BUNDLE_V1.json', 'CORPUS370_MATERIALIZATION_BUNDLE_V1.json')
    print(json.dumps({'schema': seal['schema'], 'file_count': len(seal['files']),
                      'model_subject_count': len(seal['model_subjects']), 'zip_sha256': sha256(a.zip), 'training_authorized': False}))


if __name__ == '__main__': main()
