#!/usr/bin/env python3
"""Resume a sealed corpus copy and verify model consumers before publishing PASS.

The bundle carries original manifest bytes plus an exact source-path relocation
map. Derived local manifests change only hash-bound file locations; the original
manifests, split decisions, family states and truth bytes remain preserved.
No eligibility, split, truth target or optimizer is computed here.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path, PurePosixPath
import shutil
import time

from models.materialization_contract_v1 import require, sha256
from models.iris.v32.training_input_contract_v1 import validate_manifest as validate_iris
from models.tessa.v32.training_input_contract_v1 import validate_manifest as validate_tessa


def inside(root, relative):
    path = PurePosixPath(relative)
    require(bool(relative) and not path.is_absolute() and '..' not in path.parts and '\\' not in relative,
            'MATERIALIZE_RELATIVE_PATH_REQUIRED')
    result = (root / relative).resolve()
    require(result.is_relative_to(root), 'MATERIALIZE_PATH_ESCAPE')
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix('.partial')
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    staged.replace(path)


def relocate(value, mapping, output):
    if isinstance(value, list): return [relocate(v, mapping, output) for v in value]
    if not isinstance(value, dict): return value
    result = {k: relocate(v, mapping, output) for k, v in value.items()}
    if 'path' in value and 'sha256' in value:
        key = (value['path'], value['sha256'])
        require(key in mapping, 'MATERIALIZE_UNSEALED_MANIFEST_FILE_REF:' + str(value['path']))
        result['path'] = str(inside(output, mapping[key]))
    return result


def materialize(seal_path, expected_sha256, bundle_root, output):
    seal_path, bundle_root, output = seal_path.resolve(), bundle_root.resolve(), output.resolve()
    require(sha256(seal_path) == expected_sha256, 'MATERIALIZE_SEAL_DRIFT')
    seal = json.loads(seal_path.read_text())
    require(seal.get('schema') == 'RealSaS.Corpus370MaterializationBundle.v1'
            and seal.get('status') == 'PASS_SEALED_STATIC_CORPUS_BUNDLE'
            and seal.get('training_authorized') is False, 'MATERIALIZE_SEAL_NOT_READY')
    files = seal['files']
    names = [r['relative_path'] for r in files]
    require(len(names) == len(set(names)), 'MATERIALIZE_DUPLICATE_FILE_PATH')
    output.mkdir(parents=True, exist_ok=True)
    identity = output / 'MATERIALIZATION_SCOPE.json'
    scope = {'schema': 'RealSaS.Corpus370MaterializationScope.v1', 'bundle_sha256': expected_sha256, 'training_authorized': False}
    if identity.exists():
        require(json.loads(identity.read_text()) == scope, 'MATERIALIZE_RESUME_SCOPE_DRIFT')
    else: write(identity, scope)
    write(output / 'CORPUS370_MATERIALIZATION_RECEIPT_V1.json',
          {'schema': 'RealSaS.Corpus370MaterializationReceipt.v1', 'status': 'IN_PROGRESS__REVERIFICATION_REQUIRED',
           'seal_sha256': expected_sha256, 'training_authorized': False})
    mapping = {}
    start = time.monotonic()
    for number, row in enumerate(files):
        source, destination = inside(bundle_root, row['relative_path']), inside(output, row['relative_path'])
        require(source.is_file() and source.stat().st_size == row['size_bytes'] and sha256(source) == row['sha256'],
                'MATERIALIZE_INPUT_BYTES_DRIFT:' + row['relative_path'])
        if destination.exists():
            require(destination.is_file() and destination.stat().st_size == row['size_bytes'] and sha256(destination) == row['sha256'],
                    'MATERIALIZE_EXISTING_BYTES_DRIFT:' + row['relative_path'])
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.with_suffix(destination.suffix + '.partial')
            shutil.copyfile(source, staged)
            require(sha256(staged) == row['sha256'], 'MATERIALIZE_COPIED_BYTES_DRIFT')
            staged.replace(destination)
        for original in row.get('original_paths', []):
            key = (original, row['sha256'])
            require(key not in mapping or mapping[key] == row['relative_path'], 'MATERIALIZE_RELOCATION_COLLISION')
            mapping[key] = row['relative_path']
        if number % 25 == 0 or number + 1 == len(files):
            print(json.dumps({'event': 'VERIFIED_FILE_PROGRESS', 'done': number + 1, 'total': len(files),
                              'elapsed_seconds': round(time.monotonic() - start, 2)}), flush=True)
    checkpoint_ref = seal['checkpoint']
    checkpoint_path = inside(output, checkpoint_ref['relative_path'])
    require(sha256(checkpoint_path) == checkpoint_ref['sha256'], 'MATERIALIZE_CHECKPOINT_BYTES_DRIFT')
    checkpoint = json.loads(checkpoint_path.read_text())
    require(checkpoint.get('schema') == 'RealSaS.Corpus370MaterializationCheckpoint.v1'
            and checkpoint.get('status') == 'PASS_PRE_MATERIALIZATION_CHECKPOINT'
            and checkpoint.get('counts_are_final') is True and checkpoint.get('materialization_authorized') is True
            and checkpoint.get('training_authorized') is False, 'MATERIALIZE_CHECKPOINT_NOT_READY')
    samples = []
    seen = set()
    validators = {'IRIS': validate_iris, 'TESSA': validate_tessa}
    # Demand exact coverage of the sealed admitted rosters, including unseen.
    expected = {(model, s['subject_id']): s for model, payload in checkpoint['models'].items()
                for s in payload['subjects'] if s['disposition'] in {'ADMIT_FULL', 'ADMIT_PARTIAL'}}
    for row in seal['model_subjects']:
        model, sid = row['model'], row['subject_id']
        key = (model, sid)
        require(key in expected and key not in seen, 'MATERIALIZE_MODEL_SUBJECT_SET_DRIFT')
        seen.add(key)
        original_path = inside(output, row['manifest_relative_path'])
        require(sha256(original_path) == row['manifest_sha256'], 'MATERIALIZE_ORIGINAL_MANIFEST_DRIFT')
        manifest = json.loads(original_path.read_text())
        decision = expected[key]
        require(manifest['subject_id'] == sid and manifest['source_sha256'] == decision['source_sha256']
                and manifest['split'] == decision['split'] and manifest['family_status'] == decision['family_status'],
                'MATERIALIZE_AUTHORITY_DECISION_DRIFT')
        derived = relocate(copy.deepcopy(manifest), mapping, output)
        path = inside(output, 'local_manifests/' + model + '/' + sid + '.json')
        write(path, derived)
        validated = validators[model](path)
        samples.append({'model': model, 'subject_id': sid, 'original_manifest_sha256': row['manifest_sha256'],
                        'local_manifest': str(path), 'local_manifest_sha256': sha256(path), 'consumer_validation': validated})
        print(json.dumps({'event': 'VERIFIED_MODEL_SUBJECT', 'model': model, 'subject_id': sid,
                          'done': len(samples), 'total': len(expected)}), flush=True)
    require(seen == set(expected), 'MATERIALIZE_MISSING_ELIGIBLE_SUBJECTS')
    receipt = {'schema': 'RealSaS.Corpus370MaterializationReceipt.v1', 'status': 'PASS_STATIC_CORPUS_MATERIALIZED_AND_CONSUMER_VERIFIED',
        'seal_sha256': expected_sha256, 'verified_file_count': len(files), 'verified_payload_bytes': sum(r['size_bytes'] for r in files),
        'elapsed_seconds': round(time.monotonic() - start, 2), 'subjects': samples,
        'original_truth_bytes_mutated': False, 'eligibility_computed_by_notebook': False,
        'split_computed_by_notebook': False, 'component_sign_promoted_to_global_sdf': False,
        'tessa_training_ready': False, 'training_authorized': False}
    write(output / 'CORPUS370_MATERIALIZATION_RECEIPT_V1.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seal', type=Path, required=True)
    p.add_argument('--seal-sha256', required=True)
    p.add_argument('--bundle-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    receipt = materialize(a.seal, a.seal_sha256, a.bundle_root, a.output)
    print(json.dumps({k: receipt[k] for k in ('status', 'verified_file_count', 'elapsed_seconds', 'training_authorized')}))


if __name__ == '__main__': main()
