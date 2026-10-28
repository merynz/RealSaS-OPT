"""CI reproducibility of sealed import bytes; no new teacher or Platform writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

from tools.observation.pinned_evaluator_runtime_v1 import launch_options, sha


def require(value, code):
    if not value:
        raise ValueError(code)


def bound_json(ref):
    require(sha(ref['path']) == ref['sha256'], 'IMPORT_REPLAY_BOUND_BYTES_DRIFT')
    return json.loads(Path(ref['path']).read_text())


def replay(request, readback, *, repo, authority, artifact_root, output, api_root='http://127.0.0.1:8080'):
    require(readback['execution_allowed'] is False and readback['training_allowed'] is False,
            'IMPORT_REPLAY_READ_ONLY_SCOPE_REQUIRED')
    authority, artifact_root, output = map(lambda path: Path(path).resolve(), (authority, artifact_root, output))
    preparation = (authority / readback['preparation_relative_path']).resolve()
    require(preparation.is_relative_to(authority), 'IMPORT_REPLAY_PREPARATION_PATH_ESCAPE')
    value = bound_json({'path': str(preparation), 'sha256': readback['preparation_sha256']})
    require(value['campaign_id'] == readback['campaign_id'], 'IMPORT_REPLAY_CAMPAIGN_DRIFT')
    bootstrap = bound_json(value['bootstrap_receipt'])
    members = {row['subject_id']: row for row in value['members']}
    require({row['subject_id'] for row in bootstrap['run_bindings']} == set(members), 'IMPORT_REPLAY_ROSTER_DRIFT')
    executable = Path(request['evaluator_executable'])
    options = launch_options(executable, request['evaluator_runtime'])
    require(sha(executable) == request['expected_evaluator_binary_sha256'], 'IMPORT_REPLAY_BINARY_DRIFT')
    output.mkdir(parents=True, exist_ok=False)

    def get(path):
        with urllib.request.urlopen(api_root + path, timeout=30) as source:
            return json.load(source)

    results = []
    for binding in bootstrap['run_bindings']:
        member = members[binding['subject_id']]
        if member['role'] != 'COHORT':
            continue
        manifest = json.loads(Path(binding['run_manifest_path']).read_text())
        records = get('/v1/subjects/' + member['subject_id'] + '/artifacts?artifact_type=RealSaS.RunManifestSection&schema_version=v1&limit=500')
        for role in ('source_files', 'source_audit'):
            selected = [row for row in records if row['source_kind'] == 'SUBJECT_INPUT'
                and row['source_id'] == binding['subject_input_id'] and row['role'] == 'manifest:' + role]
            require(len(selected) == 1, 'IMPORT_REPLAY_EXACT_INPUT_SECTION_REQUIRED')
            record = get('/v1/artifacts/' + selected[0]['artifact_id'])
            path = (artifact_root / record['storage_key']).resolve()
            require(path.is_relative_to(artifact_root) and sha(path) == record['content_sha256'] == selected[0]['content_sha256'],
                    'IMPORT_REPLAY_REGISTRY_SECTION_BYTES_DRIFT')
            require(json.loads(path.read_text()) == manifest[role], 'IMPORT_REPLAY_LOCAL_MANIFEST_DRIFT')
        audit = manifest['source_audit']
        admission = bound_json(audit['iris_observation_admission'])
        candidate = bound_json(audit['source_rest_candidate'])
        sources = [row for row in manifest['source_files'] if row['source_kind'] == 'PRIMARY']
        require(len(sources) == 1, 'IMPORT_REPLAY_PRIMARY_SOURCE_REQUIRED')
        source = sources[0]
        require(sha(source['path']) == source['expected_sha256'] == admission['source_sha256'] == candidate['source_sha256']
                and admission['candidate_sha256'] == audit['source_rest_candidate']['sha256'], 'IMPORT_REPLAY_SOURCE_LINEAGE_DRIFT')
        folder = output / member['attempt_id']; folder.mkdir()
        imported = folder / 'imported.json'
        with (folder / 'replay.log').open('w') as log:
            subprocess.run([str(executable), '-b', '--factory-startup', '--python-exit-code', '2', '--python',
                str(Path(repo).resolve() / 'tools/observation/blender_source_parity_export_v1.py'), '--', source['path'], str(imported)],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120, **options)
        actual = sha(imported)
        results.append({'subject_key': manifest['subject_id'], 'attempt_id': member['attempt_id'],
            'subject_input_id': binding['subject_input_id'], 'admitted_import_sha256': admission['imported_geometry_sha256'],
            'replayed_import_sha256': actual, 'exact_bytes_equal': actual == admission['imported_geometry_sha256']})
    contract = json.loads((Path(repo) / request['campaign_contract_repo_path']).read_text())
    require({row['subject_key'] for row in results} == set(contract['cohort_subject_keys']), 'IMPORT_REPLAY_COHORT_KEYS_DRIFT')
    report = {'schema': 'RealSaS.SealedObservationImportReplay.v1', 'campaign_id': value['campaign_id'],
        'subjects': results, 'all_admitted_import_bytes_equal': all(row['exact_bytes_equal'] for row in results),
        'new_teacher_measured': False, 'platform_mutation_performed': False, 'training_executed': False}
    (output / 'REPLAY_REPORT.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    require(report['all_admitted_import_bytes_equal'], 'IMPORT_REPLAY_NOT_EXACT')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'readback', 'repo', 'authority', 'artifact-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = replay(json.loads(Path(args.request).read_text()), json.loads(Path(args.readback).read_text()),
        repo=args.repo, authority=args.authority, artifact_root=args.artifact_root, output=args.output)
    print(json.dumps({'subjects_replayed': len(result['subjects']), 'exact_bytes_equal': True,
                      'new_teacher_measured': False, 'platform_mutation_performed': False}))
