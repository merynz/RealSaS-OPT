#!/usr/bin/env python3
"""Continue the exact corpus cohort into materialization-ready family evidence.

Keep unchanged SubjectInputs and qualified prefix artifact identities. Repair only
omitted external source dependencies in a new hash-bound closure; preserve every
old source byte, input packet and failed Attempt. This coordinator submits no
optimizer and computes no model eligibility or truth PASS itself.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from tools.platform_artifact_store import sha256_file
from tools.platform_cohort_bootstrap_v1 import api, put_import, json_bytes
from tools import platform_truth_cohort_bootstrap_v1 as bootstrap
from tools.platform_release_snapshot import snapshot
from tools.corpus.prepare_corpus370_platform_truth_bundle_v2 import gltf_dependencies


def require(value, code):
    if not value:
        raise RuntimeError(code)


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix('.partial')
    staged.write_bytes(json_bytes(value))
    staged.replace(path)


def bound(ref):
    path = Path(ref['path']).resolve()
    require(path.is_file() and sha256_file(path) == ref['sha256'], 'CORPUS_CONTINUATION_PARENT_DRIFT:' + str(path))
    return path


def copy_exact(source, destination, expected):
    require(source.is_file() and sha256_file(source) == expected, 'CORPUS_CONTINUATION_SOURCE_DRIFT')
    if destination.exists():
        require(destination.is_file() and sha256_file(destination) == expected, 'CORPUS_CONTINUATION_COPY_DRIFT')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_suffix(destination.suffix + '.partial')
        shutil.copyfile(source, staged)
        require(sha256_file(staged) == expected, 'CORPUS_CONTINUATION_COPY_DRIFT')
        staged.replace(destination)


def repair_dependencies(sections, source_row, source_root, source_index, output_root):
    primary = [row for row in sections['source_files'] if row.get('source_kind') == 'PRIMARY']
    require(len(primary) == 1, 'CORPUS_CONTINUATION_PRIMARY_CARDINALITY')
    primary = primary[0]
    original = (source_root / source_row['source_relative_path']).resolve()
    require(original.is_relative_to(source_root) and sha256_file(original) == primary['expected_sha256'] == source_row['source_sha256'],
            'CORPUS_CONTINUATION_EXACT_SOURCE_IDENTITY')
    dependencies = []
    missing = False
    for source in gltf_dependencies(original):
        rel = source.relative_to(original.parent).as_posix()
        row = source_index[source.relative_to(source_root).as_posix()]
        require(sha256_file(source) == row['sha256'] and source.stat().st_size == row['size_bytes'],
                'CORPUS_CONTINUATION_DEPENDENCY_SOURCE_DRIFT')
        expected_path = Path(primary['path']).parent / rel
        declared = any(Path(r['path']).resolve() == expected_path.resolve() and r['expected_sha256'] == row['sha256']
                       for r in sections['source_files'] if r.get('source_kind') == 'DEPENDENCY')
        missing |= not (declared and expected_path.is_file() and sha256_file(expected_path) == row['sha256'])
        dependencies.append({'relative_path': rel, 'sha256': row['sha256'], 'size_bytes': source.stat().st_size, 'original_path': str(source)})
    if not missing:
        return sections, {'status': 'UNCHANGED_SUBJECT_INPUT', 'source_sha256': source_row['source_sha256']}
    identity = {'source_sha256': source_row['source_sha256'], 'source_filename': original.name,
                'dependencies': [{k: r[k] for k in ('relative_path', 'sha256', 'size_bytes')} for r in dependencies]}
    root = output_root / digest(identity)
    target = root / original.name
    copy_exact(original, target, source_row['source_sha256'])
    updated = copy.deepcopy(sections)
    updated['source_files'] = [r for r in updated['source_files'] if r.get('source_kind') != 'DEPENDENCY']
    next(r for r in updated['source_files'] if r.get('source_kind') == 'PRIMARY')['path'] = str(target)
    for number, row in enumerate(dependencies):
        destination = root / row['relative_path']
        copy_exact(Path(row['original_path']), destination, row['sha256'])
        updated['source_files'].append({'role': f'source_dependency_{number:04d}', 'source_kind': 'DEPENDENCY',
            'path': str(destination), 'expected_sha256': row['sha256'], 'expected_size_bytes': row['size_bytes']})
    receipt = {'schema': 'RealSaS.SourceDependencyClosureRepair.v1', 'status': 'EXACT_SOURCE_DEPENDENCIES_RESTORED_IN_NEW_CLOSURE',
               **identity, 'original_source': str(original), 'prepared_source': str(target),
               'raw_source_mutated': False, 'truth_qualification_minted': False, 'training_authorized': False}
    write(root / 'SOURCE_DEPENDENCY_CLOSURE_REPAIR.json', receipt)
    return updated, receipt


def continue_corpus(args):
    repo, authority, platform, output = args.repo.resolve(), args.authority_root.resolve(), args.platform_root.resolve(), args.output.resolve()
    descriptor = load(args.request)
    require(descriptor.get('schema') == 'RealSaS.Corpus370IsolatedContinuation.v1' and descriptor.get('training_authorized') is False,
            'CORPUS_CONTINUATION_SCOPE')
    require(descriptor.get('enabled') is True, 'CORPUS_CONTINUATION_DISABLED')
    require(load(platform / 'deployment.json')['code_sha'] == args.expected_main_sha, 'CORPUS_CONTINUATION_DEPLOYED_MAIN_DRIFT')
    require(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == args.expected_main_sha,
            'CORPUS_CONTINUATION_CHECKOUT_DRIFT')
    require(subprocess.check_output(['git', '-C', str(repo), 'ls-remote', 'origin', 'refs/heads/main'], text=True).split()[0] == args.expected_main_sha,
            'CORPUS_CONTINUATION_REMOTE_MAIN_DRIFT')
    parent_path = bound(descriptor['original_preparation'])
    parent = load(parent_path)
    bootstrap_path = bound(parent['bootstrap_receipt'])
    old = load(bootstrap_path)
    descriptor_parent = parent_path
    for _ in range(8):
        descriptor_path = descriptor_parent.parent / 'BOOTSTRAP_DESCRIPTOR.json'
        if descriptor_path.is_file(): break
        descriptor_parent = bound(load(descriptor_parent)['parent_preparation'])
    require(descriptor_path.is_file(), 'CORPUS_CONTINUATION_BOOTSTRAP_DESCRIPTOR_MISSING')
    old_descriptor = load(descriptor_path)
    parent_status = api(args.api, 'GET', '/v1/research/campaigns/' + parent['campaign_id'])
    require(len(parent_status['members']) == parent['member_count'] == 294
            and all(m['final_state'] in {'COMPLETED', 'FAILED'} for m in parent_status['members']),
            'CORPUS_CONTINUATION_PARENT_NOT_TERMINAL')
    preparation_root = authority / 'corpus370_preparation_v1'
    archive = preparation_root / 'CORPUS370_PLATFORM_FAMILY_INPUT_BUNDLE_V2.zip'
    require(sha256_file(archive) == parent['bundle_sha256'], 'CORPUS_CONTINUATION_BASE_BUNDLE_DRIFT')
    with zipfile.ZipFile(archive) as zipped:
        base_manifest = json.loads(zipped.read('BOOTSTRAP_MANIFEST.json'))
    source_manifest = load(bound(base_manifest['parents']['source_bundle_manifest']))
    source_rows = {r['subject_id']: r for r in source_manifest['subjects']}
    source_index = {r['archive_path']: r for r in source_manifest['files']}
    hydration = load(preparation_root / 'source_bundle/CORPUS370_SOURCE_BUNDLE_HYDRATION_V1.json')
    require(hydration['source_bundle_manifest']['sha256'] == base_manifest['parents']['source_bundle_manifest']['sha256'],
            'CORPUS_CONTINUATION_SOURCE_TRANSPORT_DRIFT')
    source_root = Path(hydration['extracted_source_root']).resolve()
    plan = load(repo / descriptor['compiler_plan'])
    keys = bootstrap.plan_manifest_keys(plan)
    state_path = output / 'CONTINUATION_STATE.json'
    identity = digest({'descriptor': descriptor, 'code_sha': args.expected_main_sha})
    state = load(state_path) if state_path.exists() else {'identity': identity, 'subjects': {}}
    require(state['identity'] == identity, 'CORPUS_CONTINUATION_RESUME_SCOPE_DRIFT')
    def checkpoint(): write(state_path, state)
    write(output / 'BOOTSTRAP_DESCRIPTOR.json', old_descriptor)
    rows = []
    actor = 'corpus370-isolated-materialization-inputs'
    for binding in old['run_bindings']:
        manifest = load(binding['run_manifest_path'])
        sid = manifest['subject_id']
        sections = {key: manifest[key] for key in keys if key in manifest}
        repaired, receipt = repair_dependencies(sections, source_rows[sid], source_root, source_index, authority / 'source_dependency_closures')
        changed = repaired != sections
        item_identity = digest(repaired)
        saved = state['subjects'].get(sid)
        if saved:
            require(saved['sections_sha256'] == item_identity, 'CORPUS_CONTINUATION_INPUT_DRIFT:' + sid)
            input_id = saved['subject_input_id']
        elif changed:
            bindings = []
            original_origin = f"urn:realsas:corpus370-observation-truth-v2:{old_descriptor['campaign_name']}:{sid}"
            for key in sorted(repaired):
                origin = f'urn:realsas:corpus370-dependency-repair:{sid}' if key == 'source_files' else original_origin
                imported = put_import(args.api, json_bytes(repaired[key]), artifact_type='RealSaS.RunManifestSection', schema='v1',
                    source_uri=origin + ':manifest:' + key, actor=actor)
                bindings.append({'role': 'manifest:' + key, 'artifact_id': imported['artifact_id']})
            input_id = api(args.api, 'POST', '/v1/subject-inputs', {'subject_id': binding['subject_id'], 'bindings': bindings, 'created_by': actor})['subject_input_id']
        else:
            input_id = binding['subject_input_id']
        state['subjects'][sid] = {'subject_input_id': input_id, 'sections_sha256': item_identity,
                                  'changed_input_roles': ['manifest:source_files'] if changed else [], 'repair': receipt}
        checkpoint()
        rows.append({'key': sid, 'platform_subject_id': binding['subject_id'], 'subject_input_id': input_id, 'sections': repaired,
                     'changed_input_roles': state['subjects'][sid]['changed_input_roles']})
    repair_index = output / 'SOURCE_DEPENDENCY_REPAIRS.json'
    write(repair_index, {'schema': 'RealSaS.Corpus370SourceDependencyRepairIndex.v1', 'subjects': state['subjects'], 'training_authorized': False})
    if 'release_id' not in state:
        release_request = snapshot(plan, name=descriptor['campaign_name'] + '__' + args.expected_main_sha,
            purpose='RESEARCH', created_by=actor, stage_parameters=bootstrap.release_stage_parameters(old_descriptor, plan))
        require(release_request['manifest']['pipeline_plan_sha256'] == old['pipeline_plan_sha256'], 'CORPUS_CONTINUATION_GRAPH_SCOPE_DRIFT')
        write(output / 'ENGINE_RELEASE_REQUEST.json', release_request)
        state['release_id'] = api(args.api, 'POST', '/v1/releases', release_request)['ReleaseID']
        checkpoint()
    baseline_versions = api(args.api, 'GET', '/v1/releases/' + parent['engine_release_id'] + '/capabilities')['versions']
    candidate_versions = api(args.api, 'GET', '/v1/releases/' + state['release_id'] + '/capabilities')['versions']
    stage_ids = [r['id'] for r in plan['stages']]
    require(all('compiler.stage/' + s in baseline_versions and 'compiler.stage/' + s in candidate_versions for s in stage_ids),
            'CORPUS_CONTINUATION_RELEASE_STAGE_SCOPE')
    direct = [s for s in stage_ids if baseline_versions['compiler.stage/' + s] != candidate_versions['compiler.stage/' + s]]
    require(direct == descriptor['direct_changed_stage_ids'], 'CORPUS_CONTINUATION_DIRECT_CHANGE_DRIFT:' + str(direct))
    if 'campaign_id' not in state:
        interventions = {r['platform_subject_id']: {'direct_changed_stage_ids': direct, 'changed_input_roles': r['changed_input_roles']} for r in rows}
        campaign = api(args.api, 'POST', '/v1/research/campaigns', {'name': descriptor['campaign_name'],
            'baseline_engine_release_id': parent['engine_release_id'], 'candidate_engine_release_id': state['release_id'],
            'parent_campaign_id': parent['campaign_id'], 'member_interventions': interventions,
            'target_stage_id': parent['target_stage_id'], 'cohort_subject_ids': [m['subject_id'] for m in parent_status['members']],
            'regression_subject_ids': [], 'created_by': actor})
        state['campaign_id'] = campaign['campaign_id']; checkpoint()
    campaign_id = state['campaign_id']
    campaign = api(args.api, 'GET', '/v1/research/campaigns/' + campaign_id)
    experiment_request = {'slug': descriptor['experiment_slug'], 'name': descriptor['campaign_name'],
        'research_question': 'Which exact source subjects support qualified partial supervision after transport and native execution repair?',
        'hypothesis': 'Exact dependency restoration and isolated native execution preserve upstream evidence and yield independently courted model inputs.',
        'initial_state': 'PREREGISTERED', 'created_by': actor, 'metadata': {'source_code_sha': args.expected_main_sha,
            'parent_campaign_id': parent['campaign_id'], 'training_allowed': False, 'candidate_subject_count': 370}}
    if 'experiment_id' not in state:
        matches = [r for r in api(args.api, 'GET', '/v1/experiments') if r['slug'] == descriptor['experiment_slug']]
        require(len(matches) <= 1, 'CORPUS_CONTINUATION_EXPERIMENT_AMBIGUOUS')
        if matches:
            require(matches[0]['metadata'] == experiment_request['metadata'], 'CORPUS_CONTINUATION_EXPERIMENT_DRIFT')
            state['experiment_id'] = matches[0]['id']
        else:
            state['experiment_id'] = api(args.api, 'POST', '/v1/experiments', experiment_request)['experiment']['id']
        checkpoint()
    if not state.get('linked'):
        for kind, target in [('CAMPAIGN', campaign_id), *[('ATTEMPT', m['attempt_id']) for m in campaign['members']]]:
            api(args.api, 'POST', '/v1/experiments/link', {'experiment_id': state['experiment_id'], 'target_kind': kind, 'target_id': target, 'actor': actor})
        state['linked'] = True; checkpoint()
    run_bindings = []
    for row in rows:
        run_id = f"CAMPAIGN_{campaign_id}_{row['platform_subject_id']}"
        run_root = authority / 'runs' / run_id
        manifest_path = run_root / 'run_manifest.json'
        manifest = {'run_id': run_id, 'subject_id': row['key'], 'execution_class': 'WITNESS', **row['sections']}
        if manifest_path.exists(): require(manifest_path.read_bytes() == json_bytes(manifest), 'CORPUS_CONTINUATION_RUN_MANIFEST_DRIFT')
        else: write(manifest_path, manifest)
        run_bindings.append({'subject_id': row['platform_subject_id'], 'subject_input_id': row['subject_input_id'],
            'run_manifest_path': str(manifest_path), 'run_ledger_path': str(run_root / 'ACTIVE_RUN_V2.json')})
    bootstrap_receipt = output / 'CAMPAIGN_BOOTSTRAP.json'
    write(bootstrap_receipt, {**old, 'campaign_id': campaign_id, 'release_id': state['release_id'], 'run_bindings': run_bindings,
          'campaign_name': descriptor['campaign_name'],
          'source_dependency_repair_index': {'path': str(repair_index), 'sha256': sha256_file(repair_index)}})
    prepared = {**parent, 'source_code_sha': args.expected_main_sha, 'campaign_id': campaign_id, 'experiment_id': state['experiment_id'],
        'engine_release_id': state['release_id'], 'parent_campaign_id': parent['campaign_id'], 'superseded_stage_ids': direct,
        'parent_preparation': descriptor['original_preparation'],
        'bootstrap_receipt': {'path': str(bootstrap_receipt), 'sha256': sha256_file(bootstrap_receipt)},
        'source_dependency_repair_index': {'path': str(repair_index), 'sha256': sha256_file(repair_index)},
        'bundle_role': 'BASE_INPUTS_PLUS_EXPLICIT_HASH_BOUND_DEPENDENCY_CLOSURES', 'measurement_submitted': False}
    prepared.pop('campaign_run', None)
    write(output / 'PREPARATION.json', prepared)
    if campaign['state'] == 'READY':
        experiment = api(args.api, 'GET', '/v1/experiments/' + state['experiment_id'])
        if experiment.get('lifecycle_state', (experiment.get('experiment') or {}).get('lifecycle_state')) == 'PREREGISTERED':
            api(args.api, 'POST', '/v1/experiments/transition', {'experiment_id': state['experiment_id'], 'to_state': 'RUNNING', 'actor': actor})
        prepared['campaign_run'] = api(args.api, 'POST', '/v1/research/campaigns/run', {'campaign_id': campaign_id,
            'members': run_bindings, 'pipeline_plan_sha256': old['pipeline_plan_sha256'], 'requested_by': actor})
    else:
        require(campaign['state'] in {'RUNNING', 'FAILED', 'COMPLETED'}, 'CORPUS_CONTINUATION_RESUME_STATE')
    prepared['measurement_submitted'] = True
    write(output / 'PREPARATION.json', prepared)
    print(json.dumps({'campaign_id': campaign_id, 'parent_campaign_id': parent['campaign_id'], 'member_count': len(rows),
          'repaired_source_dependency_subject_count': sum(bool(r['changed_input_roles']) for r in rows),
          'unchanged_subject_inputs_reused': sum(not r['changed_input_roles'] for r in rows), 'training_submitted': False}))


def main():
    parser = argparse.ArgumentParser()
    for name in ('request', 'repo', 'authority-root', 'platform-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-main-sha', required=True)
    parser.add_argument('--api', default='http://127.0.0.1:8080')
    continue_corpus(parser.parse_args())


if __name__ == '__main__':
    main()
