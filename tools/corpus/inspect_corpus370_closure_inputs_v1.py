#!/usr/bin/env python3
"""Read-only host/Platform evidence discovery; never mints family authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def partial_matrix(authority_root, platform_root, output, preparations, api_root):
    """Replay existing court decisions; explicitly no final readiness seal."""
    from tools.corpus.build_corpus370_subject_family_authority_matrix_v1 import build
    from tools.corpus.derive_corpus370_model_rosters_v1 import derive
    from tools.corpus.render_corpus370_authority_matrix_markdown_v1 import render
    from tools.corpus.build_corpus370_rig_skin_evidence_v1 import build as rig_evidence
    from tools.platform_cohort_bootstrap_v1 import api
    from tools.truth_foundry.observe_corpus370_truth_v1 import verified_cas

    prep = authority_root / 'corpus370_preparation_v1'
    roster = prep / 'CORPUS370_EXACT_SOURCE_ROSTER_V1.json'
    indexes = [prep / 'source_camera_court_v2/CORPUS370_SOURCE_CAMERA_FAMILY_EVIDENCE_V2.json']
    fbx = authority_root / 'corpus370_fbx_rig_skin_v1/CORPUS370_FBX_RIG_SKIN_COURT_INDEX_V1.json'
    if fbx.is_file():
        path = output / 'CURRENT_FBX_FAMILY_EVIDENCE.json'
        path.write_text(json.dumps(rig_evidence(roster, fbx), sort_keys=True, indent=2) + '\n')
        indexes.append(path)
    rows = {}
    campaign_states = []
    for entry in preparations:
        prepared = entry.get('PREPARATION.json')
        if not prepared:
            continue
        campaign = api(api_root, 'GET', '/v1/research/campaigns/' + prepared['campaign_id'])
        counts = {}
        for member in campaign['members']:
            attempt = api(api_root, 'GET', '/v1/attempts/' + member['attempt_id'])
            state = attempt['state']
            counts[state] = counts.get(state, 0) + 1
            # Only Registry-committed stage manifests are considered, never a
            # partially written run directory or an uncommitted candidate.
            bindings = api(api_root, 'GET', f"/v1/subjects/{member['subject_id']}/artifacts?artifact_type=RealSaS.StageResultManifest&schema_version=v1&limit=500")
            matches = [b for b in bindings if b.get('source_id') == member['attempt_id']
                       and b.get('source_kind') == 'ATTEMPT'
                       and b.get('role') == 'stage:49_SUBJECT_TRUTH_FAMILIES']
            if not matches:
                continue
            if len(matches) != 1:
                raise ValueError('READBACK_DUPLICATE_REGISTRY_FAMILY_STAGE')
            record = api(api_root, 'GET', '/v1/artifacts/' + matches[0]['artifact_id'])
            manifest_path = verified_cas(platform_root / 'artifacts', record)
            manifest = json.loads(manifest_path.read_text())
            outputs = manifest.get('outputs', [])
            selected = [r for r in outputs if r['payload_schema'] == 'RealSaS.Corpus370FamilyEvidenceIndex.v1']
            if len(selected) != 1:
                raise ValueError('READBACK_FAMILY_INDEX_CARDINALITY')
            payload_path = verified_cas(platform_root / 'artifacts', selected[0])
            payload = json.loads(payload_path.read_text())
            digests = {r['content_sha256'] for r in outputs}
            for row in payload['subjects']:
                for claim in row['families'].values():
                    if claim['court']['sha256'] not in digests:
                        raise ValueError('READBACK_COURT_NOT_IN_REGISTRY_STAGE_OUTPUTS')
                sid = row['subject_id']
                if sid in rows and rows[sid] != row:
                    raise ValueError('READBACK_CONFLICTING_ATTEMPT_FAMILY_INDEXES:' + sid)
                rows[sid] = row
        campaign_states.append({'campaign_id': prepared['campaign_id'], 'attempt_counts': counts})
    if rows:
        path = output / 'CURRENT_REGISTRY_FAMILY_EVIDENCE.json'
        path.write_text(json.dumps({'schema': 'RealSaS.Corpus370FamilyEvidenceIndex.v1',
            'subject_count': len(rows), 'subjects': list(rows.values()),
            'training_authorized': False}, sort_keys=True, indent=2) + '\n')
        indexes.append(path)
    repo = Path(__file__).resolve().parents[2]
    matrix = build(repo / 'canonical/CORPUS370_SUBJECT_FAMILY_AUTHORITY_POLICY_V1.json', roster, indexes)
    matrix['readback_scope'] = 'PARTIAL_DIAGNOSTIC_PROJECTION__NOT_FINAL_ADMISSION_OR_READINESS'
    matrix['observed_campaigns'] = campaign_states
    matrix_path = output / 'CORPUS370_PARTIAL_AUTHORITY_MATRIX_V1.json'
    matrix_path.write_text(json.dumps(matrix, sort_keys=True, indent=2) + '\n')
    rosters = derive(matrix_path)
    (output / 'CORPUS370_PARTIAL_MODEL_ROSTERS_V1.json').write_text(json.dumps(rosters, sort_keys=True, indent=2) + '\n')
    (output / 'CORPUS370_PARTIAL_AUTHORITY_MATRIX_V1.md').write_text(render(matrix))
    return {'family_rollup': matrix['rollup']['families'],
            'model_rollup': matrix['rollup']['models'], 'campaign_states': campaign_states,
            'registry_family_subject_count': len(rows), 'training_authorized': False}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def inspect(authority_root: Path, platform_root: Path, output: Path, api: str):
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'schema': 'RealSaS.Corpus370ClosureInputReadback.v1',
        'scope': 'READ_ONLY_DIAGNOSTICS__NO_NEW_RESEARCH_AUTHORITY',
        'training_authorized': False, 'api': {}, 'roots': [], 'evidence': [],
        'scan_limit_reached': False,
    }
    for endpoint in ('/v1/ops/status', '/v1/research/campaigns', '/v1/experiments'):
        try:
            with urllib.request.urlopen(api.rstrip('/') + endpoint, timeout=20) as response:
                value = json.load(response)
            report['api'][endpoint] = value
        except Exception as exc:
            report['api'][endpoint] = {'read_error': str(exc)}
    deployment = platform_root / 'deployment.json'
    if deployment.is_file():
        report['deployment'] = json.loads(deployment.read_text())
    report['campaign_preparations'] = []
    preparation_root = authority_root / 'corpus370_truth_preparation'
    if preparation_root.is_dir():
        for scope in sorted(preparation_root.iterdir()):
            if not scope.is_dir() or scope.is_symlink():
                continue
            entry = {'scope': str(scope)}
            for name in ('PREPARATION.json', 'CAMPAIGN_BOOTSTRAP.json', 'CAMPAIGN_BOOTSTRAP.progress.json'):
                path = scope / name
                if path.is_file():
                    value = json.loads(path.read_text())
                    if name.endswith('.progress.json'):
                        entry[name] = {'sealed_subject_count': len(value.get('subjects', {})),
                                       'release_id': value.get('release_id'),
                                       'campaign_id': value.get('campaign_id')}
                    else:
                        entry[name] = value
            report['campaign_preparations'].append(entry)
    report['family_indexes'] = []
    runs_root = authority_root / 'runs'
    if runs_root.is_dir():
        for path in sorted(runs_root.glob('CAMPAIGN_*/artifacts/49_SUBJECT_TRUTH_FAMILIES/SUBJECT_FAMILY_EVIDENCE.json')):
            report['family_indexes'].append({'path': str(path), 'sha256': sha256(path),
                                             'payload': json.loads(path.read_text())})
    try:
        report['partial_matrix'] = partial_matrix(authority_root, platform_root, output,
                                                  report['campaign_preparations'], api)
    except Exception as exc:
        report['partial_matrix'] = {'read_error': str(exc), 'training_authorized': False}
    keywords = ('CORPUS370_', 'CORPUS400_', 'FULL139_', 'SOURCE_UNION_',
                'TRAINING_READINESS', 'TRUTH139_', 'SUBJECT_TRUTH_FAMILY_SWEEP_RECEIPT')
    summary_keys = ('schema', 'status', 'subject_count', 'candidate_subject_count',
                    'prepared_subject_count', 'failure_count', 'rollup',
                    'family_candidate_outcomes', 'training_authorized')
    count = 0
    for top in sorted(authority_root.iterdir()):
        if not top.is_dir() or top.is_symlink():
            continue
        report['roots'].append({'path': str(top), 'children': sorted(p.name for p in top.iterdir())[:100]})
        for base, dirs, files in os.walk(top, followlinks=False):
            relative = Path(base).relative_to(top)
            dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
            if len(relative.parts) >= 6:
                dirs[:] = []
            for name in sorted(files):
                count += 1
                if count > 150000:
                    report['scan_limit_reached'] = True
                    break
                if not name.endswith('.json') or not any(key in name for key in keywords):
                    continue
                path = Path(base) / name
                if path.is_symlink() or path.stat().st_size > 32 << 20:
                    continue
                try:
                    data = json.loads(path.read_text())
                    if not isinstance(data, dict):
                        continue
                    row = {'path': str(path), 'sha256': sha256(path),
                           'size_bytes': path.stat().st_size,
                           'summary': {k: data[k] for k in summary_keys if k in data}}
                    report['evidence'].append(row)
                except Exception as exc:
                    report['evidence'].append({'path': str(path), 'read_error': str(exc)})
            if report['scan_limit_reached']:
                break
        if report['scan_limit_reached']:
            break
    (output / 'CORPUS370_CLOSURE_INPUT_READBACK_V1.json').write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'schema': report['schema'], 'root_count': len(report['roots']),
                      'evidence_count': len(report['evidence']),
                      'scan_limit_reached': report['scan_limit_reached'],
                      'training_authorized': False}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority-root', type=Path, required=True)
    parser.add_argument('--platform-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api', default='http://127.0.0.1:8080')
    args = parser.parse_args()
    inspect(args.authority_root, args.platform_root, args.output, args.api)
