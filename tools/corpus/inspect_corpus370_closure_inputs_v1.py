#!/usr/bin/env python3
"""Read-only host/Platform evidence discovery; never mints family authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request


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
