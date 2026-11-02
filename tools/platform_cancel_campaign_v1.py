#!/usr/bin/env python3
"""Execute an explicit lifecycle repair through the canonical Go owner.

This pre-deployment operator can retire an exact research cohort so the normal
OPEN-Attempt deployment guard can remain intact. It never changes truth courts,
deletes artifacts, edits ledgers, cancels an undeclared Attempt or submits work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request


def require(value, code):
    if not value:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(descriptor, *, repo, platform_root, expected_main_sha, output):
    require(descriptor.get('schema') == 'RealSaS.ResearchCampaignLifecycleRepair.v1'
            and descriptor.get('training_authorized') is False, 'LIFECYCLE_REPAIR_SCOPE')
    if descriptor.get('enabled') is not True:
        return {'status': 'NOOP_DISABLED'}
    require(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == expected_main_sha,
            'LIFECYCLE_REPAIR_CHECKOUT_DRIFT')
    require(subprocess.check_output(['git', '-C', str(repo), 'ls-remote', 'origin', 'refs/heads/main'], text=True).split()[0] == expected_main_sha,
            'LIFECYCLE_REPAIR_REMOTE_MAIN_DRIFT')
    original = descriptor['original_preparation']
    path = Path(original['path'])
    require(path.is_file() and sha(path) == original['sha256'], 'LIFECYCLE_REPAIR_PREPARATION_DRIFT')
    prepared = json.loads(path.read_text())
    request = descriptor['cancellation']
    require(prepared['campaign_id'] == request['campaign_id']
            and prepared['source_code_sha'] == request['expected_campaign_code_sha']
            and prepared['engine_release_id'] == request['expected_engine_release_id']
            and prepared['member_count'] == len(request['expected_attempt_ids']), 'LIFECYCLE_REPAIR_PREPARATION_SCOPE')
    with urllib.request.urlopen('http://127.0.0.1:8080/v1/research/campaigns/' + request['campaign_id'], timeout=20) as response:
        campaign = json.load(response)
    require({m['attempt_id'] for m in campaign['members']} == set(request['expected_attempt_ids']),
            'LIFECYCLE_REPAIR_CURRENT_MEMBER_DRIFT')
    output.parent.mkdir(parents=True, exist_ok=True)
    if all(m['final_state'] in {'FAILED', 'COMPLETED'} for m in campaign['members']):
        receipt = {'schema': 'RealSaS.ResearchCampaignLifecycleRepairReceipt.v1',
                   'status': 'NOOP_EXACT_ORIGINAL_MEMBERS_ALREADY_TERMINAL',
                   'campaign_id': request['campaign_id'], 'training_authorized': False}
    else:
        binary = output.parent / ('research-lifecycle-owner-' + expected_main_sha)
        subprocess.run(['go', 'build', '-ldflags', '-X main.sourceCodeSHA=' + expected_main_sha,
                        '-o', str(binary), './cmd/realsas-control-worker'], cwd=repo / 'platform', check=True, timeout=300)
        request_path = output.with_suffix('.request.json')
        request_path.write_text(json.dumps(request, sort_keys=True, indent=2) + '\n')
        env = {**os.environ,
            'REALSAS_REPO_ROOT': str(repo),
            'REALSAS_DATABASE_URL': 'postgresql:///postgres?host=' + str(platform_root / 'data/socket') + '&port=55432',
            'REALSAS_ARTIFACT_ROOT': str(platform_root / 'artifacts'),
            'REALSAS_TEMPORAL_ADDRESS': '127.0.0.1:7233', 'REALSAS_TEMPORAL_NAMESPACE': 'default'}
        process = subprocess.run([str(binary), '--cancel-campaign-request', str(request_path),
            '--operator-repo', str(repo)], env=env, text=True, capture_output=True, timeout=120)
        output.with_suffix('.owner.log').write_text(process.stdout + '\n' + process.stderr)
        require(process.returncode == 0, 'LIFECYCLE_REPAIR_GO_OWNER_FAILED__SEE_RETAINED_LOG')
        receipts = []
        for line in process.stdout.splitlines():
            if line.startswith('{'):
                try:
                    value = json.loads(line)
                    if value.get('schema') == 'RealSaS.ResearchCampaignCancellation.v1':
                        receipts.append(value)
                except json.JSONDecodeError:
                    pass
        require(len(receipts) == 1, 'LIFECYCLE_REPAIR_GO_RECEIPT_CARDINALITY')
        receipt = receipts[0]
        require(receipt['campaign_id'] == request['campaign_id'] and receipt['training_authorized'] is False,
                'LIFECYCLE_REPAIR_RECEIPT_SCOPE')
    output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'campaign_id': request['campaign_id'], 'receipt': str(output),
                      'execution_resubmitted': False, 'training_authorized': False}))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--platform-root', type=Path, required=True)
    parser.add_argument('--expected-main-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execute(json.loads(args.request.read_text()), repo=args.repo.resolve(),
        platform_root=args.platform_root.resolve(), expected_main_sha=args.expected_main_sha, output=args.output.resolve())


if __name__ == '__main__':
    main()
