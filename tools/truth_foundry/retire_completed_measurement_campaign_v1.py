#!/usr/bin/env python3
"""Retire a pinned calibration campaign after every Stage47 measurement committed.

The historical campaign was released with Product Closure as its durable target,
while the declared operator scope was source-response measurement only.  Once
all exact Stage47 outputs are Registry-bound PASS artifacts, the downstream
Stage09 external-evidence wait is outside that measurement scope.

This recovery never marks the campaign scientifically/product complete.  It
terminates only exact idle Temporal workflows still waiting at the declared
learned-evidence boundary, then lets the existing Go reconciler mark those
full-target Attempts FAILED.  Stage47 PASS artifacts remain immutable authority.
No new execution, training, qualification, model promotion or ProductRevision is
created.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

from tools.platform_cohort_bootstrap_v1 import api
from tools.platform_deploy import runtime_environment
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256, _bound_path
from tools.truth_foundry.observe_calibration_v1 import write_json


def require(value, code):
    if not value:
        raise ValueError(code)


def _member_key(row):
    return (row['subject_id'], row['attempt_id'], row['role'])


def _current_stage47_binding(api_root, member, expected_artifact_id, stage_id):
    records = api(
        api_root,
        'GET',
        f"/v1/subjects/{member['subject_id']}/artifacts?artifact_type=RealSaS.StageResultManifest&schema_version=v1&limit=500",
    )
    rows = [
        row for row in records
        if row['source_kind'] == 'ATTEMPT'
        and row['source_id'] == member['attempt_id']
        and row['role'] == 'stage:' + stage_id
    ]
    require(len(rows) == 1, 'MEASUREMENT_RETIRE_STAGE47_BINDING_REQUIRED')
    row = rows[0]
    require(row['artifact_id'] == expected_artifact_id, 'MEASUREMENT_RETIRE_STAGE47_ARTIFACT_DRIFT')
    record = api(api_root, 'GET', '/v1/artifacts/' + row['artifact_id'])
    require(
        record['id'] == row['artifact_id']
        and record['content_sha256'] == row['content_sha256']
        and record['producer_contract'] == stage_id,
        'MEASUREMENT_RETIRE_STAGE47_REGISTRY_DRIFT',
    )
    return row


async def retire(request, preparation, observation, *, client, api_root, output, reconcile):
    require(
        request['schema'] == 'RealSaS.CompletedMeasurementCampaignRetirementRequest.v1'
        and request['terminate_waiting_workflows_allowed'] is True
        and request['new_measurements_allowed'] is False
        and request['training_allowed'] is False
        and request['truth_qualification_allowed'] is False
        and request['product_pass_allowed'] is False,
        'MEASUREMENT_RETIRE_SCOPE_REQUIRED',
    )
    require(
        preparation['schema'] == 'RealSaS.TeacherCalibrationCampaignPreparation.v1'
        and preparation['campaign_id'] == request['campaign_id']
        and preparation['engine_release_id'] == request['expected_engine_release_id']
        and preparation['source_code_sha'] == request['expected_source_code_sha'],
        'MEASUREMENT_RETIRE_PREPARATION_SCOPE_DRIFT',
    )
    require(
        observation['schema'] == 'RealSaS.TeacherCalibrationObservation.v1'
        and observation['campaign_id'] == request['campaign_id']
        and observation['source_code_sha'] == request['expected_source_code_sha']
        and observation['all_measurements_committed_and_exported'] is True
        and observation['errors'] == []
        and observation['execution_resubmitted'] is False
        and observation['training_submitted'] is False
        and observation['truth_qualification_inferred'] is False,
        'MEASUREMENT_RETIRE_OBSERVATION_NOT_COMPLETE',
    )

    campaign = api(api_root, 'GET', '/v1/research/campaigns/' + request['campaign_id'])
    require(campaign['campaign_id'] == request['campaign_id'], 'MEASUREMENT_RETIRE_CAMPAIGN_ID_DRIFT')
    require(campaign['candidate_engine_release_id'] == request['expected_engine_release_id'],
            'MEASUREMENT_RETIRE_RELEASE_DRIFT')
    require(campaign['canonical_code_sha'] == request['expected_source_code_sha'],
            'MEASUREMENT_RETIRE_CODE_DRIFT')
    require(campaign['target_stage_id'] == request['expected_full_target_stage_id'],
            'MEASUREMENT_RETIRE_FULL_TARGET_DRIFT')
    require(campaign['state'] == 'RUNNING', 'MEASUREMENT_RETIRE_CAMPAIGN_NOT_RUNNING')

    original = {_member_key(row) for row in preparation['members']}
    current = {_member_key(row) for row in campaign['members']}
    require(current == original, 'MEASUREMENT_RETIRE_CAMPAIGN_MEMBER_DRIFT')
    require(len(current) == int(request['expected_member_count']), 'MEASUREMENT_RETIRE_MEMBER_COUNT_DRIFT')

    exports = {row['attempt_id']: row for row in observation['exports']}
    require(set(exports) == {row['attempt_id'] for row in campaign['members']},
            'MEASUREMENT_RETIRE_EXPORT_MEMBER_DRIFT')
    require(all(row['measurement_committed'] is True and row['truth_qualification_inferred'] is False
                for row in exports.values()), 'MEASUREMENT_RETIRE_STAGE47_NOT_MEASURED')

    receipts = {row['AttemptID']: row for row in preparation['campaign_run']['receipts']}
    require(set(receipts) == set(exports), 'MEASUREMENT_RETIRE_COMMAND_BINDING_DRIFT')

    # Full preflight of all members before any workflow mutation.
    snapshots = {}
    workflows = {}
    for member in campaign['members']:
        attempt_id = member['attempt_id']
        attempt = api(api_root, 'GET', '/v1/attempts/' + attempt_id)
        require(attempt['attempt_id'] == attempt_id and attempt['state'] == 'OPEN',
                'MEASUREMENT_RETIRE_ATTEMPT_NOT_OPEN')
        last = attempt['events'][-1] if attempt['events'] else {}
        require(
            last.get('type') == 'STAGE_WAITING_EXTERNAL_EVIDENCE'
            and last.get('payload', {}).get('stage_id') == request['expected_wait_stage_id'],
            'MEASUREMENT_RETIRE_NOT_AT_DECLARED_WAIT',
        )
        exported = exports[attempt_id]
        require(exported['subject_id'] == member['subject_id'] and exported['role'] == member['role'],
                'MEASUREMENT_RETIRE_EXPORTED_IDENTITY_DRIFT')
        _current_stage47_binding(
            api_root, member, exported['registry_stage_artifact_id'], request['measurement_stage_id']
        )
        receipt = receipts[attempt_id]
        require(receipt['SubjectID'] == member['subject_id'], 'MEASUREMENT_RETIRE_RECEIPT_SUBJECT_DRIFT')
        workflow_id = 'realsas:compile:' + receipt['CommandID']
        description = await client.get_workflow_handle(workflow_id).describe()
        require(
            description.id == workflow_id
            and description.workflow_type == 'realsas.compile.v1'
            and bool(description.run_id),
            'MEASUREMENT_RETIRE_TEMPORAL_SCOPE_DRIFT',
        )
        status = description.status.name
        require(status in {'RUNNING', 'FAILED', 'CANCELED', 'TERMINATED', 'TIMED_OUT'},
                'MEASUREMENT_RETIRE_WORKFLOW_STATE_UNSUPPORTED')
        if status == 'RUNNING':
            require(not description.raw_description.pending_activities,
                    'MEASUREMENT_RETIRE_ACTIVE_ACTIVITY_FORBIDDEN')
        snapshots[attempt_id] = attempt
        workflows[attempt_id] = {
            'workflow_id': workflow_id,
            'run_id': description.run_id,
            'status_before': status,
        }

    # Re-read every Attempt after the complete preflight.  Any concurrent change
    # aborts the recovery before the first termination.
    for attempt_id, snapshot in snapshots.items():
        require(api(api_root, 'GET', '/v1/attempts/' + attempt_id) == snapshot,
                'MEASUREMENT_RETIRE_ATTEMPT_CHANGED_DURING_PREFLIGHT')

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        'schema': 'RealSaS.CompletedMeasurementCampaignRetirementReceipt.v1',
        'campaign_id': request['campaign_id'],
        'reason': request['reason'],
        'actor': request['actor'],
        'measurement_stage_id': request['measurement_stage_id'],
        'measurement_member_count': len(exports),
        'all_measurements_committed_and_exported': True,
        'full_target_stage_id': request['expected_full_target_stage_id'],
        'full_target_completed': False,
        'termination_requested_count': 0,
        'new_execution_submitted': False,
        'new_measurements_submitted': False,
        'training_submitted': False,
        'truth_qualification_minted': False,
        'product_pass_minted': False,
        'workflows': workflows,
    }
    receipt_path = output / 'RETIREMENT_RECEIPT.json'
    write_json(receipt_path, result)

    for attempt_id, row in workflows.items():
        if row['status_before'] != 'RUNNING':
            continue
        handle = client.get_workflow_handle(row['workflow_id'], run_id=row['run_id'])
        await handle.terminate(reason=request['actor'] + ': ' + request['reason'])
        result['termination_requested_count'] += 1
        write_json(receipt_path, result)

    for attempt_id, row in workflows.items():
        handle = client.get_workflow_handle(row['workflow_id'], run_id=row['run_id'])
        closed = await handle.describe()
        require(
            closed.status.name in {'FAILED', 'CANCELED', 'TERMINATED', 'TIMED_OUT'}
            and closed.close_time is not None,
            'MEASUREMENT_RETIRE_TERMINAL_WORKFLOW_PROOF_REQUIRED',
        )
        row['status_after'] = closed.status.name
        row['closed_at'] = closed.close_time.isoformat()
    write_json(receipt_path, result)

    # Existing Go lifecycle owner translates unsuccessful terminal Temporal runs
    # into Attempt FAILED.  This means the old full-Product target is explicitly
    # not claimed complete while Stage47 PASS artifacts remain independently valid.
    reconcile()
    started = time.monotonic()
    after = {}
    while True:
        after = {attempt_id: api(api_root, 'GET', '/v1/attempts/' + attempt_id) for attempt_id in snapshots}
        if all(row['state'] != 'OPEN' for row in after.values()):
            break
        require(time.monotonic() - started < 45, 'MEASUREMENT_RETIRE_GO_RECONCILIATION_PENDING')
        await asyncio.sleep(1)
    require(all(row['state'] == 'FAILED' for row in after.values()),
            'MEASUREMENT_RETIRE_GO_TERMINAL_STATE_REQUIRED')
    require(all(any(event['type'] == 'ATTEMPT_WORKFLOW_FAILED' for event in row['events']) for row in after.values()),
            'MEASUREMENT_RETIRE_GO_TERMINAL_EVENT_REQUIRED')

    final_campaign = api(api_root, 'GET', '/v1/research/campaigns/' + request['campaign_id'])
    require(final_campaign['any_failed'] is True and final_campaign['full_closure_complete'] is False,
            'MEASUREMENT_RETIRE_FULL_CLOSURE_FALSE_REQUIRED')
    result['attempts_after'] = after
    result['campaign_after'] = final_campaign
    write_json(receipt_path, result)
    print(json.dumps({
        'campaign_id': request['campaign_id'],
        'measurement_stage_pass_preserved': True,
        'retired_full_target_attempts': len(after),
        'full_closure_complete': False,
        'new_execution_submitted': False,
        'training_submitted': False,
    }, sort_keys=True), flush=True)
    return result


async def main(args):
    from temporalio.client import Client

    request = json.loads(Path(args.request).read_text())
    require(request['enabled'] is True, 'MEASUREMENT_RETIRE_DISABLED')
    authority = Path(args.authority_root)
    preparation_path = _bound_path(authority, request['preparation_relative_path'])
    observation_path = _bound_path(authority, request['observation_relative_path'])
    require(sha256(preparation_path) == request['preparation_sha256'],
            'MEASUREMENT_RETIRE_PREPARATION_BYTES_DRIFT')
    require(sha256(observation_path) == request['observation_sha256'],
            'MEASUREMENT_RETIRE_OBSERVATION_BYTES_DRIFT')
    preparation = json.loads(preparation_path.read_text())
    observation = json.loads(observation_path.read_text())
    root = Path(args.platform_root)
    client = await Client.connect('127.0.0.1:7233', namespace='default')

    def reconcile():
        env = {
            **os.environ,
            **runtime_environment(root, authority, os.environ.get('LD_LIBRARY_PATH', '')),
        }
        subprocess.run(
            [str(root / 'current/bin/realsas-control-worker'), '--reconcile-once'],
            env=env,
            check=True,
            timeout=45,
        )

    await retire(
        request,
        preparation,
        observation,
        client=client,
        api_root='http://127.0.0.1:8080',
        output=args.output,
        reconcile=reconcile,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'platform-root', 'authority-root', 'output'):
        parser.add_argument('--' + name, required=True)
    asyncio.run(main(parser.parse_args()))
