import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.truth_foundry.retire_completed_measurement_campaign_v1 import retire


class MeasurementRetirementTests(unittest.TestCase):
    def fixture(self):
        members = [
            {'subject_id': 's0', 'attempt_id': 'a0', 'role': 'COHORT', 'final_state': 'OPEN'},
            {'subject_id': 's1', 'attempt_id': 'a1', 'role': 'COHORT', 'final_state': 'OPEN'},
            {'subject_id': 'reg', 'attempt_id': 'ar', 'role': 'REGRESSION', 'final_state': 'OPEN'},
        ]
        request = {
            'schema': 'RealSaS.CompletedMeasurementCampaignRetirementRequest.v1',
            'campaign_id': 'campaign',
            'expected_engine_release_id': 'release',
            'expected_source_code_sha': 'c' * 40,
            'measurement_stage_id': '47_SOURCE_RESPONSE_CALIBRATION',
            'expected_wait_stage_id': '09_IRIS_FIT_PREREGISTERED',
            'expected_full_target_stage_id': '46_PRODUCT_CLOSURE_SEAL',
            'expected_member_count': 3,
            'actor': 'fixture-retirement',
            'reason': 'measurement scope complete; full target intentionally not continued',
            'terminate_waiting_workflows_allowed': True,
            'new_measurements_allowed': False,
            'training_allowed': False,
            'truth_qualification_allowed': False,
            'product_pass_allowed': False,
        }
        preparation = {
            'schema': 'RealSaS.TeacherCalibrationCampaignPreparation.v1',
            'campaign_id': 'campaign',
            'engine_release_id': 'release',
            'source_code_sha': 'c' * 40,
            'members': deepcopy(members),
            'campaign_run': {'receipts': [
                {'AttemptID': row['attempt_id'], 'SubjectID': row['subject_id'], 'CommandID': 'cmd-' + row['attempt_id']}
                for row in members
            ]},
        }
        observation = {
            'schema': 'RealSaS.TeacherCalibrationObservation.v1',
            'campaign_id': 'campaign',
            'source_code_sha': 'c' * 40,
            'all_measurements_committed_and_exported': True,
            'errors': [],
            'execution_resubmitted': False,
            'training_submitted': False,
            'truth_qualification_inferred': False,
            'exports': [
                {
                    'attempt_id': row['attempt_id'],
                    'subject_id': row['subject_id'],
                    'role': row['role'],
                    'registry_stage_artifact_id': 'artifact-' + row['attempt_id'],
                    'measurement_committed': True,
                    'truth_qualification_inferred': False,
                }
                for row in members
            ],
        }
        campaign = {
            'campaign_id': 'campaign',
            'candidate_engine_release_id': 'release',
            'canonical_code_sha': 'c' * 40,
            'target_stage_id': '46_PRODUCT_CLOSURE_SEAL',
            'state': 'RUNNING',
            'members': deepcopy(members),
            'any_failed': False,
            'full_closure_complete': False,
        }
        attempts = {
            row['attempt_id']: {
                'attempt_id': row['attempt_id'],
                'state': 'OPEN',
                'events': [{
                    'type': 'STAGE_WAITING_EXTERNAL_EVIDENCE',
                    'payload': {'stage_id': '09_IRIS_FIT_PREREGISTERED'},
                }],
            }
            for row in members
        }
        descriptions = {
            row['attempt_id']: SimpleNamespace(
                id='realsas:compile:cmd-' + row['attempt_id'],
                run_id='run-' + row['attempt_id'],
                workflow_type='realsas.compile.v1',
                status=SimpleNamespace(name='RUNNING'),
                close_time=None,
                raw_description=SimpleNamespace(pending_activities=[]),
            )
            for row in members
        }
        calls = []

        class Handle:
            def __init__(self, attempt_id):
                self.attempt_id = attempt_id
            async def describe(self):
                return descriptions[self.attempt_id]
            async def terminate(self, *, reason):
                calls.append(('terminate', self.attempt_id, reason))
                d = descriptions[self.attempt_id]
                d.status.name = 'TERMINATED'
                d.close_time = datetime.now(timezone.utc)

        class Client:
            def get_workflow_handle(self, workflow_id, *, run_id=None):
                attempt_id = workflow_id.rsplit('cmd-', 1)[1]
                calls.append(('handle', attempt_id, run_id))
                return Handle(attempt_id)

        def get(_api, method, route):
            self.assertEqual(method, 'GET')
            calls.append(('GET', route))
            if route == '/v1/research/campaigns/campaign':
                campaign['any_failed'] = any(a['state'] == 'FAILED' for a in attempts.values())
                campaign['full_closure_complete'] = all(a['state'] == 'COMPLETED' for a in attempts.values())
                current = deepcopy(campaign)
                for row in current['members']:
                    row['final_state'] = attempts[row['attempt_id']]['state']
                return current
            if route.startswith('/v1/attempts/'):
                return deepcopy(attempts[route.rsplit('/', 1)[1]])
            if '/artifacts?' in route:
                subject_id = route.split('/subjects/', 1)[1].split('/artifacts', 1)[0]
                member = next(row for row in members if row['subject_id'] == subject_id)
                aid = member['attempt_id']
                return [{
                    'source_kind': 'ATTEMPT', 'source_id': aid,
                    'role': 'stage:47_SOURCE_RESPONSE_CALIBRATION',
                    'artifact_id': 'artifact-' + aid,
                    'content_sha256': 'd' * 64,
                }]
            if route.startswith('/v1/artifacts/artifact-'):
                aid = route.rsplit('artifact-', 1)[1]
                return {
                    'id': 'artifact-' + aid,
                    'content_sha256': 'd' * 64,
                    'producer_contract': '47_SOURCE_RESPONSE_CALIBRATION',
                }
            raise AssertionError(route)

        def reconcile():
            calls.append(('go-reconcile',))
            for attempt in attempts.values():
                attempt['state'] = 'FAILED'
                attempt['events'].append({
                    'type': 'ATTEMPT_WORKFLOW_FAILED',
                    'payload': {'workflow_status': 'Terminated'},
                })

        return request, preparation, observation, campaign, attempts, descriptions, Client(), get, reconcile, calls

    def test_all_stage47_pass_members_are_retired_only_after_complete_preflight(self):
        r, p, o, c, attempts, descriptions, client, get, reconcile, calls = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, patch(
            'tools.truth_foundry.retire_completed_measurement_campaign_v1.api', side_effect=get
        ):
            result = asyncio.run(retire(r, p, o, client=client, api_root='fixture', output=tmp, reconcile=reconcile))
            self.assertTrue((Path(tmp) / 'RETIREMENT_RECEIPT.json').is_file())
        self.assertEqual(result['termination_requested_count'], 3)
        self.assertFalse(result['full_target_completed'])
        self.assertFalse(result['truth_qualification_minted'])
        self.assertFalse(result['product_pass_minted'])
        self.assertTrue(all(row['state'] == 'FAILED' for row in result['attempts_after'].values()))
        first_terminate = next(i for i, row in enumerate(calls) if row[0] == 'terminate')
        # Every attempt was read at least twice before the first mutation: initial
        # preflight and the all-members unchanged re-read.
        for attempt_id in attempts:
            reads = [i for i, row in enumerate(calls[:first_terminate]) if row == ('GET', '/v1/attempts/' + attempt_id)]
            self.assertGreaterEqual(len(reads), 2)
        self.assertGreater(calls.index(('go-reconcile',)), first_terminate)

    def test_missing_measurement_or_active_activity_fails_before_any_termination(self):
        for case in ('missing-measurement', 'active-activity', 'different-wait'):
            with self.subTest(case=case):
                r, p, o, c, attempts, descriptions, client, get, reconcile, calls = self.fixture()
                if case == 'missing-measurement':
                    o['exports'][0]['measurement_committed'] = False
                elif case == 'active-activity':
                    descriptions['a1'].raw_description.pending_activities = ['active']
                else:
                    attempts['ar']['events'][-1]['payload']['stage_id'] = '10_IRIS_FIT'
                with tempfile.TemporaryDirectory() as tmp, patch(
                    'tools.truth_foundry.retire_completed_measurement_campaign_v1.api', side_effect=get
                ):
                    with self.assertRaises(ValueError):
                        asyncio.run(retire(r, p, o, client=client, api_root='fixture', output=tmp, reconcile=reconcile))
                self.assertFalse(any(row[0] in {'terminate', 'go-reconcile'} for row in calls))


if __name__ == '__main__':
    unittest.main()
