import asyncio
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.truth_foundry.drain_failed_calibration_v1 import drain


class CalibrationDrainTests(unittest.TestCase):
    def fixture(self):
        members=[{'subject_id':f's{i}','attempt_id':f'a{i}','role':'COHORT','final_state':'FAILED'} for i in range(3)]
        members.append({'subject_id':'reg','attempt_id':'reg-a','role':'REGRESSION','final_state':'OPEN'})
        prep={'schema':'RealSaS.TeacherCalibrationCampaignPreparation.v1','campaign_id':'campaign',
              'engine_release_id':'release','members':deepcopy(members),'campaign_run':{'receipts':[
                  {'AttemptID':'reg-a','SubjectID':'reg','CommandID':'original-command'}]}}
        request={'schema':'RealSaS.FailedCalibrationDrainRequest.v1','campaign_id':'campaign','attempt_id':'reg-a',
            'terminate_idle_regression_allowed':True,'new_measurements_allowed':False,'training_allowed':False,
            'expected_wait_stage_id':'09_IRIS_FIT_PREREGISTERED','actor':'fixture-operator','reason':'declared fixture repair'}
        campaign={'candidate_engine_release_id':'release','members':members}
        attempt={'attempt_id':'reg-a','state':'OPEN','events':[{'type':'STAGE_WAITING_EXTERNAL_EVIDENCE',
            'payload':{'stage_id':'09_IRIS_FIT_PREREGISTERED'}}]}
        description=SimpleNamespace(id='realsas:compile:original-command',run_id='original-run',
            workflow_type='realsas.compile.v1',status=SimpleNamespace(name='RUNNING'),close_time=None,
            raw_description=SimpleNamespace(pending_activities=[]))
        calls=[]
        class Handle:
            async def describe(self):return description
            async def terminate(self,*,reason):
                calls.append(('terminate',reason));description.status.name='TERMINATED';description.close_time=datetime.now(timezone.utc)
        class Client:
            def get_workflow_handle(self,workflow_id,*,run_id=None):
                calls.append(('handle',workflow_id,run_id));return Handle()
        def get(api,method,route):
            calls.append((method,route));self.assertEqual(method,'GET')
            return deepcopy(campaign if route.startswith('/v1/research/') else attempt)
        def reconcile():
            calls.append(('go-reconcile',));attempt['state']='FAILED'
            attempt['events'].append({'type':'ATTEMPT_WORKFLOW_FAILED','payload':{'workflow_status':'Terminated'}})
        return request,prep,campaign,attempt,description,Client(),get,reconcile,calls

    def test_exact_idle_run_only_and_go_owner_closes_without_new_execution(self):
        r,p,c,a,d,client,get,reconcile,calls=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,patch('tools.truth_foundry.drain_failed_calibration_v1.api',side_effect=get):
            result=asyncio.run(drain(r,p,client=client,api_root='fixture',output=tmp,reconcile=reconcile))
            self.assertTrue((Path(tmp)/'DRAIN_RECEIPT.json').is_file())
        self.assertTrue(result['termination_requested']);self.assertFalse(result['qualification_minted'])
        self.assertFalse(result['new_execution_submitted']);self.assertEqual(result['after']['state'],'FAILED')
        self.assertIn(('handle','realsas:compile:original-command','original-run'),calls)
        self.assertEqual(len([x for x in calls if x[0]=='terminate']),1)
        self.assertGreater(calls.index(('go-reconcile',)),next(i for i,x in enumerate(calls) if x[0]=='terminate'))

    def test_active_cohort_or_different_wait_or_active_activity_cannot_be_drained(self):
        for case in ('active-cohort','different-wait','active-activity','wrong-workflow'):
            with self.subTest(case=case):
                r,p,c,a,d,client,get,reconcile,calls=self.fixture()
                if case=='active-cohort':c['members'][0]['final_state']='OPEN'
                elif case=='different-wait':a['events'][0]['payload']['stage_id']='10_IRIS_FIT'
                elif case=='active-activity':d.raw_description.pending_activities=['running-work']
                else:d.workflow_type='another.workflow'
                with tempfile.TemporaryDirectory() as tmp,patch('tools.truth_foundry.drain_failed_calibration_v1.api',side_effect=get):
                    with self.assertRaises(ValueError):asyncio.run(drain(r,p,client=client,api_root='fixture',output=tmp,reconcile=reconcile))
                self.assertFalse(any(x[0] in {'terminate','go-reconcile'} for x in calls))

    def test_unrelated_attempt_or_changed_members_cannot_be_selected(self):
        r,p,c,a,d,client,get,reconcile,calls=self.fixture();r['attempt_id']='a0'
        with tempfile.TemporaryDirectory() as tmp,patch('tools.truth_foundry.drain_failed_calibration_v1.api',side_effect=get):
            with self.assertRaisesRegex(ValueError,'EXACT_REGRESSION_REQUIRED'):
                asyncio.run(drain(r,p,client=client,api_root='fixture',output=tmp,reconcile=reconcile))
        self.assertFalse(any(x[0]=='terminate' for x in calls))
