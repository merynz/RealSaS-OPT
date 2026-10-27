import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.truth_foundry.observe_calibration_v1 import STAGE, export_member, export_diagnostics, observe
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256


class CalibrationObservationTests(unittest.TestCase):
    def test_failed_measurement_preserves_exact_run_ledger_and_blender_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';stage=run/'artifacts'/STAGE;stage.mkdir(parents=True)
            ledger=run/'ACTIVE_RUN_V2.json';ledger.write_text('{"status":"FAIL"}')
            manifest=run/'run_manifest.json';manifest.write_text('{}')
            (stage/'bound-job.blender.log').write_text('Traceback: measured producer error')
            (stage/'failure_diagnostics.json').write_text('{"exception_type":"FixtureFailure"}')
            bootstrap=root/'BOOTSTRAP.json';bootstrap.write_text(json.dumps({'run_bindings':[{
                'subject_id':'subject','run_ledger_path':str(ledger),'run_manifest_path':str(manifest)}]}))
            prep={'bootstrap_receipt':{'path':str(bootstrap),'sha256':sha256(bootstrap)},
                  'members':[{'subject_id':'subject','attempt_id':'attempt','role':'COHORT'}]}
            result=export_diagnostics(prep,root/'export')
            self.assertEqual(len(result[0]['files']),4)
            self.assertEqual((root/'export/attempt/diagnostics/bound-job.blender.log').read_text(),
                             'Traceback: measured producer error')
            bootstrap.write_text('{}')
            with self.assertRaisesRegex(ValueError,'BOOTSTRAP_BYTES_DRIFT'):
                export_diagnostics(prep,root/'export')

    def fixture(self,root):
        member={'subject_id':'subject','attempt_id':'original-attempt','role':'COHORT'}
        payload=root/'payload';payload.write_bytes(b'raw measured array bytes')
        ref={'storage_key':payload.name,'content_sha256':sha256(payload),'size_bytes':payload.stat().st_size,
             'relative_path':'artifacts/calibration/raw.npz','payload_schema':'RealSaS.TeacherCalibrationArrays.v1',
             'authority_class':'UNQUALIFIED_DERIVED_MEASUREMENTS'}
        manifest=root/'manifest';manifest.write_text(json.dumps({'schema':'RealSaS.StageResultManifest.v1',
            'stage_id':STAGE,'compiler_status':'PASS','outputs':[ref]}))
        record={'id':'artifact','storage_key':manifest.name,'content_sha256':sha256(manifest),
                'size_bytes':manifest.stat().st_size,'producer_contract':STAGE}
        binding={'source_kind':'ATTEMPT','source_id':member['attempt_id'],'role':'stage:'+STAGE,
                 'artifact_id':'artifact','content_sha256':sha256(manifest)}
        return member,payload,record,binding

    def test_export_uses_original_attempt_not_newest_subject_alias_and_rejects_cas_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);member,payload,record,binding=self.fixture(root)
            wrong={**binding,'source_id':'newer-unrelated-attempt','artifact_id':'wrong-artifact'}
            calls=[]
            def get(api,method,path):
                calls.append((method,path));self.assertEqual(method,'GET')
                if path.startswith('/v1/subjects/'):return [wrong,binding]
                self.assertEqual(path,'/v1/artifacts/artifact');return record
            with patch('tools.truth_foundry.observe_calibration_v1.api',side_effect=get):
                result=export_member(member,api_root='fixture',artifact_root=root,output=root/'export')
                self.assertFalse(result['truth_qualification_inferred'])
                self.assertEqual(Path(result['outputs'][0]['path']).read_bytes(),payload.read_bytes())
                payload.write_bytes(b'corrupted same data length')
                with self.assertRaisesRegex(ValueError,'CAS_BYTES_DRIFT'):
                    export_member(member,api_root='fixture',artifact_root=root,output=root/'export2')
            self.assertFalse(any(path.endswith('wrong-artifact') for method,path in calls))

    def test_timeout_preserves_original_pending_campaign_without_mutations_or_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);member,_,_,_=self.fixture(root);calls=[]
            preparation={'schema':'RealSaS.TeacherCalibrationCampaignPreparation.v1','members':[member],
                         'campaign_id':'campaign','experiment_id':'experiment','source_code_sha':'1'*40}
            def get(api,method,path):
                calls.append((method,path));self.assertEqual(method,'GET')
                if path.startswith('/v1/research/'):return {'members':[member],'state':'RUNNING'}
                if path.startswith('/v1/attempts/'):return {'attempt_id':member['attempt_id'],'state':'RUNNING','events':[]}
                return []
            with patch('tools.truth_foundry.observe_calibration_v1.api',side_effect=get):
                result=observe(preparation,api_root='fixture',artifact_root=root,output=root/'export',maximum_seconds=0)
            self.assertTrue(result['observation_timed_out'])
            self.assertFalse(result['execution_resubmitted']);self.assertFalse(result['training_submitted'])
            self.assertFalse(result['all_measurements_committed_and_exported'])
            self.assertTrue((root/'export/OBSERVATION.json').is_file())

    def test_member_identity_drift_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);member,_,_,_=self.fixture(root)
            preparation={'schema':'RealSaS.TeacherCalibrationCampaignPreparation.v1','members':[member],
                         'campaign_id':'campaign','experiment_id':'experiment','source_code_sha':'1'*40}
            with patch('tools.truth_foundry.observe_calibration_v1.api',return_value={
                    'members':[{**member,'subject_id':'another-subject'}]}):
                with self.assertRaisesRegex(ValueError,'MEMBER_SCOPE_DRIFT'):
                    observe(preparation,api_root='fixture',artifact_root=root,output=root/'export',maximum_seconds=0)
