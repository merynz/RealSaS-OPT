import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compiler.realsas_compiler_services.orchestrator.mainline import validate_plan, topological_stage_ids
from compiler.realsas_compiler_services.platform_worker.stage_inputs import released_execution_plan
from compiler.realsas_compiler_services.orchestrator.adapters.truth_foundry_v1 import measure_teacher_scene_stage
from tools.truth_foundry.platform_calibration_operator_v1 import immutable_json, prepare
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256

ROOT=Path(__file__).resolve().parents[2]


class CalibrationPlatformTests(unittest.TestCase):
    def test_additive_plan_preserves_full_target_and_real_source_parents(self):
        base=json.loads((ROOT/'canonical/MAINLINE_EXECUTION_PLAN_V2.json').read_text())
        plan=json.loads((ROOT/'canonical/TRUTH_CALIBRATION_EXECUTION_PLAN_V1.json').read_text())
        validate_plan(plan)
        nodes={r['id']:r for r in plan['stages']};original={r['id']:r for r in base['stages']}
        self.assertEqual(set(nodes)-set(original),{'47_SOURCE_RESPONSE_CALIBRATION'})
        for key,node in original.items():
            self.assertEqual(nodes[key]['adapter'],node['adapter'])
        order=topological_stage_ids(plan)
        self.assertLess(order.index('08_NORMALIZATION_DOMAIN_QUALIFIED'),order.index('47_SOURCE_RESPONSE_CALIBRATION'))
        self.assertLess(order.index('47_SOURCE_RESPONSE_CALIBRATION'),order.index('09_IRIS_FIT_PREREGISTERED'))
        self.assertEqual(order[-1],'46_PRODUCT_CLOSURE_SEAL')
        self.assertFalse(nodes['47_SOURCE_RESPONSE_CALIBRATION']['policy']['product_pass_authority'])
        self.assertEqual(set(nodes['47_SOURCE_RESPONSE_CALIBRATION']['depends_on']),{
            '01_SOURCE_BYTES_SEALED','03_SOURCE_MECHANICAL_AUDIT','08_NORMALIZATION_DOMAIN_QUALIFIED'})
        reconstructed=released_execution_plan(base,{'schema':'RealSaS.StageGraph.v1','stage_count':len(plan['stages']),
                                                    'stages':plan['stages']},mode='RESEARCH')
        self.assertEqual(validate_plan(reconstructed),validate_plan(plan))
        params=json.loads((ROOT/'canonical/TRUTH_CALIBRATION_STAGE_PARAMETERS_V1.json').read_text())
        self.assertEqual(params['stages']['09_IRIS_FIT_PREREGISTERED']['external_evidence_roles'],['qualified_truth_and_v31_training_seal'])

    def test_explicit_regression_exclusion_cannot_be_truth_or_product_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx={'run_root':Path(tmp),'stage':{'id':'47_SOURCE_RESPONSE_CALIBRATION','policy':{'product_pass_authority':False}},
                 'run_manifest':{'truth_foundry':{'mode':'SEALED_REGRESSION_EXCLUDED_FROM_NEW_CORPUS','new_truth_claimed':False}}}
            result=measure_teacher_scene_stage(ctx)
            payload=json.loads(Path(result['outputs'][0]['path']).read_text())
            self.assertFalse(payload['new_truth_produced']);self.assertFalse(payload['qualification_minted'])
            self.assertEqual(result['outputs'][0]['authority_class'],'EXPLICIT_CORPUS_EXCLUSION_ONLY')

    def test_immutable_input_staging_preserves_exact_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=immutable_json(tmp,'input.json',{'data':1});b=immutable_json(tmp,'input.json',{'data':1})
            self.assertEqual(a,b);self.assertEqual(sha256(a['path']),a['sha256'])
            Path(a['path']).write_text('drift')
            with self.assertRaisesRegex(ValueError,'IMMUTABLE_BYTES_DRIFT'):immutable_json(tmp,'input.json',{'data':1})

    def test_coordinator_links_attempts_before_submit_and_observes_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'repo';platform=root/'platform';authority=root/'authority'
            for p in (repo,platform,authority):p.mkdir()
            code='1'*40;(platform/'deployment.json').write_text(json.dumps({'code_sha':code}))
            (repo/'policy.json').write_text(json.dumps({'schema':'RealSaS.TeacherKinematicCalibrationPolicy.v1'}))
            (repo/'descriptor.json').write_text(json.dumps({'regression_witnesses':{}}))
            (repo/'parameters.json').write_text(json.dumps({'schema':'RealSaS.StageReleaseParameters.v1',
                'stages':{'09_IRIS_FIT_PREREGISTERED':{'external_evidence_roles':['qualified_truth_and_v31_training_seal']}}}))
            executable=root/'binary';executable.write_bytes(b'fixture binary')
            rows=[]
            for i in range(3):
                p=authority/f'native-{i}';p.write_bytes(f'native {i}'.encode())
                rows.append({'subject_key':f'asset-{i}','relative':p.name,'sha256':sha256(p)})
            report=authority/'report.json';report.write_text(json.dumps({'status':'EXACT_SEALED_REPLAY',
               'subjects':[{'subject_id':r['subject_key'],'exact_arrays_equal':True} for r in rows]}))
            request={'schema':'RealSaS.TeacherCalibrationOperatorRequest.v1','training_allowed':False,'new_calibration_actions_allowed':True,
                'native_baselines':rows,'source_equivalence_report':{'relative':report.name,'sha256':sha256(report)},
                'evaluator_executable':str(executable),'evaluator_version':'fixture version','maximum_seconds_per_subject':60,
                'policy_repo_path':'policy.json','input_descriptor_repo_path':'descriptor.json',
                'stage_parameters_repo_path':'parameters.json',
                'campaign_contract_repo_path':'campaign.json','campaign_name':'fixture_campaign','experiment_slug':'fixture-exp',
                'experiment_name':'fixture','research_question':'fixture question','hypothesis':'fixture hypothesis',
                'repair_of_campaign_id':'prior','baseline_engine_release_id':'prior-release'}
            calls=[];state=['READY'];prior_state=['FAILED'];captured=[]
            def fake_api(api,method,route,payload=None):
                calls.append((method,route,payload))
                if route=='/v1/experiments':return {'experiment':{'id':'exp','lifecycle_state':'PREREGISTERED'}}
                if route=='/v1/research/campaigns/prior':return {'candidate_engine_release_id':'prior-release',
                    'members':[{'subject_id':str(i),'role':'COHORT','final_state':prior_state[0]} for i in range(3)]}
                if route=='/v1/research/campaigns/campaign':return {'state':state[0],'members':[{'attempt_id':f'attempt-{i}'} for i in range(3)]}
                if route=='/v1/research/campaigns/run':state[0]='RUNNING';return {'state':'RUNNING'}
                return {}
            def fake_bootstrap(args):
                captured.append(json.loads(args.descriptor.read_text()))
                args.out.write_text(json.dumps({'campaign_id':'campaign','release_id':'release','pipeline_plan_sha256':'2'*64,
                    'run_bindings':[{'subject_id':str(i)} for i in range(3)]}))
            def command(args,**kwargs):
                if args[0]=='git':return code+'\trefs/heads/main\n' if 'ls-remote' in args else code+'\n'
                return 'fixture version\n'
            with patch('tools.truth_foundry.platform_calibration_operator_v1.subprocess.check_output',side_effect=command),\
                 patch('tools.truth_foundry.platform_calibration_operator_v1.bootstrap.api',side_effect=fake_api),\
                 patch('tools.truth_foundry.platform_calibration_operator_v1.bootstrap.main',side_effect=fake_bootstrap):
                a=prepare(request,repo=repo,platform_root=platform,authority_root=authority,expected_sha=code,
                    output=root/'out',api_root='fixture',rclone_remote='fixture',submit=True)
                b=prepare(request,repo=repo,platform_root=platform,authority_root=authority,expected_sha=code,
                    output=root/'out',api_root='fixture',rclone_remote='fixture',submit=True)
                prior_state[0]='OPEN'
                with self.assertRaisesRegex(ValueError,'PRIOR_CAMPAIGN_NOT_DRAINED'):
                    prepare(request,repo=repo,platform_root=platform,authority_root=authority,expected_sha=code,
                        output=root/'out',api_root='fixture',rclone_remote='fixture',submit=True)
            submissions=[i for i,c in enumerate(calls) if c[1]=='/v1/research/campaigns/run']
            self.assertEqual(len(submissions),1)
            self.assertEqual(len([c for c in calls[:submissions[0]] if c[1]=='/v1/experiments/link']),4)
            self.assertFalse(a['training_submitted']);self.assertTrue(b['measurement_submitted'])
            self.assertEqual(a['repair_of_campaign_id'],'prior')
            self.assertEqual(captured[0]['baseline_engine_release_id'],'prior-release')
            self.assertEqual(len(captured[0]['repair_expected_campaign_members']),3)
            fence=captured[0]['additional_stage_parameters']['09_IRIS_FIT_PREREGISTERED']['external_evidence_roles']
            self.assertEqual(fence,['qualified_truth_and_v31_training_seal'])
            cfg=captured[0]['additional_manifest_sections']['truth_foundry']
            self.assertEqual(cfg['evaluator_binary']['sha256'],sha256(executable))
            self.assertTrue(Path(cfg['policy']['path']).is_relative_to(authority))
            self.assertEqual(captured[0]['regression_additional_manifest_sections']['truth_foundry']['new_truth_claimed'],False)
