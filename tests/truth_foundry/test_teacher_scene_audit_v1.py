from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from tools.truth_foundry.audit_teacher_scene_v1 import inspect
from tools.truth_foundry.audit_teacher_scene_v2 import inspect as inspect_v2, inspect_export as inspect_export_v2
from tools.truth_foundry.audit_native_teacher_bytes_v1 import sha256


class TeacherSceneAuditTests(unittest.TestCase):
    def fixture(self,root):
        rest=np.array([[0.,0,0],[1.,0,0],[0,1.,0]])
        translation=np.array([.2,-.3,.1]);controls=[{'opaque_privileged_handle':'opaque'}]
        actions=[];positions=[]
        for ai,h in enumerate((.002,.01,.05)):
            for sign in (-1,1):
                c,s=np.cos(sign*h),np.sin(sign*h);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                positions.append(rest@rotation.T)
                actions.append({'control_index_privileged':0,'amplitude_index':ai,'amplitude':h,'sign':sign,
                    'units':'RADIAN','kind':'ISOLATED_KINEMATIC_PARAMETER_PERTURBATION','opaque_privileged_handle':'opaque'})
        arrays={'rest_vertices_source_world':rest,'faces':np.array([[0,1,2]]),
            'triangle_uv':np.full((1,3,2),np.nan),'triangle_uv_valid':np.array([False]),
            'face_owner_privileged':np.array([0]),'face_material_privileged':np.array([0]),
            'pose_identity_baseline':rest,'sham_before':rest,'sham_after':rest,
            'root_control_position':rest+translation,'isolated_position':np.array(positions),
            'source_weight_offsets':np.array([0,1,2,3]),'source_weight_bone_indices_privileged':np.array([0,0,0]),
            'source_weight_values':np.ones(3),'source_weight_coverage':np.ones(3,bool)}
        path=root/'arrays.npz';np.savez_compressed(path,**arrays)
        ref={'path':'/unavailable/old/host/arrays.npz','sha256':sha256(path),'size_bytes':path.stat().st_size,
             'arrays':{k:{'dtype':v.dtype.str,'shape':list(v.shape)} for k,v in arrays.items()}}
        scene={'schema':'RealSaS.TeacherSceneIR.v1','authority':'UNQUALIFIED_DERIVED_MEASUREMENTS',
            'qualification_minted':False,'student_input_allowed':False,'subject_key':'fixture','source_sha256':'1'*64,
            'geometry_payloads':[ref],'privileged_control_refs':controls,
            'calibration':{'actions':actions,'native_control_frames_privileged':[{'name':'privileged'}],
                'root_translation_source_units':translation.tolist(),'source_rest_to_pose_identity_max_abs':0.,
                'sham_repeat_max_abs':0.,'common_root_translation_max_abs':0.}}
        return scene,{ref['sha256']:path}

    def test_unknown_uv_is_masked_and_static_raw_audit_cannot_infer_clock_law_or_student_observability(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene,available=self.fixture(Path(tmp));result=inspect(scene,available)
            self.assertEqual(result['isolated_actions'],6);self.assertEqual(result['valid_uv_faces'],0)
            self.assertFalse(result['action_law_selected']);self.assertFalse(result['dynamic_timebase_inferred'])
            self.assertFalse(result['truth_qualification_minted'])
            self.assertEqual(result['source_weight_lbs_equivalence'],'NOT_QUALIFIED')
            self.assertEqual(len(result['adjacent_derivative_agreement']),2)

    def test_missing_sign_or_rebound_control_cannot_become_a_central_derivative(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene,available=self.fixture(Path(tmp));changed=deepcopy(scene)
            changed['calibration']['actions'][0]['sign']=1
            with self.assertRaisesRegex(ValueError,'DUPLICATE_ACTION'):inspect(changed,available)
            changed=deepcopy(scene);changed['calibration']['actions'][0]['opaque_privileged_handle']='other-control'
            with self.assertRaisesRegex(ValueError,'ACTION_IDENTITY'):inspect(changed,available)

    def test_reported_perfect_reset_does_not_hide_array_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene,available=self.fixture(Path(tmp));path=next(iter(available.values()))
            with np.load(path) as archive:arrays={k:archive[k] for k in archive.files}
            arrays['sham_after']=arrays['sham_after']+1
            np.savez_compressed(path,**arrays)
            # Simulate new immutable payload identity without updating its claimed
            # reset diagnostic. Numeric admission must inspect actual arrays.
            scene['geometry_payloads'][0].update(sha256=sha256(path),size_bytes=path.stat().st_size)
            with self.assertRaisesRegex(ValueError,'REPORTED_DIAGNOSTICS_DRIFT'):
                inspect(scene,{sha256(path):path})

    def test_v2_rechecks_actual_import_geometry_and_distinct_runtime_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scene,available=self.fixture(root)
            imported=root/'imported.json'
            imported.write_text(json.dumps({'vertices':[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
                'faces':[[0,1,2]],'objects':[{'name':'mesh','vertex_count':3}]}))
            replay_sha=sha256(imported);available[replay_sha]=imported
            scene.update(schema='RealSaS.TeacherSceneIR.v2',native_clip_refs=[],native_temporal_rest_compatible=False,
                evaluator_binary_sha256='b'*64,observation_rest_binding={
                    'schema':'RealSaS.TeacherObservationRestBinding.v2','status':'PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY',
                    'source_sha256':scene['source_sha256'],'imported_geometry_sha256':replay_sha,
                    'student_supervision_authorized':False,'training_authorized':False,
                    'native_clip_rest_compatibility_claimed':False,'product_authority_changed':False,
                    'geometry':{'vertex_count':3,'triangle_count':1,'object_count':1,
                        'vertices_exact_after_declared_object_order':True,'triangles_exact_after_declared_object_order':True,
                        'nearest_neighbour_transfer_used':False}},observation_runtime_binding={
                    'status':'PASS_EXACT_ADMITTED_IMPORT_BYTES','imported_geometry_sha256':replay_sha,
                    'evaluator_binary_sha256':'b'*64,'observation_producer_binary_sha256':'a'*64,'binary_identity_equal':False})
            result=inspect_v2(scene,available)
            self.assertEqual(result['isolated_actions'],6)
            self.assertFalse(result['truth_qualification_minted'])
            with self.assertRaisesRegex(ValueError,'AUTHORITY_DRIFT'):inspect(scene,available)
            bad=deepcopy(scene);bad['observation_runtime_binding']['binary_identity_equal']=True
            with self.assertRaisesRegex(ValueError,'RUNTIME_BINDING_DRIFT'):inspect_v2(bad,available)
            bad=deepcopy(scene);bad['native_temporal_rest_compatible']=True
            with self.assertRaisesRegex(ValueError,'AUTHORITY_DRIFT'):inspect_v2(bad,available)
            path=next(iter(available.values()))
            with np.load(path) as archive:arrays={k:archive[k] for k in archive.files}
            arrays['rest_vertices_source_world']=arrays['rest_vertices_source_world']+.1
            np.savez_compressed(path,**arrays)
            scene['geometry_payloads'][0].update(sha256=sha256(path),size_bytes=path.stat().st_size)
            available[sha256(path)]=path
            with self.assertRaisesRegex(ValueError,'VERTEX_DRIFT'):inspect_v2(scene,available)


    def test_v2_partial_or_mismatched_cohort_export_cannot_claim_complete_audit(self):
        with self.assertRaisesRegex(ValueError,'COMPLETE_MEASUREMENT_REQUIRED'):
            inspect_export_v2({'schema':'RealSaS.TeacherCalibrationObservation.v1',
                'all_measurements_committed_and_exported':False,'measurement_target_goal_complete':False})
        observation={'schema':'RealSaS.TeacherCalibrationObservation.v1',
            'all_measurements_committed_and_exported':True,'measurement_target_goal_complete':True,
            'campaign':{'execution_scope':'SOURCE_ENTRY_TO_RESEARCH_COMPONENT','full_closure_complete':False,
                'members':[{'attempt_id':'missing','role':'COHORT','final_state':'COMPLETED'}]},'exports':[]}
        with self.assertRaisesRegex(ValueError,'COHORT_DRIFT'):inspect_export_v2(observation)
