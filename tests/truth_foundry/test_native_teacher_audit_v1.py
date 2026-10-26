import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.truth_foundry.audit_native_teacher_bytes_v1 import audit_teacher, sha256
from tools.truth_foundry.replay_native_teacher_v1 import compare_arrays, replay


class NativeTeacherAuditTests(unittest.TestCase):
    def test_exact_positions_are_not_isolated_causal_or_physical_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'source').write_bytes(b'sealed source')
            admission={'native_motion_inventory':['native_clip']}
            (p/'admission.json').write_text(json.dumps(admission))
            arrays={'rest_vertices_source_world':np.array([[0.,0,0],[1,0,0],[0,1,0]]),
                    'faces':np.array([[0,1,2]]),'face_owner':np.array([0]),'face_material':np.array([0]),
                    'posed_vertices_source_world':np.zeros((2,3,3))}
            np.savez(p/'teacher.npz',**arrays)
            receipt={'schema':'RealSaS.IRISVNextAuthoredTeacherExport.v1','subject_id':'asset',
                     'source':{'sha256':sha256(p/'source')},'observation_admission_sha256':sha256(p/'admission.json'),
                     'coordinate_domain':'SOURCE_WORLD__NOT_REALSAS_CANONICAL','transport_revision':3,'teacher_only':True,
                     'teacher_rig_skin_motion_allowed_as_inference_input':False,'canonical_M_G_W_minted':False,
                     'native_motion_inventory_expected':['native_clip'],'native_motion_inventory_exported':['native_clip'],
                     'output':{'sha256':sha256(p/'teacher.npz'),'size_bytes':(p/'teacher.npz').stat().st_size},
                     'face_count':1,'sampled_frame_count':2,'rest_vertex_count':3,
                     'sampled_frames':[{'action':'native_clip','frame':0},{'action':'native_clip','frame':1}]}
            (p/'receipt.json').write_text(json.dumps(receipt))
            row={'subject_id':'asset','legacy_subject_id':'asset','teacher_file':'teacher.npz',
                 'receipt_file':'receipt.json','receipt_sha256':sha256(p/'receipt.json'),
                 'teacher_sha256':sha256(p/'teacher.npz'),'rest_vertex_count':3,'sampled_frame_count':2,
                 'native_motion_count':1,'teacher_transport_revision':3}
            source_row={'raw_source':{'path':'source','sha256':sha256(p/'source')},
                        'source_scene_evidence':{'iris_observation_admission':{'path':'admission.json','sha256':sha256(p/'admission.json')}}}
            result=audit_teacher(row,teachers=p,source_row=source_row,stage01_root=p)
            self.assertTrue(result['byte_schema_verified'])
            for k in ('isolated_intervention_available','source_timebase_qualified','source_appearance_sampler_available',
                      'complete_ray_truth_available','physics_solver_available','qualification_minted'):
                self.assertFalse(result[k])
            (p/'source').write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError,'SOURCE_BYTES_DRIFT'):
                audit_teacher(row,teachers=p,source_row=source_row,stage01_root=p)

    def test_replay_detects_changed_value_and_layout_and_forbids_new_research(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=p/'a.npz';b=p/'b.npz'
            np.savez(a,positions=np.array([1.,2.]));np.savez(b,positions=np.array([1.,2.]))
            self.assertTrue(compare_arrays(a,b)['exact_arrays_equal'])
            np.savez(b,positions=np.array([1.,2.01]))
            self.assertFalse(compare_arrays(a,b)['exact_arrays_equal'])
            np.savez(b,positions=np.array([[1.,2.]]))
            with self.assertRaisesRegex(ValueError,'LAYOUT_DRIFT'):compare_arrays(a,b)
            with self.assertRaisesRegex(ValueError,'REQUEST_SCOPE'):
                replay({'schema':'RealSaS.NativeTeacherReplayRequest.v1','training_allowed':True},
                       repo=p,authority_root=p,blender='blender',output=p/'out')

    def test_historical_transport_is_byte_identical_to_sealed_provenance(self):
        root=Path(__file__).resolve().parents[2]
        expected={1:'cafeee8de94524b58cf1345bf990c1afffb418ba039326e41f47481ddd02bfbf',
                  2:'8cb821c3d1439cb9a602fcb1c405f5a3cdf91ac6efb4deab0199a6d3eb5eedf2',
                  3:'c8335f06f1e32a18c1ace510d280aa8274bca3920976c80b06a1cfac66b1a140'}
        for version,digest in expected.items():
            self.assertEqual(sha256(root/f'tools/iris_vnext/blender_authored_teacher_export_v{version}.py'),digest)
