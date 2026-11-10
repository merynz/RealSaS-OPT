#!/usr/bin/env python3
"""Synthetic-only / no access to private corpus or Drive in CI."""
import importlib.util,pathlib,sys,tempfile,json,csv,gzip,unittest
HERE=pathlib.Path(__file__).resolve().parents[2] / 'tools' / 'corpus' / 'truth_foundry' / 'rss_vnext_inspection'

def load(mod_name):
 spec=importlib.util.spec_from_file_location(mod_name,HERE/(mod_name+'.py'))
 module=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module)
 return module

class TestContracts(unittest.TestCase):
 def test_catalog_does_not_mint_readiness(self):
  m=load('build_target_crosswalk')
  self.assertEqual(len(m.LEGACY),14)
  self.assertEqual(len(m.TARGETS),47)
  self.assertEqual(sum(row[2]=='neural' for row in m.TARGETS),43)
  self.assertEqual(sum(row[2]=='qualification_only' for row in m.TARGETS),4)
  self.assertEqual(len({row[0] for row in m.TARGETS}),47)
  self.assertFalse(any('PASS' in code for code in m.STATES_SCOPED))
  self.assertFalse(any(row[2]=='neural' and row[1].startswith('Compiler.') for row in m.TARGETS))
 def test_no_guess_semantics_from_numeric_mesh(self):
  m=load('inspect_local_sources')
  import numpy as np
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d)
   np.savez(d/'sample.npz',vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype='float32'),
     vertices_source=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype='float32'),
     faces=np.array([[0,1,2]],dtype='int32'),vertex_normals=np.array([[0.,0.,1.]]*3,dtype='float32'),
     canonical_transform=np.eye(4,dtype='float32'),inverse_canonical_transform=np.eye(4,dtype='float32'),
     parents=np.array([-1],dtype='int32'),deform_mask=np.array([1],dtype='uint8'),
     skin=np.ones((3,1),dtype='float32'),rest_local_source=np.eye(4,dtype='float32')[None],
     rest_world_source=np.eye(4,dtype='float32')[None],bone_heads=np.array([[0.,0.,0.]],dtype='float32'),
     bone_tails=np.array([[0.,0.,1.]],dtype='float32'))
   (d/'sample.json').write_text(json.dumps({'source':'object.fbx','mesh_objects':[{'name':'Head','vertex_start':0,'vertex_count':3,'face_count':1}],
     'bone_count':1,'bone_names':['Human'] }))
   r=m.judge(d/'sample.npz',d/'sample.json','TEST')
   self.assertEqual(r['status'],'OBSERVED_DERIVATIVE_ONLY_NOT_TRAINING_AUTHORIZED')
   self.assertEqual(r['mesh_object_count'],1)
   self.assertEqual(r['connected_triangle_surface_islands'],1)
   self.assertEqual(r['morphology'],'UNKNOWN')
   self.assertEqual(r['anatomical_front_axis'],'UNKNOWN')
   self.assertEqual(r['rights_training_eligibility'],'UNKNOWN')
   self.assertEqual(r['original_asset_byte_parity'],'NOT_TESTED')
 def test_bad_structure_fails_closed(self):
  m=load('inspect_local_sources')
  import numpy as np
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d)
   np.savez(d/'wrong.npz',vertices=np.ones((3,3),dtype='float32'))
   (d/'wrong.json').write_text('{}')
   r=m.judge(d/'wrong.npz',d/'wrong.json','TEST')
   self.assertEqual(r['status'],'FAIL_DERIVATIVE_STRUCTURE')
   self.assertEqual(r['supervision_admission'],'NOT_COURTED')

if __name__=='__main__':unittest.main()