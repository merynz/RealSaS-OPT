import unittest,tempfile,pathlib,sys,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'tools/corpus/truth_foundry/evidence_v13'))
from normalized_array_court_v13 import topology,court
class ArrayCourtTests(unittest.TestCase):
 def test_one_triangle_is_open(self):
  v=np.asarray([[0,0,0],[1,0,0],[0,1,0]],dtype=float);f=np.array([[0,1,2]])
  t=topology(v,f)
  self.assertEqual(t['boundary_edges'],3)
  self.assertEqual(t['nonmanifold_edges'],0)
 def test_opposite_winding_edge(self):
  v=np.zeros((4,3));f=np.array([[0,1,2],[1,0,3]])
  t=topology(v,f)
  self.assertEqual(t['boundary_edges'],4)
  self.assertEqual(t['winding_disagreement_edges'],0)
 def test_same_winding_detected(self):
  v=np.zeros((4,3));f=np.array([[0,1,2],[0,1,3]])
  self.assertEqual(topology(v,f)['winding_disagreement_edges'],1)
 def test_real_schema_cannot_mint_sdf(self):
  # Build a minimal synthetic normalized derivative; no original mesh/source authority.
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'normalized.npz'
   v=np.asarray([[0.,0,0],[1.,0,0],[0,1,0]],dtype=np.float32)
   E=np.eye(4,dtype=np.float32);w=np.ones((3,1),dtype=np.float32)
   np.savez_compressed(p,vertices_source=v,vertices=v,faces=np.array([[0,1,2]]),vertex_normals=np.array([[0,0,1]]*3,dtype=np.float32),canonical_transform=E,inverse_canonical_transform=E,parents=np.array([-1]),deform_mask=np.array([1]),skin=w,rest_local_source=E[None],rest_world_source=E[None],bone_heads=np.array([[0,0,0]],dtype=np.float32),bone_tails=np.array([[0,0,1]],dtype=np.float32))
   r,j=court(p)
   self.assertEqual(r['triangle_count'],1)
   self.assertEqual(r['boundary_edges'],3)
   self.assertFalse(r['global_sdf_sign_admission'])
   self.assertFalse(r['teacher_training_admission'])
   self.assertEqual(len(j),1)
if __name__=='__main__':unittest.main()