"""Synthetic fail-closed tests; none of these inputs are training evidence."""
from __future__ import annotations
import pathlib, tempfile, unittest
import numpy as np
from tools.corpus.truth_foundry.evidence_v20 import build_v20 as v20

class NPZNumericCourtContracts(unittest.TestCase):
    def data(self,root,**edits):
        v=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float32)
        d={'vertices':v,'vertices_source':v.copy(),'faces':np.asarray([[0,1,2]],dtype=np.int32),
           'vertex_normals':np.asarray([[0.,0.,1.]]*3,dtype=np.float32),
           'skin':np.asarray([[1.],[1.],[1.]],dtype=np.float32),
           'parents':np.asarray([-1],dtype=np.int32),
           'bone_heads':np.zeros((1,3),dtype=np.float32),'bone_tails':np.ones((1,3),dtype=np.float32),
           'rest_local_source':np.eye(4,dtype=np.float32)[None,:,:],
           'rest_world_source':np.eye(4,dtype=np.float32)[None,:,:]}
        d.update(edits)
        p=root/'test.npz';np.savez_compressed(p,**d)
        return p,['STRUCT','source','READ_STRUCTURE','1','3','1','1','1','Armature','["mesh"]']
    def test_triangle_rig_skin_counts_and_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,s=self.data(pathlib.Path(tmp));r=v20.numeric_court(p,s)
            self.assertEqual((r['vertex_count'],r['face_count'],r['joint_count']),(3,1,1))
            self.assertEqual(r['boundary_edge_count'],3)
            self.assertEqual(r['zero_skin_vertex_count'],0)
            self.assertEqual(r['nonmanifold_edge_count'],0)
            self.assertEqual(r['bad_nonzero_skin_row_sum_count'],0)
    def test_invalid_triangle_index_is_hard_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,s=self.data(pathlib.Path(tmp),faces=np.asarray([[0,1,3]],dtype=np.int32))
            with self.assertRaisesRegex(ValueError,'TRIANGLE_INDEX_OUT_OF_RANGE'):v20.numeric_court(p,s)
    def test_bad_source_structure_count_blocks_court(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,s=self.data(pathlib.Path(tmp));s[4]='4'
            with self.assertRaisesRegex(ValueError,'SOURCE_STRUCTURE_COUNT_PARITY'):v20.numeric_court(p,s)
    def test_parent_cycle_blocks_court(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,s=self.data(pathlib.Path(tmp),parents=np.asarray([0],dtype=np.int32))
            with self.assertRaisesRegex(ValueError,'PARENT_CYCLE_OR_OUT_OF_RANGE'):v20.numeric_court(p,s)
    def test_zero_weight_and_negative_weight_observed_not_silently_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,s=self.data(pathlib.Path(tmp),skin=np.asarray([[1.],[0.],[-0.2]],dtype=np.float32))
            r=v20.numeric_court(p,s)
            self.assertEqual(r['zero_skin_vertex_count'],2)
            self.assertEqual(r['negative_weight_count'],1)
            self.assertTrue(r['quality_has_irregularities'])

if __name__=='__main__':unittest.main()