"""Six-head canonical GLB census regression, original truth remains unapproved."""
import unittest
from tools.corpus.truth_foundry.uniml_glb_six_head_physical_inventory_v1 import examine,make_targets,HEADS


class SixHeadPhysicalCensusTest(unittest.TestCase):
    def root(self):
        return {
            'nodes':[{'name':'Root','children':[1]},{'name':'Arm','children':[]}],
            'skins':[{'joints':[0,1],'inverseBindMatrices':7}],
            'accessors':[
                {'type':'VEC3','count':4}, {'type':'VEC2','count':4},
                {'type':'VEC4','count':4}, {'type':'VEC4','count':4},
                {'type':'SCALAR','count':6}, {'type':'VEC3','count':4},
                {'type':'VEC3','count':3}, {'type':'MAT4','count':2},
            ],
            'meshes':[{'primitives':[{'indices':4,'material':0,'attributes':{
                'POSITION':0,'TEXCOORD_0':1,'JOINTS_0':2,'WEIGHTS_0':3,
                'NORMAL':5}}]}],
            'materials':[{}],
            'animations':[{'samplers':[{}],'channels':[{'sampler':0,'target':{'node':1,'path':'rotation'}}]}]
        }
    def pixels(self):
        return {'decoded_image_count':1,'candidate_pixel_rasterization_possible':True,'glb_sha256':'a'*64}
    def test_physical_receipt_does_not_claim_original_rights(self):
        s=examine(self.root(),self.pixels())
        self.assertEqual(s['indices_triangle_count_estimate'],2)
        self.assertEqual(s['skin_joints_unique_node_count'],2)
        self.assertEqual(s['skin_inverse_bind_matrix_accessor_pass'],1)
        self.assertEqual(s['primitive_with_joints_weights_0_count'],1)
        self.assertEqual(s['animation_rotation_channel_count'],1)
        self.assertEqual(s['uv_primitive_count'],1)
        for k in ('independent_source_sha_match','source_face_topology_parity',
                  'source_skin_deform_parity','part_semantic_concept_reviewed',
                  'author_appearance_preserved','fit_training_authorized'):
            self.assertFalse(s[k])
    def test_six_heads_have_court_gated_target_rows(self):
        rows=list(make_targets('UNIML3D:sample',examine(self.root(),self.pixels())))
        self.assertEqual({x['head'] for x in rows},set(HEADS))
        self.assertEqual(len(rows),22)
        self.assertTrue(all(x['head_target_training_admitted']=='False' for x in rows))
        self.assertTrue(all(x['independent_truth_court']=='NOT_RUN' for x in rows))
    def test_reject_broken_graph(self):
        r=self.root();r['nodes'][0]['children']=[9]
        with self.assertRaisesRegex(ValueError,'BAD_CHILD_ID'):
            examine(r,self.pixels())
    def test_sparse_missing_accessor_is_not_invented(self):
        r=self.root();del r['meshes'][0]['primitives'][0]['attributes']['WEIGHTS_0']
        s=examine(r,self.pixels())
        self.assertEqual(s['primitive_with_joints_weights_0_count'],0)
        self.assertEqual(s['skin_joints_unique_node_count'],2)

if __name__=='__main__':unittest.main()