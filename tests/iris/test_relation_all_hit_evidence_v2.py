import unittest
import numpy as np

from tools.iris_vnext.build_fit8_relation_all_hit_evidence_v2 import depth_tie_epsilon, stack_visibility_evidence, oriented_pair_evidence_v2


class RelationAllHitEvidenceV2Tests(unittest.TestCase):
    def test_unique_minimum_and_deeper_visibility(self):
        p,v,t=stack_visibility_evidence(np.asarray([1.0,2.0,3.0]),1e-10)
        np.testing.assert_array_equal(p,[1,0,0]); np.testing.assert_array_equal(v,[True,True,True]); np.testing.assert_array_equal(t,[False,False,False])

    def test_tied_minima_abstain_but_deeper_is_valid_invisible(self):
        p,v,t=stack_visibility_evidence(np.asarray([1.0,1.0,2.0]),1e-10)
        np.testing.assert_array_equal(p,[0,0,0]); np.testing.assert_array_equal(v,[False,False,True]); np.testing.assert_array_equal(t,[True,True,False])

    def test_equal_depth_pair_front_abstains_both_orientations(self):
        all_hits={
            "offsets":np.asarray([0,3],np.int64), "face":np.asarray([4,9,12],np.int64),
            "depth":np.asarray([1.0,1.0,2.0],np.float64),
            "point":np.asarray([[0,0,1],[0,0,1],[0,0,2]],np.float64),
        }
        pairs=oriented_pair_evidence_v2(all_hits=all_hits,pixels=np.asarray([[3,7]],np.int64),center=np.zeros(3),half=2.0,epsilon=1e-10)
        self.assertEqual(len(pairs["front_probability"]),6)
        tie=np.abs(pairs["depth_a"]-pairs["depth_b"])<=1e-10
        self.assertEqual(int(tie.sum()),2)
        self.assertTrue(np.all(~pairs["front_valid"][tie])); self.assertTrue(np.all(pairs["front_probability"][tie]==0))
        # Both tied minima are visibility-abstain; the deep hit is valid-invisible.
        self.assertTrue(np.all(~pairs["visibility_valid_a"][tie]))
        self.assertTrue(np.all(~pairs["visibility_valid_b"][tie]))
        deep_slots=[]
        for i in range(len(tie)):
            if pairs["rank_a"][i]==2: deep_slots.append((pairs["visibility_valid_a"][i],pairs["visibility_probability_a"][i]))
            if pairs["rank_b"][i]==2: deep_slots.append((pairs["visibility_valid_b"][i],pairs["visibility_probability_b"][i]))
        self.assertTrue(deep_slots and all(bool(valid) and float(value)==0.0 for valid,value in deep_slots))

    def test_depth_epsilon_is_scale_relative(self):
        v=np.asarray([[-2.,0,0],[3.,0,0]],np.float64)
        self.assertEqual(depth_tie_epsilon(v),5e-10)


if __name__=="__main__": unittest.main()
