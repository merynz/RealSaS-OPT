import json, unittest
from pathlib import Path
import numpy as np

from tools.iris_vnext.validate_structural_evidence_input_binding_v2 import validate_binding, _validate_view

ROOT=Path(__file__).resolve().parents[2]


class StructuralEvidenceInputBindingV2Tests(unittest.TestCase):
    def test_pending_binding_is_fail_closed_and_valid(self):
        value=json.loads((ROOT/"canonical/STRUCTURAL_EVIDENCE_INPUT_BINDING_V2.json").read_text())
        validate_binding(value)
        self.assertEqual(value["status"],"SOURCE_FREEZE_PENDING__NO_EXECUTION_AUTHORITY")
        self.assertIsNone(value["implementation_source_sha"])
        self.assertFalse(value["scientific_execution_authorized"])
        self.assertFalse(value["optimizer_or_training_authorized"])

    def _arrays(self):
        # One ray, tied minima faces 10/11 and deeper face 12; all unordered pairs, both orientations.
        d={
            "v0_stack_offsets":np.asarray([0,3],np.int64), "v0_stack_face":np.asarray([10,11,12],np.int64),
            "v0_stack_depth":np.asarray([1.,1.,2.],np.float64), "v0_stack_visibility_probability":np.asarray([0.,0.,0.],np.float32),
            "v0_stack_visibility_valid":np.asarray([False,False,True]), "v0_stack_tied_minimum":np.asarray([True,True,False]),
        }
        ranks=[(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)]
        d["v0_face_a"]=np.asarray([10,11,10,12,11,12]); d["v0_face_b"]=np.asarray([11,10,12,10,12,11])
        d["v0_rank_a"]=np.asarray([a for a,b in ranks]); d["v0_rank_b"]=np.asarray([b for a,b in ranks]); d["v0_ray_index"]=np.zeros(6,np.int32)
        d["v0_depth_a"]=np.asarray([1.,1.,1.,2.,1.,2.]); d["v0_depth_b"]=np.asarray([1.,1.,2.,1.,2.,1.])
        d["v0_front_valid"]=np.asarray([False,False,True,True,True,True]); d["v0_front_probability"]=np.asarray([0.,0.,1.,0.,1.,0.])
        sv=np.asarray([0.,0.,0.]); vv=np.asarray([False,False,True])
        d["v0_visibility_probability_a"]=np.asarray([sv[a] for a,b in ranks]); d["v0_visibility_probability_b"]=np.asarray([sv[b] for a,b in ranks])
        d["v0_visibility_valid_a"]=np.asarray([vv[a] for a,b in ranks]); d["v0_visibility_valid_b"]=np.asarray([vv[b] for a,b in ranks])
        d["v0_occlusion_positive_support"]=np.ones(6,np.float32); d["v0_base_pair_id"]=np.asarray([0,0,1,1,2,2])
        return d

    def test_semantic_view_validator_accepts_tie_abstention(self):
        stats=_validate_view(self._arrays(),"v0_",1e-10,"Synthetic",0)
        self.assertEqual(stats["rows"],6); self.assertEqual(stats["front_abstain"],2)

    def test_semantic_view_validator_rejects_rank_fabricated_tie(self):
        d=self._arrays(); d["v0_front_valid"][0]=True; d["v0_front_probability"][0]=1.0
        with self.assertRaisesRegex(ValueError,"V2_FRONT_VALIDITY"):
            _validate_view(d,"v0_",1e-10,"Synthetic",0)


if __name__=="__main__": unittest.main()
