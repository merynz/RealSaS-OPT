"""Contract tests: causal measurements cannot silently become semantic or owner labels."""
from __future__ import annotations
import copy
import unittest
from tools.corpus.generic_source_locus_counterfactual_v1 import evaluate_row,court_manifest

def fixture():
    return {
        "schema":"RealSaS.Corpus370GenericLocusCounterfactual.v1",
        "source_sha256":"a"*64, "camera_sha256":"b"*64,
        "rest_geometry_sha256":"c"*64,
        "view_index":0,"pixel_x":12,"pixel_y":22,
        "locus_a":{"face_index_teacher_only":4,
                   "barycentric_teacher_only":[0.3,0.3,0.4]},
        "locus_b":{"face_index_teacher_only":5,
                   "barycentric_teacher_only":[0.2,0.4,0.4]},
        "carrier_mode":"PREMULTIPLIED_LINEAR_RGB",
        "camera_ray_complete":True,"depth_tie":False,
        "interventions_preserve_geometry_camera_pose_alpha":True,
        "teacher_labels_to_student":False,
        "owner_identity_minted":False,"training_authorized":False,
        "semantic_or_material_name_as_label":False,
        "rgba":{
            "full":[0.8,0.2,0.1,1.0],"noop":[0.8,0.2,0.1,1.0],
            "remove_a":[0.2,0.4,0.5,1.0],
            "remove_b":[0.8,0.2,0.1,1.0],
            "remove_both":[0,0,0,0],
            "recolor_a":[0.1,0.8,0.2,1.0],
            "recolor_a_noop":[0.8,0.2,0.1,1.0],
        },
    }

class GenericSourceLocusTests(unittest.TestCase):
    def test_two_causal_loci_not_two_owners(self):
        row=evaluate_row(fixture())
        self.assertTrue(row["measured_a_causal"])
        self.assertTrue(row["measured_b_causal"])
        self.assertTrue(row["effects"]["b_revealed_by_removing_a"])
        self.assertEqual(row["same_drawable"],"UNKNOWN")
        self.assertEqual(row["attachment"],"UNKNOWN")
        self.assertFalse(row["family_authority_minted"])

    def test_same_color_cannot_prove_same_owner(self):
        row=fixture()
        row["rgba"]["recolor_a"]=row["rgba"]["full"]
        output=evaluate_row(row)
        self.assertEqual(output["effects"]["a_recolor"],0.0)
        self.assertEqual(output["same_drawable"],"UNKNOWN")

    def test_both_unobservable_abstains_not_negative(self):
        row=fixture()
        for name in row["rgba"]:
            row["rgba"][name]=[0,0,0,0]
        output=evaluate_row(row)
        self.assertEqual(output["status"],"BOTH_LOCI_ABSTAIN")
        self.assertEqual(output["graphics_owner"],"UNKNOWN")

    def test_one_locus_unobservable_abstains(self):
        row=fixture()
        row["rgba"]["remove_a"]=row["rgba"]["full"]
        row["rgba"]["remove_b"]=row["rgba"]["full"]
        row["rgba"]["remove_both"]=[0.8,0.2,0.1,1.0]
        out=evaluate_row(row)
        self.assertEqual(out["status"],"BOTH_LOCI_ABSTAIN")

    def test_required_removal_arm(self):
        row=fixture()
        del row["rgba"]["remove_both"]
        with self.assertRaisesRegex(ValueError,"MISSING_COUNTERFACTUAL_ARM"):
            evaluate_row(row)

    def test_no_op_parity(self):
        row=fixture()
        row["rgba"]["noop"]=[0,0,0,0]
        with self.assertRaisesRegex(ValueError,"NOOP_PARITY_FAILED"):
            evaluate_row(row)

    def test_alpha_preservation(self):
        row=fixture()
        row["rgba"]["recolor_a"]=[0.1,0.1,0.1,0.1]
        with self.assertRaisesRegex(ValueError,"RECOLOR_ALPHA_CHANGED"):
            evaluate_row(row)

    def test_source_locus_drift(self):
        row=fixture()
        row["locus_b"]=copy.deepcopy(row["locus_a"])
        with self.assertRaisesRegex(ValueError,"SELF_COMPARISON"):
            evaluate_row(row)

    def test_ray_tie(self):
        row=fixture()
        row["depth_tie"]=True
        with self.assertRaisesRegex(ValueError,"INCOMPLETE_OR_TIED_RAY"):
            evaluate_row(row)

    def test_uncontrolled_intervention(self):
        row=fixture()
        row["interventions_preserve_geometry_camera_pose_alpha"]=False
        with self.assertRaisesRegex(ValueError,"COUNTERFACTUAL_UNCONTROLLED"):
            evaluate_row(row)

    def test_semantic_leakage(self):
        row=fixture()
        row["semantic_or_material_name_as_label"]=True
        with self.assertRaisesRegex(ValueError,"FORGED_AUTHORITY_OR_LABEL"):
            evaluate_row(row)

    def test_fake_training_authority(self):
        row=fixture()
        row["training_authorized"]=True
        with self.assertRaisesRegex(ValueError,"FORGED_AUTHORITY_OR_LABEL"):
            evaluate_row(row)

    def test_duplicate_pixel_rejected(self):
        row=fixture()
        doc={"schema":"RealSaS.Corpus370GenericLocusCounterfactualBatch.v1",
             "training_authorized":False,"rows":[row,copy.deepcopy(row)]}
        with self.assertRaisesRegex(ValueError,"DUPLICATE_LOCUS_PAIR_ROW"):
            court_manifest(doc)

    def test_distinct_locus_pairs_at_same_pixel_are_preserved(self):
        first=fixture()
        second=copy.deepcopy(first)
        second["locus_b"]["face_index_teacher_only"]=6
        doc={"schema":"RealSaS.Corpus370GenericLocusCounterfactualBatch.v1",
             "training_authorized":False,"rows":[first,second]}
        result=court_manifest(doc)
        self.assertEqual(len(result["rows"]),2)
        self.assertEqual(result["two_causal_loci_count"],2)
        self.assertFalse(result["training_authorized"])

    def test_reversed_locus_pair_cannot_double_count(self):
        first=fixture()
        reversed_pair=copy.deepcopy(first)
        reversed_pair["locus_a"],reversed_pair["locus_b"]=(
            reversed_pair["locus_b"],reversed_pair["locus_a"])
        doc={"schema":"RealSaS.Corpus370GenericLocusCounterfactualBatch.v1",
             "training_authorized":False,"rows":[first,reversed_pair]}
        with self.assertRaisesRegex(ValueError,"DUPLICATE_LOCUS_PAIR_ROW"):
            court_manifest(doc)

    def test_valid_batch_no_training_authority(self):
        doc={"schema":"RealSaS.Corpus370GenericLocusCounterfactualBatch.v1",
             "training_authorized":False,"rows":[fixture()]}
        output=court_manifest(doc)
        self.assertEqual(output["two_causal_loci_count"],1)
        self.assertFalse(output["training_authorized"])

if __name__=="__main__":
    unittest.main()
