"""V2 independent pixel intervention tests. Synthetic only, not corpus truth."""
from __future__ import annotations
import copy
import unittest

from tools.corpus.generic_source_locus_reciprocal_palette_v2 import evaluate, court_document

def fixture():
    full=[0.2,0.2,0.2,1.0]
    return {
        "schema":"RealSaS.Corpus370GenericLocusReciprocalPalette.v2",
        "source_sha256":"a"*64,"camera_sha256":"b"*64,
        "rest_geometry_sha256":"c"*64,
        "view_index":2,"pixel_x":31,"pixel_y":24,
        "locus_a":{"face_index_teacher_only":3,"barycentric_teacher_only":[.2,.3,.5]},
        "locus_b":{"face_index_teacher_only":8,"barycentric_teacher_only":[.1,.3,.6]},
        "carrier_mode":"PREMULTIPLIED_LINEAR_RGB",
        "camera_ray_complete":True,"depth_tie":False,
        "interventions_preserve_geometry_camera_pose_alpha":True,
        "teacher_labels_to_student":False,"owner_identity_minted":False,
        "training_authorized":False,"semantic_or_material_name_as_label":False,
        "rgba":{
            "full":full.copy(),"noop":full.copy(),
            "remove_a":full.copy(),"remove_b":full.copy(),
            "remove_both":full.copy(),
            "recolor_a":[.85,.15,.4,1],"recolor_b":full.copy(),
            "recolor_a_given_remove_b":[.85,.15,.4,1],
            "recolor_b_given_remove_a":full.copy(),
            "recolor_a_noop":full.copy(),"recolor_b_noop":full.copy(),
        },
    }

def render_doc():
    f=fixture()
    return {
        "schema":"RealSaS.Corpus370SourceLocusPaletteIntervention.v1",
        "status":"MEASURED_SOURCE_PALETTE_PILOT__NO_GRAPHICS_FAMILY_AUTHORITY",
        "subject_id":"synthetic_fixture",
        "source_sha256":"a"*64,
        "parents":{
            "schema":"RealSaS.Corpus370SourceLocusPaletteJob.v1",
            "training_authorized":False,"source":{"sha256":"a"*64},
            "camera":{"sha256":"b"*64},"geometry":{"sha256":"c"*64},
        },
        "authority":{"graphics_family_minted":False,"training_authorized":False},
        "rows":[{
            "view_index":2,"pixel_x":31,"pixel_y":24,
            "status":"PASS_SOURCE_RENDER_INTERVENTION_CANDIDATE__DRAWABLE_OWNER_SEPARATE",
            "source_faces_teacher_only":[3,8],
            "full_soup_rgba":f["rgba"]["full"].copy(),
            "removal_rgba":f["rgba"]["remove_a"].copy(),
            "generic_counterfactual_v2":f,
        }],
    }

class ReciprocalCausalCourtTests(unittest.TestCase):
    def test_recolor_detects_weak_removal_a(self):
        r=evaluate(fixture())
        self.assertTrue(r["a_causal"])
        self.assertFalse(r["b_causal"])
        self.assertEqual(r["a_effects"]["remove_direct"],0)
        self.assertGreater(r["a_effects"]["recolor_direct"],.5)
        self.assertEqual(r["same_drawable"],"UNKNOWN")
        self.assertFalse(r["training_authorized"])

    def test_two_layer_redundant_identical_color_detected_after_both_removed(self):
        x=fixture()
        x["rgba"]["remove_both"]=[0,0,0,0]
        r=evaluate(x)
        self.assertTrue(r["a_causal"])
        self.assertTrue(r["b_causal"])
        self.assertEqual(r["status"],"BOTH_LOCI_CAUSALLY_OBSERVED__OWNER_UNKNOWN")
        self.assertEqual(r["qualified_same_owner_positives"],0)

    def test_b_conditional_recolor_reveals_hidden_source(self):
        x=fixture()
        x["rgba"]["recolor_a"]=x["rgba"]["full"].copy()
        x["rgba"]["recolor_a_given_remove_b"]=x["rgba"]["remove_b"].copy()
        x["rgba"]["recolor_b_given_remove_a"]=[.7,.3,.4,1]
        r=evaluate(x)
        self.assertTrue(r["b_causal"])
        self.assertEqual(r["b_effects"]["remove_direct"],0)
        self.assertGreater(r["b_effects"]["recolor_given_a_absent"],.4)

    def test_invisible_source_abstains_not_negative(self):
        x=fixture()
        x["rgba"]["recolor_a"]=x["rgba"]["full"].copy()
        x["rgba"]["recolor_a_given_remove_b"]=x["rgba"]["full"].copy()
        r=evaluate(x)
        self.assertEqual(r["status"],"INSUFFICIENT_PIXEL_CONSEQUENCE__BOTH_ABSTAIN")
        self.assertEqual(r["different_drawable"],"UNKNOWN")

    def test_missing_b_arm_is_not_implicit_zero(self):
        x=fixture()
        del x["rgba"]["recolor_b_given_remove_a"]
        with self.assertRaisesRegex(ValueError,"MISSING_OR_EXTRA_ARM"):evaluate(x)

    def test_noop_drift(self):
        x=fixture()
        x["rgba"]["recolor_b_noop"]=[0,0,0,0]
        with self.assertRaisesRegex(ValueError,"NOOP_SOURCE_DRIFT"):evaluate(x)

    def test_recolor_alpha_drift(self):
        x=fixture()
        x["rgba"]["recolor_b"]=[.2,.3,.4,.2]
        with self.assertRaisesRegex(ValueError,"RECOLOR_CHANGED_ALPHA"):evaluate(x)

    def test_conditioned_recolor_alpha_drift(self):
        x=fixture()
        x["rgba"]["recolor_b_given_remove_a"]=[.2,.3,.4,.2]
        with self.assertRaisesRegex(ValueError,"RECOLOR_CHANGED_ALPHA"):evaluate(x)

    def test_depth_tie_not_minted(self):
        x=fixture();x["depth_tie"]=True
        with self.assertRaisesRegex(ValueError,"INVALID_RAY"):evaluate(x)

    def test_semantic_name_as_label_forbidden(self):
        x=fixture();x["semantic_or_material_name_as_label"]=True
        with self.assertRaisesRegex(ValueError,"AUTHORITY_LEAK"):evaluate(x)

    def test_training_claim_forbidden(self):
        x=fixture();x["training_authorized"]=True
        with self.assertRaisesRegex(ValueError,"AUTHORITY_LEAK"):evaluate(x)

    def test_self_pair_forbidden(self):
        x=fixture();x["locus_b"]=copy.deepcopy(x["locus_a"])
        with self.assertRaisesRegex(ValueError,"SAME_FACE_LOCUS"):evaluate(x)

    def test_document_measured_replay_bound(self):
        doc=render_doc()
        r=court_document(doc)
        self.assertEqual(r["one_locus_count"],1)
        self.assertEqual(r["two_locus_count"],0)
        self.assertEqual(r["qualified_same_owner_positives"],0)
        self.assertFalse(r["graphics_family_authority_minted"])

    def test_document_forged_camera_rejected(self):
        doc=render_doc()
        doc["rows"][0]["generic_counterfactual_v2"]["camera_sha256"]="d"*64
        with self.assertRaisesRegex(ValueError,"PARENT_SOURCE_CAMERA_LOCUS_DRIFT"):court_document(doc)

    def test_document_baseline_drift_rejected(self):
        doc=render_doc()
        doc["rows"][0]["full_soup_rgba"]=[.8,.8,.8,1]
        with self.assertRaisesRegex(ValueError,"V1_RENDER_BASELINE_DRIFT"):court_document(doc)

    def test_document_unqualified_row_abstains_without_forged_owner(self):
        doc=render_doc()
        doc["rows"][0]["status"]="ABSTAIN_ALPHA_COMPOSITING_RECONSTRUCTION"
        del doc["rows"][0]["generic_counterfactual_v2"]
        r=court_document(doc)
        self.assertEqual(r["abstain_count"],1)
        self.assertEqual(r["qualified_different_owner_negatives"],0)

if __name__=="__main__":
    unittest.main()
