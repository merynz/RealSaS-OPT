"""Bounded causal visibility targets; no drawable ownership/training claim."""
from __future__ import annotations

import copy
import math
import unittest

from tools.corpus.court_corpus370_source_locus_visibility_v1 import qualify

def fixture():
    full=[.6,.26,.2,1.0]
    removed=[.9,.5,.5,1.0]
    faces=[3,8]
    bary=[[.2,.3,.5],[.1,.1,.8]]
    v2={
        "schema":"RealSaS.Corpus370GenericLocusReciprocalPalette.v2",
        "source_sha256":"b"*64,"camera_sha256":"c"*64,
        "rest_geometry_sha256":"d"*64,
        "view_index":1,"pixel_x":23,"pixel_y":41,
        "locus_a":{"face_index_teacher_only":faces[0],"barycentric_teacher_only":bary[0]},
        "locus_b":{"face_index_teacher_only":faces[1],"barycentric_teacher_only":bary[1]},
        "carrier_mode":"STRAIGHT_LINEAR_RGB",
        "camera_ray_complete":True,"depth_tie":False,
        "interventions_preserve_geometry_camera_pose_alpha":True,
        "semantic_or_material_name_as_label":False,
        "owner_identity_minted":False,"teacher_labels_to_student":False,
        "training_authorized":False,
        "rgba":{
            "full":full.copy(),"noop":full.copy(),
            "remove_a":removed.copy(),"remove_b":full.copy(),
            "remove_both":[0,0,0,0],
            "recolor_a":[.2,.9,.3,1.0],
            "recolor_b":full.copy(),
            "recolor_a_given_remove_b":[.2,.9,.3,1.0],
            "recolor_b_given_remove_a":[.2,.9,.3,1.0],
            "recolor_a_noop":full.copy(),"recolor_b_noop":full.copy(),
        },
    }
    v1=copy.deepcopy(v2)
    v1["schema"]="RealSaS.Corpus370GenericLocusCounterfactual.v1"
    for k in ("recolor_b","recolor_a_given_remove_b","recolor_b_given_remove_a","recolor_b_noop"):
        del v1["rgba"][k]
    row={
        "status":"PASS_SOURCE_RENDER_INTERVENTION_CANDIDATE__DRAWABLE_OWNER_SEPARATE",
        "view_index":1,"pixel_x":23,"pixel_y":41,
        "source_faces_teacher_only":faces,"source_barycentric_teacher_only":bary,
        "source_material_domains_teacher_only":["mesh0:0","mesh1:0"],
        "material_semantic_name_used_as_label":False,"same_drawable_label_minted":False,
        "original_rgba":full.copy(),"full_soup_rgba":full.copy(),
        "source_reimport_vs_full_soup_max_error":0.0,
        "standalone_rgba":full.copy(),"removal_rgba":removed.copy(),
        "front_face_reconstruction_error":0.0,
        "palette_distinct_rgba":[.8,.1,.1,1.0],
        "palette_collision_rgba":[.9,.15,.15,1.0],
        "palette_alpha_max_delta":0.0,
        "palette_rgb_difference_norm":math.sqrt(.1**2+.05**2+.05**2),
        "different_source_domains":True,"split_and_collision_are_color_only":True,
        "generic_counterfactual":v1,
        "generic_counterfactual_v2":v2,
    }
    return {
        "schema":"RealSaS.Corpus370SourceLocusPaletteIntervention.v1",
        "status":"MEASURED_SOURCE_PALETTE_PILOT__NO_GRAPHICS_FAMILY_AUTHORITY",
        "subject_id":"synthetic", "source_sha256":"b"*64,
        "parents":{
           "schema":"RealSaS.Corpus370SourceLocusPaletteJob.v1",
           "source":{"sha256":"b"*64},"camera":{"sha256":"c"*64},
           "geometry":{"sha256":"d"*64},"training_authorized":False,
        },
        "carrier_mode":"STRAIGHT_LINEAR_RGB",
        "renderer":"BLENDER_CYCLES_EXR_PREMULTIPLIED_CENTRE_RAY",
        "rows":[row],"qualified_intervention_rows":1,"abstain_intervention_rows":0,
        "authority":{
            "material_and_owner_names_teacher_only":True,
            "source_material_alpha_unchanged":True,
            "student_semantic_owner_labels_minted":False,
            "graphics_family_minted":False,"training_authorized":False,
        },
    }

class VisibilityQualificationTests(unittest.TestCase):
    def test_opaque_front_and_hidden_sensitive_back_labels_visibility_only(self):
        x=qualify(fixture())
        self.assertEqual(x["qualified_visible_hit_positives"],1)
        self.assertEqual(x["qualified_occluded_hit_negatives"],1)
        self.assertEqual(x["abstain_hit_count"],0)
        self.assertEqual(x["rows"][0]["targets"][0]["visibility_label"],"POSITIVE_VISIBLE_CONTRIBUTOR")
        self.assertEqual(x["rows"][0]["targets"][1]["visibility_label"],"NEGATIVE_OPAQUE_OCCLUDED_CONTRIBUTOR")
        self.assertEqual(x["rows"][0]["same_drawable"],"UNKNOWN")
        self.assertEqual(x["qualified_different_drawable_negatives"],0)
        self.assertFalse(x["training_authorized"])

    def test_nearly_opaque_not_forged_exact_occluder(self):
        x=fixture()
        row=x["rows"][0]
        row["standalone_rgba"][3]=.999
        # Rebuild recorded V1 independent front-over reconstruction exactly.
        a=.999
        full=row["full_soup_rgba"];background=row["removal_rgba"]
        reconstructed=[a*full[i]+(1-a)*background[i] for i in range(3)]
        row["front_face_reconstruction_error"]=max(abs(full[i]-reconstructed[i]) for i in range(3))
        o=qualify(x)
        self.assertEqual(o["qualified_occluded_hit_negatives"],0)
        self.assertEqual(o["abstain_hit_count"],1)

    def test_conditional_b_not_proven_abstains(self):
        x=fixture()
        row=x["rows"][0]
        row["generic_counterfactual_v2"]["rgba"]["recolor_b_given_remove_a"]=row["generic_counterfactual_v2"]["rgba"]["remove_a"].copy()
        o=qualify(x)
        self.assertEqual(o["qualified_occluded_hit_negatives"],0)
        self.assertEqual(o["abstain_hit_count"],1)

    def test_v2_missing_arm_rejected(self):
        x=fixture()
        del x["rows"][0]["generic_counterfactual_v2"]["rgba"]["recolor_b_given_remove_a"]
        with self.assertRaisesRegex(ValueError,"MISSING_OR_EXTRA_ARM"):
            qualify(x)

    def test_forged_source_sha_rejected(self):
        x=fixture()
        x["rows"][0]["generic_counterfactual_v2"]["source_sha256"]="e"*64
        with self.assertRaisesRegex(ValueError,"PARENT_SOURCE_CAMERA_LOCUS_DRIFT"):
            qualify(x)

    def test_bad_alpha_recolor_rejected(self):
        x=fixture()
        x["rows"][0]["generic_counterfactual_v2"]["rgba"]["recolor_b_given_remove_a"]=[.5,.2,.2,.2]
        with self.assertRaisesRegex(ValueError,"RECOLOR_CHANGED_ALPHA"):
            qualify(x)

    def test_false_alpha_reconstruction_rejected(self):
        x=fixture()
        x["rows"][0]["front_face_reconstruction_error"]=.4
        with self.assertRaisesRegex(ValueError,"RECONSTRUCTION_METRIC_DRIFT"):
            qualify(x)

    def test_fake_training_authority_rejected(self):
        x=fixture()
        x["authority"]["training_authorized"]=True
        with self.assertRaisesRegex(ValueError,"FORGED_AUTHORITY"):
            qualify(x)

    def test_opaque_front_back_direct_response_contradiction_refused(self):
        x=fixture()
        x["rows"][0]["generic_counterfactual_v2"]["rgba"]["recolor_b"]=[.7,.3,.2,1]
        with self.assertRaisesRegex(ValueError,"OPAQUE_FRONT_BACK_RESPONSE_CONTRADICTION"):
            qualify(x)

if __name__=="__main__":
    unittest.main()
