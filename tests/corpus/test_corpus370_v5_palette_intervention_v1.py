from __future__ import annotations

import copy
import math

import pytest

from tools.corpus.validate_corpus370_v5_palette_intervention_v1 import court


def fixture():
    # Source FULL is exact front-face-over-REMOVAL alpha composition.
    # Standalone straight RGB=(0.4,0.1,0), alpha=0.6.
    # Removal straight RGB=(0.9,0.5,0.5), alpha=1.
    # FULL RGB=(0.6,0.26,0.2), alpha=1.
    full=[0.6,0.26,0.2,1.0]
    split=[0.8,0.1,0.1,1.]
    collision=[0.9,0.15,0.15,1.]
    row={
        "view_index":1,"pixel_x":23,"pixel_y":41,
        "source_faces_teacher_only":[3,8],
        "source_barycentric_teacher_only":[[0.2,0.3,0.5],[0.1,0.1,0.8]],
        "source_material_domains_teacher_only":["mesh0:2","mesh1:2"],
        "material_semantic_name_used_as_label":False,
        "same_drawable_label_minted":False,
        "original_rgba":full.copy(),
        "full_soup_rgba":full.copy(),
        "source_reimport_vs_full_soup_max_error":0.,
        "standalone_rgba":[.4,.1,0.,.6],
        "removal_rgba":[.9,.5,.5,1.],
        "front_face_reconstruction_error":0.,
        "palette_distinct_rgba":split,
        "palette_collision_rgba":collision,
        "palette_alpha_max_delta":0.,
        "palette_rgb_difference_norm":math.sqrt(.1**2+.05**2+.05**2),
        "different_source_domains":True,
        "split_and_collision_are_color_only":True,
        "status":"PASS_SOURCE_RENDER_INTERVENTION_CANDIDATE__DRAWABLE_OWNER_SEPARATE",
    }
    return {
        "schema":"RealSaS.Corpus370SourceLocusPaletteIntervention.v1",
        "status":"MEASURED_SOURCE_PALETTE_PILOT__NO_GRAPHICS_FAMILY_AUTHORITY",
        "subject_id":"fixture","source_sha256":"b"*64,
        "parents":{"schema":"RealSaS.Corpus370SourceLocusPaletteJob.v1",
                   "subject_id":"fixture","source":{"sha256":"b"*64},
                   "training_authorized":False},
        "carrier_mode":"STRAIGHT_LINEAR_RGB",
        "renderer":"BLENDER_CYCLES_EXR_PREMULTIPLIED_CENTRE_RAY",
        "rows":[row],"qualified_intervention_rows":1,
        "abstain_intervention_rows":0,
        "authority":{
            "material_and_owner_names_teacher_only":True,
            "source_material_alpha_unchanged":True,
            "student_semantic_owner_labels_minted":False,
            "graphics_family_minted":False,
            "training_authorized":False,
        },
    }


def test_valid_measured_palette_stays_candidate():
    r=court(fixture())
    assert r["measured_valid_palette_rows"]==1
    assert r["measured_source_domain_pair_controls"]==1
    assert r["family_authority_minted"] is False
    assert r["student_drawable_ownership_qualified"] is False
    assert r["training_authorized"] is False


def test_false_semantic_owner_label_rejected():
    data=fixture()
    data["rows"][0]["same_drawable_label_minted"]=True
    with pytest.raises(ValueError,match="OWNER_LEAKAGE"):
        court(data)


def test_false_graphics_authority_rejected():
    data=fixture()
    data["authority"]["graphics_family_minted"]=True
    with pytest.raises(ValueError,match="FORGED_AUTHORITY"):
        court(data)


def test_source_sha_relabel_rejected():
    data=fixture()
    data["source_sha256"]="c"*64
    with pytest.raises(ValueError,match="SUBJECT_SOURCE_ID"):
        court(data)


def test_bad_barycentric_rejected():
    data=fixture()
    data["rows"][0]["source_barycentric_teacher_only"][0]=[1.,1.,1.]
    with pytest.raises(ValueError,match="BARYCENTRIC"):
        court(data)


def test_reconstruction_drift_rejected():
    data=fixture()
    data["rows"][0]["full_soup_rgba"][0]=0.1
    # First bypass parity metric tampering: failure stays in independent court.
    with pytest.raises(ValueError,match="FULL_PARITY_METRIC_DRIFT"):
        court(data)


def test_false_bounded_alpha_claim_rejected():
    data=fixture()
    data["rows"][0]["palette_distinct_rgba"][3]=0.4
    with pytest.raises(ValueError,match="ALPHA_METRIC_DRIFT"):
        court(data)


def test_missing_control_rejected():
    data=fixture()
    data["rows"][0]["split_and_collision_are_color_only"]=False
    with pytest.raises(ValueError,match="SPLIT_PALETTE_CONTROL"):
        court(data)


def test_new_measured_validity_can_be_abstain_not_fake_negative():
    data=fixture()
    row=data["rows"][0]
    # Incomplete reconstruction is an ABSTAIN; not valid ownership NEGATIVE.
    row["status"]="ABSTAIN_ALPHA_COMPOSITING_RECONSTRUCTION"
    row["front_face_reconstruction_error"]=0.10
    row["standalone_rgba"]=[0.,0.,0.,0.]
    data["qualified_intervention_rows"]=0
    data["abstain_intervention_rows"]=1
    # The independent court catches measured drift instead of accepting
    # a reason label unsupported by the computed RGB/alpha.
    with pytest.raises(ValueError,match="RECONSTRUCTION_METRIC_DRIFT"):
        court(data)
