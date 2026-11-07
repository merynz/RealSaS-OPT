from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image, ImageDraw

from tools.corpus.audit_corpus4008_2d_art_orbit_v1 import audit
from tools.corpus.finish_corpus4008_2d_art_orbit_v1 import run


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene(tmp_path, *, nviews=8, alpha_offset=0):
    rows=[]
    for i in range(nviews):
        for name,rgb in (
            ("source_material_diagnostic", (151,119,85,255)),
            ("cel_3band",(174,125,60,255)),
        ):
            im=Image.new("RGBA",(256,256),(0,0,0,0))
            ImageDraw.Draw(im).ellipse((35+alpha_offset,24,221,240),fill=rgb)
            p=tmp_path/(f"{name}_orbit_{i:02d}.png")
            im.save(p)
        ref=tmp_path/(f"source_material_diagnostic_orbit_{i:02d}.png")
        cel=tmp_path/(f"cel_3band_orbit_{i:02d}.png")
        matrix=[[1.,0.,0.,0.],[0.,1.,0.,0.],
                [0.,0.,1.,float(i)],[0.,0.,0.,1.]]
        rows.append({
            "view_id":f"SOURCE_ORBIT_{i:02d}",
            "source_space_orbit_degrees":45*i,
            "camera_source_space_matrix":matrix,
            "source_reference_png":ref.name,
            "source_reference_sha256":sha(ref),
            "cel_candidate_png":cel.name,
            "cel_candidate_sha256":sha(cel),
            "inked_2d_candidate_png":f"inked_2d_orbit_{i:02d}.png",
            "inked_2d_candidate_sha256":None,
            "inked_2d_alpha_identical_to_cel":None,
            "view_2d_art_truth_admitted":False,
        })
    manifest={
        "schema":"RealSaS.Graphics2DArtOrbitCandidate.v1",
        "status":"EIGHT_SOURCE_SPACE_SOURCE_AND_CEL_READY_AWAITING_INK",
        "asset_id":"character",
        "source_sha256":"a"*64,
        "view_count":nviews,
        "views":rows,
        "style_origin":"DETERMINISTIC_NON_GENERATIVE_RENDER",
        "original_source_rgb_authority_qualified":False,
        "graphics_style_2d_truth_qualified":False,
        "training_authorized":False,
    }
    (tmp_path/"GRAPHICS_2D_ART_ORBIT_CANDIDATE_V1.json").write_text(
        json.dumps(manifest))
    return manifest


def test_eight_source_matched_style_candidates_are_observation_only(tmp_path):
    scene(tmp_path)
    finished=run(tmp_path)
    assert finished["inked_png_count"]==8
    r=audit(tmp_path,enforce=True)
    assert r["numerically_suitable_stylized_2d_candidate"]
    assert r["training_authorized"] is False
    assert r["2d_art_human_visual_quality_status"]=="UNKNOWN"
    assert len(r["per_view"])==8


def test_shuffled_or_missing_view_is_not_a_valid_orbit(tmp_path):
    scene(tmp_path,nviews=7)
    with pytest.raises(ValueError,match="VIEW_COUNT"):
        run(tmp_path)


def test_art_style_alpha_cannot_invent_body_geometry(tmp_path):
    scene(tmp_path)
    run(tmp_path)
    target=tmp_path/"inked_2d_orbit_01.png"
    im=Image.open(target).convert("RGBA")
    im.putpixel((0,0),(255,0,0,255))
    im.save(target)
    # Cryptographic hash before any vision metric.
    with pytest.raises(ValueError,match="PIXEL_HASH"):
        audit(tmp_path)


def test_source_changed_during_render_pipeline_fails(tmp_path):
    scene(tmp_path)
    p=tmp_path/"source_material_diagnostic_orbit_03.png"
    p.write_bytes(b"not original")
    with pytest.raises(ValueError,match="SOURCE_PNG_SHA"):
        run(tmp_path)


def test_cannot_spoof_training_pass_even_if_all_pixels_are_valid(tmp_path):
    scene(tmp_path)
    run(tmp_path)
    p=tmp_path/"GRAPHICS_2D_ART_ORBIT_CANDIDATE_V1.json"
    d=json.loads(p.read_text())
    d["training_authorized"]=True
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="AUTHORITY_FORGERY"):
        audit(tmp_path)


def test_bad_view_matrices_fail_even_if_images_look_good(tmp_path):
    scene(tmp_path)
    run(tmp_path)
    p=tmp_path/"GRAPHICS_2D_ART_ORBIT_CANDIDATE_V1.json"
    d=json.loads(p.read_text())
    d["views"][3]["camera_source_space_matrix"]=d["views"][2]["camera_source_space_matrix"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="DUPLICATE_CAMERA"):
        audit(tmp_path)
