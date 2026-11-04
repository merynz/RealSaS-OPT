"""Independent heldout view4 vs three-view geometry hypothesis grouping tests."""
import copy
import numpy as np
import pytest

from tools.corpus.compile_corpus370_multiview_geometry_groups_v2 import (
    build,geometry_constraints,refine_only_conflicts,
)
from tools.corpus.court_corpus370_multiview_geometry_groups_v2 import qualify
from tests.corpus.test_corpus370_hierarchical_layer_groups_v1 import fixtures


def rays(n=80):
    rows=[]
    pairs={0:(0,n-1),2:(10,n-10),6:(20,n-20)}
    for view in range(8):
        for px in range(8):
            if view in pairs:
                a,b=pairs[view]
                hits=[
                    {"face_index_teacher_only":a},
                    {"face_index_teacher_only":b},
                ]
            else:
                hits=[{"face_index_teacher_only":1}]
            rows.append({
                "view_index":view,"pixel_x":px,"pixel_y":7,
                "complete":True,"has_depth_tie":False,
                "status":"VALID_GEOMETRIC_FULL_STACK_CANDIDATE",
                "hits":hits,
                "drawable_ownership_label_minted":False,
                "source_alpha_compositing_qualified":False,
            })
    return {
        "schema":"RealSaS.Corpus370V5SourceLocusGeometricRayCandidate.v1",
        "status":"SOURCE_GEOMETRIC_RAY_CANDIDATES_MEASURED__RENDERED_ALPHA_AND_OWNER_COURTS_REQUIRED",
        "source_sha256":"a"*64,
        "authority":{"training_authorized":False},
        "counts":{"proposed_pixels":64},
        "rays":rows,
    }


def test_geometry_constraints_only_reads_0_2_6_never_view4():
    doc=rays()
    links,counts=geometry_constraints(doc,source_sha="a"*64,face_count=80)
    assert len(links)==3
    assert counts=={0:8,2:8,6:8}
    assert (0,79) in links and (10,70) in links
    doc["rays"][4*8]["hits"]=[{"face_index_teacher_only":40},
                               {"face_index_teacher_only":41}]
    unchanged,_=geometry_constraints(doc,source_sha="a"*64,face_count=80)
    assert unchanged==links


def test_original_patch_refined_only_when_geometric_candidate_requires():
    original=[0]*80
    refined,broken=refine_only_conflicts(original,[(0,79),(10,70)])
    assert len(broken)==2
    assert refined[0]!=refined[79]
    assert refined[10]!=refined[70]
    assert len(set(refined))==3


def test_multiview_real_shape_and_independent_render_proxy_not_semantic_truth():
    v,f,p,pairs,parent,base=fixtures()
    doc=rays()
    q=build(v,f,p,parent,base,doc)
    c=qualify(q,doc,pairs,parent)
    assert q["geometry_candidate_constraint_count"]==3
    assert q["heldout_render_outcomes_used_for_selection"] is False
    assert c["selected_group_count"] in (8,16,32,64,128)
    assert c["geometry_source_ray_constraint_count"]==3
    assert c["graphics_owner_family_authorized"] is False
    assert c["training_authorized"] is False
    assert c["real_full_frame_render_edit_qualified"] is False


def test_view4_render_evidence_never_changes_candidate_generation_or_winner():
    v,f,p,pairs,parent,base=fixtures()
    doc=rays()
    first=build(v,f,p,parent,base,doc)
    c1=qualify(first,doc,pairs,parent)
    modified=copy.deepcopy(pairs)
    modified["rows"][1]["source_face_a_teacher_only"]=0
    modified["rows"][1]["source_face_b_teacher_only"]=79
    second=build(v,f,p,parent,base,doc)
    c2=qualify(second,doc,modified,parent)
    assert first["candidates"]==second["candidates"]
    assert c1["selected_candidate_train_only"]==c2["selected_candidate_train_only"]


def test_training_source_ray_without_alpha_must_not_be_owner_negative_label():
    v,f,p,pairs,parent,base=fixtures()
    doc=rays()
    q=build(v,f,p,parent,base,doc)
    c=qualify(q,doc,pairs,parent)
    assert c["author_drawable_truth_qualified"] is False
    assert c["geometry_occlusion_is_operational_heuristic_only"] is True
    assert c["training_authorized"] is False


def test_forged_render_authority_or_source_sha_rejected():
    v,f,p,pairs,parent,base=fixtures()
    doc=rays()
    doc["source_sha256"]="f"*64
    with pytest.raises(ValueError,match="SOURCE_RAY_SCHEMA"):
        build(v,f,p,parent,base,doc)
    doc=rays()
    doc["authority"]["training_authorized"]=True
    with pytest.raises(ValueError,match="SOURCE_RAY_SCHEMA"):
        build(v,f,p,parent,base,doc)


def test_proposal_source_partitions_cannot_merge_original_geometric_constraints():
    v,f,p,pairs,parent,base=fixtures()
    doc=rays()
    q=build(v,f,p,parent,base,doc)
    q["candidates"][0]["face_slot_teacher_only"]=[0]*80
    q["candidates"][0]["slot_count"]=1
    with pytest.raises(ValueError,match="PARTITION_FORGERY|SOURCE_GEOMETRY_VIOLATED"):
        qualify(q,doc,pairs,parent)
