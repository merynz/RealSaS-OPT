"""Adversarial all-eight-view geometric grouping; only *edit response* held out.

No author semantic labels are inferred. An included heldout camera's raw
geometry is not held out, unlike v2. The rendered edit consequences are.
"""
import copy
import numpy as np
import pytest

from tools.corpus.compile_corpus370_allview_geometry_groups_v3 import (
    allview_geometry_links,build,
)
from tools.corpus.court_corpus370_allview_geometry_groups_v3 import court


def fixtures(n=80):
    vertices=np.zeros((3*n,3),dtype=np.float64)
    for i in range(n):
        x=(i%10)*2.;y=(i//10)*2.
        vertices[i*3:i*3+3]=[(x,y,0.),(x+.3,y,0.),(x,y+.3,0.)]
    faces=np.arange(n*3,dtype=np.int32).reshape(n,3)
    posed=np.stack([vertices,vertices.copy(),vertices.copy()])
    posed[1,::3,2]+=.05
    def pair(view,a,b):
        return {"view_index":view,
                "source_face_a_teacher_only":a,
                "source_face_b_teacher_only":b,
                "source_hit_visibility_owner":"SOURCE_LOCUS_ONLY",
                "same_drawable":"UNKNOWN","different_drawable":"UNKNOWN",
                "training_authorized":False}
    rendered={
        "schema":"RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1",
        "parents":{"source_sha256":"a"*64},
        "rows":[pair(0,0,n-1),pair(4,14,n-14)],
        "qualified_same_drawable_positives":0,
        "qualified_different_drawable_negatives":0,
        "training_authorized":False,
    }
    previous={
        "schema":"RealSaS.Corpus370AnonymousLayerProposal.v1",
        "parents":{"source_sha256":"a"*64,
                   "canonical_arrays_sha256":"b"*64,
                   "source_pair_receipt_sha256":"c"*64},
        "training_authorized":False,
        "source_semantic_or_material_names_used":False,
        "graphics_owner_minted":False,
        "face_count":n,
        "candidates":[{"candidate":"MANIFOLD_COMPONENTS",
                       "slot_count":n,
                       "face_slot_teacher_only":list(range(n))}],
    }
    previous_court={
        "schema":"RealSaS.Corpus370AnonymousLayerEditProxyCourt.v1",
        "selected_candidate_train_only":"MANIFOLD_COMPONENTS",
        "selected_slot_count":n,
        "training_authorized":False,
        "graphics_family_authorized":False,
    }
    rays=[]
    for view in range(8):
        for p in range(8):
            a=(view*7+p)%n
            b=(a+n//2)%n
            rays.append({
                "view_index":view,"pixel_x":p*4,"pixel_y":25,
                "status":"VALID_GEOMETRIC_FULL_STACK_CANDIDATE",
                "complete":True,"has_depth_tie":False,
                "hits":[{"face_index_teacher_only":a},
                        {"face_index_teacher_only":b}],
                "drawable_ownership_label_minted":False,
                "source_alpha_compositing_qualified":False,
            })
    geo={
        "schema":"RealSaS.Corpus370V5SourceLocusGeometricRayCandidate.v1",
        "status":"SOURCE_GEOMETRIC_RAY_CANDIDATES_MEASURED__RENDERED_ALPHA_AND_OWNER_COURTS_REQUIRED",
        "source_sha256":"a"*64,
        "authority":{"graphics_family_minted":False,
                     "training_authorized":False},
        "counts":{"proposed_pixels":64},
        "rays":rays,
    }
    return vertices,faces,posed,previous,previous_court,geo,rendered


def test_all_camera_geometry_included_and_independent_edit_outcome_not_in_producer():
    v,f,p,original,previous,geo,rendered=fixtures()
    proposal=build(v,f,p,original,previous,geo)
    decision=court(proposal,geo,rendered,original)
    assert proposal["source_geometry_views_used_for_compilation"]==list(range(8))
    assert proposal["view4_geometry_seen_by_compiler"] is True
    assert proposal["view4_rendered_edit_consequences_used_by_compiler"] is False
    assert decision["view4_not_unseen_geometry"] is True
    assert decision["not_view_holdout_generalization"] is True
    assert decision["training_authorized"] is False
    assert decision["same_drawable_positive_labels"]==0
    assert decision["different_drawable_negative_labels"]==0


def test_opposite_edit_court_mutation_does_not_change_compiler_or_winner():
    v,f,p,original,previous,geo,rendered=fixtures()
    x=build(v,f,p,original,previous,geo)
    a=court(x,geo,rendered,original)
    changed=copy.deepcopy(rendered)
    changed["rows"][1]["source_face_a_teacher_only"]=0
    changed["rows"][1]["source_face_b_teacher_only"]=79
    y=build(v,f,p,original,previous,geo)
    b=court(y,geo,changed,original)
    assert x["candidates"]==y["candidates"]
    assert a["selected_candidate_train_only"]==b["selected_candidate_train_only"]


def test_view4_geometry_may_affect_hypotheses_because_not_unseen_view():
    v,f,p,original,previous,geo,rendered=fixtures()
    a=build(v,f,p,original,previous,geo)
    updated=copy.deepcopy(geo)
    for r in updated["rays"]:
        if r["view_index"]==4:
            r["hits"]=[{"face_index_teacher_only":5},
                       {"face_index_teacher_only":65}]
    b=build(v,f,p,original,previous,updated)
    assert (a["geometry_candidate_link_count"]!=b["geometry_candidate_link_count"]
            or a["candidates"]!=b["candidates"])


def test_bad_sample_count_and_owner_labels_cannot_pass():
    v,f,p,original,previous,geo,rendered=fixtures()
    geo["rays"].pop()
    with pytest.raises(ValueError,match="SOURCE_RAY_SHA_CENSUS"):
        build(v,f,p,original,previous,geo)
    geo=fixtures()[5]
    geo["rays"][0]["drawable_ownership_label_minted"]=True
    with pytest.raises(ValueError,match="FAKED_AUTHORITY"):
        build(v,f,p,original,previous,geo)


def test_invalid_geometry_source_face_indices_fail_closed():
    v,f,p,original,previous,geo,rendered=fixtures()
    geo["rays"][20]["hits"][1]["face_index_teacher_only"]=9000
    with pytest.raises(ValueError,match="INVALID_FACE"):
        build(v,f,p,original,previous,geo)


def test_forged_same_drawable_training_authority_rejected():
    v,f,p,original,previous,geo,rendered=fixtures()
    candidate=build(v,f,p,original,previous,geo)
    candidate["training_authorized"]=True
    with pytest.raises(ValueError,match="SOURCE_PROPOSAL_SCOPE"):
        court(candidate,geo,rendered,original)
    candidate["training_authorized"]=False
    rendered["qualified_different_drawable_negatives"]=1
    with pytest.raises(ValueError,match="PARENTS_AND_SOURCE"):
        court(candidate,geo,rendered,original)


def test_corrupted_partition_or_repaired_full_image_claim_rejected():
    v,f,p,original,previous,geo,rendered=fixtures()
    candidate=build(v,f,p,original,previous,geo)
    candidate["candidates"][0]["full_image_edit_render_courted"]=True
    with pytest.raises(ValueError,match="GROUP_PROPOSAL_INTEGRITY"):
        court(candidate,geo,rendered,original)
    candidate=build(v,f,p,original,previous,geo)
    candidate["candidates"][0]["face_slot_teacher_only"]=[0]*len(f)
    with pytest.raises(ValueError,match="GROUP_PROPOSAL_INTEGRITY"):
        court(candidate,geo,rendered,original)


def test_real_8view_rays_with_some_abstentions_not_automatic_negative():
    v,f,p,original,previous,geo,rendered=fixtures()
    geo["rays"][0].update(
        status="ABSTAIN_INCOMPLETE_OR_TIED_OR_NO_HITS",hits=[])
    p=build(v,f,p,original,previous,geo)
    result=court(p,geo,rendered,original)
    assert result["graphics_owner_family_authorized"] is False
