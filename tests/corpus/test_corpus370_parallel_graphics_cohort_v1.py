"""Parallel graphics: deterministic source-only selection, no outcome cherry-picking."""
from __future__ import annotations

import copy
import pytest

from tools.corpus.select_corpus370_parallel_graphics_cohort_v1 import (
    choose, EXCLUDED_PRIOR_THREE, SCHEMA,
)


def inputs():
    excluded=sorted(EXCLUDED_PRIOR_THREE)
    others=[f"subject_{i:04}" for i in range(367)]
    ids=excluded+others
    assert len(ids)==370
    v5ids=ids[:352]
    v5={
        "schema":"RealSaS.Corpus370PreparedObservationPackets.v5",
        "prepared_subject_count":352,"failure_count":18,
        "training_authorized":False,
        "subjects":[{"subject_id":sid,"source_sha256":f"{i+1:064x}"}
                    for i,sid in enumerate(v5ids)],
    }
    source={
        "schema":"RealSaS.Corpus370ExactSourceRoster.v1",
        "subject_count":370,"training_authorized":False,
        "subjects":[{"subject_id":sid,"source_sha256":f"{i+1:064x}",
                     "source_format":(".fbx" if i%2 else ".glb"),
                     "source_id":f"pack_{i%7}"}
                    for i,sid in enumerate(ids)],
    }
    rays={
        "schema":"RealSaS.Corpus370V5GeometricRayPilotIndex.v1",
        "status":"PILOT_SUBJECTS_ACCOUNTED__GRAPHICS_AUTHORITY_SEPARATE",
        "training_authorized":False,"graphics_owner_truth_minted":False,
        "subject_results":{
            sid:{"source_sha256":row["source_sha256"],
                 "status":"MEASURED_GEOMETRIC_RAY_CANDIDATE",
                 "counts":{"multihit_geometric_rays":2}}
            for row in v5["subjects"] for sid in [row["subject_id"]]
        },
    }
    return v5,source,rays


def test_eight_new_subjects_deterministic_stratified_without_owner_truth():
    v5,source,rays=inputs()
    selected,review=choose(v5,source,rays)
    assert len(selected["subject_ids"])==8
    assert not (set(selected["subject_ids"]) & EXCLUDED_PRIOR_THREE)
    assert selected["training_authorized"] is False
    assert review["schema"]==SCHEMA
    assert review["selection_uses_graphics_outcome"] is False
    assert review["training_authorized"] is False
    scrambled=copy.deepcopy(v5)
    scrambled["subjects"].reverse()
    selected_reversed,_=choose(scrambled,source,rays)
    assert selected_reversed["subject_ids"]==selected["subject_ids"]


def test_invalid_geometry_eligibility_is_excluded_not_false_negative():
    v5,source,rays=inputs()
    for sid in list(rays["subject_results"])[:340]:
        if sid not in EXCLUDED_PRIOR_THREE:
            rays["subject_results"][sid]["counts"]["multihit_geometric_rays"]=0
    chosen,report=choose(v5,source,rays)
    assert len(chosen["subject_ids"])==8
    assert report["geometry_candidate_census"]["insufficient_multihit"]>0
    assert report["same_drawable_labels_minted"] is False


def test_source_sha_drift_rejected():
    v5,source,rays=inputs()
    first=v5["subjects"][0]["subject_id"]
    rays["subject_results"][first]["source_sha256"]="f"*64
    with pytest.raises(ValueError,match="SOURCE_SHA_DRIFT"):
        choose(v5,source,rays)


def test_nonqualified_ray_not_promoted():
    v5,source,rays=inputs()
    sid="subject_0000"
    rays["subject_results"][sid]["status"]="FAIL_GEOMETRIC_RAY_MEASUREMENT"
    chosen,report=choose(v5,source,rays)
    assert sid not in chosen["subject_ids"]
    assert report["geometry_candidate_census"]["invalid_ray_status"]==1


def test_forbidden_new_training_scope():
    v5,source,rays=inputs()
    rays["training_authorized"]=True
    with pytest.raises(ValueError,match="RAY_SCOPE"):
        choose(v5,source,rays)


def test_insufficient_geometry_is_explicit_not_silent_replacement():
    v5,source,rays=inputs()
    for sid,row in rays["subject_results"].items():
        row["counts"]["multihit_geometric_rays"]=0
    with pytest.raises(ValueError,match="INSUFFICIENT_GEOMETRIC_CANDIDATES"):
        choose(v5,source,rays)
