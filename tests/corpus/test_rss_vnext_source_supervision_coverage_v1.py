from __future__ import annotations

from hashlib import sha256
from copy import deepcopy

import pytest

from tools.corpus.rss_vnext_source_supervision_coverage_v1 import (
    SCHEMA, audit_rss_vnext_coverage_v1,
)


def h(value):
    return sha256(value.encode("utf8")).hexdigest()


def row(i,split="TRAIN",license_id="CC0-1.0",teacher_families=(), *,
        group=None,noai=False,attribution=None):
    return {
        "schema":SCHEMA,
        "asset_sha256":h(f"asset-{i}"),
        "source_group_sha256":h(group or f"source-group-{i}"),
        "split":split,
        "original_source_rgba_sha256":[h(f"front-{i}"),h(f"back-{i}")],
        "license":{
            "spdx_id":license_id,
            "explicit_noai":noai,
            "license_evidence_sha256":h(f"license-evidence-{i}"),
            "attribution_sha256":attribution,
        },
        "qualified_teacher_receipts":{
            name:h(f"external-receipt-{i}-{name}") for name in teacher_families
        },
    }


def test_coverage_inventory_counts_only_licensed_explicit_qualified_families():
    trained=row(
        1,teacher_families=(
            "SOURCE_RENDER_RGB_ALPHA","SOURCE_RENDER_SEMANTIC_LAYER",
            "SOURCE_SDF_SIGNED","SOURCE_TESSA_TOPOLOGY",
            "SOURCE_RIG_JOINTS","SOURCE_MIRA_DENSE_SKIN",
            "SOURCE_POSED_LBS",
        ),
    )
    unseen=row(2,split="UNSEEN",teacher_families=("SOURCE_TESSA_TOPOLOGY",))
    noai=row(3,noai=True,teacher_families=("SOURCE_SDF_SIGNED",))
    noncommercial=row(4,license_id="CC-BY-NC-4.0",teacher_families=("SOURCE_SDF_SIGNED",))
    sa=row(5,license_id="CC-BY-SA-4.0",teacher_families=("SOURCE_SDF_SIGNED",))
    by=row(6,license_id="CC-BY-4.0",attribution=h("attribution"))
    report=audit_rss_vnext_coverage_v1(
        [trained,unseen,noai,noncommercial,sa,by],train_asset_target=2,
    )
    assert report["accepted_license_counts_by_split"]=={
        "TRAIN":2,"VAL":0,"UNSEEN":1,
    }
    assert report["rejected_license_counts_by_reason"]=={
        "EXPLICIT_NOAI":1,"LICENSE_NOT_IN_ALLOWED_SET":2,
    }
    assert report["declared_stage_candidate_counts_by_split"]["TRAIN"]["PRODUCT_POSED_CONSEQUENCE"]==1
    assert report["declared_stage_candidate_counts_by_split"]["TRAIN"]["IRIS_SEMANTIC_LAYER"]==1
    assert report["declared_stage_candidate_counts_by_split"]["TRAIN"]["GRAPHICS_SOURCE_RGB"]==0
    assert report["train_asset_target_met"]
    assert report["unseen_split_nonempty"]
    assert not report["training_authorized"]
    assert not report["unseen_product_quality_proven"]


def test_source_group_leakage_and_asset_duplicates_fail_closed():
    left=row(1,group="same-source-group")
    right=row(2,split="UNSEEN",group="same-source-group")
    with pytest.raises(ValueError,match="SOURCE_GROUP_SPLIT_LEAKAGE"):
        audit_rss_vnext_coverage_v1([left,right])
    with pytest.raises(ValueError,match="DUPLICATE_ASSET_SHA"):
        audit_rss_vnext_coverage_v1([left,left])


def test_teacher_receipts_cannot_be_empty_or_unknown_and_two_views_required():
    example=row(1)
    invalid=deepcopy(example)
    invalid["original_source_rgba_sha256"]=[h("front-1")]
    with pytest.raises(ValueError,match="TWO_DISTINCT_ORIGINAL_IMAGES"):
        audit_rss_vnext_coverage_v1([invalid])
    invalid=deepcopy(example)
    invalid["qualified_teacher_receipts"]={"SOURCE_SDF_SIGNED":""}
    with pytest.raises(ValueError,match="TEACHER_RECEIPT_DOMAIN"):
        audit_rss_vnext_coverage_v1([invalid])
    invalid=deepcopy(example)
    invalid["qualified_teacher_receipts"]={"UNKNOWN_PSEUDO_TRUTH":h("fake")}
    with pytest.raises(ValueError,match="TEACHER_RECEIPT_DOMAIN"):
        audit_rss_vnext_coverage_v1([invalid])


def test_cc_by_requires_attribution_and_is_not_cc_by_noncommercial():
    no_attribution=row(1,license_id="CC-BY-4.0")
    noncommercial=row(2,license_id="CC-BY-NC-4.0",attribution=h("attribution"))
    allowed=row(3,license_id="CC-BY-4.0",attribution=h("attribution"))
    report=audit_rss_vnext_coverage_v1([no_attribution,noncommercial,allowed],train_asset_target=3)
    assert report["accepted_license_counts_by_split"]["TRAIN"]==1
    assert report["rejected_license_counts_by_reason"]=={
        "MISSING_CC_BY_ATTRIBUTION_RECEIPT":1,
        "LICENSE_NOT_IN_ALLOWED_SET":1,
    }
    assert not report["train_asset_target_met"]


def test_inventory_digest_is_order_independent():
    a=row(1)
    b=row(2,split="VAL")
    x=audit_rss_vnext_coverage_v1([a,b],train_asset_target=1)
    y=audit_rss_vnext_coverage_v1([b,a],train_asset_target=1)
    assert x["inventory_sha256"]==y["inventory_sha256"]
    assert x["declared_teacher_receipt_counts_by_split"]==y["declared_teacher_receipt_counts_by_split"]
