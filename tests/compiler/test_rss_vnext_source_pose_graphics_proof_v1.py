from __future__ import annotations

"""A source frame court can inform ProductProof, never mint ProductPASS by CAS.

The fixture is not real licensed source truth. The proof worker cannot elevate
a self-contained hash-sealed JSON because its independently qualified source
and renderer registries are not replayed by that consumer.
"""

from dataclasses import replace
import json

import pytest

from compiler.realsas_compiler_core.rss_vnext_source_pose_graphics_court_v1 import (
    evaluate_registered_source_pose_graphics_v1,
    source_pose_graphics_court_hash_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_proof_capability_v1,
)
from tests.compiler.test_rss_vnext_source_pose_graphics_court_v1 import _fixture
from tests.compiler.test_rss_vnext_raw_motion_proof_replay_v1 import (
    _complete_product_preflight,
)


def _ref(cas,request,kind):
    return next(r for r in request["input_artifacts"]
                if r["artifact_type"]==kind)


def _append_court(cas,request,*,tamper=None):
    proof=evaluate_registered_source_pose_graphics_v1(**_fixture())
    mesh=json.loads(cas.read_object(_ref(cas,request,"RealSaS.CanonicalMeshIR")))
    rig=json.loads(cas.read_object(_ref(cas,request,"RealSaS.CanonicalRigIR")))
    skin=json.loads(cas.read_object(_ref(cas,request,"RealSaS.CanonicalSkinIR")))
    graphics=json.loads(cas.read_object(_ref(cas,request,"RealSaS.CompiledGraphics")))
    motion=json.loads(cas.read_object(_ref(cas,request,"RealSaS.CanonicalMotionIR")))
    # A deliberately self-attested local CAS-only claim with rehashed actual
    # parents must not become independent source or ProductProof authority.
    proof=replace(
        proof,
        canonical_mesh_lineage_hash=mesh["mesh_lineage_hash"],
        canonical_rig_lineage_hash=rig["rig_lineage_hash"],
        canonical_skin_lineage_hash=skin["skin_lineage_hash"],
        compiled_graphics_hash=graphics["graph_hash"],
        canonical_motion_lineage_hash=motion["motion_lineage_hash"],
        court_hash="",
    )
    if tamper:
        proof=replace(proof,**tamper)
    proof=replace(proof,court_hash=source_pose_graphics_court_hash_v1(proof))
    blob=(json.dumps(proof.to_dict(),sort_keys=True,separators=(",",":"))+"\n").encode()
    row={
        **cas.put_bytes(blob),
        "role":"source_pose_graphics_court",
        "artifact_type":"RealSaS.RSSvNextSourcePoseGraphicsCourt",
        "schema_version":"v1",
        "authority_class":"COMPILER_CANDIDATE_NOT_PRODUCT_REVISION",
    }
    request["input_artifacts"].append(row)
    return row


def _run(cas,request):
    return execute_proof_capability_v1(
        request,read_object=cas.read_object,put_bytes=cas.put_bytes,
    )


def test_product_proof_reopens_measured_frame_but_does_not_self_promote(tmp_path):
    cas,request,_,_=_complete_product_preflight(tmp_path)
    _append_court(cas,request)
    result=_run(cas,request)
    assert result["status"]=="FAIL"
    d=result["diagnostics"]
    row=d["source_pose_graphics_courts"]
    assert row["typed_source_pose_graphics_frame_receipts_reopened"]==1
    assert row["source_pose_graphics_local_positive_claim_count"]==1
    assert row["registered_source_rest_views_claimed"]==[0]
    assert row["registered_artist_motion_views_claimed"]==[]
    assert row["independent_native_and_source_pixel_replay_in_proof"] is False
    assert row["source_graphics_product_authority_promoted"] is False
    assert "SOURCE_APPEARANCE_AND_POSE_VISIBILITY" in d["missing_independent_witnesses"]
    assert d["product_pass_authority"] is False
    assert d["product_revision_minted"] is False


@pytest.mark.parametrize("field,value",[
    ("hidden_surface_color_proven",True),
    ("product_release_admission",True),
    ("compiled_graphics_hash","9"*64),
])
def test_product_proof_rejects_rehashed_false_authority_or_parent(tmp_path,field,value):
    cas,request,_,_=_complete_product_preflight(tmp_path)
    _append_court(cas,request,tamper={field:value})
    with pytest.raises(QualificationError,match="REQUALIFICATION_DRIFT"):
        _run(cas,request)


def test_product_proof_refuses_duplicate_or_wrong_schema_graphics_witness(tmp_path):
    cas,request,_,_=_complete_product_preflight(tmp_path)
    row=_append_court(cas,request)
    request["input_artifacts"].append(row)
    with pytest.raises(RuntimeError,match="DUPLICATE_FRAME"):
        _run(cas,request)
    request["input_artifacts"].pop()
    request["input_artifacts"][-1]={**row,"schema_version":"v9"}
    with pytest.raises(RuntimeError,match="CARDINALITY_OR_SCHEMA"):
        _run(cas,request)


def test_product_proof_without_source_pose_court_reports_zero(tmp_path):
    cas,request,_,_=_complete_product_preflight(tmp_path)
    d=_run(cas,request)["diagnostics"]["source_pose_graphics_courts"]
    assert d["typed_source_pose_graphics_frame_receipts_reopened"]==0
    assert d["source_pose_graphics_local_positive_claim_count"]==0
    assert d["independent_native_and_source_pixel_replay_in_proof"] is False
