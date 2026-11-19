from __future__ import annotations

"""ProductProof replays independently pinned raw pixel evidence, not self-CAS.

Test doubles for trusted source/native registry are NOT product source truth.
Even independent one-frame replay never clears full source appearance proof.
"""

import json
from hashlib import sha256

import pytest

from compiler.realsas_compiler_core.rss_vnext_source_pose_graphics_court_v1 import (
    evaluate_registered_source_pose_graphics_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_proof_capability_v1,
)
from compiler.rss_vnext.workers.rss_vnext_source_pose_graphics_evidence_replay_v1 import (
    SOURCE_BUNDLE_SCHEMA,NATIVE_BUNDLE_SCHEMA,
    independently_replay_registered_source_pose_graphics_v1,
)
from tests.compiler.test_rss_vnext_source_pose_graphics_court_v1 import _fixture
from tests.compiler.test_rss_vnext_source_pose_graphics_evidence_replay_v1 import _npz
from tests.compiler.test_rss_vnext_raw_motion_proof_replay_v1 import (
    _complete_product_preflight,
)


def _record(cas,request):
    def artifact(kind):
        ref=next(r for r in request["input_artifacts"]
                 if r["artifact_type"]==kind)
        return json.loads(cas.read_object(ref))
    mesh=artifact("RealSaS.CanonicalMeshIR")
    rig=artifact("RealSaS.CanonicalRigIR")
    skin=artifact("RealSaS.CanonicalSkinIR")
    graphics=artifact("RealSaS.CompiledGraphics")
    motion=artifact("RealSaS.CanonicalMotionIR")
    raw=_fixture()
    parents={
        "canonical_mesh_lineage_hash":mesh["mesh_lineage_hash"],
        "canonical_rig_lineage_hash":rig["rig_lineage_hash"],
        "canonical_skin_lineage_hash":skin["skin_lineage_hash"],
        "compiled_graphics_hash":graphics["graph_hash"],
        "canonical_motion_lineage_hash":motion["motion_lineage_hash"],
    }
    # This test's independent witness issuer is explicitly fabricated.
    # Its authority comes from the host fixture, never MODULE metadata.
    for key,val in parents.items():
        raw[key]=val
        raw["trusted_source_registry"][
            raw["source_witness_key"]
        ][key]=val
        raw["trusted_native_renderer_registry"][
            raw["renderer_witness_key"]
        ][key]=val
    raw["live_pose_receipt"].update({
        "mesh_lineage_hash":parents["canonical_mesh_lineage_hash"],
        "rig_lineage_hash":parents["canonical_rig_lineage_hash"],
        "skin_lineage_hash":parents["canonical_skin_lineage_hash"],
    })
    claim=evaluate_registered_source_pose_graphics_v1(**raw)
    source=_npz(
        target_rgba_u8=raw["target_rgba_u8"],
        target_face_owner_i32=raw["target_face_owner_i32"],
        target_known_mask=raw["target_known_mask"],
    )
    native=_npz(
        native_rgba_u8=raw["native_rgba_u8"],
        native_front_face_i32=raw["native_front_face_i32"],
        native_color_authority_u8=raw["native_color_authority_u8"],
        native_unknown_occlusion_u8=raw["native_unknown_occlusion_u8"],
    )
    sr=raw["trusted_source_registry"][raw["source_witness_key"]]
    nr=raw["trusted_native_renderer_registry"][raw["renderer_witness_key"]]
    sr.update({
        "raster_bundle_schema":SOURCE_BUNDLE_SCHEMA,
        "raster_bundle_sha256":sha256(source).hexdigest(),
        "source_raster_extraction_independently_verified":True,
        "raster_artifact_not_student_prediction":True,
    })
    nr.update({
        "raster_bundle_schema":NATIVE_BUNDLE_SCHEMA,
        "raster_bundle_sha256":sha256(native).hexdigest(),
        "cpp_native_raster_execution_independently_verified":True,
        "raster_artifact_not_student_prediction":True,
    })
    raw_court=(json.dumps(
        claim.to_dict(),sort_keys=True,separators=(",",":"),
    )+"\n").encode()
    request["input_artifacts"].append({
        **cas.put_bytes(raw_court),
        "role":"source_pose_graphics_frame_court",
        "artifact_type":"RealSaS.RSSvNextSourcePoseGraphicsCourt",
        "schema_version":"v1",
    })
    replay_kwargs={
        "source_raster_bundle":source,
        "native_raster_bundle":native,
        "native_face_count":raw["native_face_count"],
        "live_pose_receipt":raw["live_pose_receipt"],
        "source_witness_key":raw["source_witness_key"],
        "renderer_witness_key":raw["renderer_witness_key"],
        "source_view_index":raw["source_view_index"],
        "frame_index":raw["frame_index"],
        "source_target_kind":raw["source_target_kind"],
        "original_source_asset_sha256":raw["original_source_asset_sha256"],
        "original_source_file_sha256":raw["original_source_file_sha256"],
        "source_registration_sha256":raw["source_registration_sha256"],
        "native_renderer_binary_sha256":raw["native_renderer_binary_sha256"],
        "frozen_thresholds":raw["thresholds"],
        "trusted_source_registry":raw["trusted_source_registry"],
        "trusted_native_renderer_registry":raw["trusted_native_renderer_registry"],
        **parents,
    }
    def trusted_host_replay(measured):
        return independently_replay_registered_source_pose_graphics_v1(
            **replay_kwargs,claimed_court=measured.to_dict(),
        )
    return trusted_host_replay, raw, replay_kwargs


def _proof(cas,request,**params):
    return execute_proof_capability_v1(
        request,read_object=cas.read_object,
        put_bytes=cas.put_bytes,**params,
    )


def test_product_proof_trusted_raw_replay_counts_known_source_pixels_not_product_pass(
    tmp_path,
):
    cas,request,_,_=_complete_product_preflight(tmp_path)
    replayer,_,_=_record(cas,request)
    d=_proof(cas,request,trusted_source_native_pixel_replayer=replayer)[
        "diagnostics"
    ]
    assert d["source_visibility_independent_witness_present"] is True
    row=d["source_pose_graphics_courts"]
    assert row["typed_source_pose_graphics_frame_receipts_reopened"]==1
    assert row["independent_native_and_source_pixel_replay_in_proof"] is True
    assert row["independently_raw_replayed_frame_count"]==1
    assert row["independently_raw_positive_frame_count"]==1
    assert row["source_graphics_product_authority_promoted"] is False
    assert "SOURCE_APPEARANCE_AND_POSE_VISIBILITY" in d[
        "missing_independent_witnesses"
    ]
    assert d["product_pass_authority"] is False
    assert d["product_revision_minted"] is False


def test_product_proof_refuses_missing_or_forged_raw_source_bytes(tmp_path):
    cas,request,_,_=_complete_product_preflight(tmp_path)
    replayer,raw,args=_record(cas,request)
    result=_proof(cas,request)
    assert result["diagnostics"]["source_visibility_independent_witness_present"] is False
    # Raw source bytes changed with a valid unrelated CAS/hash would not
    # match the out-of-band independently trusted source registrar.
    source=args["source_raster_bundle"]+b"new-disallowed-source-bytes"
    def tampered(measured):
        return independently_replay_registered_source_pose_graphics_v1(
            **{**args,"source_raster_bundle":source},
            claimed_court=measured.to_dict(),
        )
    with pytest.raises(QualificationError,match="RAW_INDEPENDENT_RASTER_MISSING"):
        _proof(cas,request,trusted_source_native_pixel_replayer=tampered)
    # A callback is never imported or evaluated from untrusted MODULE metadata.
    request["capability_metadata"]["trusted_source_native_pixel_replayer"]=True
    with pytest.raises(RuntimeError,match="TRUSTED_GRAPHICS_REPLAYER_HOST_ONLY"):
        _proof(cas,request,trusted_source_native_pixel_replayer=replayer)


def test_product_proof_rejects_callback_that_only_forges_court_object(tmp_path):
    cas,request,_,_=_complete_product_preflight(tmp_path)
    _,_,_=_record(cas,request)
    with pytest.raises(RuntimeError,match="RAW_SOURCE_GRAPHICS_REPLAY_DRIFT"):
        _proof(cas,request,trusted_source_native_pixel_replayer=lambda court: None)
