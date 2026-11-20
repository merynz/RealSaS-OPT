from __future__ import annotations

"""Real CAS -> ProductProof -> host-rerun of user edited native C++ RGB/face."""

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_proof_capability_v1,
)
from compiler.rss_vnext.workers.rss_vnext_canonical_authoring_revision_v1 import (
    compile_rss_vnext_canonical_user_edit_revision_v1,
)
from compiler.rss_vnext.workers.rss_vnext_edited_graphics_preview_v1 import (
    compile_rss_vnext_edited_graphics_preview_v1,
)
from compiler.rss_vnext.workers.rss_vnext_native_preview_package_v1 import (
    reopen_rss_vnext_native_preview_package_v1,
)
from compiler.rss_vnext.workers.rss_vnext_canonical_edit_native_execution_replay_v1 import (
    execute_canonical_user_edit_native_cpp_capture_v1,
    native_user_edit_court_hash_v1,
)
from compiler.rss_vnext.workers.rss_vnext_product_edit_native_evidence_bridge_v1 import (
    COURT_TYPE, RASTER_TYPE, EDIT_TYPE, PREVIEW_TYPE,
    make_trusted_native_edit_replayer_v1,
)
from tests.compiler.test_rss_vnext_raw_motion_proof_replay_v1 import (
    _complete_product_preflight,
)
from tests.compiler.test_rss_vnext_native_preview_package_v1 import _build_player


@pytest.fixture(scope="module")
def native_product_case(tmp_path_factory):
    root=tmp_path_factory.mktemp("rss-native-productproof-edit")
    cas,request,_,_=_complete_product_preflight(root)
    original_ref=next(
        ref for ref in request["input_artifacts"]
        if ref["artifact_type"]=="RealSaS.RSSvNextNativeInteractivePuppetPreview"
    )
    original=root/"raw-native.rss"
    original.write_bytes(cas.read_object(original_ref))
    state=reopen_rss_vnext_native_preview_package_v1(original)
    vertex=state["mesh"].vertices[0]
    revision_bytes,_=compile_rss_vnext_canonical_user_edit_revision_v1(
        source_rss_path=original,
        mesh_vertex_position_edits={
            vertex.canonical_mesh_vertex_id:(
                vertex.P[0]+.002,vertex.P[1],vertex.P[2],
            ),
        },
    )
    revision=root/"edited.rsedit"
    revision.write_bytes(revision_bytes)
    preview_bytes,_=compile_rss_vnext_edited_graphics_preview_v1(
        source_rss_path=original,revision_path=revision,
    )
    preview=root/"edited.rss"
    preview.write_bytes(preview_bytes)
    executable=_build_player(root)
    expected_sha=sha256(executable.read_bytes()).hexdigest()
    raster,court=execute_canonical_user_edit_native_cpp_capture_v1(
        source_rss_path=original,rsedit_path=revision,
        edited_preview_rss_path=preview,
        renderer_executable=executable,
        expected_renderer_executable_sha256=expected_sha,
        frame_index=0,view_index=0,
    )
    def _ref(raw,kind):
        ref=cas.put_bytes(raw)
        return {**ref,"artifact_type":kind,"schema_version":"v1"}
    extra=[
        _ref(json.dumps(court.to_dict(),sort_keys=True,separators=(",",":")).encode(),COURT_TYPE),
        _ref(raster,RASTER_TYPE),
        _ref(revision_bytes,EDIT_TYPE),
        _ref(preview_bytes,PREVIEW_TYPE),
    ]
    replayer=make_trusted_native_edit_replayer_v1(
        renderer_executable=executable,
        expected_renderer_executable_sha256=expected_sha,
    )
    return cas,request,extra,court,replayer


def _proof(cas,request,**kwargs):
    return execute_proof_capability_v1(
        request,read_object=cas.read_object,
        put_bytes=cas.put_bytes,**kwargs,
    )


def test_real_product_proof_cas_native_edit_court_replays_without_productpass(
    native_product_case,
):
    cas,base,extra,court,replayer=native_product_case
    request={**base,"input_artifacts":base["input_artifacts"]+extra}
    untrusted=_proof(cas,request)
    assert untrusted["status"]=="FAIL"
    assert untrusted["outputs"]==[]
    local=untrusted["diagnostics"]["independent_native_user_edit_evidence"]
    assert local["typed_user_edit_court_reopened"] is True
    assert local["candidate_edit_native_execution_claim_count"]==1
    assert local["independently_native_replayed_edit_frame_count"]==0
    assert untrusted["diagnostics"]["native_user_edit_independent_sample_present"] is False
    assert local["native_edit_sample"]["independent_host_replay_passed"] is False
    assert local["native_edit_product_revision_minted"] is False

    trusted=_proof(cas,request,trusted_native_edit_replayer=replayer)
    assert trusted["status"]=="FAIL"
    assert trusted["outputs"]==[]
    evidence=trusted["diagnostics"]["independent_native_user_edit_evidence"]
    assert evidence["independently_native_replayed_edit_frame_count"]==1
    assert evidence["native_edit_sample"]["independent_host_replay_passed"] is True
    assert evidence["native_edit_sample"]["original_source_camera_or_rgb_qualified"] is False
    assert evidence["native_edit_sample"]["full_temporal_edit_coverage_qualified"] is False
    assert evidence["native_edit_sample"]["authoring_revision_id"]==court.authoring_revision_id
    assert trusted["diagnostics"]["native_user_edit_independent_sample_present"] is True
    assert "EDIT_CONSEQUENCE_AND_RUNTIME_RSS_REOPEN" in trusted["diagnostics"][
        "missing_independent_witnesses"
    ]
    assert trusted["diagnostics"]["product_pass_authority"] is False
    assert trusted["diagnostics"]["product_revision_minted"] is False


def test_proof_rejects_orphan_native_raster_even_without_callback(native_product_case):
    cas,base,extra,_,_=native_product_case
    for kind in (COURT_TYPE,RASTER_TYPE,EDIT_TYPE,PREVIEW_TYPE):
        request={
            **base,
            "input_artifacts":base["input_artifacts"]+[
                row for row in extra if row["artifact_type"]==kind
            ],
        }
        with pytest.raises(QualificationError,match="INCOMPLETE_OR_UNBOUND_CAS"):
            _proof(cas,request)


def test_proof_refuses_claimed_host_replayer_from_untrusted_metadata(
    native_product_case,
):
    cas,base,extra,_,_=native_product_case
    request={
        **base,
        "capability_metadata":{
            **base["capability_metadata"],
            "trusted_native_edit_replayer":"execute /tmp/payload",
        },
        "input_artifacts":base["input_artifacts"]+extra,
    }
    with pytest.raises(RuntimeError,match="TRUSTED_EDIT_REPLAYER_HOST_ONLY"):
        _proof(cas,request)


def test_rehashed_productpass_claim_or_tampered_native_raster_is_not_independent(
    native_product_case,
):
    cas,base,extra,court,replayer=native_product_case
    forged=replace(court,product_release_admission=True,court_hash="")
    forged=replace(forged,court_hash=native_user_edit_court_hash_v1(forged))
    ref={**cas.put_bytes(json.dumps(forged.to_dict(),sort_keys=True,separators=(",",":")).encode()),"artifact_type":COURT_TYPE,"schema_version":"v1"}
    request={
        **base,
        "input_artifacts":base["input_artifacts"]+
            [ref if row["artifact_type"]==COURT_TYPE else row for row in extra],
    }
    with pytest.raises(QualificationError,match="FALSE_OR_INVALID_AUTHORITY"):
        _proof(cas,request,trusted_native_edit_replayer=replayer)

    ref=next(row for row in extra if row["artifact_type"]==RASTER_TYPE)
    blob=bytearray(cas.read_object(ref))
    blob[-1]^=1
    raster_ref={
        **cas.put_bytes(bytes(blob)),
        "artifact_type":RASTER_TYPE,"schema_version":"v1",
    }
    request={
        **base,
        "input_artifacts":base["input_artifacts"]+
            [raster_ref if row["artifact_type"]==RASTER_TYPE else row for row in extra],
    }
    with pytest.raises(QualificationError,match="COURT_PARENT_DRIFT"):
        _proof(cas,request,trusted_native_edit_replayer=replayer)


def test_host_echo_and_foreign_original_preview_never_qualify_edit(
    native_product_case,
):
    cas,base,extra,court,replayer=native_product_case
    request={**base,"input_artifacts":base["input_artifacts"]+extra}
    with pytest.raises(QualificationError,match="INDEPENDENT_HOST_REPLAY_DRIFT"):
        _proof(cas,request,trusted_native_edit_replayer=lambda **_: court.to_dict())
    old=next(
        row for row in base["input_artifacts"]
        if row["artifact_type"]=="RealSaS.RSSvNextNativeInteractivePuppetPreview"
    )
    fake={
        **cas.put_bytes(cas.read_object(old)+b"alien"),
        "artifact_type":old["artifact_type"],"schema_version":"v1",
    }
    request={
        **base,
        "input_artifacts":[
            fake if row is old else row for row in base["input_artifacts"]
        ]+extra,
    }
    with pytest.raises(Exception):
        _proof(cas,request,trusted_native_edit_replayer=replayer)


def test_native_face_owner_abi_accepts_production_mesh_above_4096_faces():
    import numpy as np
    from compiler.rss_vnext.workers.rss_vnext_canonical_edit_native_execution_replay_v1 import (
        _pack, reopen_native_edited_puppet_rasters_v1,
    )
    def rows():
        return {
            "rgba":np.zeros((16,16,4),dtype=np.uint8),
            "owners":np.full((16,16),-1,dtype="<i4"),
            "authority":np.zeros((16,16),dtype=np.uint8),
            "unknown":np.zeros((16,16),dtype=np.uint8),
        }
    original=rows()
    edited=rows()
    edited["owners"][3,5]=65_536
    edited["authority"][3,5]=2
    edited["rgba"][3,5]=[120,20,10,255]
    binary=_pack(original,edited,16)
    decoded=reopen_native_edited_puppet_rasters_v1(
        binary,resolution=16,face_count=65_537,
    )
    assert decoded["edited_front_face_i32"][3,5]==65_536
    with pytest.raises(QualificationError,match="FACE_OWNER"):
        reopen_native_edited_puppet_rasters_v1(
            binary,resolution=16,face_count=65_536,
        )
