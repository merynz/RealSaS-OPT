from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import pytest

from compiler.realsas_compiler_core.motion_source_v1 import (
    build_motion_source_asset,build_motion_source_set,build_qualified_motion_source_seal,
    validate_qualified_motion_source_seal,
)
from compiler.realsas_compiler_core.rest_preservation_policy_v1 import RestPreservationViewDecisionIR
from compiler.realsas_compiler_core.types import QualificationError


def _rest(*,authorized=True,failed_view=False):
    return SimpleNamespace(
        preservation_lineage_hash="rest-preservation-hash",
        view_decisions=tuple(
            RestPreservationViewDecisionIR(i,"FAIL" if failed_view and i==3 else "PASS",())
            for i in range(8)
        ),
        qualification_report={"motion_authorization_precondition_satisfied":authorized},
    )


def test_source_identity_is_reusable_and_seal_is_product_rest_bound():
    asset=build_motion_source_asset(
        clip_id="idle_a",clip_kind="IDLE",source_kind="INLINE_PRESET_SPEC_V1",
        source_space="PROCEDURAL_PARAMETER_SPACE_V1",duration_seconds=1.0,loop=True,
        channel_contract=("ROTATION_DEG",),
        source_payload={"preset_id":"MECHANICAL_SWAY_V1","amplitude_deg":4.0},
        source_ref="manifest:inline:idle_a",
    )
    sources=build_motion_source_set((asset,))
    product=SimpleNamespace(product_state_hash="product-a")
    rest=_rest()
    seal=build_qualified_motion_source_seal(
        source_set=sources,product_state=product,rest_preservation=rest
    )
    assert sources.metadata["motion_compilation_performed"] is False
    assert seal.qualification_report["motion_quality_claimed"] is False
    assert seal.product_state_binding_hash=="product-a"
    assert seal.rest_preservation_binding_hash=="rest-preservation-hash"
    changed_product=SimpleNamespace(product_state_hash="product-b")
    with pytest.raises(QualificationError,match="PRODUCT_STATE_DRIFT"):
        validate_qualified_motion_source_seal(
            seal,source_set=sources,product_state=changed_product,rest_preservation=rest
        )


def test_rest_preservation_is_hard_precondition_for_source_seal():
    asset=build_motion_source_asset(
        clip_id="run_a",clip_kind="RUN",source_kind="EXTERNAL_ARTIST_CLIP_V1",
        source_space="SOURCE_RIG_TRACKS_V1",duration_seconds=0.8,loop=True,
        channel_contract=("ROTATION_DEG","TRANSLATION_XY"),
        source_payload={"schema":"RealSaS.MotionSourceClip.v1","tracks":[]},
        source_ref="file:artist_run.json",
    )
    sources=build_motion_source_set((asset,))
    with pytest.raises(QualificationError,match="NOT_AUTHORIZED"):
        build_qualified_motion_source_seal(
            source_set=sources,product_state=SimpleNamespace(product_state_hash="p"),
            rest_preservation=_rest(authorized=False),
        )
    with pytest.raises(QualificationError,match="REST_VIEW_NOT_PASS"):
        build_qualified_motion_source_seal(
            source_set=sources,product_state=SimpleNamespace(product_state_hash="p"),
            rest_preservation=_rest(failed_view=True),
        )


def test_source_seal_does_not_confuse_rig_space_source_with_compiled_motion():
    asset=build_motion_source_asset(
        clip_id="run_artist",clip_kind="RUN",source_kind="EXTERNAL_ARTIST_CLIP_V1",
        source_space="SOURCE_RIG_TRACKS_V1",duration_seconds=1.2,loop=True,
        channel_contract=("ROTATION_DEG",),
        source_payload={"schema":"RealSaS.MotionSourceClip.v1","source_rig":"anonymous","tracks":[1,2,3]},
        source_ref="file:run.json",
        metadata={"requires_stage34_retargeting":True},
    )
    sources=build_motion_source_set((asset,))
    seal=build_qualified_motion_source_seal(
        source_set=sources,product_state=SimpleNamespace(product_state_hash="p"),
        rest_preservation=_rest(),
    )
    assert seal.source_assets[0].source_space=="SOURCE_RIG_TRACKS_V1"
    assert seal.qualification_report["retargeting_performed"] is False
    assert seal.qualification_report["stage34_compile_required"] is True
