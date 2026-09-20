from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.product_state_v2 import (
    CompletePuppetStateV2IR,
    QualifiedPresentationGraphV2IR,
    QualifiedPresentationStructureV2IR,
    complete_puppet_state_hash,
    complete_puppet_state_v2_from_dict,
    presentation_graph_v2_from_dict,
    presentation_graph_v2_hash,
    presentation_structure_v2_hash,
    presentation_structure_v2_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    CanonicalDynamicFrameV2IR,
    DynamicMotionClipProofV2IR,
    QualifiedDynamicMotionV2IR,
    canonical_dynamic_frame_v2_hash,
    dynamic_motion_clip_proof_v2_hash,
    qualified_dynamic_motion_v2_from_dict,
    qualified_dynamic_motion_v2_hash,
)
from compiler.realsas_compiler_core.runtime_authority_v2 import (
    DynamicVisualIntegrityV2IR,
    NativePlaybackProbeV2IR,
    NativePlaybackV2IR,
    RuntimeClipV2IR,
    RuntimePackageSealV2IR,
    RuntimeProjectionV2IR,
    RuntimeViewV2IR,
    dynamic_visual_integrity_from_dict,
    dynamic_visual_integrity_hash,
    native_playback_from_dict,
    native_playback_hash,
    native_playback_probe_hash,
    runtime_package_hash,
    runtime_package_seal_from_dict,
    runtime_projection_from_dict,
    runtime_projection_hash,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    PresentationViewOverlayIR,
)
from compiler.realsas_compiler_core.visibility_v2 import VISIBILITY_CONTRACT_V2_HASH
from compiler.realsas_compiler_core.types import QualificationError


H = "a" * 64


def _complete_puppet():
    value = CompletePuppetStateV2IR(
        mechanical_state_binding_hash="1" * 64,
        skeleton_binding_hash="2" * 64,
        mesh_binding_hash="3" * 64,
        mesh_skin_binding_hash="4" * 64,
        presentation_structure_binding_hash="5" * 64,
        presentation_graph_binding_hash="6" * 64,
        complete_appearance_asset_binding_hash="7" * 64,
        complete_appearance_qualification_binding_hash="8" * 64,
        output_direction_set_binding_hash="9" * 64,
        qualification_ledger=(
            {"authority": "MESH", "hash": "3" * 64},
            {"authority": "COMPLETE_APPEARANCE_ASSET", "hash": "7" * 64},
        ),
        complete_puppet_hash="",
        metadata={"geometry_mechanics_appearance_coequal": True},
    )
    return replace(value, complete_puppet_hash=complete_puppet_state_hash(value))




def test_presentation_graph_v2_has_explicit_caa_bindings_and_rejects_legacy_drift():
    overlay = PresentationViewOverlayIR(
        view_index=0,
        camera_binding_hash="c" * 64,
        appearance_binding_hash="7" * 64,
        composition_binding_hash="d" * 64,
        metadata={"visibility_contract_hash": VISIBILITY_CONTRACT_V2_HASH},
    )
    value = QualifiedPresentationGraphV2IR(
        slots=(),
        attachments=(),
        view_overlays=(overlay,),
        decisions=(),
        skeleton_binding_hash="1" * 64,
        mesh_binding_hash="2" * 64,
        partition_binding_hash="3" * 64,
        carrier_policy_binding_hash="4" * 64,
        mechanical_state_binding_hash="5" * 64,
        presentation_structure_binding_hash="6" * 64,
        complete_appearance_asset_binding_hash="7" * 64,
        complete_appearance_qualification_binding_hash="8" * 64,
        composition_policy_binding_hash="d" * 64,
        qualification_report={
            "status": "PASS_CAA_BOUND_PRESENTATION_V2",
            "visibility_contract_hash": VISIBILITY_CONTRACT_V2_HASH,
        },
        presentation_lineage_hash="",
        metadata={"legacy_appearance_set_semantics_used": False},
    )
    value = replace(
        value,
        presentation_lineage_hash=presentation_graph_v2_hash(value),
    )
    decoded = presentation_graph_v2_from_dict(value.to_dict())
    assert decoded.complete_appearance_asset_binding_hash == "7" * 64
    assert decoded.complete_appearance_qualification_binding_hash == "8" * 64
    assert decoded.mechanical_state_binding_hash == "5" * 64
    assert not hasattr(decoded, "appearance_set_binding_hash")
    assert not hasattr(decoded, "composition_set_binding_hash")
    assert not hasattr(decoded, "product_state_binding_hash")

    tampered = value.to_dict()
    tampered["complete_appearance_asset_binding_hash"] = "f" * 64
    with pytest.raises(QualificationError, match="PRESENTATION_GRAPH_V2_HASH_DRIFT"):
        presentation_graph_v2_from_dict(tampered)


def test_complete_puppet_decoder_rejects_any_unrehash_binding_drift():
    value = _complete_puppet()
    complete_puppet_state_v2_from_dict(value.to_dict())
    tampered = value.to_dict()
    tampered["complete_appearance_asset_binding_hash"] = "f" * 64
    with pytest.raises(QualificationError, match="COMPLETE_PUPPET_V2_HASH_DRIFT"):
        complete_puppet_state_v2_from_dict(tampered)


def test_presentation_structure_decoder_rejects_unrehash_drift():
    value = QualifiedPresentationStructureV2IR(
        slots=(),
        attachments=(),
        decisions=(),
        skeleton_binding_hash="1" * 64,
        mesh_binding_hash="2" * 64,
        mesh_skin_binding_hash="3" * 64,
        partition_binding_hash="4" * 64,
        carrier_policy_binding_hash="5" * 64,
        structure_hash="",
        metadata={
            "role_free": True,
            "appearance_authority_owned_elsewhere": True,
            "categorical_recognition_used": False,
        },
    )
    value = replace(value, structure_hash=presentation_structure_v2_hash(value))
    presentation_structure_v2_from_dict(value.to_dict())
    tampered = value.to_dict()
    tampered["mesh_binding_hash"] = "f" * 64
    with pytest.raises(QualificationError, match="PRESENTATION_V2_STRUCTURE_HASH_DRIFT"):
        presentation_structure_v2_from_dict(tampered)




def test_dynamic_motion_v2_decoder_rejects_mechanical_and_frame_drift():
    frame = CanonicalDynamicFrameV2IR(
        time_seconds=0.0,
        joint_world_positions=(("j0", (0.0, 0.0, 0.0)),),
        posed_vertex_xyz=(("v0", (0.0, 0.0, 0.0)),),
        max_vertex_displacement=0.0,
        min_triangle_area_ratio=1.0,
        max_triangle_area_ratio=1.0,
        max_triangle_condition_number=1.0,
        frame_hash="",
        metadata={"canonical_3d_authority": True},
    )
    frame = replace(frame, frame_hash=canonical_dynamic_frame_v2_hash(frame))
    clip = DynamicMotionClipProofV2IR(
        clip_id="idle",
        clip_kind="IDLE",
        classification="ARTIST_SOURCE",
        duration_seconds=1.0,
        loop=True,
        frames=(frame,),
        contact_proofs=(),
        max_vertex_displacement=0.0,
        professional_motion_evidence=True,
        clip_proof_hash="",
        metadata={},
    )
    clip = replace(clip, clip_proof_hash=dynamic_motion_clip_proof_v2_hash(clip))
    value = QualifiedDynamicMotionV2IR(
        qualified_motion_binding_hash="1" * 64,
        constraint_set_binding_hash="2" * 64,
        mechanical_state_binding_hash="3" * 64,
        skeleton_binding_hash="4" * 64,
        mesh_binding_hash="5" * 64,
        mesh_skin_binding_hash="6" * 64,
        presentation_binding_hash="7" * 64,
        mesh_policy_binding_hash="8" * 64,
        evaluator_semantic_version="RealSaS.CanonicalDynamicMotionEvaluator.QuaternionV3",
        clips=(clip,),
        qualification_report={"status": "PASS_DYNAMIC_CANONICAL_3D"},
        dynamic_motion_hash="",
        metadata={},
    )
    value = replace(
        value,
        dynamic_motion_hash=qualified_dynamic_motion_v2_hash(value),
    )
    qualified_dynamic_motion_v2_from_dict(value.to_dict())

    mechanical_tamper = value.to_dict()
    mechanical_tamper["mechanical_state_binding_hash"] = "f" * 64
    with pytest.raises(QualificationError, match="MOTION_V2_DYNAMIC_HASH_DRIFT"):
        qualified_dynamic_motion_v2_from_dict(mechanical_tamper)

    frame_tamper = value.to_dict()
    frame_tamper["clips"][0]["frames"][0]["max_vertex_displacement"] = 99.0
    with pytest.raises(QualificationError, match="MOTION_V2_DYNAMIC_FRAME_HASH_DRIFT"):
        qualified_dynamic_motion_v2_from_dict(frame_tamper)


def _projection():
    views = tuple(
        RuntimeViewV2IR(
            view_index=i,
            view_id=f"V{i}",
            camera={
                "origin": [0.0, 0.0, -2.0],
                "right": [1.0, 0.0, 0.0],
                "screen_up": [0.0, 1.0, 0.0],
                "forward": [0.0, 0.0, 1.0],
                "half_extent": 1.0,
                "resolution": 16,
            },
            texture_path=f"/tmp/V{i}.png",
            texture_sha256=f"{i:x}" * 64,
            metadata={"appearance": "SEALED_CAA_V2"},
        )
        for i in range(8)
    )
    value = RuntimeProjectionV2IR(
        complete_puppet_binding_hash="1" * 64,
        mechanical_state_binding_hash="2" * 64,
        mesh_binding_hash="3" * 64,
        dynamic_motion_binding_hash="4" * 64,
        appearance_asset_binding_hash="5" * 64,
        appearance_qualification_binding_hash="6" * 64,
        camera_set_binding_hash="7" * 64,
        visibility_contract_hash="8" * 64,
        projection_npz_path="/tmp/projection.npz",
        projection_npz_sha256="9" * 64,
        provenance_npz_path="/tmp/provenance.npz",
        provenance_npz_sha256="a" * 64,
        views=views,
        clips=(
            RuntimeClipV2IR(
                clip_id="idle",
                duration_seconds=1.0,
                loop=True,
                frame_count=3,
                array_prefix="clip_0",
            ),
        ),
        projection_hash="",
        metadata={
            "single_mesh_truth": True,
            "donor_search_at_runtime": False,
            "runtime_generation": False,
        },
    )
    return replace(value, projection_hash=runtime_projection_hash(value))


def test_runtime_projection_decoder_rejects_authority_drift():
    value = _projection()
    runtime_projection_from_dict(value.to_dict())
    tampered = value.to_dict()
    tampered["mesh_binding_hash"] = "f" * 64
    with pytest.raises(QualificationError, match="RUNTIME_V2_PROJECTION_HASH_DRIFT"):
        runtime_projection_from_dict(tampered)


def test_runtime_package_decoder_rejects_archive_identity_drift():
    value = RuntimePackageSealV2IR(
        projection_binding_hash="1" * 64,
        archive_path="/tmp/product.rss",
        archive_sha256="2" * 64,
        archive_bytes=123,
        package_format="REALSAS_RSS_V2_UNCOMPRESSED_CONTAINER",
        entry_names=("mesh.bin", "manifest.txt"),
        package_hash="",
        metadata={"contains_only_sealed_runtime_authorities": True},
    )
    value = replace(value, package_hash=runtime_package_hash(value))
    runtime_package_seal_from_dict(value.to_dict())
    tampered = value.to_dict()
    tampered["archive_bytes"] += 1
    with pytest.raises(QualificationError, match="RUNTIME_V2_PACKAGE_HASH_DRIFT"):
        runtime_package_seal_from_dict(tampered)


def _native_playback():
    probe = NativePlaybackProbeV2IR(
        clip_id="idle",
        view_id="V0",
        frame_index=1,
        rgba_raw_path="/tmp/a.rgba",
        rgba_raw_sha256="1" * 64,
        provenance_raw_path="/tmp/a.prov",
        provenance_raw_sha256="2" * 64,
        owner_raw_path="/tmp/a.owner",
        owner_raw_sha256="3" * 64,
        stdout_sha256="4" * 64,
        probe_hash="",
        metadata={"native_reference_mismatch_pixels": 0},
    )
    probe = replace(probe, probe_hash=native_playback_probe_hash(probe))
    value = NativePlaybackV2IR(
        package_binding_hash="5" * 64,
        projection_binding_hash="6" * 64,
        native_player_sha256="7" * 64,
        probes=(probe,),
        playback_hash="",
        metadata={"python_reference_byte_parity": True},
    )
    return replace(value, playback_hash=native_playback_hash(value))


def test_native_playback_decoder_rejects_probe_and_parent_drift():
    value = _native_playback()
    native_playback_from_dict(value.to_dict())

    probe_tampered = value.to_dict()
    probe_tampered["probes"][0]["frame_index"] = 2
    with pytest.raises(QualificationError, match="RUNTIME_V2_NATIVE_PROBE_HASH_DRIFT"):
        native_playback_from_dict(probe_tampered)

    parent_tampered = value.to_dict()
    parent_tampered["package_binding_hash"] = "f" * 64
    with pytest.raises(QualificationError, match="RUNTIME_V2_NATIVE_PLAYBACK_HASH_DRIFT"):
        native_playback_from_dict(parent_tampered)


def test_dynamic_visual_integrity_decoder_rejects_quality_metric_drift():
    value = DynamicVisualIntegrityV2IR(
        package_binding_hash="1" * 64,
        projection_binding_hash="2" * 64,
        native_playback_binding_hash="3" * 64,
        frame_count=24,
        geometry_visible_pixel_count=1000,
        final_alpha_hole_pixel_count=0,
        final_alpha_hole_fraction=0.0,
        compiled_unobserved_visible_pixel_count=100,
        compiled_unobserved_visible_fraction=0.1,
        native_reference_mismatch_pixel_count=0,
        maximum_frame_native_reference_mismatch_fraction=0.0,
        dynamic_conditioning_sample_count=24,
        relative_conditioning_sample_count=24,
        temporal_conditioning_sample_count=16,
        maximum_uv_to_surface_condition_number=1.2,
        maximum_relative_surface_condition_number=1.1,
        maximum_relative_surface_principal_stretch=1.15,
        maximum_adjacent_frame_surface_principal_stretch=1.05,
        qualification_report={
            "status": "PASS_DYNAMIC_VISUAL_INTEGRITY",
            "undefined_visible_pixel_count": 0,
        },
        visual_integrity_hash="",
        metadata={"geometry_visibility_appearance_sampling_attribution": True},
    )
    value = replace(
        value,
        visual_integrity_hash=dynamic_visual_integrity_hash(value),
    )
    dynamic_visual_integrity_from_dict(value.to_dict())
    tampered = value.to_dict()
    tampered["maximum_relative_surface_principal_stretch"] = 9.0
    with pytest.raises(QualificationError, match="RUNTIME_V2_VISUAL_INTEGRITY_HASH_DRIFT"):
        dynamic_visual_integrity_from_dict(tampered)
