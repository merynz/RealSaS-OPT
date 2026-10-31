from __future__ import annotations

"""CAS-native product.motion executor for control-space CanonicalMotionIR.v2."""

import json
import math
from typing import Any, Callable

MOTION_EXECUTOR_CONTRACT_V2 = "RealSaS.ProductMotionCompile.v2"


def _one(request: dict[str, Any], artifact_type: str, schema_version: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in request.get("input_artifacts") or ()
        if str(row.get("artifact_type") or "") == artifact_type
        and str(row.get("schema_version") or "") == schema_version
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"MOTION_V2_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}"
        )
    return rows[0]


def _optional_one(
    request: dict[str, Any], artifact_type: str, schema_version: str
) -> dict[str, Any] | None:
    rows = [
        dict(row)
        for row in request.get("input_artifacts") or ()
        if str(row.get("artifact_type") or "") == artifact_type
        and str(row.get("schema_version") or "") == schema_version
    ]
    if len(rows) > 1:
        raise RuntimeError(
            f"MOTION_V2_INPUT_ARTIFACT_CARDINALITY:{artifact_type}:{schema_version}:{len(rows)}"
        )
    return rows[0] if rows else None


def _json_payload(read_object, ref):
    try:
        value = json.loads(read_object(ref).decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("MOTION_V2_TYPED_JSON_DECODE_FAILED") from exc
    if not isinstance(value, dict):
        raise RuntimeError("MOTION_V2_TYPED_JSON_OBJECT_REQUIRED")
    return value


def _json_bytes(value: Any) -> bytes:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _studio_motion_proposal(payload: dict[str, Any], *, rig, mechanics):
    from compiler.realsas_compiler_core.control_motion_ir_v1 import (
        ControlMotionClipProposalIRV1,
        ControlMotionKeyProposalIRV1,
        ControlMotionTrackProposalIRV1,
        seal_control_motion_proposal_v1,
    )

    if str(payload.get("schema_version") or "") != "RealSaS.UserPuppetEditLayer.v3":
        raise RuntimeError("MOTION_V2_STUDIO_AUTHORING_SCHEMA_DRIFT")
    if str(payload.get("authority_class") or "") != "OPTIONAL_USER_AUTHORING_LAYER":
        raise RuntimeError("MOTION_V2_STUDIO_AUTHORING_AUTHORITY_DRIFT")
    if payload.get("can_promote_product_directly") is not False:
        raise RuntimeError("MOTION_V2_STUDIO_DIRECT_PROMOTION_FORBIDDEN")
    if payload.get("requires_compiler_requalification") is not True:
        raise RuntimeError("MOTION_V2_STUDIO_REQUALIFICATION_FLAG_REQUIRED")
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("canonical_bundle_mutated") is not False:
        raise RuntimeError("MOTION_V2_STUDIO_CANONICAL_MUTATION_FORBIDDEN")

    forbidden_lists = (
        "rig_pose_edits",
        "mesh_vertex_edits",
        "mesh_topology_edits",
        "weight_edits",
        "binding_edits",
        "rig_topology_edits",
        "ik_constraints",
    )
    for field in forbidden_lists:
        if list(payload.get(field) or ()):  # animation-only shipping slice
            raise RuntimeError(f"MOTION_V2_STUDIO_UNSUPPORTED_EDIT_DOMAIN:{field}")
    presentation = payload.get("presentation_override")
    if isinstance(presentation, dict) and (
        presentation.get("graphics_evidence_departure") is True
        or list(presentation.get("hidden_mesh_ids") or ())
    ):
        raise RuntimeError("MOTION_V2_STUDIO_UNSUPPORTED_EDIT_DOMAIN:presentation_override")

    clips = list(payload.get("animation_edits") or ())
    if len(clips) != 1 or not isinstance(clips[0], dict):
        raise RuntimeError(f"MOTION_V2_STUDIO_ANIMATION_CLIP_CARDINALITY:{len(clips)}")
    clip = dict(clips[0])
    clip_id = str(clip.get("clip_id") or "").strip()
    if not clip_id:
        raise RuntimeError("MOTION_V2_STUDIO_CLIP_ID_REQUIRED")
    duration = float(clip.get("duration_seconds") or 0.0)
    fps = float(clip.get("fps") or 30.0)
    if not math.isfinite(duration) or duration <= 0.0:
        raise RuntimeError("MOTION_V2_STUDIO_DURATION_INVALID")
    if not math.isfinite(fps) or fps < 1.0 or fps > 120.0:
        raise RuntimeError("MOTION_V2_STUDIO_FPS_INVALID")

    tracks = []
    for track_index, raw_track in enumerate(list(clip.get("tracks") or ())):
        if not isinstance(raw_track, dict):
            raise RuntimeError("MOTION_V2_STUDIO_TRACK_OBJECT_REQUIRED")
        control_id = str(raw_track.get("control_id") or raw_track.get("bone_id") or "").strip()
        if not control_id:
            raise RuntimeError("MOTION_V2_STUDIO_CONTROL_ID_REQUIRED")
        keys = []
        for key_index, raw_key in enumerate(list(raw_track.get("keys") or ())):
            if not isinstance(raw_key, dict):
                raise RuntimeError("MOTION_V2_STUDIO_KEY_OBJECT_REQUIRED")
            curve = str(raw_key.get("curve") or "linear").strip().lower()
            if curve != "linear":
                raise RuntimeError(f"MOTION_V2_STUDIO_CURVE_NOT_YET_QUALIFIED:{curve}")
            translation = list(raw_key.get("translation_xy") or (0.0, 0.0))
            scale = list(raw_key.get("scale_xy") or (1.0, 1.0))
            if len(translation) != 2 or len(scale) != 2:
                raise RuntimeError("MOTION_V2_STUDIO_2D_TRANSFORM_INVALID")
            degrees = float(raw_key.get("rotation_degrees") or 0.0)
            radians = math.radians(degrees) * 0.5
            keys.append(
                ControlMotionKeyProposalIRV1(
                    time_seconds=float(raw_key["time_seconds"]),
                    translation=(float(translation[0]), float(translation[1]), 0.0),
                    rotation_quat_wxyz=(math.cos(radians), 0.0, 0.0, math.sin(radians)),
                    scale=(float(scale[0]), float(scale[1]), 1.0),
                    metadata={
                        "source": "STUDIO_AUTHORING_REVISION",
                        "curve": "linear",
                        "source_key_index": key_index,
                    },
                )
            )
        tracks.append(
            ControlMotionTrackProposalIRV1(
                proposal_track_id=f"studio:{clip_id}:{track_index}:{control_id}",
                canonical_control_id=control_id,
                keys=tuple(keys),
                metadata={"source": "STUDIO_AUTHORING_REVISION"},
            )
        )

    proposal = ControlMotionClipProposalIRV1(
        proposal_clip_id=f"studio:{clip_id}",
        display_name=str(clip.get("display_name") or clip_id),
        intent="user_authored",
        duration_seconds=duration,
        loop=bool(clip.get("loop", False)),
        tracks=tuple(tracks),
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
        metadata={
            "provider": "STUDIO_AUTHORING_REVISION",
            "source_clip_id": clip_id,
            "source_fps": fps,
            "source_view_id": str(payload.get("view_id") or ""),
            "curve_support": "LINEAR_ONLY_V1",
        },
    )
    return seal_control_motion_proposal_v1(proposal)


def execute_motion_capability_v2(
    request: dict[str, Any],
    *,
    read_object: Callable[[dict[str, Any]], bytes],
    put_bytes: Callable[[bytes], dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(request.get("capability_metadata") or {})
    if metadata.get("executor_contract") != MOTION_EXECUTOR_CONTRACT_V2:
        raise RuntimeError("MOTION_V2_EXECUTOR_CONTRACT_DRIFT")
    if str(request.get("kind") or "") != "MODULE":
        raise RuntimeError("MOTION_V2_CAPABILITY_KIND_DRIFT")
    if request.get("model_releases"):
        raise RuntimeError("MOTION_V2_LEARNED_MODEL_RELEASE_FORBIDDEN")

    from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
        canonical_mechanics_from_dict_v1,
    )
    from compiler.realsas_compiler_core.canonical_motion_ir_v2 import mint_canonical_motion_v2
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import canonical_rig_from_dict_v2
    from compiler.realsas_compiler_core.control_motion_ir_v1 import (
        control_motion_proposal_from_dict_v1,
        qualify_control_motion_v1,
    )
    from compiler.realsas_compiler_core.default_idle_provider_v1 import (
        DEFAULT_IDLE_PROVIDER_V1,
        build_default_idle_proposal_v1,
    )

    rig = canonical_rig_from_dict_v2(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalRigIR", "v2"))
    )
    mechanics = canonical_mechanics_from_dict_v1(
        _json_payload(read_object, _one(request, "RealSaS.CanonicalMechanicsIR", "v1"))
    )
    proposal_ref = _optional_one(request, "RealSaS.ControlMotionClipProposalIR", "v1")
    studio_ref = _optional_one(request, "RealSaS.StudioAuthoringRevision", "v1")
    if proposal_ref is not None and studio_ref is not None:
        raise RuntimeError("MOTION_V2_PROPOSAL_SOURCE_AMBIGUOUS")
    if studio_ref is not None:
        proposal = _studio_motion_proposal(
            _json_payload(read_object, studio_ref),
            rig=rig,
            mechanics=mechanics,
        )
        provider = "STUDIO_AUTHORING_REVISION"
        proposal_source = "STUDIO_AUTHORING_REVISION"
    elif proposal_ref is not None:
        proposal = control_motion_proposal_from_dict_v1(
            _json_payload(read_object, proposal_ref)
        )
        provider = str(proposal.metadata.get("provider") or "EXTERNAL_CONTROL_MOTION_PROVIDER")
        proposal_source = "EXTERNAL_ARTIFACT"
    else:
        proposal = build_default_idle_proposal_v1(rig, mechanics)
        provider = DEFAULT_IDLE_PROVIDER_V1
        proposal_source = "PRODUCT_DEFAULT_PROVIDER"

    qualified = qualify_control_motion_v1(proposal, rig=rig, mechanics=mechanics)
    motion = mint_canonical_motion_v2(qualified)
    sealed = put_bytes(_json_bytes(motion))
    output = {
        "role": "canonical_motion",
        "artifact_type": "RealSaS.CanonicalMotionIR",
        "schema_version": "v2",
        "storage_key": sealed["storage_key"],
        "content_sha256": sealed["content_sha256"],
        "size_bytes": sealed["size_bytes"],
        "authority_class": "COMPILER_CANONICAL_MOTION",
    }
    return {
        "capability_id": str(request.get("capability_id") or ""),
        "status": "PASS",
        "outputs": [output],
        "diagnostics": {
            "schema": "RealSaS.ProductMotionDiagnostics.v2",
            "motion_lineage_hash": motion.motion_lineage_hash,
            "rig_lineage_hash": motion.rig_lineage_hash,
            "control_program_hash": motion.control_program_hash,
            "mechanics_lineage_hash": motion.mechanics_lineage_hash,
            "track_count": len(motion.control_tracks),
            "intent": motion.intent,
            "motion_provider": provider,
            "proposal_source": proposal_source,
            "default_idle_provider_used": proposal_ref is None and studio_ref is None,
            "studio_authoring_revision_used": studio_ref is not None,
            "studio_curve_support": "LINEAR_ONLY_V1" if studio_ref is not None else None,
            "editable_motion_space": "ANIMATOR_CONTROLS",
            "direct_joint_track_authority": False,
            "constraint_solver_executed_by_compiler": False,
            "compiler_retargeting_performed": False,
            "product_revision_pass_claimed": False,
        },
        "failure": None,
    }


__all__ = ["MOTION_EXECUTOR_CONTRACT_V2", "execute_motion_capability_v2"]
