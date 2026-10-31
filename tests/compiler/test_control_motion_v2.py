from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    CanonicalMechanicsIRV1,
    canonical_mechanics_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_motion_ir_v2 import (
    canonical_motion_from_dict_v2,
    mint_canonical_motion_v2,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
    _hash_control_program,
    _hash_deform_structure,
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.control_motion_ir_v1 import (
    ControlMotionClipProposalIRV1,
    ControlMotionKeyProposalIRV1,
    ControlMotionTrackProposalIRV1,
    qualify_control_motion_v1,
    seal_control_motion_proposal_v1,
)
from compiler.realsas_compiler_core.control_motion_runtime_v2 import (
    sample_control_transforms_v2,
    solve_motion_pose_v2,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.platform_worker.motion_executor_v2 import (
    MOTION_EXECUTOR_CONTRACT_V2,
    execute_motion_capability_v2,
)


def _rig():
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return CanonicalRigIRV2(
        joints=(CanonicalDeformJointIRV2("J0", (0.0, 0.0, 0.0), None, "PJ0"),),
        local_frames=(CanonicalJointFrameIRV2("J0", identity, ("RX", "RY", "RZ")),),
        controls=(CanonicalRigControlIRV2("C0", "ROOT", ("J0",), None, (0.0, 0.0, 0.0), "PC0"),),
        chains=(), constraints=(), attachment_anchors=(), deform_root_joint_ids=("J0",),
        mesh_lineage_hash="1" * 64,
        feature_evidence_binding_hash="2" * 64,
        proposal_evidence_hash="3" * 64,
        rig_program_qualification_hash="4" * 64,
        deform_structure_hash="5" * 64,
        control_program_hash="6" * 64,
        rig_lineage_hash="7" * 64,
        metadata={"authority": "COMPILER_CANONICAL_RIG"},
    )


def _mechanics(rig):
    return CanonicalMechanicsIRV1(
        mesh_lineage_hash=rig.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash="8" * 64,
        constraint_schedule=(),
        executable_operator_ids=(),
        rest_pose_hash="9" * 64,
        qualification_hash="a" * 64,
        mechanics_lineage_hash="b" * 64,
        metadata={"authority": "COMPILER_CANONICAL_MECHANICS"},
    )


def _sealed_executor_rig():
    rig = replace(_rig(), deform_structure_hash="", control_program_hash="", rig_lineage_hash="")
    rig = replace(rig, deform_structure_hash=_hash_deform_structure(rig))
    rig = replace(rig, control_program_hash=_hash_control_program(rig))
    return replace(rig, rig_lineage_hash=canonical_rig_lineage_hash_v2(rig))


def _sealed_executor_mechanics(rig):
    mechanics = replace(_mechanics(rig), mechanics_lineage_hash="")
    return replace(
        mechanics,
        mechanics_lineage_hash=canonical_mechanics_lineage_hash_v1(mechanics),
    )


def _proposal(rig, mechanics):
    return seal_control_motion_proposal_v1(ControlMotionClipProposalIRV1(
        proposal_clip_id="run",
        display_name="Run",
        intent="RUN",
        duration_seconds=1.0,
        loop=True,
        tracks=(ControlMotionTrackProposalIRV1(
            proposal_track_id="root-track",
            canonical_control_id="C0",
            keys=(
                ControlMotionKeyProposalIRV1(0.0, translation=(0.0, 0.0, 0.0)),
                ControlMotionKeyProposalIRV1(1.0, translation=(2.0, 0.0, 0.0)),
            ),
        ),),
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
    ))


def _studio_layer(*, curve="linear", mesh_edit=False):
    return {
        "schema_version": "RealSaS.UserPuppetEditLayer.v3",
        "source_product_state_hash": "c" * 64,
        "authority_class": "OPTIONAL_USER_AUTHORING_LAYER",
        "can_promote_product_directly": False,
        "requires_compiler_requalification": True,
        "view_id": "V0",
        "rig_pose_edits": [],
        "mesh_vertex_edits": (
            [{"mesh_id": "V0:BODY", "vertex_index": 0, "position_xy": [1.0, 2.0]}]
            if mesh_edit else []
        ),
        "mesh_topology_edits": [],
        "weight_edits": [],
        "binding_edits": [],
        "rig_topology_edits": [],
        "ik_constraints": [],
        "presentation_override": None,
        "animation_edits": [{
            "clip_id": "USER_WAVE",
            "display_name": "Wave",
            "duration_seconds": 1.0,
            "fps": 30.0,
            "loop": True,
            "tracks": [{
                "control_id": "C0",
                "keys": [
                    {
                        "time_seconds": 0.0,
                        "rotation_degrees": 0.0,
                        "translation_xy": [0.0, 0.0],
                        "scale_xy": [1.0, 1.0],
                        "curve": curve,
                    },
                    {
                        "time_seconds": 1.0,
                        "rotation_degrees": 90.0,
                        "translation_xy": [2.0, 3.0],
                        "scale_xy": [1.25, 0.75],
                        "curve": curve,
                    },
                ],
            }],
        }],
        "metadata": {
            "canonical_bundle_mutated": False,
            "animation_track_space": "ANIMATOR_CONTROLS",
        },
    }


def _execute_studio_motion(layer):
    rig = _sealed_executor_rig()
    mechanics = _sealed_executor_mechanics(rig)
    objects = {
        "rig": json.dumps(rig.to_dict(), sort_keys=True).encode("utf-8"),
        "mechanics": json.dumps(mechanics.to_dict(), sort_keys=True).encode("utf-8"),
        "studio": json.dumps(layer, sort_keys=True).encode("utf-8"),
    }
    stored = {}

    def read_object(ref):
        return objects[ref["storage_key"]]

    def put_bytes(payload):
        digest = hashlib.sha256(payload).hexdigest()
        stored[digest] = payload
        return {
            "storage_key": f"test/{digest}",
            "content_sha256": digest,
            "size_bytes": len(payload),
        }

    request = {
        "capability_id": "product.motion",
        "kind": "MODULE",
        "capability_metadata": {"executor_contract": MOTION_EXECUTOR_CONTRACT_V2},
        "model_releases": [],
        "input_artifacts": [
            {"artifact_type": "RealSaS.CanonicalRigIR", "schema_version": "v2", "storage_key": "rig"},
            {"artifact_type": "RealSaS.CanonicalMechanicsIR", "schema_version": "v1", "storage_key": "mechanics"},
            {"artifact_type": "RealSaS.StudioAuthoringRevision", "schema_version": "v1", "storage_key": "studio"},
        ],
    }
    result = execute_motion_capability_v2(request, read_object=read_object, put_bytes=put_bytes)
    output = result["outputs"][0]
    payload = stored[output["content_sha256"]]
    return result, canonical_motion_from_dict_v2(json.loads(payload.decode("utf-8")))


def test_control_motion_is_canonicalized_without_joint_track_authority():
    rig = _rig()
    mechanics = _mechanics(rig)
    qualified = qualify_control_motion_v1(_proposal(rig, mechanics), rig=rig, mechanics=mechanics)
    motion = mint_canonical_motion_v2(qualified)
    assert motion.metadata["editable_motion_space"] == "ANIMATOR_CONTROLS"
    assert motion.metadata["direct_joint_track_authority"] is False
    assert motion.metadata["constraint_execution_owner"] == "MECHANICS"
    assert canonical_motion_from_dict_v2(motion.to_dict()) == motion


def test_control_motion_sampling_drives_mechanics_controls_not_joint_tracks():
    rig = _rig()
    mechanics = _mechanics(rig)
    motion = mint_canonical_motion_v2(
        qualify_control_motion_v1(_proposal(rig, mechanics), rig=rig, mechanics=mechanics)
    )
    controls = sample_control_transforms_v2(motion, rig=rig, mechanics=mechanics, time_seconds=0.5)
    assert len(controls) == 1
    assert controls[0].canonical_control_id == "C0"
    assert np.allclose(controls[0].translation, (1.0, 0.0, 0.0))
    pose = solve_motion_pose_v2(motion, rig=rig, mechanics=mechanics, time_seconds=0.5)
    assert np.allclose(pose.joints[0].solved_P, (1.0, 0.0, 0.0))


def test_motion_rejects_unknown_control_without_retargeting():
    rig = _rig()
    mechanics = _mechanics(rig)
    proposal = _proposal(rig, mechanics)
    bad_track = replace(proposal.tracks[0], canonical_control_id="C_MISSING")
    bad = seal_control_motion_proposal_v1(replace(proposal, tracks=(bad_track,), proposal_lineage_hash=""))
    with pytest.raises(QualificationError, match="MOTION_V2_CONTROL_UNKNOWN"):
        qualify_control_motion_v1(bad, rig=rig, mechanics=mechanics)


def test_motion_rejects_non_unit_quaternion_instead_of_repairing_it():
    rig = _rig()
    mechanics = _mechanics(rig)
    proposal = _proposal(rig, mechanics)
    keys = list(proposal.tracks[0].keys)
    keys[0] = replace(keys[0], rotation_quat_wxyz=(2.0, 0.0, 0.0, 0.0))
    bad_track = replace(proposal.tracks[0], keys=tuple(keys))
    bad = seal_control_motion_proposal_v1(replace(proposal, tracks=(bad_track,), proposal_lineage_hash=""))
    with pytest.raises(QualificationError, match="MOTION_V2_ROTATION_NOT_UNIT"):
        qualify_control_motion_v1(bad, rig=rig, mechanics=mechanics)


def test_studio_linear_animation_is_qualified_into_canonical_motion():
    result, motion = _execute_studio_motion(_studio_layer())
    assert result["status"] == "PASS"
    assert result["diagnostics"]["studio_authoring_revision_used"] is True
    assert result["diagnostics"]["proposal_source"] == "STUDIO_AUTHORING_REVISION"
    assert motion.intent == "user_authored"
    assert len(motion.control_tracks) == 1
    assert motion.control_tracks[0].canonical_control_id == "C0"
    assert np.allclose(motion.control_tracks[0].keys[-1].translation, (2.0, 3.0, 0.0))
    assert np.allclose(motion.control_tracks[0].keys[-1].scale, (1.25, 0.75, 1.0))
    assert np.allclose(
        motion.control_tracks[0].keys[-1].rotation_quat_wxyz,
        (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
    )


def test_studio_bezier_animation_is_fail_closed_until_curve_execution_is_qualified():
    with pytest.raises(RuntimeError, match="MOTION_V2_STUDIO_CURVE_NOT_YET_QUALIFIED:bezier"):
        _execute_studio_motion(_studio_layer(curve="bezier"))


def test_studio_motion_executor_rejects_non_animation_authoring_domains():
    with pytest.raises(RuntimeError, match="MOTION_V2_STUDIO_UNSUPPORTED_EDIT_DOMAIN:mesh_vertex_edits"):
        _execute_studio_motion(_studio_layer(mesh_edit=True))
