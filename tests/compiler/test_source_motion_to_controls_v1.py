from __future__ import annotations

"""Rig-source clip -> animator-control MotionV2, no inferred correspondences."""
from copy import deepcopy
import json
import numpy as np
import pytest

from compiler.realsas_compiler_core.source_motion_to_controls_v1 import (
    retarget_source_motion_to_canonical_controls_v1 as retarget,
)
from compiler.realsas_compiler_core.canonical_motion_ir_v2 import (
    canonical_motion_from_dict_v2,mint_canonical_motion_v2,
)
from compiler.realsas_compiler_core.control_motion_ir_v1 import (
    qualify_control_motion_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_motion_capability_v2,MOTION_CONTRACT,
)
from tests.compiler.test_control_motion_v2 import _rig,_mechanics
from tests.compiler.test_motion_retarget_frames_v3 import RetargetFrameTests
from tests.compiler.test_product_downstream_cas_v1 import _Cas


def _source():
    out=RetargetFrameTests().source()
    out["schema_version"]="RealSaS.SourceRigMotion.v1"
    return out


def _binding():
    return {
        "schema_version":"RealSaS.ControlMotionRetargetBinding.v1",
        "target_joint_to_source":{"J0":"root"},
        "target_joint_to_control":{"J0":"C0"},
        "object_rotation":np.eye(3).tolist(),
        "target_body_scale":2.,
        "clip_id":"source_run",
        "display_name":"Source Run",
        "intent":"IMPORTED_SOURCE_CLIP",
        "loop":True,
    }


def _retarget(source=None,binding=None):
    binding=binding or _binding()
    rig=_rig()
    mechanics=_mechanics(rig)
    return retarget(
        source or _source(),rig=rig,mechanics=mechanics,
        target_joint_to_source=binding["target_joint_to_source"],
        target_joint_to_control=binding["target_joint_to_control"],
        object_rotation=binding["object_rotation"],
        target_body_scale=binding["target_body_scale"],
        clip_id=binding["clip_id"],
        display_name=binding["display_name"],
        intent=binding["intent"],
        loop=binding["loop"],
    )


def test_exact_source_motion_retarget_yields_control_space_clip():
    rig=_rig()
    mechanics=_mechanics(rig)
    proposal=_retarget()
    assert proposal.metadata["source_animation_rights_qualified"] is False
    assert proposal.metadata["visual_motion_parity_proven"] is False
    assert len(proposal.tracks)==1
    assert proposal.tracks[0].canonical_control_id=="C0"
    assert all(len(k.rotation_quat_wxyz)==4 for k in proposal.tracks[0].keys)
    clip=mint_canonical_motion_v2(qualify_control_motion_v1(
        proposal,rig=rig,mechanics=mechanics,
    ))
    assert clip.control_tracks[0].canonical_control_id=="C0"
    assert clip.duration_seconds==1.


def test_correspondence_must_be_exact_fk_control_not_nearest_joint():
    b=_binding()
    b["target_joint_to_source"]={"J0":"missing"}
    with pytest.raises(QualificationError,match="CORRESPONDENCE_UNKNOWN"):
        _retarget(binding=b)
    b=_binding()
    b["target_joint_to_control"]={"J0":"FAKE"}
    with pytest.raises(QualificationError,match="NOT_EXACT_FK_CONTROL"):
        _retarget(binding=b)
    b=_binding()
    b["object_rotation"]=np.diag([-1.,1.,1.]).tolist()
    with pytest.raises(QualificationError,match="OBJECT_BASIS_INVALID"):
        _retarget(binding=b)


def test_source_motion_worker_reopens_two_exact_cas_artifacts():
    cas=_Cas()
    rig=_rig()
    mechanics=_mechanics(rig)
    refs=[
        cas.put(rig,"RealSaS.CanonicalRigIR","v2"),
        cas.put(mechanics,"RealSaS.CanonicalMechanicsIR","v1"),
        cas.put(_source(),"RealSaS.SourceRigMotion","v1"),
        cas.put(_binding(),"RealSaS.ControlMotionRetargetBinding","v1"),
    ]
    request={
        "capability_id":"product.motion","kind":"MODULE",
        "capability_metadata":{"executor_contract":MOTION_CONTRACT},
        "model_releases":[],"input_artifacts":refs,
    }
    out=execute_motion_capability_v2(
        request,read_object=cas.read_object,put_bytes=cas.put_bytes,
    )
    assert out["status"]=="PASS"
    assert out["diagnostics"]["motion_source_kind"]==(
        "EXPLICIT_SOURCE_RIG_RETARGET_CANDIDATE_NOT_SOURCE_TRUTH"
    )
    assert out["diagnostics"]["source_motion_independently_verified"] is False
    clip=canonical_motion_from_dict_v2(json.loads(
        cas.read_object(out["outputs"][1]),
    ))
    assert len(clip.control_tracks)==1
    assert clip.motion_lineage_hash
    # A user cannot sneak two sources into the one-clip product contract.
    request["input_artifacts"].append(
        cas.put(_source(),"RealSaS.SourceRigMotion","v1"),
    )
    with pytest.raises(RuntimeError,match="SOURCE_AMBIGUOUS"):
        execute_motion_capability_v2(
            request,read_object=cas.read_object,put_bytes=cas.put_bytes,
        )


def test_worker_rejects_claimed_source_without_exact_binding():
    cas=_Cas()
    rig=_rig()
    mechanics=_mechanics(rig)
    request={
        "capability_id":"product.motion","kind":"MODULE",
        "capability_metadata":{"executor_contract":MOTION_CONTRACT},
        "model_releases":[],
        "input_artifacts":[
            cas.put(rig,"RealSaS.CanonicalRigIR","v2"),
            cas.put(mechanics,"RealSaS.CanonicalMechanicsIR","v1"),
            cas.put(_source(),"RealSaS.SourceRigMotion","v1"),
        ],
    }
    with pytest.raises(RuntimeError,match="SOURCE_AMBIGUOUS"):
        execute_motion_capability_v2(
            request,read_object=cas.read_object,put_bytes=cas.put_bytes,
        )
