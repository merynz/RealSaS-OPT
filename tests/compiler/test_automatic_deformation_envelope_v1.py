import numpy as np

from compiler.realsas_compiler_core.camera_authority_v1 import build_qualified_camera_set
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import derive_deformation_envelope_v1
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.types import QualifiedJoint, QualifiedSkeletonIR

def test_stage25_generic_derivation_needs_no_character_rows():
    skeleton=QualifiedSkeletonIR(
        (
            QualifiedJoint("r",(0,0,0),None,(),"p0"),
            QualifiedJoint("c",(0,1,0),"r",(),"p1"),
        ),
        "r",{"status":"PASS"},"skel",
    )
    cameras=tuple(CameraProjectionV3(f"V{i}",i,(0,0,-2),(1,0,0),(0,1,0),(0,0,1),1.0,32) for i in range(8))
    cset=build_qualified_camera_set(cameras,source_bundle_sha256="a"*64)
    axis,env=derive_deformation_envelope_v1(skeleton=skeleton,camera_set=cset)
    assert len(axis["joint_axes"])==2
    assert axis["actual_motion_capability_claimed"] is False
    assert all(len(x["axis_xyz"])==3 for x in axis["joint_axes"])
    assert all(np.isfinite(x["axis_xyz"]).all() if isinstance(x["axis_xyz"],np.ndarray) else all(np.isfinite(x["axis_xyz"])) for x in axis["joint_axes"])
    assert all(r.min_rotation_deg==-10.0 and r.max_rotation_deg==10.0 for r in env.joint_ranges)
    assert env.metadata["actual_motion_capability_claimed"] is False
    assert env.metadata["actual_motion_capability_authority"]=="STAGE35_EXACT_QUATERNION_CLIP_EXECUTION"
    assert env.metadata["per_character_joint_range_authoring"] is False
    assert env.metadata["per_character_axis_authoring"] is False
