from compiler.realsas_compiler_core.types import RiggingSurfaceIR, QualifiedJoint, QualifiedSkinIR
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from compiler.realsas_compiler_core.v4 import build_mechanical_state
from compiler.realsas_compiler_core.motion import build_deterministic_preset_motion

def mechanical(ids=('J:0','J:1')):
    surface=RiggingSurfaceIR((),geometry_lineage_hash='S')
    joints=(QualifiedJoint(ids[0],(0,0,0),None),QualifiedJoint(ids[1],(.2,0,0),ids[0]))
    sk=QualifiedSkeletonIRV2(joints,(ids[0],),{}, {'status':'PASS'},'')
    from compiler.realsas_compiler_core.v4 import qualified_skeleton_v2_lineage_hash
    from dataclasses import replace
    sk=replace(sk,skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(sk))
    skin=QualifiedSkinIR((),surface.geometry_lineage_hash,sk.skeleton_lineage_hash,{'status':'PASS'},'W')
    return build_mechanical_state(surface,sk,skin)

def test_deterministic_puppet_local_preset():
    a=build_deterministic_preset_motion(mechanical()); b=build_deterministic_preset_motion(mechanical())
    assert a.motion_state_hash==b.motion_state_hash
    assert a.joint_tracks[0].canonical_joint_id=='J:1'
    assert a.joint_tracks[0].transform_space=='PUPPET_LOCAL_2D_2P5D'
    assert len({k.rotation_deg for k in a.joint_tracks[0].keys})>1
    assert a.metadata['authored_joint_names_used'] is False

def test_joint_identity_changes_motion_hash():
    a=build_deterministic_preset_motion(mechanical(('J:A','J:B')))
    b=build_deterministic_preset_motion(mechanical(('J:A','J:C')))
    assert a.motion_state_hash!=b.motion_state_hash
    assert all(not hasattr(k,'rotation_xyzw') for k in a.joint_tracks[0].keys)
