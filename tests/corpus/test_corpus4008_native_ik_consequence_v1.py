import pytest

from tools.corpus.blender_corpus4008_native_ik_consequence_v1 import (
    max_motion,
)


def test_named_joint_and_vertex_identity_is_required():
    baseline={"left::knee":(0,0,0),"left::foot":(1,0,0)}
    altered={"left::knee":(0,0,0),"left::foot":(1.1,0,0)}
    assert max_motion(baseline,altered)==pytest.approx(.1)
    assert max_motion(baseline,{"right::foot":(1.1,0,0)}) is None
    assert max_motion({}, {})==0.0


def test_skin_motion_is_not_same_as_joint_motion():
    bones={"rig::ankle":(0,0,0)}
    after={"rig::ankle":(0,0,0)}
    mesh={"mesh::1":(2,0,0)}
    deformed={"mesh::1":(2.025,0,0)}
    assert max_motion(bones,after)==0.0
    assert max_motion(mesh,deformed)==pytest.approx(.025)
    # A moving mesh alone does not qualify as an IK solver behavior court.
