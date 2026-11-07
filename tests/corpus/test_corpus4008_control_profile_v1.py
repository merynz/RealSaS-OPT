"""Fail-closed synthetic tests for imported-control evidence, never authored-rig proof."""
from __future__ import annotations

import math
from types import SimpleNamespace as NS

import pytest

from tools.corpus.blender_corpus4008_control_profile_v1 import (
    _action_curves, _number, bone_inventory, constraint_inventory,
    property_inventory, source_control_profile,
)


class Props:
    def __init__(self, d):
        self.d = d

    def keys(self):
        return self.d.keys()

    def __getitem__(self, k):
        return self.d[k]


def xmat():
    return [[1., 0., 0., 0.], [0., 1., 0., 0.],
            [0., 0., 1., 0.], [0., 0., 0., 1.]]


def fake_constraint(kind="IK", *, influence=1.0, target=None):
    return NS(type=kind, name="test", mute=False, enabled=True,
              influence=influence, target=target, subtarget="leg",
              owner_space="POSE", target_space="POSE", chain_count=2,
              pole_target=None, pole_subtarget="", pole_angle=0.,
              use_stretch=False, iterations=100)


def test_imported_ik_is_not_functional_ik_authority():
    c = constraint_inventory([fake_constraint("IK")])[0]
    assert c["type"] == "IK"
    assert c["chain_count_imported"] == 2
    assert c["enabled_imported"] is True
    assert c["pole_target_id"] is None
    assert "ik_solver_functional" not in c
    disabled = constraint_inventory([fake_constraint("IK", influence=0.)])[0]
    assert disabled["enabled_imported"] is False


def test_custom_props_keep_safe_numeric_switch_only():
    props = property_inventory(Props({
        "_RNA_UI": "ignored", "IK_FK": .5,
        "artist_note": "teacher-only metadata", "active": True,
    }))
    lookup = {r["key_teacher_only"]: r for r in props}
    assert "_RNA_UI" not in lookup
    assert lookup["IK_FK"]["scalar_imported_value"] == .5
    assert lookup["active"]["scalar_imported_value"] is True
    assert lookup["artist_note"]["scalar_imported_value"] is None
    assert _number(math.inf) is None


def test_layered_actions_and_legacy_actions_have_distinct_evidence():
    fc = NS(data_path="pose.bones[\\\"Foot\\\"].location", array_index=0,
            keyframe_points=[1, 2])
    bag = NS(fcurves=[fc])
    new = NS(layers=[NS(strips=[NS(channelbags=[bag])])], fcurves=[])
    method, result = _action_curves(new)
    assert method == "LAYERED_CHANNELBAGS" and len(result[0][3]) == 1
    legacy = NS(layers=[], fcurves=[fc])
    assert _action_curves(legacy)[0] == "LEGACY_FCURVES"
    unknown = NS(layers=[], fcurves=[])
    assert _action_curves(unknown) == ("NO_CURVES_EXPOSED_BY_SUPPORTED_APIS", [])


def fake_pose_bone(name, use_deform, constraints=()):
    data = NS(
        use_deform=use_deform, use_connect=False, inherit_scale="FULL",
        collections=[NS(name="Controls")] if not use_deform else [],
        matrix_local=xmat(), length=1., roll=0.,
    )
    return NS(
        name=name, parent=None, bone=data, matrix=xmat(),
        rotation_mode="XYZ", lock_location=(False, False, False),
        lock_rotation=(False, False, False), lock_rotation_w=False,
        lock_rotations_4d=False, lock_scale=(False, False, False),
        constraints=list(constraints), keys=lambda: [], __getitem__=lambda k: None,
    )


def test_pose_bone_control_candidate_is_not_confirmed_controller():
    ctrl = fake_pose_bone("IK.Target", False, [fake_constraint()])
    arm = NS(pose=NS(bones=[ctrl]))
    rows = bone_inventory(arm)
    assert rows[0]["bone_collections_teacher_only"] == ["Controls"]
    assert rows[0]["use_deform"] is False
    assert rows[0]["is_control_bone_courted"] is False
    assert rows[0]["ik_solver_functional_courted"] is False


def test_source_profile_preserves_negative_unknown_distinction():
    mesh = NS(
        name="mesh", type="MESH", constraints=[],
        data=NS(shape_keys=None, animation_data=None),
        animation_data=None,
    )
    bpy = NS(data=NS(objects=[mesh], actions=[]))
    p = source_control_profile(bpy)
    assert p["imported_ik_constraint_count"] == 0
    assert p["actual_ik_functionality"] == "UNKNOWN"
    assert p["ik_absence_in_authored_source"] == "UNKNOWN"
    assert p["character_vs_prop"] == "UNKNOWN"
    assert p["training_authorized"] is False


def test_real_source_budget_must_fail_closed_not_truncate():
    objs = [NS(name=str(i)) for i in range(10001)]
    with pytest.raises(ValueError, match="OBJECT_COUNT_BUDGET"):
        source_control_profile(NS(data=NS(objects=objs, actions=[])))
