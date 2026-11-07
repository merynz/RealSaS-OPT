#!/usr/bin/env python3
"""Source-bound imported rig-control inventory. OBSERVATION ONLY, never training authority.

Blender 4.5.14 importer can bake/strip native controller constructs. Therefore
"no imported IK" NEVER means "no IK in authored source". External source-format
parity and intervention tests are separate required courts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

# Blender bundled Python may ignore PYTHONPATH; bind package root explicitly.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_native_source_probe_v1 import (
    import_source, finite_matrix, require, sha,
)

SCHEMA = "RealSaS.Corpus4008ImportedControlProfile.v1"
MAX_OBJECTS = 10000
MAX_BONES = 15000
MAX_CONSTRAINTS = 200000
MAX_ACTION_CURVES = 400000
MAX_NLA_STRIPS = 100000


def _string(x):
    return "" if x is None else str(x)


def _number(x):
    if x is None:
        return None
    try:
        n = float(x)
    except (ValueError, TypeError):
        return None
    return n if math.isfinite(n) else None


def _id_reference(x):
    if x is None:
        return None
    return {
        "name_teacher_only": _string(getattr(x, "name", "")),
        "id_type": _string(getattr(getattr(x, "bl_rna", None), "identifier", type(x).__name__)),
    }


def property_inventory(obj):
    # Arbitrary source strings/custom driver expressions must not be executed
    # or taught to students. Scalar switch values are informative but not proof.
    out = []
    keys = getattr(obj, "keys", None)
    for key in sorted(keys() if callable(keys) else []):
        if key == "_RNA_UI":
            continue
        value = obj[key]
        rec = {"key_teacher_only": _string(key), "type": type(value).__name__}
        if isinstance(value, bool):
            rec["scalar_imported_value"] = value
        elif isinstance(value, (int, float)):
            rec["scalar_imported_value"] = _number(value)
        else:
            rec["scalar_imported_value"] = None
        out.append(rec)
    return out


def constraint_inventory(constraints):
    out = []
    for c in constraints:
        rec = {
            "name_teacher_only": _string(getattr(c, "name", "")),
            "type": _string(c.type),
            "enabled_imported": bool(
                not getattr(c, "mute", False)
                and bool(getattr(c, "enabled", True))
                and (_number(getattr(c, "influence", 1.0)) or 0) > 0
            ),
            "influence": _number(getattr(c, "influence", 1.0)),
            "target_id": _id_reference(getattr(c, "target", None)),
            "subtarget_teacher_only": _string(getattr(c, "subtarget", "")),
            "owner_space": _string(getattr(c, "owner_space", "")),
            "target_space": _string(getattr(c, "target_space", "")),
            "chain_count_imported": getattr(c, "chain_count", None),
            "pole_target_id": _id_reference(getattr(c, "pole_target", None)),
            "pole_subtarget_teacher_only": _string(getattr(c, "pole_subtarget", "")),
            "pole_angle": _number(getattr(c, "pole_angle", None)),
            "use_stretch": getattr(c, "use_stretch", None),
            "iterations": getattr(c, "iterations", None),
        }
        # "IK" on constraint is not independently verified IK functionality.
        out.append(rec)
    return out


def driver_inventory(owner, kind):
    ad = getattr(owner, "animation_data", None)
    curves = list(getattr(ad, "drivers", []) or [])
    result = []
    for c in curves:
        driver = c.driver
        variables = []
        for var in driver.variables:
            variables.append({
                "name_teacher_only": _string(var.name),
                "type": _string(var.type),
                "targets": [{
                    "id": _id_reference(getattr(t, "id", None)),
                    "data_path_teacher_only": _string(getattr(t, "data_path", "")),
                    "bone_target_teacher_only": _string(getattr(t, "bone_target", "")),
                    "transform_type": _string(getattr(t, "transform_type", "")),
                    "transform_space": _string(getattr(t, "transform_space", "")),
                } for t in var.targets],
            })
        expression = _string(getattr(driver, "expression", ""))
        result.append({
            "owner_kind": kind,
            "owner_id": _id_reference(owner),
            "data_path_teacher_only": _string(c.data_path),
            "array_index": int(c.array_index),
            "driver_type": _string(driver.type),
            "expression_sha256_teacher_only": hashlib.sha256(
                expression.encode("utf-8")).hexdigest(),
            "variables": variables,
            "imported_functionality_verified": False,
        })
    return result


def _action_curves(action):
    # Blender 4.4+ layered actions store F-curves in strips/channelbags.
    # Legacy actions expose action.fcurves. Absence in one API is not absence
    # of animation curves; we preserve status when introspection unavailable.
    bags = []
    layers = getattr(action, "layers", None)
    if layers is not None:
        for li, layer in enumerate(layers):
            for si, strip in enumerate(getattr(layer, "strips", [])):
                for bi, bag in enumerate(getattr(strip, "channelbags", [])):
                    bags.append((li, si, bi, list(getattr(bag, "fcurves", []))))
    if bags:
        return "LAYERED_CHANNELBAGS", bags
    try:
        old = list(getattr(action, "fcurves", []))
    except (AttributeError, RuntimeError):
        old = []
    if old:
        return "LEGACY_FCURVES", [(None, None, None, old)]
    return "NO_CURVES_EXPOSED_BY_SUPPORTED_APIS", []


def actions_inventory(bpy):
    out = []
    total_curves = 0
    for action in sorted(bpy.data.actions, key=lambda a: a.name):
        method, bags = _action_curves(action)
        curves = []
        for li, si, bi, fcurves in bags:
            for fc in fcurves:
                total_curves += 1
                require(total_curves <= MAX_ACTION_CURVES, "ACTION_CURVES_BUDGET")
                curves.append({
                    "layer_index": li, "strip_index": si,
                    "channelbag_index": bi,
                    "data_path_teacher_only": _string(fc.data_path),
                    "array_index": int(fc.array_index),
                    "keyframe_count": len(getattr(fc, "keyframe_points", [])),
                })
        fr = getattr(action, "frame_range", None)
        frame_range = ([_number(fr[0]), _number(fr[1])] if fr is not None else None)
        out.append({
            "action_name_teacher_only": action.name,
            "frame_range_imported": frame_range,
            "curve_discovery": method,
            "curve_count": len(curves),
            "curves": curves,
            "evaluated_motion_courted": False,
        })
    return out, total_curves


def nla_inventory(bpy):
    rows = []
    for obj in sorted(bpy.data.objects, key=lambda o: o.name):
        ad = getattr(obj, "animation_data", None)
        if ad is None:
            continue
        for track in getattr(ad, "nla_tracks", []):
            for strip in getattr(track, "strips", []):
                rows.append({
                    "owner_node_teacher_only": obj.name,
                    "track_teacher_only": track.name,
                    "strip_teacher_only": strip.name,
                    "action_teacher_only": getattr(getattr(strip, "action", None), "name", None),
                    "frame_start": _number(getattr(strip, "frame_start", None)),
                    "frame_end": _number(getattr(strip, "frame_end", None)),
                    "blend_type": _string(getattr(strip, "blend_type", "")),
                    "influence": _number(getattr(strip, "influence", None)),
                    "mute_imported": bool(getattr(track, "mute", False)),
                    "source_motion_accepted": False,
                })
                require(len(rows) <= MAX_NLA_STRIPS, "NLA_STRIPS_BUDGET")
    return rows


def bone_inventory(armature):
    rows = []
    for pb in sorted(armature.pose.bones, key=lambda x: x.name):
        bone = pb.bone
        collections = sorted(_string(x.name) for x in getattr(bone, "collections", []))
        c = constraint_inventory(pb.constraints)
        lock_rotation = [bool(x) for x in getattr(pb, "lock_rotation", [])]
        rows.append({
            "bone_name_teacher_only": pb.name,
            "parent_name_teacher_only": getattr(pb.parent, "name", None),
            "use_deform": bool(bone.use_deform),
            "use_connect": bool(getattr(bone, "use_connect", False)),
            "inherit_scale": _string(getattr(bone, "inherit_scale", "")),
            "bone_collections_teacher_only": collections,
            "bone_group_teacher_only": getattr(getattr(pb, "bone_group", None), "name", None),
            "matrix_rest_imported": finite_matrix(bone.matrix_local),
            "matrix_pose_imported": finite_matrix(pb.matrix),
            "length_imported": _number(bone.length),
            "roll_imported": _number(getattr(bone, "roll", None)),
            "rotation_mode": _string(pb.rotation_mode),
            "lock_location": [bool(x) for x in pb.lock_location],
            "lock_rotation": lock_rotation,
            "lock_rotation_w": bool(getattr(pb, "lock_rotation_w", False)),
            "lock_rotations_4d": bool(getattr(pb, "lock_rotations_4d", False)),
            "lock_scale": [bool(x) for x in pb.lock_scale],
            "use_ik_limit_x": bool(getattr(pb, "use_ik_limit_x", False)),
            "use_ik_limit_y": bool(getattr(pb, "use_ik_limit_y", False)),
            "use_ik_limit_z": bool(getattr(pb, "use_ik_limit_z", False)),
            "ik_min_x": _number(getattr(pb, "ik_min_x", None)),
            "ik_max_x": _number(getattr(pb, "ik_max_x", None)),
            "ik_min_y": _number(getattr(pb, "ik_min_y", None)),
            "ik_max_y": _number(getattr(pb, "ik_max_y", None)),
            "ik_min_z": _number(getattr(pb, "ik_min_z", None)),
            "ik_max_z": _number(getattr(pb, "ik_max_z", None)),
            "ik_stiffness_x": _number(getattr(pb, "ik_stiffness_x", None)),
            "ik_stiffness_y": _number(getattr(pb, "ik_stiffness_y", None)),
            "ik_stiffness_z": _number(getattr(pb, "ik_stiffness_z", None)),
            "custom_props_teacher_only": property_inventory(pb),
            "constraints": c,
            "is_control_bone_courted": False,
            "fk_controller_courted": False,
            "ik_solver_functional_courted": False,
        })
    return rows


def source_control_profile(bpy):
    objs = sorted(bpy.data.objects, key=lambda o: o.name)
    require(len(objs) <= MAX_OBJECTS, "OBJECT_COUNT_BUDGET")
    controllers = []
    obj_constraints = []
    drivers = []
    armature_count = 0
    bone_count = 0
    for o in objs:
        cs = constraint_inventory(o.constraints)
        if cs:
            obj_constraints.append({
                "owner_object_teacher_only": o.name,
                "object_type": o.type,
                "constraints": cs,
            })
        drivers.extend(driver_inventory(o, "OBJECT"))
        if getattr(o, "data", None) is not None:
            drivers.extend(driver_inventory(o.data, "OBJECT_DATA"))
            # Shape-key animation can carry independent facial/deform controls.
            shape_keys = getattr(o.data, "shape_keys", None)
            if shape_keys is not None:
                drivers.extend(driver_inventory(shape_keys, "SHAPE_KEYS"))
        if o.type == "ARMATURE":
            armature_count += 1
            bones = bone_inventory(o)
            bone_count += len(bones)
            require(bone_count <= MAX_BONES, "BONE_COUNT_BUDGET")
            controllers.append({
                "armature_name_teacher_only": o.name,
                "object_matrix_imported": finite_matrix(o.matrix_world),
                "armature_bone_collections_teacher_only": [
                    x.name for x in getattr(o.data, "collections", [])],
                "bone_count": len(bones),
                "deform_bone_count": sum(x["use_deform"] for x in bones),
                "nondeform_bone_count": sum(not x["use_deform"] for x in bones),
                "bones": bones,
                "armature_custom_props_teacher_only": property_inventory(o),
                "source_ik_rig_functional_courted": False,
            })
    actions, total_curves = actions_inventory(bpy)
    nla = nla_inventory(bpy)
    all_constraints = [
        c for arm in controllers for bone in arm["bones"]
        for c in bone["constraints"]
    ] + [
        c for row in obj_constraints for c in row["constraints"]
    ]
    require(len(all_constraints) <= MAX_CONSTRAINTS, "CONSTRAINT_BUDGET")
    kinds = Counter(c["type"] for c in all_constraints)
    cameras = [{
        "source_camera_name_teacher_only": o.name,
        "world_matrix_imported": finite_matrix(o.matrix_world),
        "lens_imported": _number(getattr(o.data, "lens", None)),
        "camera_type": _string(getattr(o.data, "type", "")),
        "native_character_orbit_courted": False,
    } for o in objs if o.type == "CAMERA"]
    return {
        "imported_object_count": len(objs),
        "imported_armature_count": armature_count,
        "imported_pose_bone_count": bone_count,
        "imported_nondeform_bone_count": sum(
            a["nondeform_bone_count"] for a in controllers),
        "imported_constraint_count": len(all_constraints),
        "imported_constraint_type_counts": dict(sorted(kinds.items())),
        "imported_ik_constraint_count": kinds.get("IK", 0),
        "imported_driver_count": len(drivers),
        "imported_nla_strip_count": len(nla),
        "imported_action_count": len(actions),
        "imported_action_curve_count": total_curves,
        "imported_camera_count": len(cameras),
        "armatures": controllers,
        "object_constraints": obj_constraints,
        "drivers": drivers,
        "actions": actions,
        "nla_strips": nla,
        "source_cameras": cameras,
        "actual_ik_functionality": "UNKNOWN",
        "ik_absence_in_authored_source": "UNKNOWN",
        "control_group_semantics": "UNKNOWN",
        "controller_bone_semantics": "UNKNOWN",
        "fk_ik_switch_functionality": "UNKNOWN",
        "retargetability": "UNKNOWN",
        "bind_deformation_consequence": "UNKNOWN",
        "source_pose_functionality": "UNKNOWN",
        "source_anatomical_up": "UNKNOWN",
        "source_anatomical_front": "UNKNOWN",
        "source_rest_pose_authored_parity": "UNKNOWN",
        "character_vs_prop": "UNKNOWN",
        "training_authorized": False,
    }


def run(bpy, *, source, expected_sha256, asset_id, candidate_id):
    require(source.is_file(), "SOURCE_NOT_FOUND")
    require(sha(source) == expected_sha256, "EXACT_ORIGINAL_SHA")
    require(
        candidate_id.endswith(expected_sha256) or
        candidate_id.endswith(expected_sha256[:24]),
        "CANDIDATE_SOURCE_SHA_DRIFT")
    import_source(bpy, source, load_anim=True)
    return {
        "schema": SCHEMA,
        "status": "PINNED_IMPORTED_CONTROL_OBSERVATION_ONLY",
        "asset_id": asset_id,
        "candidate_id": candidate_id,
        "original_sha256": expected_sha256,
        "imported_file_ext": source.suffix.lower(),
        "blender_importer_version": bpy.app.version_string,
        "importer_controls_may_be_baked_or_stripped": True,
        "imported_control": source_control_profile(bpy),
        "all_control_claims_require_native_and_heldout_intervention_courts": True,
        "source_original_bytes_preserved": True,
        "semantic_truth_authorized": False,
        "model_family_authorized": False,
        "training_authorized": False,
    }


def main():
    import bpy
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = run(bpy, source=args.source.resolve(),
                 expected_sha256=args.expected_sha256,
                 asset_id=args.asset_id, candidate_id=args.candidate_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    c = result["imported_control"]
    print(json.dumps({k: c[k] for k in (
        "imported_armature_count", "imported_pose_bone_count",
        "imported_ik_constraint_count", "imported_driver_count",
        "imported_action_count", "imported_action_curve_count",
        "imported_nla_strip_count", "actual_ik_functionality"
    )}, sort_keys=True))


if __name__ == "__main__":
    import sys
    old = sys.argv
    try:
        sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1:]
        main()
    finally:
        sys.argv = old
