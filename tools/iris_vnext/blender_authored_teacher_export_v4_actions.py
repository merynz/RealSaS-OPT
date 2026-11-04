"""Generic imported Blender Action bindings: armature, object and shape-key.

Capture the *authored importer owners* before neutralising animations. Binding
an Action to a convenient armature, or using its semantic name as an owner, is
forbidden. Exact imported NLA/active Action and Blender slot evidence decides
owners. Preserve all source-graph nodes for node/shape-key-only morphologies.
"""
from __future__ import annotations

import bpy

from tools.iris_vnext import blender_authored_teacher_export_v1 as base
from tools.iris_vnext import blender_authored_teacher_export_v2 as rig_adapter


def _slot_identifier(slot) -> str | None:
    if slot is None:
        return None
    return str(slot.identifier)


def capture_imported_action_bindings(actions: dict[str, object]) -> dict[str, list[dict]]:
    """Map each exact Blender Action to its imported active/NLA owners."""
    imported_actions = {action.name: action for action in actions.values()}
    bindings = {name: [] for name in imported_actions}
    for owner in base._animation_owners():
        animation = getattr(owner, "animation_data", None)
        if animation is None:
            continue
        references = []
        action = getattr(animation, "action", None)
        if action is not None:
            references.append((action, getattr(animation, "action_slot", None), "ACTIVE"))
        for track in animation.nla_tracks:
            for strip in track.strips:
                if getattr(strip, "action", None) is not None:
                    references.append(
                        (strip.action, getattr(strip, "action_slot", None), "NLA_IMPORTER")
                    )
        for action, slot, origin in references:
            if action.name not in bindings or imported_actions[action.name] is not action:
                continue
            bindings[action.name].append({
                "owner": owner,
                "owner_name": str(getattr(owner, "name", "")),
                "owner_type": str(owner.bl_rna.identifier),
                "source_binding": origin,
                "slot_identifier": _slot_identifier(slot),
            })
    for key in bindings:
        unique = {}
        for row in bindings[key]:
            owner_key = id(row["owner"])
            if owner_key in unique:
                original = unique[owner_key]
                if (original["slot_identifier"] is not None
                        and row["slot_identifier"] is not None
                        and original["slot_identifier"] != row["slot_identifier"]):
                    raise RuntimeError(
                        "VNEXT_TEACHER_CONFLICTING_IMPORTED_ACTION_SLOTS:" + key
                    )
                if original["slot_identifier"] is None:
                    unique[owner_key] = row
            else:
                unique[owner_key] = row
        bindings[key] = sorted(
            unique.values(),
            key=lambda row: (row["owner_type"], row["owner_name"]),
        )
    return bindings


def _assign_imported_slot(owner, action, wanted: str | None) -> dict:
    animation = owner.animation_data
    suitable = (
        list(animation.action_suitable_slots)
        if hasattr(animation, "action_suitable_slots") else []
    )
    if not suitable:
        return {
            "identifier": "LEGACY_ACTION_NO_SLOT_API",
            "name": str(getattr(owner, "name", "")),
            "target_id_type": "LEGACY",
        }
    if wanted is not None:
        matches = [slot for slot in suitable if str(slot.identifier) == wanted]
        if len(matches) != 1:
            raise RuntimeError("VNEXT_TEACHER_IMPORTER_SLOT_NOT_REPLAYABLE:"
                               + action.name + ":" + str(getattr(owner, "name", "")))
        slot = matches[0]
    elif len(suitable) == 1:
        slot = suitable[0]
    else:
        matches = [slot for slot in suitable
                   if str(slot.name_display) == str(getattr(owner, "name", ""))]
        if len(matches) != 1:
            raise RuntimeError("VNEXT_TEACHER_UNBOUND_SLOT_AMBIGUOUS:"
                               + action.name + ":" + str(getattr(owner, "name", "")))
        slot = matches[0]
    animation.action_slot = slot
    return {
        "identifier": str(slot.identifier),
        "name": str(slot.name_display),
        "target_id_type": str(slot.target_id_type),
    }


def activate_action(action, bindings: dict[str, list[dict]]) -> dict:
    """Replay all exact imported holders for one authored Action."""
    recorded = list(bindings.get(action.name) or ())
    if not recorded:
        # Preserve a separately typed rig inference path only when the source
        # truly has an armature. Never pretend a rig for non-rigged objects.
        armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
        if not armatures:
            raise RuntimeError("VNEXT_TEACHER_UNBOUND_RIGLESS_ACTION:"
                               + action.name)
        armature_name, slot = rig_adapter._activate_action(action)
        return {
            "carrier_kind": "LEGACY_RIG_OWNER_FALLBACK",
            "armature": armature_name,
            "action_slot_identifier": slot["identifier"],
            "action_slot_name": slot["name"],
            "action_slot_target_id_type": slot["target_id_type"],
            "imported_owners": [{
                "name": armature_name,
                "id_type": "ARMATURE_OBJECT",
                "binding": "UNIQUE_RIG_INFERENCE",
                "slot_identifier": slot["identifier"],
            }],
        }
    # The captured owner set is scoped exactly to this action, so replay does
    # not activate unrelated imported NLA strips or animation controls.
    for owner in base._animation_owners():
        animation = getattr(owner, "animation_data", None)
        if animation is None:
            continue
        animation.action = None
        if hasattr(animation, "use_nla"):
            animation.use_nla = False
        for track in animation.nla_tracks:
            track.mute = True
    result = []
    for row in recorded:
        owner = row["owner"]
        if owner.animation_data is None:
            owner.animation_data_create()
        if isinstance(owner, bpy.types.Object) and owner.type == "ARMATURE":
            owner.data.pose_position = "POSE"
        owner.animation_data.action = action
        selected = _assign_imported_slot(owner, action, row["slot_identifier"])
        result.append({
            "name": row["owner_name"],
            "id_type": row["owner_type"],
            "binding": row["source_binding"],
            "slot_identifier": selected["identifier"],
            "slot_name": selected["name"],
            "slot_target_id_type": selected["target_id_type"],
        })
    has_rig = any(
        isinstance(row["owner"], bpy.types.Object) and row["owner"].type == "ARMATURE"
        for row in recorded
    )
    has_shape_key = any(
        isinstance(row["owner"], bpy.types.Key) for row in recorded
    )
    kind = (
        "MIXED_IMPORTED_CONTROLS" if has_rig and len(recorded) > 1 else
        "IMPORTED_RIG_CONTROL" if has_rig else
        "IMPORTED_SHAPE_KEY_CONTROL" if has_shape_key and len(recorded) == 1 else
        "IMPORTED_NODE_OR_OBJECT_CONTROLS"
    )
    bpy.context.view_layer.update()
    return {
        "carrier_kind": kind,
        "armature": next(
            (row["owner_name"] for row in recorded
             if isinstance(row["owner"], bpy.types.Object)
             and row["owner"].type == "ARMATURE"),
            None,
        ),
        "imported_owners": result,
    }
