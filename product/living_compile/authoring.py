from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .common import Json, LivingCompileError, USER_LAYER_SCHEMA, finite, finite_pair, json_write, load_product, parse_view, safe_id, view_id
from .scene import build_scene


def _joint_ids(product: Mapping[str, Any]) -> set[str]:
    joints = ((product.get("mechanical_state") or {}).get("skeleton") or {}).get("joints") or []
    return {str(row.get("canonical_joint_id")) for row in joints}


def _parent_map(product: Mapping[str, Any]) -> dict[str, str | None]:
    joints = ((product.get("mechanical_state") or {}).get("skeleton") or {}).get("joints") or []
    return {str(row.get("canonical_joint_id")): (str(row.get("parent_canonical_id")) if row.get("parent_canonical_id") else None) for row in joints}


def _validate_topology(product: Mapping[str, Any], raw_rows: Any) -> tuple[list[Json], set[str]]:
    raw_rows = [] if raw_rows is None else raw_rows
    if not isinstance(raw_rows, list) or len(raw_rows) > 1024:
        raise LivingCompileError("rig_topology_edits must be a list with <=1024 entries")
    canonical = _joint_ids(product); parent = _parent_map(product); user_bones: set[str] = set(); deleted: set[str] = set(); clean = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict): raise LivingCompileError("each rig topology edit must be an object")
        op = str(row.get("operation") or "").strip().lower()
        if op == "add_bone":
            bone = safe_id(row.get("bone_id"), field="bone_id")
            if bone in canonical or bone in user_bones: raise LivingCompileError(f"duplicate bone id: {bone}")
            par = str(row.get("parent_id") or "").strip() or None
            if par and par not in canonical and par not in user_bones: raise LivingCompileError(f"add_bone parent does not exist: {par}")
            anchor = finite_pair(row.get("anchor_xy"), field=f"add_bone[{index}].anchor_xy")
            user_bones.add(bone); parent[bone] = par
            clean.append({"operation": op, "bone_id": bone, "parent_id": par, "anchor_xy": list(anchor), "display_name": str(row.get("display_name") or bone)[:128]})
        elif op == "delete_bone":
            bone = safe_id(row.get("bone_id"), field="bone_id")
            if bone not in canonical and bone not in user_bones: raise LivingCompileError(f"delete_bone target does not exist: {bone}")
            deleted.add(bone); clean.append({"operation": op, "bone_id": bone})
        elif op == "reparent":
            bone = safe_id(row.get("bone_id"), field="bone_id")
            if bone not in canonical and bone not in user_bones: raise LivingCompileError(f"reparent target does not exist: {bone}")
            par = str(row.get("parent_id") or "").strip() or None
            if par and par not in canonical and par not in user_bones: raise LivingCompileError(f"reparent parent does not exist: {par}")
            if par == bone: raise LivingCompileError("bone cannot parent itself")
            parent[bone] = par; clean.append({"operation": op, "bone_id": bone, "parent_id": par})
        else:
            raise LivingCompileError(f"unsupported rig topology operation: {op}")
    for bone in deleted: parent.pop(bone, None)
    for bone, par in list(parent.items()):
        if par in deleted: parent[bone] = None
    for start in parent:
        seen = set(); node: str | None = start
        while node is not None and node in parent:
            if node in seen: raise LivingCompileError(f"rig topology edit creates cycle at {node}")
            seen.add(node); node = parent.get(node)
    return clean, (set(parent) | user_bones) - deleted


def _validate_ik(raw_rows: Any, valid_bones: set[str]) -> list[Json]:
    raw_rows = [] if raw_rows is None else raw_rows
    if not isinstance(raw_rows, list) or len(raw_rows) > 256: raise LivingCompileError("ik_constraints must be a list with <=256 entries")
    out = []; ids = set()
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict): raise LivingCompileError("each IK constraint must be an object")
        cid = safe_id(row.get("constraint_id") or f"IK:{index}", field="constraint_id")
        if cid in ids: raise LivingCompileError(f"duplicate IK constraint id: {cid}")
        ids.add(cid)
        start = safe_id(row.get("start_bone_id"), field="start_bone_id"); end = safe_id(row.get("end_bone_id"), field="end_bone_id"); target = safe_id(row.get("target_bone_id"), field="target_bone_id")
        for bone in (start, end, target):
            if bone not in valid_bones: raise LivingCompileError(f"IK constraint references unknown bone: {bone}")
        mix = finite(row.get("mix", 1.0), field=f"ik[{index}].mix")
        if not 0 <= mix <= 1: raise LivingCompileError("IK mix must be within [0,1]")
        iterations = int(row.get("iterations", 12))
        if iterations < 1 or iterations > 128: raise LivingCompileError("IK iterations must be within [1,128]")
        out.append({"constraint_id": cid, "start_bone_id": start, "end_bone_id": end, "target_bone_id": target, "mix": mix, "iterations": iterations, "bend_direction": -1 if int(row.get("bend_direction", 1)) < 0 else 1, "enabled": bool(row.get("enabled", True))})
    return out


def _validate_animation(raw_rows: Any, valid_bones: set[str]) -> list[Json]:
    raw_rows = [] if raw_rows is None else raw_rows
    if not isinstance(raw_rows, list) or len(raw_rows) > 128: raise LivingCompileError("animation_edits must be a list with <=128 clips")
    out = []; clip_ids = set()
    for ci, clip in enumerate(raw_rows):
        if not isinstance(clip, dict): raise LivingCompileError("each animation edit must be an object")
        clip_id = safe_id(clip.get("clip_id") or f"USER_CLIP_{ci}", field="clip_id")
        if clip_id in clip_ids: raise LivingCompileError(f"duplicate user clip id: {clip_id}")
        clip_ids.add(clip_id); duration = finite(clip.get("duration_seconds", 1.0), field="duration_seconds")
        if duration <= 0 or duration > 600: raise LivingCompileError("animation duration must be within (0,600]")
        tracks_out = []; seen_tracks = set()
        for track in list(clip.get("tracks") or []):
            if not isinstance(track, dict): raise LivingCompileError("animation track must be an object")
            bone = safe_id(track.get("bone_id"), field="animation bone_id")
            if bone not in valid_bones: raise LivingCompileError(f"animation track references unknown bone: {bone}")
            if bone in seen_tracks: raise LivingCompileError(f"duplicate animation track for {bone}")
            seen_tracks.add(bone); keys_out = []; previous = -math.inf
            for ki, key in enumerate(list(track.get("keys") or [])):
                if not isinstance(key, dict): raise LivingCompileError("animation key must be an object")
                time = finite(key.get("time_seconds"), field=f"key[{ki}].time_seconds")
                if time < 0 or time > duration or time <= previous: raise LivingCompileError("animation key times must be strictly increasing and inside clip")
                previous = time; translation = finite_pair(key.get("translation_xy", (0, 0)), field="key.translation_xy"); rotation = finite(key.get("rotation_degrees", 0), field="key.rotation_degrees"); scale = finite_pair(key.get("scale_xy", (1, 1)), field="key.scale_xy")
                if max(abs(rotation), abs(translation[0]), abs(translation[1])) > 100000: raise LivingCompileError("animation key exceeds bounded authoring preview limits")
                if not (0.01 <= scale[0] <= 100 and 0.01 <= scale[1] <= 100): raise LivingCompileError("animation key scale must be within [0.01,100]")
                curve = str(key.get("curve") or "linear").lower()
                if curve not in {"linear", "step", "cubic"}: raise LivingCompileError("animation curve must be linear, step, or cubic")
                keys_out.append({"time_seconds": time, "translation_xy": list(translation), "rotation_degrees": rotation, "scale_xy": list(scale), "curve": curve})
            tracks_out.append({"bone_id": bone, "keys": keys_out})
        out.append({"clip_id": clip_id, "display_name": str(clip.get("display_name") or clip_id)[:128], "duration_seconds": duration, "loop": bool(clip.get("loop", False)), "tracks": tracks_out})
    return out


def save_user_layer(root: Path, payload: Mapping[str, Any]) -> Json:
    product = load_product(root); view = parse_view(payload.get("view_id", "V0")); scene = build_scene(root)
    controls = {row["control_id"] for row in scene["puppet"]["rig"]["controls"] if row["view_id"] == view_id(view)}
    meshes = {row["mesh_id"]: row for row in scene["puppet"]["meshes"] if row["view_id"] == view_id(view)}
    bindings = {row["binding_id"]: row for row in scene["binding"]["bindings"] if row["view_id"] == view_id(view)}
    rig_edits = []
    for raw in list(payload.get("edits") or []):
        control = safe_id(raw.get("control_id"), field="control_id")
        if control not in controls: raise LivingCompileError(f"rig edit references unknown control in selected view: {control}")
        rig_edits.append({"control_id": control, "rotation_degrees": finite(raw.get("rotation_degrees", 0), field="rotation_degrees"), "translation_xy": list(finite_pair(raw.get("translation_xy", (0, 0)), field="translation_xy"))})
    mesh_edits = []
    for raw in list(payload.get("mesh_vertex_edits") or []):
        mid = str(raw.get("mesh_id") or ""); vi = int(raw.get("vertex_index", -1))
        if mid not in meshes or vi < 0 or vi >= len(meshes[mid]["vertices"]): raise LivingCompileError(f"mesh edit vertex outside range: {mid}:{vi}")
        mesh_edits.append({"mesh_id": mid, "vertex_index": vi, "translation_xy": list(finite_pair(raw.get("translation_xy", (0, 0)), field="mesh translation_xy"))})
    weight_edits = []
    for raw in list(payload.get("weight_edits") or []):
        mid = str(raw.get("mesh_id") or ""); vi = int(raw.get("vertex_index", -1)); control = safe_id(raw.get("control_id"), field="weight control_id"); delta = finite(raw.get("weight_delta", 0), field="weight_delta")
        if mid not in meshes or vi < 0 or vi >= len(meshes[mid]["vertices"]): raise LivingCompileError("weight edit vertex outside range")
        if control not in controls: raise LivingCompileError(f"weight edit references unknown control: {control}")
        if not -1 <= delta <= 1: raise LivingCompileError("weight_delta must be within [-1,1]")
        weight_edits.append({"mesh_id": mid, "vertex_index": vi, "control_id": control, "weight_delta": delta})
    binding_edits = []
    for raw in list(payload.get("binding_edits") or []):
        bid = str(raw.get("binding_id") or ""); control = safe_id(raw.get("primary_final_control_id"), field="binding primary control")
        if bid not in bindings: raise LivingCompileError(f"binding edit references unknown selected-view binding: {bid}")
        if control not in controls: raise LivingCompileError(f"binding edit references unknown control: {control}")
        binding_edits.append({"binding_id": bid, "primary_final_control_id": control})
    topology, valid_bones = _validate_topology(product, payload.get("rig_topology_edits")); ik = _validate_ik(payload.get("ik_constraints"), valid_bones); animation = _validate_animation(payload.get("animation_edits"), valid_bones)
    layer = {
        "schema_version": USER_LAYER_SCHEMA, "source_product_state_hash": product.get("product_state_hash"), "view_index": view, "view_id": view_id(view),
        "authority_class": "OPTIONAL_USER_AUTHORING_LAYER", "can_promote_product_directly": False, "requires_compiler_requalification": True,
        "requires_dynamic_revalidation": True, "deployment_status": "BLOCKED_PENDING_DYNAMIC_REVALIDATION",
        "rig_pose_edits": rig_edits, "mesh_vertex_edits": mesh_edits, "weight_edits": weight_edits, "binding_edits": binding_edits,
        "rig_topology_edits": topology, "ik_constraints": ik, "animation_edits": animation,
        "metadata": {"canonical_bundle_mutated": False, "topology_edit_count": len(topology), "ik_constraint_count": len(ik), "animation_clip_count": len(animation), "authoring_surface": "LIVING_COMPILE_V4_EDITOR"},
    }
    path = root.parent / f"{root.name}.user" / "layers" / f"view_{view:02d}.json"; json_write(path, layer)
    return {"schema_version": USER_LAYER_SCHEMA, "path": str(path), "edit_count": len(rig_edits), "mesh_vertex_edit_count": len(mesh_edits), "weight_edit_count": len(weight_edits), "binding_edit_count": len(binding_edits), "rig_topology_edit_count": len(topology), "ik_constraint_count": len(ik), "animation_clip_count": len(animation), "requires_dynamic_revalidation": True, "canonical_bundle_mutated": False}
