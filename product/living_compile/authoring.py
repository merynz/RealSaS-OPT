from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .common import (
    Json,
    LivingCompileError,
    USER_LAYER_SCHEMA,
    finite,
    finite_pair,
    json_write,
    load_product,
    parse_view,
    safe_id,
    view_id,
)
from .scene import build_scene


def _joint_ids(product: Mapping[str, Any]) -> set[str]:
    joints = ((product.get("mechanical_state") or {}).get("skeleton") or {}).get("joints") or []
    return {str(row.get("canonical_joint_id")) for row in joints if str(row.get("canonical_joint_id") or "")}


def _control_ids(product: Mapping[str, Any]) -> set[str]:
    controls = ((product.get("mechanical_state") or {}).get("control_program") or {}).get("controls") or []
    return {str(row.get("canonical_control_id")) for row in controls if str(row.get("canonical_control_id") or "")}


def _parent_map(product: Mapping[str, Any]) -> dict[str, str | None]:
    joints = ((product.get("mechanical_state") or {}).get("skeleton") or {}).get("joints") or []
    return {
        str(row.get("canonical_joint_id")): (
            str(row.get("parent_canonical_id")) if row.get("parent_canonical_id") else None
        )
        for row in joints
    }


def _validate_topology(product: Mapping[str, Any], raw_rows: Any) -> tuple[list[Json], set[str]]:
    raw_rows = [] if raw_rows is None else raw_rows
    if not isinstance(raw_rows, list) or len(raw_rows) > 1024:
        raise LivingCompileError("rig_topology_edits must be a list with <=1024 entries")
    canonical = _joint_ids(product)
    parent = _parent_map(product)
    user_bones: set[str] = set()
    deleted: set[str] = set()
    clean: list[Json] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise LivingCompileError("each rig topology edit must be an object")
        op = str(row.get("operation") or "").strip().lower()
        if op == "add_bone":
            bone = safe_id(row.get("bone_id"), field="bone_id")
            if bone in canonical or bone in user_bones:
                raise LivingCompileError(f"duplicate bone id: {bone}")
            par = str(row.get("parent_id") or "").strip() or None
            if par and par not in canonical and par not in user_bones:
                raise LivingCompileError(f"add_bone parent does not exist: {par}")
            anchor = finite_pair(row.get("anchor_xy"), field=f"add_bone[{index}].anchor_xy")
            user_bones.add(bone)
            parent[bone] = par
            clean.append(
                {
                    "operation": op,
                    "bone_id": bone,
                    "parent_id": par,
                    "anchor_xy": list(anchor),
                    "display_name": str(row.get("display_name") or bone)[:128],
                }
            )
        elif op == "delete_bone":
            bone = safe_id(row.get("bone_id"), field="bone_id")
            if bone not in canonical and bone not in user_bones:
                raise LivingCompileError(f"delete_bone target does not exist: {bone}")
            deleted.add(bone)
            clean.append({"operation": op, "bone_id": bone})
        elif op == "reparent":
            bone = safe_id(row.get("bone_id"), field="bone_id")
            if bone not in canonical and bone not in user_bones:
                raise LivingCompileError(f"reparent target does not exist: {bone}")
            par = str(row.get("parent_id") or "").strip() or None
            if par and par not in canonical and par not in user_bones:
                raise LivingCompileError(f"reparent parent does not exist: {par}")
            if par == bone:
                raise LivingCompileError("bone cannot parent itself")
            parent[bone] = par
            clean.append({"operation": op, "bone_id": bone, "parent_id": par})
        else:
            raise LivingCompileError(f"unsupported rig topology operation: {op}")
    for bone in deleted:
        parent.pop(bone, None)
    for bone, par in list(parent.items()):
        if par in deleted:
            parent[bone] = None
    for start in parent:
        seen = set()
        node: str | None = start
        while node is not None and node in parent:
            if node in seen:
                raise LivingCompileError(f"rig topology edit creates cycle at {node}")
            seen.add(node)
            node = parent.get(node)
    return clean, (set(parent) | user_bones) - deleted


def _validate_ik(raw_rows: Any, valid_bones: set[str]) -> list[Json]:
    raw_rows = [] if raw_rows is None else raw_rows
    if not isinstance(raw_rows, list) or len(raw_rows) > 256:
        raise LivingCompileError("ik_constraints must be a list with <=256 entries")
    out: list[Json] = []
    ids = set()
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            raise LivingCompileError("each IK constraint must be an object")
        cid = safe_id(row.get("constraint_id") or f"IK:{index}", field="constraint_id")
        if cid in ids:
            raise LivingCompileError(f"duplicate IK constraint id: {cid}")
        ids.add(cid)
        start = safe_id(row.get("start_bone_id"), field="start_bone_id")
        end = safe_id(row.get("end_bone_id"), field="end_bone_id")
        target = safe_id(row.get("target_bone_id"), field="target_bone_id")
        for bone in (start, end, target):
            if bone not in valid_bones:
                raise LivingCompileError(f"IK constraint references unknown bone: {bone}")
        mix = finite(row.get("mix", 1.0), field=f"ik[{index}].mix")
        if not 0 <= mix <= 1:
            raise LivingCompileError("IK mix must be within [0,1]")
        iterations = int(row.get("iterations", 12))
        if iterations < 1 or iterations > 128:
            raise LivingCompileError("IK iterations must be within [1,128]")
        out.append(
            {
                "constraint_id": cid,
                "start_bone_id": start,
                "end_bone_id": end,
                "target_bone_id": target,
                "mix": mix,
                "iterations": iterations,
                "bend_direction": -1 if int(row.get("bend_direction", 1)) < 0 else 1,
                "enabled": bool(row.get("enabled", True)),
            }
        )
    return out


def _mesh_vertex_id(index: int) -> str:
    return f"v:{int(index)}"


def _mesh_face_id(index: int) -> str:
    return f"f:{int(index)}"


def _mesh_area2(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _mesh_edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _validate_mesh_model(model: dict[str, Any]) -> None:
    vertices: dict[str, tuple[float, float]] = model["vertices"]
    faces: dict[str, list[str]] = model["faces"]
    face_signatures: set[tuple[str, str, str]] = set()
    edge_counts: dict[tuple[str, str], int] = {}
    for face_id, ids in faces.items():
        if len(ids) != 3 or len(set(ids)) != 3:
            raise LivingCompileError(f"mesh topology contains degenerate face: {face_id}")
        if any(vertex_id not in vertices for vertex_id in ids):
            raise LivingCompileError(f"mesh topology face references missing vertex: {face_id}")
        a, b, c = (vertices[vertex_id] for vertex_id in ids)
        if abs(_mesh_area2(a, b, c)) <= 1e-8:
            raise LivingCompileError(f"mesh topology contains zero-area face: {face_id}")
        signature = tuple(sorted(ids))
        if signature in face_signatures:
            raise LivingCompileError(f"mesh topology contains duplicate face: {face_id}")
        face_signatures.add(signature)
        for u, v in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            edge = _mesh_edge_key(u, v)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            if edge_counts[edge] > 2:
                raise LivingCompileError(f"mesh topology contains non-manifold edge: {edge[0]}|{edge[1]}")


def _split_face(ids: list[str], a: str, b: str, new_vertex_id: str) -> tuple[list[str], list[str]] | None:
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        if (ids[i] == a and ids[j] == b) or (ids[i] == b and ids[j] == a):
            return [ids[i], new_vertex_id, ids[k]], [new_vertex_id, ids[j], ids[k]]
    return None


def _base_mesh_model(mesh: Mapping[str, Any]) -> dict[str, Any]:
    vertices = {
        _mesh_vertex_id(index): finite_pair(point, field=f"mesh vertex {index}")
        for index, point in enumerate(list(mesh.get("vertices") or []))
    }
    faces = {
        _mesh_face_id(index): [_mesh_vertex_id(int(vertex_index)) for vertex_index in triangle]
        for index, triangle in enumerate(list(mesh.get("triangles") or []))
    }
    model = {"vertices": vertices, "faces": faces}
    _validate_mesh_model(model)
    return model


def _validate_mesh_topology(meshes: Mapping[str, Mapping[str, Any]], raw_rows: Any) -> list[Json]:
    raw_rows = [] if raw_rows is None else raw_rows
    if not isinstance(raw_rows, list) or len(raw_rows) > 4096:
        raise LivingCompileError("mesh_topology_edits must be a list with <=4096 entries")
    models = {mesh_id: _base_mesh_model(mesh) for mesh_id, mesh in meshes.items()}
    clean: list[Json] = []
    for operation_index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise LivingCompileError("each mesh topology edit must be an object")
        mesh_id = str(raw.get("mesh_id") or "")
        if mesh_id not in models:
            raise LivingCompileError(f"mesh topology edit references unknown selected-view mesh: {mesh_id}")
        model = models[mesh_id]
        vertices: dict[str, tuple[float, float]] = model["vertices"]
        faces: dict[str, list[str]] = model["faces"]
        op = str(raw.get("operation") or "").strip().lower()
        row: Json = {"operation": op, "mesh_id": mesh_id}
        if op == "add_vertex":
            vertex_id = safe_id(raw.get("vertex_id"), field="mesh vertex_id")
            if vertex_id in vertices:
                raise LivingCompileError(f"duplicate mesh vertex id: {vertex_id}")
            point = finite_pair(raw.get("position_xy"), field=f"mesh_topology[{operation_index}].position_xy")
            vertices[vertex_id] = point
            row.update({"vertex_id": vertex_id, "position_xy": list(point)})
        elif op == "move_vertex":
            vertex_id = safe_id(raw.get("vertex_id"), field="mesh vertex_id")
            if vertex_id not in vertices:
                raise LivingCompileError(f"move_vertex target missing: {vertex_id}")
            point = finite_pair(raw.get("position_xy"), field=f"mesh_topology[{operation_index}].position_xy")
            vertices[vertex_id] = point
            row.update({"vertex_id": vertex_id, "position_xy": list(point)})
        elif op == "delete_vertex":
            vertex_id = safe_id(raw.get("vertex_id"), field="mesh vertex_id")
            if vertex_id not in vertices:
                raise LivingCompileError(f"delete_vertex target missing: {vertex_id}")
            if any(vertex_id in ids for ids in faces.values()):
                raise LivingCompileError(f"delete_vertex still referenced: {vertex_id}")
            del vertices[vertex_id]
            row["vertex_id"] = vertex_id
        elif op == "add_face":
            face_id = safe_id(raw.get("face_id"), field="mesh face_id")
            vertex_ids = [safe_id(value, field="mesh face vertex_id") for value in list(raw.get("vertex_ids") or [])]
            if face_id in faces:
                raise LivingCompileError(f"duplicate mesh face id: {face_id}")
            faces[face_id] = vertex_ids
            row.update({"face_id": face_id, "vertex_ids": vertex_ids})
        elif op == "delete_face":
            face_id = safe_id(raw.get("face_id"), field="mesh face_id")
            if face_id not in faces:
                raise LivingCompileError(f"delete_face target missing: {face_id}")
            del faces[face_id]
            row["face_id"] = face_id
        elif op == "split_edge":
            a = safe_id(raw.get("vertex_a_id"), field="split vertex_a_id")
            b = safe_id(raw.get("vertex_b_id"), field="split vertex_b_id")
            vertex_id = safe_id(raw.get("new_vertex_id"), field="split new_vertex_id")
            if a == b or a not in vertices or b not in vertices:
                raise LivingCompileError("split_edge requires two existing distinct vertices")
            if vertex_id in vertices:
                raise LivingCompileError(f"duplicate mesh vertex id: {vertex_id}")
            incident = [(face_id, ids) for face_id, ids in list(faces.items()) if a in ids and b in ids]
            if len(incident) < 1 or len(incident) > 2:
                raise LivingCompileError("split_edge requires a manifold edge")
            t = finite(raw.get("t", 0.5), field="split_edge t")
            if not 0 < t < 1:
                raise LivingCompileError("split_edge t must be inside (0,1)")
            pa, pb = vertices[a], vertices[b]
            vertices[vertex_id] = (pa[0] * (1 - t) + pb[0] * t, pa[1] * (1 - t) + pb[1] * t)
            for face_id, ids in incident:
                replacements = _split_face(ids, a, b, vertex_id)
                if replacements is None:
                    raise LivingCompileError(f"split_edge orientation failure: {face_id}")
                del faces[face_id]
                faces[f"user:split:{operation_index}:{face_id}:0"] = replacements[0]
                faces[f"user:split:{operation_index}:{face_id}:1"] = replacements[1]
            row.update({"vertex_a_id": a, "vertex_b_id": b, "new_vertex_id": vertex_id, "t": t})
        elif op == "weld_vertices":
            keep = safe_id(raw.get("keep_vertex_id"), field="weld keep_vertex_id")
            remove = safe_id(raw.get("remove_vertex_id"), field="weld remove_vertex_id")
            if keep == remove or keep not in vertices or remove not in vertices:
                raise LivingCompileError("weld_vertices requires two existing distinct vertices")
            for face_id, ids in list(faces.items()):
                next_ids = [keep if vertex_id == remove else vertex_id for vertex_id in ids]
                if len(set(next_ids)) < 3:
                    del faces[face_id]
                else:
                    faces[face_id] = next_ids
            del vertices[remove]
            row.update({"keep_vertex_id": keep, "remove_vertex_id": remove})
        else:
            raise LivingCompileError(f"unsupported mesh topology operation: {op}")
        _validate_mesh_model(model)
        clean.append(row)
    return clean


def _validate_bezier(value: Any, *, field: str) -> list[float]:
    raw = [0.25, 0.1, 0.25, 1.0] if value is None else value
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise LivingCompileError(f"{field} must be [x1,y1,x2,y2]")
    clean = [finite(item, field=f"{field}[{index}]") for index, item in enumerate(raw)]
    if any(item < 0 or item > 1 for item in clean):
        raise LivingCompileError(f"{field} values must be within [0,1]")
    return clean


def _validate_animation(raw_rows: Any, valid_controls: set[str]) -> list[Json]:
    raw_rows = [] if raw_rows is None else raw_rows
    if not isinstance(raw_rows, list) or len(raw_rows) > 128:
        raise LivingCompileError("animation_edits must be a list with <=128 clips")
    out: list[Json] = []
    clip_ids = set()
    for ci, clip in enumerate(raw_rows):
        if not isinstance(clip, dict):
            raise LivingCompileError("each animation edit must be an object")
        clip_id = safe_id(clip.get("clip_id") or f"USER_CLIP_{ci}", field="clip_id")
        if clip_id in clip_ids:
            raise LivingCompileError(f"duplicate user clip id: {clip_id}")
        clip_ids.add(clip_id)
        duration = finite(clip.get("duration_seconds", 1.0), field="duration_seconds")
        if duration <= 0 or duration > 600:
            raise LivingCompileError("animation duration must be within (0,600]")
        fps = finite(clip.get("fps", 30), field="animation fps")
        if fps < 1 or fps > 120:
            raise LivingCompileError("animation fps must be within [1,120]")
        tracks_out: list[Json] = []
        seen_tracks = set()
        for track in list(clip.get("tracks") or []):
            if not isinstance(track, dict):
                raise LivingCompileError("animation track must be an object")
            control = safe_id(track.get("control_id") or track.get("bone_id"), field="animation control_id")
            if control not in valid_controls:
                raise LivingCompileError(f"animation track references unknown animator control: {control}")
            if control in seen_tracks:
                raise LivingCompileError(f"duplicate animation track for {control}")
            seen_tracks.add(control)
            keys_out: list[Json] = []
            previous = -math.inf
            for ki, key in enumerate(list(track.get("keys") or [])):
                if not isinstance(key, dict):
                    raise LivingCompileError("animation key must be an object")
                time = finite(key.get("time_seconds"), field=f"key[{ki}].time_seconds")
                if time < 0 or time > duration or time <= previous:
                    raise LivingCompileError("animation key times must be strictly increasing and inside clip")
                previous = time
                translation = finite_pair(key.get("translation_xy", (0, 0)), field="key.translation_xy")
                rotation = finite(key.get("rotation_degrees", 0), field="key.rotation_degrees")
                scale = finite_pair(key.get("scale_xy", (1, 1)), field="key.scale_xy")
                if max(abs(rotation), abs(translation[0]), abs(translation[1])) > 100000:
                    raise LivingCompileError("animation key exceeds bounded authoring preview limits")
                if not (0.01 <= scale[0] <= 100 and 0.01 <= scale[1] <= 100):
                    raise LivingCompileError("animation key scale must be within [0.01,100]")
                curve = str(key.get("curve") or "linear").lower()
                if curve == "cubic":
                    curve = "bezier"
                if curve not in {"linear", "step", "bezier"}:
                    raise LivingCompileError("animation curve must be linear, step, or bezier")
                clean_key: Json = {
                    "time_seconds": time,
                    "translation_xy": list(translation),
                    "rotation_degrees": rotation,
                    "scale_xy": list(scale),
                    "curve": curve,
                }
                if curve == "bezier":
                    clean_key["bezier"] = _validate_bezier(key.get("bezier"), field=f"key[{ki}].bezier")
                keys_out.append(clean_key)
            tracks_out.append({"bone_id": control, "control_id": control, "keys": keys_out})
        out.append(
            {
                "clip_id": clip_id,
                "display_name": str(clip.get("display_name") or clip_id)[:128],
                "duration_seconds": duration,
                "fps": fps,
                "loop": bool(clip.get("loop", False)),
                "tracks": tracks_out,
            }
        )
    return out


def save_user_layer(root: Path, payload: Mapping[str, Any]) -> Json:
    product = load_product(root)
    view = parse_view(payload.get("view_id", "V0"))
    scene = build_scene(root)
    rig = scene["puppet"]["rig"]
    deform_rows = list(rig.get("deform_controls") or rig.get("controls") or [])
    controls = {row["control_id"] for row in deform_rows if row["view_id"] == view_id(view)}
    meshes = {row["mesh_id"]: row for row in scene["puppet"]["meshes"] if row["view_id"] == view_id(view)}
    bindings = {row["binding_id"]: row for row in scene["binding"]["bindings"] if row["view_id"] == view_id(view)}

    rig_edits: list[Json] = []
    for raw in list(payload.get("edits") or []):
        control = safe_id(raw.get("control_id"), field="control_id")
        if control not in controls:
            raise LivingCompileError(f"rig edit references unknown deform joint in selected view: {control}")
        rig_edits.append(
            {
                "control_id": control,
                "rotation_degrees": finite(raw.get("rotation_degrees", 0), field="rotation_degrees"),
                "translation_xy": list(finite_pair(raw.get("translation_xy", (0, 0)), field="translation_xy")),
            }
        )

    mesh_edits: list[Json] = []
    for raw in list(payload.get("mesh_vertex_edits") or []):
        mid = str(raw.get("mesh_id") or "")
        vi = int(raw.get("vertex_index", -1))
        if mid not in meshes or vi < 0 or vi >= len(meshes[mid]["vertices"]):
            raise LivingCompileError(f"mesh edit vertex outside range: {mid}:{vi}")
        mesh_edits.append(
            {
                "mesh_id": mid,
                "vertex_index": vi,
                "translation_xy": list(finite_pair(raw.get("translation_xy", (0, 0)), field="mesh translation_xy")),
            }
        )

    mesh_topology = _validate_mesh_topology(meshes, payload.get("mesh_topology_edits"))

    weight_edits: list[Json] = []
    for raw in list(payload.get("weight_edits") or []):
        mid = str(raw.get("mesh_id") or "")
        vi = int(raw.get("vertex_index", -1))
        control = safe_id(raw.get("control_id"), field="weight control_id")
        delta = finite(raw.get("weight_delta", 0), field="weight_delta")
        if mid not in meshes or vi < 0 or vi >= len(meshes[mid]["vertices"]):
            raise LivingCompileError("weight edit vertex outside range")
        if control not in controls:
            raise LivingCompileError(f"weight edit references unknown deform joint: {control}")
        if not -1 <= delta <= 1:
            raise LivingCompileError("weight_delta must be within [-1,1]")
        weight_edits.append(
            {"mesh_id": mid, "vertex_index": vi, "control_id": control, "weight_delta": delta}
        )

    binding_edits: list[Json] = []
    for raw in list(payload.get("binding_edits") or []):
        bid = str(raw.get("binding_id") or "")
        control = safe_id(raw.get("primary_final_control_id"), field="binding primary control")
        if bid not in bindings:
            raise LivingCompileError(f"binding edit references unknown selected-view binding: {bid}")
        if control not in controls:
            raise LivingCompileError(f"binding edit references unknown deform joint: {control}")
        binding_edits.append({"binding_id": bid, "primary_final_control_id": control})

    topology, valid_bones = _validate_topology(product, payload.get("rig_topology_edits"))
    ik = _validate_ik(payload.get("ik_constraints"), valid_bones)
    canonical_controls = _control_ids(product)
    user_bones = valid_bones - _joint_ids(product)
    valid_animation_controls = (canonical_controls | user_bones) if canonical_controls else valid_bones
    animation = _validate_animation(payload.get("animation_edits"), valid_animation_controls)

    layer = {
        "schema_version": USER_LAYER_SCHEMA,
        "source_product_state_hash": product.get("product_state_hash"),
        "view_index": view,
        "view_id": view_id(view),
        "authority_class": "OPTIONAL_USER_AUTHORING_LAYER",
        "can_promote_product_directly": False,
        "requires_compiler_requalification": True,
        "requires_dynamic_revalidation": True,
        "deployment_status": "BLOCKED_PENDING_DYNAMIC_REVALIDATION",
        "rig_pose_edits": rig_edits,
        "mesh_vertex_edits": mesh_edits,
        "mesh_topology_edits": mesh_topology,
        "weight_edits": weight_edits,
        "binding_edits": binding_edits,
        "rig_topology_edits": topology,
        "ik_constraints": ik,
        "animation_edits": animation,
        "metadata": {
            "canonical_bundle_mutated": False,
            "mesh_topology_edit_count": len(mesh_topology),
            "topology_edit_count": len(topology),
            "ik_constraint_count": len(ik),
            "animation_clip_count": len(animation),
            "authoring_surface": "LIVING_COMPILE_V4_EDITOR",
            "animation_track_space": "ANIMATOR_CONTROLS" if canonical_controls else "LEGACY_DEFORM_CONTROL_FALLBACK",
        },
    }
    path = root.parent / f"{root.name}.user" / "layers" / f"view_{view:02d}.json"
    json_write(path, layer)
    return {
        "schema_version": USER_LAYER_SCHEMA,
        "path": str(path),
        "edit_count": len(rig_edits),
        "mesh_vertex_edit_count": len(mesh_edits),
        "mesh_topology_edit_count": len(mesh_topology),
        "weight_edit_count": len(weight_edits),
        "binding_edit_count": len(binding_edits),
        "rig_topology_edit_count": len(topology),
        "ik_constraint_count": len(ik),
        "animation_clip_count": len(animation),
        "requires_dynamic_revalidation": True,
        "canonical_bundle_mutated": False,
    }
