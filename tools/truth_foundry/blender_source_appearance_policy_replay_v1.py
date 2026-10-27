#!/usr/bin/env python3
from __future__ import annotations

"""Replay the admitted source-observation material policy on a packed evaluator.

The packed Truth Foundry scene may come from a different Blender import lane than
source observation admission.  This adapter therefore does not infer opacity
from the packed material's current blend mode.  It replays only the exact sealed
``material_transform`` consequence recorded by the admitted
``SOURCE_REST_RENDER_CANDIDATE.json`` and uses the exact admitted camera bundle.

This is a source-equivalence court adapter, not a qualified appearance sampler.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.observation import blender_controlled_8view_v1 as orbit
from tools.observation import blender_controlled_8view_v2 as controlled

SCHEMA = "RealSaS.SourceAppearancePolicyReplay.v1"
_ALLOWED_ALPHA_POLICIES = {
    "IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE",
    "IMPORTED_SHADER_ALPHA_LINK_RETAINED",
    "IMPORTED_SHADER_ALPHA_CONSTANT_RETAINED",
}


def _args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--camera-bundle", type=Path, required=True)
    p.add_argument("--expected-blend-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args(raw)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _materials_from_render_meshes(meshes) -> dict[str, bpy.types.Material]:
    rows: dict[str, bpy.types.Material] = {}
    for obj in meshes:
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                continue
            if material.name in rows and rows[material.name] != material:
                raise RuntimeError("SOURCE_APPEARANCE_REPLAY_MATERIAL_NAME_COLLISION:" + material.name)
            rows[material.name] = material
    return rows


def _single_principled(material: bpy.types.Material):
    if not material.use_nodes or material.node_tree is None:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_NODE_GRAPH_REQUIRED:" + material.name)
    rows = [node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"]
    if len(rows) != 1:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_PRINCIPLED_CARDINALITY:" + material.name)
    return rows[0]


def _replay_transform(materials: dict[str, bpy.types.Material], transform_rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    receipts: list[dict] = []
    for row in transform_rows:
        name = str(row.get("material") or "")
        if not name or name in seen:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_MATERIAL_IDENTITY_INVALID")
        seen.add(name)
        material = materials.get(name)
        if material is None:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_MATERIAL_MISSING:" + name)
        operation = str(row.get("operation") or "")
        if operation == "NATIVE_EMISSION_GRAPH_PRESERVED":
            if not material.use_nodes or material.node_tree is None:
                raise RuntimeError("SOURCE_APPEARANCE_REPLAY_NATIVE_EMISSION_GRAPH_MISSING:" + name)
            if not any(node.type == "EMISSION" for node in material.node_tree.nodes):
                raise RuntimeError("SOURCE_APPEARANCE_REPLAY_NATIVE_EMISSION_NODE_MISSING:" + name)
            receipts.append({"material": name, "operation": operation, "replayed": True})
            continue
        if operation != "BSDF_LIGHT_RESPONSE_TO_EMISSION__ALPHA_GRAPH_RETAINED":
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_OPERATION_UNSUPPORTED:" + name + ":" + operation)

        bsdf = _single_principled(material)
        tree = material.node_tree
        color = bsdf.inputs.get("Base Color")
        alpha = bsdf.inputs.get("Alpha")
        if color is None or alpha is None:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_BASE_COLOR_ALPHA_SOCKET_MISSING:" + name)
        expected_color_linked = row.get("source_color_linked")
        expected_alpha_linked = row.get("source_alpha_linked")
        if type(expected_color_linked) is not bool or type(expected_alpha_linked) is not bool:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_LINK_RECEIPT_INVALID:" + name)
        if bool(color.is_linked) != expected_color_linked or bool(alpha.is_linked) != expected_alpha_linked:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_UPSTREAM_LINK_DRIFT:" + name)
        if not expected_alpha_linked:
            expected_default = float(row.get("source_alpha_default"))
            if not math.isfinite(expected_default) or abs(float(alpha.default_value) - expected_default) > 1.0e-12:
                raise RuntimeError("SOURCE_APPEARANCE_REPLAY_ALPHA_DEFAULT_DRIFT:" + name)

        consumers = [link.to_socket for link in list(bsdf.outputs["BSDF"].links)]
        if not consumers:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_SHADER_UNCONSUMED:" + name)
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0
        if color.is_linked:
            tree.links.new(color.links[0].from_socket, emission.inputs["Color"])
        else:
            emission.inputs["Color"].default_value = tuple(color.default_value)
        transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
        mix = tree.nodes.new("ShaderNodeMixShader")

        alpha_policy = str(row.get("alpha_policy") or "")
        if alpha_policy not in _ALLOWED_ALPHA_POLICIES:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_ALPHA_POLICY_UNSUPPORTED:" + name + ":" + alpha_policy)
        if alpha_policy == "IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE":
            if str(row.get("imported_blend_method") or "") != "OPAQUE":
                raise RuntimeError("SOURCE_APPEARANCE_REPLAY_OPAQUE_POLICY_RECEIPT_DRIFT:" + name)
            mix.inputs[0].default_value = 1.0
        elif alpha_policy == "IMPORTED_SHADER_ALPHA_LINK_RETAINED":
            if not alpha.is_linked:
                raise RuntimeError("SOURCE_APPEARANCE_REPLAY_ALPHA_LINK_MISSING:" + name)
            tree.links.new(alpha.links[0].from_socket, mix.inputs[0])
        else:
            if alpha.is_linked:
                raise RuntimeError("SOURCE_APPEARANCE_REPLAY_ALPHA_CONSTANT_LINKED:" + name)
            mix.inputs[0].default_value = float(row["source_alpha_default"])

        tree.links.new(transparent.outputs[0], mix.inputs[1])
        tree.links.new(emission.outputs[0], mix.inputs[2])
        for consumer in consumers:
            tree.links.new(mix.outputs[0], consumer)
        receipts.append({
            "material": name,
            "operation": operation,
            "alpha_policy": alpha_policy,
            "packed_blend_method_observed": str(getattr(material, "blend_method", "UNAVAILABLE")),
            "packed_surface_render_method_observed": str(getattr(material, "surface_render_method", "UNKNOWN")),
            "replayed": True,
        })

    if seen != set(materials):
        missing = sorted(set(materials) - seen)
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_UNSEALED_RENDER_MATERIALS:" + ",".join(missing))
    return receipts


def _render_masks(bundle: dict, output: Path) -> list[dict]:
    cameras = tuple(bundle.get("cameras") or ())
    if len(cameras) != 8:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CAMERA_COUNT")
    resolution_values = {int(row["resolution"]) for row in cameras}
    if len(resolution_values) != 1:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CAMERA_RESOLUTION_DRIFT")
    resolution = next(iter(resolution_values))
    controlled.configure(resolution)
    scene = bpy.context.scene
    data = bpy.data.cameras.new("RealSaS_SourceAppearanceReplay_Camera")
    data.type = "ORTHO"
    camera = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    views = []
    for expected_index, row in enumerate(cameras):
        index = int(row["view_index"])
        if index != expected_index or str(row["view_id"]) != f"V{index}":
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CAMERA_ORDER")
        origin = Vector(tuple(map(float, row["origin"])))
        forward = Vector(tuple(map(float, row["forward"])))
        if not all(math.isfinite(float(v)) for v in (*origin, *forward)) or forward.length == 0.0:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CAMERA_NONFINITE_OR_ZERO_FORWARD")
        # Reproduce the admitted renderer exactly: the sealed camera bundle is
        # already hash-bound by admission, and the original renderer consumed
        # the stored forward vector directly through ``to_track_quat``.  Do not
        # add a new unit-length tolerance that was not part of that authority.
        camera.location = origin
        camera.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
        data.ortho_scale = 2.0 * float(row["half_extent"])
        data.clip_start = float(row["clip_start"])
        data.clip_end = float(row["clip_end"])
        bpy.context.view_layer.update()

        png = output / f"V{index}.png"
        scene.render.filepath = str(png)
        bpy.ops.render.render(write_still=True)
        if not png.is_file() or png.stat().st_size <= 0:
            raise RuntimeError("SOURCE_APPEARANCE_REPLAY_RASTER_MISSING")
        rendered = bpy.data.images.load(str(png), check_existing=False)
        try:
            if tuple(map(int, rendered.size)) != (resolution, resolution):
                raise RuntimeError("SOURCE_APPEARANCE_REPLAY_RASTER_LAYOUT")
            pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
            rendered.pixels.foreach_get(pixels)
            alpha = pixels.reshape(resolution, resolution, 4)[::-1, :, 3]
            mask = np.asarray(alpha > 1.0e-6, dtype=np.uint8)
        finally:
            bpy.data.images.remove(rendered)
        binary = output / f"V{index}.mask.bin"
        binary.write_bytes(mask.tobytes())
        views.append({
            "view_index": index,
            "raster_sha256": _sha(png),
            "mask_sha256": _sha(binary),
            "foreground_pixel_count": int(mask.sum()),
        })
    return views


def main() -> None:
    a = _args()
    candidate = _load(a.candidate)
    bundle = _load(a.camera_bundle)
    if candidate.get("schema") != "RealSaS.Controlled8ViewSourceRestCandidate.v2" or candidate.get("subject_id") != a.subject_id:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CANDIDATE_BINDING")
    if candidate.get("material_mode") != "unlit_preserve_alpha":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_MATERIAL_MODE")
    if candidate.get("training_admission_claimed") is not False or candidate.get("product_pass_claimed") is not False:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CANDIDATE_AUTHORITY")
    if bundle.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CAMERA_SCHEMA")
    if _sha(a.camera_bundle) != str(candidate.get("camera_bundle", {}).get("sha256")):
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_CAMERA_HASH_DRIFT")
    current_blend = Path(bpy.data.filepath).resolve()
    if not current_blend.is_file() or _sha(current_blend) != a.expected_blend_sha256:
        raise RuntimeError("SOURCE_APPEARANCE_REPLAY_BLEND_HASH_DRIFT")

    a.output_dir.mkdir(parents=True, exist_ok=False)
    meshes, _points, _by_object = orbit.full_subject()
    materials = _materials_from_render_meshes(meshes)
    replay_receipts = _replay_transform(materials, list(candidate.get("material_transform") or ()))
    views = _render_masks(bundle, a.output_dir)
    receipt = {
        "schema": SCHEMA,
        "subject_id": a.subject_id,
        "candidate_sha256": _sha(a.candidate),
        "camera_bundle_sha256": _sha(a.camera_bundle),
        "packed_source_scene_blend_sha256": a.expected_blend_sha256,
        "blender_version": str(bpy.app.version_string),
        "material_policy_source": "SEALED_SOURCE_REST_RENDER_CANDIDATE__NO_PACKED_BLEND_MODE_INFERENCE",
        "material_replay": replay_receipts,
        "views": views,
        "source_lit_appearance_parity_claimed": False,
        "rgb_equivalence_claimed": False,
        "sampler_qualified": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "authority": "SOURCE_APPEARANCE_MASK_REPLAY_CANDIDATE_ONLY",
    }
    (a.output_dir / "SOURCE_APPEARANCE_POLICY_REPLAY_V1.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"material_replay", "views"}}, sort_keys=True))


if __name__ == "__main__":
    main()
