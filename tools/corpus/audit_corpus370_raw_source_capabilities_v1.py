#!/usr/bin/env python3
from __future__ import annotations

"""Measure raw authored capabilities for the exact Corpus370 source bundle.

This is a source census, never a family-admission court. It deliberately avoids
Blender, learned models, historical training flags and imported teacher state.
The output answers only what is authored in the immutable source bytes: glTF/GLB
skins/joints/weight attributes/animations and FBX object/deformer/animation
presence. Current independent family courts are still required before any cell
in the subject-family authority matrix can become PASS.
"""

import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.Corpus370RawSourceCapabilityCensus.v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    require(data[:4] == b"glTF", "CORPUS370_GLB_MAGIC")
    _, version, total = struct.unpack_from("<4sII", data, 0)
    require(version == 2 and total <= len(data), "CORPUS370_GLB_HEADER")
    offset = 12
    while offset + 8 <= total:
        length, kind = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk = data[offset : offset + length]
        offset += length
        if kind == b"JSON":
            return json.loads(chunk.rstrip(b" \t\r\n\x00"))
    raise ValueError("CORPUS370_GLB_JSON_CHUNK")


def _scan_gltf(path: Path, source_format: str) -> dict[str, Any]:
    doc = _glb_json(path) if source_format == ".glb" else json.loads(path.read_text(errors="strict"))
    meshes = doc.get("meshes") or []
    nodes = doc.get("nodes") or []
    skins = doc.get("skins") or []
    animations = doc.get("animations") or []
    primitives = [primitive for mesh in meshes for primitive in (mesh.get("primitives") or [])]
    skinned_mesh_nodes = [node for node in nodes if "mesh" in node and "skin" in node]
    unique_joints = sorted({int(joint) for skin in skins for joint in (skin.get("joints") or [])})
    weighted_primitives = [
        primitive
        for primitive in primitives
        if {"JOINTS_0", "WEIGHTS_0"}.issubset((primitive.get("attributes") or {}).keys())
    ]
    return {
        "parser": "RAW_GLTF2_JSON_V1",
        "mesh_count": len(meshes),
        "primitive_count": len(primitives),
        "node_count": len(nodes),
        "skin_count": len(skins),
        "joint_count_unique": len(unique_joints),
        "skinned_mesh_node_count": len(skinned_mesh_nodes),
        "weighted_primitive_count": len(weighted_primitives),
        "all_primitives_weighted": bool(primitives) and len(weighted_primitives) == len(primitives),
        "animation_count": len(animations),
        "animation_channel_count": sum(len(animation.get("channels") or []) for animation in animations),
        "material_count": len(doc.get("materials") or []),
        "image_count": len(doc.get("images") or []),
        "texture_count": len(doc.get("textures") or []),
        "has_rig": bool(skins and unique_joints),
        "has_skin_weights": bool(weighted_primitives),
        "has_animation": bool(animations),
        "generator": (doc.get("asset") or {}).get("generator"),
    }


_FBX_SCALAR = {
    "Y": ("<h", 2),
    "C": ("<?", 1),
    "I": ("<i", 4),
    "F": ("<f", 4),
    "D": ("<d", 8),
    "L": ("<q", 8),
}
_FBX_ARRAY = set("fdlibc")


def _fbx_property(data: bytes, offset: int) -> tuple[Any, int]:
    kind = chr(data[offset])
    offset += 1
    if kind in _FBX_SCALAR:
        fmt, size = _FBX_SCALAR[kind]
        return struct.unpack_from(fmt, data, offset)[0], offset + size
    if kind in {"S", "R"}:
        length = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        raw = data[offset : offset + length]
        offset += length
        return (raw.decode("utf-8", "replace") if kind == "S" else None), offset
    if kind in _FBX_ARRAY:
        length, encoding, compressed_length = struct.unpack_from("<III", data, offset)
        offset += 12 + compressed_length
        return {"array_len": length, "encoding": encoding, "compressed_len": compressed_length}, offset
    raise ValueError("CORPUS370_FBX_PROPERTY_" + kind)


def _scan_fbx(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    require(data.startswith(b"Kaydara FBX Binary  \x00\x1a\x00"), "CORPUS370_FBX_BINARY_REQUIRED")
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    null_len = 25 if wide else 13
    counts: collections.Counter[str] = collections.Counter()
    model_types: collections.Counter[str] = collections.Counter()
    deformer_types: collections.Counter[str] = collections.Counter()

    def parse_node(offset: int) -> int:
        if wide:
            end, property_count, property_bytes = struct.unpack_from("<QQQ", data, offset)
            name_len = data[offset + 24]
            header = 25
        else:
            end, property_count, property_bytes = struct.unpack_from("<III", data, offset)
            name_len = data[offset + 12]
            header = 13
        if end == 0:
            return offset + null_len
        require(end <= len(data), "CORPUS370_FBX_NODE_END")
        name_start = offset + header
        name = data[name_start : name_start + name_len].decode("utf-8", "replace")
        property_offset = name_start + name_len
        values = []
        try:
            for index in range(property_count):
                value, property_offset = _fbx_property(data, property_offset)
                if index < 4:
                    values.append(value)
        except Exception:
            values = []
            property_offset = name_start + name_len + property_bytes
        counts[name] += 1
        if name == "Model" and len(values) >= 3:
            model_types[str(values[2])] += 1
        if name == "Deformer" and len(values) >= 3:
            deformer_types[str(values[2])] += 1
        child = name_start + name_len + property_bytes
        limit = end - null_len
        while child < limit:
            next_child = parse_node(child)
            require(next_child > child, "CORPUS370_FBX_CHILD_PROGRESS")
            child = next_child
        return end

    offset = 27
    while offset + null_len <= len(data):
        if data[offset : offset + null_len] == b"\x00" * null_len:
            break
        next_offset = parse_node(offset)
        require(next_offset > offset, "CORPUS370_FBX_TOPLEVEL_PROGRESS")
        offset = next_offset

    return {
        "parser": "RAW_FBX_BINARY_OBJECT_CENSUS_V1",
        "fbx_version": version,
        "geometry_count": counts["Geometry"],
        "model_count": counts["Model"],
        "model_types": dict(sorted(model_types.items())),
        "deformer_count": counts["Deformer"],
        "deformer_types": dict(sorted(deformer_types.items())),
        "skin_deformer_count": deformer_types.get("Skin", 0),
        "cluster_deformer_count": deformer_types.get("Cluster", 0),
        "animation_stack_count": counts["AnimationStack"],
        "animation_layer_count": counts["AnimationLayer"],
        "animation_curve_node_count": counts["AnimationCurveNode"],
        "animation_curve_count": counts["AnimationCurve"],
        "material_count": counts["Material"],
        "texture_count": counts["Texture"],
        "video_count": counts["Video"],
        "has_rig": bool(model_types.get("LimbNode", 0) or deformer_types.get("Skin", 0)),
        "has_skin_weights": bool(deformer_types.get("Skin", 0) and deformer_types.get("Cluster", 0)),
        "has_animation": bool(counts["AnimationStack"] and counts["AnimationCurveNode"]),
    }


def audit(bundle_root: Path, expected_subject_count: int = 370) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    manifest_path = bundle_root / "SOURCE_BUNDLE_MANIFEST_V1.json"
    require(manifest_path.is_file(), "CORPUS370_SOURCE_BUNDLE_MANIFEST_REQUIRED")
    manifest = json.loads(manifest_path.read_text())
    subjects = list(manifest.get("subjects") or ())
    require(len(subjects) == expected_subject_count, "CORPUS370_RAW_CENSUS_SUBJECT_COUNT")
    rows = []
    for raw in subjects:
        source = bundle_root / str(raw["source_relative_path"])
        require(source.is_file(), "CORPUS370_RAW_CENSUS_SOURCE_MISSING")
        require(sha256(source) == raw["source_sha256"], "CORPUS370_RAW_CENSUS_SOURCE_HASH")
        source_format = str(raw["source_format"])
        if source_format in {".gltf", ".glb"}:
            measurement = _scan_gltf(source, source_format)
        elif source_format == ".fbx":
            measurement = _scan_fbx(source)
        else:
            raise ValueError("CORPUS370_RAW_CENSUS_SOURCE_FORMAT:" + source_format)
        rows.append(
            {
                "subject_id": raw["subject_id"],
                "origin": raw.get("origin"),
                "source_id": raw.get("source_id"),
                "source_format": source_format,
                "source_relative_path": raw["source_relative_path"],
                "source_sha256": raw["source_sha256"],
                "raw_capabilities": measurement,
            }
        )
    summary = {}
    for key in ("has_rig", "has_skin_weights", "has_animation"):
        counter = collections.Counter(row["raw_capabilities"][key] for row in rows)
        summary[key] = {str(k).lower(): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}
    return {
        "schema": SCHEMA,
        "status": "MEASUREMENT_ONLY__NOT_FAMILY_AUTHORITY",
        "subject_count": len(rows),
        "source_bundle_manifest_sha256": sha256(manifest_path),
        "subjects": sorted(rows, key=lambda row: row["subject_id"]),
        "summary": summary,
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-subject-count", type=int, default=370)
    args = parser.parse_args()
    result = audit(args.bundle_root, args.expected_subject_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": result["schema"], "subject_count": result["subject_count"], "summary": result["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
