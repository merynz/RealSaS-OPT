from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF, Y_TO_Z
from tools.corpus.fit8_f_qualified_fa_bundle_v2 import canonical_source_carrier_sha256_v1
from tools.corpus.fit8_render_vertex_equivalence_v1 import (
    RenderVertexEvidenceV1,
    qualify_render_vertex_equivalence_v1,
)

SCHEMA_VERSION = "RealSaS.FIT8GLTFStaticCarrier.v1"
OWNER_SCOPE = "DIAGNOSTIC_DOMINANT_NORMALIZED_SKIN_JOINT"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_skin_signature(
    source: SourceGLTF,
    primitive: dict[str, Any],
    skin_index: int | None,
    vertex_count: int,
) -> tuple[list[str], list[dict[int, float]]]:
    attributes = primitive.get("attributes", {})
    joint_sets = sorted(
        int(name.split("_", 1)[1])
        for name in attributes
        if name.startswith("JOINTS_") and f"WEIGHTS_{name.split('_', 1)[1]}" in attributes
    )
    if skin_index is None or not joint_sets:
        return (["RIGID_UNSKINNED"] * vertex_count, [{} for _ in range(vertex_count)])
    joints = source.doc["skins"][skin_index]["joints"]
    accumulators: list[dict[int, float]] = [dict() for _ in range(vertex_count)]
    for set_id in joint_sets:
        joint_index = source.accessor(attributes[f"JOINTS_{set_id}"]).astype(np.int64)
        weight = source.accessor(attributes[f"WEIGHTS_{set_id}"]).astype(np.float64)
        if joint_index.shape != weight.shape or joint_index.shape[0] != vertex_count:
            raise ValueError("FIT8_GLTF_CARRIER_SKIN_SHAPE_INVALID")
        if np.any(joint_index < 0) or np.any(joint_index >= len(joints)):
            raise ValueError("FIT8_GLTF_CARRIER_JOINT_INDEX_OOB")
        for vertex_id in range(vertex_count):
            row = accumulators[vertex_id]
            for local_joint, value in zip(joint_index[vertex_id].tolist(), weight[vertex_id].tolist()):
                if float(value) == 0.0:
                    continue
                node_id = int(joints[int(local_joint)])
                row[node_id] = row.get(node_id, 0.0) + float(value)
    signatures: list[str] = []
    normalized_rows: list[dict[int, float]] = []
    for row in accumulators:
        total = float(sum(row.values()))
        if total <= 0.0:
            signatures.append("SKIN_ZERO_WEIGHT")
            normalized_rows.append({})
            continue
        normalized = {key: float(value / total) for key, value in sorted(row.items()) if value != 0.0}
        normalized_rows.append(normalized)
        payload = ";".join(f"{key}:{float(value).hex()}" for key, value in normalized.items())
        signatures.append("SKIN_NORMALIZED:" + payload)
    return signatures, normalized_rows


def _render_signature(
    source: SourceGLTF,
    primitive: dict[str, Any],
    local_vertex_id: int,
) -> str:
    attributes = primitive.get("attributes", {})
    digest = hashlib.sha256()
    digest.update(f"material:{primitive.get('material', -1)}".encode("utf-8"))
    for name in sorted(attributes):
        if name.startswith("JOINTS_") or name.startswith("WEIGHTS_") or name == "POSITION":
            continue
        array = np.asarray(source.accessor(attributes[name]))
        row = np.ascontiguousarray(array[local_vertex_id])
        digest.update(name.encode("utf-8"))
        digest.update(row.dtype.str.encode("ascii"))
        digest.update(row.tobytes())
    return digest.hexdigest()


def build_gltf_static_carrier_v1(source_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    source = SourceGLTF(source_path)
    raw_vertices: list[np.ndarray] = []
    raw_faces: list[np.ndarray] = []
    face_material: list[np.ndarray] = []
    face_owner: list[np.ndarray] = []
    evidence: list[RenderVertexEvidenceV1] = []
    render_vertex_offset = 0
    primitive_count = 0

    for node_index, node in enumerate(source.doc.get("nodes", [])):
        if "mesh" not in node:
            continue
        mesh = source.doc["meshes"][node["mesh"]]
        skin_index = node.get("skin")
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError("FIT8_GLTF_CARRIER_NON_TRIANGLE_PRIMITIVE")
            primitive_count += 1
            position = source.accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
            weights = node.get("weights", mesh.get("weights", []))
            for morph_weight, target in zip(weights, primitive.get("targets", [])):
                if "POSITION" in target:
                    position += float(morph_weight) * source.accessor(target["POSITION"])
            world = source.world(node_index)
            transformed = (position @ world[:3, :3].T + world[:3, 3]) @ Y_TO_Z.T
            indices = (
                source.accessor(primitive["indices"]).ravel()
                if "indices" in primitive
                else np.arange(len(position), dtype=np.int64)
            ).astype(np.int64).reshape(-1, 3)

            deformation_signature, normalized_skin = _canonical_skin_signature(
                source,
                primitive,
                skin_index,
                len(position),
            )
            for local_vertex_id, xyz in enumerate(transformed.tolist()):
                evidence.append(
                    RenderVertexEvidenceV1(
                        render_vertex_id=render_vertex_offset + local_vertex_id,
                        rest_xyz=tuple(float(v) for v in xyz),
                        deformation_signature=(
                            f"NODE:{node_index}:" + deformation_signature[local_vertex_id]
                            if skin_index is None
                            else f"SKIN:{skin_index}:" + deformation_signature[local_vertex_id]
                        ),
                        render_signature=_render_signature(source, primitive, local_vertex_id),
                    )
                )

            owners = np.full(len(indices), -1, dtype=np.int32)
            for face_index, face in enumerate(indices.tolist()):
                aggregate: dict[int, float] = {}
                for local_vertex_id in face:
                    for joint_node_id, value in normalized_skin[int(local_vertex_id)].items():
                        aggregate[joint_node_id] = aggregate.get(joint_node_id, 0.0) + float(value) / 3.0
                if aggregate:
                    owners[face_index] = max(
                        aggregate.items(), key=lambda item: (item[1], -item[0])
                    )[0]
            raw_vertices.append(transformed)
            raw_faces.append(indices + render_vertex_offset)
            face_material.append(np.full(len(indices), primitive.get("material", -1), dtype=np.int32))
            face_owner.append(owners)
            render_vertex_offset += len(position)

    if not raw_vertices or not raw_faces:
        raise ValueError("FIT8_GLTF_CARRIER_EMPTY")
    raw_xyz = np.vstack(raw_vertices).astype(np.float64, copy=False)
    raw_tri = np.vstack(raw_faces).astype(np.int64, copy=False)
    material = np.concatenate(face_material).astype(np.int32, copy=False)
    owner = np.concatenate(face_owner).astype(np.int32, copy=False)
    report = qualify_render_vertex_equivalence_v1(evidence)
    if not report.mechanical_conflict_free:
        raise ValueError(
            f"FIT8_GLTF_CARRIER_MECHANICAL_POSITION_CONFLICT:{report.mechanically_conflicted_position_count}"
        )
    mapping = np.asarray(report.render_to_geometric, dtype=np.int64)
    geometric_xyz = np.asarray([row.rest_xyz for row in report.groups], dtype=np.float64)
    geometric_tri = mapping[raw_tri]
    if np.any(
        (geometric_tri[:, 0] == geometric_tri[:, 1])
        | (geometric_tri[:, 1] == geometric_tri[:, 2])
        | (geometric_tri[:, 2] == geometric_tri[:, 0])
    ):
        raise ValueError("FIT8_GLTF_CARRIER_EQUIVALENCE_COLLAPSED_FACE")

    arrays = {
        "vertices": geometric_xyz,
        "faces": geometric_tri.astype(np.int64),
        "face_material_id": material,
        "face_owner_id": owner,
        "render_to_geometric": mapping,
    }
    manifest = {
        "schema": SCHEMA_VERSION,
        "source_sha256": _sha256(source_path),
        "source_scope": "RAW_BIND_MESH__DEFAULT_POSE_IS_NOT_BIND_ASSUMED",
        "raw_render_vertex_count": int(len(raw_xyz)),
        "geometric_vertex_count": int(len(geometric_xyz)),
        "face_count": int(len(geometric_tri)),
        "primitive_count": int(primitive_count),
        "material_region_count": int(len(set(int(v) for v in material.tolist() if int(v) >= 0))),
        "diagnostic_mechanical_owner_count": int(len(set(int(v) for v in owner.tolist() if int(v) >= 0))),
        "owner_scope": OWNER_SCOPE,
        "equivalence_schema": report.schema_version,
        "coincident_position_count": int(report.coincident_position_count),
        "mechanically_conflicted_position_count": int(report.mechanically_conflicted_position_count),
        "render_seam_group_count": int(report.render_seam_group_count),
        "source_carrier_sha256": canonical_source_carrier_sha256_v1(geometric_xyz, geometric_tri),
        "native_motion_names": [
            row.get("name", str(index))
            for index, row in enumerate(source.doc.get("animations", []))
        ],
        "claims": {
            "material_id_source_exact": True,
            "mechanical_owner_diagnostic_only": True,
            "semantic_owner_authority": False,
            "default_pose_is_bind_assumed": False,
        },
    }
    return manifest, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    manifest, arrays = build_gltf_static_carrier_v1(args.source)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


__all__ = ["OWNER_SCOPE", "SCHEMA_VERSION", "build_gltf_static_carrier_v1"]

if __name__ == "__main__":
    main()
