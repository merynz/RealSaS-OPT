"""Source-scene preparation and internal consistency, without model/rig authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
from urllib.parse import unquote, urlparse

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_and_chunks(source: Path) -> tuple[dict, list[tuple[int, bytes]]]:
    data = source.read_bytes()
    if source.suffix.lower() == ".gltf":
        return json.loads(data.decode("utf-8-sig")), []
    _require(source.suffix.lower() == ".glb", "SOURCE_GLTF_FORMAT")
    _require(len(data) >= 20, "SOURCE_GLB_TRUNCATED")
    magic, version, size = struct.unpack_from("<III", data)
    _require(magic == 0x46546C67 and version == 2 and size == len(data), "SOURCE_GLB_HEADER")
    chunks = []
    offset = 12
    while offset < len(data):
        _require(offset + 8 <= len(data), "SOURCE_GLB_CHUNK_HEADER")
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        _require(length % 4 == 0 and offset + length <= len(data), "SOURCE_GLB_CHUNK_SIZE")
        chunks.append((kind, data[offset:offset + length]))
        offset += length
    _require(bool(chunks) and chunks[0][0] == 0x4E4F534A, "SOURCE_GLB_JSON_FIRST")
    _require(sum(kind == 0x4E4F534A for kind, _ in chunks) == 1, "SOURCE_GLB_DUPLICATE_JSON")
    return json.loads(chunks[0][1].rstrip(b" \x00").decode()), chunks[1:]


def prepare_animation_free_gltf(source: Path, destination: Path) -> dict:
    """Copy a declared derivative; preserve nodes, skins, materials and binary data.

    Removing animation before import avoids mixing bind/default object transforms
    with a selected action or NLA evaluation. This is not a pose repair or source
    admission; original bytes are never changed.
    """
    source = source.resolve()
    destination = destination.resolve()
    _require(not destination.is_relative_to(source.parent),
             "REST_DERIVATIVE_MUST_BE_OUTSIDE_RAW_SOURCE_DIRECTORY")
    _require(source.suffix.lower() == destination.suffix.lower(), "REST_DERIVATIVE_FORMAT")
    document, other_chunks = _json_and_chunks(source)
    _require(document.get("asset", {}).get("version") == "2.0", "SOURCE_GLTF_VERSION")
    animations = document.pop("animations", [])
    dependencies = []
    reserved = {destination.relative_to(destination.parent).as_posix()}
    copied = {}
    for item in [*document.get("buffers", []), *document.get("images", [])]:
        uri = item.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        parsed = urlparse(uri)
        _require(not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment,
                 "SOURCE_EXTERNAL_RESOURCE_MUST_BE_LOCAL_DECLARED_BYTES")
        name = unquote(parsed.path)
        relative = PurePosixPath(name)
        _require(not relative.is_absolute() and "\\" not in name and ":" not in name,
                 "SOURCE_EXTERNAL_RESOURCE_PATH")
        _require(all(part not in ("", ".", "..") for part in name.split("/")),
                 "SOURCE_EXTERNAL_RESOURCE_TRAVERSAL")
        _require(name not in reserved, "REST_DERIVATIVE_RESOURCE_COLLISION")
        original = (source.parent / name).resolve()
        _require(original.is_relative_to(source.parent) and original.is_file(),
                 "SOURCE_EXTERNAL_RESOURCE_MISSING_OR_ESCAPED")
        target = destination.parent / name
        _require(target.resolve().is_relative_to(destination.parent), "REST_RESOURCE_TARGET_ESCAPE")
        current_sha = sha256(original)
        if name not in copied:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            _require(sha256(target) == current_sha, "REST_RESOURCE_COPY_HASH")
            copied[name] = current_sha
            dependencies.append({"uri": uri, "sha256": current_sha,
                                 "byte_size": original.stat().st_size})
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
    if source.suffix.lower() == ".gltf":
        destination.write_bytes(encoded)
    else:
        encoded += b" " * (-len(encoded) % 4)
        body = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
        body += b"".join(struct.pack("<II", len(chunk), kind) + chunk for kind, chunk in other_chunks)
        destination.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
    return {
        "schema": "RealSaS.SourceRestImportDerivative.v2",
        "source_sha256": sha256(source), "derivative_sha256": sha256(destination),
        "removed_animation_count": len(animations),
        "source_animation_names": [row.get("name") for row in animations],
        "operation": "REMOVE_ANIMATION_ONLY__PRESERVE_DEFAULT_NODE_AND_MORPH_STATE",
        "external_resources": dependencies, "raw_source_mutated": False,
        "source_bind_parity_claimed": False,
    }


def prepare_bind_rest_gltf(source: Path, destination: Path) -> dict:
    """Declare a hierarchy anchor for importers; retain authored inverse binds.

    An absent optional skin.skeleton is legal glTF. Some importers otherwise
    discard the common root's inverse-bind translation. The hint below locates
    an existing ancestor; no node, joint, buffer, weight or transform is repaired.
    """
    receipt = prepare_animation_free_gltf(source, destination)
    document, chunks = _json_and_chunks(destination)
    nodes = document.get("nodes", [])
    parents = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            _require(0 <= child < len(nodes) and child not in parents,
                     "SOURCE_GLTF_INVALID_PARENT_GRAPH")
            parents[child] = index

    def chain(index):
        result = []
        while True:
            _require(index not in result, "SOURCE_GLTF_PARENT_CYCLE")
            result.append(index)
            if index not in parents:
                return result
            index = parents[index]

    for index in range(len(nodes)):
        chain(index)
    hints = []
    for index, skin in enumerate(document.get("skins", [])):
        joints = skin["joints"]
        _require(bool(joints) and all(0 <= j < len(nodes) for j in joints),
                 "SOURCE_GLTF_SKIN_JOINTS")
        paths = [chain(j) for j in joints]
        if "skeleton" in skin:
            anchor = skin["skeleton"]
            _require(all(anchor in p for p in paths), "SOURCE_GLTF_SKELETON_NOT_ANCESTOR")
            operation = "AUTHORED_ANCHOR_RETAINED"
        else:
            common = [j for j in paths[0] if all(j in p for p in paths)]
            _require(bool(common), "SOURCE_GLTF_SKIN_HAS_NO_COMMON_ANCESTOR")
            anchor = common[0]
            if anchor in joints and anchor in parents:
                anchor = parents[anchor]
            skin["skeleton"] = anchor
            operation = "EXISTING_ANCESTOR_HINT_DECLARED"
        hints.append({"skin_index": index, "skeleton_node": anchor, "operation": operation})
    encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
    if destination.suffix.lower() == ".gltf":
        destination.write_bytes(encoded)
    else:
        encoded += b" " * (-len(encoded) % 4)
        body = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
        body += b"".join(struct.pack("<II", len(c), k) + c for k, c in chunks)
        destination.write_bytes(struct.pack("<III", 0x46546C67, 2, len(body) + 12) + body)
    receipt.update(operation="ANIMATION_FREE_BIND_IMPORT__EXISTING_ANCESTOR_HINT",
                   skeleton_hints=hints, derivative_sha256=sha256(destination))
    return receipt


def _matrix(value: list, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.shape == (4, 4) and bool(np.isfinite(matrix).all()), f"SCENE_MATRIX:{label}")
    _require(bool(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)),
             f"SCENE_AFFINE_MATRIX:{label}")
    _require(np.linalg.matrix_rank(matrix[:3, :3]) == 3, f"SCENE_SINGULAR_MATRIX:{label}")
    return matrix


def validate_imported_scene_snapshot(snapshot: dict, *, tolerance: float = 1e-6) -> dict:
    """Detect stale/tampered imported bindings; never prove source/import fidelity."""
    _require(snapshot.get("schema") == "RealSaS.ImportedSourceSceneSnapshot.v2", "SCENE_SCHEMA")
    _require(np.isfinite(tolerance) and tolerance > 0, "SCENE_TOLERANCE")
    nodes = snapshot["nodes"]
    _require(bool(nodes), "SCENE_EMPTY")
    index = {row["node_id"]: row for row in nodes}
    _require(len(index) == len(nodes), "SCENE_DUPLICATE_NODE_ID")
    norm = _matrix(snapshot["source_world_to_canonical"], "normalization")
    inverse = _matrix(snapshot["canonical_to_source_world"], "normalization_inverse")
    _require(np.allclose(norm @ inverse, np.eye(4), atol=tolerance, rtol=0), "SCENE_NORMALIZATION_INVERSE")
    gram = norm[:3, :3].T @ norm[:3, :3]
    _require(np.allclose(gram / np.trace(gram), np.eye(3) / 3, atol=tolerance, rtol=0),
             "SCENE_NONUNIFORM_CANONICAL_NORMALIZATION")
    _require(np.linalg.slogdet(norm[:3, :3])[0] > 0, "SCENE_REFLECTED_CANONICAL_NORMALIZATION")
    max_error = 0.0
    for row in nodes:
        world = _matrix(row["node_to_source_world"], row["node_id"])
        local = _matrix(row["effective_node_to_parent"], row["node_id"] + ":local")
        parent_id = row["parent_node_id"]
        seen = {row["node_id"]}
        current = parent_id
        while current is not None:
            _require(current in index, "SCENE_PARENT_MISSING")
            _require(current not in seen, "SCENE_PARENT_CYCLE")
            seen.add(current)
            current = index[current]["parent_node_id"]
        parent = np.eye(4) if parent_id is None else _matrix(index[parent_id]["node_to_source_world"], parent_id)
        error = float(np.max(np.abs(world - parent @ local)))
        max_error = max(max_error, error)
        _require(error <= tolerance * max(1., float(np.max(np.abs(world)))), "SCENE_PARENT_WORLD_DRIFT")
        canonical = _matrix(row["node_to_canonical"], row["node_id"] + ":canonical")
        _require(np.allclose(canonical, norm @ world, atol=tolerance, rtol=tolerance),
                 "SCENE_CANONICAL_BINDING_DRIFT")
        _require(row["active_action_count"] == 0 and row["active_nla_track_count"] == 0,
                 "SCENE_REST_HAS_LIVE_ANIMATION")
        _require(row.get("driver_count", 0) == 0, "SCENE_REST_HAS_LIVE_DRIVERS")
    return {
        "schema": "RealSaS.ImportedSceneInternalConsistency.v2", "node_count": len(nodes),
        "max_parent_world_residual": max_error, "normalization_uniform": True,
        "animation_free_rest": True, "scope": "IMPORTED_SNAPSHOT_INTERNAL_CONSISTENCY_ONLY",
        "independent_source_bind_parity_claimed": False, "training_admission_claimed": False,
    }
