#!/usr/bin/env python3
"""Audit the sealed Quaternius GLTF subset before Truth139 apparatus materialization.

This is a raw-byte capability audit, not admission and not training authority.  It reproduces the
historical 395 -> 143 rigged+animated -> 139 deduplicated census from immutable source bytes, emits
strong leakage signatures, excludes known development/regression subjects, and classifies which
FIT8-derived truth families each remaining subject can even attempt.

No render, Blender import, Stage47/48 measurement, feature extraction, split, or optimizer step is
performed here.  Those remain later Platform-owned gates.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF

ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCHEMA = "RealSaS.Truth139RawCapabilityAudit.v1"
SUBJECT_SCHEMA = "RealSaS.Truth139RawSubjectCapability.v1"
EXPECTED_CORPUS_ROOT = "2f15bf84694417cee3862c39a89e04bfa131f51331d6da5880739d40b7eab5ab"
EXPECTED_GLTF_FILES = 395
EXPECTED_RIGGED_ANIMATED_FILES = 143
EXPECTED_DEDUP_SUBJECTS = 139


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest_json(value: object) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _array_digest(value: np.ndarray, dtype: str) -> str:
    array = np.asarray(value).astype(dtype, copy=False)
    h = hashlib.sha256()
    h.update(str(tuple(array.shape)).encode("ascii"))
    h.update(array.tobytes())
    return h.hexdigest()


def _norm_asset(value: str) -> str:
    text = "".join(ch.lower() for ch in str(value) if ch.isalnum())
    for suffix in ("textured", "flat", "material", "materials"):
        if text.endswith(suffix) and len(text) > len(suffix) + 2:
            text = text[: -len(suffix)]
    return text


def _pack_identity(relative: Path) -> tuple[str, str]:
    parts = relative.as_posix().split("/")
    if "quaternius" in parts:
        index = parts.index("quaternius")
        pack = parts[index + 1] if index + 1 < len(parts) else ""
        return "quaternius", pack
    return "unknown", ""


def _development_aliases(path: Path) -> tuple[set[str], set[tuple[str, str, str]]]:
    payload = json.loads(path.read_text())
    shas: set[str] = set()
    aliases: set[tuple[str, str, str]] = set()
    for row in payload.get("selected") or ():
        source = dict(row.get("source") or {})
        digest = str(source.get("sha256") or "")
        if len(digest) == 64:
            shas.add(digest)
        relative = Path(str(source.get("corpus_relative_path") or ""))
        aliases.add((str(row.get("creator") or "").lower(), str(row.get("pack_id") or "").lower(), _norm_asset(relative.stem)))
    return shas, aliases


def _view_bytes(gltf: SourceGLTF, view_index: int) -> bytes:
    row = gltf.doc["bufferViews"][int(view_index)]
    start = int(row.get("byteOffset", 0))
    length = int(row["byteLength"])
    data = gltf.buffers[int(row["buffer"])]
    return bytes(data[start : start + length])


def _image_bytes(gltf: SourceGLTF, image_index: int) -> bytes:
    row = gltf.doc["images"][int(image_index)]
    if "bufferView" in row:
        return _view_bytes(gltf, int(row["bufferView"]))
    uri = str(row.get("uri") or "")
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    if not uri:
        raise ValueError("TRUTH139_IMAGE_SOURCE_MISSING")
    path = (gltf.path.parent / unquote(uri)).resolve()
    if not path.is_relative_to(gltf.path.parent.resolve()) or not path.is_file():
        raise ValueError("TRUTH139_IMAGE_RESOURCE_UNRESOLVED")
    return path.read_bytes()


def _material_tokens(gltf: SourceGLTF) -> tuple[list[str], bool, list[str]]:
    doc = gltf.doc
    image_hashes: dict[int, str] = {}
    errors: list[str] = []
    for index in range(len(doc.get("images", []))):
        try:
            image_hashes[index] = _sha_bytes(_image_bytes(gltf, index))
        except Exception as exc:
            errors.append(f"IMAGE_{index}:{type(exc).__name__}:{exc}")

    texture_to_image = {
        index: int(row["source"])
        for index, row in enumerate(doc.get("textures", []))
        if "source" in row
    }

    def normalize(value):
        if isinstance(value, dict):
            out = {}
            for key, item in sorted(value.items()):
                if key == "name":
                    continue
                if key == "index" and isinstance(item, int):
                    image_index = texture_to_image.get(int(item))
                    out[key] = {
                        "texture_index": int(item),
                        "image_sha256": image_hashes.get(image_index, "UNRESOLVED") if image_index is not None else "NO_SOURCE",
                    }
                else:
                    out[key] = normalize(item)
            return out
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    tokens = ["MATCFG:" + _digest_json(normalize(row)) for row in doc.get("materials", [])]
    tokens.extend("IMAGE:" + digest for digest in sorted(set(image_hashes.values())))
    if not tokens:
        # Do not connect every material-less asset into one leakage component.
        tokens = ["NO_MATERIAL_SOURCE:" + _sha_file(gltf.path)]
    return sorted(set(tokens)), not errors, errors


def _primitive_rows(gltf: SourceGLTF) -> list[dict]:
    rows: list[dict] = []
    for node_index, node in enumerate(gltf.doc.get("nodes", [])):
        if "mesh" not in node:
            continue
        mesh_index = int(node["mesh"])
        mesh = gltf.doc["meshes"][mesh_index]
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            mode = int(primitive.get("mode", 4))
            attrs = dict(primitive.get("attributes") or {})
            if "POSITION" not in attrs:
                rows.append({"node_index": node_index, "mesh_index": mesh_index, "primitive_index": primitive_index, "mode": mode, "error": "POSITION_MISSING"})
                continue
            positions = gltf.accessor(int(attrs["POSITION"])).astype(np.float64)
            if "indices" in primitive:
                indices = gltf.accessor(int(primitive["indices"])).reshape(-1).astype(np.int64)
            else:
                indices = np.arange(len(positions), dtype=np.int64)
            faces = indices.reshape(-1, 3) if mode == 4 and len(indices) % 3 == 0 else np.empty((0, 3), dtype=np.int64)
            rows.append(
                {
                    "node_index": node_index,
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    "mode": mode,
                    "skin_index": node.get("skin"),
                    "material_index": primitive.get("material"),
                    "attributes": attrs,
                    "positions": positions,
                    "faces": faces,
                    "position_count": int(len(positions)),
                }
            )
    return rows


def _skeleton_tokens(gltf: SourceGLTF) -> list[str]:
    nodes = gltf.doc.get("nodes", [])
    tokens = []
    for skin_index, skin in enumerate(gltf.doc.get("skins", [])):
        joints = [int(value) for value in skin.get("joints", [])]
        ordinal = {node_index: index for index, node_index in enumerate(joints)}
        rows = []
        for node_index in joints:
            node = nodes[node_index]
            parent = gltf.parents.get(node_index)
            rows.append(
                {
                    "name": str(node.get("name") or ""),
                    "parent_joint": ordinal.get(parent, -1),
                    "children_joint_ordinals": sorted(ordinal[child] for child in node.get("children", []) if child in ordinal),
                }
            )
        tokens.append("SKEL:" + _digest_json({"joint_count": len(joints), "rows": rows}))
    return sorted(tokens)


def _topology_tokens(primitives: list[dict]) -> list[str]:
    tokens = []
    for row in primitives:
        if int(row.get("mode", -1)) != 4 or "faces" not in row:
            continue
        faces = np.asarray(row["faces"], dtype=np.int64)
        payload = {
            "vertex_count": int(row["position_count"]),
            "face_count": int(len(faces)),
            "indices_sha256": _array_digest(faces, "<i8"),
        }
        tokens.append("TOPO:" + _digest_json(payload))
    return sorted(tokens)


def _subject_identity(gltf: SourceGLTF, primitives: list[dict], skeleton_tokens: list[str]) -> str:
    primitive = []
    for row in primitives:
        if int(row.get("mode", -1)) != 4 or "positions" not in row:
            continue
        attrs = row["attributes"]
        skin_payload = {}
        for name in ("JOINTS_0", "WEIGHTS_0"):
            if name in attrs:
                value = gltf.accessor(int(attrs[name]))
                dtype = "<i8" if name.startswith("JOINTS") else "<f4"
                skin_payload[name] = _array_digest(value, dtype)
        primitive.append(
            {
                "position_sha256": _array_digest(np.asarray(row["positions"]), "<f8"),
                "faces_sha256": _array_digest(np.asarray(row["faces"]), "<i8"),
                "skin": skin_payload,
            }
        )
    return _digest_json({"primitive": primitive, "skeleton": skeleton_tokens})


def _animation_targets_rig(gltf: SourceGLTF) -> tuple[int, int]:
    joints = {int(joint) for skin in gltf.doc.get("skins", []) for joint in skin.get("joints", [])}
    channels = 0
    on_rig = 0
    for animation in gltf.doc.get("animations", []):
        for channel in animation.get("channels", []):
            channels += 1
            target = channel.get("target") or {}
            if int(target.get("node", -1)) in joints:
                on_rig += 1
    return channels, on_rig


def _skin_metrics(gltf: SourceGLTF, primitives: list[dict]) -> dict:
    zero_weight = 0
    vertex_count = 0
    max_influences = 0
    maximum_sum_error = 0.0
    skinned_primitives = 0
    complete_weight_primitives = 0
    for row in primitives:
        if row.get("skin_index") is None or "attributes" not in row:
            continue
        skinned_primitives += 1
        attrs = row["attributes"]
        if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
            continue
        joints = gltf.accessor(int(attrs["JOINTS_0"]))
        weights = gltf.accessor(int(attrs["WEIGHTS_0"])).astype(np.float64)
        if joints.shape != weights.shape or len(weights) != int(row["position_count"]):
            continue
        complete_weight_primitives += 1
        sums = weights.sum(axis=1)
        vertex_count += int(len(weights))
        zero_weight += int(np.count_nonzero(sums <= 1e-12))
        maximum_sum_error = max(maximum_sum_error, float(np.max(np.abs(sums - 1.0))) if len(sums) else 0.0)
        max_influences = max(max_influences, int(np.max(np.count_nonzero(weights > 1e-8, axis=1))) if len(weights) else 0)
    return {
        "skinned_primitive_count": skinned_primitives,
        "complete_joint_weight_primitive_count": complete_weight_primitives,
        "skinned_vertex_count": vertex_count,
        "zero_weight_vertex_count": zero_weight,
        "maximum_influences_per_vertex": max_influences,
        "maximum_weight_sum_error": maximum_sum_error,
    }


def _audit_file(path: Path, raw_root: Path, development_shas: set[str], development_aliases: set[tuple[str, str, str]]) -> dict:
    relative = path.relative_to(raw_root)
    source_sha = _sha_file(path)
    creator, pack = _pack_identity(relative)
    stem = _norm_asset(path.stem)
    gltf = SourceGLTF(path)
    primitives = _primitive_rows(gltf)
    triangle = [row for row in primitives if int(row.get("mode", -1)) == 4 and "positions" in row and len(row.get("faces", ())) > 0]
    finite = bool(triangle) and all(np.isfinite(np.asarray(row["positions"])).all() for row in triangle)
    nondegenerate = True
    for row in triangle:
        p = np.asarray(row["positions"], dtype=np.float64)
        f = np.asarray(row["faces"], dtype=np.int64)
        if len(f):
            cross = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
            if np.any(np.linalg.norm(cross, axis=1) <= 1e-12):
                nondegenerate = False
                break
    skins = list(gltf.doc.get("skins", []))
    animations = list(gltf.doc.get("animations", []))
    inverse_bind_complete = bool(skins) and all("inverseBindMatrices" in skin for skin in skins)
    skeleton = _skeleton_tokens(gltf)
    topology = _topology_tokens(triangle)
    material, materials_resolved, material_errors = _material_tokens(gltf)
    identity = _subject_identity(gltf, triangle, skeleton)
    skin = _skin_metrics(gltf, triangle)
    animation_channels, animation_channels_on_rig = _animation_targets_rig(gltf)
    joint_weight_complete = skin["skinned_primitive_count"] > 0 and skin["complete_joint_weight_primitive_count"] == skin["skinned_primitive_count"]
    rigged_animated = bool(skins) and bool(animations) and skin["skinned_primitive_count"] > 0
    appearance_ok = materials_resolved
    observation_ok = bool(triangle) and finite and appearance_ok
    normals_ok = bool(triangle) and finite and nondegenerate
    mechanics_ok = (
        rigged_animated
        and inverse_bind_complete
        and joint_weight_complete
        and skin["zero_weight_vertex_count"] == 0
        and skin["maximum_weight_sum_error"] <= 1e-4
        and animation_channels_on_rig > 0
    )
    tessa_ok = bool(triangle) and finite and all(int(row["position_count"]) <= 32768 and len(row["faces"]) <= 32768 for row in triangle)
    stage48_ok = observation_ok and mechanics_ok and appearance_ok
    full = observation_ok and normals_ok and mechanics_ok and stage48_ok and tessa_ok

    development_reason = None
    if source_sha in development_shas:
        development_reason = "FIT8_EXACT_SOURCE_SHA256"
    elif (creator.lower(), pack.lower(), stem) in development_aliases:
        development_reason = "FIT8_SAME_CREATOR_PACK_NORMALIZED_ASSET_STEM"
    elif stem == "knight":
        development_reason = "KNIGHT_NAME_GUARD__REQUIRES_MANUAL_SOURCE_IDENTITY_CONFIRMATION"

    blockers = []
    for name, ok in (
        ("OBSERVATION_FSTAR", observation_ok),
        ("SURFACE_NORMAL_TRUTH", normals_ok),
        ("STAGE47_MECHANICS", mechanics_ok),
        ("STAGE48_APPEARANCE_PRESENTATION", stage48_ok),
        ("TESSA_TOPOLOGY_TEACHER", tessa_ok),
    ):
        if not ok:
            blockers.append(name)
    if development_reason:
        blockers.append("DEVELOPMENT_OR_REGRESSION_EXCLUSION")

    return {
        "schema": SUBJECT_SCHEMA,
        "source": {
            "relative_path": relative.as_posix(),
            "sha256": source_sha,
            "size_bytes": path.stat().st_size,
            "format": path.suffix.upper().lstrip("."),
            "creator": creator,
            "pack_id": pack,
            "normalized_asset_stem": stem,
        },
        "raw_structure": {
            "mesh_primitive_count": len(primitives),
            "triangle_primitive_count": len(triangle),
            "vertex_count": int(sum(int(row.get("position_count", 0)) for row in triangle)),
            "face_count": int(sum(len(row.get("faces", ())) for row in triangle)),
            "skin_count": len(skins),
            "joint_counts": [len(skin.get("joints", [])) for skin in skins],
            "animation_count": len(animations),
            "animation_names": [str(row.get("name") or index) for index, row in enumerate(animations)],
            "animation_channel_count": animation_channels,
            "animation_channel_count_targeting_rig": animation_channels_on_rig,
            "inverse_bind_matrices_complete": inverse_bind_complete,
            **skin,
            "material_count": len(gltf.doc.get("materials", [])),
            "image_count": len(gltf.doc.get("images", [])),
            "material_resources_resolved": materials_resolved,
            "material_resolution_errors": material_errors,
            "finite_triangle_geometry": finite,
            "nondegenerate_surface": nondegenerate,
        },
        "lineage": {
            "source_sha256": [source_sha],
            "source_identity": ["SUBJECT:" + identity],
            "skeleton_hierarchy_signature": skeleton or ["NO_SKIN:" + source_sha],
            "topology_lineage_signature": topology or ["NO_TOPOLOGY:" + source_sha],
            "material_lineage_signature": material,
        },
        "subject_identity_signature": identity,
        "historical_rigged_animated_candidate": rigged_animated,
        "capabilities": {
            "OBSERVATION_FSTAR": observation_ok,
            "SURFACE_NORMAL_TRUTH": normals_ok,
            "STAGE47_MECHANICS": mechanics_ok,
            "STAGE48_APPEARANCE_PRESENTATION": stage48_ok,
            "TESSA_TOPOLOGY_TEACHER": tessa_ok,
            "FULL_TRUTH139_RAW_ELIGIBLE": full,
        },
        "development_exclusion": development_reason,
        "raw_eligibility_blockers": blockers,
        "platform_import_render_truth_admission_required": full and development_reason is None,
        "admitted_for_truth139": False,
        "training_authorized": False,
    }


def _choose_representative(rows: list[dict]) -> dict:
    def rank(row: dict):
        path = str(row["source"]["relative_path"]).lower()
        flat = 1 if "/flat/" in path or _norm_asset(Path(path).stem) != _norm_asset(Path(path).stem.replace("flat", "")) else 0
        textured = 0 if "textured" in path else 1
        return (flat, textured, path)
    return sorted(rows, key=rank)[0]


def audit(raw_root: Path, *, fit8_selection: Path, expected_corpus_root_sha256: str = EXPECTED_CORPUS_ROOT) -> dict:
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    development_shas, development_aliases = _development_aliases(fit8_selection)
    scan_root = raw_root / "raw" / "quaternius"
    if not scan_root.is_dir():
        scan_root = raw_root
    paths = sorted({*scan_root.rglob("*.gltf"), *scan_root.rglob("*.glb")})
    file_rows = []
    failures = []
    for index, path in enumerate(paths, 1):
        try:
            row = _audit_file(path, raw_root, development_shas, development_aliases)
            file_rows.append(row)
            print(f"[TRUTH139_RAW_AUDIT] {index}/{len(paths)} {row['source']['relative_path']} rigged_animated={int(row['historical_rigged_animated_candidate'])} full={int(row['capabilities']['FULL_TRUTH139_RAW_ELIGIBLE'])}", flush=True)
        except Exception as exc:
            failures.append({"relative_path": path.relative_to(raw_root).as_posix(), "error": f"{type(exc).__name__}:{exc}"})
            print(f"[TRUTH139_RAW_AUDIT_FAIL] {index}/{len(paths)} {path.relative_to(raw_root)} {type(exc).__name__}:{exc}", flush=True)

    candidates = [row for row in file_rows if row["historical_rigged_animated_candidate"]]
    grouped: dict[str, list[dict]] = {}
    for row in candidates:
        grouped.setdefault(row["subject_identity_signature"], []).append(row)

    subjects = []
    duplicate_groups = []
    for identity, group in sorted(grouped.items()):
        representative = _choose_representative(group)
        subject_id = "truth139_" + identity[:20]
        row = {
            **representative,
            "subject_id": subject_id,
            "subject_identity_signature": identity,
            "duplicate_source_files": [item["source"] for item in sorted(group, key=lambda item: item["source"]["relative_path"])],
            "duplicate_source_file_count": len(group),
            "raw_eligible_for_truth139": bool(representative["capabilities"]["FULL_TRUTH139_RAW_ELIGIBLE"] and representative["development_exclusion"] is None),
            "admitted_for_truth139": False,
        }
        subjects.append(row)
        if len(group) > 1:
            duplicate_groups.append({"subject_id": subject_id, "identity": identity, "source_files": [item["source"]["relative_path"] for item in group]})

    counts = {
        "gltf_glb_files_scanned": len(paths),
        "files_audited": len(file_rows),
        "file_audit_failures": len(failures),
        "rigged_animated_files": len(candidates),
        "deduplicated_rigged_animated_subjects": len(subjects),
        "duplicate_files_collapsed": len(candidates) - len(subjects),
        "development_or_regression_excluded_subjects": sum(row["development_exclusion"] is not None for row in subjects),
        "raw_full_family_eligible_subjects_before_development_exclusion": sum(row["capabilities"]["FULL_TRUTH139_RAW_ELIGIBLE"] for row in subjects),
        "raw_full_family_eligible_subjects_after_development_exclusion": sum(row["raw_eligible_for_truth139"] for row in subjects),
    }
    historical_match = (
        counts["gltf_glb_files_scanned"] == EXPECTED_GLTF_FILES
        and counts["rigged_animated_files"] == EXPECTED_RIGGED_ANIMATED_FILES
        and counts["deduplicated_rigged_animated_subjects"] == EXPECTED_DEDUP_SUBJECTS
        and not failures
    )
    payload = {
        "schema": AUDIT_SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_RAW_CAPABILITY_AUDIT__PLATFORM_ADMISSION_PENDING" if historical_match else "FAIL_OR_INCOMPLETE_RAW_CAPABILITY_AUDIT__DO_NOT_SPLIT_OR_TRAIN",
        "corpus_root_path": str(raw_root),
        "expected_corpus_root_sha256": expected_corpus_root_sha256,
        "corpus_root_hash_recomputed_by_this_tool": False,
        "fit8_selection_path": str(fit8_selection.resolve()),
        "historical_expectations": {
            "gltf_glb_files": EXPECTED_GLTF_FILES,
            "rigged_animated_files": EXPECTED_RIGGED_ANIMATED_FILES,
            "deduplicated_subjects": EXPECTED_DEDUP_SUBJECTS,
        },
        "historical_cardinality_reproduced": historical_match,
        "counts": counts,
        "duplicate_groups": duplicate_groups,
        "subjects": subjects,
        "file_failures": failures,
        "split_authorized": False,
        "training_authorized": False,
        "next_gate": "PLATFORM_IMPORT_RENDER_STAGE47_STAGE48_APPARATUS",
    }
    payload["audit_identity_sha256"] = _digest_json(payload)
    return payload


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--fit8-selection", default=str(ROOT / "canonical/FIT8_DIVERSITY_SELECTION_V1.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-corpus-root-sha256", default=EXPECTED_CORPUS_ROOT)
    args = parser.parse_args()
    result = audit(
        Path(args.raw_root),
        fit8_selection=Path(args.fit8_selection),
        expected_corpus_root_sha256=str(args.expected_corpus_root_sha256),
    )
    output = Path(args.output).resolve()
    _atomic_json(output, result)
    print(json.dumps({"output": str(output), "status": result["status"], "counts": result["counts"], "audit_identity_sha256": result["audit_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
