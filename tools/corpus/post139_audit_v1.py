"""Deterministic Post139 source census, structural audit and duplicate admission.

This tool creates reviewable source evidence only. It does not mint Platform,
research, model or product authority.

Typical full audit (Blender is strongly recommended for final cross-format dedup):

    python tools/corpus/post139_audit_v1.py audit \
      --source-root /data/UltimateAnimatedCharacter \
      --out /data/post139/ultimate_animated_character \
      --family Quaternius --pack UltimateAnimatedCharacter \
      --provenance CLEAN_CC0_HUMAN_AUTHORED \
      --blender-exe blender \
      --reference-root /data/base139_audit \
      --reference-root /data/post139_already_audited

A direct GLTF/GLB extractor is included for fast census/preflight. Final admission
across FBX/GLTF/GLB should use the Blender canonicalizer so every format is
measured in the same imported scene space.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
BLENDER_EXTRACTOR = Path(__file__).with_name("blender_post139_source_structure_v1.py")

SOURCE_STRUCTURE_SCHEMA = "RealSaS.Post139.SourceStructure.v1"
TECHNICAL_AUDIT_SCHEMA = "RealSaS.Post139.TechnicalAudit.v1"
ADMISSION_SCHEMA = "RealSaS.Post139.Admission.v1"
AUDIT_INDEX_SCHEMA = "RealSaS.Post139.AuditIndex.v1"
CANDIDATE_MANIFEST_SCHEMA = "RealSaS.Post139.CandidateManifest.v1"

CLEAN_PROVENANCE = {
    "CLEAN_CC0_HUMAN_AUTHORED",
    "CLEAN_PERMISSIVE_HUMAN_AUTHORED",
}
DIRECT_EXTENSIONS = {".gltf", ".glb"}
BLENDER_EXTENSIONS = {".gltf", ".glb", ".fbx"}
ALL_EXTENSIONS = DIRECT_EXTENSIONS | {".fbx"}

COMPONENTS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
TYPE_WIDTH = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


class AuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class Candidate:
    path: Path
    subject_id: str
    source_family: str
    pack: str
    provenance: str
    license: str | None = None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not value:
        raise AuditError("POST139_EMPTY_SUBJECT_ID")
    return value


def qfloat(value: float, digits: int = 7) -> float:
    if not math.isfinite(value):
        raise AuditError("POST139_NONFINITE_FLOAT")
    rounded = round(float(value), digits)
    return 0.0 if rounded == -0.0 else rounded


def _decode_data_uri(uri: str) -> bytes:
    if not uri.startswith("data:") or "," not in uri:
        raise AuditError("POST139_INVALID_DATA_URI")
    header, payload = uri.split(",", 1)
    if ";base64" not in header:
        raise AuditError("POST139_NON_BASE64_DATA_URI_UNSUPPORTED")
    return base64.b64decode(payload, validate=True)


def _load_glb(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    raw = path.read_bytes()
    if len(raw) < 12:
        raise AuditError("POST139_GLB_TOO_SHORT")
    magic, version, declared = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2 or declared != len(raw):
        raise AuditError("POST139_GLB_HEADER_INVALID")
    offset = 12
    document: dict[str, Any] | None = None
    binary_chunks: list[bytes] = []
    while offset + 8 <= len(raw):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_len]
        offset += chunk_len
        if len(chunk) != chunk_len:
            raise AuditError("POST139_GLB_CHUNK_TRUNCATED")
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary_chunks.append(chunk)
    if document is None:
        raise AuditError("POST139_GLB_JSON_MISSING")
    return document, binary_chunks


def load_gltf(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    suffix = path.suffix.lower()
    if suffix == ".glb":
        document, glb_chunks = _load_glb(path)
    elif suffix == ".gltf":
        document = json.loads(path.read_text(encoding="utf-8"))
        glb_chunks = []
    else:
        raise AuditError(f"POST139_DIRECT_FORMAT_UNSUPPORTED:{suffix}")

    buffers: list[bytes] = []
    glb_index = 0
    for row in document.get("buffers", []):
        uri = row.get("uri")
        if uri is None:
            if glb_index >= len(glb_chunks):
                raise AuditError("POST139_GLB_BUFFER_CHUNK_MISSING")
            payload = glb_chunks[glb_index]
            glb_index += 1
        elif uri.startswith("data:"):
            payload = _decode_data_uri(uri)
        else:
            dependency = (path.parent / uri).resolve()
            if not dependency.is_file():
                raise AuditError(f"POST139_EXTERNAL_BUFFER_MISSING:{uri}")
            payload = dependency.read_bytes()
        declared = row.get("byteLength")
        if declared is not None and len(payload) < int(declared):
            raise AuditError("POST139_BUFFER_SHORTER_THAN_DECLARED")
        buffers.append(payload)
    return document, buffers


def read_accessor(document: dict[str, Any], buffers: Sequence[bytes], index: int) -> list[Any]:
    accessor = document["accessors"][index]
    if "sparse" in accessor:
        raise AuditError("POST139_SPARSE_ACCESSOR_REQUIRES_BLENDER")
    if "bufferView" not in accessor:
        raise AuditError("POST139_ACCESSOR_WITHOUT_BUFFERVIEW_REQUIRES_BLENDER")
    component_type = int(accessor["componentType"])
    if component_type not in COMPONENTS:
        raise AuditError(f"POST139_COMPONENT_TYPE_UNSUPPORTED:{component_type}")
    fmt, component_size = COMPONENTS[component_type]
    element_width = TYPE_WIDTH[accessor["type"]]
    view = document["bufferViews"][accessor["bufferView"]]
    payload = buffers[int(view["buffer"])]
    view_offset = int(view.get("byteOffset", 0))
    accessor_offset = int(accessor.get("byteOffset", 0))
    packed_size = component_size * element_width
    stride = int(view.get("byteStride", packed_size))
    if stride < packed_size:
        raise AuditError("POST139_ACCESSOR_STRIDE_INVALID")
    count = int(accessor["count"])
    unpacker = struct.Struct("<" + fmt * element_width)
    values: list[Any] = []
    for row_index in range(count):
        start = view_offset + accessor_offset + row_index * stride
        end = start + packed_size
        if end > len(payload):
            raise AuditError("POST139_ACCESSOR_OUT_OF_RANGE")
        row = unpacker.unpack(payload[start:end])
        values.append(row[0] if element_width == 1 else row)
    return values


def _normalized_positions(position_rows: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    if not position_rows:
        return []
    xs = [float(row[0]) for row in position_rows]
    ys = [float(row[1]) for row in position_rows]
    zs = [float(row[2]) for row in position_rows]
    center = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
    centered = [(x - center[0], y - center[1], z - center[2]) for x, y, z in zip(xs, ys, zs)]
    rms = math.sqrt(sum(x * x + y * y + z * z for x, y, z in centered) / len(centered))
    scale = rms if rms > 1e-12 else 1.0
    return [(qfloat(x / scale), qfloat(y / scale), qfloat(z / scale)) for x, y, z in centered]


def _node_parent_map(document: dict[str, Any]) -> dict[int, int]:
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(document.get("nodes", [])):
        for child in node.get("children", []):
            parents[int(child)] = parent_index
    return parents


def _direct_gltf_structure(path: Path) -> dict[str, Any]:
    document, buffers = load_gltf(path)
    primitive_records = []
    geometry_present = False
    skin_present = False
    total_vertices = 0
    total_indices = 0

    for mesh_index, mesh in enumerate(document.get("meshes", [])):
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            attrs = primitive.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            positions = read_accessor(document, buffers, int(attrs["POSITION"]))
            normalized = _normalized_positions(positions)
            total_vertices += len(normalized)
            geometry_present = geometry_present or bool(normalized)
            indices: list[Any] = []
            if "indices" in primitive:
                indices = read_accessor(document, buffers, int(primitive["indices"]))
                total_indices += len(indices)
            joints = read_accessor(document, buffers, int(attrs["JOINTS_0"])) if "JOINTS_0" in attrs else []
            weights = read_accessor(document, buffers, int(attrs["WEIGHTS_0"])) if "WEIGHTS_0" in attrs else []
            if joints and weights and len(joints) == len(weights) == len(normalized):
                skin_present = True
            primitive_records.append(
                {
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    "vertex_count": len(normalized),
                    "index_count": len(indices),
                    "mode": int(primitive.get("mode", 4)),
                    "positions_hash": canonical_hash(normalized),
                    "indices_hash": canonical_hash(indices),
                    "skin_hash": canonical_hash(
                        [
                            {
                                "j": [int(x) for x in joint_row],
                                "w": [qfloat(float(x), 6) for x in weight_row],
                            }
                            for joint_row, weight_row in zip(joints, weights)
                        ]
                    )
                    if joints and weights
                    else None,
                }
            )

    parents = _node_parent_map(document)
    joint_records = []
    for skin_index, skin in enumerate(document.get("skins", [])):
        joint_set = {int(x) for x in skin.get("joints", [])}
        for joint in sorted(joint_set):
            node = document.get("nodes", [])[joint]
            parent_index = parents.get(joint)
            parent_name = None
            if parent_index in joint_set:
                parent_name = document["nodes"][parent_index].get("name", f"node_{parent_index}")
            joint_records.append(
                {
                    "skin": skin_index,
                    "name": node.get("name", f"node_{joint}"),
                    "parent": parent_name,
                    "translation": [qfloat(float(v)) for v in node.get("translation", [0.0, 0.0, 0.0])],
                    "rotation": [qfloat(float(v)) for v in node.get("rotation", [0.0, 0.0, 0.0, 1.0])],
                    "scale": [qfloat(float(v)) for v in node.get("scale", [1.0, 1.0, 1.0])],
                }
            )

    materials = document.get("materials", [])
    images = document.get("images", [])
    textures = document.get("textures", [])
    appearance_payload = {
        "materials": materials,
        "images": [{k: row[k] for k in sorted(row) if k not in {"uri"}} | ({"uri": row.get("uri")} if row.get("uri", "").startswith("data:") else {"uri": Path(row.get("uri", "")).name}) for row in images],
        "textures": textures,
    }

    animations = []
    for index, animation in enumerate(document.get("animations", [])):
        animations.append(
            {
                "name": animation.get("name", f"animation_{index}"),
                "channels": len(animation.get("channels", [])),
                "samplers": len(animation.get("samplers", [])),
            }
        )

    geometry_payload = [
        {
            "vertex_count": row["vertex_count"],
            "index_count": row["index_count"],
            "mode": row["mode"],
            "positions_hash": row["positions_hash"],
            "indices_hash": row["indices_hash"],
        }
        for row in primitive_records
    ]
    skin_payload = [row["skin_hash"] for row in primitive_records if row["skin_hash"]]
    rig_payload = sorted(joint_records, key=lambda row: (row["skin"], row["name"], row["parent"] or ""))
    geometry_hash = canonical_hash(geometry_payload)
    rig_hash = canonical_hash(rig_payload)
    skin_hash = canonical_hash(skin_payload)
    structural_hash = canonical_hash(
        {"geometry_topology": geometry_hash, "rig_rest": rig_hash, "skin_weights": skin_hash}
    )

    return {
        "schema": SOURCE_STRUCTURE_SCHEMA,
        "extractor": {
            "name": "DIRECT_GLTF_V1",
            "canonical_space": "FORMAT_LOCAL_PRELIMINARY",
            "cross_format_dedup_ready": False,
        },
        "source": {
            "path": str(path),
            "name": path.name,
            "extension": path.suffix.lower(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        },
        "metrics": {
            "mesh_count": len(document.get("meshes", [])),
            "primitive_count": len(primitive_records),
            "vertex_count": total_vertices,
            "index_count": total_indices,
            "skin_count": len(document.get("skins", [])),
            "joint_count": len({(row["skin"], row["name"]) for row in joint_records}),
            "animation_count": len(animations),
            "material_count": len(materials),
        },
        "capabilities": {
            "geometry": geometry_present,
            "appearance": bool(materials or images or textures),
            "rig": bool(joint_records),
            "skin": skin_present,
            "animation": bool(animations),
        },
        "fingerprints": {
            "geometry_topology": geometry_hash,
            "rig_rest": rig_hash,
            "skin_weights": skin_hash,
            "structural_subject": structural_hash,
            "appearance": canonical_hash(appearance_payload),
            "animation_inventory": canonical_hash(animations),
        },
        "inventory": {
            "primitives": primitive_records,
            "joints": rig_payload,
            "animations": animations,
        },
    }


def _run_blender_structure(path: Path, blender_exe: str, scratch: Path) -> dict[str, Any]:
    scratch.parent.mkdir(parents=True, exist_ok=True)
    command = [
        blender_exe,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(BLENDER_EXTRACTOR),
        "--",
        "--source",
        str(path.resolve()),
        "--out",
        str(scratch.resolve()),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr)[-8000:]
        raise AuditError(f"POST139_BLENDER_EXTRACTION_FAILED:{path.name}\n{tail}")
    if not scratch.is_file():
        raise AuditError("POST139_BLENDER_OUTPUT_MISSING")
    result = json.loads(scratch.read_text(encoding="utf-8"))
    if result.get("schema") != SOURCE_STRUCTURE_SCHEMA:
        raise AuditError("POST139_BLENDER_OUTPUT_SCHEMA")
    return result


def extract_structure(path: Path, blender_exe: str | None, scratch: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if blender_exe and suffix in BLENDER_EXTENSIONS:
        return _run_blender_structure(path, blender_exe, scratch)
    if suffix in DIRECT_EXTENSIONS:
        return _direct_gltf_structure(path)
    if suffix == ".fbx":
        raise AuditError("POST139_FBX_REQUIRES_BLENDER")
    raise AuditError(f"POST139_SOURCE_FORMAT_UNSUPPORTED:{suffix}")


def discover_candidates(
    source_root: Path,
    source_family: str,
    pack: str,
    provenance: str,
    license_name: str | None,
) -> list[Candidate]:
    paths = sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in ALL_EXTENSIONS
    )
    candidates = []
    used_ids: dict[str, int] = {}
    for path in paths:
        base = slugify(path.stem)
        count = used_ids.get(base, 0)
        used_ids[base] = count + 1
        subject_id = base if count == 0 else f"{base}-{count + 1}"
        candidates.append(Candidate(path, subject_id, source_family, pack, provenance, license_name))
    return candidates


def load_manifest(path: Path, source_root: Path) -> list[Candidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CANDIDATE_MANIFEST_SCHEMA:
        raise AuditError("POST139_CANDIDATE_MANIFEST_SCHEMA")
    source_family = payload["source_family"]
    pack = payload["pack"]
    provenance = payload["provenance"]
    license_name = payload.get("license")
    candidates = []
    seen_ids: set[str] = set()
    for row in payload.get("candidates", []):
        relative = Path(row["path"])
        candidate_path = (source_root / relative).resolve()
        if not candidate_path.is_relative_to(source_root.resolve()):
            raise AuditError("POST139_CANDIDATE_PATH_ESCAPE")
        if not candidate_path.is_file():
            raise AuditError(f"POST139_CANDIDATE_MISSING:{relative.as_posix()}")
        subject_id = slugify(row.get("subject_id", candidate_path.stem))
        if subject_id in seen_ids:
            raise AuditError(f"POST139_DUPLICATE_SUBJECT_ID:{subject_id}")
        seen_ids.add(subject_id)
        candidates.append(
            Candidate(
                candidate_path,
                subject_id,
                row.get("source_family", source_family),
                row.get("pack", pack),
                row.get("provenance", provenance),
                row.get("license", license_name),
            )
        )
    return candidates


def technical_audit(structure: dict[str, Any]) -> dict[str, Any]:
    capabilities = structure["capabilities"]
    required = {name: bool(capabilities.get(name)) for name in ("geometry", "appearance", "rig", "skin")}
    missing = sorted(name for name, ok in required.items() if not ok)
    extractor = structure["extractor"]
    canonical_ready = bool(extractor.get("cross_format_dedup_ready"))
    warnings = []
    if not canonical_ready:
        warnings.append("FINAL_CROSS_FORMAT_DEDUP_REQUIRES_BLENDER_CANONICALIZER")
    if not capabilities.get("animation"):
        warnings.append("ANIMATION_TRUTH_ABSENT__STATIC_TRUTH_MAY_STILL_BE_USABLE")
    return {
        "schema": TECHNICAL_AUDIT_SCHEMA,
        "source_sha256": structure["source"]["sha256"],
        "required_truth": required,
        "animation_truth": bool(capabilities.get("animation")),
        "missing_full_admission_truth": missing,
        "full_static_truth_ready": not missing,
        "canonical_dedup_ready": canonical_ready,
        "warnings": warnings,
        "status": "PASS" if not missing else "PARTIAL",
    }


def iter_reference_structures(roots: Sequence[Path]) -> Iterable[tuple[str, dict[str, Any]]]:
    for root in roots:
        if not root.exists():
            raise AuditError(f"POST139_REFERENCE_ROOT_MISSING:{root}")
        for path in sorted(root.rglob("source_structure.json")):
            try:
                structure = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AuditError(f"POST139_REFERENCE_STRUCTURE_INVALID:{path}:{exc}") from exc
            if structure.get("schema") != SOURCE_STRUCTURE_SCHEMA:
                continue
            subject_id = path.parent.name
            yield subject_id, structure


def build_reference_map(roots: Sequence[Path]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for subject_id, structure in iter_reference_structures(roots):
        fingerprint = structure.get("fingerprints", {}).get("structural_subject")
        if not fingerprint:
            continue
        extractor = structure.get("extractor", {})
        result.setdefault(fingerprint, {
            "subject_id": subject_id,
            "source_sha256": structure.get("source", {}).get("sha256", ""),
            "canonical_space": extractor.get("canonical_space", "UNKNOWN"),
        })
    return result


def decide_admissions(
    rows: list[dict[str, Any]], reference_map: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    by_fingerprint: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        fingerprint = row["structure"]["fingerprints"]["structural_subject"]
        by_fingerprint.setdefault(fingerprint, []).append(row)

    decisions = []
    for fingerprint, group in sorted(by_fingerprint.items()):
        group.sort(key=lambda row: row["candidate"].subject_id)
        external = reference_map.get(fingerprint)
        canonical_subject = external["subject_id"] if external else group[0]["candidate"].subject_id
        for position, row in enumerate(group):
            candidate: Candidate = row["candidate"]
            structure = row["structure"]
            audit = row["technical_audit"]
            canonical_ready = bool(audit["canonical_dedup_ready"])
            duplicate_kind = None
            duplicate_of = None
            if external:
                duplicate_kind = "REFERENCE_STRUCTURAL_DUPLICATE"
                duplicate_of = external["subject_id"]
            elif position > 0:
                duplicate_kind = "BATCH_STRUCTURAL_DUPLICATE"
                duplicate_of = group[0]["candidate"].subject_id

            if candidate.provenance not in CLEAN_PROVENANCE:
                status = "QUARANTINE_PROVENANCE"
            elif audit["missing_full_admission_truth"]:
                status = "PARTIAL_TRUTH_AUXILIARY"
            elif duplicate_of:
                status = "DUPLICATE_COLLAPSE"
            elif not canonical_ready:
                status = "AUDIT_READY_PENDING_CANONICAL_DEDUP"
            else:
                status = "ADMIT_CANDIDATE"

            decisions.append(
                {
                    "schema": ADMISSION_SCHEMA,
                    "subject_id": candidate.subject_id,
                    "source_family": candidate.source_family,
                    "pack": candidate.pack,
                    "provenance": candidate.provenance,
                    "license": candidate.license,
                    "status": status,
                    "canonical_subject_id": canonical_subject,
                    "duplicate_kind": duplicate_kind,
                    "duplicate_of": duplicate_of,
                    "source_sha256": structure["source"]["sha256"],
                    "structural_subject_hash": fingerprint,
                    "appearance_hash": structure["fingerprints"]["appearance"],
                    "animation_truth": audit["animation_truth"],
                    "canonical_dedup_ready": canonical_ready,
                    "policy": {
                        "same_rig_alone_is_duplicate": False,
                        "format_pairs_are_distinct_subjects": False,
                        "color_only_variants_with_identical_geometry_topology_weights_collapse": True,
                        "full_admission_requires": ["geometry", "appearance", "rig", "skin"],
                    },
                }
            )
    return sorted(decisions, key=lambda row: row["subject_id"])


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_batch(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).resolve()
    out = Path(args.out).resolve()
    if not source_root.is_dir():
        raise AuditError(f"POST139_SOURCE_ROOT_MISSING:{source_root}")
    if args.manifest:
        candidates = load_manifest(Path(args.manifest).resolve(), source_root)
    else:
        if not args.family or not args.pack or not args.provenance:
            raise AuditError("POST139_DISCOVERY_REQUIRES_FAMILY_PACK_PROVENANCE")
        candidates = discover_candidates(
            source_root, args.family, args.pack, args.provenance, args.license_name
        )
    if not candidates:
        raise AuditError("POST139_NO_CANDIDATES")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    scratch_root = out / ".scratch"
    for candidate in candidates:
        subject_root = out / "subjects" / candidate.subject_id
        try:
            structure = extract_structure(
                candidate.path,
                args.blender_exe,
                scratch_root / f"{candidate.subject_id}.json",
            )
            structure["subject"] = {
                "subject_id": candidate.subject_id,
                "source_family": candidate.source_family,
                "pack": candidate.pack,
            }
            audit = technical_audit(structure)
            _write_json(subject_root / "source_structure.json", structure)
            _write_json(subject_root / "technical_audit.json", audit)
            rows.append({"candidate": candidate, "structure": structure, "technical_audit": audit})
        except (AuditError, OSError, json.JSONDecodeError) as exc:
            failures.append({"subject_id": candidate.subject_id, "source": str(candidate.path), "error": str(exc)})
            _write_json(
                subject_root / "technical_audit.json",
                {
                    "schema": TECHNICAL_AUDIT_SCHEMA,
                    "status": "FAIL",
                    "source": str(candidate.path),
                    "error": str(exc),
                },
            )

    references = [Path(value).resolve() for value in args.reference_root]
    reference_map = build_reference_map(references)
    decisions = decide_admissions(rows, reference_map)
    decision_map = {row["subject_id"]: row for row in decisions}
    for row in decisions:
        _write_json(out / "subjects" / row["subject_id"] / "admission.json", row)

    status_counts: dict[str, int] = {}
    for row in decisions:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    index = {
        "schema": AUDIT_INDEX_SCHEMA,
        "source_root": str(source_root),
        "candidate_count": len(candidates),
        "audited_count": len(rows),
        "failure_count": len(failures),
        "status_counts": dict(sorted(status_counts.items())),
        "failures": failures,
        "subjects": [
            {
                "subject_id": candidate.subject_id,
                "source": str(candidate.path.relative_to(source_root)),
                "status": decision_map.get(candidate.subject_id, {}).get("status", "AUDIT_FAILED"),
            }
            for candidate in candidates
        ],
    }
    _write_json(out / "POST139_AUDIT_INDEX.json", index)
    if scratch_root.exists():
        for child in scratch_root.glob("*.json"):
            child.unlink(missing_ok=True)
        try:
            scratch_root.rmdir()
        except OSError:
            pass
    print(json.dumps(index["status_counts"], sort_keys=True))
    if failures:
        print(f"POST139_AUDIT_FAILURES={len(failures)}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Audit one source pack into per-subject evidence")
    audit.add_argument("--source-root", required=True)
    audit.add_argument("--out", required=True)
    audit.add_argument("--manifest")
    audit.add_argument("--family")
    audit.add_argument("--pack")
    audit.add_argument("--provenance")
    audit.add_argument("--license-name")
    audit.add_argument("--blender-exe", help="Blender executable; required for FBX and final cross-format dedup")
    audit.add_argument("--reference-root", action="append", default=[], help="Existing audit root(s) containing subjects/*/source_structure.json")
    audit.set_defaults(func=audit_batch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AuditError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
