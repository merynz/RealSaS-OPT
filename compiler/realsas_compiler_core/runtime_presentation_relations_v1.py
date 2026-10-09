"""RSS-v2 extension for qualified presentation contacts/composition.

The base source-owned package remains canonical for mesh/texture/XY/depth. This
module appends already-compiled presentation region identity and per-frame draw
order. Runtime is forbidden to infer either relation.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import hashlib
import struct

import numpy as np

from .runtime_package_v2 import build_source_owned_visual_rss_v2_entries
from .types import QualificationError
from .visual_relations_v1 import (
    COMPOSITION_OPERATOR_ID, POLICY, validate_region_order_timeline,
)
from .hashing import content_sha256


def _face_region_payload(face_region_id) -> bytes:
    values = np.asarray(face_region_id, dtype="<i4")
    if values.ndim != 1 or not len(values) or np.any(values < 0):
        raise QualificationError("RSS_V2_FACE_REGION_INVALID")
    return b"RSRG1\0\0\0" + struct.pack("<I", len(values)) + values.tobytes(order="C")


def _region_order_payload(order) -> bytes:
    rows = np.asarray(order, dtype="<i4")
    if rows.ndim != 2 or not len(rows) or not rows.shape[1]:
        raise QualificationError("RSS_V2_REGION_ORDER_INVALID")
    validate_region_order_timeline(rows, region_count=rows.shape[1])
    return b"RSRO1\0\0\0" + struct.pack("<II", *rows.shape) + rows.tobytes(order="C")


def build_relational_source_owned_visual_rss_v2_entries(projection) -> OrderedDict[str, bytes]:
    entries = build_source_owned_visual_rss_v2_entries(projection)
    operator = str(dict(projection.metadata or {}).get("semantic_composition_operator_id") or "")
    if not operator:
        return entries
    if operator != COMPOSITION_OPERATOR_ID:
        raise QualificationError("RSS_V2_SEMANTIC_COMPOSITION_OPERATOR_DRIFT")
    if dict(projection.metadata or {}).get("semantic_composition_policy_hash") != content_sha256(POLICY):
        raise QualificationError("RSS_V2_SEMANTIC_COMPOSITION_POLICY_DRIFT")
    path = Path(projection.projection_npz_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != projection.projection_npz_sha256:
        raise QualificationError("RSS_V2_RELATIONAL_PROJECTION_BYTES_DRIFT")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    manifest = entries["manifest.txt"].decode("utf-8").rstrip("\n").split("\n")
    manifest.extend([
        f"semantic_composition_contract={COMPOSITION_OPERATOR_ID}",
        f"semantic_composition_policy_hash={content_sha256(POLICY)}",
        "inter_region_occlusion=COMPILED_REGION_ORDER",
        "same_region_occlusion=CANONICAL_DEPTH",
        "runtime_semantic_order_inference=0",
    ])
    views = tuple(sorted(projection.views, key=lambda row: int(row.view_index)))
    for view in views:
        vi = int(view.view_index)
        region_key = f"view_{vi}_face_region_id"
        if region_key not in arrays:
            raise QualificationError("RSS_V2_FACE_REGION_MISSING")
        face_region = np.asarray(arrays[region_key], dtype=np.int32)
        if face_region.shape != (int(view.visual_face_count),) or np.any(face_region < 0):
            raise QualificationError("RSS_V2_FACE_REGION_SHAPE_INVALID")
        region_count = int(np.max(face_region)) + 1
        if not np.array_equal(np.unique(face_region), np.arange(region_count)):
            raise QualificationError("RSS_V2_FACE_REGION_IDS_NOT_DENSE")
        entry = f"visual_face_regions_v{vi}.bin"
        entries[entry] = _face_region_payload(face_region)
        manifest.extend([
            f"view.{vi}.region_count={region_count}",
            f"view.{vi}.face_region_entry={entry}",
        ])
    for clip_index, clip in enumerate(projection.clips):
        for view in views:
            vi = int(view.view_index)
            key = f"{clip.array_prefix}_view_{vi}_region_order"
            if key not in arrays:
                raise QualificationError("RSS_V2_REGION_ORDER_MISSING")
            rows = np.asarray(arrays[key], dtype=np.int32)
            region_count = int(np.max(arrays[f"view_{vi}_face_region_id"])) + 1
            if rows.shape != (int(clip.frame_count), region_count):
                raise QualificationError("RSS_V2_REGION_ORDER_SHAPE_INVALID")
            validate_region_order_timeline(rows, region_count=region_count)
            entry = f"clip_{clip_index}_v{vi}.region_order.bin"
            entries[entry] = _region_order_payload(rows)
            manifest.append(f"clip.{clip_index}.view.{vi}.region_order_entry={entry}")
    entries["manifest.txt"] = ("\n".join(manifest) + "\n").encode("utf-8")
    return entries
