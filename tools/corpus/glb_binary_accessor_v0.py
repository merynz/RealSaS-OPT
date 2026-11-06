"""Strict glTF 2.0 GLB JSON/BIN and accessor reader, pure stdlib + numpy.

No image/Pillow/Blender dependence; usable in native Blender's embedded Python.
Research-only narrow source binary parser, does not create source owner labels.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct
import numpy as np

COMPONENT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
DIMENSION = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}


def _require(ok, code):
    if not ok:
        raise ValueError("SOURCE_GLB_ACCESSOR_" + code)


def glb_document(source):
    b = Path(source).read_bytes()
    _require(len(b) >= 28, "CONTAINER_TOO_SHORT")
    magic, version, declared_length = struct.unpack_from("<III", b)
    _require(magic == 0x46546c67 and version == 2 and declared_length == len(b),
             "CONTAINER_HEADER_INVALID")
    json_length, json_kind = struct.unpack_from("<II", b, 12)
    _require(json_kind == 0x4E4F534A and json_length > 0
             and 20 + json_length + 8 <= len(b), "JSON_CHUNK_INVALID")
    document = json.loads(b[20:20 + json_length].decode("utf-8"))
    binary_length, binary_kind = struct.unpack_from("<II", b, 20 + json_length)
    _require(binary_kind == 0x004E4942
             and 20 + json_length + 8 + binary_length == len(b),
             "BIN_CHUNK_INVALID")
    _require(document.get("asset", {}).get("version") == "2.0",
             "GLTF_SCHEMA_UNEXPECTED")
    return document, memoryview(b)[20 + json_length + 8:]


def accessor(document, blob, index):
    rows = document.get("accessors", [])
    _require(type(index) is int and 0 <= index < len(rows),
             "ACCESSOR_INDEX_INVALID")
    a = rows[index]
    _require("sparse" not in a and "bufferView" in a and
             a.get("componentType") in COMPONENT and a.get("type") in DIMENSION
             and not a.get("normalized", False),
             "ACCESSOR_LAYOUT_UNSUPPORTED")
    views = document.get("bufferViews", [])
    view_index = a["bufferView"]
    _require(type(view_index) is int and 0 <= view_index < len(views),
             "BUFFER_VIEW_INDEX_INVALID")
    v = views[view_index]
    _require(v.get("buffer") == 0, "EXTERNAL_BINARY_BUFFER_UNSUPPORTED")
    dtype = np.dtype(COMPONENT[a["componentType"]]).newbyteorder("<")
    width = DIMENSION[a["type"]]
    count = a.get("count")
    _require(type(count) is int and count > 0, "ACCESSOR_COUNT_INVALID")
    stride = v.get("byteStride", width * dtype.itemsize)
    rel = a.get("byteOffset", 0)
    base = v.get("byteOffset", 0)
    length = v.get("byteLength", 0)
    _require(all(type(x) is int for x in (stride, rel, base, length))
             and stride >= width * dtype.itemsize
             and rel >= 0 and base >= 0 and length > 0,
             "BUFFER_STRIDE_OFFSET_INVALID")
    last = rel + (count - 1) * stride + width * dtype.itemsize
    _require(last <= length and base + last <= len(blob),
             "BUFFER_OUT_OF_RANGE")
    return np.ndarray(
        (count, width), dtype=dtype, buffer=blob, offset=base + rel,
        strides=(stride, dtype.itemsize),
    ).copy()
