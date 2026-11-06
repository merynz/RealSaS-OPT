"""Pillow-free GLB parser tests, including strict byte and accessor bounds."""
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from tools.corpus.glb_binary_accessor_v0 import glb_document, accessor


def write_glb(path: Path, document, content: bytes):
    a = json.dumps(document, separators=(",", ":")).encode("utf-8")
    a += b" " * ((-len(a)) % 4)
    b = content + b"\x00" * ((-len(content)) % 4)
    file = (struct.pack("<III", 0x46546c67, 2, 12 + 8 + len(a) + 8 + len(b))
            + struct.pack("<II", len(a), 0x4E4F534A) + a
            + struct.pack("<II", len(b), 0x004E4942) + b)
    path.write_bytes(file)


def test_source_glb_reader_preserves_exact_strided_accessor(tmp_path):
    payload = np.asarray([[0.2, 0.4, 100.], [0.6, 0.8, 900.]],
                         dtype="<f4").tobytes()
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(payload)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(payload),
                         "byteStride": 12}],
        "accessors": [{"bufferView": 0, "componentType": 5126,
                       "type": "VEC2", "count": 2, "byteOffset": 0}],
    }
    path = tmp_path / "example.glb"
    write_glb(path, doc, payload)
    parsed, blob = glb_document(path)
    values = accessor(parsed, blob, 0)
    assert np.allclose(values, [[0.2, 0.4], [0.6, 0.8]])


def test_source_glb_accessor_rejects_bad_declared_buffer_bounds(tmp_path):
    doc = {"asset": {"version": "2.0"},
           "bufferViews": [{"buffer": 0, "byteLength": 4}],
           "accessors": [{"bufferView": 0, "componentType": 5126,
                          "type": "VEC2", "count": 1}]}
    path = tmp_path / "malformed.glb"
    write_glb(path, doc, np.zeros((1,), dtype="<f4").tobytes())
    parsed, blob = glb_document(path)
    with pytest.raises(ValueError, match="BUFFER_OUT_OF_RANGE"):
        accessor(parsed, blob, 0)


def test_source_glb_reader_rejects_mismatched_container_size(tmp_path):
    path = tmp_path / "sample.glb"
    write_glb(path, {"asset": {"version": "2.0"}}, b"\x00\x00\x00\x00")
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="CONTAINER_HEADER_INVALID"):
        glb_document(path)


def test_sparse_indices_are_explicitly_unsupported(tmp_path):
    doc = {"asset": {"version": "2.0"},
           "bufferViews": [{"buffer": 0, "byteLength": 12}],
           "accessors": [{"bufferView": 0, "componentType": 5126,
                          "type": "VEC2", "count": 1, "sparse": {}}]}
    path = tmp_path / "sparse.glb"
    write_glb(path, doc, np.zeros((3,), dtype="<f4").tobytes())
    parsed, blob = glb_document(path)
    with pytest.raises(ValueError, match="LAYOUT_UNSUPPORTED"):
        accessor(parsed, blob, 0)
