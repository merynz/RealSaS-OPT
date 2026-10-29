from __future__ import annotations

import json

import numpy as np
import pytest

from tools.corpus.iris_observation_admission_v1 import digest
from tools.corpus.prepare_iris_observations_v1 import (
    SURFACE_GEOMETRY_MANIFEST_NAME,
    SURFACE_GEOMETRY_NAME,
    verify_surface_geometry_packet,
)


def _row():
    return {"subject_key": "truth139_test", "source": {"sha256": "a" * 64}}


def _packet(tmp_path, *, include_response=False):
    packet = tmp_path / "truth139_test"
    packet.mkdir()
    geometry = packet / SURFACE_GEOMETRY_NAME
    arrays = {
        "rest_vertices_source_world": np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
        "face_owner_privileged": np.asarray([0], dtype=np.int32),
        "face_material_privileged": np.asarray([0], dtype=np.int32),
        "triangle_uv": np.asarray([[[0, 0], [1, 0], [0, 1]]], dtype=np.float64),
        "triangle_uv_valid": np.asarray([True], dtype=np.bool_),
    }
    if include_response:
        arrays["isolated_position"] = np.zeros((1, 3, 3), dtype=np.float64)
    np.savez_compressed(geometry, **arrays)
    manifest = {
        "schema": "RealSaS.ResponseIndependentRestSurfaceGeometry.v1",
        "status": "PASS_IMPORTED_REST_SURFACE_GEOMETRY__RAW_PARITY_ADMISSION_SEPARATE",
        "source_sha256": "a" * 64,
        "geometry_npz": {"path": str(geometry), "sha256": digest(geometry)},
    }
    (packet / SURFACE_GEOMETRY_MANIFEST_NAME).write_text(json.dumps(manifest))
    return packet


def test_surface_geometry_packet_accepts_exact_rest_only_payload(tmp_path):
    packet = _packet(tmp_path)
    result = verify_surface_geometry_packet(packet, _row())
    assert result["geometry_sha256"] == digest(packet / SURFACE_GEOMETRY_NAME)
    assert result["vertex_count"] == 3
    assert result["triangle_count"] == 1


def test_surface_geometry_packet_rejects_response_leak(tmp_path):
    packet = _packet(tmp_path, include_response=True)
    with pytest.raises(ValueError, match="RESPONSE_OR_SKIN_LEAK"):
        verify_surface_geometry_packet(packet, _row())


def test_surface_geometry_packet_rejects_hash_drift(tmp_path):
    packet = _packet(tmp_path)
    manifest_path = packet / SURFACE_GEOMETRY_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["geometry_npz"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="HASH_DRIFT"):
        verify_surface_geometry_packet(packet, _row())
