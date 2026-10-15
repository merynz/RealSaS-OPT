import base64
import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.fit8_dynamic_fixed_frame_presentation_v1 import FRAME_SCOPE
from tools.corpus.fit8_gltf_dynamic_census_v2 import (
    GLTFDynamicSamplingPolicyV2,
    build_gltf_dynamic_census_v2,
)


def _align(blob: bytearray, alignment: int = 4) -> None:
    while len(blob) % alignment:
        blob.append(0)


def _write_rigid_skinned_tetra(path: Path) -> None:
    blob = bytearray()
    views = []

    def add(array: np.ndarray, target=None) -> int:
        _align(blob)
        offset = len(blob)
        raw = np.ascontiguousarray(array).tobytes()
        blob.extend(raw)
        row = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            row["target"] = target
        views.append(row)
        return len(views) - 1

    position = np.asarray(
        [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
        dtype="<f4",
    )
    indices = np.asarray(
        [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
        dtype="<u2",
    )
    joints = np.asarray([[0, 0, 0, 0]] * 4, dtype="u1")
    weights = np.asarray([[1, 0, 0, 0]] * 4, dtype="<f4")
    ibm = np.eye(4, dtype="<f4").reshape(1, 16)
    times = np.asarray([0.0, 1.0], dtype="<f4")
    translations = np.asarray([[0, 0, 0], [2, 0, 0]], dtype="<f4")
    rotations = np.asarray([[0, 0, 0, 1], [0, 0, 1, 0]], dtype="<f4")

    vp = add(position, 34962)
    vi = add(indices, 34963)
    vj = add(joints, 34962)
    vw = add(weights, 34962)
    vb = add(ibm)
    vt = add(times)
    vtr = add(translations)
    vro = add(rotations)
    accessors = []

    def acc(view, component, count, kind):
        accessors.append({"bufferView": view, "componentType": component, "count": count, "type": kind})
        return len(accessors) - 1

    ap = acc(vp, 5126, 4, "VEC3")
    ai = acc(vi, 5123, 12, "SCALAR")
    aj = acc(vj, 5121, 4, "VEC4")
    aw = acc(vw, 5126, 4, "VEC4")
    ab = acc(vb, 5126, 1, "MAT4")
    at = acc(vt, 5126, 2, "SCALAR")
    atr = acc(vtr, 5126, 2, "VEC3")
    ar = acc(vro, 5126, 2, "VEC4")
    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode("ascii")
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "joint"}, {"name": "mesh", "mesh": 0, "skin": 0}],
        "buffers": [{"byteLength": len(blob), "uri": uri}],
        "bufferViews": views,
        "accessors": accessors,
        "meshes": [{"primitives": [{"attributes": {"POSITION": ap, "JOINTS_0": aj, "WEIGHTS_0": aw}, "indices": ai}]}],
        "skins": [{"joints": [0], "inverseBindMatrices": ab}],
        "animations": [{
            "name": "RigidMove",
            "samplers": [
                {"input": at, "output": atr, "interpolation": "LINEAR"},
                {"input": at, "output": ar, "interpolation": "LINEAR"},
            ],
            "channels": [
                {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                {"sampler": 1, "target": {"node": 0, "path": "rotation"}},
            ],
        }],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_integrated_v2_uses_one_dynamic_frame_and_preserves_rigid_mechanics(tmp_path: Path):
    source = tmp_path / "rigid_tetra.gltf"
    _write_rigid_skinned_tetra(source)
    policy = GLTFDynamicSamplingPolicyV2(
        normalized_times=(0.25, 0.75),
        presentation_resolution=32,
        frame_scale=1.2,
        provenance="SYNTHETIC_DYNAMIC_V2_INTEGRATION",
    )
    result, execution = build_gltf_dynamic_census_v2(
        source_path=source,
        subject_key="fixture_rigid_tetra",
        policy=policy,
    )
    assert execution["native_clip_count"] == 1
    assert execution["sample_count"] == 2
    assert execution["dynamic_frame_scope"] == FRAME_SCOPE
    assert execution["per_pose_recenter_or_rescale"] is False
    assert result.dynamic_frame_sha256 == execution["dynamic_frame_sha256"]
    assert result.dynamic_frame_scope == FRAME_SCOPE
    assert len(result.samples) == 2
    for sample in result.samples:
        assert sample.clip_name == "RigidMove"
        assert sample.quotient_worst_separation == 0.0
        assert sample.edge_stretch_p99 == pytest.approx(1.0, abs=1.0e-7)
        assert sample.face_condition_p99 == pytest.approx(1.0, abs=1.0e-7)
        assert sample.face_area_ratio_p01 == pytest.approx(1.0, abs=1.0e-7)
        assert sample.face_area_ratio_p99 == pytest.approx(1.0, abs=1.0e-7)
        assert sample.directed_cycle_view_count == 0
        assert sample.mixed_pair_count_sum == 0
        assert sample.maximum_distinct_component_stack == 1
        assert sample.new_directed_edges_vs_static == 0
        assert sample.strict_pair_flips_vs_static == 0
