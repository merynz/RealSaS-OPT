import base64
import json
from pathlib import Path

import numpy as np

from tools.corpus.fit8_gltf_dynamic_pose_v1 import (
    build_gltf_dynamic_source_v1,
    evaluate_authored_pose_v1,
)


def _align(blob: bytearray, alignment: int = 4) -> None:
    while len(blob) % alignment:
        blob.append(0)


def _write_animated_skin_gltf(path: Path) -> None:
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

    position = np.asarray([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype="<f4")
    indices = np.asarray([0, 1, 2], dtype="<u2")
    joints = np.asarray([[0, 0, 0, 0]] * 3, dtype="u1")
    weights = np.asarray([[1, 0, 0, 0]] * 3, dtype="<f4")
    ibm = np.eye(4, dtype="<f4").reshape(1, 16)
    times = np.asarray([0.0, 1.0], dtype="<f4")
    translations = np.asarray([[0, 0, 0], [2, 0, 0]], dtype="<f4")
    rotations = np.asarray(
        [[0, 0, 0, 1], [0, 0, 1, 0]],
        dtype="<f4",
    )

    p0 = add(position, 34962)
    i0 = add(indices, 34963)
    j0 = add(joints, 34962)
    w0 = add(weights, 34962)
    p1 = add(position, 34962)
    i1 = add(indices, 34963)
    j1 = add(joints, 34962)
    w1 = add(weights, 34962)
    ib = add(ibm)
    ti = add(times)
    tr = add(translations)
    ro = add(rotations)

    accessors = []

    def accessor(view, component_type, count, kind, normalized=False):
        row = {
            "bufferView": view,
            "componentType": component_type,
            "count": count,
            "type": kind,
        }
        if normalized:
            row["normalized"] = True
        accessors.append(row)
        return len(accessors) - 1

    ap0 = accessor(p0, 5126, 3, "VEC3")
    ai0 = accessor(i0, 5123, 3, "SCALAR")
    aj0 = accessor(j0, 5121, 3, "VEC4")
    aw0 = accessor(w0, 5126, 3, "VEC4")
    ap1 = accessor(p1, 5126, 3, "VEC3")
    ai1 = accessor(i1, 5123, 3, "SCALAR")
    aj1 = accessor(j1, 5121, 3, "VEC4")
    aw1 = accessor(w1, 5126, 3, "VEC4")
    aib = accessor(ib, 5126, 1, "MAT4")
    ati = accessor(ti, 5126, 2, "SCALAR")
    atr = accessor(tr, 5126, 2, "VEC3")
    aro = accessor(ro, 5126, 2, "VEC4")

    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode("ascii")
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "joint"},
            {"name": "mesh", "mesh": 0, "skin": 0},
        ],
        "buffers": [{"byteLength": len(blob), "uri": uri}],
        "bufferViews": views,
        "accessors": accessors,
        "materials": [{"name": "A"}, {"name": "B"}],
        "meshes": [{
            "primitives": [
                {"attributes": {"POSITION": ap0, "JOINTS_0": aj0, "WEIGHTS_0": aw0}, "indices": ai0, "material": 0},
                {"attributes": {"POSITION": ap1, "JOINTS_0": aj1, "WEIGHTS_0": aw1}, "indices": ai1, "material": 1},
            ]
        }],
        "skins": [{"joints": [0], "inverseBindMatrices": aib}],
        "animations": [{
            "name": "MoveRotate",
            "samplers": [
                {"input": ati, "output": atr, "interpolation": "LINEAR"},
                {"input": ati, "output": aro, "interpolation": "LINEAR"},
            ],
            "channels": [
                {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                {"sampler": 1, "target": {"node": 0, "path": "rotation"}},
            ],
        }],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_linear_trs_skinning_and_quotient_motion_witness(tmp_path: Path):
    source_path = tmp_path / "animated.gltf"
    _write_animated_skin_gltf(source_path)
    dynamic = build_gltf_dynamic_source_v1(source_path)
    assert dynamic.render_vertex_count == 6
    assert dynamic.geometric_vertex_count == 3
    pose = evaluate_authored_pose_v1(dynamic, clip_index=0, normalized_time=0.5)
    assert pose.quotient_worst_separation < 1.0e-12
    # +90deg about glTF Z, plus +1 X translation, then RealSaS Y-up -> Z-up.
    expected = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    actual = np.asarray(sorted(pose.geometric_xyz.tolist()), dtype=np.float64)
    target = np.asarray(sorted(expected.tolist()), dtype=np.float64)
    np.testing.assert_allclose(actual, target, atol=1.0e-7, rtol=0.0)


def test_normalized_time_endpoints_are_valid(tmp_path: Path):
    source_path = tmp_path / "animated.gltf"
    _write_animated_skin_gltf(source_path)
    dynamic = build_gltf_dynamic_source_v1(source_path)
    start = evaluate_authored_pose_v1(dynamic, clip_index=0, normalized_time=0.0)
    end = evaluate_authored_pose_v1(dynamic, clip_index=0, normalized_time=1.0)
    assert start.authored_time == 0.0
    assert end.authored_time == 1.0
    assert start.quotient_worst_separation == 0.0
    assert end.quotient_worst_separation == 0.0
