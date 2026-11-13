from __future__ import annotations

"""Raw Source→Graphics Python-native binary ABI court.

This asserts real C++ parser/face-index/addressing consumption, *not* RGB
pixel equivalence, original-source depth or dynamic occlusion qualification.
"""

import json
from pathlib import Path
import struct
import subprocess

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_from_dict_v1,
)
from compiler.realsas_compiler_core.graphics_reopen_v1 import (
    reopen_compiled_graphics_v1,
)
from compiler.realsas_compiler_core.runtime_graphics_binary_v1 import (
    encode_runtime_graphics_v1,
)
from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_graphics_capability_v1,
)
from tests.compiler.test_rss_vnext_raw_graphics_consumer_v1 import (
    _full_parent_fixture,
)


REPO = Path(__file__).resolve().parents[2]
NATIVE_PROBE = (
    REPO / "runtime/realsas_cpp/tests/rss_vnext_raw_graphics_cas_probe.cpp"
)


def test_real_original_images_reopen_to_eight_unqualified_native_graphics_views(
    tmp_path,
):
    cas, req, _, _ = _full_parent_fixture(
        tmp_path, active_mechanics_courts=True,
    )
    result = execute_graphics_capability_v1(
        req, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )
    assert result["status"] == "PASS"
    mesh_ref = next(
        r for r in req["input_artifacts"]
        if r["artifact_type"] == "RealSaS.CanonicalMeshIR"
    )
    mesh = canonical_mesh_from_dict_v1(
        json.loads(cas.read_object(mesh_ref))
    )
    compiled_json = json.loads(cas.read_object(result["outputs"][0]))
    compiled, runtime = reopen_compiled_graphics_v1(
        compiled_json, face_count=len(mesh.faces),
    )
    assert len(compiled.views) == len(runtime.views) == 8
    assert [v.view_index for v in runtime.views] == list(range(8))
    assert compiled.metadata["source_view_visibility_unqualified"] is True
    assert compiled.metadata["eight_virtual_runtime_views_not_source_observations"] is True
    for view in compiled.views:
        assert view.visible_fragment_ids == tuple(sorted(
            f.fragment_id for f in compiled.fragments
        ))
        assert view.order_edges == ()
        assert view.metadata["visibility_known_from_original_source"] is False
        assert view.metadata["source_order_edges_observed"] is False
        assert tuple(view.metadata["visibility_unknown_fragment_ids"]) == (
            view.visible_fragment_ids
        )
    runtime_json = json.loads(cas.read_object(result["outputs"][1]))
    assert runtime_json["runtime_graphics_hash"] == runtime.runtime_graphics_hash
    assert len(runtime_json["views"]) == 8

    raw = encode_runtime_graphics_v1(runtime)
    binary = tmp_path / "raw_runtime_graphics.bin"
    binary.write_bytes(raw)
    executable = tmp_path / "rss_raw_graphics_native_probe"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
         str(NATIVE_PROBE), "-o", str(executable)],
        check=True, capture_output=True, text=True,
    )
    verified = subprocess.run(
        [str(executable), str(binary)],
        capture_output=True, text=True, check=True,
    )
    assert "RAW_GFX_BINARY_CAS_NATIVE_CONSUMPTION_PASS" in verified.stdout
    assert "views=8" in verified.stdout
    assert f"mesh={mesh.mesh_lineage_hash}" in verified.stdout
    assert f"graphics={compiled.graph_hash}" in verified.stdout
    assert f"runtime={runtime.runtime_graphics_hash}" in verified.stdout

    # The same immutable original alpha silhouettes used for Mesh
    # admission must have the SAME half-integer/top-left triangle coverage
    # in the native runtime. Cameras here are explicitly fitted HYPOTHESES.
    from compiler.rss_vnext.workers.rss_vnext_raw_mesh_admission_v1 import (
        _observed_alpha, _project_fitted,
    )
    from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
        rasterize_triangles_half_integer_top_left,
    )
    from tests.compiler.test_rss_vnext_raw_mesh_admission_v1 import (
        _triangle_source_and_scene,
    )
    source, scene, _ = _triangle_source_and_scene()
    xyz = scene["tessa_candidate_rest_vertices"][0]
    faces = scene["tessa_candidate_faces"]
    for view_index in range(2):
        observed, blocker = _observed_alpha(source, view_index)
        assert blocker is None
        height, width = observed.shape
        projected, camera = _project_fitted(xyz, observed, view_index)
        assert camera["source_camera_was_observed"] is False
        triangles = [
            tuple(
                tuple(float(x) for x in projected[int(vertex_id)])
                for vertex_id in face
            ) for face in faces
        ]
        expected = rasterize_triangles_half_integer_top_left(
            triangles, width=width, height=height,
        )
        raster_input = tmp_path / f"fitted_triangle_view_{view_index}.bin"
        payload = bytearray(struct.pack("<III", width, height, len(triangles)))
        for tri in triangles:
            payload.extend(struct.pack("<6d", *(v for point in tri for v in point)))
        raster_input.write_bytes(payload)
        expected_file = tmp_path / f"expected_view_{view_index}.bin"
        expected_file.write_bytes(expected)
        native = subprocess.run(
            [str(executable), str(binary), str(raster_input),
             str(expected_file)],
            capture_output=True, text=True, check=True,
        )
        assert "RAW_GFX_STATIC_G5_RASTER_PIXEL_EQUIVALENCE_PASS" in native.stdout
        # A different expected source pixel cannot be accepted as parity.
        bad_alpha = bytearray(expected)
        bad_alpha[len(expected) // 2] ^= 1
        bad_expected_file = tmp_path / f"forged_view_{view_index}.bin"
        bad_expected_file.write_bytes(bad_alpha)
        denied = subprocess.run(
            [str(executable), str(binary), str(raster_input),
             str(bad_expected_file)],
            capture_output=True, text=True,
        )
        assert denied.returncode != 0
        assert "RAW_GFX_NATIVE_STATIC_RASTER_PIXEL_MISMATCH" in denied.stderr

    # A missing view-domain must never be "repaired" to eight on C++ read.
    # Magic[8], version[4], faces[4], fragments[4], view_count[4].
    malformed = bytearray(raw)
    struct.pack_into("<I", malformed, 20, 0)
    bad_file = tmp_path / "forged_no_views.bin"
    bad_file.write_bytes(malformed)
    denied = subprocess.run(
        [str(executable), str(bad_file)],
        capture_output=True, text=True,
    )
    assert denied.returncode != 0
    assert "GRAPHICS_V1_CARDINALITY_INVALID" in denied.stderr
