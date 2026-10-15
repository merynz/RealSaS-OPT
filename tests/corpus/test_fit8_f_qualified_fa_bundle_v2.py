from pathlib import Path

import json
import numpy as np

from tools.corpus.fit8_f_flexicubes_reference_adapter_v1 import build_dense_flexicubes_grid_v1
from tools.corpus.fit8_f_qualified_fa_bundle_v1 import QualifiedFAAuthorityRefsV1
from tools.corpus.fit8_f_qualified_fa_bundle_v2 import (
    build_qualified_fa_bundle_v2,
    canonical_array_content_sha256_v1,
    canonical_source_carrier_sha256_v1,
)
from tools.corpus.fit8_mesh_field_teacher_v1 import MeshFieldTeacherPolicyV1
from tools.corpus.fit8_static_self_intersection_qualification_v1 import StaticSelfIntersectionPolicyV1


def _carrier():
    vertices = np.asarray(
        [
            [-0.43, -0.31, -0.27],
            [0.51, -0.22, -0.18],
            [-0.17, 0.57, -0.11],
            [-0.08, -0.05, 0.63],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int64,
    )
    return vertices, faces


def _wide_grid():
    return build_dense_flexicubes_grid_v1(
        cell_min_ijk=(0, 0, 0),
        cell_max_exclusive_ijk=(8, 8, 8),
        lattice_origin_xyz=(-1.24, -1.24, -1.24),
        extraction_pitch=0.31,
    )


def _tight_grid():
    return build_dense_flexicubes_grid_v1(
        cell_min_ijk=(0, 0, 0),
        cell_max_exclusive_ijk=(4, 4, 4),
        lattice_origin_xyz=(-0.50, -0.50, -0.50),
        extraction_pitch=0.28,
    )


def _refs():
    return QualifiedFAAuthorityRefsV1(
        source_sha256="1" * 64,
        teacher_bundle_sha256="2" * 64,
        extraction_truth_sha256="3" * 64,
        canonical_camera_contract_sha256="4" * 64,
        extraction_scale_manifest_sha256="5" * 64,
        measurement_protocol_sha256="6" * 64,
        source_exact_bytes_validation="synthetic exact bytes witness",
        source_geometry_frame_validation="synthetic frame witness",
        teacher_bundle_validation="synthetic teacher witness",
        extraction_truth_validation="synthetic extraction witness",
        canonical_camera_contract_validation="synthetic camera witness",
        extraction_scale_manifest_validation="synthetic scale witness",
        measurement_protocol_preregistration="synthetic protocol witness",
        provenance="synthetic v2 authority refs",
    )


def _field_policy():
    return MeshFieldTeacherPolicyV1(
        spatial_tolerance=1.0e-10,
        minimum_triangle_area2=1.0e-12,
        distance_tie_tolerance=1.0e-11,
        bvh_leaf_size=4,
        query_batch_size=128,
        provenance="synthetic v2 field policy",
    )


def _intersection_policy():
    return StaticSelfIntersectionPolicyV1(
        provenance="synthetic v2 intersection policy",
    )


def test_semantic_content_hash_is_name_order_invariant_and_value_sensitive():
    a = {
        "z": np.asarray([1, 2, 3], dtype=np.int32),
        "a": np.asarray([[1.0, 2.0]], dtype=np.float64),
    }
    b = {"a": a["a"].copy(), "z": a["z"].copy()}
    assert canonical_array_content_sha256_v1(a) == canonical_array_content_sha256_v1(b)
    b["z"][1] = 99
    assert canonical_array_content_sha256_v1(a) != canonical_array_content_sha256_v1(b)


def test_source_carrier_hash_changes_with_geometry_not_dict_or_zip_state():
    vertices, faces = _carrier()
    first = canonical_source_carrier_sha256_v1(vertices, faces)
    changed = vertices.copy()
    changed[0, 0] += 1.0e-9
    second = canonical_source_carrier_sha256_v1(changed, faces)
    assert first != second


def test_v2_build_requires_margin_and_outside_shell(tmp_path: Path):
    vertices, faces = _carrier()
    report = build_qualified_fa_bundle_v2(
        subject_key="UNIT_TETRA_V2",
        vertices=vertices,
        faces=faces,
        grid=_wide_grid(),
        authority_refs=_refs(),
        mesh_field_policy=_field_policy(),
        self_intersection_policy=_intersection_policy(),
        minimum_source_margin_cells=1.0,
        output_npz=tmp_path / "input.npz",
        output_manifest=tmp_path / "input.json",
        output_grid_authority=tmp_path / "grid.json",
        provenance="synthetic v2 qualification",
    )
    report.validate()
    assert report.volume.qualified
    assert report.volume.measured_min_margin_cells >= 1.0
    assert report.volume.boundary_inside_count == 0
    assert report.volume.boundary_nonpositive_count == 0
    assert report.array_content_sha256 != report.transport_npz_sha256

    manifest = json.loads((tmp_path / "input.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    assert metadata["outside_shell_qualified"] is True
    assert metadata["array_content_sha256"] == report.array_content_sha256
    assert metadata["source_carrier_sha256"] == report.source_carrier_sha256
    assert metadata["transport_npz_sha256"] == report.transport_npz_sha256


def test_v2_rejects_source_without_preregistered_volume_margin(tmp_path: Path):
    vertices, faces = _carrier()
    try:
        build_qualified_fa_bundle_v2(
            subject_key="UNIT_TETRA_TIGHT",
            vertices=vertices,
            faces=faces,
            grid=_tight_grid(),
            authority_refs=_refs(),
            mesh_field_policy=_field_policy(),
            self_intersection_policy=_intersection_policy(),
            minimum_source_margin_cells=1.0,
            output_npz=tmp_path / "input.npz",
            output_manifest=tmp_path / "input.json",
            output_grid_authority=tmp_path / "grid.json",
            provenance="synthetic v2 tight-volume rejection",
        )
    except ValueError as exc:
        assert "SOURCE_MARGIN_INSUFFICIENT" in str(exc)
    else:
        raise AssertionError("expected insufficient extraction-volume margin rejection")
