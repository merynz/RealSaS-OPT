from pathlib import Path

import numpy as np

from tools.corpus.fit8_f_flexicubes_reference_adapter_v1 import build_dense_flexicubes_grid_v1
from tools.corpus.fit8_f_qualified_fa_bundle_v1 import (
    QualifiedFAAuthorityRefsV1,
    build_qualified_fa_bundle_v1,
    deterministic_npz_bytes_v1,
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
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _grid():
    return build_dense_flexicubes_grid_v1(
        cell_min_ijk=(0, 0, 0),
        cell_max_exclusive_ijk=(6, 6, 6),
        lattice_origin_xyz=(-0.91, -0.87, -0.83),
        extraction_pitch=0.31,
    )


def _refs():
    return QualifiedFAAuthorityRefsV1(
        source_sha256="1" * 64,
        teacher_bundle_sha256="2" * 64,
        extraction_truth_sha256="3" * 64,
        canonical_camera_contract_sha256="4" * 64,
        extraction_scale_manifest_sha256="5" * 64,
        measurement_protocol_sha256="6" * 64,
        source_exact_bytes_validation="synthetic-test exact bytes witness",
        source_geometry_frame_validation="synthetic-test common frame witness",
        teacher_bundle_validation="synthetic-test teacher validation witness",
        extraction_truth_validation="synthetic-test extraction truth witness",
        canonical_camera_contract_validation="synthetic-test camera contract witness",
        extraction_scale_manifest_validation="synthetic-test scale manifest witness",
        measurement_protocol_preregistration="synthetic-test preregistration witness",
        provenance="synthetic qualified-builder regression guard",
    )


def _field_policy():
    return MeshFieldTeacherPolicyV1(
        spatial_tolerance=1.0e-10,
        minimum_triangle_area2=1.0e-12,
        distance_tie_tolerance=1.0e-11,
        bvh_leaf_size=4,
        query_batch_size=128,
        provenance="synthetic qualified-builder field policy",
    )


def _intersection_policy():
    return StaticSelfIntersectionPolicyV1(
        provenance="synthetic qualified-builder self-intersection policy",
    )


def test_deterministic_npz_bytes_are_name_order_invariant():
    a = {
        "z": np.asarray([3, 4], dtype=np.int64),
        "a": np.asarray([[1.25, 2.5]], dtype=np.float32),
    }
    b = {"a": a["a"].copy(), "z": a["z"].copy()}
    assert deterministic_npz_bytes_v1(a) == deterministic_npz_bytes_v1(b)


def test_authority_refs_reject_placeholder_sha():
    refs = _refs()
    bad = QualifiedFAAuthorityRefsV1(
        **{**refs.__dict__, "teacher_bundle_sha256": "0" * 64}
    )
    try:
        bad.validate()
    except ValueError as exc:
        assert "ZERO_SHA_FORBIDDEN:teacher_bundle_sha256" in str(exc)
    else:
        raise AssertionError("expected placeholder SHA rejection")


def test_closed_tetra_builds_byte_stable_qualified_fa_bundle(tmp_path: Path):
    vertices, faces = _carrier()
    grid = _grid()

    def build(root: Path):
        root.mkdir()
        return build_qualified_fa_bundle_v1(
            subject_key="UNIT_TETRA",
            vertices=vertices,
            faces=faces,
            grid=grid,
            authority_refs=_refs(),
            mesh_field_policy=_field_policy(),
            self_intersection_policy=_intersection_policy(),
            output_npz=root / "input.npz",
            output_manifest=root / "input.json",
            output_grid_authority=root / "grid.json",
            provenance="synthetic qualified F_A bundle test",
        )

    report_a = build(tmp_path / "a")
    report_b = build(tmp_path / "b")
    report_a.validate()
    report_b.validate()

    assert report_a.bundle_npz_sha256 == report_b.bundle_npz_sha256
    assert (tmp_path / "a" / "input.npz").read_bytes() == (tmp_path / "b" / "input.npz").read_bytes()
    assert (tmp_path / "a" / "input.json").read_bytes() == (tmp_path / "b" / "input.json").read_bytes()
    assert (tmp_path / "a" / "grid.json").read_bytes() == (tmp_path / "b" / "grid.json").read_bytes()

    archive = np.load(tmp_path / "a" / "input.npz", allow_pickle=False)
    assert archive["x_nx3"].dtype == np.float32
    assert archive["s_n"].dtype == np.float32
    assert archive["teacher_oriented_level_set_f64"].dtype == np.float64
    assert archive["orientation_valid"].all()
    assert report_a.orientation_valid_count == len(grid.vertex_xyz)
    assert report_a.topology_component_count == 1
    assert report_a.topology_boundary_edge_count == 0
    assert report_a.topology_nonmanifold_edge_count == 0
    assert report_a.static_self_intersection_free


def test_open_carrier_cannot_be_promoted_to_qualified_fa(tmp_path: Path):
    vertices, faces = _carrier()
    open_faces = faces[:-1]
    try:
        build_qualified_fa_bundle_v1(
            subject_key="UNIT_OPEN_TETRA",
            vertices=vertices,
            faces=open_faces,
            grid=_grid(),
            authority_refs=_refs(),
            mesh_field_policy=_field_policy(),
            self_intersection_policy=_intersection_policy(),
            output_npz=tmp_path / "input.npz",
            output_manifest=tmp_path / "input.json",
            output_grid_authority=tmp_path / "grid.json",
            provenance="synthetic open-carrier rejection test",
        )
    except ValueError as exc:
        assert "GLOBAL_SIGN_NOT_ADMITTED" in str(exc)
        assert "OPEN_BOUNDARY" in str(exc)
    else:
        raise AssertionError("expected open global-sign rejection")
