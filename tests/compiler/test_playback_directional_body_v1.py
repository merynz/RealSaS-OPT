from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from compiler.realsas_compiler_core.playback_directional_body_v1 import (
    DirectionalBodyAssetV1,
    build_directional_body_asset_v1,
    build_directional_body_runtime_v4_clip_v1,
    build_directional_body_runtime_v4_contract_v1,
)
from compiler.realsas_compiler_core.playback_directional_motion_cert_v1 import (
    certify_directional_body_motion_v1,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    AttachmentKind,
    RuntimeV3FrameComposition,
    RuntimeV3Slot,
    TopologyClass,
)
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    RuntimeV4AttachmentAsset,
    RuntimeV4Clip,
    RuntimeV4Frame,
    RuntimeV4PlaybackContract,
    RuntimeV4ViewAssetOverlay,
    RuntimeV4ViewOverlay,
    provenance_from_code,
)
from compiler.realsas_compiler_core.types import QualificationError


def _camera(view_id: str = "V0", index: int = 0) -> CameraProjectionV3:
    return CameraProjectionV3(
        view_id=view_id,
        view_index=index,
        origin=(0.0, 0.0, -5.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=2.0,
        resolution=128,
    )


def _directional_row(
    rest_xyz: np.ndarray,
    triangles: np.ndarray,
    *,
    view_id: str = "V0",
    view_index: int = 0,
) -> DirectionalBodyAssetV1:
    asset = RuntimeV4AttachmentAsset(
        asset_id=f"body_{view_id}",
        slot_id="BODY_UNDERLAY",
        attachment_id=f"BODY_UNDERLAY__{view_id}",
        attachment_kind=AttachmentKind.DEFORMABLE_BODY,
        topology_class=TopologyClass.STATIC,
        rest_xyz=rest_xyz,
        triangles=triangles,
        sealed_source_hash=("a" if view_index == 0 else "b") * 64,
    )
    return DirectionalBodyAssetV1(
        view_id=view_id,
        view_index=view_index,
        runtime_asset=asset,
        uv=np.zeros((len(rest_xyz), 2), dtype=np.float32),
        weight_matrix=np.ones((len(rest_xyz), 1), dtype=np.float64),
        joint_ids=("root",),
        mesh_lineage_hash=("c" if view_index == 0 else "d") * 64,
        mesh_skin_lineage_hash=("e" if view_index == 0 else "f") * 64,
        source_texture_sha256=("1" if view_index == 0 else "2") * 64,
        source_alpha_recall=0.99,
        precision_inside_alpha=1.0,
    )


def _single_view_contract_and_clip(rest_xyz, triangles, frame0, frame1):
    row = _directional_row(rest_xyz, triangles)
    contract = build_directional_body_runtime_v4_contract_v1(
        (row,),
        {"V0": _camera()},
        root_bone_id="root",
        required_view_ids=("V0",),
    )
    composition = {
        "V0": RuntimeV3FrameComposition(
            "V0",
            ("BODY_UNDERLAY",),
            {"BODY_UNDERLAY": "BODY_UNDERLAY__V0"},
        )
    }
    clip = RuntimeV4Clip(
        "run",
        "Run",
        "RUN",
        1.0,
        30.0,
        False,
        (
            RuntimeV4Frame(0.0, {"body_V0": np.asarray(frame0, dtype=np.float32)}, composition),
            RuntimeV4Frame(1.0, {"body_V0": np.asarray(frame1, dtype=np.float32)}, composition),
        ),
        False,
    )
    return row, contract, clip


def test_directional_contract_never_labels_source_owned_asset_unseen():
    tri = np.asarray(((0, 1, 2),), dtype=np.uint32)
    rest0 = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32)
    rest1 = rest0 + np.asarray((0.0, 0.0, 0.1), dtype=np.float32)
    rows = (
        _directional_row(rest0, tri, view_id="V0", view_index=0),
        _directional_row(rest1, tri, view_id="V1", view_index=1),
    )
    contract = build_directional_body_runtime_v4_contract_v1(
        rows,
        {"V0": _camera("V0", 0), "V1": _camera("V1", 1)},
        root_bone_id="root",
        required_view_ids=("V0", "V1"),
    )

    for target_index, view in enumerate(contract.views):
        for owner_index, overlay in enumerate(view.assets):
            provenance = provenance_from_code(int(overlay.provenance_codes[0]))
            expected = (
                AppearanceProvenance.DIRECT_SOURCE
                if target_index == owner_index
                else AppearanceProvenance.OTHER_VIEW_SOURCE
            )
            assert provenance == expected
            assert int(overlay.donor_view_indices[0]) == owner_index


def test_continuous_certificate_passes_rigid_translation():
    rest = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.uint32)
    moved = rest + np.asarray((0.25, 0.1, 0.0), dtype=np.float32)
    row, contract, clip = _single_view_contract_and_clip(rest, triangles, rest, moved)

    cert = certify_directional_body_motion_v1(
        contract,
        clip,
        (row,),
        required_view_ids=("V0",),
    )
    assert cert.continuous_orientation_certified
    assert cert.boundary_motion_clearance_certified
    assert cert.global_embedding_certified
    assert cert.source_alpha_recall_floor == pytest.approx(0.99)


def test_continuous_certificate_catches_mid_interval_collapse_even_when_endpoints_have_same_orientation():
    rest = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    triangles = np.asarray(((0, 1, 2),), dtype=np.uint32)
    end = np.asarray(
        ((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        dtype=np.float32,
    )
    row, contract, clip = _single_view_contract_and_clip(rest, triangles, rest, end)

    with pytest.raises(
        QualificationError,
        match="DIRECTIONAL_BODY_TRIANGLE_INTERVAL_INVERSION_OR_COLLAPSE",
    ):
        certify_directional_body_motion_v1(
            contract,
            clip,
            (row,),
            required_view_ids=("V0",),
        )


def test_asset_builder_carries_source_coverage_and_exact_skin_rows():
    vertices = (
        SimpleNamespace(canonical_mesh_vertex_id="A", P=(0.0, 0.0, 0.0), metadata={"raster_xy": (0.0, 0.0)}),
        SimpleNamespace(canonical_mesh_vertex_id="B", P=(1.0, 0.0, 0.0), metadata={"raster_xy": (1.0, 0.0)}),
        SimpleNamespace(canonical_mesh_vertex_id="C", P=(0.0, 1.0, 0.0), metadata={"raster_xy": (0.0, 1.0)}),
    )
    mesh = SimpleNamespace(
        vertices=vertices,
        faces=(("A", "B", "C"),),
        view_index=0,
        mesh_lineage_hash="3" * 64,
        qualification_report={
            "source_alpha_recall": 0.991,
            "precision_inside_alpha": 0.999,
        },
    )
    mesh_skin = SimpleNamespace(
        rows=tuple(
            SimpleNamespace(
                canonical_mesh_vertex_id=v.canonical_mesh_vertex_id,
                influences=(("root", 1.0),),
            )
            for v in vertices
        ),
        mesh_skin_lineage_hash="4" * 64,
    )
    row = build_directional_body_asset_v1(
        mesh,
        mesh_skin,
        view_id="V0",
        view_index=0,
        joint_ids=("root",),
        source_width=2,
        source_height=2,
        source_texture_sha256="5" * 64,
    )
    assert row.source_alpha_recall == pytest.approx(0.991)
    assert row.precision_inside_alpha == pytest.approx(0.999)
    assert np.allclose(row.weight_matrix, 1.0)
