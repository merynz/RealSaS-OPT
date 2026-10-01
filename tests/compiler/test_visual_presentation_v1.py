from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    VisualMesh2D,
    visual_mesh_semantic_hash,
)
from compiler.realsas_compiler_core.visual_presentation_v1 import (
    QualifiedVisualPresentationSetIR,
    QualifiedVisualPresentationViewIR,
    load_qualified_visual_presentation_view,
    qualified_visual_presentation_set_from_dict,
    qualified_visual_presentation_set_hash,
    qualified_visual_presentation_view_hash,
)


def _view(tmp_path, view_index: int) -> QualifiedVisualPresentationViewIR:
    positions = np.asarray(
        ((0.0, 0.0), (4.0, 0.0), (0.0, 4.0)),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2),), dtype=np.uint32)
    uv = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = VisualMesh2D(
        positions=positions,
        faces=faces,
        uv=uv,
        width=8,
        height=8,
    )
    path = tmp_path / f"V{view_index}.npz"
    region = np.full((8, 8), -1, dtype=np.int32)
    region[:4, :4] = 0
    seed = region.copy()
    np.savez_compressed(
        path,
        positions=positions,
        faces=faces,
        uv=uv,
        vertex_region_id=np.asarray((0, 0, 0), dtype=np.int32),
        face_region_id=np.asarray((0,), dtype=np.int32),
        region_labels=region,
        seed_region_labels=seed,
    )
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    value = QualifiedVisualPresentationViewIR(
        view_index=view_index,
        direction_id=f"V{view_index}",
        width=8,
        height=8,
        vertex_count=3,
        face_count=1,
        region_count=1,
        mesh_npz_path=str(path),
        mesh_npz_sha256=digest,
        source_visual_mesh_hash="1" * 64,
        source_raster_sha256="2" * 64,
        source_foreground_mask_sha256="3" * 64,
        visual_mesh_hash=visual_mesh_semantic_hash(mesh),
        view_hash="",
        metadata={"dynamic_deformation_qualified": False},
    )
    return replace(
        value,
        view_hash=qualified_visual_presentation_view_hash(value),
    )


def _set(tmp_path) -> QualifiedVisualPresentationSetIR:
    value = QualifiedVisualPresentationSetIR(
        source_visual_mesh_set_binding_hash="4" * 64,
        observation_set_binding_hash="5" * 64,
        output_direction_set_binding_hash="6" * 64,
        mechanical_mesh_binding_hash="7" * 64,
        skin_topology_compatibility_report_hash="8" * 64,
        appearance_asset_binding_hash="9" * 64,
        appearance_qualification_binding_hash="a" * 64,
        presentation_policy_binding_hash="b" * 64,
        views=tuple(_view(tmp_path, i) for i in range(8)),
        set_hash="",
        metadata={
            "authority": "STAGE37_QUALIFIED_VISUAL_PRESENTATION_TOPOLOGY",
            "dynamic_deformation_qualified": False,
        },
    )
    return replace(
        value,
        set_hash=qualified_visual_presentation_set_hash(value),
    )


def test_qualified_visual_presentation_roundtrip_and_npz_bytes(tmp_path):
    value = _set(tmp_path)
    restored = qualified_visual_presentation_set_from_dict(value.to_dict())
    assert restored == value
    loaded = load_qualified_visual_presentation_view(restored.views[0])
    mesh = loaded["mesh"]
    assert mesh.positions.shape == (3, 2)
    assert mesh.faces.shape == (1, 3)
    assert loaded["region_labels"].shape == (8, 8)


def test_qualified_visual_presentation_rejects_set_hash_drift(tmp_path):
    value = _set(tmp_path)
    payload = value.to_dict()
    payload["mechanical_mesh_binding_hash"] = "c" * 64
    with pytest.raises(
        QualificationError,
        match="QUALIFIED_VISUAL_PRESENTATION_SET_HASH_DRIFT",
    ):
        qualified_visual_presentation_set_from_dict(payload)


def test_qualified_visual_presentation_rejects_npz_byte_drift(tmp_path):
    value = _set(tmp_path)
    path = value.views[0].mesh_npz_path
    with open(path, "ab") as handle:
        handle.write(b"drift")
    with pytest.raises(
        QualificationError,
        match="QUALIFIED_VISUAL_PRESENTATION_VIEW_NPZ_BYTES_DRIFT",
    ):
        load_qualified_visual_presentation_view(value.views[0])
