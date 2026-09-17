from __future__ import annotations

import numpy as np
import pytest

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
    provenance_code,
    validate_playback_runtime_v4_contract,
    validate_runtime_v4_clip,
)
from compiler.realsas_compiler_core.types import QualificationError


def _camera(view_id: str, view_index: int) -> CameraProjectionV3:
    return CameraProjectionV3(
        view_id=view_id,
        view_index=view_index,
        origin=(0.0, 0.0, -1.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=1.0,
        resolution=2,
    )


def _fixture():
    rest = np.asarray(((-1.0, 1.0, 0.2), (1.0, 1.0, 0.2), (-1.0, -1.0, 0.2)), dtype=np.float32)
    faces = np.asarray(((0, 1, 2),), dtype=np.uint32)
    asset = RuntimeV4AttachmentAsset(
        asset_id="body_asset",
        slot_id="body_slot",
        attachment_id="body_attachment",
        attachment_kind=AttachmentKind.DEFORMABLE_BODY,
        topology_class=TopologyClass.STATIC,
        rest_xyz=rest,
        triangles=faces,
        sealed_source_hash="a" * 64,
    )
    views = []
    for i in range(8):
        views.append(RuntimeV4ViewOverlay(
            view_id=f"V{i}",
            view_index=i,
            camera=_camera(f"V{i}", i),
            assets=(RuntimeV4ViewAssetOverlay(
                asset_id="body_asset",
                uv=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
                provenance_codes=np.asarray((provenance_code(AppearanceProvenance.DIRECT_SOURCE),), dtype=np.uint8),
                donor_view_indices=np.asarray((i,), dtype=np.int16),
            ),),
        ))
    contract = RuntimeV4PlaybackContract(
        slots=(RuntimeV3Slot("body_slot", "root", 0, "body_attachment"),),
        assets=(asset,),
        views=tuple(views),
    )
    compositions = {
        f"V{i}": RuntimeV3FrameComposition(
            view_id=f"V{i}",
            draw_order_slot_ids=("body_slot",),
            active_attachment_by_slot={"body_slot": "body_attachment"},
        )
        for i in range(8)
    }
    frame0 = RuntimeV4Frame(0.0, {"body_asset": rest.copy()}, compositions)
    frame1 = RuntimeV4Frame(1.0, {"body_asset": rest.copy()}, compositions)
    clip = RuntimeV4Clip("idle", "Idle", "idle", 1.0, 30.0, False, (frame0, frame1), True)
    return contract, clip


def test_v4_contract_shares_one_canonical_asset_across_eight_views():
    contract, clip = _fixture()
    h1 = validate_playback_runtime_v4_contract(contract)
    h2 = validate_playback_runtime_v4_contract(contract)
    validate_runtime_v4_clip(contract, clip)
    assert h1 == h2
    assert len(contract.assets) == 1
    assert len(contract.views) == 8
    assert all(view.assets[0].asset_id == "body_asset" for view in contract.views)


def test_v4_frame_stores_canonical_pose_once_not_once_per_view():
    contract, clip = _fixture()
    frame = clip.frames[0]
    assert tuple(frame.canonical_posed_xyz_by_asset) == ("body_asset",)
    assert frame.canonical_posed_xyz_by_asset["body_asset"].nbytes == 3 * 3 * 4
    assert len(frame.composition_by_view) == 8
    validate_runtime_v4_clip(contract, clip)


def test_direct_source_overlay_donor_must_equal_target_view():
    contract, _ = _fixture()
    bad_view = contract.views[3]
    bad_row = RuntimeV4ViewAssetOverlay(
        asset_id="body_asset",
        uv=bad_view.assets[0].uv,
        provenance_codes=bad_view.assets[0].provenance_codes,
        donor_view_indices=np.asarray((0,), dtype=np.int16),
    )
    views = list(contract.views)
    views[3] = RuntimeV4ViewOverlay("V3", 3, bad_view.camera, (bad_row,))
    bad = RuntimeV4PlaybackContract(contract.slots, contract.assets, tuple(views))
    with pytest.raises(QualificationError, match="RUNTIME_V4_DIRECT_SOURCE_DONOR_MUST_MATCH_TARGET_VIEW"):
        validate_playback_runtime_v4_contract(bad)


def test_v4_frame_asset_set_is_exact_and_fail_closed():
    contract, clip = _fixture()
    bad_frame = RuntimeV4Frame(0.0, {}, clip.frames[0].composition_by_view)
    bad_clip = RuntimeV4Clip("idle", "Idle", "idle", 1.0, 30.0, False, (bad_frame, clip.frames[1]))
    with pytest.raises(QualificationError, match="RUNTIME_V4_FRAME_ASSET_SET_MISMATCH"):
        validate_runtime_v4_clip(contract, bad_clip)


def test_v4_completion_remains_disabled_by_default():
    contract, _ = _fixture()
    views = list(contract.views)
    row = views[0].assets[0]
    completion = RuntimeV4ViewAssetOverlay(
        row.asset_id,
        row.uv,
        np.asarray((provenance_code(AppearanceProvenance.COMPLETION),), dtype=np.uint8),
        np.asarray((-1,), dtype=np.int16),
    )
    views[0] = RuntimeV4ViewOverlay("V0", 0, views[0].camera, (completion,))
    bad = RuntimeV4PlaybackContract(contract.slots, contract.assets, tuple(views))
    with pytest.raises(QualificationError, match="RUNTIME_V4_COMPLETION_NOT_ALLOWED_BY_PRODUCT_POLICY"):
        validate_playback_runtime_v4_contract(bad)
