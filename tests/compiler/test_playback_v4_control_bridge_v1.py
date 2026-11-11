from __future__ import annotations

"""Generic editable control -> canonical 3D clip -> eight-view native bridge."""

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import zlib

import numpy as np
import pytest

from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetSessionV1, VIEW_IDS_V1,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import qualify_camera_v3
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AttachmentKind, AppearanceProvenance, RuntimeV3FrameComposition,
    TopologyClass,
)
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    RuntimeV4AttachmentAsset, RuntimeV4PlaybackContract, RuntimeV4ViewAssetOverlay,
    RuntimeV4ViewOverlay, provenance_code, validate_runtime_v4_clip,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.export.playback_v4_control_bridge_v1 import (
    capture_puppet_pose_keyframe_v1,
    build_runtime_v4_clip_from_puppet_keyframes_v1,
)
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload
from compiler.realsas_compiler_services.export.runtime_v4 import materialize_runtime_v4_archive
from test_controllable_puppet_runtime_v1 import qualified_synthetic_bundle


def _contract(bundle):
    ids = bundle.vertex_ids_by_attachment["A_BODY"]
    vertices = {v.canonical_mesh_vertex_id: v.P for v in bundle.mesh.vertices}
    rest = np.asarray([vertices[i] for i in ids], np.float32)
    asset = RuntimeV4AttachmentAsset(
        "asset:body", "S_BODY", "A_BODY",
        AttachmentKind.DEFORMABLE_BODY, TopologyClass.STATIC,
        rest, np.asarray(((0, 1, 2),), np.uint32), "a" * 64,
    )
    views = []
    for i, name in enumerate(VIEW_IDS_V1):
        mesh = next(m for m in bundle.playback.meshes if m.view_id == name)
        uv = np.asarray([(v.u, v.v) for v in mesh.vertices], np.float32)
        views.append(RuntimeV4ViewOverlay(
            name, i,
            qualify_camera_v3(bundle.cameras_by_view[name], view_id=name, view_index=i),
            (RuntimeV4ViewAssetOverlay(
                "asset:body", uv,
                np.asarray((provenance_code(AppearanceProvenance.DIRECT_SOURCE),), np.uint8),
                np.asarray((i,), np.int16),
            ),),
        ))
    return RuntimeV4PlaybackContract(bundle.playback.slots, (asset,), tuple(views))


def _compositions(session):
    return {
        view: RuntimeV3FrameComposition(
            view, ("S_BODY",), dict(session.active_attachments),
        ) for view in VIEW_IDS_V1
    }


def _fixture():
    bundle = qualified_synthetic_bundle()
    contract = _contract(bundle)
    session = ControllablePuppetSessionV1(bundle)
    before = capture_puppet_pose_keyframe_v1(
        session, time_seconds=0., composition_by_view=_compositions(session),
    )
    session.drag_screen("C_IK", 80., 16.)
    after = capture_puppet_pose_keyframe_v1(
        session, time_seconds=1., composition_by_view=_compositions(session),
    )
    clip, report = build_runtime_v4_clip_from_puppet_keyframes_v1(
        bundle=bundle, contract=contract, captured=(before, after),
        clip_id="custom_001", display_name="Custom motion",
        intent="user_authored", duration_seconds=1., fps=30.,
    )
    return bundle, contract, (before, after), clip, report


def _rebuild(bundle, contract, frames):
    return build_runtime_v4_clip_from_puppet_keyframes_v1(
        bundle=bundle, contract=contract, captured=frames,
        clip_id="custom_001", display_name="Custom motion",
        intent="user_authored", duration_seconds=1., fps=30.,
    )


def test_authoring_preserves_editable_controls_and_exact_3d_poses():
    bundle, contract, keyframes, clip, report = _fixture()
    validate_runtime_v4_clip(contract, clip)
    assert not clip.runtime_qualified and not report.runtime_qualified
    assert report.product_revision_id == bundle.product_revision_id
    assert report.mesh_lineage_hash == bundle.mesh.mesh_lineage_hash
    assert report.skin_lineage_hash == bundle.skin.skin_lineage_hash
    assert report.control_program_hash == bundle.rig.control_program_hash
    assert report.keyframe_times == (0., 1.)
    assert report.control_keyframes[0] == ()
    assert report.control_keyframes[1][0].canonical_control_id == "C_IK"
    assert len(clip.frames[0].canonical_posed_xyz_by_asset) == 1
    assert len(clip.frames[0].composition_by_view) == 8
    first = clip.frames[0].canonical_posed_xyz_by_asset["asset:body"]
    second = clip.frames[1].canonical_posed_xyz_by_asset["asset:body"]
    assert not np.allclose(first, second)
    assert np.allclose(first, np.asarray(keyframes[0].frame.deformed.positions), atol=1e-6)
    assert np.allclose(second, np.asarray(keyframes[1].frame.deformed.positions), atol=1e-6)
    assert _rebuild(bundle, contract, keyframes)[1].bridge_hash == report.bridge_hash


def test_missing_view_or_changed_slot_fails_closed():
    session = ControllablePuppetSessionV1(qualified_synthetic_bundle())
    comp = _compositions(session)
    with pytest.raises(QualificationError, match="EIGHT_VIEW_COMPOSITION_REQUIRED"):
        capture_puppet_pose_keyframe_v1(
            session, time_seconds=0.,
            composition_by_view={k: v for k, v in comp.items() if k != "V7"},
        )
    bad = dict(comp)
    bad["V0"] = replace(comp["V0"], active_attachment_by_slot={"S_BODY": None})
    with pytest.raises(QualificationError, match="ACTIVE_ATTACHMENT_POSE_DRIFT"):
        capture_puppet_pose_keyframe_v1(
            session, time_seconds=0., composition_by_view=bad,
        )


def test_authoring_rejects_fabricated_geometry_and_other_revision():
    bundle, contract, keyframes, _, _ = _fixture()
    bad = replace(keyframes[1].frame, product_revision_id="FOREIGN_REV")
    with pytest.raises(QualificationError, match="PRODUCT_REVISION_DRIFT"):
        _rebuild(bundle, contract, (keyframes[0], replace(keyframes[1], frame=bad)))
    wrong = replace(keyframes[1].frame.deformed, positions=((0., 0., 0.),) * 3)
    bad = replace(keyframes[1].frame, deformed=wrong)
    with pytest.raises(QualificationError, match="DEFORMATION_NOT_REPLAYABLE"):
        _rebuild(bundle, contract, (keyframes[0], replace(keyframes[1], frame=bad)))
    with pytest.raises(QualificationError, match="KEYFRAME_ENDPOINTS_REQUIRED"):
        _rebuild(bundle, contract, (keyframes[1], keyframes[0]))
    altered_asset = replace(contract.assets[0], rest_xyz=contract.assets[0].rest_xyz + 0.1)
    other = replace(contract, assets=(altered_asset,))
    with pytest.raises(QualificationError, match="REST_MESH_ADDRESS_DRIFT"):
        _rebuild(bundle, other, keyframes)


@pytest.mark.skipif(not os.environ.get("REALSAS_V4_NATIVE_DEMO"), reason="native C++ runtime unavailable")
def test_authored_control_clip_changes_real_native_pixels_in_eight_view_package(tmp_path: Path):
    from PIL import Image

    bundle, contract, _, clip, report = _fixture()
    texture_root = tmp_path / "assets"
    textures = []
    for view_id in VIEW_IDS_V1:
        rel = f"textures/{view_id}.png"
        path = texture_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.full((128, 128, 4), (80, 140, 230, 255), dtype=np.uint8),
        ).save(path)
        payload = path.read_bytes()
        textures.append(RuntimeV3TexturePayload(
            view_id, rel, sha256(payload).hexdigest(),
            zlib.crc32(payload) & 0xffffffff, 128, 128,
        ))
    archive = tmp_path / "authored.rss"
    materialize_runtime_v4_archive(
        out_path=archive, texture_root=texture_root,
        contract=contract, textures=tuple(textures), clips=(clip,),
        source_product_state_hash="c" * 64, source_proof_bundle_hash="d" * 64,
    )
    demo = os.environ["REALSAS_V4_NATIVE_DEMO"]
    for i in range(8):
        images = []
        for t in (0., 1.):
            out = tmp_path / f"V{i}-t{t}.png"
            proc = subprocess.run(
                [demo, str(archive), "--clip", clip.clip_id, "--view", f"V{i}",
                 "--time", str(t), "--out", str(out)],
                capture_output=True, text=True,
            )
            assert proc.returncode == 0, proc.stderr
            images.append(np.asarray(Image.open(out).convert("RGBA")))
        assert images[0].shape == (128, 128, 4)
        # The test mesh is exactly planar in world XY. At 90° / 270° it
        # projects to a line. A *correct* full-surface renderer must not
        # fabricate silhouette pixels for an edge-on triangle.
        for pixels in images:
            covered = np.count_nonzero(pixels[..., 3])
            if i in (2, 6):
                assert covered == 0, f"edge-on source was invented at V{i}"
            else:
                assert covered > 0, f"visible canonical surface absent at V{i}"
        if i in (0, 1, 7):
            assert not np.array_equal(images[0], images[1]), f"control effect absent V{i}"
    assert report.runtime_qualified is False
