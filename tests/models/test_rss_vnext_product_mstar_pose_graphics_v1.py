from __future__ import annotations

"""CPU source-admission regression, NOT native corpus/GPU or cloth proof."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.rss_vnext.product_mstar_pose_graphics_v1 import (
    NativeMstarPoseRasterTargetV1, NativeMstarPoseRasterWeightsV1,
    qualified_product_mstar_pose_loss_v1, render_posed_indexed_mstar_v1,
    SCHEMA, VISIBLE_POLICY,
)
from models.rss_vnext.training_engine import source_surface_tensor_sha256
from models.rss_vnext.realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)
from tests.models.test_rss_vnext_product_seeded_training_route_v1 import (
    _scenario,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(20261120)
    torch.set_num_threads(1)


def _pose_case(monkeypatch):
    model, batch, *_ = _scenario(monkeypatch)
    state = model.encode_generated_multiview(
        batch.rgba, None,
        generated_rgba=batch.product_generated_rgba,
        generated_normals=batch.product_generated_normals,
        generated_cameras=batch.product_generated_cameras,
        generation_checkpoint_sha256=batch.product_generation_checkpoint_sha256,
        source_binding_sha256=batch.product_source_binding_sha256,
    )
    original = batch.seeded_tessa_source_teacher.source_xyz
    faces = torch.tensor(
        [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]],
        dtype=torch.long,
    )
    _, six = target_canonical_and_six_cameras_v1(
        1, device="cpu", dtype=original.dtype,
    )
    return model, batch, state, original[None], faces, six[:, :1]


def test_canonical_surface_appearance_stays_stable_under_pose_depth_translation(
    monkeypatch,
):
    model, batch, state, rest, faces, cameras = _pose_case(monkeypatch)
    raster = lambda pose: render_posed_indexed_mstar_v1(
        rest_xyz=rest, posed_xyz=pose, faces=faces,
        scene_triplanes=state.scene_triplanes, iris=model.iris,
        graphics_head=model.graphics, role_head=model.presentation_role_head,
        cameras=cameras, height=16, width=16,
    )
    a = raster(rest)
    # Camera-forward translation changes depth, NOT canonical UV/color.
    forward = cameras[0, 0, 9:12]
    b = raster(rest + 0.055 * forward)
    torch.testing.assert_close(a.rgb_white, b.rgb_white, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(a.alpha, b.alpha, atol=1e-4, rtol=1e-4)
    assert not torch.equal(a.depth, b.depth)
    assert a.raster_backend_id.startswith("BUSTED_TORCH")
    assert a.authority == "MODEL_POSED_MSTAR_RENDER_NOT_SOURCE_TRUTH"


def test_posed_mstar_raster_backpropagates_to_geometry_and_graphics(
    monkeypatch,
):
    model, batch, state, rest, faces, cameras = _pose_case(monkeypatch)
    xyz = rest.detach().clone().requires_grad_()
    posed = (xyz + torch.tensor((0.027, 0.015, 0.01))).clone()
    out = render_posed_indexed_mstar_v1(
        rest_xyz=xyz, posed_xyz=posed, faces=faces,
        scene_triplanes=state.scene_triplanes, iris=model.iris,
        graphics_head=model.graphics, role_head=model.presentation_role_head,
        cameras=cameras, height=16, width=16,
    )
    loss = (
        (out.rgb_white - 0.32).square().mean()
        + (out.alpha - 0.72).square().mean()
        + out.normal_xyz.square().mean()
        + out.depth.square().mean() * 0.01
    )
    loss.backward()
    assert xyz.grad is not None and torch.isfinite(xyz.grad).all()
    assert bool(torch.any(xyz.grad.abs() > 0))
    assert model.graphics[-1].weight.grad is not None
    assert bool(torch.any(model.graphics[-1].weight.grad.abs() > 0))


def _admitted_case(monkeypatch):
    model, batch, state, rest, faces, cameras = _pose_case(monkeypatch)
    num_pixels = 16 * 16
    target = NativeMstarPoseRasterTargetV1(
        pose_index=0, cameras=cameras,
        height=16, width=16,
        rgb_white=torch.full((1, num_pixels, 3), .67),
        alpha=torch.full((1, num_pixels, 1), .8),
        rgb_mask=torch.ones((1, num_pixels), dtype=torch.bool),
        alpha_mask=torch.ones((1, num_pixels), dtype=torch.bool),
        source_3d_truth_binding_sha256="e"*64,
    )
    source = batch.seeded_tessa_source_teacher
    support = SimpleNamespace(receipt_sha256=lambda: "c"*64)
    native = SimpleNamespace(
        native_pose_transport=support, source_asset_sha256=source.source_asset_sha256,
        source_pose_matrices=torch.eye(4).reshape(1, 1, 1, 4, 4),
        source_pose_mask=torch.ones((1, 1, rest.shape[1]), dtype=torch.bool),
    )
    b = SimpleNamespace(
        rgba=batch.rgba, camera_features=batch.camera_features,
        product_source_binding_sha256=batch.product_source_binding_sha256,
        product_generation_checkpoint_sha256=batch.product_generation_checkpoint_sha256,
        source_qualification_key=batch.source_qualification_key,
        axis_teacher_sequence=object(),
    )
    mesh = SimpleNamespace(
        rest_vertices_iris=rest.detach().clone().requires_grad_(),
        faces=faces,
        seeded_surface=object(),
    )
    pose = mesh.rest_vertices_iris[:, None].clone()
    dig = source_surface_tensor_sha256
    registry = {
        "valid-pose-raster": {
            "schema": SCHEMA,
            "admission_status": "QUALIFIED_SOURCE_NATIVE_MSTAR_POSE_RENDER",
            "dataset_manifest_sha256": "f"*64,
            "source_asset_sha256": source.source_asset_sha256,
            "source_qualification_key": b.source_qualification_key,
            "observed_rgba_sha256": dig(b.rgba),
            "original_camera_sha256": dig(b.camera_features),
            "source_binding_sha256": b.product_source_binding_sha256,
            "generation_checkpoint_sha256": b.product_generation_checkpoint_sha256,
            "candidate_mstar_topology_address_sha256": "a"*64,
            "axis_teacher_sequence_sha256": "b"*64,
            "native_pose_transport_receipt_sha256": "c"*64,
            "source_pose_matrices_sha256": dig(native.source_pose_matrices),
            "source_pose_mask_sha256": dig(native.source_pose_mask),
            "pose_index": 0,
            "render_height": 16, "render_width": 16,
            "source_camera_sha256": dig(cameras),
            "target_rgb_white_sha256": dig(target.rgb_white),
            "target_alpha_sha256": dig(target.alpha),
            "target_rgb_mask_sha256": dig(target.rgb_mask),
            "target_alpha_mask_sha256": dig(target.alpha_mask),
            "target_role_sha256": None,
            "target_role_mask_sha256": None,
            "target_normal_sha256": None,
            "target_normal_mask_sha256": None,
            "target_depth_sha256": None,
            "target_depth_mask_sha256": None,
            "source_3d_truth_binding_sha256": "e"*64,
            "source_visibility_authority": VISIBLE_POLICY,
            "source_renderer_independent_of_student": True,
            "generated_six_views_are_not_source_gt": True,
            "source_native_pose_was_verified_before_renderer": True,
            "source_material_opacity_policy": "OPAQUE_SURFACE_ONLY",
            "source_truth_is_loss_only": True,
        }
    }
    def run(*, t=target, reg=registry, p=pose, n=native):
        return qualified_product_mstar_pose_loss_v1(
            model=model, state=state, mesh=mesh,
            predicted_posed_xyz=p, batch=b, seeded_teacher=n,
            target=t, admission_key="valid-pose-raster",
            trusted_registry=reg, dataset_manifest_sha256="f"*64,
            loss_weights=NativeMstarPoseRasterWeightsV1(),
            tensor_digest=dig,
            axis_sequence_digest=lambda seq:"b"*64,
            seeded_mstar_address_digest=lambda seed:"a"*64,
        )
    return run, target, registry, native, mesh


def test_independently_admitted_pose_raster_loss_is_differentiable(monkeypatch):
    run, target, registry, native, mesh = _admitted_case(monkeypatch)
    loss, parts, pixels = run()
    assert set(parts) == {"rgb", "alpha"}
    assert pixels == 256 and loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert mesh.rest_vertices_iris.grad is not None
    assert bool(torch.any(mesh.rest_vertices_iris.grad.abs() > 0))


def test_pose_teacher_tampering_or_invented_motion_is_rejected(monkeypatch):
    run, target, registry, native, mesh = _admitted_case(monkeypatch)
    with pytest.raises(ValueError, match="SOURCE_RECEIPT_DRIFT"):
        run(t=replace(target, rgb_white=target.rgb_white*.5))
    forged = {"valid-pose-raster":dict(registry["valid-pose-raster"])}
    forged["valid-pose-raster"]["source_renderer_independent_of_student"] = False
    with pytest.raises(ValueError, match="SOURCE_RECEIPT_DRIFT"):
        run(reg=forged)
    with pytest.raises(ValueError, match="NATIVE_POSE_REQUIRED"):
        run(n=SimpleNamespace(
            native_pose_transport=None,
            source_pose_matrices=native.source_pose_matrices,
            source_pose_mask=native.source_pose_mask,
        ))
    with pytest.raises(ValueError, match="POSE_ADDRESS_INVALID"):
        run(n=SimpleNamespace(
            native_pose_transport=native.native_pose_transport,
            source_asset_sha256=native.source_asset_sha256,
            source_pose_matrices=native.source_pose_matrices,
            source_pose_mask=torch.zeros_like(native.source_pose_mask),
        ))
    with pytest.raises(ValueError, match="OPTIONAL_PAIR"):
        run(t=replace(target, normal_xyz=torch.zeros(1,256,3)))


def test_2d_mstar_color_and_role_interpolation_matches_compiler_vertex_logit_abi(
    monkeypatch,
):
    """Compiler UV: barycentric average of vertex LOGITS, then sigmoid.

    Semantic role sublayer: mean vertex softmax per face, NOT softmax of
    interpolated logits. These two ABI invariants are independently checked
    against the native first-hit raster, preventing trainer-vs-Compiler drift.
    """
    from models.rss_vnext.busted_raster_backend_v1 import BustedTorchRasterV1

    model, _, state, rest, faces, cameras = _pose_case(monkeypatch)
    posed = rest + torch.tensor([0.025, -0.01, 0.04])
    renderer = render_posed_indexed_mstar_v1(
        rest_xyz=rest, posed_xyz=posed, faces=faces,
        scene_triplanes=state.scene_triplanes, iris=model.iris,
        graphics_head=model.graphics, role_head=model.presentation_role_head,
        cameras=cameras, height=16, width=16,
    )
    # Compiler consumes the exact same final M* vertex graphics logits.
    with torch.no_grad():
        h = model.iris.query(state.scene_triplanes, rest)["H"]
        logits = model.graphics(h)[0].float()
        role_logits = model.presentation_role_head(h)[0].float()
        vertex_role_probs = torch.softmax(role_logits, dim=-1)
    camera = cameras[0, 0].float()
    xyz = posed[0].float()
    delta = xyz - camera[None, :3]
    right = (delta @ camera[3:6]) / camera[12]
    up = (delta @ camera[6:9]) / camera[12]
    forward = delta @ camera[9:12]
    low = forward.min() - .05
    high = forward.max() + .05
    clip = torch.stack((
        right, up, (forward-low)*2/(high-low)-1,
        torch.ones_like(right),
    ), dim=-1)[None]
    raster, _ = BustedTorchRasterV1.rasterize(
        BustedTorchRasterV1.RasterizeCudaContext(),
        clip, faces.to(torch.int32), (16, 16),
    )
    ids = raster[0, ..., 3].long() - 1
    coverage = ids >= 0
    # The Blender / UV compiler operates in per-face barycentric texel
    # coordinates, which the first-hit raster reports as w0,w1,w2.
    face_logit = logits[faces[ids.clamp_min(0)]]
    w0, w1 = raster[0, ..., 0:1], raster[0, ..., 1:2]
    mixed_logits = (
        face_logit[..., 0, :]*w0
        + face_logit[..., 1, :]*w1
        + face_logit[..., 2, :]*(1.-w0-w1)
    )
    face_mean_role = vertex_role_probs[faces].mean(dim=1)
    role_expect = (
        face_mean_role[ids.clamp_min(0)]*coverage[...,None]
    ).flip(dims=(0,)).reshape(1, 256, -1)
    # Native raster AA multiplies predicted RGB once by edge coverage.
    base_color = torch.sigmoid(mixed_logits[..., :3])*coverage[...,None]
    alpha = raster[0, ..., 2:3] * coverage[...,None]
    rgb_expect = (base_color*alpha + 1. - alpha).flip(dims=(0,)).reshape(1,256,3)
    torch.testing.assert_close(
        renderer.rgb_white, rgb_expect, rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        renderer.roles, role_expect, rtol=1e-5, atol=1e-6,
    )
