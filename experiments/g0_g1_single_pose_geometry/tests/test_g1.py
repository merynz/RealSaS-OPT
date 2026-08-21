import numpy as np
from PIL import Image
import torch
from realsas_iris_sees.g1_model import IRISG1SinglePose, G1Config, migrate_n1d_state_dict
from realsas_iris_sees.g1_losses import g1_geometry_loss
from realsas_iris_sees.g1_metrics import g1_geometry_metrics, aggregate_g1_metric_rows, g1_checkpoint_selection_key
from realsas_iris_sees.g1_io import load_g1_sample, G1_FORBIDDEN_GPU_TARGET_KEYS
from realsas_iris_sees.g1_targets import build_g1_target_from_sidecar


def _target(B=1, N=12, image_size=256):
    P = torch.randn(B, N, 3).clamp(-.4, .4)
    Nrm = torch.nn.functional.normalize(torch.randn(B, N, 3), dim=-1)
    XY = torch.rand(B, 8, N, 2) * (image_size - 1)
    V = torch.rand(B, 8, N) > .2
    return {'P_A':P,'N_A':Nrm,'XY_A':XY,'V_A':V,'direct_obs_A':V.any(1)}


def test_forward_contract():
    model = IRISG1SinglePose(G1Config(image_size=128))
    x = torch.randn(1,8,4,128,128)
    with torch.no_grad(): out = model(x)
    assert out['point'].shape[:3] == (1,8,3)
    assert out['normal'].shape == out['point'].shape
    assert out['log_sigma'].shape[:3] == (1,8,1)
    assert out['visibility_logit'].shape[:3] == (1,8,1)
    assert 'descriptor_z' in out
    assert not any(k.startswith('delta') or k.startswith('transport') for k in out)


def test_loss_and_metrics_finite():
    model = IRISG1SinglePose(G1Config(image_size=128))
    x = torch.randn(1,8,4,128,128)
    out = model(x)
    t = _target()
    loss, parts = g1_geometry_loss(out,t,image_size=256)
    assert torch.isfinite(loss)
    assert set(parts) == {'P','N','V','total'}
    with torch.no_grad():
        metrics = g1_geometry_metrics(out,t,image_size=256)
    for k,v in metrics.items():
        if isinstance(v,float): assert v == v, k


def test_migration_report():
    model = IRISG1SinglePose()
    src = {k:v.clone() for k,v in model.state_dict().items()}
    src['differential.fake'] = torch.zeros(1)
    r = migrate_n1d_state_dict(model,src)
    assert r['loaded_fraction'] == 1.0
    assert 'differential.fake' in r['archived_only_source_keys']


def test_passive_modules_are_frozen():
    model = IRISG1SinglePose()
    assert model.descriptor is not None
    assert not any(p.requires_grad for p in model.descriptor.parameters())
    assert not any(p.requires_grad for p in model.camera_residual.parameters())


def test_g1_io_excludes_pose_b_and_mechanics_truth(tmp_path):
    pose = tmp_path / "poseA"
    pose.mkdir()
    for i in range(8):
        a = np.zeros((16, 16, 3), np.uint8)
        a[..., 1] = 255
        a[4:12, 4:12] = np.array([255, 0, 0], np.uint8)
        Image.fromarray(a).save(pose / f"{i:02d}_V{i}.png")
    n = 12
    target = tmp_path / "target.npz"
    np.savez_compressed(
        target,
        family_id=np.array(42, np.int32),
        P_A=np.zeros((n, 3), np.float32),
        N_A=np.tile(np.array([[0, 1, 0]], np.float32), (n, 1)),
        XY_A=np.zeros((8, n, 2), np.float32),
        V_A=np.ones((8, n), np.uint8),
        direct_obs_A=np.ones((n,), np.uint8),
        P_B=np.ones((n, 3), np.float32),
        scene_flow=np.ones((n, 3), np.float32),
        carrier_id_TRAINING_ONLY=np.arange(n, dtype=np.int16),
    )
    images, g1, fid = load_g1_sample(pose, target, image_size=16)
    assert tuple(images.shape) == (1, 8, 4, 16, 16)
    assert fid == 42
    assert set(g1) == {"P_A", "N_A", "XY_A", "V_A", "direct_obs_A"}
    assert not set(g1).intersection(G1_FORBIDDEN_GPU_TARGET_KEYS)


def test_panel_aggregation_has_hard_tail_and_stable_key():
    rows = []
    for i in range(4):
        rows.append({
            "family_id": i,
            "point_p95": 0.01 + i * 0.01,
            "point_mean": 0.005 + i * 0.005,
            "crossview_point_spread_p95": 0.004 + i * 0.002,
            "normal_p95_deg": 10.0 + i,
            "visibility_brier": 0.1 + i * 0.01,
            "visible_sample_count": 100,
            "direct_carrier_count": 50,
        })
    panel = aggregate_g1_metric_rows(rows)
    assert panel["family_count"] == 4
    assert "point_p95__family_p95" in panel
    assert panel["visible_sample_count"] == 400
    key = g1_checkpoint_selection_key(panel)
    assert len(key) == 6 and all(v < 0 for v in key)


def test_a_only_sidecar_projection_ignores_pose_b(tmp_path):
    n = 6
    side = tmp_path / "sidecar.npz"
    pts = np.arange(n * 3, dtype=np.float32).reshape(n, 3) * 0.01
    normals = np.tile(np.array([[0.0, 2.0, 0.0]], np.float32), (n, 1))
    xy = np.zeros((8, n, 2), np.float32)
    vis = np.full((8, n), 2, np.uint8)
    np.savez_compressed(
        side,
        family_id=np.array(7, np.int32),
        camera_center=np.zeros(3, np.float32),
        camera_half_extent=np.array(0.5, np.float32),
        surface_points_A=pts,
        surface_normals_A=normals,
        surface_xy_A=xy,
        surface_visibility_A=vis,
        # Deliberately nonsensical B payload: G1 projector must never depend on it.
        surface_points_B=np.full((1, 1), np.nan, np.float32),
        surface_normals_B=np.full((2,), np.nan, np.float32),
    )
    out = build_g1_target_from_sidecar(side)
    assert int(out["family_id"]) == 7
    assert out["P_A"].shape == (n, 3)
    assert np.allclose(out["N_A"][:, 1], 1.0)
    assert out["V_A"].all() and out["direct_obs_A"].all()
    assert not any(k.endswith("_B") for k in out)
