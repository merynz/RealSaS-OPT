import torch
from realsas_iris_sees.g1_model import IRISG1SinglePose, G1Config, migrate_n1d_state_dict
from realsas_iris_sees.g1_losses import g1_geometry_loss
from realsas_iris_sees.g1_metrics import g1_geometry_metrics


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
