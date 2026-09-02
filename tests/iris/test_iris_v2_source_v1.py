from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from compiler.realsas_compiler_core.local_geometry import (
    DTB_ND1_HISTORICAL_OPERATOR_BLOB_SHA,
    robust_local_plane_normals,
    dtb_nd1_operator_hash,
)
from experiments.iris_reprojection_v2_20260831.observation_contract_v2 import (
    OrthographicCameraV2,
    ObservationContractV2,
    ObservationFileManifestV2,
    camera_from_renderer_json_v2,
    load_observation_manifest_v2,
)
from experiments.iris_reprojection_v2_20260831.foundation_adapter_v2 import (
    FoundationFeatureContractV2,
    FrozenFoundationAdapterV2,
    save_foundation_cache_v2,
    load_foundation_cache_v2,
)
from experiments.iris_reprojection_v2_20260831.q_domain_v2 import (
    QCandidatePolicyV2,
    build_q_domain_v2,
)
from experiments.iris_reprojection_v2_20260831.model_v2 import IrisReprojectionV2
from experiments.iris_reprojection_v2_20260831.ray_modes_v2 import extract_ray_modes_v2
from experiments.iris_reprojection_v2_20260831.train_v2 import iris_v2_loss
from experiments.iris_reprojection_v2_20260831.world_regularizer_v2 import isotropic_world_regularizer_v2
from experiments.iris_reprojection_v2_20260831.observation_evidence_emitter_v2 import emit_observation_evidence_v2
from experiments.iris_reprojection_v2_20260831.persistence_adapter_v2 import compile_surface_v2
from experiments.iris_reprojection_v2_20260831.checkpoint_v2 import save_checkpoint_v2, load_checkpoint_v2


def camera(v: int, half: float = 1.5) -> OrthographicCameraV2:
    a = 2 * math.pi * v / 8
    f = (-math.sin(a), 0.0, -math.cos(a))
    r = (math.cos(a), 0.0, -math.sin(a))
    u = (0.0, 1.0, 0.0)
    return OrthographicCameraV2(v, (0.0, 0.0, 0.0), r, u, f, half, 1024)


def contract() -> ObservationContractV2:
    return ObservationContractV2("synthetic", tuple(camera(v) for v in range(8)), tuple(f"r{v}" for v in range(8)))


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_offcenter_orthographic_ray_roundtrip():
    c = camera(3); g = np.asarray([0.37, -0.29]); d = 0.8125; o = c.ray_origin_for_grid(g); p = c.point_for_grid_depth(g, d); recovered = o + d * np.asarray(c.forward)
    assert np.max(np.abs(p - recovered)) < 1e-12
    assert np.max(np.abs(c.project_grid(p) - g)) < 1e-12
    assert abs(float(c.depth_for_point(p)) - d) < 1e-12


def test_explicit_manifest_loader_never_guesses_variant(tmp_path: Path):
    from PIL import Image
    rgba_paths=[]; camera_paths=[]
    for v in range(8):
        rgba=tmp_path/f"chosen_source_textured_V{v}.png"; arr=np.zeros((1024,1024,4),np.uint8); arr[...,0]=v; arr[...,3]=255; Image.fromarray(arr,mode="RGBA").save(rgba)
        cam=camera(v,0.54); cp=tmp_path/f"camera_V{v}.json"; cp.write_text(json.dumps({"contract":"realsas.level_orthographic_z_orbit.v1","yaw_deg":45.0*v,"right":cam.right,"screen_up":cam.up,"forward":cam.forward,"half_extent":cam.half_extent,"resolution":1024}),encoding="utf-8")
        rgba_paths.append(str(rgba)); camera_paths.append(str(cp))
    m=ObservationFileManifestV2("synthetic",tuple(rgba_paths),tuple(camera_paths),tuple(sha(Path(x)) for x in rgba_paths),tuple(sha(Path(x)) for x in camera_paths)); images,c=load_observation_manifest_v2(m)
    assert images.shape==(8,1024,1024,4); assert c.raster_authority=="MASTER_SOURCE_TEXTURED_RGBA"; assert tuple(x.view_index for x in c.cameras)==tuple(range(8))


def test_renderer_camera_json_parser_preserves_order(tmp_path: Path):
    c=camera(2,0.54); p=tmp_path/"camera.json"; p.write_text(json.dumps({"contract":"realsas.level_orthographic_z_orbit.v1","yaw_deg":90.0,"right":c.right,"screen_up":c.up,"forward":c.forward,"half_extent":0.54}),encoding="utf-8"); got=camera_from_renderer_json_v2(p,2); assert got.camera_hash
    with pytest.raises(ValueError): camera_from_renderer_json_v2(p,1)


class DummyBackbone(nn.Module):
    def __init__(self): super().__init__(); self.conv=nn.Conv2d(3,4,1,bias=False)
    def forward(self,x): return self.conv(x)


def test_frozen_foundation_adapter_and_cache_parity(tmp_path: Path):
    torch.manual_seed(0); bb=DummyBackbone(); fc=FoundationFeatureContractV2("DINO-S-DUMMY","source-sha",("l0",),(4,),"rgb-unit"); adapter=FrozenFoundationAdapterV2(bb,contract=fc,extract_levels=lambda m,x:(m(x),)); assert not any(p.requires_grad for p in adapter.backbone.parameters())
    images=torch.randn(1,8,3,8,8); maps=adapter(images); path=tmp_path/"foundation.pt"; save_foundation_cache_v2(path,maps=maps,contract=fc,observation_contract_hash="obs"); loaded=load_foundation_cache_v2(path,expected_contract=fc,expected_observation_contract_hash="obs"); assert torch.equal(maps[0],loaded[0])
    with pytest.raises(ValueError): load_foundation_cache_v2(path,expected_contract=fc,expected_observation_contract_hash="wrong")


def test_q_domain_exact_projection_and_explicit_nonuniversal_hull_policy():
    c=contract(); B,Q,D=1,3,5; av=torch.zeros((B,Q),dtype=torch.long); ag=torch.tensor([[[-0.3,0.2],[0.0,0.0],[0.4,-0.1]]],dtype=torch.float32); depths=torch.linspace(0.2,1.0,D); masks=torch.ones((1,8,32,32),dtype=torch.bool); masks[:,7]=False; policy=QCandidatePolicyV2(minimum_foreground_support_views=2); domain=build_q_domain_v2((c,),av,ag,depths,foreground_masks=masks,candidate_policy=policy)
    assert domain.candidate_valid.all(); assert not domain.foreground_support[...,7].any(); assert domain.candidate_policy.minimum_foreground_support_views==2; p=domain.q_points[0,1,2].detach().cpu().numpy()
    for v,cam in enumerate(c.cameras):
        g=cam.project_grid(p); d=cam.depth_for_point(p); assert np.max(np.abs(g-domain.projected_grid[0,1,2,v].cpu().numpy()))<1e-6; assert abs(float(d)-float(domain.projected_depth[0,1,2,v]))<1e-6


def tiny_domain():
    c=contract(); B,Q,D=1,4,9; av=torch.zeros((B,Q),dtype=torch.long); ag=torch.linspace(-0.4,0.4,Q).view(1,Q,1).repeat(1,1,2); ag[...,1]=0.0; depths=torch.linspace(0.2,1.8,D); return c,build_q_domain_v2((c,),av,ag,depths)


def test_model_has_no_learned_P_or_N_head_and_backpropagates_depth_only():
    torch.manual_seed(1); c,domain=tiny_domain(); images=torch.randn(1,8,4,24,24); foundation=(torch.randn(1,8,6,6,6),torch.randn(1,8,10,3,3)); model=IrisReprojectionV2((6,10),hidden_dim=24,max_modes=3); names=tuple(name.lower() for name,_ in model.named_modules()); assert not any(name.endswith("p_head") or name.endswith("n_head") for name in names); out=model(images,foundation,domain); teacher=torch.full((1,4),1.0); support=torch.ones((1,4),dtype=torch.bool); losses=iris_v2_loss(out,domain,teacher,support); losses["total"].backward(); assert torch.isfinite(losses["total"]); assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_multimodal_ray_modes_preserve_separated_modes_and_unknown():
    s=torch.full((1,1,9),-8.0); s[...,2]=4.0; s[...,6]=3.7; m=extract_ray_modes_v2(s,max_modes=3,min_probability=0.01); assert set(m.mode_indices[0,0,:2].tolist())=={2,6}; invalid=extract_ray_modes_v2(s,torch.zeros_like(s,dtype=torch.bool),max_modes=3); assert not invalid.mode_valid.any()


def test_world_regularizer_rotation_and_scale_invariant():
    torch.manual_seed(2); q=torch.randn(1,5,7,3); logits=torch.randn(1,5,7); valid=torch.ones_like(logits,dtype=torch.bool); a=isotropic_world_regularizer_v2(q,logits,valid_mask=valid); theta=0.73; R=torch.tensor([[math.cos(theta),-math.sin(theta),0.0],[math.sin(theta),math.cos(theta),0.0],[0.0,0.0,1.0]]); b=isotropic_world_regularizer_v2((q@R.T)*3.7,logits,valid_mask=valid); assert torch.allclose(a,b,atol=1e-5,rtol=1e-5); valid[:,2]=False; c=isotropic_world_regularizer_v2(q,logits,valid_mask=valid); assert torch.isfinite(c)


def test_emitter_persistence_surface_exact_common_frame_roundtrip():
    torch.manual_seed(3); c,domain=tiny_domain(); images=torch.randn(1,8,4,24,24); foundation=(torch.randn(1,8,6,6,6),torch.randn(1,8,10,3,3)); model=IrisReprojectionV2((6,10),hidden_dim=24,max_modes=1); out=model(images,foundation,domain); out.depth_output.support_probability=torch.ones_like(out.depth_output.support_probability); ev=emit_observation_evidence_v2((c,),domain,out.modes,out.refined_depth,out.depth_output)[0]; surf=compile_surface_v2(ev,max_common_frame_error=1e-5); assert surf.surface_nodes; by_id={s.observation_id:s for s in ev.samples}
    for node in surf.surface_nodes:
        pts=[]
        for oid in node.source_observation_ids:
            s=by_id[oid]; o=np.asarray(s.ray_origin,np.float64); f=np.asarray(s.ray_forward,np.float64); f/=np.linalg.norm(f); pts.append(o+float(s.depth)*f)
        assert np.max(np.abs(np.mean(pts,axis=0)-np.asarray(node.P)))<1e-6


def test_dtb_nd1_plane_behavior_and_frozen_authority():
    res=32; yy,xx=np.meshgrid(np.arange(res),np.arange(res),indexing="ij"); pix=(yy*res+xx).reshape(-1).astype(np.int64); P=np.stack([xx.reshape(-1)*0.01,yy.reshape(-1)*0.01,np.zeros(res*res)],axis=1).astype(np.float32); n,valid=robust_local_plane_normals(pix,P,res); assert valid[res*16+16]; assert abs(abs(float(n[res*16+16,2]))-1.0)<1e-5; assert DTB_ND1_HISTORICAL_OPERATOR_BLOB_SHA=="4e612978c3f70a537dcf3189948dc19a38b6e582"; assert len(dtb_nd1_operator_hash())==64


def test_checkpoint_contract_roundtrip(tmp_path: Path):
    torch.manual_seed(4); model=IrisReprojectionV2((6,),hidden_dim=24,max_modes=1); opt=torch.optim.Adam(model.parameters(),lr=1e-3); p=tmp_path/"iris.pt"; meta=save_checkpoint_v2(p,model=model,optimizer=opt,step=7,config={"hidden":24},source_contract_hash="source-hash"); assert meta["sha256"]; other=IrisReprojectionV2((6,),hidden_dim=24,max_modes=1); payload=load_checkpoint_v2(p,model=other,expected_source_contract_hash="source-hash"); assert payload["step"]==7
    for a,b in zip(model.parameters(),other.parameters()): assert torch.equal(a,b)
    with pytest.raises(ValueError): load_checkpoint_v2(p,model=other,expected_source_contract_hash="wrong")
