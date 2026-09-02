from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import QualifiedJoint, RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from experiments.geppetto_arachne_r6_20260901.arachne_candidate_v2 import ArachneCandidateConfigV2, ArachneCandidateV2
from experiments.geppetto_arachne_r6_20260901.codec_deformation_loss_v1 import codec_deformation_loss_v1
from experiments.geppetto_arachne_r6_20260901.conditioning_v2 import ArachneConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.eval_arachne_r6_a1_v1 import eval_arachne_r6_a1_v1
from experiments.geppetto_arachne_r6_20260901.eval_codec_r6_a0_v1 import eval_codec_r6_a0_v1
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_checkpoint_v1 import load_skin_field_codec_checkpoint_v1, save_skin_field_codec_checkpoint_v1
from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import SkinFieldCodecV1
from experiments.geppetto_arachne_r6_20260901.skin_target_rebind_v1 import rebind_skin_target_v1
from experiments.geppetto_arachne_r6_20260901.train_arachne_r6_a1_v1 import train_arachne_r6_a1_step_v1
from experiments.geppetto_arachne_r6_20260901.train_codec_r6_a0_v1 import CodecA0QualificationTokenV1, train_codec_r6_a0_step_v1
from experiments.geppetto_arachne_r6_20260901.verified_lbs_v1 import apply_verified_lbs_v1, mutate_weights_swap_mass_v1, verified_lbs_report_v1
from experiments.geppetto_arachne_r6_20260901.candidate_config_v1 import SkinFieldCodecConfigV1
from experiments.single_family_e2e_v1.mechanical_truth_adapter_v1 import align_mechanical_skin_truth_v1

OP_HASH = "dtb-nd1-test-operator-hash"


def _surface(prefix: str, count: int, *, op_hash: str = OP_HASH) -> RiggingSurfaceIR:
    nodes=[]
    for i in range(count):
        nodes.append(SurfaceNode(
            f"{prefix}:S:{i}",
            (float(i)*0.2, float(i%2)*0.15, float(i%3)*0.07),
            (0,1,2,3,4,5,6,7),
            (f"o:{i}",),
            (f"o:{i}",),
            raster_bindings=((0,(-0.8+0.2*i,-0.7+0.1*i)),),
            derived_normal=(0.0,0.0,1.0),
        ))
    rel=tuple(SurfaceRelation(f"R:{i}",nodes[i].surface_id,nodes[i+1].surface_id,"LOCAL",1.0) for i in range(max(0,count-1)))
    return RiggingSurfaceIR(tuple(nodes),rel,geometry_lineage_hash=f"surface-{prefix}",metadata={"Nd_operator_sha256":op_hash,"raster_coordinate_system":"GRID_XY","resolution":1024})


def _skeleton(prefix: str, parents=(-1,0,1)) -> QualifiedSkeletonIRV2:
    joints=[]
    for i,parent in enumerate(parents):
        joints.append(QualifiedJoint(
            f"J:{prefix}:{i}",
            (float(i)*0.35, 0.0, 0.0),
            None if parent < 0 else f"J:{prefix}:{parent}",
            (f"{prefix}:S:{min(i,2)}",),
            f"P:{prefix}:{i}",
        ))
    roots=tuple(j.canonical_joint_id for j in joints if j.parent_canonical_id is None)
    return QualifiedSkeletonIRV2(tuple(joints),roots,{"deforming":False},{"status":"PASS"},f"skeleton-{prefix}-{'-'.join(map(str,parents))}")


def _codec() -> SkinFieldCodecV1:
    return SkinFieldCodecV1(SkinFieldCodecConfigV1(hidden_dim=32,latent_dim=8,encoder_layers=2,decoder_layers=2))


def _teacher(n: int, j: int) -> torch.Tensor:
    raw=torch.arange(1,n*j+1,dtype=torch.float32).reshape(n,j)
    return (raw/raw.sum(dim=1,keepdim=True))[None]


def _probe_transforms(j: int) -> torch.Tensor:
    t=torch.eye(4,dtype=torch.float32)[None,None].repeat(1,2,j,1,1)
    for ji in range(j):
        t[0,1,ji,0,3]=0.05*(ji+1)
        t[0,1,ji,1,3]=-0.03*ji
    return t


def _conditioning(prefix="A", parents=(-1,0,1)):
    surface=_surface(prefix,6); skeleton=_skeleton(prefix,parents); cond=ArachneConditioningAdapterV2()([surface],[skeleton]); return surface,skeleton,cond


def test_skin_target_rebind_exact_lineage_and_axes():
    surface,skeleton,cond=_conditioning("A")
    source_w=np.asarray(_teacher(6,3)[0].numpy(),np.float32)
    mechanical=align_mechanical_skin_truth_v1(surface,skeleton,teacher_surface_ids=tuple(reversed(cond.surface_ids[0])),anonymous_control_ids=tuple(reversed([j.source_proposal_id for j in skeleton.joints])),dense_weights=source_w[::-1,::-1])
    target=rebind_skin_target_v1(mechanical,surface_ids=cond.surface_ids[0],canonical_joint_ids=cond.joint_ids[0],surface_binding_hash=surface.geometry_lineage_hash,skeleton_binding_hash=skeleton.skeleton_lineage_hash)
    target.validate(); assert target.weights.shape==(6,3); np.testing.assert_allclose(target.weights.sum(1),1.0,atol=1e-6)
    with pytest.raises(ValueError,match="skeleton lineage mismatch"):
        rebind_skin_target_v1(mechanical,surface_ids=cond.surface_ids[0],canonical_joint_ids=cond.joint_ids[0],surface_binding_hash=surface.geometry_lineage_hash,skeleton_binding_hash="wrong")


def test_codec_variable_nj_simplex_and_checkpoint_reproducibility(tmp_path):
    s1,g1,c1=_conditioning("B"); s2=_surface("C",4); g2=_skeleton("C",(-1,0)); cond=ArachneConditioningAdapterV2()([s1,s2],[g1,g2]); codec=_codec(); sf=torch.tensor(cond.surface_features); jf=torch.tensor(cond.joint_features); sm=torch.tensor(cond.surface_mask); jm=torch.tensor(cond.joint_mask); teacher=torch.zeros((2,sf.shape[1],jf.shape[1]),dtype=torch.float32); teacher[0,:6,:3]=_teacher(6,3)[0]; teacher[1,:4,:2]=_teacher(4,2)[0]; out=codec(sf,jf,teacher,sm,jm)
    for b,(n,j) in enumerate(((6,3),(4,2))):
        torch.testing.assert_close(out.decoded_weights[b,:n,:j].sum(-1),torch.ones(n),atol=1e-6,rtol=1e-6); assert (out.decoded_weights[b,:n,:j]>=0).all()
    path=tmp_path/"codec.pt"; save_skin_field_codec_checkpoint_v1(path,codec,metadata={"optimizer_steps":0}); restored=_codec(); load_skin_field_codec_checkpoint_v1(path,restored); out2=restored(sf,jf,teacher,sm,jm); torch.testing.assert_close(out.decoded_weights,out2.decoded_weights,atol=0,rtol=0)


def test_verified_lbs_teacher_exact_and_controlled_weight_mutation_worsens():
    rest=np.asarray([[0,0,0],[0.3,0.1,0],[0.8,0,0]],np.float32); w=np.asarray([[1,0],[0.5,0.5],[0,1]],np.float32); transforms=_probe_transforms(2)[0].numpy(); expected=apply_verified_lbs_v1(rest,w,transforms); report=verified_lbs_report_v1(apply_verified_lbs_v1(rest,w,transforms),expected); assert report.max_error==0.0; mutated=mutate_weights_swap_mass_v1(w,joint_a=0,joint_b=1,fraction=0.5); bad=verified_lbs_report_v1(apply_verified_lbs_v1(rest,mutated,transforms),expected); assert bad.rms>report.rms and bad.max_error>0


def test_codec_deformation_loss_is_causal_and_a0_token_is_thresholded():
    surface,skeleton,cond=_conditioning("D"); codec=_codec(); sf=torch.tensor(cond.surface_features); jf=torch.tensor(cond.joint_features); sm=torch.tensor(cond.surface_mask); jm=torch.tensor(cond.joint_mask); teacher=_teacher(6,3); rest=torch.tensor([n.P for n in surface.surface_nodes],dtype=torch.float32)[None]; transforms=_probe_transforms(3); exact=codec_deformation_loss_v1(teacher,teacher,rest,transforms,sm,jm); mutated=teacher.clone(); moved=0.4*mutated[:,:,0].clone(); mutated[:,:,0]-=moved; mutated[:,:,1]+=moved; bad=codec_deformation_loss_v1(mutated,teacher,rest,transforms,sm,jm); assert float(exact["deformation_rms"])<1e-5; assert float(bad["deformation_rms"])>float(exact["deformation_rms"])
    metrics,token=eval_codec_r6_a0_v1(codec,sf,jf,teacher,sm,jm,rest,transforms,source_gate="SYNTHETIC",optimizer_steps=0,max_reconstruction=100.0,max_deformation_rms=100.0,max_simplex_residual=1e-5); token.validate_for(codec); assert token.status=="PASS"
    _,failed=eval_codec_r6_a0_v1(codec,sf,jf,teacher,sm,jm,rest,transforms,source_gate="SYNTHETIC",optimizer_steps=0,max_reconstruction=0.0,max_deformation_rms=0.0,max_simplex_residual=0.0)
    assert failed.status=="FAIL"
    with pytest.raises(ValueError,match="not PASS"):
        failed.validate_for(codec)


def test_codec_a0_train_step_has_deformation_consequence():
    surface,skeleton,cond=_conditioning("E"); codec=_codec(); sf=torch.tensor(cond.surface_features); jf=torch.tensor(cond.joint_features); sm=torch.tensor(cond.surface_mask); jm=torch.tensor(cond.joint_mask); teacher=_teacher(6,3); rest=torch.tensor([n.P for n in surface.surface_nodes],dtype=torch.float32)[None]; transforms=_probe_transforms(3); opt=torch.optim.AdamW(codec.parameters(),lr=1e-4); metrics=train_codec_r6_a0_step_v1(codec,opt,sf,jf,teacher,sm,jm,rest,transforms); assert np.isfinite(metrics["total"]) and metrics["deformation_rms"]>=0


def test_segment_geometry_and_parent_topology_are_causal():
    surface=_surface("F",6); g_chain=_skeleton("F",(-1,0,1)); g_roots=_skeleton("F",(-1,-1,1)); c_chain=ArachneConditioningAdapterV2()([surface],[g_chain]); c_roots=ArachneConditioningAdapterV2()([surface],[g_roots]); assert c_chain.conditioning_hashes[0]!=c_roots.conditioning_hashes[0]; assert not np.allclose(c_chain.pair_geometry,c_roots.pair_geometry); seg_idx=4; assert np.any(c_chain.pair_geometry[0,:6,1,seg_idx] < c_chain.pair_geometry[0,:6,1,3] + 1e-6)


def test_arachne_v2_shared_frozen_decoder_uncertainty_a1_gate_and_compiler_roundtrip():
    surface,skeleton,cond=_conditioning("G"); codec=_codec(); model=ArachneCandidateV2(codec,ArachneCandidateConfigV2(model_dim=32,surface_encoder_layers=1,attention_heads=4,feedforward_dim=64)); assert model.codec is codec; assert all(not p.requires_grad for p in codec.parameters()); sf=torch.tensor(cond.surface_features); jf=torch.tensor(cond.joint_features); sm=torch.tensor(cond.surface_mask); jm=torch.tensor(cond.joint_mask); pi=torch.tensor(cond.parent_indices); pg=torch.tensor(cond.pair_geometry); pm=torch.tensor(cond.pair_mask); out=model(sf,jf,sm,jm,pi,pg,pm); assert torch.isfinite(out.joint_latent_log_sigma).all(); torch.testing.assert_close(out.decoded_weights[0,:6,:3].sum(-1),torch.ones(6),atol=1e-6,rtol=1e-6)
    teacher=_teacher(6,3); rest=torch.tensor([n.P for n in surface.surface_nodes],dtype=torch.float32)[None]; transforms=_probe_transforms(3); _,token=eval_codec_r6_a0_v1(codec,sf,jf,teacher,sm,jm,rest,transforms,source_gate="SYNTHETIC",optimizer_steps=0,max_reconstruction=100.0,max_deformation_rms=100.0,max_simplex_residual=1e-5); fail=CodecA0QualificationTokenV1("FAIL",codec.config.config_hash,"SYNTH",0,"c","m")
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    with pytest.raises(ValueError,match="not PASS"):
        train_arachne_r6_a1_step_v1(model,opt,cond,teacher,rest,transforms,a0_token=fail,expected_surface_hashes=cond.source_surface_hashes,expected_skeleton_hashes=cond.source_skeleton_hashes)
    metrics=train_arachne_r6_a1_step_v1(model,opt,cond,teacher,rest,transforms,a0_token=token,expected_surface_hashes=cond.source_surface_hashes,expected_skeleton_hashes=cond.source_skeleton_hashes); assert np.isfinite(metrics["total"]); ev=eval_arachne_r6_a1_v1(model,cond,teacher,rest,transforms,a0_token=token,expected_surface_hashes=cond.source_surface_hashes,expected_skeleton_hashes=cond.source_skeleton_hashes); assert np.isfinite(ev["latent_nll"])
    proposal=model.propose(cond)[0]; assert proposal.metadata["codec_decoder_object_shared"] is True; assert proposal.metadata["codec_frozen"] is True; qualified=qualify_skin(surface,skeleton,proposal); assert len(qualified.rows)==6; assert qualified.surface_binding_hash==surface.geometry_lineage_hash; assert qualified.skeleton_binding_hash==skeleton.skeleton_lineage_hash


def test_tree_perturbation_changes_arachne_output():
    surface=_surface("H",6); c1=ArachneConditioningAdapterV2()([surface],[_skeleton("H",(-1,0,1))]); c2=ArachneConditioningAdapterV2()([surface],[_skeleton("H",(-1,-1,1))]); codec=_codec(); model=ArachneCandidateV2(codec,ArachneCandidateConfigV2(model_dim=32,surface_encoder_layers=1,attention_heads=4,feedforward_dim=64))
    def run(c):
        return model(torch.tensor(c.surface_features),torch.tensor(c.joint_features),torch.tensor(c.surface_mask),torch.tensor(c.joint_mask),torch.tensor(c.parent_indices),torch.tensor(c.pair_geometry),torch.tensor(c.pair_mask)).decoded_weights
    a,b=run(c1),run(c2); assert not torch.allclose(a,b)
