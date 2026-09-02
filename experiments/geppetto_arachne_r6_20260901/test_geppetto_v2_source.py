from __future__ import annotations

import inspect
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from compiler.realsas_compiler_core.rig import qualify_skeleton_v2
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from experiments.geppetto_arachne_r6_20260901.geppetto_candidate_v2 import GeppettoCandidateConfigV2, GeppettoCandidateV2
from experiments.geppetto_arachne_r6_20260901.geppetto_checkpoint_v2 import load_geppetto_checkpoint_v2, save_geppetto_checkpoint_v2
from experiments.geppetto_arachne_r6_20260901.geppetto_conditioning_v2 import FEATURE_CONTRACT_V2, GeppettoConditioningAdapterV2
from experiments.geppetto_arachne_r6_20260901.geppetto_cpu_capacity_v2 import geppetto_cpu_capacity_probe_v2
from experiments.geppetto_arachne_r6_20260901.geppetto_loss_v2 import GeppettoLossV2
from experiments.geppetto_arachne_r6_20260901.geppetto_train_v2 import geppetto_train_step_v2
from experiments.geppetto_arachne_r6_20260901.training_targets_v1 import GeppettoTeacherTargetV1
from experiments.geppetto_arachne_r6_20260901.training_targets_v2 import permute_geppetto_teacher_target_v2

OP_HASH = "dtb-nd1-test-operator"


def _surface(prefix: str, count: int, *, reverse: bool = False, op_hash: str = OP_HASH) -> RiggingSurfaceIR:
    nodes=[]
    for i in range(count):
        nodes.append(SurfaceNode(surface_id=f"{prefix}:S:{i:04d}",P=(float(i)*0.013,float(i%7)*0.019,float(i%5)*0.011),support_views=(0,1,2,3,4,5,6,7) if i%2==0 else (0,2,4,6),provenance_refs=(f"obs:{i}",),source_observation_ids=(f"obs:{i}",),raster_bindings=((0,(-0.8+1.6*((i%17)/16.0),-0.7+1.4*((i%13)/12.0))),),derived_normal=(0.0,0.0,1.0)))
    relations=tuple(SurfaceRelation(f"R:{i}",nodes[i].surface_id,nodes[i+1].surface_id,"LOCAL",0.8) for i in range(max(0,count-1)))
    if reverse:
        nodes=list(reversed(nodes)); relations=tuple(reversed(relations))
    return RiggingSurfaceIR(tuple(nodes),relations,geometry_lineage_hash=f"surface-hash-{prefix}",metadata={"raster_coordinate_system":"GRID_XY","resolution":1024,"Nd_operator_sha256":op_hash})


def _tiny_config() -> GeppettoCandidateConfigV2:
    return GeppettoCandidateConfigV2(model_dim=24,knn_k=4,local_layers=1,global_layers=1,decoder_layers=1,attention_heads=4,feedforward_dim=48,support_topk=4)


def _target() -> GeppettoTeacherTargetV1:
    return GeppettoTeacherTargetV1(positions_normalized=np.asarray([[-0.35,-0.10,0.00],[0.00,0.05,0.02],[0.31,0.14,-0.01]],np.float32),parent_indices=np.asarray([-1,0,1],np.int64),root_mask=np.asarray([True,False,False]),valid=True)


def test_conditioning_v2_is_audited_24d_permutation_safe_teacher_free_and_operator_bound():
    adapter=GeppettoConditioningAdapterV2(); a=adapter([_surface("A",8)]); b=adapter([_surface("A",8,reverse=True)]); c=adapter([_surface("A",8,op_hash="different-operator")])
    assert len(FEATURE_CONTRACT_V2)==24 and a.features.shape==(1,8,24); assert a.surface_ids==b.surface_ids; np.testing.assert_allclose(a.features,b.features,atol=0.0,rtol=0.0); assert a.conditioning_hashes==b.conditioning_hashes; assert a.conditioning_hashes!=c.conditioning_hashes; assert a.local_geometry_operator_hashes==(OP_HASH,); assert set(inspect.signature(adapter.__call__).parameters)=={"surfaces"}; assert "teacher" not in repr(vars(a)).lower()
    with pytest.raises(ValueError,match="Nd_operator_sha256"):
        adapter([RiggingSurfaceIR(_surface("Z",2).surface_nodes,geometry_lineage_hash="z",metadata={"raster_coordinate_system":"GRID_XY","resolution":1024})])


def test_dynamic_cardinality_has_no_product_max_k_and_328_capacity_is_executable():
    cfg=_tiny_config(); assert not hasattr(cfg,"max_joint_queries"); probe=geppetto_cpu_capacity_probe_v2(surface_token_count=328,decode_steps=328); assert probe["finite"] is True; assert probe["parent_matrix_rows"]==328; assert probe["parent_matrix_cols"]==328; assert probe["support_width"]==328; assert probe["optimizer_steps"]==0; assert probe["product_max_joint_count"] is None


def test_heteroscedastic_loss_is_teacher_permutation_invariant_and_backpropagates():
    cond=GeppettoConditioningAdapterV2()([_surface("B",12)]); model=GeppettoCandidateV2(_tiny_config()); f=torch.tensor(cond.features,dtype=torch.float32); p=torch.tensor(cond.positions_normalized,dtype=torch.float32); m=torch.tensor(cond.valid_mask,dtype=torch.bool); out=model(f,p,m,decode_steps=4); target=_target(); permuted=permute_geppetto_teacher_target_v2(target,np.asarray([2,0,1],np.int64)); loss_fn=GeppettoLossV2(support_topk=4); a=loss_fn(out,[target],p,m); b=loss_fn(out,[permuted],p,m)
    for key in ("total","position_nll","existence","stop","root","parent","support","support_presence","abstain"): torch.testing.assert_close(a[key],b[key],atol=2e-5,rtol=2e-5)
    assert torch.isfinite(a["total"]) and torch.isfinite(out.position_log_sigma).all(); a["total"].backward(); assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters() if param.requires_grad)


def test_train_step_is_executable_without_fixed_cardinality():
    cond=GeppettoConditioningAdapterV2()([_surface("C",10)]); model=GeppettoCandidateV2(_tiny_config()); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4); metrics=geppetto_train_step_v2(model,optimizer,cond,[_target()]); assert np.isfinite(metrics["total"]); assert metrics["matched_joint_count"]==3.0


def test_proposal_is_anonymous_soft_evidence_and_compiler_owns_canonical_ids():
    surface=_surface("D",6); cond=GeppettoConditioningAdapterV2()([surface]); model=GeppettoCandidateV2(_tiny_config())
    with torch.no_grad():
        model.stop.weight.zero_(); model.stop.bias.fill_(12.0); model.existence.weight.zero_(); model.existence.bias.fill_(12.0); model.root.weight.zero_(); model.root.bias.fill_(12.0); model.support_presence.weight.zero_(); model.support_presence.bias.fill_(12.0); model.log_sigma.weight.zero_(); model.log_sigma.bias.fill_(-4.0)
    proposal=model.propose(cond,resource_step_limit=1)[0]; assert len(proposal.joints)==1; assert proposal.edges==(); assert proposal.joints[0].proposal_id.startswith("P:G2:") and not proposal.joints[0].proposal_id.startswith("J:"); assert proposal.metadata["generation_index_is_not_identity"] is True; assert proposal.metadata["compiler_owns_root_tree_ids"] is True; assert proposal.metadata["full_3d_reconstruction_claim"] is False; qualified=qualify_skeleton_v2(surface,proposal); assert len(qualified.joints)==1; assert qualified.joints[0].canonical_joint_id.startswith("J:"); assert qualified.joints[0].source_proposal_id==proposal.joints[0].proposal_id


def test_checkpoint_binds_config_and_feature_contract(tmp_path):
    model=GeppettoCandidateV2(_tiny_config()); path=tmp_path/"geppetto-v2.pt"; save_geppetto_checkpoint_v2(path,model,extra_metadata={"optimizer_steps":0}); authority=load_geppetto_checkpoint_v2(path,model); assert authority["dynamic_cardinality"] is True; assert authority["product_max_joint_count"] is None; mismatch=GeppettoCandidateV2(GeppettoCandidateConfigV2(model_dim=48,knn_k=4,local_layers=1,global_layers=1,decoder_layers=1,attention_heads=4,feedforward_dim=96,support_topk=4))
    with pytest.raises(ValueError,match="config_hash"): load_geppetto_checkpoint_v2(path,mismatch)
