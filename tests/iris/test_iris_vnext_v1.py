from __future__ import annotations
import numpy as np,pytest,torch
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.model_v1 import IRISVNextV1
from models.iris.vnext.teacher_truth_v1 import build_mechanical_demand_truth_v1
from models.iris.vnext.training_v1 import assert_batched_fit8_v1

def _cfg():return IRISVNextConfigV1(image_token_dim=16,camera_feature_dim=5,scene_dim=32,camera_embed_dim=16,plane_size=2,plane_channels=8,field_hidden_dim=32,field_residual_blocks=2,local_feature_dim=32,attention_heads=4,scene_layers=1,feedforward_multiplier=2,topology_width=32,topology_heads=4,topology_decoder_layers=2,topology_memory_tokens=64,topology_refinement_layers=1,topology_source_width=16,appearance_descriptor_dim=8,relation_embedding_dim=8,maximum_extraction_delta=.05)
def test_vnext_single_scene_geometry_mechanical_structure_contract():
    torch.manual_seed(0);cfg=_cfg();m=IRISVNextV1(cfg);b,v,t=2,8,4;im=torch.randn(b,v,t,16);xy=torch.randn(b,v,t,2).clamp(-1,1);cam=torch.randn(b,v,5);planes=m.encode_scene(im,xy,cam);assert planes.shape==(b,3,8,4,4);points=torch.rand(b,7,3)*2-1;e=m.query_evidence(planes,points);assert e["H"].shape==(b,7,32);assert e["mechanical_demand"]["tangent_direction_2theta"].shape==(b,7,2);assert m.semantic_taxonomy_used is False and m.independent_visual_mesh_allowed is False;xyz=points[:,:5];normal=torch.nn.functional.normalize(torch.randn(b,5,3),dim=-1);w=m.query_evidence(planes,xyz);ids=torch.full((b,6),cfg.tessa_decoder_config().BOS,dtype=torch.long);o=m.decode_topology(scene_planes=planes,witness_xyz=xyz,witness_normal=normal,witness_h=w["H"],witness_log_uncertainty=w["log_uncertainty"],witness_mechanical=w["mechanical_demand"],input_ids=ids);assert o.logits.shape[:2]==ids.shape;assert o.surface_latents.shape==(b,64,32)
def test_vnext_parameter_groups_are_disjoint():
    m=IRISVNextV1(_cfg());groups=m.parameter_groups();sets=[{id(p) for p in x} for x in groups.values()];assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2]);assert set().union(*sets)=={id(p) for p in m.parameters()}
def test_mechanical_truth_is_training_only_motion_supervision():
    rest=np.array([[0.,0,0],[1,0,0],[0,1,0]]);faces=np.array([[0,1,2]]);posed=np.stack((rest,np.array([[0.,0,0],[1.25,0,0],[0,1,0]])));truth=build_mechanical_demand_truth_v1(rest,faces,posed);assert truth.stretch_major[0]>0 and truth.edge_strain_p95[0]>0;assert truth.metadata["teacher_only"] is True and truth.metadata["rig_or_skin_inference_input"] is False
def test_fit8_kernel_rejects_subject_serial_shape():
    one={"image_tokens":torch.zeros(1,8,2,16),"token_xy":torch.zeros(1,8,2,2),"camera_features":torch.zeros(1,8,5)}
    with pytest.raises(ValueError,match="IRIS_VNEXT_REQUIRES_BATCHED_COHORT"):assert_batched_fit8_v1(one,expected=8)
    eight={"image_tokens":torch.zeros(8,8,2,16),"token_xy":torch.zeros(8,8,2,2),"camera_features":torch.zeros(8,8,5)};assert_batched_fit8_v1(eight,expected=8)
