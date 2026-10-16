from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from models.iris.vnext.cohort_v1 import IRISVNextFit8TrainerV1,SUBJECT_IDS,VNextFit8PolicyV1,VNextStageV1
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _cfg():return IRISVNextConfigV1(image_token_dim=16,camera_feature_dim=5,scene_dim=32,camera_embed_dim=16,plane_size=2,plane_channels=8,field_hidden_dim=32,field_residual_blocks=1,local_feature_dim=32,attention_heads=4,scene_layers=1,feedforward_multiplier=2,topology_width=32,topology_heads=4,topology_decoder_layers=1,topology_memory_tokens=64,topology_refinement_layers=1,topology_source_width=16,appearance_descriptor_dim=8,relation_embedding_dim=8)
def test_vnext_fit8_trainer_executes_one_real_batch8_forward_and_step(tmp_path):
    rows=[]
    for si,subject in enumerate(SUBJECT_IDS):
        g=torch.Generator().manual_seed(100+si);payload={"subject_id":subject,"teacher_inputs_in_inference_forward":False,"image_tokens":torch.randn(8,2,16,generator=g),"token_xy":torch.rand(8,2,2,generator=g)*2-1,"camera_features":torch.randn(8,5,generator=g),"geometry_points":torch.rand(6,3,generator=g)*2-1,"geometry_targets":torch.zeros(6),"mechanical_points":torch.rand(5,3,generator=g)*2-1,"mechanical_normals":torch.nn.functional.normalize(torch.randn(5,3,generator=g),dim=-1),"mechanical_stretch_major":torch.full((5,),.1),"mechanical_stretch_minor":torch.full((5,),.05),"mechanical_tangent_direction_2theta":torch.tensor([[1.,0.]]).repeat(5,1),"mechanical_bend_demand":torch.full((5,),.02),"mechanical_rigidity_probability":torch.full((5,),.8)};path=tmp_path/f"{si}.pt";torch.save(payload,path);rows.append({"subject_id":subject,"shard_file":path.name,"shard_sha256":_sha(path)})
    manifest={"schema":"RealSaS.IRISVNextFit8Manifest.v1","teacher_inputs_in_inference_forward":False,"subjects":rows};policy=VNextFit8PolicyV1(geometry_samples_per_subject=3,mechanical_samples_per_subject=2,checkpoint_interval=10,eval_interval=10,stages=(VNextStageV1("G1_TEST",1,scene_lr_scale=1.,evidence_lr_scale=1.,structure_lr_scale=0.,field_weight=1.,mechanical_weight=1.),));trainer=IRISVNextFit8TrainerV1(data_root=tmp_path,manifest=manifest,checkpoint_root=tmp_path/"cp",device="cpu",model_config=_cfg(),policy=policy);row=trainer.one_step();assert row["batch_subjects"]==8;assert row["optimizer_steps"]==1;assert row["global_step"]==1;assert row["stage"]=="G1_TEST";assert torch.isfinite(torch.tensor(row["loss"]))
