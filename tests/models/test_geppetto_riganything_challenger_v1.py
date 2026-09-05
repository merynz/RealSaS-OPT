import torch

from models.geppetto.challengers.riganything_mechanisms_v1 import (
    ConditionalJointDiffusionV1,
    randomize_bfs_depth_order_v1,
)


def test_joint_diffusion_loss_and_sampling_are_finite_cpu():
    torch.manual_seed(7)
    head=ConditionalJointDiffusionV1(context_dim=12,hidden_dim=24,depth=2,train_steps=32,sample_steps=8,position_clip=1.25)
    context=torch.randn(4,12)
    target=torch.tanh(torch.randn(4,3))
    loss=head.loss(target,context)
    assert loss.ndim==0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in head.parameters())
    with torch.no_grad():
        sample=head.sample(context,sample_steps=6)
    assert sample.shape==(4,3)
    assert torch.isfinite(sample).all()
    assert float(sample.abs().max()) <= 1.250001


def test_randomized_bfs_order_preserves_parent_before_child_and_geometry():
    pos=torch.tensor([
        [0.,0.,0.],  # root
        [-1.,0.,0.], [1.,0.,0.],
        [-2.,0.,0.], [-1.,1.,0.], [2.,0.,0.], [1.,1.,0.],
    ])
    parent=torch.tensor([-1,0,0,1,1,2,2])
    g=torch.Generator().manual_seed(123)
    p2,par2,perm=randomize_bfs_depth_order_v1(pos,parent,generator=g)
    assert set(map(tuple,p2.tolist()))==set(map(tuple,pos.tolist()))
    assert sorted(perm.tolist())==list(range(len(pos)))
    for child,p in enumerate(par2.tolist()):
        assert p < child if p >= 0 else child == 0


def test_full_challenger_teacher_forcing_uses_surface_memory_and_diffusion():
    from models.geppetto.challengers.riganything_mechanisms_v1 import (
        GeppettoRigAnythingMechanismChallengerV1,
        RigAnythingMechanismConfigV1,
    )
    from models.geppetto.v2.geppetto_candidate_v2 import GeppettoCandidateConfigV2
    base=GeppettoCandidateConfigV2(surface_feature_dim=24,model_dim=12,attention_heads=3,position_scale=1.25)
    cfg=RigAnythingMechanismConfigV1(base=base,diffusion_hidden=24,diffusion_depth=2,diffusion_train_steps=32,diffusion_sample_steps=8,max_sequence_steps=16)
    model=GeppettoRigAnythingMechanismChallengerV1(cfg)
    features=torch.randn(2,20,24)
    pos=torch.randn(2,20,3)
    mask=torch.ones(2,20,dtype=torch.bool)
    teacher=torch.tanh(torch.randn(2,4,3))
    parents=torch.tensor([[-1,0,1,1],[-1,0,0,2]])
    out=model(features,pos,mask,decode_steps=4,teacher_positions=teacher,teacher_parent_indices=parents)
    assert out.positions_normalized.shape==(2,4,3)
    assert out.control_contexts.shape==(2,4,12)
    assert out.parent_logits.shape==(2,4,4)
    assert out.surface_attention.shape==(2,4,20)
    assert out.diffusion_loss is not None and torch.isfinite(out.diffusion_loss)
    out.diffusion_loss.backward()
