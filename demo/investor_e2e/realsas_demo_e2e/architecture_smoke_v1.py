from __future__ import annotations

"""Zero-specimen executable smoke test.

No optimizer/training step occurs.  This only proves that the generic learned
producers instantiate, emit the expected shapes, and bind to current Compiler
IR types before any specimen is selected.
"""

import torch

from realsas_compiler_core.hashing import content_sha256
from realsas_compiler_core.types import (
    QualifiedJoint, QualifiedSkeletonIR, RiggingSurfaceIR, SurfaceNode,
)
from realsas_demo_e2e import (
    ArachneDemoConfig, ArachneDemoV1,
    GeppettoDemoConfig, GeppettoDemoV1,
    IrisDemoConfig, IrisDemoV1,
)


def synthetic_surface() -> RiggingSurfaceIR:
    nodes=[]
    pts=[(-.4,-.5,0.),(.4,-.5,0.),(-.4,.5,0.),(.4,.5,0.),(0.,0.,.15)]
    for i,p in enumerate(pts):
        nodes.append(SurfaceNode(
            surface_id=f"SYN:S:{i}",P=p,support_views=(0,1),
            provenance_refs=("SYNTHETIC_ARCHITECTURE_SMOKE",),source_observation_ids=(),
            raster_bindings=((0,(20.+i*8,20.+i*6)),(1,(22.+i*8,20.+i*6))),
            persistence_group_id=f"SYN:G:{i}",
        ))
    surface=RiggingSurfaceIR(tuple(nodes),geometry_lineage_hash="")
    return RiggingSurfaceIR(**{**surface.__dict__,"geometry_lineage_hash":content_sha256(surface.to_dict())})


def synthetic_skeleton(surface: RiggingSurfaceIR) -> QualifiedSkeletonIR:
    joints=(
        QualifiedJoint("J:ROOT",(0.,0.,0.),None,(surface.surface_nodes[4].surface_id,),"SYN:Q:0"),
        QualifiedJoint("J:L",(-.25,.25,0.),"J:ROOT",(surface.surface_nodes[2].surface_id,),"SYN:Q:1"),
        QualifiedJoint("J:R",(.25,.25,0.),"J:ROOT",(surface.surface_nodes[3].surface_id,),"SYN:Q:2"),
    )
    return QualifiedSkeletonIR(joints,"J:ROOT",{"status":"SYNTHETIC_TYPED_SMOKE"},"SYN:SKEL:HASH")


def main() -> None:
    torch.manual_seed(20260901)

    iris=IrisDemoV1(IrisDemoConfig(model_resolution=32,base_width=16,evidence_stride=4))
    images=torch.rand(1,8,4,32,32)
    yaw=torch.arange(8,dtype=torch.float32)*45.0
    out=iris(images,yaw)
    assert out.depth.shape==(1,8,1,32,32)
    assert out.support_logits.shape==out.depth.shape
    assert torch.isfinite(out.depth).all()

    surface=synthetic_surface()
    gp=GeppettoDemoV1(GeppettoDemoConfig(max_joints=8,token_dim=32,hidden_dim=64,encoder_layers=1,query_layers=1,heads=4,support_k=2))
    p=torch.tensor([[n.P for n in surface.surface_nodes]],dtype=torch.float32)
    gout=gp(p)
    assert gout.position_norm.shape==(1,8,3)
    assert gout.edge_logits.shape==(1,8,8)
    assert torch.isfinite(gout.position_norm).all()

    skeleton=synthetic_skeleton(surface)
    ar=ArachneDemoV1(ArachneDemoConfig(hidden_dim=32,depth=2,top_k=3))
    sp=torch.tensor([n.P for n in surface.surface_nodes],dtype=torch.float32)
    jp=torch.tensor([j.position for j in skeleton.joints],dtype=torch.float32)
    parent_by={j.canonical_joint_id:j.parent_canonical_id for j in skeleton.joints}
    pos_by={j.canonical_joint_id:torch.tensor(j.position,dtype=torch.float32) for j in skeleton.joints}
    pp=torch.stack([pos_by[parent_by[j.canonical_joint_id]] if parent_by[j.canonical_joint_id] else pos_by[j.canonical_joint_id] for j in skeleton.joints])
    root=torch.tensor([1.0 if j.canonical_joint_id==skeleton.root_id else 0.0 for j in skeleton.joints])
    logits=ar(sp,jp,pp,root)
    assert logits.shape==(len(surface.surface_nodes),len(skeleton.joints))
    assert torch.isfinite(logits).all()

    print("DEMO_ARCHITECTURE_SMOKE = PASS")
    print("SPECIMEN_BOUND = FALSE")
    print("OPTIMIZER_STEPS = 0")
    print("GENERALIZATION_CLAIM = FALSE")


if __name__=="__main__":
    main()
