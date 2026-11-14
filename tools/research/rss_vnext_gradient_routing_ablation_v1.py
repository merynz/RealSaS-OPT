#!/usr/bin/env python3
"""Matched-seed, full-shared-IRIS two-task gradient routing micro-ablation.

Synthetic source-label fixtures from existing tests; NOT real qualified corpus,
trained checkpoint, held-out generalization, or a ProductProof. Uses EXACT
same initial weights, source batch, per-step dropout seed, 2 task backward
probes + summed backward, AdamW hyperparameters and total update count.

Only model.iris.scene is routed. All expert-private task gradients retain
the ordinary summed backward. PCGrad/MGDA/CAGrad direction changes have
different vector norms; results show both norms and cannot be attributed to
"direction only". Same compute means same model forward/backward count,
NOT identical solver flops/time. We select no production winner here.

CLI:
 python tools/research/rss_vnext_gradient_routing_ablation_v1.py --quick
 python tools/research/rss_vnext_gradient_routing_ablation_v1.py --steps 5 --seeds 11 23 47
"""
from __future__ import annotations

import argparse
import copy
import json
from math import isfinite

import torch
from torch import Tensor

from models.rss_vnext.training_gradient_routing_research_v1 import (
    ROUTES, route_two_rss_vnext_task_gradients_v1,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
)
from tests.models.test_rss_vnext_early_semantics import _qualified_3d_role_case
from tests.models.test_rss_vnext_structural_trainer import make_sample


_DATASET = "3"*64
_WEIGHTS = RSSvNextTrainingWeights(
    topology=1., surface_roles=1., motion_preference=0.,
    source_semantics=0., flexicubes_surface=0.,
)


def _source_model_and_batch(seed: int):
    torch.manual_seed(seed)
    source_model, receipt, batch = _qualified_3d_role_case()
    sample = make_sample(source_model)
    batch.topology_teacher_tokens = sample.topology_teacher_tokens
    batch.topology_valid_mask = sample.topology_valid_mask
    return source_model, receipt, batch


def _vector(grads: tuple[Tensor | None, ...],params: list[torch.nn.Parameter]) -> Tensor:
    return torch.cat([
        grad.detach().reshape(-1) if grad is not None
        else torch.zeros_like(param).reshape(-1)
        for grad,param in zip(grads,params)
    ])


def _losses(result) -> dict[str,float]:
    if result.topology_loss is None or result.surface_role_loss is None:
        raise RuntimeError("RSS_VNEXT_ABLATION_QUALIFIED_TASK_MISSING")
    return {
        "topology":float(result.topology_loss.detach().item()),
        "surface_roles":float(result.surface_role_loss.detach().item()),
    }


def _private_head_grad_norms(model) -> dict[str,float]:
    return {
        "tessa":float(torch.sqrt(sum(
            (p.grad.detach().double().square().sum()
             for p in model.tessa.parameters() if p.grad is not None),
            torch.tensor(0.,dtype=torch.float64),
        )).item()),
        "presentation_role_head":float(torch.sqrt(sum(
            (p.grad.detach().double().square().sum()
             for p in model.presentation_role_head.parameters() if p.grad is not None),
            torch.tensor(0.,dtype=torch.float64),
        )).item()),
    }


def train_one_policy(
    source_model, batch, registry, *,
    route_mode: str, seed: int, steps: int,
) -> dict:
    model=copy.deepcopy(source_model)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    trainer=RSSvNextStructuralTrainer(
        model,opt,dataset_manifest_sha256=_DATASET,
        weights=_WEIGHTS,trusted_surface_role_registry=registry,
    )
    shared=[(name,p) for name,p in model.iris.scene.named_parameters()
            if p.requires_grad]
    if len(shared)<2:
        raise RuntimeError("RSS_VNEXT_ABLATION_SHARED_SCENE_OWNER_MISSING")
    params=[p for _,p in shared]
    n_shared_elements=sum(p.numel() for p in params)
    if n_shared_elements<1:
        raise RuntimeError("RSS_VNEXT_ABLATION_EMPTY_SHARED_VECTOR")
    report=[]
    initial=None
    for step in range(steps):
        trainer._assert_optimizer_ownership()
        model.train()
        opt.zero_grad(set_to_none=True)
        dropout_seed=seed*10000+step*101+17
        torch.manual_seed(dropout_seed)
        out=trainer.objective(batch)
        before=_losses(out)
        if initial is None:
            initial=before
        a=torch.autograd.grad(
            out.topology_loss * _WEIGHTS.topology, params,
            retain_graph=True, allow_unused=True,
        )
        b=torch.autograd.grad(
            out.surface_role_loss * _WEIGHTS.surface_roles,params,
            retain_graph=True,allow_unused=True,
        )
        vec_a=_vector(a,params)
        vec_b=_vector(b,params)
        if not bool(torch.isfinite(vec_a).all()) or not bool(torch.isfinite(vec_b).all()):
            raise RuntimeError("RSS_VNEXT_ABLATION_TASK_NONFINITE")
        if not bool((vec_a.norm()>0.) & (vec_b.norm()>0.)):
            raise RuntimeError("RSS_VNEXT_ABLATION_TASK_SHARED_GRADIENT_EMPTY")
        route=route_two_rss_vnext_task_gradients_v1(
            vec_a,vec_b,mode=route_mode,cagrad_radius_fraction=.5,
        )
        out.total_loss.backward()
        # Compare exact flat shared gradient to independent task sum before
        # any routing, optimizer preconditioning, clipping or weight decay.
        ordinary=_vector(tuple(p.grad for p in params),params)
        if not torch.allclose(
            ordinary,vec_a+vec_b,atol=5.e-5,rtol=2.e-4,
        ):
            raise RuntimeError("RSS_VNEXT_ABLATION_UNACCOUNTED_SHARED_OBJECTIVE")
        head_grads=_private_head_grad_norms(model)
        if min(head_grads.values())<=0.:
            raise RuntimeError("RSS_VNEXT_ABLATION_PRIVATE_HEAD_GRADIENT_MISSING")
        # Source-specific TESSA/Semantic heads keep ordinary gradients;
        # FREEZE_SHARED must use grad=None to skip AdamW weight decay/moments.
        start=0
        for p,g_a,g_b in zip(params,a,b):
            n=p.numel()
            if route_mode=="FREEZE_SHARED":
                p.grad=None
            elif g_a is None and g_b is None:
                p.grad=None
            else:
                p.grad=route.candidate[start:start+n].view_as(p).clone()
            start+=n
        assert start == n_shared_elements
        grad_norm=torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            _WEIGHTS.clip_gradient_norm,error_if_nonfinite=True,
        )
        opt.step()
        trainer.global_step+=1
        # Identical dropout seed BEFORE/AFTER evaluation. This is one-batch
        # training progression, not independent held-out source evaluation.
        torch.manual_seed(dropout_seed)
        after_out=trainer.objective(batch)
        after=_losses(after_out)
        if any(not isfinite(v) for v in after.values()):
            raise RuntimeError("RSS_VNEXT_ABLATION_NONFINITE_POST_LOSS")
        report.append({
            "step":step,
            "before":before,"after":after,
            "task_loss_delta":{k:after[k]-before[k] for k in before},
            "shared_baseline_grad_norm":route.baseline_norm,
            "shared_routed_grad_norm":route.candidate_norm,
            "shared_route_cosine_with_baseline":route.cosine_with_baseline,
            "shared_first_order_task_dots":list(route.first_order_task_dots),
            "shared_first_order_adverse_task_count":
                route.first_order_adverse_task_count,
            "cagrad_ball_radius":route.cagrad_ball_radius,
            "cagrad_ball_distance":route.cagrad_ball_distance,
            "pre_routing_private_head_grad_norms":head_grads,
            "total_preclip_grad_norm":float(grad_norm.detach()),
        })
        del out,after_out,a,b
    return {
        "mode":route_mode,"seed":seed,"steps":steps,
        "initial":initial,"final":report[-1]["after"],
        "final_minus_initial":{
            k:report[-1]["after"][k]-initial[k] for k in initial
        },
        "shared_trainable_parameter_tensors":len(params),
        "shared_trainable_scalar_elements":n_shared_elements,
        "full_shared_IRIS_scene_routed":True,
        "nonshared_private_grads_modified":False,
        "trajectory":report,
    }


def run(*, seeds: tuple[int,...], steps: int) -> dict:
    if not seeds or steps<1:
        raise ValueError("RSS_VNEXT_ABLATION_INVALID_EXPERIMENT")
    torch.set_num_threads(1)
    all_results=[]
    for seed in seeds:
        source_model,registry,batch=_source_model_and_batch(seed)
        per_seed=[]
        for mode in ROUTES:
            per_seed.append(train_one_policy(
                source_model,batch,registry,
                route_mode=mode,seed=seed,steps=steps,
            ))
        # Before any updates, private gradients must be equal across arms;
        # otherwise the seed/data/forward matching is not a valid experiment.
        reference=per_seed[0]["trajectory"][0]["pre_routing_private_head_grad_norms"]
        for arm in per_seed[1:]:
            actual=arm["trajectory"][0]["pre_routing_private_head_grad_norms"]
            for key in reference:
                if abs(actual[key]-reference[key])>1.e-5:
                    raise RuntimeError("RSS_VNEXT_ABLATION_EXPERT_GRADIENT_ARM_DRIFT")
            if arm["initial"]!=per_seed[0]["initial"]:
                raise RuntimeError("RSS_VNEXT_ABLATION_INITIAL_LOSS_ARM_DRIFT")
        all_results.extend(per_seed)
    return {
        "schema":"RealSaS.RSSvNext.TwoTaskGradientRoutingAblation.v1",
        "qualification":"RESEARCH_ONLY_SYNTHETIC_RANDOM_NO_SOURCE_CORPUS",
        "policy_authority":"NO_PRODUCT_TRAINER_ALGORITHM_SELECTION",
        "source_teacher":"TEST_ONLY_3D_ROLE_FIXTURE_NOT_COMMERCIAL_SOURCE_TRUTH",
        "trained_checkpoints_or_unseen_qualified":False,
        "arms":list(ROUTES),"seeds":list(seeds),"steps":steps,
        "matched_forward_per_step":1,
        "matched_task_backward_probes_per_step":2,
        "matched_sum_backward_per_step":1,
        "matched_optimizer_class":"torch.optim.AdamW",
        "matched_lr":1e-3,
        "gradient_routing_scope":"FULL_IRIS_SCENE_ONLY",
        "other_head_gradients":"UNCHANGED_WEIGHTED_SUM",
        "matched_optimization_flops":False,
        "shared_grad_norms_are_not_matched_between_algorithms":True,
        "individual_results":all_results,
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--steps",type=int,default=5)
    parser.add_argument("--seeds",nargs="+",type=int,default=[11,23,47])
    args=parser.parse_args()
    data=run(
        seeds=(11,) if args.quick else tuple(args.seeds),
        steps=2 if args.quick else args.steps,
    )
    print(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False))


if __name__=="__main__":
    main()
