#!/usr/bin/env python3
"""Deterministic *synthetic* RSS.vNext shared-H gradient-interference microcourt.

Runs real unified IRIS/TESSA/semantic-source-role PyTorch gradients on small
teacher-bound fixtures. This is NOT a corpus study, learned checkpoint or
product-quality gate. It records precise measurement ranges for debugging
gradient routing before introducing any gradient surgery or phase policy.

Usage: python tools/research/rss_vnext_training_gradient_microcourt_v1.py
"""
from __future__ import annotations

from hashlib import sha256
import json
import torch

from models.rss_vnext.training_gradient_interference_v1 import (
    RSSvNextGradientCourtPolicyV1,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
)
from tests.models.test_rss_vnext_early_semantics import _qualified_3d_role_case
from tests.models.test_rss_vnext_structural_trainer import make_sample

SEEDS = (11, 23, 47)


def main() -> None:
    torch.set_num_threads(1)
    records = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model, registry, batch = _qualified_3d_role_case()
        topo = make_sample(model)
        batch.topology_teacher_tokens = topo.topology_teacher_tokens
        batch.topology_valid_mask = topo.topology_valid_mask
        trainer = RSSvNextStructuralTrainer(
            model, torch.optim.AdamW(model.parameters(), lr=1.e-3),
            dataset_manifest_sha256="3"*64,
            trusted_surface_role_registry=registry,
            weights=RSSvNextTrainingWeights(
                topology=1., surface_roles=1., motion_preference=0.,
                source_semantics=0., flexicubes_surface=0.,
            ),
            gradient_court_policy=RSSvNextGradientCourtPolicyV1(
                every_n_steps=1, max_parameters_per_owner=12,
                max_elements_per_owner=65536,
            ),
        )
        # Paired pre/post objective on the exact same synthetic training
        # example and deterministic dropout masks. This tests the ACTUAL
        # AdamW step rather than inferring AdamW effects from the SGD
        # first-order gradient direction. It still does NOT measure unseen
        # or held-out generalization.
        dropout_seed=seed+100000
        torch.manual_seed(dropout_seed)
        before=trainer.objective(batch)
        initial={
            "topology":float(before.topology_loss.detach()),
            "surface_roles":float(before.surface_role_loss.detach()),
        }
        del before
        torch.manual_seed(dropout_seed)
        measured=trainer.train_step(batch).gradient_interference_report
        assert measured is not None
        torch.manual_seed(dropout_seed)
        after=trainer.objective(batch)
        actual_deltas={
            "topology":float(after.topology_loss.detach())-initial["topology"],
            "surface_roles":float(after.surface_role_loss.detach())-initial["surface_roles"],
        }
        del after
        scene=measured["owners"]["SHARED_IRIS_SCENE"]
        pair=next(x for x in scene["pairwise"]
                  if set((x["left"],x["right"]))=={"topology","surface_roles"})
        assert pair["cosine"] is not None
        assert scene["per_family_gradient_norms"]["topology"]>0.
        assert scene["per_family_gradient_norms"]["surface_roles"]>0.
        records.append({
            "seed":seed,
            "iris_scene_pair_cosine":pair["cosine"],
            "iris_scene_negative_interference":pair["negative_interference"],
            "iris_scene_cancellation_ratio":scene["cancellation_ratio"],
            "iris_scene_weighted_dominance_ratio":
                scene["weighted_dominance_ratio"],
            "iris_scene_first_order_descent_alignment":
                scene["first_order_plain_gradient_descent_alignment"],
            "iris_scene_adverse_first_order_families":
                scene["adverse_first_order_family_count"],
            "actual_AdamW_same_batch_one_step_loss_delta":actual_deltas,
            "actual_AdamW_any_task_loss_increased":any(
                value>0. for value in actual_deltas.values()
            ),
            "iris_scene_worst_named_parameter_pairs":
                scene["worst_sampled_parameters"][:3],
            "iris_scene_sampled_parameters":scene["sampled_parameter_count"],
            "iris_scene_eligible_parameters":scene["eligible_parameter_count"],
            "iris_scene_sampled_elements":scene["sampled_scalar_elements"],
            "iris_scene_eligible_elements":scene["eligible_scalar_elements"],
        })
    report={
        "schema":"RealSaS.RSSvNext.SyntheticGradientMicrocourt.v1",
        "qualification":"SYNTHETIC_RANDOM_MODEL_NO_LEARNED_CHECKPOINT",
        "source_claim":"TEST_ONLY_SOURCE_FIXTURES_NOT_INDEPENDENTLY_SOURCE_QUALIFIED",
        "optimizer_policy":"WEIGHTED_SUM_UNCHANGED_OBSERVATION_ONLY",
        "objective_families":["topology","surface_roles"],
        "seeds":list(SEEDS),
        "conflicting_iris_scene_seeds":sum(
            int(r["iris_scene_negative_interference"]) for r in records
        ),
        "seeds_with_adverse_first_order_plain_gradient_descent":sum(
            int(r["iris_scene_adverse_first_order_families"]>0)
            for r in records
        ),
        "seeds_with_any_increased_loss_after_real_AdamW_step":sum(
            int(r["actual_AdamW_any_task_loss_increased"])
            for r in records
        ),
        "records":records,
    }
    print(json.dumps(report, sort_keys=True, separators=(",",":"), allow_nan=False))


if __name__=="__main__":
    main()
