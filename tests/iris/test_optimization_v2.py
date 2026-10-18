import unittest

import torch
from torch import nn

from models.iris.vnext.optimization_v2 import (
    SPECIALIST_OPTIMIZER_GROUPS_V2,
    SpecialistOptimizationPolicyV2,
    build_specialist_optimizer_v2,
    capture_group_parameters_v2,
    group_l2_telemetry_v2,
    optimizer_group_lrs_v2,
    specialist_parameter_groups_v2,
)


class _TinyPostV8Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_core = nn.Linear(2, 2)
        self.mechanical_demand = nn.Linear(2, 1)
        self.appearance = nn.Linear(2, 1)
        self.relations = nn.Linear(2, 1)
        self.structure = nn.Linear(2, 1)


class PostV8OptimizationV2Tests(unittest.TestCase):
    def _policy(self):
        return SpecialistOptimizationPolicyV2(
            mechanical_learning_rate=1.0e-4,
            appearance_learning_rate=2.0e-4,
            relation_learning_rate=3.0e-4,
            structure_learning_rate=4.0e-4,
        )

    def test_parameter_groups_are_disjoint_complete_and_named(self):
        model = _TinyPostV8Model()
        groups = specialist_parameter_groups_v2(model)
        self.assertEqual(tuple(groups), SPECIALIST_OPTIMIZER_GROUPS_V2)
        grouped = {id(p) for params in groups.values() for p in params}
        expected = {id(p) for p in model.parameters()}
        self.assertEqual(grouped, expected)

    def test_scene_core_nonzero_lr_is_forbidden(self):
        policy = SpecialistOptimizationPolicyV2(
            scene_learning_rate=1.0e-8,
            mechanical_learning_rate=1.0e-4,
            appearance_learning_rate=1.0e-4,
            relation_learning_rate=1.0e-4,
            structure_learning_rate=1.0e-4,
        )
        with self.assertRaisesRegex(ValueError, "SCENE_LR_MUST_BE_ZERO"):
            build_specialist_optimizer_v2(_TinyPostV8Model(), policy)

    def test_optimizer_exposes_head_specific_lrs_with_sealed_scene(self):
        optimizer = build_specialist_optimizer_v2(_TinyPostV8Model(), self._policy())
        lrs = optimizer_group_lrs_v2(optimizer)
        self.assertEqual(tuple(lrs), SPECIALIST_OPTIMIZER_GROUPS_V2)
        self.assertEqual(lrs["scene_core"], 0.0)
        self.assertEqual(lrs["mechanical_head"], 1.0e-4)
        self.assertEqual(lrs["appearance_head"], 2.0e-4)
        self.assertEqual(lrs["relation_head"], 3.0e-4)
        self.assertEqual(lrs["structure"], 4.0e-4)

    def test_group_telemetry_separates_gradients_and_updates(self):
        model = _TinyPostV8Model()
        optimizer = build_specialist_optimizer_v2(model, self._policy())
        before = capture_group_parameters_v2(optimizer)

        x = torch.ones(1, 2)
        loss = (
            model.mechanical_demand(x).sum()
            + model.appearance(x).sum()
            + model.relations(x).sum()
            + model.structure(x).sum()
        )
        loss.backward()
        pre_step = group_l2_telemetry_v2(optimizer)
        self.assertEqual(pre_step["scene_core"]["gradient_l2"], 0.0)
        for name in ("mechanical_head", "appearance_head", "relation_head", "structure"):
            self.assertGreater(pre_step[name]["gradient_l2"], 0.0)

        optimizer.step()
        post_step = group_l2_telemetry_v2(optimizer, before=before)
        self.assertEqual(post_step["scene_core"]["update_l2"], 0.0)
        for name in ("mechanical_head", "appearance_head", "relation_head", "structure"):
            self.assertGreater(post_step[name]["update_l2"], 0.0)
            self.assertGreater(post_step[name]["update_parameter_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
