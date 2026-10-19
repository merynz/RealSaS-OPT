from __future__ import annotations

import unittest

import torch

from models.iris.vnext.surface_rescue_training_v1 import SurfA1ScheduleV1, module_state_sha256_v1


class SurfaceRescueTrainingTests(unittest.TestCase):
    def test_schedule_boundaries_are_preregistered(self):
        schedule = SurfA1ScheduleV1()
        self.assertEqual(schedule.total_steps, 2400)
        self.assertEqual(schedule.resolution(0), 24)
        self.assertEqual(schedule.resolution(399), 24)
        self.assertEqual(schedule.resolution(400), 32)
        self.assertEqual(schedule.resolution(999), 32)
        self.assertEqual(schedule.resolution(1000), 48)
        self.assertEqual(schedule.resolution(2399), 48)
        self.assertGreater(schedule.learning_rate(119), 0.0)
        self.assertGreaterEqual(schedule.learning_rate(120), schedule.minimum_lr)
        self.assertAlmostEqual(schedule.learning_rate(2399), schedule.minimum_lr, places=12)

    def test_module_state_hash_detects_parameter_mutation(self):
        module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.GELU(), torch.nn.Linear(4, 2))
        before = module_state_sha256_v1(module)
        with torch.no_grad():
            module[0].weight[0, 0].add_(1.0)
        after = module_state_sha256_v1(module)
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
