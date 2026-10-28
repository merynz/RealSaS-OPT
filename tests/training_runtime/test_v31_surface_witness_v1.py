from __future__ import annotations

import unittest

import torch

from models.iris.v31.surface_witness_v1 import (
    SAMPLER_ID_V1,
    SurfaceWitnessSamplerReceiptV1,
    xyz_fps_indices_v31,
)


def _rows():
    xyz = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, -1.5, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    normal = torch.tensor([[0.0, 0.0, 1.0]] * len(xyz), dtype=torch.float32)
    return xyz, normal


class V31SurfaceWitnessSamplerTests(unittest.TestCase):
    def test_selected_sampler_receipt_does_not_select_extraction_or_budget(self):
        receipt = SurfaceWitnessSamplerReceiptV1()
        self.assertEqual(receipt.sampler_id, SAMPLER_ID_V1)
        self.assertFalse(receipt.production_resolution_selected)
        self.assertFalse(receipt.production_witness_count_selected)
        self.assertFalse(receipt.product_authority_changed)
        self.assertFalse(receipt.model_release_changed)
        self.assertFalse(receipt.genericity_claimed)

    def test_xyz_fps_is_permutation_invariant_in_selected_geometry(self):
        xyz, normal = _rows()
        reference = None
        for seed in range(12):
            permutation = torch.randperm(len(xyz), generator=torch.Generator().manual_seed(700 + seed))
            selected = xyz_fps_indices_v31(xyz[permutation], normal[permutation], count=4)
            chosen = tuple(
                sorted(tuple(float(v) for v in row) for row in xyz[permutation][selected].tolist())
            )
            if reference is None:
                reference = chosen
            self.assertEqual(chosen, reference)

    def test_xyz_fps_captures_distant_thin_support_and_exact_quota(self):
        body = torch.stack(
            (
                torch.linspace(-0.2, 0.2, 32),
                torch.zeros(32),
                torch.zeros(32),
            ),
            dim=-1,
        )
        rare = torch.tensor([[2.5, 0.0, 0.0], [2.6, 0.0, 0.0]], dtype=torch.float32)
        xyz = torch.cat((body, rare), dim=0)
        normal = torch.tensor([[0.0, 0.0, 1.0]] * len(xyz), dtype=torch.float32)
        selected = xyz_fps_indices_v31(xyz, normal, count=6)
        self.assertEqual(int(selected.numel()), 6)
        self.assertEqual(int(torch.unique(selected).numel()), 6)
        self.assertTrue(bool((selected >= len(body)).any()))

    def test_xyz_fps_all_points_are_returned_in_canonical_order_when_budget_exceeds_n(self):
        xyz, normal = _rows()
        permutation = torch.tensor([7, 2, 4, 0, 6, 1, 5, 3])
        selected = xyz_fps_indices_v31(xyz[permutation], normal[permutation], count=99)
        chosen = xyz[permutation][selected]
        expected = torch.tensor(sorted(tuple(float(v) for v in row) for row in xyz.tolist()))
        self.assertTrue(torch.equal(chosen, expected))

    def test_duplicate_canonical_rows_fail_closed(self):
        xyz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
        normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "IRIS_V31_XYZ_FPS_DUPLICATE_CANONICAL_ROW"):
            xyz_fps_indices_v31(xyz, normal, count=1)

    def test_non_unit_normals_fail_closed(self):
        xyz, normal = _rows()
        normal[0] = torch.tensor([0.0, 0.0, 2.0])
        with self.assertRaisesRegex(ValueError, "IRIS_V31_XYZ_FPS_UNIT_NORMAL_REQUIRED"):
            xyz_fps_indices_v31(xyz, normal, count=3)


if __name__ == "__main__":
    unittest.main()
