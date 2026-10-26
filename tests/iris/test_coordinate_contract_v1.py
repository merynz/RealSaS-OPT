from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.coordinate_contract_v1 import (
    iris_xyz_to_tessa_v1,
    tessa_xyz_to_iris_v1,
    triplane_cell_center_xy_tessa_v1,
)
from models.iris.vnext.presentation_evidence_v1 import (
    PRESENTATION_KIND_PAIR,
    build_role_free_presentation_evidence_v1,
)
from models.iris.vnext.structure_v5 import IRISVNextStructuralizationBridgeV5


class CoordinateContractV1Test(unittest.TestCase):
    def test_round_trip_domain_conversion(self):
        xyz_iris = torch.tensor(
            [[[-1.0, -0.25, 1.0], [0.5, 0.0, -0.75]]], dtype=torch.float32
        )
        xyz_tessa = iris_xyz_to_tessa_v1(xyz_iris)
        self.assertLessEqual(float(xyz_tessa.abs().max()), 0.5)
        self.assertTrue(torch.equal(tessa_xyz_to_iris_v1(xyz_tessa), xyz_iris))

    def test_align_corners_false_carrier_centers_use_tessa_units(self):
        xy = triplane_cell_center_xy_tessa_v1(
            64, 64, device=torch.device("cpu"), dtype=torch.float32
        )
        expected = 0.5 * (1.0 - 1.0 / 64.0)
        self.assertAlmostEqual(float(xy[..., 0].max()), expected, places=7)
        self.assertAlmostEqual(float(xy[..., 0].min()), -expected, places=7)
        self.assertAlmostEqual(float(xy[..., 1].max()), expected, places=7)
        self.assertAlmostEqual(float(xy[..., 1].min()), -expected, places=7)

    def test_v5_scene_source_positions_share_tessa_domain(self):
        cfg = IRISVNextConfigV1()
        dummy = SimpleNamespace(iris_config=cfg)
        planes = torch.zeros(1, 3, cfg.plane_channels, 64, 64)
        source = IRISVNextStructuralizationBridgeV5.scene_source(dummy, planes)
        pos = source[..., cfg.plane_channels : cfg.plane_channels + 2]
        expected = 0.5 * (1.0 - 1.0 / 64.0)
        self.assertLessEqual(float(pos.abs().max()), expected + 1.0e-7)
        self.assertAlmostEqual(float(pos[..., 0].min()), -expected, places=7)
        self.assertAlmostEqual(float(pos[..., 0].max()), expected, places=7)

    def test_pair_endpoint_conversion_prepares_q_domain(self):
        xyz_a = torch.tensor([[[0.5, -0.25, 0.0]]], dtype=torch.float32)
        xyz_b = torch.tensor([[[-0.5, 0.25, 0.125]]], dtype=torch.float32)
        evidence = build_role_free_presentation_evidence_v1(
            xyz=xyz_a,
            kind_id=torch.tensor([[PRESENTATION_KIND_PAIR]], dtype=torch.long),
            token_valid_mask=torch.tensor([[True]]),
            paired_xyz=xyz_b,
            paired_xyz_valid=torch.tensor([[True]]),
        )
        a_iris, b_iris, valid = (
            IRISVNextStructuralizationBridgeV5.presentation_endpoint_query_xyz_iris_v1(
                evidence
            )
        )
        self.assertTrue(bool(valid.item()))
        self.assertTrue(
            torch.equal(a_iris, torch.tensor([[[1.0, -0.5, 0.0]]]))
        )
        self.assertTrue(
            torch.equal(b_iris, torch.tensor([[[-1.0, 0.5, 0.25]]]))
        )


if __name__ == "__main__":
    unittest.main()
