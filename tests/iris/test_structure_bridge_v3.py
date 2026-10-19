import unittest

import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.structure_v3 import IRISVNextStructuralizationBridgeV3
from models.tessa.v2 import TESSAStructuralizerConfigV2


class IRISVNextStructuralizationBridgeV3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77)
        self.iris = IRISVNextConfigV1()
        self.tessa = TESSAStructuralizerConfigV2(
            d_model=64,
            n_heads=8,
            decoder_layers=1,
            coordinate_bins=128,
            scene_memory_tokens=4,
            surface_memory_tokens=4,
            memory_source_width=32,
            memory_refinement_layers=1,
            local_attention_window=16,
            query_chunk_size=4,
            gradient_checkpointing=False,
        )
        self.bridge = IRISVNextStructuralizationBridgeV3(
            iris_config=self.iris,
            tessa_config=self.tessa,
        )
        self.xyz = torch.randn(2, 5, 3) * 0.20
        self.normal = torch.nn.functional.normalize(torch.randn(2, 5, 3), dim=-1)
        self.h = torch.randn(2, 5, self.iris.local_feature_dim)
        self.mechanics = {
            "stretch_major": torch.rand(2, 5) + 1.0,
            "stretch_minor": torch.rand(2, 5),
            "tangent_direction_2theta": torch.nn.functional.normalize(
                torch.randn(2, 5, 2), dim=-1
            ),
            "bend_demand": torch.rand(2, 5),
            "rigidity_probability": torch.rand(2, 5),
            # Historical V1 key is intentionally extra and must not enter V3.
            "uncertainty": torch.rand(2, 5),
        }

    def test_uncalibrated_uncertainty_is_not_consumed(self):
        a = self.bridge.surface_source(
            witness_xyz=self.xyz,
            witness_normal=self.normal,
            witness_h=self.h,
            witness_mechanical=self.mechanics,
        )
        changed = {**self.mechanics, "uncertainty": self.mechanics["uncertainty"] + 1000.0}
        b = self.bridge.surface_source(
            witness_xyz=self.xyz,
            witness_normal=self.normal,
            witness_h=self.h,
            witness_mechanical=changed,
        )
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape[-1], 3 + 3 + self.iris.local_feature_dim + 6 + 2)

    def test_unknown_contact_is_typed_as_invalid_not_negative_truth(self):
        source = self.bridge.surface_source(
            witness_xyz=self.xyz,
            witness_normal=self.normal,
            witness_h=self.h,
            witness_mechanical=self.mechanics,
        )
        # final two channels are contact value and contact validity.
        self.assertTrue(torch.equal(source[..., -2], torch.zeros_like(source[..., -2])))
        self.assertTrue(torch.equal(source[..., -1], torch.zeros_like(source[..., -1])))

        contact = torch.ones(2, 5)
        valid = torch.tensor(
            [[True, False, True, False, True], [False, True, False, True, False]]
        )
        typed = self.bridge.surface_source(
            witness_xyz=self.xyz,
            witness_normal=self.normal,
            witness_h=self.h,
            witness_mechanical=self.mechanics,
            contact_persistence=contact,
            contact_persistence_valid=valid,
        )
        self.assertTrue(torch.equal(typed[..., -1].bool(), valid))
        self.assertTrue(torch.equal(typed[..., -2][valid], torch.ones_like(typed[..., -2][valid])))
        self.assertTrue(torch.equal(typed[..., -2][~valid], torch.zeros_like(typed[..., -2][~valid])))

    def test_forward_produces_v2_topology_objective(self):
        scene = torch.randn(2, 3, self.iris.plane_channels, 4, 4)
        input_ids = torch.tensor(
            [[self.tessa.BOS, self.tessa.COMPONENT_BEGIN, self.tessa.CHART_BEGIN, self.tessa.FACE_BREAK],
             [self.tessa.BOS, self.tessa.COMPONENT_BEGIN, self.tessa.CHART_BEGIN, self.tessa.FACE_BREAK]],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [[self.tessa.COMPONENT_BEGIN, self.tessa.CHART_BEGIN, self.tessa.FACE_BREAK, 6],
             [self.tessa.COMPONENT_BEGIN, self.tessa.CHART_BEGIN, self.tessa.FACE_BREAK, 8]],
            dtype=torch.long,
        )
        out = self.bridge(
            scene_planes=scene,
            witness_xyz=self.xyz,
            witness_normal=self.normal,
            witness_h=self.h,
            witness_mechanical=self.mechanics,
            input_ids=input_ids,
            labels=labels,
        )
        self.assertEqual(out.conditioning_memory.shape, (2, 8, 64))
        self.assertIsNotNone(out.objective)
        self.assertTrue(torch.isfinite(out.loss))


if __name__ == "__main__":
    unittest.main()
