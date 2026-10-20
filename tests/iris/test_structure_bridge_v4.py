import unittest

import torch
import torch.nn.functional as F

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.presentation_evidence_v1 import (
    PRESENTATION_KIND_RAY_HIT,
    build_role_free_presentation_evidence_v1,
)
from models.iris.vnext.structure_v4 import IRISVNextStructuralizationBridgeV4
from models.tessa.v3 import TESSAStructuralizerConfigV3


class IRISVNextStructuralizationBridgeV4Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.iris = IRISVNextConfigV1()
        self.tessa = TESSAStructuralizerConfigV3(
            d_model=64,
            n_heads=8,
            decoder_layers=1,
            coordinate_bins=128,
            scene_memory_tokens=4,
            surface_memory_tokens=4,
            presentation_memory_tokens=2,
            conditioning_token_budget=10,
            memory_source_width=32,
            memory_refinement_layers=1,
            local_attention_window=16,
            query_chunk_size=4,
            gradient_checkpointing=False,
        )
        self.bridge = IRISVNextStructuralizationBridgeV4(
            iris_config=self.iris,
            tessa_config=self.tessa,
        )
        self.scene = torch.randn(1, 3, self.iris.plane_channels, 3, 3)
        self.xyz = torch.randn(1, 5, 3).clamp(-0.3, 0.3)
        self.normal = F.normalize(torch.randn(1, 5, 3), dim=-1)
        self.h = torch.randn(1, 5, self.iris.local_feature_dim)
        self.mechanics = {
            "stretch_major": torch.rand(1, 5) + 1.0,
            "stretch_minor": torch.rand(1, 5),
            "tangent_direction_2theta": F.normalize(torch.randn(1, 5, 2), dim=-1),
            "bend_demand": torch.rand(1, 5),
            "rigidity_probability": torch.rand(1, 5),
        }
        self.input_ids = torch.tensor([[self.tessa.BOS, self.tessa.COMPONENT_BEGIN, self.tessa.CHART_BEGIN, self.tessa.FACE_BREAK]])

    def test_null_control_and_evidence_arm_have_same_conditioning_shape(self):
        null_out = self.bridge(
            scene_planes=self.scene,
            witness_xyz=self.xyz,
            witness_normal=self.normal,
            witness_h=self.h,
            witness_mechanical=self.mechanics,
            input_ids=self.input_ids,
        )
        p_xyz = self.xyz[:, :3]
        visibility = torch.tensor([[1.0, 0.0, 0.0]])
        presentation = build_role_free_presentation_evidence_v1(
            xyz=p_xyz,
            kind_id=torch.full((1, 3), PRESENTATION_KIND_RAY_HIT, dtype=torch.long),
            token_valid_mask=torch.ones(1, 3, dtype=torch.bool),
            global_visibility=visibility,
            global_visibility_valid=torch.ones(1, 3, dtype=torch.bool),
        )
        evidence_out = self.bridge(
            scene_planes=self.scene,
            witness_xyz=self.xyz,
            witness_normal=self.normal,
            witness_h=self.h,
            witness_mechanical=self.mechanics,
            input_ids=self.input_ids,
            presentation=presentation,
        )
        self.assertEqual(null_out.conditioning_memory.shape, evidence_out.conditioning_memory.shape)
        self.assertEqual(null_out.conditioning_memory.shape, (1, 10, 64))

    def test_v4_does_not_admit_teacher_topology_or_named_semantics(self):
        self.assertFalse(self.bridge.named_semantic_taxonomy_required)
        self.assertFalse(self.bridge.teacher_topology_in_inference)
        self.assertEqual(self.bridge.output_authority, "LEARNED_MESH_CANDIDATE_PROPOSAL_ONLY")

    def test_presentation_dtype_mismatch_fails_closed(self):
        presentation = build_role_free_presentation_evidence_v1(
            xyz=self.xyz[:, :1].double(),
            kind_id=torch.full((1, 1), PRESENTATION_KIND_RAY_HIT, dtype=torch.long),
            token_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, "PRESENTATION_DTYPE_MISMATCH"):
            self.bridge(
                scene_planes=self.scene,
                witness_xyz=self.xyz,
                witness_normal=self.normal,
                witness_h=self.h,
                witness_mechanical=self.mechanics,
                input_ids=self.input_ids,
                presentation=presentation,
            )


if __name__ == "__main__":
    unittest.main()
