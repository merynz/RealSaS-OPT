from __future__ import annotations

import torch
from torch import nn

from .contracts_v1 import IRISVNextConfigV1
from .evidence_heads_v3 import AppearanceEvidenceHeadV3, MechanicalDemandHeadV3
from .heads_v3 import (
    GraphicsRelationHeadV3,
    GraphicsRelationHeadV3Config,
    RayStackVisibilityHeadV3,
)
from .presentation_prediction_v1 import (
    PREDICTED_PRESENTATION_PROVENANCE_V1,
    pair_presentation_from_predictions_v1,
    ray_hit_presentation_from_predictions_v1,
    surface_presentation_from_predictions_v1,
)
from .scene_core_v1 import IRISVNextSceneCoreV1


class IRISVNextEvidenceModelV3(nn.Module):
    """IRIS continuous-scene/evidence model with topology factored out.

    TESSA is intentionally *not* a child head of this model. A single IRIS
    checkpoint can therefore be evaluated through either direct surface->Compiler
    or TESSA structuralization->Compiler without changing the scene model.

    Role-free presentation evidence is produced only from observation-time IRIS
    specialist predictions plus optional source observations. Materialized V2
    truth may supervise/evaluate those heads offline but is not an inference input.
    """

    architecture_id = "RealSaS.IRIS.VNextEvidenceModel.v3"
    output_semantics = "CONTINUOUS_SCENE_AND_TRAINED_EVIDENCE_ONLY"
    topology_decoder_embedded = False
    uncalibrated_uncertainty_public_output = False
    presentation_provenance = PREDICTED_PRESENTATION_PROVENANCE_V1
    materialized_v2_truth_in_inference_forward = False
    teacher_topology_in_inference_forward = False
    named_semantic_taxonomy_in_inference_forward = False

    def __init__(
        self,
        config: IRISVNextConfigV1 = IRISVNextConfigV1(),
        relation_config: GraphicsRelationHeadV3Config = GraphicsRelationHeadV3Config(),
        *,
        ray_visibility_width: int = 512,
        ray_visibility_heads: int = 8,
        ray_visibility_layers: int = 6,
        ray_visibility_mlp_ratio: int = 4,
        ray_visibility_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        config.validate()
        relation_config.validate()
        self.config = config
        self.scene_core = IRISVNextSceneCoreV1(config)
        # Historical V1 contains an uncertainty row for checkpoint compatibility,
        # but V8 supplied no calibration objective for it. V3 neither exposes nor
        # optimizes those parameters. A future calibrated uncertainty head must be
        # introduced as a new typed objective, not silently revive these weights.
        for parameter in self.scene_core.field.uncertainty_head.parameters():
            parameter.requires_grad_(False)
        self.mechanical_demand = MechanicalDemandHeadV3(config)
        self.appearance = AppearanceEvidenceHeadV3(config)
        self.relations = GraphicsRelationHeadV3(config, relation_config)
        self.ray_visibility = RayStackVisibilityHeadV3(
            config,
            width=ray_visibility_width,
            heads=ray_visibility_heads,
            layers=ray_visibility_layers,
            mlp_ratio=ray_visibility_mlp_ratio,
            dropout=ray_visibility_dropout,
        )

    def encode_scene(
        self,
        image_tokens: torch.Tensor,
        token_xy: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.scene_core.encode_scene(image_tokens, token_xy, camera_features)

    def query_local(
        self,
        scene_planes: torch.Tensor,
        points_normalized: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local = self.scene_core.query(scene_planes, points_normalized)
        # Historical scene_core still computes log_uncertainty for exact source
        # compatibility. V3 deliberately removes it from the public evidence API.
        return {
            "H": local["H"],
            "field": local["field"],
            "extraction_delta": local["extraction_delta"],
        }

    def query_evidence(
        self,
        scene_planes: torch.Tensor,
        points_normalized: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        local = self.query_local(scene_planes, points_normalized)
        return {
            **local,
            "mechanical_demand": self.mechanical_demand(local["H"]),
        }

    def query_appearance(
        self,
        h: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.appearance(h, camera_features)

    def query_pair_relations(
        self,
        *,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
        point_a: torch.Tensor,
        point_b: torch.Tensor,
        camera_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.relations.relation_probabilities(
            feature_a,
            feature_b,
            point_a,
            point_b,
            camera_features,
        )

    def query_ray_visibility(
        self,
        *,
        hit_h: torch.Tensor,
        hit_xyz: torch.Tensor,
        hit_depth: torch.Tensor,
        camera_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.ray_visibility(
            hit_h=hit_h,
            hit_xyz=hit_xyz,
            hit_depth=hit_depth,
            camera_features=camera_features,
            valid_mask=valid_mask,
        )

    def predict_surface_presentation(
        self,
        *,
        scene_planes: torch.Tensor,
        xyz: torch.Tensor,
        camera_features: torch.Tensor,
        token_valid_mask: torch.Tensor,
        source_rgba: torch.Tensor | None = None,
        source_rgba_valid: torch.Tensor | None = None,
        source_support: torch.Tensor | None = None,
        source_support_valid: torch.Tensor | None = None,
    ):
        local = self.query_local(scene_planes, xyz)
        appearance = self.query_appearance(local["H"], camera_features)
        return surface_presentation_from_predictions_v1(
            xyz=xyz,
            camera_features=camera_features,
            token_valid_mask=token_valid_mask,
            appearance=appearance,
            source_rgba=source_rgba,
            source_rgba_valid=source_rgba_valid,
            source_support=source_support,
            source_support_valid=source_support_valid,
        )

    def predict_pair_presentation(
        self,
        *,
        scene_planes: torch.Tensor,
        xyz_a: torch.Tensor,
        xyz_b: torch.Tensor,
        camera_features: torch.Tensor,
        token_valid_mask: torch.Tensor,
    ):
        h_a = self.query_local(scene_planes, xyz_a)["H"]
        h_b = self.query_local(scene_planes, xyz_b)["H"]
        relations = self.query_pair_relations(
            feature_a=h_a,
            feature_b=h_b,
            point_a=xyz_a,
            point_b=xyz_b,
            camera_features=camera_features,
        )
        return pair_presentation_from_predictions_v1(
            xyz_a=xyz_a,
            xyz_b=xyz_b,
            camera_features=camera_features,
            token_valid_mask=token_valid_mask,
            relations=relations,
        )

    def predict_ray_presentation(
        self,
        *,
        scene_planes: torch.Tensor,
        hit_xyz: torch.Tensor,
        hit_depth: torch.Tensor,
        camera_features: torch.Tensor,
        hit_valid_mask: torch.Tensor,
    ):
        hit_h = self.query_local(scene_planes, hit_xyz)["H"]
        visibility = self.query_ray_visibility(
            hit_h=hit_h,
            hit_xyz=hit_xyz,
            hit_depth=hit_depth,
            camera_features=camera_features,
            valid_mask=hit_valid_mask,
        )
        return ray_hit_presentation_from_predictions_v1(
            hit_xyz=hit_xyz,
            camera_features=camera_features,
            hit_valid_mask=hit_valid_mask,
            global_visibility_probability=visibility,
        )

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        scene = tuple(p for p in self.scene_core.parameters() if p.requires_grad)
        mechanical = tuple(p for p in self.mechanical_demand.parameters() if p.requires_grad)
        appearance = tuple(p for p in self.appearance.parameters() if p.requires_grad)
        relation = tuple(p for p in self.relations.parameters() if p.requires_grad)
        visibility = tuple(p for p in self.ray_visibility.parameters() if p.requires_grad)
        groups = {
            "scene_core": scene,
            "mechanical": mechanical,
            "appearance": appearance,
            "relation": relation,
            "visibility": visibility,
        }
        seen: set[int] = set()
        for name, parameters in groups.items():
            ids = {id(p) for p in parameters}
            if seen.intersection(ids):
                raise RuntimeError("IRIS_V3_PARAMETER_GROUP_OVERLAP:" + name)
            seen.update(ids)
        return groups

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
