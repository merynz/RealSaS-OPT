from __future__ import annotations

import torch
from torch import nn

from .contracts_v1 import IRISVNextConfigV1
from .evidence_heads_v3 import AppearanceEvidenceHeadV3, MechanicalDemandHeadV3
from .heads_v3 import GraphicsRelationHeadV3, GraphicsRelationHeadV3Config
from .scene_core_v1 import IRISVNextSceneCoreV1


class IRISVNextEvidenceModelV3(nn.Module):
    """IRIS continuous-scene/evidence model with topology factored out.

    TESSA is intentionally *not* a child head of this model. A single IRIS
    checkpoint can therefore be evaluated through either direct surface->Compiler
    or TESSA structuralization->Compiler without changing the scene model.
    """

    architecture_id = "RealSaS.IRIS.VNextEvidenceModel.v3"
    output_semantics = "CONTINUOUS_SCENE_AND_TRAINED_EVIDENCE_ONLY"
    topology_decoder_embedded = False
    uncalibrated_uncertainty_public_output = False

    def __init__(
        self,
        config: IRISVNextConfigV1 = IRISVNextConfigV1(),
        relation_config: GraphicsRelationHeadV3Config = GraphicsRelationHeadV3Config(),
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

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        scene = tuple(p for p in self.scene_core.parameters() if p.requires_grad)
        mechanical = tuple(p for p in self.mechanical_demand.parameters() if p.requires_grad)
        appearance = tuple(p for p in self.appearance.parameters() if p.requires_grad)
        relation = tuple(p for p in self.relations.parameters() if p.requires_grad)
        groups = {
            "scene_core": scene,
            "mechanical": mechanical,
            "appearance": appearance,
            "relation": relation,
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
