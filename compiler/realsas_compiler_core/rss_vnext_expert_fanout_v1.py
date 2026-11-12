from __future__ import annotations

"""One RSS.vNext shared neural carrier; independent typed expert consumers.

Semantic, Geometry, Graphics/2D Presentation, and Mechanics are specialist
readouts over the SAME H(x)/region graph. Nothing is flattened into a proxy
mesh, arbitrarily remapped to body-part classes, or promoted to attachment,
source appearance, visible occlusion, signed solid or executable mechanics.

This reader validates the source-bound neural archive ONCE, then provides
index-addressed views without copying/reducing the shared tensors. A TESSA
ABSTAIN does not suppress Semantic, Graphics, or latent mechanical evidence.
"""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from typing import Any, Mapping

import torch
from torch import Tensor

from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1, validate_inference_evidence_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.rss_vnext.contracts import RSSvNextConfig
from models.rss_vnext.neural_scene_archive_v1 import read_neural_scene_carrier_v1

SCHEMA = "RealSaS.RSSvNext.ExpertEvidenceFanout.v1"
REGION_CHANNELS = (
    "same_drawable_affinity", "cut_affinity", "directed_attachment_affinity",
)
GRAPHICS_CHANNELS = (
    "rgb_logit_r", "rgb_logit_g", "rgb_logit_b",
    "density_logit", "uncertainty_residual_logit",
)
NO_PRODUCT_AUTHORITY = "NEURAL_EVIDENCE_ONLY_NO_PRODUCT_AUTHORITY"


def _float_tuple(value: Tensor) -> tuple[float, ...]:
    return tuple(float(x) for x in value.detach().float().reshape(-1).tolist())


def _softmax(value: Tensor) -> tuple[float, ...]:
    return _float_tuple(torch.softmax(value.detach().float(), dim=-1))


def _prob(value: Tensor) -> float:
    return float(torch.sigmoid(value.detach().float()).item())


def _checked_index(value: int, maximum: int, code: str) -> None:
    if type(value) is not int or not 0 <= value < maximum:
        raise QualificationError(code)


@dataclass(frozen=True)
class RSSvNextTypedSourcePatchV1:
    observed_view_index: int
    patch_index: int
    semantic_evidence_logits: tuple[float, ...]
    predicted_patch_risk_logit: float
    authority: str = NO_PRODUCT_AUTHORITY
    is_original_source_label: bool = False


@dataclass(frozen=True)
class RSSvNextTypedQueryV1:
    query_index: int
    xyz: tuple[float, float, float]
    # Raw H(x) is intentionally retained; the shared full carrier is not
    # replaced by this small, on-demand readout.
    local_h: tuple[float, ...]
    geometry_field_values: tuple[tuple[str, tuple[float, ...]], ...]
    region_membership_logits: tuple[float, ...]
    region_membership_probabilities: tuple[float, ...]  # last = UNKNOWN
    presentation_role_logits: tuple[float, ...]
    presentation_role_probabilities: tuple[float, ...]
    graphics_field_logits: tuple[float, float, float, float, float]
    deformation_affordance_logits: tuple[float, ...]  # last = UNKNOWN
    authority: str = NO_PRODUCT_AUTHORITY
    canonical_vertex_id: str | None = None
    source_rgb_qualification: bool = False
    source_visibility_qualification: bool = False
    executable_mechanics_qualification: bool = False


@dataclass(frozen=True)
class RSSvNextTypedRegionRelationV1:
    from_slot_index: int
    to_slot_index: int
    relation_channel_names: tuple[str, str, str]
    raw_logits: tuple[float, float, float]
    same_drawable_affinity: float
    cut_affinity: float
    directed_attachment_affinity: float
    authority: str = NO_PRODUCT_AUTHORITY
    attachment_parent_joint_id: str | None = None
    attachment_anchor_xyz: tuple[float, float, float] | None = None
    source_contact_proven: bool = False
    independent_drawable_proven: bool = False


@dataclass(frozen=True)
class RSSvNextExpertAvailabilityV1:
    semantic_and_presentation: str
    geometry_and_graphics_field: str
    relational_mechanics: str
    decoded_mesh_candidate: str
    causal_rig_program_candidate: str
    canonical_mesh: str = "NOT_QUALIFIED"
    canonical_rig_skin: str = "NOT_QUALIFIED"
    pose_conditioned_source_visibility: str = "NOT_QUALIFIED"


class RSSvNextExpertEvidenceFanoutV1:
    """Bounded typed access to ONE qualified RSS inference neural carrier.

    Distinct expert readouts are cheap views on full, immutable tensor
    evidence. They do not pretend that unknown faces, trained confidence or
    source material can be calculated from categorical slots.
    """

    def __init__(
        self,
        *,
        evidence: RSSvNextInferenceEvidenceIRV1,
        carrier_bytes: bytes,
        expected_checkpoint_sha256: str,
        expected_model_config_sha256: str,
        model_config: RSSvNextConfig,
    ):
        validate_inference_evidence_v1(
            evidence, expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        if type(model_config) is not RSSvNextConfig:
            raise QualificationError("RSS_VNEXT_FANOUT_CONFIG_REQUIRED")
        model_config.validate()
        model_config_sha256 = sha256(json.dumps(
            asdict(model_config), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf8")).hexdigest()
        if (
            type(expected_model_config_sha256) is not str
            or model_config_sha256 != expected_model_config_sha256
        ):
            raise QualificationError("RSS_VNEXT_FANOUT_MODEL_CONFIG_BINDING_DRIFT")
        tensors = read_neural_scene_carrier_v1(carrier_bytes, evidence=evidence)
        n = model_config.query_grid_side ** 3
        r = model_config.region_slots
        h = model_config.iris.local_feature_dim
        if (
            tensors["query_xyz"].shape != (1, n, 3)
            or tensors["local_features"].shape != (1, n, h)
            or tensors["geometry__H"].shape != (1, n, h)
            or not torch.equal(tensors["geometry__H"], tensors["local_features"])
            or tensors["semantic_logits"].shape != (1, n, r + 1)
            or tensors["affordance_logits"].shape != (
                1, n, model_config.affordance_classes + 1
            )
            or tensors["graphics_logits"].shape != (1, n, 5)
            or tensors["presentation_role_logits"].shape != (
                1, n, len(model_config.semantic_role_names)
            )
            or tensors["region_relation_logits"].shape != (1, r, r, 3)
            or tensors["region_relation_valid_mask"].shape != (1, r, r)
            or tensors["source_semantic_patch_logits"].shape[-1]
               != model_config.semantic_evidence_channels
        ):
            raise QualificationError("RSS_VNEXT_FANOUT_MODEL_EXPERT_SHAPE_DRIFT")
        mask = tensors["region_relation_valid_mask"][0]
        if any(bool(mask[i, j]) != (i != j) for i in range(r) for j in range(r)):
            raise QualificationError("RSS_VNEXT_FANOUT_RELATION_SELF_EDGE_DRIFT")
        relation = tensors["region_relation_logits"][0]
        if not torch.allclose(
            relation[..., :2], relation.transpose(0, 1)[..., :2],
            rtol=0.0, atol=1.0e-5,
        ):
            raise QualificationError("RSS_VNEXT_FANOUT_SYMMETRIC_REGION_CHANNEL_DRIFT")
        self.schema = SCHEMA
        self.evidence = evidence
        self.model_config = model_config
        self.model_config_sha256 = model_config_sha256
        self.neural_carrier_sha256 = sha256(carrier_bytes).hexdigest()
        # Internal references, no dense aggregate summary or tensor copies.
        self._tensors = tensors

    @property
    def witness_count(self) -> int:
        return self._tensors["query_xyz"].shape[1]

    @property
    def region_slot_count(self) -> int:
        return self.model_config.region_slots

    @property
    def availability(self) -> RSSvNextExpertAvailabilityV1:
        return RSSvNextExpertAvailabilityV1(
            semantic_and_presentation="TYPED_NEURAL_CANDIDATE",
            geometry_and_graphics_field="TYPED_NEURAL_CANDIDATE",
            relational_mechanics="TYPED_AFFORDANCE_AND_DIRECTED_SLOT_RELATIONS",
            decoded_mesh_candidate=(
                "UNQUALIFIED" if self.evidence.mesh_status != "ABSTAIN"
                else "ABSTAIN_NO_TOPOLOGY"
            ),
            causal_rig_program_candidate=(
                "UNQUALIFIED_EVENTS_REQUIRE_CODEC_COURT"
                if self.evidence.control_events else "NO_TYPED_EVENTS"
            ),
        )

    def original_source_patch(
        self, observed_view_index: int, patch_index: int,
    ) -> RSSvNextTypedSourcePatchV1:
        patches = self._tensors["source_semantic_patch_logits"]
        _checked_index(observed_view_index, 2, "RSS_VNEXT_FANOUT_SOURCE_VIEW_NOT_OBSERVED")
        _checked_index(
            patch_index, patches.shape[2], "RSS_VNEXT_FANOUT_SOURCE_PATCH_ADDRESS_INVALID",
        )
        risk = self._tensors["source_semantic_log_uncertainty"][
            0, observed_view_index, patch_index
        ]
        if risk.numel() != 1:
            raise QualificationError("RSS_VNEXT_FANOUT_PATCH_RISK_WIDTH_DRIFT")
        return RSSvNextTypedSourcePatchV1(
            observed_view_index=observed_view_index,
            patch_index=patch_index,
            semantic_evidence_logits=_float_tuple(
                patches[0, observed_view_index, patch_index],
            ),
            predicted_patch_risk_logit=float(risk.item()),
        )

    def query(self, query_index: int) -> RSSvNextTypedQueryV1:
        _checked_index(
            query_index, self.witness_count, "RSS_VNEXT_FANOUT_QUERY_ADDRESS_INVALID",
        )
        t = self._tensors
        member = t["semantic_logits"][0, query_index]
        roles = t["presentation_role_logits"][0, query_index]
        geometry = tuple(
            (name.removeprefix("geometry__"), _float_tuple(value[0, query_index]))
            for name, value in sorted(t.items())
            if name.startswith("geometry__")
        )
        rgb = _float_tuple(t["graphics_logits"][0, query_index])
        return RSSvNextTypedQueryV1(
            query_index=query_index,
            xyz=_float_tuple(t["query_xyz"][0, query_index]),
            local_h=_float_tuple(t["local_features"][0, query_index]),
            geometry_field_values=geometry,
            region_membership_logits=_float_tuple(member),
            region_membership_probabilities=_softmax(member),
            presentation_role_logits=_float_tuple(roles),
            presentation_role_probabilities=_softmax(roles),
            graphics_field_logits=rgb,  # raw, NOT calibrated source RGB
            deformation_affordance_logits=_float_tuple(
                t["affordance_logits"][0, query_index],
            ),
        )

    def region_relation(
        self, from_slot_index: int, to_slot_index: int,
    ) -> RSSvNextTypedRegionRelationV1:
        _checked_index(
            from_slot_index, self.region_slot_count,
            "RSS_VNEXT_FANOUT_FROM_SLOT_INVALID",
        )
        _checked_index(
            to_slot_index, self.region_slot_count,
            "RSS_VNEXT_FANOUT_TO_SLOT_INVALID",
        )
        if from_slot_index == to_slot_index:
            raise QualificationError("RSS_VNEXT_FANOUT_RELATION_SELF_EDGE_UNOBSERVED")
        raw = self._tensors["region_relation_logits"][
            0, from_slot_index, to_slot_index,
        ]
        return RSSvNextTypedRegionRelationV1(
            from_slot_index=from_slot_index,
            to_slot_index=to_slot_index,
            relation_channel_names=REGION_CHANNELS,
            raw_logits=_float_tuple(raw),
            same_drawable_affinity=_prob(raw[0]),
            cut_affinity=_prob(raw[1]),
            directed_attachment_affinity=_prob(raw[2]),
        )

    def region_relation_matrix_v1(self) -> Tensor:
        """Bounded R×R typed view; no giant N×N or F×F duplication.

        Return a detached *copy* of the small learned relation expert, never
        a mutable alias to the shared scene. Consumers must retain the raw
        directed attachment channel and reject diagonal self-edges.
        """
        return self._tensors["region_relation_logits"][0].detach().float().clone()

    def causal_control_events(self) -> tuple[Mapping[str, Any], ...]:
        """Do not execute until AXIS/MIRA independent codec court is passed."""
        return self.evidence.control_events

    def qualify_causal_rig_proposal(self, *, mira_codec_binding: Any) -> Any:
        """Use the existing AXIS grammar/codec court, not latent interpretation."""
        from models.rss_vnext.candidate_rig_court_v1 import (
            qualify_candidate_rig_program_v1,
        )
        return qualify_candidate_rig_program_v1(
            self.evidence,
            model_config=self.model_config,
            expected_checkpoint_sha256=self.evidence.checkpoint_sha256,
            mira_codec_binding=mira_codec_binding,
        )


def open_rss_vnext_expert_fanout_v1(
    evidence: RSSvNextInferenceEvidenceIRV1,
    carrier_bytes: bytes, *,
    expected_checkpoint_sha256: str,
    expected_model_config_sha256: str,
    model_config: RSSvNextConfig,
) -> RSSvNextExpertEvidenceFanoutV1:
    return RSSvNextExpertEvidenceFanoutV1(
        evidence=evidence, carrier_bytes=carrier_bytes,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_model_config_sha256=expected_model_config_sha256,
        model_config=model_config,
    )


__all__ = [
    "SCHEMA", "REGION_CHANNELS", "GRAPHICS_CHANNELS",
    "RSSvNextTypedSourcePatchV1", "RSSvNextTypedQueryV1",
    "RSSvNextTypedRegionRelationV1", "RSSvNextExpertAvailabilityV1",
    "RSSvNextExpertEvidenceFanoutV1", "open_rss_vnext_expert_fanout_v1",
]
