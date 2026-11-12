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
class RSSvNextTypedQueryPairV1:
    """Computed, query-addressed relation, not an executable attachment.

    Predicted relation is the SAME anonymous-slot marginal used by RSS
    source-supervised typed relation training. Source labels, region identity,
    contact persistence and anchor coordinates are not inferred by this court.
    """
    from_query_index: int
    to_query_index: int
    same_drawable_affinity: float
    cut_affinity: float
    directed_attachment_affinity: float
    parent_predicted_known_region_mass: float
    child_predicted_known_region_mass: float
    nonself_slot_pair_mass: float
    parent_predicted_unknown_mass: float
    child_predicted_unknown_mass: float
    inference_evidence_hash: str
    neural_carrier_sha256: str
    authority: str = NO_PRODUCT_AUTHORITY
    source_contact_proven: bool = False
    attachment_anchor_proven: bool = False
    canonical_face_rebinding_proven: bool = False
    source_drawable_proven: bool = False
    source_visibility_proven: bool = False


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

    def query_pair_relations(
        self, ordered_query_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    ) -> tuple[RSSvNextTypedQueryPairV1, ...]:
        """Vectorized Compiler lowering of the *trained* typed relation.

        O(P*R^2) arithmetic, O(P*R + R^2) temporary memory: no neural
        reinference, no full N^2 query-pair materialization and no copying
        the persistent scene or 3D H(x) carrier. The diagonal does not count
        as a relation, and UNKNOWN stays a separate, reported mass.

        This computes the same conditional known-slot marginal as
        qualified_source_typed_relation_loss; an argmax slot or arbitrary
        source part mapping would be an irreversible evidence bottleneck.
        """
        if not isinstance(ordered_query_pairs, (tuple, list)):
            raise QualificationError("RSS_VNEXT_FANOUT_PAIR_BATCH_TYPE_INVALID")
        if len(ordered_query_pairs) > self.model_config.max_pair_queries:
            raise QualificationError("RSS_VNEXT_FANOUT_PAIR_BATCH_RESOURCE_LIMIT")
        if not ordered_query_pairs:
            return ()
        for pair in ordered_query_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise QualificationError("RSS_VNEXT_FANOUT_QUERY_PAIR_ADDRESS_INVALID")
            _checked_index(
                pair[0], self.witness_count,
                "RSS_VNEXT_FANOUT_QUERY_PAIR_ADDRESS_INVALID",
            )
            _checked_index(
                pair[1], self.witness_count,
                "RSS_VNEXT_FANOUT_QUERY_PAIR_ADDRESS_INVALID",
            )
            if pair[0] == pair[1]:
                raise QualificationError("RSS_VNEXT_FANOUT_QUERY_PAIR_SELF_EDGE_FORBIDDEN")

        t = self._tensors
        device = t["semantic_logits"].device
        indices = torch.tensor(ordered_query_pairs, dtype=torch.long, device=device)
        member = t["semantic_logits"][0].float()
        known = torch.softmax(member[..., :-1], dim=-1)
        unknown = torch.softmax(member, dim=-1)[..., -1]
        parent = known.index_select(0, indices[:, 0])
        child = known.index_select(0, indices[:, 1])
        region_valid = t["region_relation_valid_mask"][0].float()
        # IMPORTANT: region logits [:2] are symmetric, [2] is directed.
        # Train and compile share the EXACT conditional normalized formula.
        relation = torch.sigmoid(t["region_relation_logits"][0].float())
        # Never form an [P,R,R,3] dense temporary or a full [N,N] matrix.
        parent_to_other = torch.einsum("pi,ij->pj", parent, region_valid)
        denominator = (parent_to_other * child).sum(dim=-1)
        # Force the contracted intermediate to [P,R,C], not [P,R,R,C].
        # This guards CPU/GPU memory regardless of einsum path selection.
        parent_to_relation = torch.einsum(
            "pi,ijc->pjc", parent, relation * region_valid[..., None],
        )
        numerator = (parent_to_relation * child[..., None]).sum(dim=1)
        candidate = (numerator / denominator.clamp_min(1.e-10)[:, None]).clamp(
            1.e-6, 1. - 1.e-6,
        )
        full_known = 1. - unknown
        rows = []
        for index, pair in enumerate(ordered_query_pairs):
            # A vanishing cross-region normalization means the model has
            # not distinguished two region nodes. No positive attachment
            # claim may be emitted. We keep the evidence otherwise intact.
            d = float(denominator[index].item())
            if d <= 1.e-10:
                raise QualificationError("RSS_VNEXT_FANOUT_NO_CROSS_REGION_SUPPORT")
            values = candidate[index].tolist()
            rows.append(RSSvNextTypedQueryPairV1(
                from_query_index=int(pair[0]),
                to_query_index=int(pair[1]),
                same_drawable_affinity=float(values[0]),
                cut_affinity=float(values[1]),
                directed_attachment_affinity=float(values[2]),
                parent_predicted_known_region_mass=float(full_known[pair[0]].item()),
                child_predicted_known_region_mass=float(full_known[pair[1]].item()),
                nonself_slot_pair_mass=d,
                parent_predicted_unknown_mass=float(unknown[pair[0]].item()),
                child_predicted_unknown_mass=float(unknown[pair[1]].item()),
                inference_evidence_hash=self.evidence.evidence_hash,
                neural_carrier_sha256=self.neural_carrier_sha256,
            ))
        return tuple(rows)

    def query_pair_relation(
        self, from_query_index: int, to_query_index: int,
    ) -> RSSvNextTypedQueryPairV1:
        return self.query_pair_relations([(from_query_index, to_query_index)])[0]

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
    "RSSvNextTypedRegionRelationV1", "RSSvNextTypedQueryPairV1",
    "RSSvNextExpertAvailabilityV1",
    "RSSvNextExpertEvidenceFanoutV1", "open_rss_vnext_expert_fanout_v1",
]
