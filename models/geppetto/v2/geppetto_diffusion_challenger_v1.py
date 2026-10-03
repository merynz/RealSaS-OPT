from __future__ import annotations

"""D1 Geppetto challenger: frozen D0 recurrence + conditional diffusion loci.

The class deliberately subclasses the current Geppetto candidate so every
non-locus mechanism stays identical. Only _locus_distribution is overridden.
Because parent evidence consumes realized XYZ, parent logits are then
recomputed by the inherited decoder from the D1 representative positions.

This is research-only and emits the same SkeletonProposalIR authority class.
"""

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json

import torch

from .geppetto_candidate_v2 import (
    GeppettoCandidateConfigV2,
    GeppettoCandidateV2,
)
from .joint_locus_diffusion_v1 import (
    ConditionalDiffusionLocusConfigV1,
    ConditionalDiffusionLocusHeadV1,
)


def _hash(payload) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class GeppettoDiffusionChallengerConfigV1:
    diffusion: ConditionalDiffusionLocusConfigV1 = field(
        default_factory=ConditionalDiffusionLocusConfigV1
    )
    inference_seed: int = 91373
    architecture_id: str = (
        "RealSaS.GeppettoCandidate.D1ConditionalDiffusionLocus.v1"
    )

    def validate(self, base: GeppettoCandidateConfigV2) -> None:
        self.diffusion.validate()
        if self.diffusion.condition_dim != base.model_dim:
            raise ValueError("D1 diffusion condition_dim must equal Geppetto model_dim")
        if self.diffusion.sample_count != base.position_modes:
            raise ValueError("D1 sample_count must equal Geppetto position_modes")
        if abs(
            float(self.diffusion.position_scale)
            - float(base.position_scale)
        ) > 1e-12:
            raise ValueError("D1 position scale must equal Geppetto position scale")
        if int(self.inference_seed) < 0:
            raise ValueError("D1 inference seed invalid")

    def config_hash(self, base: GeppettoCandidateConfigV2) -> str:
        self.validate(base)
        return _hash(
            {
                "schema":"RealSaS.GeppettoDiffusionChallengerConfig.v1",
                "base_config_hash":base.config_hash,
                "challenger":asdict(self),
            }
        )


class GeppettoCandidateDiffusionLocusV1(GeppettoCandidateV2):
    def __init__(
        self,
        base_config: GeppettoCandidateConfigV2 = GeppettoCandidateConfigV2(),
        challenger_config: GeppettoDiffusionChallengerConfigV1 | None = None,
    ):
        challenger_config = (
            challenger_config
            if challenger_config is not None
            else GeppettoDiffusionChallengerConfigV1(
                diffusion=ConditionalDiffusionLocusConfigV1(
                    condition_dim=base_config.model_dim,
                    sample_count=base_config.position_modes,
                    position_scale=base_config.position_scale,
                )
            )
        )
        challenger_config.validate(base_config)
        super().__init__(base_config)
        self.challenger_config = challenger_config
        self.diffusion_locus_head = ConditionalDiffusionLocusHeadV1(
            challenger_config.diffusion
        )

    @property
    def challenger_config_hash(self) -> str:
        return self.challenger_config.config_hash(self.config)

    def _locus_distribution(
        self, h: torch.Tensor, *, step_index: int
    ):
        # Prime step stride avoids accidental identical RNG streams between
        # adjacent autoregressive controls while remaining exact-replay stable.
        seed = (
            int(self.challenger_config.inference_seed)
            + 104729 * int(step_index)
        )
        return self.diffusion_locus_head.distribution(
            h,
            seed=seed,
            sample_count=self.config.position_modes,
        )

    def load_frozen_base_state_dict(self, state_dict) -> dict:
        result = self.load_state_dict(state_dict, strict=False)
        unexpected = tuple(result.unexpected_keys)
        missing = tuple(result.missing_keys)
        invalid_missing = tuple(
            k for k in missing
            if not k.startswith("diffusion_locus_head.")
        )
        if unexpected or invalid_missing:
            raise RuntimeError(
                "D1_BASE_CHECKPOINT_CONTRACT_DRIFT:"
                f"missing={invalid_missing}:unexpected={unexpected}"
            )
        return {
            "missing_diffusion_keys":missing,
            "unexpected_keys":unexpected,
            "base_checkpoint_compatible":True,
        }

    def freeze_base_for_head_fit(self) -> dict:
        trainable = []
        frozen = []
        for name, param in self.named_parameters():
            flag = name.startswith("diffusion_locus_head.")
            param.requires_grad_(flag)
            (trainable if flag else frozen).append(name)
        if not trainable:
            raise RuntimeError("D1_DIFFUSION_HEAD_HAS_NO_TRAINABLE_PARAMETERS")
        return {
            "trainable":tuple(trainable),
            "frozen":tuple(frozen),
        }

    @torch.no_grad()
    def propose(
        self,
        conditioning,
        *,
        device=None,
        resource_step_limit: int | None = None,
    ):
        proposals = super().propose(
            conditioning,
            device=device,
            resource_step_limit=resource_step_limit,
        )
        out = []
        for proposal in proposals:
            md = dict(proposal.metadata or {})
            md.update(
                {
                    "candidate_architecture":
                        self.challenger_config.architecture_id,
                    "locus_distribution_kind":
                        self.diffusion_locus_head.distribution_kind,
                    "locus_distribution_config_hash":
                        self.challenger_config_hash,
                    "locus_recurrence_feedback":
                        "LATENT_ONLY__NO_GENERATED_XYZ_FEEDBACK",
                    "diffusion_inference_seed":
                        int(self.challenger_config.inference_seed),
                    "compiler_owns_root_tree_ids":True,
                }
            )
            out.append(
                replace(
                    proposal,
                    model_provenance=self.challenger_config_hash,
                    metadata=md,
                )
            )
        return tuple(out)


__all__ = [
    "GeppettoDiffusionChallengerConfigV1",
    "GeppettoCandidateDiffusionLocusV1",
]
