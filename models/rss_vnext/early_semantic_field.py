from __future__ import annotations

"""RSS.vNext semantic evidence is learned DURING construction of shared Z.

This is NOT a fixed human-anatomy ontology and NOT StdGEN's hard-coded
hair/cloth/canonical-solid manipulation. The independent evidence channels
can be masked per channel, per source view and per patch by admitted teacher
data; missing observations are UNKNOWN, not a negative category.

The feature update is residual: neither original source patch tokens nor
shared H(x) is replaced by a small class-probability vector.
"""
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RSSvNextEarlySemanticOutput:
    source_patch_logits: Tensor  # [B,2,T,K], independent evidence axes
    conditioned_image_tokens: Tensor  # [B,2,T,C], residual original
    source_evidence_uncertainty: Tensor  # [B,2,T,1], unqualified estimate
    authority: str = "TWO_SOURCE_VIEW_SEMANTIC_PREDICTION"


class RSSvNextEarlySemanticField(nn.Module):
    def __init__(self, image_dim: int, carrier_dim: int, evidence_channels: int, *, residual_scale: float = .1):
        super().__init__()
        if min(image_dim, carrier_dim, evidence_channels) < 1:
            raise ValueError("RSS_VNEXT_SEMANTIC_FIELD_DIMENSIONS")
        if not 0 < residual_scale <= .25:
            raise ValueError("RSS_VNEXT_SEMANTIC_FIELD_RESIDUAL_SCALE")
        self.image_dim = image_dim
        self.carrier_dim = carrier_dim
        self.evidence_channels = evidence_channels
        self.residual_scale = residual_scale
        self.image_evidence = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, image_dim),
            nn.GELU(),
            nn.Linear(image_dim, evidence_channels),
        )
        self.image_uncertainty = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, 1))
        # Feature generation retains image information, it is not just a one-hot label.
        self.image_residual = nn.Sequential(
            nn.LayerNorm(image_dim + evidence_channels),
            nn.Linear(image_dim + evidence_channels, image_dim),
            nn.GELU(),
            nn.Linear(image_dim, image_dim, bias=False),
        )
        self.field_evidence = nn.Sequential(
            nn.LayerNorm(carrier_dim),
            nn.Linear(carrier_dim, carrier_dim),
            nn.GELU(),
            nn.Linear(carrier_dim, evidence_channels),
        )

    def forward(self, image_tokens: Tensor) -> RSSvNextEarlySemanticOutput:
        if image_tokens.ndim != 4 or image_tokens.shape[1] != 2:
            raise ValueError("RSS_VNEXT_SEMANTIC_ONLY_TWO_PATCH_VIEW_INPUT")
        return self.forward_variable_views(image_tokens)

    def forward_variable_views(self, image_tokens: Tensor) -> RSSvNextEarlySemanticOutput:
        if (image_tokens.ndim != 4 or image_tokens.shape[1] not in (1,2,6)
                or image_tokens.shape[2] < 1 or image_tokens.shape[-1] != self.image_dim):
            raise ValueError("RSS_VNEXT_STDGEN_SEMANTIC_INPUT_VIEW_COUNT")
        if not bool(torch.isfinite(image_tokens).all()):
            raise ValueError("RSS_VNEXT_SEMANTIC_IMAGE_TOKENS_NONFINITE")
        logits = self.image_evidence(image_tokens)
        # Differentiable multi-label evidence, no mutually exclusive
        # "body OR garment" assumption; unlabeled axes do not become false.
        expected_roles = torch.sigmoid(logits)
        delta = self.image_residual(torch.cat((image_tokens, expected_roles), dim=-1))
        return RSSvNextEarlySemanticOutput(
            source_patch_logits=logits,
            conditioned_image_tokens=image_tokens + self.residual_scale * delta,
            source_evidence_uncertainty=self.image_uncertainty(image_tokens).clamp(-8, 4),
            authority=(
                "TWO_SOURCE_VIEW_SEMANTIC_PREDICTION"
                if image_tokens.shape[1]==2
                else "STDGEN_NON_SOURCE_REFERENCE_SEMANTIC_HYPOTHESIS_ONLY"
            ),
        )

    def query_field(self, local_h: Tensor) -> Tensor:
        if local_h.ndim != 3 or local_h.shape[-1] != self.carrier_dim:
            raise ValueError("RSS_VNEXT_SEMANTIC_FIELD_QUERY_SHAPE")
        return self.field_evidence(local_h)


def masked_source_semantic_evidence_loss(
    logits: Tensor,
    target: Tensor,
    qualified_mask: Tensor,
) -> Tensor:
    """Teacher-only BCE on independently qualified 2D evidence axes.

    No teacher map is accepted by the student's forward/scene encoder.
    A mask means *qualified source-target evidence exists*, not that the pixel
    is visible, surface-owned, mechanically attached or a source-depth hit.
    """
    if logits.ndim != 4 or logits.shape[1] != 2 or logits.shape != target.shape:
        raise ValueError("RSS_VNEXT_SEMANTIC_SUPERVISION_SHAPE")
    if qualified_mask.shape != logits.shape or qualified_mask.dtype != torch.bool:
        raise ValueError("RSS_VNEXT_SEMANTIC_SUPERVISION_MASK")
    if not bool(qualified_mask.any()):
        raise ValueError("RSS_VNEXT_SEMANTIC_NO_QUALIFIED_TARGET")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("RSS_VNEXT_SEMANTIC_PREDICTION_NONFINITE")
    valid_target = target[qualified_mask]
    if not bool(torch.isfinite(valid_target).all()) or bool(((valid_target < 0) | (valid_target > 1)).any()):
        raise ValueError("RSS_VNEXT_SEMANTIC_QUALIFIED_LABEL_RANGE")
    return F.binary_cross_entropy_with_logits(
        logits[qualified_mask].float(), valid_target.float(), reduction="mean",
    )


def masked_semantic_expected_brier_risk_loss(
    logits: Tensor,
    log_uncertainty: Tensor,
    target: Tensor,
    qualified_mask: Tensor,
) -> Tensor:
    """Supervise predicted patch uncertainty as conditional expected Brier risk.

    Exact per-axis teacher masks decide which semantic observations exist.
    Unobserved axes may contain NaN but never become positive/negative labels.
    A patch with no teacher axes never trains uncertainty. This measures
    source-patch classification error, not source depth or drawable truth.

    The historical 'log_uncertainty' ABI name is kept. Its sigmoid decodes
    expected Brier risk, not a Gaussian log-variance or proof certificate.
    """
    if (
        logits.ndim != 4 or logits.shape[1] != 2
        or target.shape != logits.shape
        or qualified_mask.shape != logits.shape
        or log_uncertainty.shape != (*logits.shape[:3], 1)
    ):
        raise ValueError("RSS_VNEXT_SEMANTIC_RISK_SHAPE")
    if qualified_mask.dtype != torch.bool:
        raise ValueError("RSS_VNEXT_SEMANTIC_RISK_MASK_DTYPE")
    if not bool(qualified_mask.any()):
        raise ValueError("RSS_VNEXT_SEMANTIC_RISK_NO_QUALIFIED_AXES")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(log_uncertainty).all()):
        raise ValueError("RSS_VNEXT_SEMANTIC_RISK_PREDICTION_NONFINITE")
    observed = target[qualified_mask]
    if not bool(torch.isfinite(observed).all()) or bool(((observed < 0) | (observed > 1)).any()):
        raise ValueError("RSS_VNEXT_SEMANTIC_RISK_TARGET_INVALID")
    safe_target = torch.where(qualified_mask, target, torch.zeros_like(target))
    squared_error = (torch.sigmoid(logits) - safe_target).square()
    axes_per_patch = qualified_mask.sum(dim=-1)
    measured_risk = (
        squared_error.masked_fill(~qualified_mask, 0.).sum(dim=-1)
        / axes_per_patch.clamp_min(1)
    ).detach()
    admitted_patch = axes_per_patch > 0
    predicted_risk = torch.sigmoid(log_uncertainty[..., 0])
    return F.mse_loss(
        predicted_risk[admitted_patch],
        measured_risk[admitted_patch],
    )


__all__ = [
    "RSSvNextEarlySemanticOutput", "RSSvNextEarlySemanticField",
    "masked_source_semantic_evidence_loss",
    "masked_semantic_expected_brier_risk_loss",
]
