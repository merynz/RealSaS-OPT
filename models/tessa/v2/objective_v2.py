from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config_v2 import TESSAStructuralizerConfigV2


@dataclass(frozen=True)
class TESSAObjectiveWeightsV2:
    token_cross_entropy: float = 1.0
    coordinate_metric: float = 0.5
    structural_special_token: float = 0.25

    def validate(self) -> None:
        values = (
            self.token_cross_entropy,
            self.coordinate_metric,
            self.structural_special_token,
        )
        if min(values) < 0.0 or sum(values) <= 0.0:
            raise ValueError("TESSA_V2_OBJECTIVE_WEIGHT_INVALID")


@dataclass
class TESSAObjectiveResultV2:
    total: torch.Tensor
    token_cross_entropy: torch.Tensor
    coordinate_metric: torch.Tensor
    structural_special_token: torch.Tensor
    coordinate_expected_abs_error_bins: torch.Tensor
    valid_token_count: int
    coordinate_token_count: int
    structural_token_count: int


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.new_zeros(())


def topology_sequence_objective_v2(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: TESSAStructuralizerConfigV2 | None = None,
    weights: TESSAObjectiveWeightsV2 = TESSAObjectiveWeightsV2(),
) -> TESSAObjectiveResultV2:
    """Topology-token objective with metric awareness for coordinate tokens.

    Hard next-token cross entropy remains the grammar/generation objective, but
    coordinates live on an ordered lattice.  A one-bin coordinate miss and a
    five-hundred-bin miss are not geometrically equivalent, so V2 adds a metric
    objective on the expected coordinate under the coordinate-token posterior.

    Free-running mesh validity/topology quality is deliberately *not* folded
    into this differentiable loss.  It belongs to an independent decoded-mesh
    court so teacher forcing cannot masquerade as a topology PASS.
    """

    cfg = config or TESSAStructuralizerConfigV2()
    cfg.validate()
    weights.validate()

    if logits.ndim != 3 or logits.shape[-1] != cfg.vocab_size:
        raise ValueError("TESSA_V2_LOGIT_SHAPE_INVALID")
    if labels.shape != logits.shape[:2]:
        raise ValueError("TESSA_V2_LABEL_SHAPE_INVALID")
    if labels.numel() and (
        int(labels.min()) < 0 or int(labels.max()) >= cfg.vocab_size
    ):
        raise ValueError("TESSA_V2_LABEL_OUT_OF_RANGE")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid = flat_labels != cfg.PAD
    coordinate = valid & (flat_labels < cfg.coordinate_bins)
    structural = valid & (flat_labels >= cfg.coordinate_bins)

    valid_count = int(valid.sum().item())
    coordinate_count = int(coordinate.sum().item())
    structural_count = int(structural.sum().item())
    if valid_count <= 0:
        raise ValueError("TESSA_V2_OBJECTIVE_EMPTY")

    raw_ce = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="none",
    )
    token_ce = raw_ce[valid].mean()
    special_ce = raw_ce[structural].mean() if structural_count else _zero_like(logits)

    coordinate_metric = _zero_like(logits)
    expected_abs_error_bins = _zero_like(logits)
    if coordinate_count:
        selected_logits = flat_logits[coordinate, : cfg.coordinate_bins].float()
        selected_labels = flat_labels[coordinate]
        probability = torch.softmax(selected_logits, dim=-1)
        bins = torch.arange(
            cfg.coordinate_bins,
            dtype=probability.dtype,
            device=probability.device,
        )
        expected_bin = (probability * bins[None, :]).sum(dim=-1)
        target_bin = selected_labels.to(dtype=expected_bin.dtype)

        # Normalize the metric term to the canonical coordinate span so its
        # scale does not grow merely because V2 keeps the finer 1024-bin lattice.
        expected_coordinate = (expected_bin + 0.5) / float(cfg.coordinate_bins) - 0.5
        target_coordinate = (target_bin + 0.5) / float(cfg.coordinate_bins) - 0.5
        coordinate_metric = F.smooth_l1_loss(
            expected_coordinate,
            target_coordinate,
        )
        expected_abs_error_bins = (expected_bin - target_bin).abs().mean()

    total = (
        float(weights.token_cross_entropy) * token_ce
        + float(weights.coordinate_metric) * coordinate_metric
        + float(weights.structural_special_token) * special_ce
    )
    if not bool(torch.isfinite(total)):
        raise RuntimeError("TESSA_V2_OBJECTIVE_NONFINITE")

    return TESSAObjectiveResultV2(
        total=total,
        token_cross_entropy=token_ce,
        coordinate_metric=coordinate_metric,
        structural_special_token=special_ce,
        coordinate_expected_abs_error_bins=expected_abs_error_bins,
        valid_token_count=valid_count,
        coordinate_token_count=coordinate_count,
        structural_token_count=structural_count,
    )
