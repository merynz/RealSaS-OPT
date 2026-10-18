from __future__ import annotations

"""IRIS vNext structure head v2: bounded projection into TESSA canonical space.

The differentiable extractor operates on an IRIS grid in [-1, 1] and permits a
bounded learned extraction_delta. Therefore a valid extracted witness may lie
slightly outside [-1, 1] by at most config.maximum_extraction_delta. TESSA's
structure memory, however, has an exact canonical XYZ contract of [-0.5, 0.5].

V2 reconciles those two contracts without changing extraction geometry or its
loss: it verifies that any overshoot is within the model-authorized extraction
delta, then projects only the topology-memory coordinate into the canonical
TESSA cube. Larger excursions remain fail-closed.

Sequence-position semantics remain backward compatible with StructureHeadV1:
omitting `sequence_position_offset` means scalar offset 0. Batched per-subject
offsets remain supported when an explicit tensor is supplied.
"""

import torch

from .structure_v1 import IRISVNextStructureHeadV1


class IRISVNextStructureHeadV2(IRISVNextStructureHeadV1):
    architecture_id = "RealSaS.IRIS.VNext.StructureHead.v2"
    witness_coordinate_semantics = (
        "IRIS_EXTRACTED_XYZ_WITHIN_AUTHORIZED_DELTA__"
        "PROJECT_TO_TESSA_CANONICAL_MINUS_HALF_TO_HALF"
    )

    def forward(
        self,
        *,
        scene_planes,
        witness_xyz,
        witness_normal,
        witness_h,
        witness_log_uncertainty,
        witness_mechanical,
        input_ids,
        labels=None,
        sequence_position_offset=0,
        witness_valid_mask=None,
    ):
        if sequence_position_offset is None:
            # Compatibility with older call sites that explicitly forwarded
            # None to mean "use the decoder default".
            sequence_position_offset = 0

        if witness_xyz.numel():
            observed_max = float(witness_xyz.detach().abs().max())
            # witness_xyz is already in TESSA scale (IRIS * 0.5). The extractor
            # may move the IRIS grid by at most maximum_extraction_delta.
            authorized_max = 0.5 * (1.0 + float(self.cfg.maximum_extraction_delta))
            if observed_max > authorized_max + 1.0e-6:
                raise ValueError(
                    "IRIS_VNEXT_WITNESS_COORDINATE_EXCEEDS_AUTHORIZED_EXTRACTION_DELTA:"
                    f"observed={observed_max}:authorized={authorized_max}"
                )
            witness_xyz = witness_xyz.clamp(-0.5, 0.5)

        return super().forward(
            scene_planes=scene_planes,
            witness_xyz=witness_xyz,
            witness_normal=witness_normal,
            witness_h=witness_h,
            witness_log_uncertainty=witness_log_uncertainty,
            witness_mechanical=witness_mechanical,
            input_ids=input_ids,
            labels=labels,
            sequence_position_offset=sequence_position_offset,
            witness_valid_mask=witness_valid_mask,
        )
