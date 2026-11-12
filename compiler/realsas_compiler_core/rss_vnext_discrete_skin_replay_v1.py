from __future__ import annotations

"""Executable product Compiler audit: hard FSQ skin replay, without promotion.

SkinTokens motivates keeping *discrete* code identity plus decoder weights.
AniGen motivates the same mesh/joint spatial domain. The Compiler independently
replays the candidate dense weights and validates their declared discrepancy
without hallucinating an executable CanonicalRig, a motion pose, or a drawable.

All inputs are sourced from source/checkpoint-sealed CAS children by the caller;
this function never replaces source evidence or performs model inference.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import _dynamic
from compiler.realsas_compiler_core.types import QualificationError
from models.axis.rig_program_model_v6 import RigTokenKindV6
from models.rss_vnext.contracts import RSSvNextConfig
from models.rss_vnext.unified_character_model import RSSvNextSkinCoupler

SCHEMA = "RealSaS.RSSvNext.DiscreteSkinReplayReceipt.v1"
_MAX_REPLAY_WEIGHTS = 2_000_000
_REPLAY_ATOL = 5.e-5


@dataclass(frozen=True)
class RSSvNextDiscreteSkinReplayReceiptV1:
    checkpoint_sha256: str
    inference_evidence_hash: str
    embedded_decoder_sha256: str
    candidate_mesh_sha256: str
    vertex_count: int
    joint_count: int
    skin_code_count: int
    dense_weight_max_abs_replay_error: float
    soft_to_discrete_max_abs_difference: float
    soft_to_discrete_mean_abs_difference: float
    state: str = "REPLAYED_UNQUALIFIED_DISCRETE_SKIN_CANDIDATE"
    schema: str = SCHEMA
    canonical_mesh_authority: bool = False
    canonical_rig_authority: bool = False
    canonical_skin_authority: bool = False
    deform_motion_parity_qualified: bool = False


@torch.no_grad()
def replay_rss_vnext_discrete_skin_candidate_v1(
    *, mesh: Mapping[str, Any],
    dynamic: Mapping[str, Any],
    control_events: Sequence[Mapping[str, Any]],
    model_config: RSSvNextConfig,
    decoder: RSSvNextSkinCoupler,
    checkpoint_sha256: str,
    inference_evidence_hash: str,
    embedded_decoder_sha256: str,
) -> RSSvNextDiscreteSkinReplayReceiptV1:
    """Recompute stored hard weights from actual AXIS token IDs and joint H.

    A successful equality proves producer/consumer replay only; it does NOT
    prove skeleton geometry, skin plausibility, IK, or 2D composition.
    """
    for field in (checkpoint_sha256, inference_evidence_hash, embedded_decoder_sha256):
        if (
            type(field) is not str or len(field) != 64
            or any(ch not in "0123456789abcdef" for ch in field)
        ):
            raise QualificationError("RSS_VNEXT_SKIN_REPLAY_LINEAGE_REQUIRED")
    if (
        type(model_config) is not RSSvNextConfig
        or type(decoder) is not RSSvNextSkinCoupler
        or decoder.cfg != model_config
        or decoder.training
        or any(p.requires_grad for p in decoder.parameters())
    ):
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_PINNED_DECODER_REQUIRED")
    if not isinstance(mesh, Mapping) or not isinstance(dynamic, Mapping):
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_BOUND_MESH_AND_SKIN_REQUIRED")
    for field in ("local_features", "rest_xyz", "normals_xyz"):
        if field not in mesh:
            raise QualificationError("RSS_VNEXT_SKIN_REPLAY_VERTEX_FIELDS_MISSING")
    n = len(mesh["rest_xyz"])
    if not 1 <= n <= _MAX_REPLAY_WEIGHTS:
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_VERTEX_BUDGET")
    _dynamic(dynamic, n)
    if "discrete_skin_weights" not in dynamic:
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_SOFT_ONLY_ABSTAIN")
    j = len(dynamic["weights"][0])
    if not 1 <= j <= model_config.control_max_seq or n * j > _MAX_REPLAY_WEIGHTS:
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_WEIGHT_BUDGET")
    if not isinstance(control_events, (tuple, list)):
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_CAUSAL_EVENTS_REQUIRED")
    if sum(e.get("kind") == int(RigTokenKindV6.JOINT) for e in control_events) != j:
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_JOINT_ORDER_DRIFT")
    codes = [
        e.get("skin_code_id") for e in control_events
        if e.get("kind") == int(RigTokenKindV6.SKIN_CODE)
    ]
    if len(codes) != j * model_config.skin_tokens_per_joint:
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_CODE_CARDINALITY_DRIFT")
    if any(type(c) is not int for c in codes):
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_CODE_TYPE_DRIFT")
    t = model_config.skin_tokens_per_joint
    token_ids = torch.tensor(codes, dtype=torch.long).reshape(1, j, t)
    features = torch.as_tensor(np.asarray(mesh["local_features"], dtype=np.float32))[None]
    xyz = torch.as_tensor(np.asarray(mesh["rest_xyz"], dtype=np.float32))[None]
    normals = torch.as_tensor(np.asarray(mesh["normals_xyz"], dtype=np.float32))[None]
    joint_hidden = torch.as_tensor(np.asarray(
        dynamic["skin_joint_hidden_states"], dtype=np.float32,
    ))[None]
    candidate_weights = torch.as_tensor(np.asarray(
        dynamic["discrete_skin_weights"], dtype=np.float32,
    ))[None]
    replay = decoder.decode_discrete_skin_v1(
        features, xyz, joint_hidden, token_ids, vertex_normals=normals,
    )
    diff = (replay - candidate_weights).abs()
    maximum = float(diff.max().item())
    if maximum > _REPLAY_ATOL:
        raise QualificationError("RSS_VNEXT_SKIN_REPLAY_DISCRETE_WEIGHT_MISMATCH")
    return RSSvNextDiscreteSkinReplayReceiptV1(
        checkpoint_sha256=checkpoint_sha256,
        inference_evidence_hash=inference_evidence_hash,
        embedded_decoder_sha256=embedded_decoder_sha256,
        candidate_mesh_sha256=content_sha256(mesh),
        vertex_count=n, joint_count=j, skin_code_count=len(codes),
        dense_weight_max_abs_replay_error=maximum,
        soft_to_discrete_max_abs_difference=float(
            dynamic["soft_to_discrete_max_abs_difference"],
        ),
        soft_to_discrete_mean_abs_difference=float(
            dynamic["soft_to_discrete_mean_abs_difference"],
        ),
    )


__all__ = [
    "SCHEMA", "RSSvNextDiscreteSkinReplayReceiptV1",
    "replay_rss_vnext_discrete_skin_candidate_v1",
]
