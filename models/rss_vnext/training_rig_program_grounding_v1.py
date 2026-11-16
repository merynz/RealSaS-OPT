from __future__ import annotations

"""Source-qualified, semantic-class-free AXIS/MIRA RigProgram mathematics.

The module is intentionally opt-in. It never creates CONTROL, CONSTRAINT or
ANCHOR truth from a skeleton, invents material coefficients, or turns missing
observations into negative labels. Caller supplies a trusted foundry receipt
bound to the source tensors and *exact* teacher TESSA / AXIS joint addresses.

TailorRig's chain-loss insight is adapted independently: MIRA already predicts
per-bone FSQ-decoded weights; the chain objective supervises their aggregation.
This is NOT a claim to implement TailorRig's separate chain-first decoder.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from models.axis.rig_program_model_v6 import AXISSequenceV6, RigTokenKindV6


GROUNDING_SCHEMA_V1 = "RealSaS.RigProgram.SourceGrounding.v1"


@dataclass(frozen=True)
class RigProgramGroundingWeightsV1:
    joint_position: float = 1.0
    bone_projection: float = 0.5
    chain_skin: float = 1.0
    bone_skin: float = 1.0
    frame_orientation: float = 0.25
    observed_dof: float = 0.25
    control_rest: float = 0.25
    attachment_anchor: float = 0.25
    posed_lbs: float = 1.0

    def validate(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("RIG_MATH_WEIGHT_INVALID:" + name)
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError("RIG_MATH_WEIGHT_INVALID:" + name)


def maximal_source_bone_chains_v1(parents: Tensor) -> tuple[tuple[int, ...], ...]:
    """Partition incoming bones into maximal unbranched chains.

    Every non-root joint represents its incoming bone. Roots form separate
    singleton chains, preserving source skin mass assigned directly to roots.
    The partition is exhaustive and disjoint even for multi-root rigs.
    No anatomical names, humanoid templates or semantic classifications.
    """
    if parents.ndim != 1 or parents.dtype != torch.long or parents.numel() < 1:
        raise ValueError("RIG_MATH_PARENT_DOMAIN_INVALID")
    p = [int(x) for x in parents.detach().cpu().tolist()]
    n = len(p)
    if any(v < -1 or v >= n or v == i for i, v in enumerate(p)):
        raise ValueError("RIG_MATH_PARENT_ADDRESS_INVALID")
    children = [[] for _ in range(n)]
    for j, parent in enumerate(p):
        if parent >= 0:
            children[parent].append(j)
    for j in range(n):
        seen = set()
        k = j
        while k != -1:
            if k in seen:
                raise ValueError("RIG_MATH_PARENT_CYCLE")
            seen.add(k)
            k = p[k]
    if not any(x == -1 for x in p):
        raise ValueError("RIG_MATH_NO_ROOT")

    chains: list[tuple[int, ...]] = []
    for j in range(n):
        if p[j] == -1:
            chains.append((j,))
            continue
        parent = p[j]
        if p[parent] != -1 and len(children[parent]) == 1:
            continue
        chain = [j]
        cursor = j
        while len(children[cursor]) == 1:
            cursor = children[cursor][0]
            chain.append(cursor)
        chains.append(tuple(chain))
    covered = [j for chain in chains for j in chain]
    if len(covered) != n or set(covered) != set(range(n)):
        raise ValueError("RIG_MATH_CHAIN_PARTITION_INVALID")
    return tuple(chains)


def _digest_tensor(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    if tensor.requires_grad:
        raise ValueError("RIG_MATH_SOURCE_GRADIENT_FORBIDDEN")
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return sha256(header + b"\0" + tensor.numpy().tobytes()).hexdigest()


def source_rig_grounding_digest_v1(**fields: Tensor | None) -> str:
    """Hash ALL available typed source observations, including missing fields."""
    return sha256(json.dumps(
        {key: None if value is None else _digest_tensor(value)
         for key, value in sorted(fields.items())},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _expected_xyz(logits: Tensor, lo: float, hi: float) -> Tensor:
    # Differentiable expectation of the *same* discrete bins as AXIS V6.
    centers = lo + (torch.arange(
        logits.shape[-1], dtype=logits.dtype, device=logits.device
    ) + 0.5) * ((hi - lo) / logits.shape[-1])
    return (torch.softmax(logits, dim=-1) * centers).sum(dim=-1)


def _pred_rot6d_rows(value: Tensor) -> Tensor:
    x = F.normalize(value[..., :3], dim=-1, eps=1e-8)
    y0 = value[..., 3:] - (value[..., 3:] * x).sum(-1, keepdim=True) * x
    y = F.normalize(y0, dim=-1, eps=1e-8)
    z = torch.linalg.cross(x, y)
    return torch.stack((x, y, z), dim=-2)


def _source_data_validate(
    source: Mapping[str, Tensor | None],
    *, batch_size: int, vertex_count: int, joint_count: int,
    device: torch.device,
) -> None:
    xyz, parent, weight = (
        source["joint_xyz"], source["parents"], source["skin_weights"],
    )
    assert isinstance(xyz, Tensor) and isinstance(parent, Tensor) and isinstance(weight, Tensor)
    if (xyz.shape != (batch_size, joint_count, 3)
        or parent.shape != (joint_count,)
        or weight.shape != (batch_size, vertex_count, joint_count)
        or parent.dtype != torch.long):
        raise ValueError("RIG_MATH_SOURCE_SHAPE_INVALID")
    if any(
        tensor.device != device or tensor.requires_grad
        for tensor in source.values() if tensor is not None
    ):
        raise ValueError("RIG_MATH_SOURCE_DEVICE_OR_GRAD_INVALID")
    if not (
        torch.isfinite(xyz).all()
        and torch.isfinite(weight).all()
        and bool((weight >= 0).all())
        and torch.allclose(
            weight.sum(-1), torch.ones_like(weight.sum(-1)),
            atol=1e-5, rtol=0,
        )
    ):
        raise ValueError("RIG_MATH_SOURCE_NUMERIC_OR_SKIN_INVALID")
    for key in ("joint_xyz", "skin_weights", "frame_basis",
                "control_rest_xyz", "anchor_local_xyz",
                "pose_matrices", "posed_vertices"):
        tensor = source.get(key)
        if tensor is not None and not bool(torch.isfinite(tensor).all()):
            raise ValueError("RIG_MATH_SOURCE_NONFINITE:" + key)
    for key in ("frame_basis", "dof_known_mask", "dof_allowed"):
        tensor = source.get(key)
        if tensor is not None:
            expected = ((batch_size, joint_count, 3, 3) if key == "frame_basis"
                        else (batch_size, joint_count, 9))
            if tensor.shape != expected:
                raise ValueError("RIG_MATH_SOURCE_FRAME_OR_DOF_SHAPE")


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        raise ValueError("RIG_MATH_NO_QUALIFIED_LABELS")
    return value[mask].mean()


def qualified_rig_program_grounding_objective_v1(
    *,
    teacher: AXISSequenceV6,
    outputs: Mapping[str, Tensor],
    predicted_skin: Tensor,
    source_joint_xyz: Tensor,
    source_parent_indices: Tensor,
    source_skin_weights: Tensor,
    source_frame_basis: Tensor | None = None,
    source_dof_known_mask: Tensor | None = None,
    source_dof_allowed: Tensor | None = None,
    source_control_rest_xyz: Tensor | None = None,
    source_anchor_local_xyz: Tensor | None = None,
    source_rest_vertices: Tensor | None = None,
    source_pose_matrices: Tensor | None = None,
    source_posed_vertices: Tensor | None = None,
    source_pose_mask: Tensor | None = None,
    trusted_receipt: Mapping[str, object] | None,
    expected_dataset_manifest_sha256: str,
    expected_source_qualification_key: str,
    expected_axis_teacher_sha256: str,
    expected_mesh_teacher_topology_digest: str,
    weights: RigProgramGroundingWeightsV1 = RigProgramGroundingWeightsV1(),
    coord_min: float = -1.0,
    coord_max: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Source rig geometry + source skin + optionally observed DOF/controls/poses.

    CONTROL/CONSTRAINT tokens remain supervised by the existing AXIS causal
    token objective, which requires qualified source intervention truth.
    This function adds geometric/kinematic *consequences* to that objective,
    and cannot itself invent a controller or a differentiable IK solver.
    """
    weights.validate()
    if (predicted_skin.ndim != 3 or predicted_skin.shape[0] != 1
        or not torch.isfinite(predicted_skin).all()
        or not torch.isfinite(source_joint_xyz).all()):
        raise ValueError("RIG_MATH_PREDICTION_INVALID")
    b, v, j = predicted_skin.shape
    if (teacher.kind_ids.ndim != 2 or teacher.kind_ids.shape[0] != b
        or not math.isfinite(coord_min) or not math.isfinite(coord_max)
        or coord_max <= coord_min):
        raise ValueError("RIG_MATH_TEACHER_OR_COORD_DOMAIN_INVALID")
    source: dict[str, Tensor | None] = {
        "joint_xyz": source_joint_xyz, "parents": source_parent_indices,
        "skin_weights": source_skin_weights, "frame_basis": source_frame_basis,
        "dof_known_mask": source_dof_known_mask,
        "dof_allowed": source_dof_allowed,
        "control_rest_xyz": source_control_rest_xyz,
        "anchor_local_xyz": source_anchor_local_xyz,
        "rest_vertices": source_rest_vertices,
        "pose_matrices": source_pose_matrices,
        "posed_vertices": source_posed_vertices,
        "pose_mask": source_pose_mask,
    }
    _source_data_validate(source, batch_size=b, vertex_count=v,
                          joint_count=j, device=predicted_skin.device)
    digest = source_rig_grounding_digest_v1(**source)
    if (trusted_receipt is None
        or trusted_receipt.get("admission_status") != "QUALIFIED_SOURCE_RIG_PROGRAM_MATH_V1"
        or trusted_receipt.get("schema") != GROUNDING_SCHEMA_V1
        or trusted_receipt.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256
        or trusted_receipt.get("source_qualification_key") != expected_source_qualification_key
        or trusted_receipt.get("axis_teacher_sequence_sha256") != expected_axis_teacher_sha256
        or trusted_receipt.get("mesh_teacher_topology_digest") != expected_mesh_teacher_topology_digest
        or trusted_receipt.get("source_tensors_sha256") != digest
        or trusted_receipt.get("joint_index_and_mesh_vertex_reindex_proven") is not True
        or trusted_receipt.get("source_semantics") != "NATIVE_RIG_SKIN_NOT_PSEUDOLABEL"):
        raise ValueError("RIG_MATH_SOURCE_RECEIPT_UNQUALIFIED")
    # Static rig names/positions are never evidence that a joint axis is
    # locked, or that a user control actually has a behavioral consequence.
    if source_dof_known_mask is not None and bool(source_dof_known_mask.any()):
        if (
            trusted_receipt.get("dof_truth_authority")
            not in ("SOURCE_AUTHORED_LIMITS", "INDEPENDENT_SOURCE_INTERVENTION")
            or not isinstance(trusted_receipt.get("dof_truth_witness_sha256"), str)
            or len(trusted_receipt["dof_truth_witness_sha256"]) != 64
        ):
            raise ValueError("RIG_MATH_DOF_WITNESS_REQUIRED")
    if source_control_rest_xyz is not None and (
        trusted_receipt.get("control_rest_authority") != "NATIVE_AUTHORED_CONTROL"
    ):
        raise ValueError("RIG_MATH_SOURCE_CONTROL_AUTHORING_REQUIRED")
    if source_anchor_local_xyz is not None and (
        trusted_receipt.get("anchor_local_authority") != "SOURCE_NATIVE_ANCHOR"
    ):
        raise ValueError("RIG_MATH_SOURCE_ANCHOR_AUTHORING_REQUIRED")
    if source_pose_matrices is not None and (
        trusted_receipt.get("pose_truth_authority") != "INDEPENDENT_SOURCE_LBS_POSES"
    ):
        raise ValueError("RIG_MATH_SOURCE_POSE_WITNESS_REQUIRED")
    chains = maximal_source_bone_chains_v1(source_parent_indices)
    membership = predicted_skin.new_zeros((len(chains), j))
    for ci, chain in enumerate(chains):
        membership[ci, list(chain)] = 1.0

    if (not bool(torch.isfinite(predicted_skin).all())
        or bool((predicted_skin < 0).any())
        or not bool(torch.allclose(
            predicted_skin.sum(-1), torch.ones_like(predicted_skin.sum(-1)),
            atol=1e-4, rtol=0,
        ))):
        raise ValueError("RIG_MATH_PREDICTED_SKIN_SIMPLEX_INVALID")
    # Forward uses the hard-token MIRA skin distribution. Chain labels are
    # sums of admitted source per-bone skin; no object categories are needed.
    chain_pred = torch.einsum("bvj,kj->bvk", predicted_skin, membership)
    chain_gt = torch.einsum("bvj,kj->bvk", source_skin_weights, membership)
    eps = 1e-8
    def kl(gt: Tensor, pred: Tensor) -> Tensor:
        return (gt * (gt.clamp_min(eps).log() - pred.clamp_min(eps).log())).sum(-1).mean()
    parts: dict[str, Tensor] = {
        "chain_skin": kl(chain_gt, chain_pred),
        "bone_skin": kl(source_skin_weights, predicted_skin),
    }

    # The teacher event at position t is predicted from logits[t - 1].
    events = teacher.kind_ids[:, 1:]
    coord_logits = outputs["coord_logits"][:, :-1]
    if coord_logits.shape[:3] != (*events.shape, 3):
        raise ValueError("RIG_MATH_COORD_LOGIT_ALIGNMENT")
    xyz_pred = _expected_xyz(coord_logits, coord_min, coord_max)
    # Runtime inference uses discrete argmax bins. This expectation is an
    # additional smooth training signal, not an inference-time replacement.
    joints = torch.where(events[0] == int(RigTokenKindV6.JOINT))[0]
    if len(joints) != j:
        raise ValueError("RIG_MATH_JOINT_TEACHER_REINDEX_DRIFT")
    predicted_joint_xyz = xyz_pred[:, joints]
    parts["joint_position"] = F.smooth_l1_loss(predicted_joint_xyz, source_joint_xyz)

    nonroot = source_parent_indices >= 0
    if bool(nonroot.any()):
        parent = source_parent_indices[nonroot]
        start = source_joint_xyz[:, parent, :]
        end = source_joint_xyz[:, nonroot, :]
        segment = end - start
        denom = (segment * segment).sum(-1, keepdim=True)
        if bool((denom < 1e-12).any()):
            raise ValueError("RIG_MATH_DEGENERATE_SOURCE_BONE")
        joint = predicted_joint_xyz[:, nonroot, :]
        t = (((joint - start) * segment).sum(-1, keepdim=True) / denom).clamp(0, 1)
        close = start + t * segment
        parts["bone_projection"] = (joint - close).square().sum(-1).mean()

    if source_frame_basis is not None:
        frame_idx = torch.where(events[0] == int(RigTokenKindV6.FRAME))[0]
        frame_ptr = teacher.pointer_a[0, 1:][frame_idx]
        if (len(frame_idx) != j or sorted(frame_ptr.tolist()) != list(range(j))):
            raise ValueError("RIG_MATH_FRAME_POINTER_COVERAGE")
        predicted_frames = _pred_rot6d_rows(outputs["frame_6d"][:, :-1][:, frame_idx])
        true_frames = source_frame_basis[:, frame_ptr]
        # Basis rows are orthonormal vectors; cosine loss has finite gradient.
        gram = torch.matmul(true_frames, true_frames.transpose(-1, -2))
        eye = torch.eye(3, device=gram.device, dtype=gram.dtype)
        if not bool(torch.allclose(gram, eye, atol=2e-3, rtol=0)):
            raise ValueError("RIG_MATH_GT_FRAME_NOT_ORTHONORMAL")
        parts["frame_orientation"] = (1 - (predicted_frames * true_frames).sum(-1)).mean()
        if (source_dof_known_mask is None) != (source_dof_allowed is None):
            raise ValueError("RIG_MATH_DOF_PARTIAL_TEACHER")
        if source_dof_known_mask is not None and source_dof_allowed is not None:
            if source_dof_known_mask.dtype != torch.bool:
                raise ValueError("RIG_MATH_DOF_KNOWN_MASK_INVALID")
            if bool(source_dof_known_mask.any()):
                truth = source_dof_allowed[:, frame_ptr].float()
                known = source_dof_known_mask[:, frame_ptr]
                if bool((((truth != 0) & (truth != 1)) & known).any()):
                    raise ValueError("RIG_MATH_DOF_NOT_BINARY")
                logits = outputs["dof_logits"][:, :-1][:, frame_idx]
                parts["observed_dof"] = _masked_mean(
                    F.binary_cross_entropy_with_logits(logits, truth.clamp(0, 1), reduction="none"),
                    known,
                )
    elif source_dof_known_mask is not None or source_dof_allowed is not None:
        raise ValueError("RIG_MATH_DOF_WITHOUT_SOURCE_FRAME")

    for kind, key, target, loss_name in (
        (RigTokenKindV6.CONTROL, "control_rest", source_control_rest_xyz, "control_rest"),
        (RigTokenKindV6.ANCHOR, "anchor_local", source_anchor_local_xyz, "attachment_anchor"),
    ):
        idx = torch.where(events[0] == int(kind))[0]
        if target is not None:
            if target.shape != (b, len(idx), 3):
                raise ValueError("RIG_MATH_SOURCE_CONTROL_ANCHOR_REINDEX:" + key)
            if len(idx):
                parts[loss_name] = F.smooth_l1_loss(xyz_pred[:, idx], target)
        elif len(idx) and trusted_receipt.get(key + "_source_required") is True:
            raise ValueError("RIG_MATH_REQUIRED_CONTROL_ANCHOR_TRUTH_MISSING:" + key)

    if any(x is not None for x in (
        source_rest_vertices, source_pose_matrices,
        source_posed_vertices, source_pose_mask,
    )):
        if any(x is None for x in (
            source_rest_vertices, source_pose_matrices,
            source_posed_vertices, source_pose_mask,
        )):
            raise ValueError("RIG_MATH_POSED_SOURCE_INCOMPLETE")
        assert source_rest_vertices is not None
        assert source_pose_matrices is not None
        assert source_posed_vertices is not None
        assert source_pose_mask is not None
        if (source_rest_vertices.shape != (b, v, 3)
            or source_pose_matrices.ndim != 5
            or source_pose_matrices.shape[0] != b
            or source_pose_matrices.shape[2:] != (j, 4, 4)
            or source_posed_vertices.shape != (
                b, source_pose_matrices.shape[1], v, 3,
            )
            or source_pose_mask.shape != source_pose_matrices.shape[:2]
            or source_pose_mask.dtype != torch.bool
            or not bool(source_pose_mask.any())):
            raise ValueError("RIG_MATH_POSED_DOMAIN_INVALID")
        matrices = source_pose_matrices
        last_row = matrices[..., 3, :]
        identity_last = matrices.new_tensor((0., 0., 0., 1.))
        if not bool(torch.allclose(last_row, identity_last, atol=1e-5, rtol=0)):
            raise ValueError("RIG_MATH_SOURCE_LBS_MATRIX_INVALID")
        # A global rigid move alone cannot teach skinning. Require at least
        # one qualified pose with genuinely different joint transports.
        delta = matrices - matrices[:, :, :1]
        excited = delta.abs().amax(dim=(-1, -2, -3)) > 1e-5
        if not bool((excited & source_pose_mask).any()):
            raise ValueError("RIG_MATH_UNINFORMATIVE_RIGID_ONLY_POSES")
        hom = torch.cat((source_rest_vertices, source_rest_vertices.new_ones((b, v, 1))), -1)
        transported = torch.einsum("bpjik,bvk->bpvji", matrices, hom)[..., :3]
        posed_pred = (
            transported * predicted_skin[:, None, :, :, None]
        ).sum(-2)
        parts["posed_lbs"] = _masked_mean(
            (posed_pred - source_posed_vertices).abs().mean(dim=(-2, -1)),
            source_pose_mask,
        )

    total = sum(getattr(weights, k) * x for k, x in parts.items())
    if not bool(torch.isfinite(total)):
        raise ValueError("RIG_MATH_NONFINITE_OBJECTIVE")
    return total, parts


__all__ = [
    "GROUNDING_SCHEMA_V1", "RigProgramGroundingWeightsV1",
    "maximal_source_bone_chains_v1", "source_rig_grounding_digest_v1",
    "qualified_rig_program_grounding_objective_v1",
]
