from __future__ import annotations

"""Deterministic CanonicalMesh + CanonicalRig.v2 conditioning for MIRA V6.

No learned geometry encoder lives here. The codec receives exact product geometry,
local mechanical frames and DOF state; learned skin information is carried only by
the AXIS-generated discrete skin codes.
"""

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import torch
from torch import Tensor

from compiler.realsas_compiler_core.carrier_skin_v1 import (
    CarrierSkinInfluenceProposal,
    CarrierSkinProposalIR,
)

MIRA_V6_VERTEX_FEATURE_DIM = 7
MIRA_V6_JOINT_FEATURE_DIM = 25
MIRA_V6_CONDITIONING_ID = "RealSaS.MIRA.CanonicalMeshRigConditioning.v6_0"


@dataclass(frozen=True)
class MIRACarrierViewV2:
    ordered_vertex_ids: tuple[str, ...]
    carrier_evidence_hash: str
    topology_hash: str
    geometry_hash: str


@dataclass(frozen=True)
class _SkeletonJointViewV2:
    canonical_joint_id: str


@dataclass(frozen=True)
class MIRADeformSkeletonViewV2:
    joints: tuple[_SkeletonJointViewV2, ...]
    skeleton_lineage_hash: str


@dataclass(frozen=True)
class MIRAConditioningV6:
    vertex_features: Tensor
    joint_features: Tensor
    carrier: MIRACarrierViewV2
    skeleton: MIRADeformSkeletonViewV2
    joint_ids: tuple[str, ...]
    conditioning_id: str = MIRA_V6_CONDITIONING_ID


def _mesh_arrays(mesh):
    vertices = np.asarray([v.P for v in mesh.vertices], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise RuntimeError("MIRA_V6_MESH_VERTEX_DOMAIN_INVALID")
    index = {str(v.canonical_mesh_vertex_id): i for i, v in enumerate(mesh.vertices)}
    if len(index) != len(mesh.vertices):
        raise RuntimeError("MIRA_V6_MESH_VERTEX_ID_DUPLICATE")
    try:
        faces = np.asarray([[index[str(x)] for x in face] for face in mesh.faces], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError("MIRA_V6_MESH_FACE_VERTEX_UNKNOWN") from exc
    if faces.size == 0:
        faces = np.empty((0, 3), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("MIRA_V6_MESH_FACE_DOMAIN_INVALID")
    return vertices, faces


def _normalization(vertices: np.ndarray):
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) * 0.5
    scale = float(np.max(hi - lo))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("MIRA_V6_MESH_NORMALIZATION_DEGENERATE")
    return center, scale


def _normals(vertices: np.ndarray, faces: np.ndarray):
    out = np.zeros_like(vertices)
    if len(faces):
        fn = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        for c in range(3):
            np.add.at(out, faces[:, c], fn)
    mag = np.linalg.norm(out, axis=1)
    valid = mag > 1e-12
    out[valid] /= mag[valid, None]
    out[~valid] = 0.0
    return out.astype(np.float32), valid.astype(np.float32)


def build_mira_conditioning_v6(mesh, rig, *, device: torch.device | str) -> MIRAConditioningV6:
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise RuntimeError("MIRA_V6_RIG_MESH_PARENT_DRIFT")
    vertices, faces = _mesh_arrays(mesh)
    center, scale = _normalization(vertices)
    p = ((vertices - center[None, :]) / scale).astype(np.float32)
    n, valid = _normals(vertices, faces)
    vertex_features_np = np.concatenate([p, n, valid[:, None]], axis=1)
    if vertex_features_np.shape[1] != MIRA_V6_VERTEX_FEATURE_DIM:
        raise RuntimeError("MIRA_V6_VERTEX_FEATURE_WIDTH_DRIFT")

    frame_by = {str(f.canonical_joint_id): f for f in rig.local_frames}
    joint_by = {str(j.canonical_joint_id): j for j in rig.joints}
    if len(frame_by) != len(rig.local_frames) or set(frame_by) != set(joint_by):
        raise RuntimeError("MIRA_V6_RIG_FRAME_COVERAGE_DRIFT")
    joint_ids = tuple(str(j.canonical_joint_id) for j in rig.joints)
    dof_vocab = ("TX", "TY", "TZ", "RX", "RY", "RZ", "SX", "SY", "SZ")
    rows = []
    for j in rig.joints:
        jid = str(j.canonical_joint_id)
        pos = (np.asarray(j.P, dtype=np.float64) - center) / scale
        parent = j.parent_canonical_joint_id
        if parent is None:
            delta = np.zeros(3, dtype=np.float64)
            root = 1.0
        else:
            pj = joint_by.get(str(parent))
            if pj is None:
                raise RuntimeError("MIRA_V6_RIG_PARENT_UNKNOWN")
            delta = (np.asarray(j.P, dtype=np.float64) - np.asarray(pj.P, dtype=np.float64)) / scale
            root = 0.0
        frame = frame_by[jid]
        basis = np.asarray(frame.basis, dtype=np.float64).reshape(-1)
        if basis.shape != (9,) or not np.isfinite(basis).all():
            raise RuntimeError("MIRA_V6_FRAME_BASIS_INVALID")
        labels = set(map(str, frame.dof_labels))
        dof = np.asarray([1.0 if name in labels else 0.0 for name in dof_vocab], dtype=np.float64)
        row = np.concatenate([pos, delta, basis, dof, np.asarray([root])], axis=0)
        if row.shape != (MIRA_V6_JOINT_FEATURE_DIM,) or not np.isfinite(row).all():
            raise RuntimeError("MIRA_V6_JOINT_FEATURE_INVALID")
        rows.append(row.astype(np.float32))
    if not rows:
        raise RuntimeError("MIRA_V6_RIG_EMPTY")
    joint_features_np = np.stack(rows, axis=0)

    dev = torch.device(device)
    carrier = MIRACarrierViewV2(
        ordered_vertex_ids=tuple(str(v.source_candidate_vertex_id) for v in mesh.vertices),
        carrier_evidence_hash=str(mesh.carrier_evidence_hash),
        topology_hash=str(mesh.carrier_topology_hash),
        geometry_hash=str(mesh.carrier_geometry_hash),
    )
    skeleton = MIRADeformSkeletonViewV2(
        joints=tuple(_SkeletonJointViewV2(jid) for jid in joint_ids),
        skeleton_lineage_hash=str(rig.deform_structure_hash),
    )
    return MIRAConditioningV6(
        vertex_features=torch.as_tensor(vertex_features_np, device=dev, dtype=torch.float32).unsqueeze(0),
        joint_features=torch.as_tensor(joint_features_np, device=dev, dtype=torch.float32).unsqueeze(0),
        carrier=carrier,
        skeleton=skeleton,
        joint_ids=joint_ids,
    )


def carrier_skin_proposal_from_decoded_v6(
    weights: Tensor,
    conditioning: MIRAConditioningV6,
    *,
    model_provenance: str,
) -> CarrierSkinProposalIR:
    if weights.ndim == 3:
        if weights.shape[0] != 1:
            raise RuntimeError("MIRA_V6_SKIN_BATCH_ONE_REQUIRED")
        weights = weights[0]
    n = len(conditioning.carrier.ordered_vertex_ids)
    j = len(conditioning.joint_ids)
    if tuple(weights.shape) != (n, j):
        raise RuntimeError("MIRA_V6_DECODED_WEIGHT_SHAPE_DRIFT")
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("MIRA_V6_DECODED_WEIGHT_NONFINITE")
    influences = tuple(
        CarrierSkinInfluenceProposal(
            carrier_vertex_id=conditioning.carrier.ordered_vertex_ids[vi],
            canonical_joint_id=conditioning.joint_ids[ji],
            weight=float(weights[vi, ji].item()),
        )
        for vi in range(n)
        for ji in range(j)
    )
    return CarrierSkinProposalIR(
        influences=influences,
        carrier_evidence_hash=conditioning.carrier.carrier_evidence_hash,
        carrier_topology_hash=conditioning.carrier.topology_hash,
        carrier_geometry_hash=conditioning.carrier.geometry_hash,
        skeleton_binding_hash=conditioning.skeleton.skeleton_lineage_hash,
        model_provenance=str(model_provenance),
        metadata={
            "output_domain": "MECHANICAL_CARRIER_M",
            "semantic_skin_transfer_performed": False,
            "gsa_skin_field_authority_minted": False,
            "direct_simplex_readout": True,
            "source_skin_runtime_authority": False,
            "mesh_mutation_invalidates_prediction": True,
            "decoded_from_axis_generated_discrete_skin_codes": True,
        },
    )


__all__ = [
    "MIRAConditioningV6", "MIRACarrierViewV2", "MIRADeformSkeletonViewV2",
    "MIRA_V6_CONDITIONING_ID", "MIRA_V6_JOINT_FEATURE_DIM", "MIRA_V6_VERTEX_FEATURE_DIM",
    "build_mira_conditioning_v6", "carrier_skin_proposal_from_decoded_v6",
]
