from __future__ import annotations

"""Zero-train MIRA query construction on an exact mechanical carrier.

The MIRA backbone remains a GSA/skeleton perception-semantic model. This module
constructs a new query domain for the existing V6 readout:

- query XYZ/normals come from the exact Stage19 carrier;
- carrier->GSA semantic memory transport is explicit and support-bound;
- point<->joint geometry is recomputed on carrier vertices;
- no surface skin weights are transferred.

This is a research/compatibility bridge. It does not authorize a new MIRA
architecture or product promotion by itself.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np
import torch

from models.arachne.v2.arachne_geometry_v2 import (
    PAIR_GEOMETRY_CONTRACT_V2,
    arachne_pair_geometry_v2,
)


SCHEMA = "RealSaS.MIRAMechanicalCarrierQuery.v1"


def _hash(payload: Any) -> str:
    def norm(x):
        if isinstance(x, np.ndarray):
            return {
                "dtype": str(x.dtype),
                "shape": list(x.shape),
                "sha256": sha256(x.tobytes(order="C")).hexdigest(),
            }
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (tuple, list)):
            return [norm(v) for v in x]
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        return x
    raw = json.dumps(norm(payload), sort_keys=True, separators=(",", ":")).encode()
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class MIRAMechanicalCarrierQueryV1:
    carrier_vertex_ids: tuple[str, ...]
    positions_normalized: np.ndarray      # [V,3]
    normals: np.ndarray                   # [V,3]
    normal_valid: np.ndarray              # [V]
    geometry7: np.ndarray                 # [V,7]
    pair_geometry: np.ndarray             # [V,J,10]
    legal_pair: np.ndarray                # [V,J]
    support_row_index: np.ndarray         # [K]
    support_source_index: np.ndarray      # [K]
    support_coefficients: np.ndarray      # [K]
    source_surface_ids: tuple[str, ...]
    joint_ids: tuple[str, ...]
    carrier_evidence_hash: str
    carrier_topology_hash: str
    query_normal_authority_hash: str
    query_normal_authority_class: str
    gsa_tensorization_hash: str
    skeleton_lineage_hash: str
    query_hash: str
    schema_version: str = SCHEMA

    @property
    def vertex_count(self) -> int:
        return len(self.carrier_vertex_ids)

    @property
    def joint_count(self) -> int:
        return len(self.joint_ids)


def _support_coefficients(vertex) -> tuple[tuple[str, float], ...]:
    """Mechanical-safe source support for semantic-memory transport.

    Ordinary carrier vertices use exact geometry support. A legacy seam geometry
    vertex may carry a separate mechanically admitted support set; use it when
    present so semantic/mechanical evidence never crosses a declared seam merely
    because the geometry interpolator did.
    """
    binding = vertex.support_binding
    if str(binding.mode) == "SEAM_GEOMETRY_INTERPOLATION":
        md = dict(binding.metadata or {})
        raw = tuple(md.get("skin_support_coefficients") or ())
        if raw:
            rows = tuple((str(sid), float(weight)) for sid, weight in raw)
        else:
            rows = tuple((str(sid), float(weight)) for sid, weight in binding.coefficients)
    else:
        rows = tuple((str(sid), float(weight)) for sid, weight in binding.coefficients)

    if not rows:
        raise ValueError("MIRA_CARRIER_QUERY_SUPPORT_EMPTY")
    seen = set()
    total = 0.0
    out = []
    for sid, weight in rows:
        if sid in seen or not math.isfinite(weight) or weight < 0.0:
            raise ValueError("MIRA_CARRIER_QUERY_SUPPORT_INVALID")
        seen.add(sid)
        total += weight
        out.append((sid, weight))
    if abs(total - 1.0) > 1e-8:
        raise ValueError("MIRA_CARRIER_QUERY_SUPPORT_SIMPLEX_DRIFT")
    return tuple(out)


def build_mira_mechanical_carrier_query_v1(
    *,
    candidate,
    carrier_evidence,
    surface_tensor,
    conditioning,
    signed_query_normal_evidence=None,
    allow_unoriented_carrier_normals_for_diagnostic: bool = False,
) -> MIRAMechanicalCarrierQueryV1:
    if str(candidate.candidate_lineage_hash) != str(carrier_evidence.candidate_mesh_binding_hash):
        raise ValueError("MIRA_CARRIER_QUERY_CANDIDATE_BINDING_DRIFT")
    if tuple(conditioning.surface_ids[0]) != tuple(surface_tensor.surface_ids):
        raise ValueError("MIRA_CARRIER_QUERY_SURFACE_ORDER_DRIFT")
    if str(conditioning.surface_tensorization_hashes[0]) != str(surface_tensor.tensorization_hash):
        raise ValueError("MIRA_CARRIER_QUERY_TENSORIZATION_DRIFT")

    ordered = tuple(sorted(candidate.vertices, key=lambda row: str(row.candidate_vertex_id)))
    vertex_ids = tuple(str(row.candidate_vertex_id) for row in ordered)
    if vertex_ids != tuple(carrier_evidence.ordered_vertex_ids):
        raise ValueError("MIRA_CARRIER_QUERY_VERTEX_ORDER_DRIFT")

    positions_world = np.asarray(carrier_evidence.positions, np.float64)
    carrier_meta = dict(getattr(carrier_evidence, "metadata", {}) or {})
    learned_use_forbidden = bool(
        carrier_meta.get("learned_query_normal_use_forbidden", False)
    )

    if signed_query_normal_evidence is not None:
        evidence = signed_query_normal_evidence
        if str(evidence.candidate_mesh_binding_hash) != str(candidate.candidate_lineage_hash):
            raise ValueError("MIRA_CARRIER_QUERY_NORMAL_CANDIDATE_BINDING_DRIFT")
        if (
            str(evidence.mechanical_carrier_evidence_binding_hash)
            != str(carrier_evidence.carrier_evidence_hash)
        ):
            raise ValueError("MIRA_CARRIER_QUERY_NORMAL_CARRIER_BINDING_DRIFT")
        if tuple(evidence.ordered_vertex_ids) != vertex_ids:
            raise ValueError("MIRA_CARRIER_QUERY_NORMAL_VERTEX_ORDER_DRIFT")
        normals = np.asarray(evidence.signed_normals, np.float32)
        normal_valid = np.asarray(evidence.normal_valid, bool)
        query_normal_authority_hash = str(evidence.query_normal_evidence_hash)
        query_normal_authority_class = "CARRIER_BOUND_SIGNED_QUERY_NORMAL_EVIDENCE"
    else:
        if learned_use_forbidden and not allow_unoriented_carrier_normals_for_diagnostic:
            raise ValueError("MIRA_CARRIER_QUERY_SIGNED_NORMAL_EVIDENCE_REQUIRED")
        if learned_use_forbidden:
            query_normal_authority_class = "DIAGNOSTIC_UNORIENTED_CARRIER_NORMALS"
        else:
            query_normal_authority_class = "LEGACY_OR_DIAGNOSTIC_CARRIER_NORMALS"
        normals = np.asarray(carrier_evidence.normals, np.float32)
        normal_valid = np.asarray(carrier_evidence.normal_valid, bool)
        query_normal_authority_hash = _hash(
            {
                "schema": "RealSaS.MIRAQueryNormalDiagnosticAuthority.v1",
                "carrier_evidence_hash": str(carrier_evidence.carrier_evidence_hash),
                "authority_class": query_normal_authority_class,
                "normals": normals,
                "normal_valid": normal_valid.astype(np.uint8),
            }
        )

    if positions_world.shape != (len(ordered), 3) or normals.shape != positions_world.shape:
        raise ValueError("MIRA_CARRIER_QUERY_GEOMETRY_SHAPE")
    center = np.asarray(surface_tensor.normalization_center, np.float64)
    scale = float(surface_tensor.normalization_scale)
    if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("MIRA_CARRIER_QUERY_NORMALIZATION_INVALID")
    positions_normalized = ((positions_world - center[None]) / scale).astype(np.float32)

    source_ids = tuple(surface_tensor.surface_ids)
    source_index = {str(sid): i for i, sid in enumerate(source_ids)}
    rows = []
    cols = []
    coeffs = []
    for qi, vertex in enumerate(ordered):
        for sid, weight in _support_coefficients(vertex):
            if sid not in source_index:
                raise ValueError("MIRA_CARRIER_QUERY_SUPPORT_OUTSIDE_GSA")
            rows.append(qi)
            cols.append(source_index[sid])
            coeffs.append(float(weight))

    support_row = np.asarray(rows, np.int64)
    support_source = np.asarray(cols, np.int64)
    support_coeff = np.asarray(coeffs, np.float32)
    if len(support_row) == 0:
        raise ValueError("MIRA_CARRIER_QUERY_NO_SUPPORT")

    joint_positions = np.asarray(conditioning.joint_positions_normalized[:, : len(conditioning.joint_ids[0])], np.float32)
    parent_indices = np.asarray(conditioning.parent_indices[:, : len(conditioning.joint_ids[0])], np.int64)
    joint_mask = np.ones((1, len(conditioning.joint_ids[0])), bool)
    query_mask = np.ones((1, len(ordered)), bool)
    pair, legal = arachne_pair_geometry_v2(
        positions_normalized[None],
        normals[None],
        normal_valid[None],
        joint_positions,
        parent_indices,
        query_mask,
        joint_mask,
    )
    geometry7 = np.concatenate(
        [
            2.0 * positions_normalized,
            normals,
            normal_valid[:, None].astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    payload = {
        "schema": SCHEMA,
        "carrier_vertex_ids": vertex_ids,
        "positions_normalized": positions_normalized,
        "normals": normals,
        "normal_valid": normal_valid.astype(np.uint8),
        "geometry7": geometry7,
        "pair_geometry": pair[0],
        "legal_pair": legal[0].astype(np.uint8),
        "support_row_index": support_row,
        "support_source_index": support_source,
        "support_coefficients": support_coeff,
        "source_surface_ids": source_ids,
        "joint_ids": tuple(conditioning.joint_ids[0]),
        "carrier_evidence_hash": str(carrier_evidence.carrier_evidence_hash),
        "carrier_topology_hash": str(carrier_evidence.topology_hash),
        "query_normal_authority_hash": query_normal_authority_hash,
        "query_normal_authority_class": query_normal_authority_class,
        "gsa_tensorization_hash": str(surface_tensor.tensorization_hash),
        "skeleton_lineage_hash": str(conditioning.source_skeleton_hashes[0]),
        "pair_geometry_contract": PAIR_GEOMETRY_CONTRACT_V2,
    }
    query_hash = _hash(payload)
    return MIRAMechanicalCarrierQueryV1(
        carrier_vertex_ids=vertex_ids,
        positions_normalized=positions_normalized,
        normals=normals,
        normal_valid=normal_valid,
        geometry7=geometry7,
        pair_geometry=pair[0],
        legal_pair=legal[0],
        support_row_index=support_row,
        support_source_index=support_source,
        support_coefficients=support_coeff,
        source_surface_ids=source_ids,
        joint_ids=tuple(conditioning.joint_ids[0]),
        carrier_evidence_hash=str(carrier_evidence.carrier_evidence_hash),
        carrier_topology_hash=str(carrier_evidence.topology_hash),
        query_normal_authority_hash=query_normal_authority_hash,
        query_normal_authority_class=query_normal_authority_class,
        gsa_tensorization_hash=str(surface_tensor.tensorization_hash),
        skeleton_lineage_hash=str(conditioning.source_skeleton_hashes[0]),
        query_hash=query_hash,
    )


def transport_surface_memory_to_carrier_v1(
    surface_memory: torch.Tensor,
    query: MIRAMechanicalCarrierQueryV1,
) -> torch.Tensor:
    """Convexly transport learned GSA semantic memory to exact carrier vertices."""
    if surface_memory.ndim != 2:
        raise ValueError("MIRA_CARRIER_MEMORY_SHAPE")
    if surface_memory.shape[0] != len(query.source_surface_ids):
        raise ValueError("MIRA_CARRIER_MEMORY_SOURCE_CARDINALITY_DRIFT")

    device = surface_memory.device
    row = torch.as_tensor(query.support_row_index, dtype=torch.long, device=device)
    col = torch.as_tensor(query.support_source_index, dtype=torch.long, device=device)
    coeff = torch.as_tensor(
        query.support_coefficients,
        dtype=surface_memory.dtype,
        device=device,
    )
    if row.numel() != col.numel() or row.numel() != coeff.numel():
        raise ValueError("MIRA_CARRIER_MEMORY_SUPPORT_SHAPE")

    out = torch.zeros(
        (query.vertex_count, surface_memory.shape[1]),
        dtype=surface_memory.dtype,
        device=device,
    )
    out.index_add_(0, row, surface_memory[col] * coeff[:, None])
    mass = torch.zeros(
        (query.vertex_count,),
        dtype=surface_memory.dtype,
        device=device,
    )
    mass.index_add_(0, row, coeff)
    if not torch.allclose(
        mass,
        torch.ones_like(mass),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("MIRA_CARRIER_MEMORY_SUPPORT_MASS_DRIFT")
    return out


__all__ = [
    "MIRAMechanicalCarrierQueryV1",
    "build_mira_mechanical_carrier_query_v1",
    "transport_surface_memory_to_carrier_v1",
]
