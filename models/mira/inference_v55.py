from __future__ import annotations

"""Exact carrier-native MIRA V5.5 inference assembly.

This is the productized form of the sealed V5.5 Run-All predictor path:
RiggingSurface + canonical Rig -> frozen Arachne backbone evidence; TESSA material
support composes that evidence onto the exact canonical Mesh; M owns query
geometry; the direct V5.5 readout predicts one simplex row per Mesh vertex.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MiraCarrierViewV1:
    ordered_vertex_ids: tuple[str, ...]
    carrier_evidence_hash: str
    topology_hash: str
    geometry_hash: str


def _qualified_skeleton_view(rig):
    from compiler.realsas_compiler_core.types import QualifiedJoint, QualifiedSkeletonIR

    if len(rig.root_joint_ids) != 1:
        raise RuntimeError("MIRA_V55_CURRENT_RIG_ROOT_COUNT_DRIFT")
    joints = tuple(
        QualifiedJoint(
            canonical_joint_id=str(joint.canonical_joint_id),
            position=tuple(map(float, joint.P)),
            parent_canonical_id=(
                None
                if joint.parent_canonical_joint_id is None
                else str(joint.parent_canonical_joint_id)
            ),
            support_surface_ids=tuple(map(str, joint.proposal_support_evidence_ids)),
            source_proposal_id=str(joint.source_proposal_id),
        )
        for joint in sorted(rig.joints, key=lambda row: str(row.canonical_joint_id))
    )
    return QualifiedSkeletonIR(
        joints=joints,
        root_id=str(rig.root_joint_ids[0]),
        qualification_report={
            "status": "CANONICAL_RIG_COMPATIBILITY_VIEW_ONLY",
            "canonical_rig_lineage_hash": str(rig.rig_lineage_hash),
        },
        skeleton_lineage_hash=str(rig.structure_qualification_hash),
    )


def _material_support_rows(payload: Mapping[str, Any]):
    from compiler.realsas_compiler_core.tessa_surface_field_v1 import (
        TESSAMaterialSupportFieldIR,
        TESSAMaterialSupportRowIR,
        tessa_material_support_field_lineage_hash,
    )

    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.TESSAMaterialSupportFieldIR.v1":
        raise RuntimeError("MIRA_V55_MATERIAL_SUPPORT_SCHEMA_DRIFT")
    rows = tuple(
        TESSAMaterialSupportRowIR(
            proposal_vertex_id=str(row["proposal_vertex_id"]),
            decoded_component_index=int(row["decoded_component_index"]),
            mechanical_component_id=str(row["mechanical_component_id"]),
            coefficients=tuple(
                (str(surface_id), float(weight))
                for surface_id, weight in row.get("coefficients") or ()
            ),
            nearest_surface_distance=float(row["nearest_surface_distance"]),
            seed_surface_id=str(row["seed_surface_id"]),
            support_graph_hops=int(row["support_graph_hops"]),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("rows") or ()
    )
    value = TESSAMaterialSupportFieldIR(
        rows=rows,
        surface_binding_hash=str(payload["surface_binding_hash"]),
        partition_binding_hash=str(payload["partition_binding_hash"]),
        topology_sequence_hash=str(payload["topology_sequence_hash"]),
        field_lineage_hash=str(payload["field_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.field_lineage_hash != tessa_material_support_field_lineage_hash(value):
        raise RuntimeError("MIRA_V55_MATERIAL_SUPPORT_LINEAGE_DRIFT")
    return value


def _area_weighted_vertex_normals(vertices: np.ndarray, faces: np.ndarray):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("MIRA_V55_MESH_VERTEX_SHAPE_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("MIRA_V55_MESH_FACE_SHAPE_INVALID")
    normals = np.zeros_like(vertices)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    magnitude = np.linalg.norm(normals, axis=1)
    valid = magnitude > 1e-12
    normals[valid] /= magnitude[valid, None]
    normals[~valid] = 0.0
    return normals.astype(np.float32), valid


def _conditioning_to_torch(conditioning, device):
    import torch

    floats = (
        "surface_positions_normalized",
        "surface_normals",
        "surface_raster_xy",
        "edge_features",
        "joint_positions_normalized",
        "joint_depth_normalized",
        "pair_geometry",
        "view_yaw_code",
    )
    booleans = (
        "surface_normal_valid",
        "surface_support_views",
        "surface_raster_valid",
        "surface_observed",
        "surface_completed",
        "surface_mask",
        "edge_mask",
        "joint_mask",
        "root_mask",
        "deform_root_mask",
        "support_anchor_matrix",
        "pair_mask",
    )
    integers = ("edge_index", "parent_indices")
    out = {}
    for name in floats:
        key = "view_yaw_fourier" if name == "view_yaw_code" else name
        out[key] = torch.as_tensor(
            getattr(conditioning, name), device=device, dtype=torch.float32
        )
    for name in booleans:
        out[name] = torch.as_tensor(
            getattr(conditioning, name), device=device, dtype=torch.bool
        )
    for name in integers:
        out[name] = torch.as_tensor(
            getattr(conditioning, name), device=device, dtype=torch.long
        )
    return out


def _compose_mesh_to_gsa_support(mesh, support_field, conditioning_surface_ids):
    support_by_tessa = {
        str(row.proposal_vertex_id): tuple(row.coefficients)
        for row in support_field.rows
    }
    if len(support_by_tessa) != len(support_field.rows):
        raise RuntimeError("MIRA_V55_MATERIAL_SUPPORT_DUPLICATE_VERTEX")
    gsa_row = {
        str(surface_id): index
        for index, surface_id in enumerate(conditioning_surface_ids)
    }
    if len(gsa_row) != len(conditioning_surface_ids):
        raise RuntimeError("MIRA_V55_GSA_SURFACE_ID_DUPLICATE")

    composed: list[tuple[tuple[int, float], ...]] = []
    for vertex in mesh.vertices:
        accum: dict[int, float] = {}
        tessa_total = 0.0
        for proposal_vertex_id, outer_weight in vertex.support_binding.coefficients:
            outer_weight = float(outer_weight)
            if outer_weight < -1e-12:
                raise RuntimeError("MIRA_V55_MESH_SUPPORT_NEGATIVE")
            row = support_by_tessa.get(str(proposal_vertex_id))
            if row is None:
                raise RuntimeError("MIRA_V55_MATERIAL_SUPPORT_VERTEX_MISSING")
            tessa_total += outer_weight
            for surface_id, inner_weight in row:
                surface_id = str(surface_id)
                if surface_id not in gsa_row:
                    raise RuntimeError("MIRA_V55_MATERIAL_SUPPORT_GSA_ID_UNKNOWN")
                value = outer_weight * float(inner_weight)
                if value > 1e-15:
                    index = gsa_row[surface_id]
                    accum[index] = accum.get(index, 0.0) + value
        if abs(tessa_total - 1.0) > 2e-6:
            raise RuntimeError("MIRA_V55_MESH_SUPPORT_NOT_SIMPLEX")
        total = sum(accum.values())
        if total <= 0.0:
            raise RuntimeError("MIRA_V55_COMPOSED_SUPPORT_EMPTY")
        normalized = tuple(
            (index, float(weight / total))
            for index, weight in sorted(accum.items())
            if weight / total > 1e-15
        )
        if abs(sum(weight for _, weight in normalized) - 1.0) > 2e-6:
            raise RuntimeError("MIRA_V55_COMPOSED_SUPPORT_NOT_SIMPLEX")
        composed.append(normalized)
    return tuple(composed)


def infer_dense_wm_v55(
    *,
    backbone,
    decoder,
    mesh,
    rig,
    surface,
    material_support_payload: Mapping[str, Any],
    device,
    decode_chunk: int = 512,
):
    import torch

    from models.arachne.v2.arachne_geometry_v2 import arachne_pair_geometry_v2
    from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
    from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
        tensorize_rigging_surface_v1,
    )

    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise RuntimeError("MIRA_V55_RIG_MESH_PARENT_DRIFT")
    if surface.geometry_lineage_hash != rig.feature_evidence_binding_hash:
        raise RuntimeError("MIRA_V55_FEATURE_EVIDENCE_RIG_BINDING_DRIFT")
    if surface.geometry_lineage_hash != mesh.source_evidence_binding_hash:
        raise RuntimeError("MIRA_V55_FEATURE_EVIDENCE_MESH_BINDING_DRIFT")

    support_field = _material_support_rows(material_support_payload)
    if support_field.surface_binding_hash != surface.geometry_lineage_hash:
        raise RuntimeError("MIRA_V55_MATERIAL_SUPPORT_SURFACE_BINDING_DRIFT")
    if support_field.partition_binding_hash != mesh.partition_binding_hash:
        raise RuntimeError("MIRA_V55_MATERIAL_SUPPORT_PARTITION_BINDING_DRIFT")

    skeleton = _qualified_skeleton_view(rig)
    conditioning = ArachneRichConditioningAdapterV3(require_scene_first=True)(
        [surface], [skeleton]
    )
    conditioning_joint_ids = tuple(map(str, conditioning.joint_ids[0]))
    skeleton_joint_ids = tuple(
        str(joint.canonical_joint_id) for joint in skeleton.joints
    )
    if conditioning_joint_ids != skeleton_joint_ids:
        raise RuntimeError("MIRA_V55_JOINT_ORDER_DRIFT")

    torch_device = torch.device(device)
    ci = _conditioning_to_torch(conditioning, torch_device)
    backbone = backbone.to(device=torch_device, dtype=torch.float32)
    decoder = decoder.to(device=torch_device, dtype=torch.float32)
    backbone.eval()
    decoder.eval()

    if torch_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    autocast_enabled = torch_device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=torch_device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        raw = backbone(**ci)
    surface_memory_gsa = raw.surface_memory[0].float().detach()
    field_tokens = raw.field_tokens[0].float().detach()

    conditioning_surface_ids = tuple(map(str, conditioning.surface_ids[0]))
    composed = _compose_mesh_to_gsa_support(
        mesh, support_field, conditioning_surface_ids
    )
    surface_memory = torch.zeros(
        (len(mesh.vertices), surface_memory_gsa.shape[-1]),
        device=torch_device,
        dtype=torch.float32,
    )
    for row_index, row in enumerate(composed):
        if not row:
            raise RuntimeError("MIRA_V55_COMPOSED_SUPPORT_ROW_EMPTY")
        indices = torch.as_tensor(
            [index for index, _ in row], device=torch_device, dtype=torch.long
        )
        coeff = torch.as_tensor(
            [weight for _, weight in row], device=torch_device, dtype=torch.float32
        )
        surface_memory[row_index] = (
            surface_memory_gsa[indices] * coeff[:, None]
        ).sum(dim=0)

    vertices = np.asarray([vertex.P for vertex in mesh.vertices], dtype=np.float64)
    canonical_index = {
        str(vertex.canonical_mesh_vertex_id): index
        for index, vertex in enumerate(mesh.vertices)
    }
    if len(canonical_index) != len(mesh.vertices):
        raise RuntimeError("MIRA_V55_CANONICAL_MESH_ID_DUPLICATE")
    try:
        faces = np.asarray(
            [
                [canonical_index[str(vertex_id)] for vertex_id in face]
                for face in mesh.faces
            ],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RuntimeError("MIRA_V55_CANONICAL_MESH_FACE_ID_UNKNOWN") from exc

    normals, normal_valid = _area_weighted_vertex_normals(vertices, faces)
    gsa_tensor = tensorize_rigging_surface_v1(surface, require_scene_first=True)
    if tuple(gsa_tensor.surface_ids) != conditioning_surface_ids:
        raise RuntimeError("MIRA_V55_GSA_TENSORIZATION_ORDER_DRIFT")
    if (
        np.max(
            np.abs(
                np.asarray(gsa_tensor.positions_normalized, np.float32)
                - np.asarray(
                    conditioning.surface_positions_normalized[0], np.float32
                )
            )
        )
        > 1e-7
    ):
        raise RuntimeError("MIRA_V55_GSA_NORMALIZATION_DRIFT")
    center = np.asarray(gsa_tensor.normalization_center, dtype=np.float64)
    scale = float(gsa_tensor.normalization_scale)
    normalized = ((vertices - center[None, :]) / scale).astype(np.float32)
    geom_np = np.concatenate(
        [
            2.0 * normalized,
            normals,
            normal_valid[:, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    geom = torch.as_tensor(geom_np, device=torch_device, dtype=torch.float32)

    joint_pos = np.asarray(conditioning.joint_positions_normalized[0:1], np.float32)
    parent = np.asarray(conditioning.parent_indices[0:1], np.int64)
    surface_mask = np.ones((1, len(mesh.vertices)), dtype=bool)
    joint_mask = np.asarray(conditioning.joint_mask[0:1], dtype=bool)
    pair_np, legal_np = arachne_pair_geometry_v2(
        normalized[None],
        normals[None],
        normal_valid[None],
        joint_pos,
        parent,
        surface_mask,
        joint_mask,
    )
    pair_geometry = torch.as_tensor(
        pair_np[0], device=torch_device, dtype=torch.float32
    )
    legal = torch.as_tensor(legal_np[0], device=torch_device, dtype=torch.bool)

    with torch.no_grad():
        _, weights = decoder.decode_all(
            surface_memory,
            geom,
            pair_geometry,
            field_tokens,
            legal,
            chunk=int(decode_chunk),
        )
    if weights.shape != (len(mesh.vertices), len(conditioning_joint_ids)):
        raise RuntimeError("MIRA_V55_OUTPUT_SHAPE_DRIFT")
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("MIRA_V55_OUTPUT_NONFINITE")
    row_residual = (weights.sum(dim=-1) - 1.0).abs()
    if float(row_residual.max().item()) > 1e-6:
        raise RuntimeError("MIRA_V55_OUTPUT_SIMPLEX_DRIFT")
    if float(weights.min().item()) < -1e-8:
        raise RuntimeError("MIRA_V55_OUTPUT_NEGATIVE")

    carrier = MiraCarrierViewV1(
        ordered_vertex_ids=tuple(
            str(vertex.source_candidate_vertex_id) for vertex in mesh.vertices
        ),
        carrier_evidence_hash=str(mesh.carrier_evidence_hash),
        topology_hash=str(mesh.carrier_topology_hash),
        geometry_hash=str(mesh.carrier_geometry_hash),
    )
    return (
        weights.detach().cpu().numpy().astype(np.float64),
        carrier,
        skeleton,
        {
            "joint_ids": conditioning_joint_ids,
            "mesh_vertex_count": len(mesh.vertices),
            "gsa_surface_count": len(conditioning_surface_ids),
            "decode_chunk": int(decode_chunk),
            "cuda_bfloat16_backbone_autocast": bool(autocast_enabled),
        },
    )


__all__ = [
    "MiraCarrierViewV1",
    "infer_dense_wm_v55",
]
