from __future__ import annotations

"""Runtime-only MIRA V5.5 inference on exact CanonicalMesh + CanonicalRig.

Current FIT1 V5.5 is intentionally carrier/rig-bound. RiggingSurface evidence is
used only to run the frozen rich backbone. CanonicalMesh owns query geometry and
support coefficients lift learned surface memory to exact M rows; no skin field
is transferred from GSA. The returned proposal binds the CanonicalRig lineage.
"""

from dataclasses import dataclass
import hashlib
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import CanonicalMeshIR
from compiler.realsas_compiler_core.canonical_rig_ir_v1 import CanonicalRigIR
from compiler.realsas_compiler_core.carrier_skin_v1 import CarrierSkinProposalIR
from compiler.realsas_compiler_core.types import QualifiedJoint, QualifiedSkeletonIR, QualificationError, RiggingSurfaceIR
from models.mira.carrier_native_v55 import MIRA_V55_ARCHITECTURE_ID, proposal_from_dense_wm_v55
from models.mira.runtime_checkpoint_v1 import validate_mira_runtime_checkpoint_v1


@dataclass(frozen=True)
class MIRAV55InferenceResult:
    proposal: CarrierSkinProposalIR
    diagnostics: dict[str, Any]


def _require_runtime():
    try:
        import torch
        from models.arachne.v2.arachne_geometry_v2 import arachne_pair_geometry_v2
        from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
        from models.arachne.v4.arachne_candidate_v4 import ArachneA1ConfigV4, ArachneA1V4
        from models.mira.v6_readout_v1 import MIRAV6RawReadout
    except Exception as exc:  # pragma: no cover - runtime environment guard
        raise QualificationError("MIRA_RUNTIME_DEPENDENCY_REQUIRED") from exc
    return torch, arachne_pair_geometry_v2, ArachneRichConditioningAdapterV3, ArachneA1ConfigV4, ArachneA1V4, MIRAV6RawReadout


def _array_contract_hash(rows) -> str:
    h = hashlib.sha256()
    for name, value in rows:
        array = np.ascontiguousarray(value)
        h.update(str(name).encode())
        h.update(str(array.dtype).encode())
        h.update(str(tuple(array.shape)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def canonical_mesh_fit1_carrier_basis_hash_v1(mesh: CanonicalMeshIR) -> str:
    vertices = np.asarray([vertex.P for vertex in mesh.vertices], np.float64)
    row = {str(vertex.canonical_mesh_vertex_id): i for i, vertex in enumerate(mesh.vertices)}
    if len(row) != len(mesh.vertices):
        raise QualificationError("MIRA_RUNTIME_CANONICAL_MESH_VERTEX_ID_DRIFT")
    try:
        faces = np.asarray([[row[str(vertex_id)] for vertex_id in face] for face in mesh.faces], np.int64)
    except KeyError as exc:
        raise QualificationError("MIRA_RUNTIME_CANONICAL_MESH_FACE_ID_DRIFT") from exc
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise QualificationError("MIRA_RUNTIME_CANONICAL_MESH_BASIS_INVALID")
    return _array_contract_hash((
        ("M.vertices_world", vertices),
        ("M.faces", faces),
    ))


def _conditioning_skeleton_view_v1(rig: CanonicalRigIR) -> QualifiedSkeletonIR:
    if len(rig.root_joint_ids) != 1:
        raise QualificationError("MIRA_V55_CURRENT_RUNTIME_REQUIRES_SINGLE_ROOT_RIG")
    joints = tuple(
        QualifiedJoint(
            canonical_joint_id=str(joint.canonical_joint_id),
            position=tuple(map(float, joint.P)),
            parent_canonical_id=(
                None if joint.parent_canonical_joint_id is None else str(joint.parent_canonical_joint_id)
            ),
            support_surface_ids=tuple(map(str, joint.proposal_support_evidence_ids)),
            source_proposal_id=str(joint.source_proposal_id),
        )
        for joint in rig.joints
    )
    return QualifiedSkeletonIR(
        joints=joints,
        root_id=str(rig.root_joint_ids[0]),
        qualification_report={
            "status": "MIRA_RUNTIME_CONDITIONING_VIEW_ONLY",
            "canonical_rig_lineage_hash": rig.rig_lineage_hash,
            "feature_evidence_is_canonical_parent": False,
        },
        # Preserve the exact historical evidence-frame lineage used by the
        # frozen MIRA backbone. Product authority still binds rig_lineage_hash.
        skeleton_lineage_hash=str(rig.structure_qualification_hash),
    )


def _conditioning_to_torch_v1(conditioning, device, torch):
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
    bools = (
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
    longs = ("edge_index", "parent_indices")
    out = {}
    for name in floats:
        out["view_yaw_fourier" if name == "view_yaw_code" else name] = torch.as_tensor(
            getattr(conditioning, name), device=device, dtype=torch.float32
        )
    for name in bools:
        out[name] = torch.as_tensor(getattr(conditioning, name), device=device, dtype=torch.bool)
    for name in longs:
        out[name] = torch.as_tensor(getattr(conditioning, name), device=device, dtype=torch.long)
    return out


def _area_weighted_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    normals = np.zeros_like(v)
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for corner in range(3):
        np.add.at(normals, f[:, corner], face_normals)
    magnitude = np.linalg.norm(normals, axis=1)
    valid = magnitude > 1e-12
    normals[valid] /= magnitude[valid, None]
    normals[~valid] = 0.0
    return normals.astype(np.float32), valid


def _surface_normalization_frame(conditioning) -> tuple[np.ndarray, float]:
    world = np.asarray(conditioning.surface_positions_world[0], np.float64)
    normalized = np.asarray(conditioning.surface_positions_normalized[0], np.float32)
    mask = np.asarray(conditioning.surface_mask[0], bool)
    world = world[mask]
    normalized = normalized[mask]
    if world.ndim != 2 or world.shape[1] != 3 or len(world) == 0:
        raise QualificationError("MIRA_RUNTIME_SURFACE_NORMALIZATION_DOMAIN_INVALID")
    lo = world.min(axis=0)
    hi = world.max(axis=0)
    center = (lo + hi) * 0.5
    scale = float(np.max(hi - lo))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise QualificationError("MIRA_RUNTIME_SURFACE_NORMALIZATION_SCALE_INVALID")
    reproduced = ((world - center[None]) / scale).astype(np.float32)
    if reproduced.shape != normalized.shape or float(np.max(np.abs(reproduced - normalized))) > 1e-7:
        raise QualificationError("MIRA_RUNTIME_SURFACE_NORMALIZATION_POLICY_DRIFT")
    return center.astype(np.float64), scale


def _mesh_basis(mesh: CanonicalMeshIR) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([vertex.P for vertex in mesh.vertices], np.float64)
    row = {str(vertex.canonical_mesh_vertex_id): i for i, vertex in enumerate(mesh.vertices)}
    faces = np.asarray([[row[str(vertex_id)] for vertex_id in face] for face in mesh.faces], np.int64)
    return vertices, faces


def _lift_surface_memory_to_mesh_v1(mesh: CanonicalMeshIR, conditioning, surface_memory_gsa, torch, device):
    surface_ids = tuple(map(str, conditioning.surface_ids[0]))
    surface_row = {surface_id: i for i, surface_id in enumerate(surface_ids)}
    if len(surface_row) != len(surface_ids):
        raise QualificationError("MIRA_RUNTIME_GSA_SURFACE_ID_DRIFT")
    rows: list[int] = []
    cols: list[int] = []
    coeffs: list[float] = []
    for mesh_row, vertex in enumerate(mesh.vertices):
        coefficients = tuple(vertex.support_binding.coefficients)
        if not coefficients:
            raise QualificationError("MIRA_RUNTIME_MESH_FEATURE_SUPPORT_EMPTY")
        total = 0.0
        seen: set[str] = set()
        for raw_surface_id, raw_coeff in coefficients:
            surface_id = str(raw_surface_id)
            coeff = float(raw_coeff)
            if surface_id in seen or surface_id not in surface_row:
                raise QualificationError("MIRA_RUNTIME_MESH_FEATURE_SUPPORT_ID_DRIFT")
            if not math.isfinite(coeff) or coeff < -1e-12:
                raise QualificationError("MIRA_RUNTIME_MESH_FEATURE_SUPPORT_WEIGHT_INVALID")
            seen.add(surface_id)
            total += max(0.0, coeff)
            rows.append(mesh_row)
            cols.append(surface_row[surface_id])
            coeffs.append(max(0.0, coeff))
        if abs(total - 1.0) > 1e-8:
            raise QualificationError("MIRA_RUNTIME_MESH_FEATURE_SUPPORT_SIMPLEX_DRIFT")
    row_t = torch.as_tensor(rows, device=device, dtype=torch.long)
    col_t = torch.as_tensor(cols, device=device, dtype=torch.long)
    coeff_t = torch.as_tensor(coeffs, device=device, dtype=torch.float32)
    lifted = torch.zeros(
        (len(mesh.vertices), int(surface_memory_gsa.shape[-1])),
        device=device,
        dtype=torch.float32,
    )
    lifted.index_add_(0, row_t, surface_memory_gsa[col_t].float() * coeff_t[:, None])
    return lifted


def infer_mira_v55_fit1_v1(
    *,
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIR,
    surface: RiggingSurfaceIR,
    runtime_checkpoint: dict[str, Any],
    device_name: str | None = None,
) -> MIRAV55InferenceResult:
    validate_mira_runtime_checkpoint_v1(runtime_checkpoint)
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise QualificationError("MIRA_RUNTIME_RIG_MESH_PARENT_DRIFT")
    if str(rig.feature_evidence_binding_hash) != str(surface.geometry_lineage_hash):
        raise QualificationError("MIRA_RUNTIME_FEATURE_EVIDENCE_BINDING_DRIFT")
    basis_hash = canonical_mesh_fit1_carrier_basis_hash_v1(mesh)
    if basis_hash != str(runtime_checkpoint["trained_carrier_basis_sha256"]):
        raise QualificationError("MIRA_RUNTIME_FIT1_CARRIER_BASIS_DRIFT")
    if str(rig.structure_qualification_hash) != str(runtime_checkpoint["trained_skeleton_lineage_hash"]):
        raise QualificationError("MIRA_RUNTIME_FIT1_RIG_LINEAGE_DRIFT")

    torch, pair_geometry_v2, ConditioningAdapter, ArachneConfig, ArachneBackbone, Decoder = _require_runtime()
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    skeleton_view = _conditioning_skeleton_view_v1(rig)
    conditioning = ConditioningAdapter(require_scene_first=True)([surface], [skeleton_view])
    if tuple(map(str, conditioning.source_skeleton_hashes)) != (str(rig.structure_qualification_hash),):
        raise QualificationError("MIRA_RUNTIME_CONDITIONING_RIG_LINEAGE_DRIFT")
    ci = _conditioning_to_torch_v1(conditioning, device, torch)

    config = ArachneConfig()
    backbone = ArachneBackbone(config).cpu().float()
    backbone.load_state_dict(runtime_checkpoint["backbone"], strict=True)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone = backbone.to(device=device, dtype=torch.float32).eval()

    decoder = Decoder(
        surface_dim=int(config.model_dim),
        token_dim=int(config.latent_channels),
        pair_dim=int(config.pair_geometry_dim),
        geom_dim=7,
        relation_dim=256,
    ).cpu().float()
    decoder.load_state_dict(runtime_checkpoint["decoder"], strict=True)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    decoder = decoder.to(device=device, dtype=torch.float32).eval()

    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if autocast_enabled else torch.float32,
        enabled=autocast_enabled,
    ):
        raw = backbone(**ci)
    surface_memory_gsa = raw.surface_memory[0].float().detach()
    field_tokens = raw.field_tokens[0].float().detach()
    surface_memory = _lift_surface_memory_to_mesh_v1(
        mesh, conditioning, surface_memory_gsa, torch, device
    )

    vertices, faces = _mesh_basis(mesh)
    normals, normal_valid = _area_weighted_vertex_normals(vertices, faces)
    center, scale = _surface_normalization_frame(conditioning)
    mesh_normalized = ((vertices - center[None]) / scale).astype(np.float32)
    geometry7 = np.concatenate(
        [2.0 * mesh_normalized, normals, normal_valid[:, None].astype(np.float32)],
        axis=1,
    ).astype(np.float32)

    joint_pos = np.asarray(conditioning.joint_positions_normalized[0:1], np.float32)
    parent = np.asarray(conditioning.parent_indices[0:1], np.int64)
    surface_mask = np.ones((1, len(vertices)), bool)
    joint_mask = np.asarray(conditioning.joint_mask[0:1], bool)
    pair_np, pair_mask_np = pair_geometry_v2(
        mesh_normalized[None],
        normals[None],
        normal_valid[None],
        joint_pos,
        parent,
        surface_mask,
        joint_mask,
    )
    geometry = torch.as_tensor(geometry7, device=device, dtype=torch.float32)
    pair_geometry = torch.as_tensor(pair_np[0], device=device, dtype=torch.float32)
    legal = torch.as_tensor(pair_mask_np[0], device=device, dtype=torch.bool)
    with torch.no_grad():
        _, prediction = decoder.decode_all(
            surface_memory,
            geometry,
            pair_geometry,
            field_tokens,
            legal,
            chunk=512,
        )
    weights = prediction.float().cpu().numpy().astype(np.float64)
    if weights.shape != (len(mesh.vertices), len(conditioning.joint_ids[0])):
        raise QualificationError("MIRA_RUNTIME_DENSE_WM_SHAPE_DRIFT")
    if not np.isfinite(weights).all() or float(np.max(np.abs(weights.sum(1) - 1.0))) > 1e-6:
        raise QualificationError("MIRA_RUNTIME_DENSE_WM_SIMPLEX_DRIFT")

    carrier_view = SimpleNamespace(
        ordered_vertex_ids=tuple(str(vertex.source_candidate_vertex_id) for vertex in mesh.vertices),
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        topology_hash=mesh.carrier_topology_hash,
        geometry_hash=mesh.carrier_geometry_hash,
    )
    proposal_skeleton_view = SimpleNamespace(
        joints=tuple(
            SimpleNamespace(canonical_joint_id=str(joint_id))
            for joint_id in conditioning.joint_ids[0]
        ),
        skeleton_lineage_hash=rig.rig_lineage_hash,
    )
    proposal = proposal_from_dense_wm_v55(
        weights=weights,
        carrier=carrier_view,
        skeleton=proposal_skeleton_view,
        model_provenance=MIRA_V55_ARCHITECTURE_ID,
    )
    return MIRAV55InferenceResult(
        proposal=proposal,
        diagnostics={
            "schema": "RealSaS.MIRAV55RuntimeInferenceDiagnostics.v1",
            "architecture_id": MIRA_V55_ARCHITECTURE_ID,
            "canonical_mesh_lineage_hash": mesh.mesh_lineage_hash,
            "canonical_rig_lineage_hash": rig.rig_lineage_hash,
            "conditioning_structure_lineage_hash": rig.structure_qualification_hash,
            "trained_carrier_basis_sha256": basis_hash,
            "feature_evidence_binding_hash": surface.geometry_lineage_hash,
            "gsa_role": "EVIDENCE_SUBSTRATE_ONLY",
            "query_geometry_authority": "CANONICAL_MESH",
            "semantic_skin_transfer_performed": False,
            "output_domain": "CANONICAL_MESH_CARRIER",
            "generalization_claimed": False,
            "runtime_device": str(device),
            "runtime_autocast_bf16": bool(autocast_enabled),
        },
    )


__all__ = [
    "MIRAV55InferenceResult",
    "canonical_mesh_fit1_carrier_basis_hash_v1",
    "infer_mira_v55_fit1_v1",
]
