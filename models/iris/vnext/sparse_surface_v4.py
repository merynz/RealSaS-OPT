from __future__ import annotations

"""Corrected vNext direct-surface bridge with exact sparse-shell containment.

This module preserves the sealed vNext scalar field, normalized coordinates,
SparseRegularTetraPolicyV5 T512->T1024 carrier, and indexed marching-tetrahedra
geometry extraction.  The only correction is the already-canonical V5 shell
gate: the historical ``boundary_crossing_tets != 0`` proxy is not treated as a
truncation proof.  Extraction proceeds only after the exact refined-carrier
boundary-face zero-contact proof passes.

No teacher truth, TESSA, optimizer, backward pass, threshold change, or model
parameter mutation is introduced here.
"""

from pathlib import Path

import torch
from torch import nn

from models.iris.v5.contained_sparse_tetra_decoder_v5 import (
    CONTAINED_DECODER_ID_V5,
    decode_indexed_sparse_tetra_mt_contained_v5,
)
from models.iris.v5.indexed_sparse_tetra_decoder_v5 import (
    IndexedSparseTetraMeshV5,
    SparseRegularTetraPolicyV5,
)
from models.iris.vnext.sparse_surface_v3 import (
    DirectIRISSurfaceReceiptV1,
    VNextScalarFieldQueryV3,
    VNextSparseSurfaceQueryPolicyV3,
    direct_surface_candidate_binding_hash_v1,
    write_direct_iris_surface_bundle_v1,
)


DIRECT_SURFACE_DECODER_ID_V4 = CONTAINED_DECODER_ID_V5


def decode_vnext_sparse_surface_v4(
    model: nn.Module,
    scene_planes: torch.Tensor,
    *,
    sparse_policy: SparseRegularTetraPolicyV5 = SparseRegularTetraPolicyV5(),
    query_policy: VNextSparseSurfaceQueryPolicyV3 = VNextSparseSurfaceQueryPolicyV3(),
    work_dir: str | Path | None = None,
) -> IndexedSparseTetraMeshV5:
    """Decode the sealed vNext field through the corrected exact-containment lane."""

    query = VNextScalarFieldQueryV3(model, scene_planes, policy=query_policy)
    mesh = decode_indexed_sparse_tetra_mt_contained_v5(
        query,
        policy=sparse_policy,
        work_dir=work_dir,
    )
    if str(mesh.decoder_id) != CONTAINED_DECODER_ID_V5:
        raise RuntimeError("IRIS_VNEXT_DIRECT_SURFACE_CONTAINED_DECODER_ID_DRIFT")

    diagnostics = dict(mesh.diagnostics)
    if diagnostics.get("exact_shell_containment_required") is not True:
        raise RuntimeError("IRIS_VNEXT_DIRECT_SURFACE_CONTAINMENT_PROOF_MISSING")
    if diagnostics.get("legacy_boundary_crossing_tets_is_truncation_gate") is not False:
        raise RuntimeError("IRIS_VNEXT_DIRECT_SURFACE_LEGACY_PROXY_REACTIVATED")
    if diagnostics.get("passed") is not True:
        raise RuntimeError("IRIS_VNEXT_DIRECT_SURFACE_CONTAINMENT_NOT_PASS")

    diagnostics.update(
        {
            "vnext_surface_bridge": "RealSaS.IRIS.VNextSparseSurfaceBridge.v4",
            "vnext_field_query_api": query.query_api,
            "field_coordinate_transform": VNextScalarFieldQueryV3.coordinate_transform,
            "field_scalar_transform": VNextScalarFieldQueryV3.scalar_transform,
            "model_query_chunk": int(query_policy.model_query_chunk),
            "shell_gate": "EXACT_REFINED_CARRIER_BOUNDARY_FACE_ZERO_CONTACT_V1",
            "legacy_boundary_proxy_role": "TELEMETRY_ONLY",
            "gate_correction_only": True,
            "teacher_truth_used": False,
            "compiler_qualified": False,
            "scientific_pass_claimed": False,
        }
    )
    return IndexedSparseTetraMeshV5(
        vertices_normalized=mesh.vertices_normalized,
        faces=mesh.faces,
        implicit_normals=mesh.implicit_normals,
        diagnostics=diagnostics,
        decoder_id=CONTAINED_DECODER_ID_V5,
    )


__all__ = [
    "DIRECT_SURFACE_DECODER_ID_V4",
    "DirectIRISSurfaceReceiptV1",
    "VNextScalarFieldQueryV3",
    "VNextSparseSurfaceQueryPolicyV3",
    "decode_vnext_sparse_surface_v4",
    "direct_surface_candidate_binding_hash_v1",
    "write_direct_iris_surface_bundle_v1",
]
