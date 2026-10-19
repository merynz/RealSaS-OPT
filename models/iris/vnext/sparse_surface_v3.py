from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import (
    IndexedSparseTetraMeshV5,
    SparseRegularTetraPolicyV5,
    decode_indexed_sparse_tetra_mt_v5,
    write_stage12_npz_v5,
)


@dataclass(frozen=True)
class VNextSparseSurfaceQueryPolicyV3:
    """Execution policy for querying a sealed vNext scalar field during extraction.

    The V5 sparse decoder already chunks the regular grid.  vNext still needs a
    smaller internal chunk because LocalEvidenceField materializes full-width H
    before producing the scalar field.  This policy changes only execution memory,
    never coordinates, field sign or extraction authority.
    """

    model_query_chunk: int = 32768
    require_eval_mode: bool = True
    enforce_normalized_domain: bool = True

    def validate(self) -> None:
        if int(self.model_query_chunk) <= 0:
            raise ValueError("IRIS_VNEXT_SPARSE_MODEL_QUERY_CHUNK_INVALID")


class VNextScalarFieldQueryV3:
    """NumPy query adapter from the V5 sparse decoder to vNext query_local().

    Contract:
    - one already-encoded scene only;
    - normalized coordinates are passed through unchanged in [-1, 1];
    - raw learned field sign is preserved (no hidden sign flip / offset);
    - inference-only and teacher-free;
    - no extraction_delta or uncertainty is mixed into the scalar field.
    """

    architecture_id = "RealSaS.IRIS.VNextSparseScalarFieldAdapter.v3"
    coordinate_transform = "IDENTITY_NORMALIZED_MINUS1_PLUS1"
    scalar_transform = "IDENTITY_LEARNED_FIELD"
    teacher_truth_used = False

    def __init__(
        self,
        model: nn.Module,
        scene_planes: torch.Tensor,
        *,
        policy: VNextSparseSurfaceQueryPolicyV3 = VNextSparseSurfaceQueryPolicyV3(),
    ) -> None:
        policy.validate()
        if policy.require_eval_mode and model.training:
            raise ValueError("IRIS_VNEXT_SPARSE_MODEL_MUST_BE_EVAL")
        if scene_planes.ndim != 5 or scene_planes.shape[0] != 1 or scene_planes.shape[1] != 3:
            raise ValueError("IRIS_VNEXT_SPARSE_SCENE_PLANES_SHAPE_INVALID")
        if not torch.is_floating_point(scene_planes):
            raise ValueError("IRIS_VNEXT_SPARSE_SCENE_PLANES_DTYPE_INVALID")
        if not bool(torch.isfinite(scene_planes).all().item()):
            raise ValueError("IRIS_VNEXT_SPARSE_SCENE_PLANES_NONFINITE")
        if not hasattr(model, "query_local"):
            raise TypeError("IRIS_VNEXT_SPARSE_QUERY_LOCAL_REQUIRED")

        self.model = model
        self.scene_planes = scene_planes.detach()
        self.policy = policy

    def __call__(self, points: np.ndarray) -> np.ndarray:
        q = np.asarray(points, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("IRIS_VNEXT_SPARSE_QUERY_POINTS_SHAPE_INVALID")
        if len(q) == 0:
            return np.empty((0,), dtype=np.float32)
        if not np.isfinite(q).all():
            raise ValueError("IRIS_VNEXT_SPARSE_QUERY_POINTS_NONFINITE")
        if self.policy.enforce_normalized_domain and float(np.max(np.abs(q))) > 1.00001:
            raise ValueError("IRIS_VNEXT_SPARSE_QUERY_POINTS_OUT_OF_DOMAIN")

        out = np.empty((len(q),), dtype=np.float32)
        chunk = int(self.policy.model_query_chunk)
        device = self.scene_planes.device
        dtype = self.scene_planes.dtype
        with torch.inference_mode():
            for start in range(0, len(q), chunk):
                end = min(len(q), start + chunk)
                points_t = torch.from_numpy(q[start:end]).to(device=device, dtype=dtype)
                local = self.model.query_local(self.scene_planes, points_t.unsqueeze(0))
                if not isinstance(local, dict) or "field" not in local:
                    raise ValueError("IRIS_VNEXT_SPARSE_FIELD_OUTPUT_MISSING")
                field = local["field"]
                if field.shape != (1, end - start):
                    raise ValueError("IRIS_VNEXT_SPARSE_FIELD_OUTPUT_SHAPE_INVALID")
                values = field.detach().to(dtype=torch.float32, device="cpu").numpy()[0]
                if not np.isfinite(values).all():
                    raise ValueError("IRIS_VNEXT_SPARSE_FIELD_OUTPUT_NONFINITE")
                out[start:end] = values
        return out


@dataclass(frozen=True)
class DirectIRISSurfaceReceiptV1:
    subject_id: str
    source_commit: str
    checkpoint_sha256: str
    scene_binding_sha256: str
    candidate_binding_hash: str
    stage12_npz_sha256: str
    decoder_id: str
    base_cells: int
    fine_cells: int
    vertex_count: int
    face_count: int
    field_coordinate_transform: str = "IDENTITY_NORMALIZED_MINUS1_PLUS1"
    field_scalar_transform: str = "IDENTITY_LEARNED_FIELD"
    teacher_truth_used: bool = False
    compiler_qualified: bool = False
    training_authorized: bool = False
    scientific_pass_claimed: bool = False
    status: str = "DIRECT_IRIS_SURFACE_MATERIALIZED__COMPILER_NOT_RUN"
    schema: str = "RealSaS.DirectIRISSurfaceReceipt.v1"

    def validate(self) -> None:
        if not self.subject_id:
            raise ValueError("DIRECT_IRIS_SURFACE_SUBJECT_EMPTY")
        if len(self.source_commit) != 40:
            raise ValueError("DIRECT_IRIS_SURFACE_SOURCE_COMMIT_INVALID")
        for name in (
            "checkpoint_sha256",
            "scene_binding_sha256",
            "candidate_binding_hash",
            "stage12_npz_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise ValueError(f"DIRECT_IRIS_SURFACE_HASH_INVALID:{name}")
        if self.base_cells <= 0 or self.fine_cells != 2 * self.base_cells:
            raise ValueError("DIRECT_IRIS_SURFACE_GRID_POLICY_INVALID")
        if self.vertex_count <= 0 or self.face_count <= 0:
            raise ValueError("DIRECT_IRIS_SURFACE_EMPTY_MESH")
        if self.teacher_truth_used or self.compiler_qualified:
            raise ValueError("DIRECT_IRIS_SURFACE_RECEIPT_AUTHORITY_ESCALATION")
        if self.training_authorized or self.scientific_pass_claimed:
            raise ValueError("DIRECT_IRIS_SURFACE_RECEIPT_CLAIM_ESCALATION")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _array_sha256(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("utf-8"))
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("utf-8"))
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def direct_surface_candidate_binding_hash_v1(
    mesh: IndexedSparseTetraMeshV5,
    *,
    subject_id: str,
    source_commit: str,
    checkpoint_sha256: str,
    scene_binding_sha256: str,
    sparse_policy: SparseRegularTetraPolicyV5,
) -> str:
    binding = {
        "schema": "RealSaS.DirectIRISSurfaceCandidateBinding.v1",
        "subject_id": str(subject_id),
        "source_commit": str(source_commit),
        "checkpoint_sha256": str(checkpoint_sha256),
        "scene_binding_sha256": str(scene_binding_sha256),
        "decoder_id": str(mesh.decoder_id),
        "base_cells": int(sparse_policy.base_cells),
        "fine_cells": int(sparse_policy.fine_cells),
        "vertices_sha256": _array_sha256(np.asarray(mesh.vertices_normalized, dtype=np.float32)),
        "faces_sha256": _array_sha256(np.asarray(mesh.faces, dtype=np.int32)),
        "implicit_normals_sha256": _array_sha256(np.asarray(mesh.implicit_normals, dtype=np.float32)),
    }
    payload = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def decode_vnext_sparse_surface_v3(
    model: nn.Module,
    scene_planes: torch.Tensor,
    *,
    sparse_policy: SparseRegularTetraPolicyV5 = SparseRegularTetraPolicyV5(),
    query_policy: VNextSparseSurfaceQueryPolicyV3 = VNextSparseSurfaceQueryPolicyV3(),
    work_dir: str | Path | None = None,
) -> IndexedSparseTetraMeshV5:
    """Decode the raw vNext field through the historical selected T512->T1024 path."""

    query = VNextScalarFieldQueryV3(model, scene_planes, policy=query_policy)
    mesh = decode_indexed_sparse_tetra_mt_v5(
        query,
        policy=sparse_policy,
        work_dir=work_dir,
    )
    diagnostics = dict(mesh.diagnostics)
    diagnostics.update(
        {
            "vnext_surface_bridge": VNextScalarFieldQueryV3.architecture_id,
            "field_coordinate_transform": VNextScalarFieldQueryV3.coordinate_transform,
            "field_scalar_transform": VNextScalarFieldQueryV3.scalar_transform,
            "model_query_chunk": int(query_policy.model_query_chunk),
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
        decoder_id=mesh.decoder_id,
    )


def write_direct_iris_surface_bundle_v1(
    mesh: IndexedSparseTetraMeshV5,
    output_dir: str | Path,
    *,
    subject_id: str,
    source_commit: str,
    checkpoint_sha256: str,
    scene_binding_sha256: str,
    sparse_policy: SparseRegularTetraPolicyV5,
) -> DirectIRISSurfaceReceiptV1:
    """Persist a model-only surface candidate and a deliberately non-qualified receipt."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stage12 = write_stage12_npz_v5(mesh, root / "DIRECT_IRIS_SURFACE.stage12.npz")
    stage12_sha = hashlib.sha256(stage12.read_bytes()).hexdigest()
    candidate_hash = direct_surface_candidate_binding_hash_v1(
        mesh,
        subject_id=subject_id,
        source_commit=source_commit,
        checkpoint_sha256=checkpoint_sha256,
        scene_binding_sha256=scene_binding_sha256,
        sparse_policy=sparse_policy,
    )
    receipt = DirectIRISSurfaceReceiptV1(
        subject_id=str(subject_id),
        source_commit=str(source_commit),
        checkpoint_sha256=str(checkpoint_sha256),
        scene_binding_sha256=str(scene_binding_sha256),
        candidate_binding_hash=candidate_hash,
        stage12_npz_sha256=stage12_sha,
        decoder_id=str(mesh.decoder_id),
        base_cells=int(sparse_policy.base_cells),
        fine_cells=int(sparse_policy.fine_cells),
        vertex_count=int(len(mesh.vertices_normalized)),
        face_count=int(len(mesh.faces)),
    )
    receipt.validate()
    (root / "DIRECT_IRIS_SURFACE_RECEIPT.json").write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
