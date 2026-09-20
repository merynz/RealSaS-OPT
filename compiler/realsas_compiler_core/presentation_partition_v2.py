from __future__ import annotations

"""Role-free V2 presentation partition from observable mechanics + appearance evidence."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from .appearance_authority_v2 import CAA_PROVENANCE
from .appearance_bake_v2 import (
    bilinear_premultiplied_rgba,
    conservative_bilinear_provenance,
)
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class PresentationPartitionEvidenceV2IR:
    mesh_binding_hash: str
    appearance_asset_binding_hash: str
    appearance_qualification_binding_hash: str
    policy_hash: str
    evaluated_shared_edge_count: int
    source_supported_edge_count: int
    cut_face_pairs: tuple[tuple[int, int], ...]
    boundary_measurements: tuple[Json, ...]
    evidence_hash: str
    schema_version: str = "RealSaS.PresentationPartitionEvidenceIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def presentation_partition_evidence_hash(
    value: PresentationPartitionEvidenceV2IR,
) -> str:
    payload = value.to_dict()
    payload.pop("evidence_hash", None)
    return content_sha256(payload)


def _nearest_scalar(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    points = np.asarray(uv, dtype=np.float64)
    if source.ndim != 2 or points.ndim != 2 or points.shape[1] != 2:
        raise QualificationError("PRESENTATION_PARTITION_SAMPLE_SHAPE_INVALID")
    height, width = source.shape
    x = np.rint(np.clip(points[:, 0], 0.0, 1.0) * float(width - 1)).astype(
        np.int64
    )
    y = np.rint(np.clip(points[:, 1], 0.0, 1.0) * float(height - 1)).astype(
        np.int64
    )
    return source[y, x]


def _face_components(mesh) -> tuple[str, ...]:
    component_by_vertex = {
        str(vertex.canonical_mesh_vertex_id): str(vertex.component_id)
        for vertex in mesh.vertices
    }
    rows = []
    for face in mesh.faces:
        components = {
            component_by_vertex.get(str(vertex_id))
            for vertex_id in face
        }
        if None in components or len(components) != 1:
            raise QualificationError("PRESENTATION_PARTITION_FACE_COMPONENT_INVALID")
        rows.append(str(next(iter(components))))
    return tuple(rows)


def _shared_edge_rows(mesh, face_components: tuple[str, ...]):
    edge_to_faces: dict[tuple[str, str], list[int]] = {}
    for face_index, face in enumerate(mesh.faces):
        vertices = tuple(map(str, face))
        if len(vertices) != 3 or len(set(vertices)) != 3:
            raise QualificationError("PRESENTATION_PARTITION_FACE_INVALID")
        for a, b in (
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0]),
        ):
            edge_to_faces.setdefault(tuple(sorted((a, b))), []).append(face_index)

    rows = []
    for edge, incident in sorted(edge_to_faces.items()):
        if len(incident) > 2:
            raise QualificationError("PRESENTATION_PARTITION_NONMANIFOLD_EDGE")
        if len(incident) != 2:
            continue
        a, b = sorted(map(int, incident))
        if face_components[a] != face_components[b]:
            continue
        rows.append((edge, a, b, face_components[a]))
    return tuple(rows)


def _inset_edge_uv(face, face_uv, shared_edge, t_values, inset: float):
    vertices = tuple(map(str, face))
    a, b = shared_edge
    if a not in vertices or b not in vertices:
        raise QualificationError("PRESENTATION_PARTITION_SHARED_EDGE_DRIFT")
    ia = vertices.index(a)
    ib = vertices.index(b)
    opposite = ({0, 1, 2} - {ia, ib}).pop()
    uv_a = np.asarray(face_uv[ia], dtype=np.float64)
    uv_b = np.asarray(face_uv[ib], dtype=np.float64)
    uv_c = np.asarray(face_uv[opposite], dtype=np.float64)
    edge_points = (
        (1.0 - t_values[:, None]) * uv_a[None, :]
        + t_values[:, None] * uv_b[None, :]
    )
    return (1.0 - inset) * edge_points + inset * uv_c[None, :]


def build_presentation_partition_evidence(
    *,
    mesh,
    appearance_asset_hash: str,
    appearance_qualification_hash: str,
    face_uv: np.ndarray,
    textures_by_direction: Mapping[int, np.ndarray],
    provenance_by_direction: np.ndarray,
    policy: Mapping[str, Any],
) -> PresentationPartitionEvidenceV2IR:
    policy = dict(policy)
    mechanical = dict(policy.get("mechanical_binding_policy") or {})
    boundary = dict(policy.get("appearance_boundary_policy") or {})
    if not mechanical or not boundary:
        raise QualificationError("PRESENTATION_PARTITION_POLICY_INCOMPLETE")

    sample_count = int(boundary.get("edge_samples_per_direction", 0))
    inset = float(boundary.get("edge_inset_fraction", -1.0))
    min_source_pairs = int(boundary.get("min_source_backed_sample_pairs", 0))
    mean_cut = float(boundary.get("mean_premultiplied_rgba_l1_cut", -1.0))
    p95_cut = float(boundary.get("p95_premultiplied_rgba_l1_cut", -1.0))
    source_names = tuple(map(str, boundary.get("source_backed_provenance") or ()))
    if (
        sample_count < 3
        or not (0.0 < inset < 0.5)
        or min_source_pairs <= 0
        or not (0.0 <= mean_cut <= 1.0)
        or not (0.0 <= p95_cut <= 1.0)
        or not source_names
    ):
        raise QualificationError("PRESENTATION_PARTITION_POLICY_INVALID")
    try:
        source_codes = {int(CAA_PROVENANCE[name]) for name in source_names}
    except KeyError as exc:
        raise QualificationError(
            "PRESENTATION_PARTITION_PROVENANCE_POLICY_INVALID"
        ) from exc

    uv = np.asarray(face_uv, dtype=np.float64)
    if uv.shape != (len(mesh.faces), 3, 2) or not np.isfinite(uv).all():
        raise QualificationError("PRESENTATION_PARTITION_FACE_UV_INVALID")
    provenance = np.asarray(provenance_by_direction, dtype=np.uint8)
    if provenance.ndim != 3 or provenance.shape[0] != 8:
        raise QualificationError("PRESENTATION_PARTITION_PROVENANCE_INVALID")
    textures = {int(key): np.asarray(value, dtype=np.uint8) for key, value in textures_by_direction.items()}
    if set(textures) != set(range(8)):
        raise QualificationError("PRESENTATION_PARTITION_TEXTURE_MATRIX_INCOMPLETE")
    for direction, texture in textures.items():
        if (
            texture.ndim != 3
            or texture.shape[2] != 4
            or texture.shape[:2] != provenance[direction].shape
        ):
            raise QualificationError("PRESENTATION_PARTITION_TEXTURE_SHAPE_INVALID")

    components = _face_components(mesh)
    shared_rows = _shared_edge_rows(mesh, components)
    t_values = np.linspace(
        1.0 / float(sample_count + 1),
        float(sample_count) / float(sample_count + 1),
        sample_count,
        dtype=np.float64,
    )

    measurements = []
    cuts = []
    supported_edges = 0
    for shared_edge, face_a, face_b, component_id in shared_rows:
        uv_a = _inset_edge_uv(
            mesh.faces[face_a], uv[face_a], shared_edge, t_values, inset
        )
        uv_b = _inset_edge_uv(
            mesh.faces[face_b], uv[face_b], shared_edge, t_values, inset
        )
        errors = []
        for direction in range(8):
            prov_a = conservative_bilinear_provenance(provenance[direction], uv_a)
            prov_b = conservative_bilinear_provenance(provenance[direction], uv_b)
            eligible = np.asarray(
                [
                    int(a) in source_codes and int(b) in source_codes
                    for a, b in zip(prov_a, prov_b)
                ],
                dtype=bool,
            )
            if not np.any(eligible):
                continue
            pm_a = bilinear_premultiplied_rgba(textures[direction], uv_a)
            pm_b = bilinear_premultiplied_rgba(textures[direction], uv_b)
            l1 = np.mean(np.abs(pm_a - pm_b), axis=1)
            errors.extend(float(value) for value in l1[eligible])

        count = len(errors)
        mean_l1 = float(np.mean(errors)) if errors else 0.0
        p95_l1 = float(np.quantile(errors, 0.95)) if errors else 0.0
        source_supported = count >= min_source_pairs
        if source_supported:
            supported_edges += 1
        cut = bool(
            source_supported
            and (mean_l1 >= mean_cut or p95_l1 >= p95_cut)
        )
        if cut:
            cuts.append((face_a, face_b))
        measurements.append(
            {
                "component_id": component_id,
                "face_a": face_a,
                "face_b": face_b,
                "shared_vertex_ids": list(shared_edge),
                "source_backed_sample_pair_count": count,
                "mean_premultiplied_rgba_l1": mean_l1,
                "p95_premultiplied_rgba_l1": p95_l1,
                "source_supported": source_supported,
                "cut": cut,
                "decision": (
                    "CUT_SOURCE_BACKED_APPEARANCE_BOUNDARY"
                    if cut
                    else (
                        "KEEP_SOURCE_BACKED_CONTINUITY"
                        if source_supported
                        else "KEEP_INSUFFICIENT_SOURCE_BOUNDARY_EVIDENCE"
                    )
                ),
            }
        )

    value = PresentationPartitionEvidenceV2IR(
        mesh_binding_hash=str(mesh.mesh_lineage_hash),
        appearance_asset_binding_hash=str(appearance_asset_hash),
        appearance_qualification_binding_hash=str(appearance_qualification_hash),
        policy_hash=content_sha256(policy),
        evaluated_shared_edge_count=len(shared_rows),
        source_supported_edge_count=supported_edges,
        cut_face_pairs=tuple(sorted(set(cuts))),
        boundary_measurements=tuple(measurements),
        evidence_hash="",
        metadata={
            "role_free": True,
            "categorical_recognition_used": False,
            "conceptual_object_identity_claimed": False,
            "evidence_supported_visual_partition": True,
            "appearance_boundary_does_not_mint_appearance": True,
        },
    )
    return replace(
        value,
        evidence_hash=presentation_partition_evidence_hash(value),
    )


def presentation_partition_evidence_from_dict(
    payload: Mapping[str, Any],
) -> PresentationPartitionEvidenceV2IR:
    value = PresentationPartitionEvidenceV2IR(
        mesh_binding_hash=str(payload["mesh_binding_hash"]),
        appearance_asset_binding_hash=str(payload["appearance_asset_binding_hash"]),
        appearance_qualification_binding_hash=str(
            payload["appearance_qualification_binding_hash"]
        ),
        policy_hash=str(payload["policy_hash"]),
        evaluated_shared_edge_count=int(payload["evaluated_shared_edge_count"]),
        source_supported_edge_count=int(payload["source_supported_edge_count"]),
        cut_face_pairs=tuple(
            (int(row[0]), int(row[1]))
            for row in (payload.get("cut_face_pairs") or ())
        ),
        boundary_measurements=tuple(
            dict(row) for row in (payload.get("boundary_measurements") or ())
        ),
        evidence_hash=str(payload["evidence_hash"]),
        schema_version=str(
            payload.get("schema_version")
            or "RealSaS.PresentationPartitionEvidenceIR.v2"
        ),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.evidence_hash != presentation_partition_evidence_hash(value):
        raise QualificationError("PRESENTATION_PARTITION_EVIDENCE_HASH_DRIFT")
    return value
