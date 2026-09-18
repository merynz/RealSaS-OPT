from __future__ import annotations

"""Exact G5 multiview component coverage using Runtime-v3/v4 raster semantics."""

from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Any, Iterable

from ..hashing import content_sha256
from ..playback_runtime_v3 import ReferenceRasterContractV1
from ..playback_full_surface_v3 import CameraProjectionV3, project_points_xyz_v3
from ..product_authority_v1 import (
    ComponentCarrierPolicyIR,
    MeshQualificationPolicyIR,
    QualifiedMeshIR,
    validate_component_carrier_policy,
    validate_mesh_qualification_policy,
)
from ..types import QualificationError, RiggingSurfaceIR

Json = dict[str, Any]


@dataclass(frozen=True)
class ComponentObservationRasterIR:
    view_index: int
    component_id: str
    carrier_class: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    component_surface_set_hash: str
    width: int
    height: int
    mask_bytes: bytes
    mask_sha256: str
    source_observation_hash: str
    camera_binding_hash: str
    raster_contract_hash: str
    schema_version: str = "RealSaS.ComponentObservationRasterIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        # bytes are represented by exact digest + size in lineage-bearing dicts.
        return {
            "view_index": self.view_index,
            "component_id": self.component_id,
            "carrier_class": self.carrier_class,
            "partition_binding_hash": self.partition_binding_hash,
            "carrier_policy_binding_hash": self.carrier_policy_binding_hash,
            "component_surface_set_hash": self.component_surface_set_hash,
            "width": self.width,
            "height": self.height,
            "mask_sha256": self.mask_sha256,
            "mask_bytes_count": len(self.mask_bytes),
            "source_observation_hash": self.source_observation_hash,
            "camera_binding_hash": self.camera_binding_hash,
            "raster_contract_hash": self.raster_contract_hash,
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
        }


def mask_sha256(mask_bytes: bytes) -> str:
    return hashlib.sha256(bytes(mask_bytes)).hexdigest()


def component_surface_set_hash(component) -> str:
    return content_sha256({
        "schema": "RealSaS.ComponentSurfaceSetBinding.v1",
        "component_id": component.component_id,
        "surface_ids": tuple(sorted(component.surface_ids)),
    })


def validate_component_observation(value: ComponentObservationRasterIR, *, partition=None, carrier_policy=None) -> None:
    if not (0 <= int(value.view_index) < 8):
        raise QualificationError("G5_OBSERVATION_VIEW_INVALID")
    if not value.component_id or not value.carrier_class:
        raise QualificationError("G5_OBSERVATION_COMPONENT_INVALID")
    if int(value.width) <= 0 or int(value.height) <= 0:
        raise QualificationError("G5_OBSERVATION_DIMENSION_INVALID")
    if len(value.mask_bytes) != int(value.width) * int(value.height):
        raise QualificationError("G5_OBSERVATION_MASK_SIZE_INVALID")
    if any(x not in (0, 1) for x in value.mask_bytes):
        raise QualificationError("G5_OBSERVATION_MASK_BINARY_REQUIRED")
    if value.mask_sha256 != mask_sha256(value.mask_bytes):
        raise QualificationError("G5_OBSERVATION_MASK_HASH_MISMATCH")
    if not value.source_observation_hash or not value.camera_binding_hash or not value.raster_contract_hash:
        raise QualificationError("G5_OBSERVATION_AUTHORITY_BINDING_MISSING")
    if not value.partition_binding_hash or not value.carrier_policy_binding_hash or not value.component_surface_set_hash:
        raise QualificationError("G5_OBSERVATION_PARTITION_BINDING_MISSING")
    if partition is not None:
        if value.partition_binding_hash != partition.partition_lineage_hash:
            raise QualificationError("G5_OBSERVATION_PARTITION_LINEAGE_MISMATCH")
        component = next((row for row in partition.components if row.component_id == value.component_id), None)
        if component is None:
            raise QualificationError("G5_OBSERVATION_COMPONENT_NOT_IN_PARTITION")
        if value.component_surface_set_hash != component_surface_set_hash(component):
            raise QualificationError("G5_OBSERVATION_COMPONENT_SURFACE_SET_MISMATCH")
    if carrier_policy is not None:
        if value.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
            raise QualificationError("G5_OBSERVATION_CARRIER_POLICY_LINEAGE_MISMATCH")
        decision = next((row for row in carrier_policy.decisions if row.component_id == value.component_id), None)
        if decision is None or decision.carrier_class != value.carrier_class:
            raise QualificationError("G5_OBSERVATION_CARRIER_POLICY_DRIFT")


def _orient2d(a, b, px: float, py: float) -> float:
    return (float(b[0]) - float(a[0])) * (py - float(a[1])) - (float(b[1]) - float(a[1])) * (px - float(a[0]))


def _is_top_left(a, b) -> bool:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    return dy < 0.0 or (dy == 0.0 and dx > 0.0)


def _edge_accept(e: float, top_left: bool) -> bool:
    eps = 1e-12
    if e > eps:
        return True
    if e < -eps:
        return False
    return bool(top_left)


def _covers_pixel_center(a, b, c, pixel_x: int, pixel_y: int) -> bool:
    signed_area = _orient2d(a, b, float(c[0]), float(c[1]))
    if abs(signed_area) <= 1e-12:
        return False
    px = float(pixel_x) + 0.5
    py = float(pixel_y) + 0.5
    positive = signed_area > 0.0
    sign = 1.0 if positive else -1.0
    e0 = sign * _orient2d(b, c, px, py)
    e1 = sign * _orient2d(c, a, px, py)
    e2 = sign * _orient2d(a, b, px, py)
    tl0 = _is_top_left(b, c) if positive else _is_top_left(c, b)
    tl1 = _is_top_left(c, a) if positive else _is_top_left(a, c)
    tl2 = _is_top_left(a, b) if positive else _is_top_left(b, a)
    return _edge_accept(e0, tl0) and _edge_accept(e1, tl1) and _edge_accept(e2, tl2)


def rasterize_triangles_half_integer_top_left(
    triangles: Iterable[tuple[tuple[float,float], tuple[float,float], tuple[float,float]]],
    *,
    width: int,
    height: int,
) -> bytes:
    predicted = bytearray(int(width) * int(height))
    for a, b, c in triangles:
        xs = (float(a[0]), float(b[0]), float(c[0]))
        ys = (float(a[1]), float(b[1]), float(c[1]))
        minx = max(0, int(math.floor(min(xs) - 0.5)))
        maxx = min(int(width) - 1, int(math.ceil(max(xs) - 0.5)))
        miny = max(0, int(math.floor(min(ys) - 0.5)))
        maxy = min(int(height) - 1, int(math.ceil(max(ys) - 0.5)))
        if minx > maxx or miny > maxy:
            continue
        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                if _covers_pixel_center(a, b, c, x, y):
                    predicted[y * int(width) + x] = 1
    return bytes(predicted)


def _largest_4_connected(mask: bytes, *, width: int, height: int) -> int:
    n = int(width) * int(height)
    visited = bytearray(n)
    largest = 0
    for seed in range(n):
        if not mask[seed] or visited[seed]:
            continue
        visited[seed] = 1
        q = deque([seed])
        size = 0
        while q:
            idx = q.popleft()
            size += 1
            x, y = idx % width, idx // width
            for nxt in (
                idx - 1 if x > 0 else -1,
                idx + 1 if x + 1 < width else -1,
                idx - width if y > 0 else -1,
                idx + width if y + 1 < height else -1,
            ):
                if nxt >= 0 and mask[nxt] and not visited[nxt]:
                    visited[nxt] = 1
                    q.append(nxt)
        largest = max(largest, size)
    return int(largest)


def _interior_mask_8_neighbor(authority: bytes, *, width: int, height: int) -> bytes:
    """One-reference-pixel boundary band removal via Chebyshev-radius-1 erosion."""
    out = bytearray(int(width) * int(height))
    for y in range(1, int(height) - 1):
        for x in range(1, int(width) - 1):
            idx = y * int(width) + x
            if not authority[idx]:
                continue
            if all(
                authority[(y + dy) * int(width) + (x + dx)]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ):
                out[idx] = 1
    return bytes(out)


def coverage_metrics(authority: bytes, predicted: bytes, *, width: int, height: int) -> dict:
    if len(authority) != int(width) * int(height) or len(predicted) != len(authority):
        raise QualificationError("G5_COVERAGE_MASK_SIZE_MISMATCH")
    foreground = sum(authority)
    predicted_count = sum(predicted)
    inside = sum(1 for a, p in zip(authority, predicted) if a and p)
    recall = 1.0 if foreground == 0 else float(inside) / float(foreground)
    precision = 1.0 if predicted_count == 0 else float(inside) / float(predicted_count)

    uncovered = bytes(1 if a and not p else 0 for a, p in zip(authority, predicted))
    largest = _largest_4_connected(uncovered, width=width, height=height)
    largest_fraction = 0.0 if foreground == 0 else float(largest) / float(foreground)

    interior = _interior_mask_8_neighbor(authority, width=width, height=height)
    interior_count = sum(interior)
    interior_uncovered = sum(1 for i, p in zip(interior, predicted) if i and not p)
    interior_fraction = 0.0 if interior_count == 0 else float(interior_uncovered) / float(interior_count)

    return {
        "foreground_pixel_count": int(foreground),
        "predicted_pixel_count": int(predicted_count),
        "inside_pixel_count": int(inside),
        "recall": float(recall),
        "precision": float(precision),
        "largest_coherent_hole_pixels": int(largest),
        "largest_coherent_hole_fraction": float(largest_fraction),
        "interior_foreground_pixel_count": int(interior_count),
        "interior_uncovered_pixel_count": int(interior_uncovered),
        "interior_uncovered_fraction": float(interior_fraction),
    }


def camera_projection_binding_hash(camera: CameraProjectionV3) -> str:
    return content_sha256({
        "schema": camera.schema_version,
        "view_id": camera.view_id,
        "view_index": int(camera.view_index),
        "origin": tuple(map(float, camera.origin)),
        "right": tuple(map(float, camera.right)),
        "screen_up": tuple(map(float, camera.screen_up)),
        "forward": tuple(map(float, camera.forward)),
        "half_extent": float(camera.half_extent),
        "resolution": int(camera.resolution),
    })


def _mesh_component_triangles(mesh: QualifiedMeshIR, camera: CameraProjectionV3, *, component_id: str):
    vertex_ids = [vertex.canonical_mesh_vertex_id for vertex in mesh.vertices]
    xyz = [tuple(map(float, vertex.P)) for vertex in mesh.vertices]
    projected = project_points_xyz_v3(xyz, camera)
    by_id = {
        vertex_ids[i]: (float(projected[i, 0]), float(projected[i, 1]))
        for i in range(len(vertex_ids))
    }
    component_by_id = {
        vertex.canonical_mesh_vertex_id: vertex.component_id
        for vertex in mesh.vertices
    }

    triangles = []
    for face in mesh.faces:
        components = {component_by_id[vid] for vid in face}
        if len(components) != 1:
            raise QualificationError("G5_FACE_CROSSES_COMPONENT_BOUNDARY")
        if next(iter(components)) != component_id:
            continue
        triangles.append(tuple(by_id[vid] for vid in face))
    if not triangles:
        raise QualificationError("G5_COMPONENT_HAS_NO_RASTERIZABLE_FACE")
    return tuple(triangles)


def build_g5_coverage_matrix(
    mesh: QualifiedMeshIR,
    *,
    surface: RiggingSurfaceIR,
    partition,
    carrier_policy: ComponentCarrierPolicyIR,
    mesh_policy: MeshQualificationPolicyIR,
    observations: Iterable[ComponentObservationRasterIR],
    cameras: Iterable[CameraProjectionV3],
    raster_contract: ReferenceRasterContractV1 = ReferenceRasterContractV1(),
) -> tuple[dict, ...]:
    validate_component_carrier_policy(carrier_policy, partition)
    validate_mesh_qualification_policy(mesh_policy)
    if raster_contract.pixel_center != "HALF_INTEGER_CENTER" or raster_contract.triangle_fill_rule != "TOP_LEFT":
        raise QualificationError("G5_REFERENCE_RASTER_CONTRACT_UNSUPPORTED")
    raster_hash = raster_contract.contract_hash

    component_ids = {component.component_id for component in partition.components}
    carrier_by_component = {row.component_id: row.carrier_class for row in carrier_policy.decisions}
    thresholds = {row.carrier_class: row for row in mesh_policy.coverage_thresholds}
    camera_by_view = {int(camera.view_index): camera for camera in cameras}
    if set(camera_by_view) != set(range(8)):
        raise QualificationError("G5_REQUIRES_EXACT_8_CAMERAS")
    if len(camera_by_view) != 8:
        raise QualificationError("G5_DUPLICATE_CAMERA_VIEW")

    by_key = {}
    for observation in observations:
        validate_component_observation(observation, partition=partition, carrier_policy=carrier_policy)
        key = (int(observation.view_index), observation.component_id)
        if key in by_key:
            raise QualificationError("G5_DUPLICATE_COMPONENT_OBSERVATION")
        if observation.component_id not in component_ids:
            raise QualificationError("G5_UNKNOWN_COMPONENT_OBSERVATION")
        if observation.carrier_class != carrier_by_component[observation.component_id]:
            raise QualificationError("G5_OBSERVATION_CARRIER_POLICY_DRIFT")
        if observation.raster_contract_hash != raster_hash:
            raise QualificationError("G5_RASTER_CONTRACT_HASH_MISMATCH")
        camera = camera_by_view[int(observation.view_index)]
        if int(camera.resolution) != int(observation.width) or int(camera.resolution) != int(observation.height):
            raise QualificationError("G5_CAMERA_OBSERVATION_DIMENSION_MISMATCH")
        if observation.camera_binding_hash != camera_projection_binding_hash(camera):
            raise QualificationError("G5_CAMERA_BINDING_HASH_MISMATCH")
        by_key[key] = observation

    expected = {(view, cid) for view in range(8) for cid in component_ids}
    if set(by_key) != expected:
        raise QualificationError("G5_COMPONENT_OBSERVATION_MATRIX_INCOMPLETE")

    rows = []
    for view, component_id in sorted(expected):
        observation = by_key[(view, component_id)]
        triangles = _mesh_component_triangles(mesh, camera_by_view[view], component_id=component_id)
        predicted = rasterize_triangles_half_integer_top_left(
            triangles,
            width=observation.width,
            height=observation.height,
        )
        metrics = coverage_metrics(
            observation.mask_bytes,
            predicted,
            width=observation.width,
            height=observation.height,
        )
        threshold = thresholds[observation.carrier_class]
        passed = (
            metrics["recall"] >= threshold.min_recall
            and metrics["precision"] >= threshold.min_precision
            and metrics["largest_coherent_hole_fraction"] <= threshold.max_largest_coherent_hole_fraction
            and metrics["interior_uncovered_fraction"] <= threshold.max_interior_uncovered_fraction
        )
        rows.append({
            "view_index": int(view),
            "component_id": component_id,
            "carrier_class": observation.carrier_class,
            "partition_binding_hash": observation.partition_binding_hash,
            "carrier_policy_binding_hash": observation.carrier_policy_binding_hash,
            "component_surface_set_hash": observation.component_surface_set_hash,
            **metrics,
            "status": "PASS" if passed else "FAIL",
            "source_mask_sha256": observation.mask_sha256,
            "source_observation_hash": observation.source_observation_hash,
            "camera_binding_hash": observation.camera_binding_hash,
            "raster_contract_hash": raster_hash,
        })
    return tuple(rows)


def g5_coverage_evidence_hash(rows: Iterable[dict]) -> str:
    return content_sha256({
        "schema": "RealSaS.G5CoverageEvidence.v1",
        "rows": tuple(rows),
    })
