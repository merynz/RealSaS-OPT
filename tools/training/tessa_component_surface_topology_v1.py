from __future__ import annotations

"""Pure TESSA source-topology adapter for Corpus400 component geometry authority.

This module intentionally lives outside the model runtime package so corpus and
truth courts can execute without importing torch or any train-time TESSA code.
TESSA consumes qualified artist-surface connectivity, not SDF/sign connectivity.
The adapter therefore exposes `surface_component_id` and deliberately refuses to
construct a mechanical topology when any source face is geometry-invalid. Other
truth families may still use the subject with their own masks.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class TessaSurfaceTopologyTruth:
    raw_to_topology_vertex: np.ndarray
    vertices_source_world: np.ndarray
    faces: np.ndarray
    surface_component_id: np.ndarray
    topology_vertex_owner: np.ndarray
    face_valid: np.ndarray

    def validate(self) -> None:
        mapping = np.asarray(self.raw_to_topology_vertex, dtype=np.int64)
        vertices = np.asarray(self.vertices_source_world, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        components = np.asarray(self.surface_component_id, dtype=np.int32)
        owner = np.asarray(self.topology_vertex_owner, dtype=np.int32)
        valid = np.asarray(self.face_valid, dtype=np.bool_)
        if mapping.ndim != 1:
            raise ValueError("TESSA_SURFACE_TOPOLOGY_RAW_MAP_LAYOUT")
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
            raise ValueError("TESSA_SURFACE_TOPOLOGY_VERTEX_LAYOUT")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
            raise ValueError("TESSA_SURFACE_TOPOLOGY_FACE_LAYOUT")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("TESSA_SURFACE_TOPOLOGY_FACE_INDEX")
        if components.shape != (len(faces),) or np.any(components < 0):
            raise ValueError("TESSA_SURFACE_TOPOLOGY_COMPONENT_LAYOUT")
        if owner.shape != (len(vertices),) or np.any(owner < 0):
            raise ValueError("TESSA_SURFACE_TOPOLOGY_OWNER_LAYOUT")
        if valid.shape != (len(faces),):
            raise ValueError("TESSA_SURFACE_TOPOLOGY_VALIDITY_LAYOUT")
        if not np.all(valid):
            raise ValueError("TESSA_SURFACE_TOPOLOGY_INVALID_FACE_REPAIR_REQUIRED")
        if np.any(owner[faces] != owner[faces][:, :1]):
            raise ValueError("TESSA_SURFACE_TOPOLOGY_FACE_OWNER_CROSS")

    @property
    def component_count(self) -> int:
        self.validate()
        return int(len(np.unique(self.surface_component_id)))


def load_tessa_surface_topology_v1(topology_npz: Path) -> TessaSurfaceTopologyTruth:
    with np.load(Path(topology_npz).resolve(), allow_pickle=False) as z:
        required = {
            "raw_to_topology_vertex",
            "topology_vertices_source_world",
            "topology_faces",
            "topology_vertex_owner",
            "face_valid",
            "surface_component_id",
            "sign_component_id",
            "sign_component_valid",
        }
        if not required.issubset(z.files):
            raise ValueError("TESSA_SURFACE_TOPOLOGY_COMPONENT_GEOMETRY_ARRAYS")
        result = TessaSurfaceTopologyTruth(
            raw_to_topology_vertex=np.asarray(z["raw_to_topology_vertex"], dtype=np.int64),
            vertices_source_world=np.asarray(z["topology_vertices_source_world"], dtype=np.float64),
            faces=np.asarray(z["topology_faces"], dtype=np.int64),
            surface_component_id=np.asarray(z["surface_component_id"], dtype=np.int32),
            topology_vertex_owner=np.asarray(z["topology_vertex_owner"], dtype=np.int32),
            face_valid=np.asarray(z["face_valid"], dtype=np.bool_),
        )
    result.validate()
    return result
