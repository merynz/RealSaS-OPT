from __future__ import annotations

"""Typed IRIS training targets for component-scoped geometry authority.

This adapter is intentionally small and fail-closed. It does not create new truth;
it only exposes already-qualified producer/court bytes in a shape that makes the
intended loss semantics hard to misuse:

- unsigned surface distance is dense over its explicitly valid query rows;
- global signed distance and occupancy remain compact and indexed;
- local oriented SDF/brackets are derived only from valid surface samples + normals;
- local oriented sign is never relabeled as global inside/outside occupancy.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ComponentGeometryTrainingTargets:
    query_xyz: np.ndarray
    unsigned_distance: np.ndarray
    unsigned_valid: np.ndarray
    signed_query_index: np.ndarray
    signed_distance: np.ndarray
    occupancy: np.ndarray
    surface_xyz: np.ndarray
    surface_normal: np.ndarray
    surface_normal_valid: np.ndarray

    def validate(self) -> None:
        q = np.asarray(self.query_xyz)
        d = np.asarray(self.unsigned_distance)
        valid = np.asarray(self.unsigned_valid)
        index = np.asarray(self.signed_query_index)
        signed = np.asarray(self.signed_distance)
        occupancy = np.asarray(self.occupancy)
        sx = np.asarray(self.surface_xyz)
        sn = np.asarray(self.surface_normal)
        nv = np.asarray(self.surface_normal_valid)

        if q.ndim != 2 or q.shape[1:] != (3,) or not np.isfinite(q).all():
            raise ValueError("IRIS_GEOMETRY_TARGET_QUERY_LAYOUT")
        if d.shape != (len(q),) or valid.shape != (len(q),):
            raise ValueError("IRIS_GEOMETRY_TARGET_UNSIGNED_LAYOUT")
        if not np.isfinite(d).all() or np.any(d < 0):
            raise ValueError("IRIS_GEOMETRY_TARGET_UNSIGNED_VALUE")
        if valid.dtype != np.bool_ or not np.any(valid):
            raise ValueError("IRIS_GEOMETRY_TARGET_UNSIGNED_VALIDITY")
        if index.ndim != 1 or signed.shape != index.shape or occupancy.shape != index.shape:
            raise ValueError("IRIS_GEOMETRY_TARGET_SIGN_LAYOUT")
        if len(index):
            if np.any(index < 0) or np.any(index >= len(q)):
                raise ValueError("IRIS_GEOMETRY_TARGET_SIGN_INDEX_RANGE")
            if len(np.unique(index)) != len(index) or not np.all(np.diff(index) > 0):
                raise ValueError("IRIS_GEOMETRY_TARGET_SIGN_INDEX_ORDER")
            if not np.all(valid[index]):
                raise ValueError("IRIS_GEOMETRY_TARGET_SIGN_ON_INVALID_UNSIGNED")
            if not np.isfinite(signed).all():
                raise ValueError("IRIS_GEOMETRY_TARGET_SIGN_VALUE")
        if occupancy.dtype != np.bool_:
            raise ValueError("IRIS_GEOMETRY_TARGET_OCCUPANCY_DTYPE")
        if sx.ndim != 2 or sx.shape[1:] != (3,) or sn.shape != sx.shape or nv.shape != (len(sx),):
            raise ValueError("IRIS_GEOMETRY_TARGET_SURFACE_LAYOUT")
        if not np.isfinite(sx).all() or not np.isfinite(sn).all():
            raise ValueError("IRIS_GEOMETRY_TARGET_SURFACE_FINITE")
        if nv.dtype != np.bool_ or not np.any(nv):
            raise ValueError("IRIS_GEOMETRY_TARGET_NORMAL_VALIDITY")
        length = np.linalg.norm(sn[nv], axis=1)
        if not np.allclose(length, 1.0, atol=2.0e-4, rtol=0):
            raise ValueError("IRIS_GEOMETRY_TARGET_NORMAL_UNIT")

    def global_sign_rows(self) -> dict[str, np.ndarray]:
        """Return only globally sign-authoritative rows; never dense-fill unknowns."""
        self.validate()
        index = np.asarray(self.signed_query_index, dtype=np.int64)
        return {
            "query_index": index,
            "query_xyz": np.asarray(self.query_xyz)[index],
            "signed_distance": np.asarray(self.signed_distance),
            "occupancy": np.asarray(self.occupancy),
        }

    def local_oriented_brackets(self, epsilons) -> dict[str, np.ndarray]:
        """Derive local oriented surface coordinates, not global occupancy labels."""
        self.validate()
        eps = np.asarray(tuple(epsilons), dtype=np.float64)
        if eps.ndim != 1 or not len(eps) or not np.isfinite(eps).all() or np.any(eps <= 0):
            raise ValueError("IRIS_GEOMETRY_TARGET_BRACKET_EPSILON")
        if len(np.unique(eps)) != len(eps):
            raise ValueError("IRIS_GEOMETRY_TARGET_BRACKET_EPSILON_DUPLICATE")
        valid_ids = np.flatnonzero(self.surface_normal_valid)
        xyz = np.asarray(self.surface_xyz, dtype=np.float64)[valid_ids]
        normal = np.asarray(self.surface_normal, dtype=np.float64)[valid_ids]
        base = xyz[:, None, :]
        delta = normal[:, None, :] * eps[None, :, None]
        plus = base + delta
        minus = base - delta
        query = np.concatenate((minus, plus), axis=1).reshape(-1, 3)
        value = np.concatenate((-np.broadcast_to(eps, (len(xyz), len(eps))), np.broadcast_to(eps, (len(xyz), len(eps)))), axis=1).reshape(-1)
        side = value > 0
        source_surface_row = np.repeat(valid_ids, 2 * len(eps)).astype(np.int64)
        return {
            "query_xyz": query.astype(np.float32),
            "local_oriented_sdf": value.astype(np.float32),
            "positive_normal_side": side.astype(np.bool_),
            "source_surface_row": source_surface_row,
            "semantics": np.asarray(["LOCAL_ORIENTED_SURFACE_COORDINATE__NOT_GLOBAL_OCCUPANCY"]),
        }


def load_component_geometry_training_targets(distance_sign_npz, surface_npz) -> ComponentGeometryTrainingTargets:
    with np.load(distance_sign_npz, allow_pickle=False) as z:
        required = {
            "query_xyz_canonical",
            "unsigned_distance",
            "unsigned_distance_valid",
            "signed_query_index",
            "signed_distance_value",
            "occupancy_value",
        }
        if not required.issubset(z.files):
            raise ValueError("IRIS_GEOMETRY_TARGET_DISTANCE_ARRAYS")
        forbidden = {"signed_distance", "occupancy", "sign_valid", "occupancy_valid"}
        if forbidden.intersection(z.files):
            raise ValueError("IRIS_GEOMETRY_TARGET_LEGACY_DENSE_SIGN_FORBIDDEN")
        query_xyz = np.asarray(z["query_xyz_canonical"], dtype=np.float32)
        unsigned_distance = np.asarray(z["unsigned_distance"], dtype=np.float32)
        unsigned_valid = np.asarray(z["unsigned_distance_valid"], dtype=np.bool_)
        signed_query_index = np.asarray(z["signed_query_index"], dtype=np.int64)
        signed_distance = np.asarray(z["signed_distance_value"], dtype=np.float32)
        occupancy = np.asarray(z["occupancy_value"], dtype=np.bool_)
    with np.load(surface_npz, allow_pickle=False) as z:
        required = {"surface_xyz_canonical", "normal_canonical", "normal_valid"}
        if not required.issubset(z.files):
            raise ValueError("IRIS_GEOMETRY_TARGET_SURFACE_ARRAYS")
        surface_xyz = np.asarray(z["surface_xyz_canonical"], dtype=np.float32)
        surface_normal = np.asarray(z["normal_canonical"], dtype=np.float32)
        normal_valid = np.asarray(z["normal_valid"], dtype=np.bool_)
    target = ComponentGeometryTrainingTargets(
        query_xyz=query_xyz,
        unsigned_distance=unsigned_distance,
        unsigned_valid=unsigned_valid,
        signed_query_index=signed_query_index,
        signed_distance=signed_distance,
        occupancy=occupancy,
        surface_xyz=surface_xyz,
        surface_normal=surface_normal,
        surface_normal_valid=normal_valid,
    )
    target.validate()
    return target
