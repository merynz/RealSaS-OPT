from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SignedZeroSurfaceDecoderConfigV2:
    """Experimental cleanroom decoder for a signed 3D field.

    Product policy: a mesh is allowed as an intermediate or persistent geometry
    substrate. Canonical promotion is not implied. The scientific contract is
    simply that the scene-first latent is queried on a bounded 3D grid and the
    zero-level set is extracted without any per-ray argmin decoder.
    """

    grid_resolution: int = 256
    iso_level: float = 0.0
    query_chunk: int = 262_144

    def __post_init__(self) -> None:
        if int(self.grid_resolution) < 32:
            raise ValueError("grid_resolution must be >= 32")
        if int(self.query_chunk) < 1:
            raise ValueError("query_chunk must be positive")
        if not np.isfinite(float(self.iso_level)):
            raise ValueError("iso_level must be finite")


def marching_cubes_zero_surface_v2(
    sdf_zyx: np.ndarray,
    *,
    bbox_min_xyz: np.ndarray,
    bbox_max_xyz: np.ndarray,
    iso_level: float = 0.0,
):
    """Extract a triangle zero-surface from a dense signed field.

    `sdf_zyx` is indexed [z,y,x]. Returned vertices/normals are xyz in the
    caller's bbox coordinates. This function intentionally does not encode any
    renderer, view identity, teacher labels, rig labels, or product topology.
    """

    from skimage import measure

    sdf = np.asarray(sdf_zyx, dtype=np.float32)
    if sdf.ndim != 3 or min(sdf.shape) < 2:
        raise ValueError("sdf_zyx must be a non-degenerate 3D array")
    if not np.isfinite(sdf).all():
        raise ValueError("sdf_zyx contains non-finite values")
    level = float(iso_level)
    if not (float(sdf.min()) <= level <= float(sdf.max())):
        raise ValueError("signed field does not bracket requested iso-level")

    lo = np.asarray(bbox_min_xyz, dtype=np.float32).reshape(3)
    hi = np.asarray(bbox_max_xyz, dtype=np.float32).reshape(3)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
        raise ValueError("invalid bbox")

    rz, ry, rx = sdf.shape
    spacing_zyx = (
        float((hi[2] - lo[2]) / (rz - 1)),
        float((hi[1] - lo[1]) / (ry - 1)),
        float((hi[0] - lo[0]) / (rx - 1)),
    )
    v_zyx, faces, n_zyx, values = measure.marching_cubes(
        sdf,
        level=level,
        spacing=spacing_zyx,
        allow_degenerate=False,
    )
    vertices = np.stack(
        [lo[0] + v_zyx[:, 2], lo[1] + v_zyx[:, 1], lo[2] + v_zyx[:, 0]],
        axis=-1,
    ).astype(np.float32)
    normals = np.stack([n_zyx[:, 2], n_zyx[:, 1], n_zyx[:, 0]], axis=-1).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    return vertices, np.asarray(faces, dtype=np.int64), normals, np.asarray(values, dtype=np.float32)


def write_obj_v2(path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write only geometry; no UV/material assumptions."""

    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("triangle mesh arrays required")
    with p.open("w", encoding="utf-8") as out:
        out.write("# RealSaS cleanroom signed zero-surface V2\n")
        for x, y, z in v:
            out.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in f + 1:
            out.write(f"f {int(a)} {int(b)} {int(c)}\n")
