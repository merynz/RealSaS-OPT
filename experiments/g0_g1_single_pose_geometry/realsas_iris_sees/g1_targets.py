from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np


G1_KEYS = ("family_id", "P_A", "N_A", "XY_A", "V_A", "direct_obs_A")
G1_SIDECAR_KEYS = (
    "family_id", "camera_center", "camera_half_extent",
    "surface_points_A", "surface_normals_A", "surface_xy_A", "surface_visibility_A",
)


def observation_target_to_g1(target: Dict[str, Any]) -> Dict[str, Any]:
    """Non-destructive projection of an existing canonical observation target onto G1."""
    missing = [k for k in G1_KEYS if k not in target]
    if missing:
        raise KeyError(f"canonical target missing G1 fields: {missing}")
    out = {k: target[k] for k in G1_KEYS}
    out["schema"] = np.array("RealSaS.SurfaceEvidence.G1.Target.v1")
    return out


def build_g1_target_from_sidecar(sidecar_path: str | Path) -> Dict[str, np.ndarray]:
    """Build G1 truth by reading Pose-A observable fields only.

    This function intentionally never indexes any Pose-B, scene-flow, differential,
    intervention, skeleton, owner, or mechanical field even if the source sidecar
    contains them. It is the canonical G1 cache-construction boundary.
    """
    with np.load(sidecar_path, allow_pickle=False) as z:
        missing = [k for k in G1_SIDECAR_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"G1 sidecar missing required Pose-A fields: {missing}")
        fid = int(z["family_id"])
        center = z["camera_center"].astype(np.float64)
        scale = 2.0 * float(z["camera_half_extent"])
        P = (z["surface_points_A"].astype(np.float64) - center) / scale
        N = z["surface_normals_A"].astype(np.float64)
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        V = (z["surface_visibility_A"] == 2)
        direct = V.any(0)
        return {
            "schema": np.array("RealSaS.SurfaceEvidence.G1.Target.v1"),
            "family_id": np.array(fid, np.int32),
            "P_A": P.astype(np.float32),
            "N_A": N.astype(np.float32),
            "XY_A": z["surface_xy_A"].astype(np.float32),
            "V_A": V.astype(np.uint8),
            "direct_obs_A": direct.astype(np.uint8),
        }


def save_g1_target_from_sidecar(sidecar_path: str | Path, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **build_g1_target_from_sidecar(sidecar_path))
    return output_path


def load_g1_target(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in G1_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"G1 target missing fields: {missing}")
        return {k: z[k] for k in ("schema",) + G1_KEYS if k in z.files}
