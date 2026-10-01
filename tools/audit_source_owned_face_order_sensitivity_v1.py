from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from compiler.realsas_compiler_services.orchestrator.adapters.runtime_v2 import (
    _source_owned_visual_reference_frame,
)


def sha_rgba(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.uint8).tobytes()).hexdigest()


def render(texture: Path, faces: np.ndarray):
    source_size = 8
    resolution = 16
    # Two coincident visual triangles. Vertices 0..2 sample red; 3..5 sample blue.
    positions = np.asarray(
        [
            (1.0, 1.0), (6.0, 1.0), (1.0, 6.0),
            (1.0, 1.0), (6.0, 1.0), (1.0, 6.0),
        ],
        dtype=np.float64,
    )
    uv = np.asarray(
        [
            (0.10, 0.50), (0.10, 0.50), (0.10, 0.50),
            (0.90, 0.50), (0.90, 0.50), (0.90, 0.50),
        ],
        dtype=np.float64,
    )
    arrays = {
        "view_0_uv": uv,
        "view_0_faces": np.asarray(faces, dtype=np.int64),
        "clip_0_view_0_positions": positions.reshape(1, 6, 2),
    }
    view = SimpleNamespace(
        view_index=0,
        visual_vertex_count=6,
        visual_face_count=2,
        source_width=source_size,
        source_height=source_size,
        texture_path=str(texture),
        camera={"resolution": resolution},
    )
    clip = SimpleNamespace(array_prefix="clip_0")
    return _source_owned_visual_reference_frame(
        None, arrays, clip=clip, view=view, frame_index=0
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    texture = a.out.parent / "face_order_texture.png"
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :4] = (255, 0, 0, 255)
    rgba[:, 4:] = (0, 0, 255, 255)
    Image.fromarray(rgba, mode="RGBA").save(texture)

    order_a = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64)
    order_b = np.asarray(((3, 4, 5), (0, 1, 2)), dtype=np.int64)
    a_frame = render(texture, order_a)
    b_frame = render(texture, order_b)

    visible = (a_frame.straight_rgba_u8[..., 3] > 0) & (b_frame.straight_rgba_u8[..., 3] > 0)
    changed = np.any(a_frame.straight_rgba_u8 != b_frame.straight_rgba_u8, axis=2)
    changed_visible = visible & changed
    ys, xs = np.nonzero(changed_visible)
    if not len(xs):
        raise SystemExit("FACE_ORDER_SENSITIVITY_NOT_OBSERVED")

    y, x = int(ys[len(ys)//2]), int(xs[len(xs)//2])
    report = {
        "schema": "RealSaS.SourceOwnedVisualFaceOrderSensitivityAudit.v1",
        "status": "ORDER_SENSITIVE",
        "only_intervention": "REVERSE_FACE_ARRAY_ORDER",
        "geometry_identical": True,
        "texture_identical": True,
        "positions_identical": True,
        "uv_by_vertex_identical": True,
        "changed_visible_pixel_count": int(np.count_nonzero(changed_visible)),
        "order_a_rgba_sha256": sha_rgba(a_frame.straight_rgba_u8),
        "order_b_rgba_sha256": sha_rgba(b_frame.straight_rgba_u8),
        "sample_pixel_xy": [x, y],
        "order_a_sample_rgba": a_frame.straight_rgba_u8[y, x].tolist(),
        "order_b_sample_rgba": b_frame.straight_rgba_u8[y, x].tolist(),
        "order_a_sample_owner": int(a_frame.owner_face_index[y, x]),
        "order_b_sample_owner": int(b_frame.owner_face_index[y, x]),
        "interpretation": (
            "Source-owned visual output changes when only triangle array order changes. "
            "The current renderer therefore uses face iteration order as an implicit "
            "presentation/overlap authority."
        ),
    }
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
