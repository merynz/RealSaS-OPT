from __future__ import annotations

import numpy as np

from tools.corpus import build_truth139_graphics_layer_peel_v3_rescue as rescue


def test_guarded_prefix_rejects_exact_outside_candidate_and_recovers_deeper_hit() -> None:
    # Resolution=2, pixel (0,0) -> PyTorch3D NDC centre (+0.5,+0.5).
    # face 0 covers it at depth 1, face 1 is a float-raster-style false edge
    # candidate outside the exact pixel centre, face 2 covers it at depth 2.
    camera = {
        "origin": [0.0, 0.0, -1.0],
        "forward": [0.0, 0.0, 1.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "half_extent": 1.0,
        "resolution": 2,
        "view_index": 0,
    }

    # world x is negated by the PyTorch3D projection convention.
    def world(ndc_x: float, ndc_y: float, z: float) -> list[float]:
        return [-ndc_x, ndc_y, z]

    vertices = np.asarray(
        [
            world(0.2, 0.2, 0.0), world(0.8, 0.2, 0.0), world(0.5, 0.9, 0.0),
            world(0.6, 0.6, 0.5), world(0.9, 0.6, 0.5), world(0.6, 0.9, 0.5),
            world(0.2, 0.2, 1.0), world(0.8, 0.2, 1.0), world(0.5, 0.9, 1.0),
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)

    raw_face = np.full((2, 2, 4), -1, dtype=np.int32)
    raw_face[0, 0, :3] = [0, 1, 2]

    face, depth, bary, overflow = rescue._float64_surface_address_guarded(
        raw_face,
        vertices,
        faces,
        camera,
        output_k=2,
    )

    assert face[0, 0].tolist() == [0, 2]
    assert np.allclose(depth[0, 0], [1.0, 2.0])
    assert np.allclose(bary[0, 0].sum(axis=-1), [1.0, 1.0])
    assert bool(overflow[0, 0]) is False


def test_guarded_prefix_preserves_unknown_when_more_than_k_exact_hits_survive() -> None:
    camera = {
        "origin": [0.0, 0.0, -1.0],
        "forward": [0.0, 0.0, 1.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "half_extent": 1.0,
        "resolution": 2,
        "view_index": 0,
    }

    def world(ndc_x: float, ndc_y: float, z: float) -> list[float]:
        return [-ndc_x, ndc_y, z]

    vertices = []
    faces = []
    for index, z in enumerate((0.0, 0.5, 1.0)):
        base = len(vertices)
        vertices.extend([world(0.2, 0.2, z), world(0.8, 0.2, z), world(0.5, 0.9, z)])
        faces.append([base, base + 1, base + 2])
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    raw_face = np.full((2, 2, 4), -1, dtype=np.int32)
    raw_face[0, 0, :3] = [0, 1, 2]
    face, _depth, _bary, overflow = rescue._float64_surface_address_guarded(
        raw_face, vertices, faces, camera, output_k=2
    )
    assert face[0, 0].tolist() == [0, 1]
    assert bool(overflow[0, 0]) is True
