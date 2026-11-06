import numpy as np

from compiler.realsas_compiler_core.single_m_caa_v1 import render_m_depth
from tools.research.presentation_order_renderer_v1 import (
    render_m_depth_with_presentation_order,
    stable_fragment_order,
)


def test_stable_fragment_order_preserves_unconstrained_geometric_order():
    order, cycle = stable_fragment_order(
        ["OWNER:body:00", "OWNER:cape:00", "OWNER:sword:00"],
        [("OWNER:cape:00", "OWNER:body:00")],
    )
    assert not cycle
    assert order.tolist() == [1, 0, 2]


def test_stable_fragment_order_cycle_falls_back():
    order, cycle = stable_fragment_order(
        ["OWNER:body:00", "OWNER:cape:00"],
        [
            ("OWNER:cape:00", "OWNER:body:00"),
            ("OWNER:body:00", "OWNER:cape:00"),
        ],
    )
    assert cycle
    assert order.tolist() == [0, 1]


def _scene():
    # Two identical triangles. Face 1 is geometrically nearer than face 0.
    positions = np.array(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 0.0], [3.0, 0.0], [0.0, 3.0]],
        dtype=float,
    )
    depths = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    uv = np.zeros_like(positions)
    texture = np.zeros((4, 4, 4), dtype=np.uint8)

    def sample_material(fi, bary):
        fi = np.asarray(fi)
        row = np.zeros((len(fi), 4), dtype=float)
        row[:, 3] = 1.0
        row[fi == 0, 0] = 1.0  # cape red
        row[fi == 1, 1] = 1.0  # body green
        sample_material.provenance_codes = np.zeros(len(fi), dtype=np.uint8)
        return row

    return positions, depths, faces, uv, texture, sample_material


def test_no_active_edge_is_bit_exact_to_canonical():
    p, z, f, uv, tex, sampler = _scene()
    canonical = render_m_depth(
        positions=p, depths=z, faces=f, uv=uv, texture=tex, resolution=4, sample_material=sampler
    )
    court = render_m_depth_with_presentation_order(
        positions=p,
        depths=z,
        faces=f,
        uv=uv,
        texture=tex,
        resolution=4,
        sample_material=sampler,
        face_owner_ids=np.array(["OWNER:cape:00", "OWNER:body:00"], dtype=object),
        precedence_edges=[],
    )
    assert np.array_equal(canonical.straight_rgba_u8, court.straight_rgba_u8)
    assert np.array_equal(canonical.owner_face_index, court.owner_face_index)
    assert np.array_equal(canonical.geometry_visible, court.geometry_visible)
    assert court.presentation_override_pixel_count == 0


def test_active_edge_changes_presentation_not_geometry_or_alpha():
    p, z, f, uv, tex, sampler = _scene()
    canonical = render_m_depth(
        positions=p, depths=z, faces=f, uv=uv, texture=tex, resolution=4, sample_material=sampler
    )
    court = render_m_depth_with_presentation_order(
        positions=p,
        depths=z,
        faces=f,
        uv=uv,
        texture=tex,
        resolution=4,
        sample_material=sampler,
        face_owner_ids=np.array(["OWNER:cape:00", "OWNER:body:00"], dtype=object),
        precedence_edges=[("OWNER:cape:00", "OWNER:body:00")],
    )
    assert court.presentation_override_pixel_count > 0
    assert np.array_equal(canonical.geometry_visible, court.geometry_visible)
    assert np.array_equal(canonical.final_alpha, court.final_alpha)
    assert np.array_equal(canonical.owner_face_index, court.geometric_owner_face_index)
    changed = np.any(canonical.straight_rgba_u8 != court.straight_rgba_u8, axis=2)
    assert np.any(changed)
    assert np.all(changed <= court.presentation_override_mask)
