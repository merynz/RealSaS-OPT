from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation, QualificationError
from compiler.realsas_compiler_core.mwb2 import build_mwb2_candidate, qualify_mwb2_mesh
from compiler.realsas_compiler_core.appearance import build_observed_appearance_binding
import pytest


RESOLUTION = 1024


def _grid_to_pixel_center(x, y):
    return ((float(x) + 1.0) * 0.5 * RESOLUTION - 0.5, (float(y) + 1.0) * 0.5 * RESOLUTION - 0.5)


def surface():
    pts = {'a': (-.4, -.4, 0.), 'b': (.4, -.4, 0.), 'c': (.4, .4, 0.)}
    nodes = tuple(
        SurfaceNode(
            s,
            p,
            (0, 1),
            ('p',),
            ('o',),
            ((0, _grid_to_pixel_center(p[0], p[1])), (1, _grid_to_pixel_center(p[0] * .9, p[1] * .9))),
            f'pg-{s}',
        )
        for s, p in pts.items()
    )
    rel = tuple(SurfaceRelation(f'r{i}', a, b, 'LOCAL_NEIGHBOR', 1., {}) for i, (a, b) in enumerate((('a', 'b'), ('b', 'c'), ('a', 'c'))))
    return RiggingSurfaceIR(nodes, rel, 'S-HASH', metadata={'raster_coordinate_system': 'PIXEL_CENTER_XY', 'resolution': RESOLUTION})


def observation_hashes():
    return {view: f'im{view}' for view in range(8)}


def test_master_observed_appearance_exact_local_coverage():
    s = surface()
    m = qualify_mwb2_mesh(s, build_mwb2_candidate(s, view_index=0, camera_binding_hash='cam0'))
    hashes = observation_hashes()
    a = build_observed_appearance_binding(
        surface=s,
        mesh=m,
        target_view_index=0,
        camera_binding_hash='cam0',
        observation_hash_by_view=hashes,
    )
    assert len(a.corner_bindings) == sum(len(f) for f in m.faces)
    assert all(c.authority_class == 'OBSERVED_LOCAL' and c.donor_view_index == 0 for c in a.corner_bindings)
    assert all(0.0 <= c.material_uv[0] <= 1.0 and 0.0 <= c.material_uv[1] <= 1.0 for c in a.corner_bindings)
    assert a.metadata['camera_refit'] is False and a.metadata['source_mesh_uv_used'] is False and len(hashes) == 8
    assert a.metadata['raster_coordinate_system'] == 'PIXEL_CENTER_XY' and a.metadata['raster_resolution'] == 1024


def test_missing_observation_hash_fails_closed():
    s = surface(); m = qualify_mwb2_mesh(s, build_mwb2_candidate(s, view_index=0, camera_binding_hash='cam0'))
    with pytest.raises(QualificationError, match='MISSING_DONOR_OBSERVATION_HASH'):
        build_observed_appearance_binding(surface=s, mesh=m, target_view_index=0, camera_binding_hash='cam0', observation_hash_by_view={})
