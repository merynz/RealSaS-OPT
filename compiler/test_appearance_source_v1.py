from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation, QualificationError
from compiler.realsas_compiler_core.mwb2 import build_mwb2_candidate, qualify_mwb2_mesh
from compiler.realsas_compiler_core.appearance import build_observed_appearance_binding
from experiments.single_family_e2e_v1.data_manifest_v1 import MasterViewBindingV1, MasterFamilyManifestV1
from experiments.single_family_e2e_v1.appearance_builder_v1 import observation_hashes_from_master_manifest, build_master_observed_appearance_v1
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


def manifest():
    views = tuple(MasterViewBindingV1(v, f'renders/V{v}/textured.png', f'im{v}', f'renders/V{v}/camera.json', f'cam{v}', f'renders/V{v}/raster_authority.npz', f'ras{v}', 45. * v, (1024, 1024)) for v in range(8))
    return MasterFamilyManifestV1('asset', 'textured.png', 'primary_geometry.npz', 'geom', views, (), 'manifest')


def test_master_observed_appearance_exact_local_coverage():
    s = surface(); m = qualify_mwb2_mesh(s, build_mwb2_candidate(s, view_index=0, camera_binding_hash='cam0')); a = build_master_observed_appearance_v1(manifest=manifest(), surface=s, mesh=m, target_view_index=0)
    assert len(a.corner_bindings) == sum(len(f) for f in m.faces); assert all(c.authority_class == 'OBSERVED_LOCAL' and c.donor_view_index == 0 for c in a.corner_bindings)
    assert all(0.0 <= c.material_uv[0] <= 1.0 and 0.0 <= c.material_uv[1] <= 1.0 for c in a.corner_bindings)
    assert a.metadata['camera_refit'] is False and a.metadata['source_mesh_uv_used'] is False and len(observation_hashes_from_master_manifest(manifest())) == 8
    assert a.metadata['raster_coordinate_system'] == 'PIXEL_CENTER_XY' and a.metadata['raster_resolution'] == 1024


def test_missing_observation_hash_fails_closed():
    s = surface(); m = qualify_mwb2_mesh(s, build_mwb2_candidate(s, view_index=0, camera_binding_hash='cam0'))
    with pytest.raises(QualificationError, match='MISSING_DONOR_OBSERVATION_HASH'):
        build_observed_appearance_binding(surface=s, mesh=m, target_view_index=0, camera_binding_hash='cam0', observation_hash_by_view={})
