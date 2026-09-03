from dataclasses import replace
from pathlib import Path

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.export.current_v4_runtime_v2 import (
    _runtime_uv_from_local_corner,
    project_current_v4_proof_bakes_to_runtime_v2,
)
from experiments.single_family_e2e_v1.run_complete_e2e_v1 import run_complete_e2e_v1


def test_complete_e2e_native_projection_is_bound_to_proof_bake_and_exact_raster_uv(tmp_path):
    out = run_complete_e2e_v1(tmp_path / 'bundle')
    projection = out['native_projection']
    bake = out['motion_bakes'][0]
    product = out['product']
    textures = out['texture_bindings']

    assert Path(out['native_runtime_archive']).is_file()
    assert out['native_runtime_result']['view_count'] == 8
    assert out['native_runtime_result']['clip_count'] == 1
    assert len(projection.projection_hash) == 64
    assert projection.source_motion_bake_hashes == {bake.clip_id: bake.bake_hash}

    # Static runtime XY must be the exact proof-owned bake rest XY, duplicated only
    # for UV seams. Mechanical mesh P.xy is not an allowed runtime-position source.
    rest = dict(bake.rest_mesh_vertices_by_id)
    for view in projection.views:
        for mesh in view.meshes:
            source_indices = projection.runtime_vertex_sources[mesh.mesh_id]
            expected_xy = tuple(rest[mesh.mesh_id][i] for i in source_indices)
            actual_xy = tuple((vertex[0], vertex[1]) for vertex in mesh.xyuv)
            assert actual_xy == expected_xy

    # Native C++ samples u*(W-1), v*(H-1). Runtime UV therefore comes from the
    # top-left donor raster directly, not from material_uv or a blind 1-v flip.
    direction = product.directional_renderables.directions[0]
    component = direction.components[0]
    texture = textures[0]
    corner = component.appearance.corner_bindings[0]
    expected_uv = (
        corner.donor_raster_xy[0] / float(texture.width - 1),
        corner.donor_raster_xy[1] / float(texture.height - 1),
    )
    actual_uv = _runtime_uv_from_local_corner(corner, texture)
    assert actual_uv == pytest.approx(expected_uv, abs=1e-12)
    runtime_uvs = {(vertex[2], vertex[3]) for vertex in projection.views[0].meshes[0].xyuv}
    assert any(abs(u - actual_uv[0]) < 1e-12 and abs(v - actual_uv[1]) < 1e-12 for u, v in runtime_uvs)


def test_runtime_projection_rejects_missing_or_tampered_proof_bake(tmp_path):
    out = run_complete_e2e_v1(tmp_path / 'bundle')
    with pytest.raises(QualificationError, match='PROVEN_BAKE_CLIP_SET_MISMATCH'):
        project_current_v4_proof_bakes_to_runtime_v2(
            product=out['product'],
            proof_bundle=out['proof'],
            motion_bakes=(),
            texture_bindings=out['texture_bindings'],
        )

    tampered = replace(out['motion_bakes'][0], bake_hash='0' * 64)
    with pytest.raises(ValueError, match='MOTION_BAKE_HASH_MISMATCH'):
        project_current_v4_proof_bakes_to_runtime_v2(
            product=out['product'],
            proof_bundle=out['proof'],
            motion_bakes=(tampered,),
            texture_bindings=out['texture_bindings'],
        )


def test_runtime_projection_rejects_texture_lineage_and_cross_view_atlas_shortcut(tmp_path):
    out = run_complete_e2e_v1(tmp_path / 'bundle')
    bad_texture = replace(out['texture_bindings'][0], atlas_payload_hash='0' * 64)
    with pytest.raises(QualificationError, match='TEXTURE_ATLAS_LINEAGE_MISMATCH'):
        project_current_v4_proof_bakes_to_runtime_v2(
            product=out['product'],
            proof_bundle=out['proof'],
            motion_bakes=out['motion_bakes'],
            texture_bindings=(bad_texture, *out['texture_bindings'][1:]),
        )

    corner = out['product'].directional_renderables.directions[0].components[0].appearance.corner_bindings[0]
    cross_view = replace(corner, donor_view_index=1, authority_class='OBSERVED_CROSS_VIEW')
    with pytest.raises(QualificationError, match='CROSS_VIEW_OR_COMPLETION_ATLAS_NOT_QUALIFIED'):
        _runtime_uv_from_local_corner(cross_view, out['texture_bindings'][0])


def test_runtime_projection_source_forbids_mechanical_xy_and_solver_replay():
    source = Path('compiler/realsas_compiler_services/export/current_v4_runtime_v2.py').read_text(encoding='utf-8')
    assert 'vertex.P' not in source
    assert 'apply_lbs' not in source
    assert 'directional_motion_evaluator' not in source
    assert 'runtime_rest_xy_authority' in source
    assert 'QUALIFICATION_OWNED_MOTION_BAKE_REST' in source
