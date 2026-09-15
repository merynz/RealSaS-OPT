from compiler.realsas_compiler_core.proof_mesh_authority_v8 import (
    _EXTERNAL_RASTER_AUTHORITY,
    _CANONICAL_SUPPORT_AREA_AUTHORITY,
    _authoritative_area_summary,
)


def test_external_render_support_uses_raster_area_for_product_nondegeneracy():
    summary = _authoritative_area_summary(
        support_area={"face_count": 4, "degenerate_faces": 4, "min_area": 0.0},
        raster={"face_count": 4, "degenerate_faces": 0, "min_raster_triangle_area": 2.5},
        external=True,
    )
    assert summary == {
        "face_count": 4,
        "degenerate_faces": 0,
        "min_area": 2.5,
        "mesh_area_authority": _EXTERNAL_RASTER_AUTHORITY,
    }


def test_canonical_mesh_keeps_support_space_area_authority():
    summary = _authoritative_area_summary(
        support_area={"face_count": 2, "degenerate_faces": 1, "min_area": 0.0},
        raster={"face_count": 2, "degenerate_faces": 0, "min_raster_triangle_area": 10.0},
        external=False,
    )
    assert summary == {
        "face_count": 2,
        "degenerate_faces": 1,
        "min_area": 0.0,
        "mesh_area_authority": _CANONICAL_SUPPORT_AREA_AUTHORITY,
    }
