from __future__ import annotations

"""Two observed source silhouettes vs base/rest TESSA, never release proof."""

from dataclasses import asdict, replace
import numpy as np
import pytest

from compiler.realsas_compiler_core.camera_geometry_v2 import (
    CameraProjectionV3, project_points_xyz_v3,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    rasterize_triangles_half_integer_top_left,
)
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1, seal_inference_evidence_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_evidence_binding_v1 import (
    validate_rss_vnext_source_evidence_binding_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_observation_v1 import (
    canonicalize_two_view_source_v1,
)
from compiler.realsas_compiler_core.rss_vnext_two_view_raster_diagnostic_v1 import (
    compile_rss_vnext_two_view_raster_diagnostic_v1,
    validate_rss_vnext_two_view_raster_diagnostic_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.rss_vnext.inference_evidence_export_v1 import _source_tensor_digest
from tests.models.test_rss_vnext_inference_checkpoint_bridge_v1 import _mesh


def _parents(*, height=14, width=20):
    mesh = _mesh()
    vertices = [[-.45,-.45,0.],[.45,-.45,0.],[0.,.45,0.]]
    mesh["base_xyz"] = vertices
    mesh["rest_xyz"] = vertices
    images = np.zeros((2,4,height,width),dtype=np.uint8)
    images[:,0] = 143
    cameras = np.zeros((2,13),dtype=np.float32)
    cameras[0,[3,7,11,12]] = 1.
    cameras[1,[3,7,12]] = [-1.,1.,1.]
    cameras[1,11] = -1.
    for view in (0,1):
        cam = CameraProjectionV3(
            view_id=("FRONT","BACK")[view], view_index=view,
            origin=(0.,0.,0.),
            right=(1.,0.,0.) if view==0 else (-1.,0.,0.),
            screen_up=(0.,1.,0.),
            forward=(0.,0.,1.) if view==0 else (0.,0.,-1.),
            half_extent=1., resolution=width,
        )
        xy=project_points_xyz_v3(np.asarray(vertices),cam)
        xy[:,1]*=height/width
        mask=rasterize_triangles_half_integer_top_left(
            [tuple(tuple(map(float,xy[i,:2])) for i in (0,1,2))],
            width=width,height=height,
        )
        images[view,3]=np.frombuffer(mask,dtype=np.uint8).reshape(height,width)*255
    source=canonicalize_two_view_source_v1(
        images,cameras,source_rights_claim_sha256="a"*64,
    )
    rgba,cams=source.to_model_tensors()
    evidence=seal_inference_evidence_v1(RSSvNextInferenceEvidenceIRV1(
        checkpoint_sha256="b"*64,
        observed_front_sha256=_source_tensor_digest(rgba[0,0]),
        observed_back_sha256=_source_tensor_digest(rgba[0,1]),
        camera_binding_sha256=_source_tensor_digest(cams),
        mesh_status="UNQUALIFIED_MESH_CANDIDATE",mesh=mesh,
    ))
    bound=validate_rss_vnext_source_evidence_binding_v1(
        source=source,evidence=evidence,
        expected_source_binding_sha256=source.source_binding_sha256,
    )
    return source,evidence,bound


def _measure(source,evidence,bound):
    return compile_rss_vnext_two_view_raster_diagnostic_v1(
        source=source,evidence=evidence,source_evidence_binding_court=bound,
    )


def test_original_two_source_views_match_base_and_rest_without_minting_authority():
    source,evidence,bound=_parents()
    before=evidence.to_dict()
    result=_measure(source,evidence,bound)
    validate_rss_vnext_two_view_raster_diagnostic_v1(result)
    assert result.state=="MEASURED_TWO_VIEW_SILHOUETTE_NOT_SOURCE_FIDELITY"
    assert len(result.views)==2
    for view in result.views:
        for variant in ("base","rest"):
            stats=view["geometry"][variant]["alpha_comparison"]["alpha_nonzero"]["metrics"]
            assert stats["recall"]==pytest.approx(1.)
            assert stats["precision"]==pytest.approx(1.)
            assert stats["silhouette_edge_p95_px"]==pytest.approx(0.)
    assert evidence.to_dict()==before
    assert not result.source_fidelity_qualified
    assert not result.depth_or_first_hit_qualified
    assert not result.source_color_owner_qualified
    assert not result.semantic_part_or_drawable_qualified
    assert not result.canonical_mesh_authority
    assert not result.animation_ready_authority


def test_tessa_refinement_changes_observed_silhouette_consequence():
    source,evidence,_=_parents()
    altered=dict(evidence.mesh)
    altered["rest_xyz"]=[[x+.28,y,z] for x,y,z in evidence.mesh["rest_xyz"]]
    changed=seal_inference_evidence_v1(replace(
        evidence,mesh=altered,evidence_hash="",
    ))
    court=validate_rss_vnext_source_evidence_binding_v1(
        source=source,evidence=changed,
        expected_source_binding_sha256=source.source_binding_sha256,
    )
    result=_measure(source,changed,court)
    for view in result.views:
        base=view["geometry"]["base"]["alpha_comparison"]["alpha_nonzero"]["metrics"]
        rest=view["geometry"]["rest"]["alpha_comparison"]["alpha_nonzero"]["metrics"]
        assert base["recall"]==pytest.approx(1.)
        assert rest["recall"]<1.
    assert source.view_pixel_sha256==_parents()[0].view_pixel_sha256


def test_topology_abstain_retains_source_without_fake_raster():
    source,evidence,_=_parents()
    empty=seal_inference_evidence_v1(replace(
        evidence,mesh=None,mesh_status="ABSTAIN",
        reason="TEST_NO_TOPOLOGY",evidence_hash="",
    ))
    court=validate_rss_vnext_source_evidence_binding_v1(
        source=source,evidence=empty,
        expected_source_binding_sha256=source.source_binding_sha256,
    )
    result=_measure(source,empty,court)
    assert result.state=="ABSTAIN_NO_TOPOLOGY"
    assert result.views==()
    assert not result.canonical_mesh_authority


def test_rehashed_source_binding_receipt_cannot_cross_another_parent():
    source,evidence,bound=_parents()
    fake=replace(bound,inference_evidence_hash="f"*64,court_hash="")
    payload=asdict(fake)
    payload.pop("court_hash")
    fake=replace(fake,court_hash=content_sha256(payload))
    with pytest.raises(QualificationError,match="SOURCE_EVIDENCE_COURT_PARENT_DRIFT"):
        _measure(source,evidence,fake)


def test_receipt_tamper_and_premature_fidelity_claim_fail():
    source,evidence,bound=_parents()
    result=_measure(source,evidence,bound)
    with pytest.raises(QualificationError,match="RECEIPT_PREMATURE_AUTHORITY"):
        validate_rss_vnext_two_view_raster_diagnostic_v1(replace(
            result,source_fidelity_qualified=True,
        ))
    with pytest.raises(QualificationError,match="RECEIPT_CONTENT_DRIFT"):
        validate_rss_vnext_two_view_raster_diagnostic_v1(replace(
            result,inference_evidence_hash="f"*64,
        ))



def test_two_view_silhouette_and_ray_evidence_cannot_uniquely_identify_3d_depth():
    source, evidence, bound = _parents(height=20, width=28)
    original = _measure(source, evidence, bound)
    original_witness = original.views[0]["two_view_source_depth_observability"]
    assert original_witness["effective_rank"] == 2
    assert original_witness["source_point_correspondence_authority"] == (
        "UNKNOWN_NOT_SOURCE_OBSERVED"
    )
    assert original_witness["counterfactual_status"] == (
        "DISTINCT_3D_GEOMETRY_SAME_TWO_SOURCE_PROJECTIONS"
    )
    assert original_witness["counterfactual_3d_vertex_displacement_world"] > 0
    assert original_witness["counterfactual_max_two_view_pixel_drift"] < 1e-6
    assert not original_witness["source_depth_or_surface_completion_authority"]
    assert not original_witness["canonical_mesh_authority"]

    # Real 3D shape changed substantially; BOTH independently observed
    # front/back masks still give perfect projected silhouette overlap.
    mesh = dict(evidence.mesh)
    mesh["rest_xyz"] = [
        [x, y, z + (0.2 if i % 2 == 0 else -0.2)]
        for i, (x, y, z) in enumerate(evidence.mesh["rest_xyz"])
    ]
    changed = seal_inference_evidence_v1(replace(
        evidence, mesh=mesh, evidence_hash="",
    ))
    rebound = validate_rss_vnext_source_evidence_binding_v1(
        source=source, evidence=changed,
        expected_source_binding_sha256=source.source_binding_sha256,
    )
    rest = _measure(source, changed, rebound)
    validate_rss_vnext_two_view_raster_diagnostic_v1(rest)
    for before, after in zip(original.views, rest.views):
        a = before["geometry"]["rest"]["binary_projected_raster_sha256"]
        b = after["geometry"]["rest"]["binary_projected_raster_sha256"]
        assert a == b
        measured = after["geometry"]["rest"]["alpha_comparison"]["alpha_nonzero"]["metrics"]
        assert measured["recall"] == pytest.approx(1.)
        assert measured["precision"] == pytest.approx(1.)
    assert original.original_mesh_sha256 != rest.original_mesh_sha256
    assert not rest.source_fidelity_qualified


def test_oblique_back_camera_improves_linear_rank_but_not_source_correspondence():
    source, evidence, _ = _parents()
    cameras = source.camera_features.copy()
    angle = .2
    from math import cos, sin
    cameras[1, 3:6] = [-cos(angle), 0., sin(angle)]
    cameras[1, 9:12] = [-sin(angle), 0., -cos(angle)]
    tilted = canonicalize_two_view_source_v1(
        source.native_rgba.copy(), cameras,
        source_rights_claim_sha256=source.source_rights_claim_sha256,
    )
    rgba, camera = tilted.to_model_tensors()
    reobserved = seal_inference_evidence_v1(replace(
        evidence,
        observed_front_sha256=_source_tensor_digest(rgba[0, 0]),
        observed_back_sha256=_source_tensor_digest(rgba[0, 1]),
        camera_binding_sha256=_source_tensor_digest(camera),
        evidence_hash="",
    ))
    court = validate_rss_vnext_source_evidence_binding_v1(
        source=tilted, evidence=reobserved,
        expected_source_binding_sha256=tilted.source_binding_sha256,
    )
    result = _measure(tilted, reobserved, court)
    witness = result.views[0]["two_view_source_depth_observability"]
    assert witness["effective_rank"] == 3
    assert witness["counterfactual_status"] == "NOT_EVALUATED_NO_COMMON_NULLSPACE"
    assert witness["source_point_correspondence_authority"] == (
        "UNKNOWN_NOT_SOURCE_OBSERVED"
    )
    assert not witness["source_depth_or_surface_completion_authority"]
    assert not result.canonical_mesh_authority
