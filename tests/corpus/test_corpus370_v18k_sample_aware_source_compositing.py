"""V18k geometry-only selection, sample-before-filter compositor redteam.

No original artist semantics, V18i failure selection or Cycles sample equivalence.
"""
import numpy as np
import pytest

from models.iris.truth.graphics_oracle_v1 import OrthographicCameraV1
from tools.corpus.measure_corpus370_v18k_sample_aware_source_compositing import (
    SCHEMA,STATUS,TOL,SAMPLE_COUNT,RED,BLUE,
    proposed_pixels,render_sampled,over_premul,compare_to_source,
    premul_source,
)


def hit(*faces,complete=True,tie=False):
    return {
        "source_subpixel_offset_x":-.375,
        "source_subpixel_offset_y":-.375,
        "complete":complete,
        "depth_tie":tie,
        "source_hit_face_ordinals_teacher_only":list(faces),
        "source_hit_depths":[float(i+1) for i,_ in enumerate(faces)],
    }


def test_disjoint_opaque_subpixel_coverage_postfilter_alpha_fails():
    # First 8 subsamples hit only G; next 8 only R.
    # Full source pixel alpha 1; each isolated postfilter alpha 0.5.
    # Posterior OVER alpha 0.75 cannot be rescued by draw order.
    rays=[hit(0) for _ in range(8)]+[hit(1) for _ in range(8)]
    g=render_sampled(rays,np.array([2,3],np.int32),2)[
        "geometric_opaque_sample_composition"]
    assert g["mixed_source_nearest_G_R_within_pixel"]
    assert g["geometrically_nonempty_subsamples"]==16
    assert g["joint_pre_filter_premultiplied_rgba"][3]==1.
    assert g["postfilter_group_premultiplied_rgba"][3]==.5
    assert g["postfilter_remainder_premultiplied_rgba"][3]==.5
    assert g["postfilter_g_over_r_premultiplied_rgba"][3]==.75
    assert g["postfilter_r_over_g_premultiplied_rgba"][3]==.75
    assert g["joint_alpha_minus_postfilter_alpha"]==.25
    assert g["joint_vs_best_postfilter_order_max_rgba_delta"]>=.25


def test_two_opaque_overlapping_layers_with_opposite_subpixel_front_order():
    # Both G and R are present at each subpixel but the frontmost source
    # surface flips. Separate postfilter composes solid red/blue, losing the
    # true mixed red+blue joint raster in both directions.
    rays=[hit(0,1) for _ in range(8)]+[hit(1,0) for _ in range(8)]
    g=render_sampled(rays,np.array([2,3],np.int32),2)[
        "geometric_opaque_sample_composition"]
    np.testing.assert_allclose(g["joint_pre_filter_premultiplied_rgba"],
      [*((RED+BLUE)/2),1.],atol=1e-12,rtol=0)
    np.testing.assert_allclose(g["postfilter_g_over_r_premultiplied_rgba"],
      [*RED,1.],atol=1e-12,rtol=0)
    np.testing.assert_allclose(g["postfilter_r_over_g_premultiplied_rgba"],
      [*BLUE,1.],atol=1e-12,rtol=0)
    assert g["joint_vs_best_postfilter_order_max_rgba_delta"]>TOL
    assert g["joint_alpha_minus_postfilter_alpha"]==0


def test_per_ray_depth_interleaving_is_not_arbitrary_artist_part_boundary():
    rays=[hit(0,1,0) for _ in range(16)]
    g=render_sampled(rays,np.array([2,3],np.int32),2)[
        "geometric_opaque_sample_composition"]
    assert g["subsample_rays_with_multiple_depth_side_alternations"]==16
    assert not g["mixed_source_nearest_G_R_within_pixel"]
    assert g["joint_vs_best_postfilter_order_max_rgba_delta"]==0


def test_camera_tie_fails_closed_into_unknown_not_empty_or_negative_layer():
    rays=[hit(0,1) for _ in range(15)]+[hit(1,0,tie=True)]
    x=render_sampled(rays,np.array([2,3],np.int32),2)
    assert x["status"]=="INCOMPLETE_OR_TIED_RAY__NO_PIXEL_COMPOSITION"
    assert x["ray_unknown_sample_count"]==1
    assert x["geometric_opaque_sample_composition"] is None


def test_invalid_source_faces_and_depth_order_fail_closed():
    with pytest.raises(ValueError,match="V18K_ORIGINAL_RAY_FACE_ORDER_DRIFT"):
        render_sampled([hit(99) for _ in range(16)],np.array([2,3],np.int32),2)
    bad=hit(0,1)
    bad["source_hit_depths"]=[2.,1.]
    with pytest.raises(ValueError,match="V18K_ORIGINAL_RAY_FACE_ORDER_DRIFT"):
        render_sampled([bad for _ in range(16)],np.array([2,3],np.int32),2)


def test_actual_source_rgba_is_compared_separately_from_geometry_idealization():
    rays=[hit(0,1) for _ in range(16)]
    geom=render_sampled(rays,np.array([2,3],np.int32),2)
    full=np.array([*RED,1.],np.float64)
    isolated=np.array([*RED,1.],np.float64)
    remainder=np.array([*BLUE,1.],np.float64)
    x=compare_to_source(geom,full,isolated,remainder)
    assert x["source_joint_sample_error"]==pytest.approx(0.,abs=1e-12)
    assert x["real_blender_v18h_postfilter_both_orders_fail"] is False
    assert x["sample_evidence_not_exact_cycles_filter"] is True
    assert not x["no_artist_layer_identity"] is False


def test_premultiplied_rgba_and_alpha_are_not_accidentally_composited_as_straight():
    a=premul_source(np.array([.9,.2,.1,.5]))
    b=premul_source(np.array([.1,.3,.9,.6]))
    x=over_premul(a,b)
    np.testing.assert_allclose(x[:3],a[:3]+b[:3]*(1-.5))
    assert x[3]==pytest.approx(.8)
    with pytest.raises(ValueError,match="V18K_REAL_SOURCE_RGBA_INVALID"):
        premul_source(np.array([np.nan,0,0,1.]))


def test_group_centroid_geometric_selection_has_no_rgba_dependency():
    verts=np.array([
      [-.3,-.3,0.],[0.,-.3,0.],[-.2,0.,0.],
      [.1,-.3,0.],[.4,-.3,0.],[.3,0.,0.]],np.float64)
    faces=np.array([[0,1,2],[3,4,5]],np.int64)
    cam=OrthographicCameraV1(origin=(0.,0.,3.),
      forward=(0.,0.,-1.),right=(1.,0.,0.),screen_up=(0.,1.,0.),
      half_extent=1.,resolution=192,view_index=0)
    labels=np.array([4,5],np.int32)
    picked=proposed_pixels(verts,faces,cam,labels,4)
    assert picked==tuple(sorted(set(picked)))
    assert len(picked)>2
    assert all(0<=x<192 and 0<=y<192 for x,y in picked)
    assert proposed_pixels(verts,faces,cam,labels,4)==picked
    with pytest.raises(ValueError,match="V18K_TARGET_GROUP_NOT_FOUND"):
        proposed_pixels(verts,faces,cam,labels,7)


def test_whole_character_k1_looks_perfect_but_never_validates_editability():
    geo=render_sampled([hit(0) for _ in range(16)],
                       np.array([8],np.int32),8)
    g=geo["geometric_opaque_sample_composition"]
    assert g["joint_vs_best_postfilter_order_max_rgba_delta"]==0
    assert g["postfilter_remainder_premultiplied_rgba"][3]==0
    assert SCHEMA.endswith(".v18k")
    assert "ARTIST_UNKNOWN" in STATUS
    assert SAMPLE_COUNT==16
