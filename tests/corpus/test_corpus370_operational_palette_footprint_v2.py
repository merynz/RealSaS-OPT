"""Synthetic independent original-rest analytic ray and subpixel gate tests."""
import numpy as np
from tools.corpus.court_corpus370_operational_palette_footprint_v2 import SourceRestRayOracle,STENCIL


def square_scene(front_colors=(0,1)):
    vertices=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],dtype=float)
    faces=np.array([[0,1,2],[0,2,3]],dtype=int)
    oracle=SourceRestRayOracle(vertices,faces,list(front_colors))
    oracle.view({"origin":[0,0,-2],"forward":[0,0,1],
                 "right":[1,0,0],"screen_up":[0,1,0],
                 "half_extent":1.0,"clip_start":0.1,"clip_end":4.0})
    return oracle


def test_source_rest_nearest_triangle_color_changes_across_diagonal():
    ray=square_scene()
    assert ray.first_hit_palette(140,140)==0
    assert ray.first_hit_palette(115,115)==1


def test_subpixel_diagonal_mixing_is_detected_when_cardinal_sampling_is_insufficient():
    ray=square_scene()
    colors=[ray.first_hit_palette(127.5+dx,127.5+dy)
            for dx in STENCIL for dy in STENCIL]
    assert 0 in colors and 1 in colors
    assert len(set(colors))==2


def test_frontmost_depth_wins_over_hidden_palette():
    vertices=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],
                       [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)
    faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],dtype=int)
    ray=SourceRestRayOracle(vertices,faces,[0,0,1,1])
    ray.view({"origin":[0,0,-2],"forward":[0,0,1],
              "right":[1,0,0],"screen_up":[0,1,0],
              "half_extent":1.0,"clip_start":0.1,"clip_end":4.0})
    assert ray.first_hit_palette(140,140)==0
    assert ray.first_hit_palette(115,115)==0


def test_unknown_ray_stays_unknown():
    ray=square_scene()
    assert ray.first_hit_palette(-200,128) is None


def test_zero_or_nonfinite_source_geometry_is_not_admitted():
    try:
        SourceRestRayOracle(np.array([[np.nan,0,0],[0,1,0],[1,0,0]]),
                           [[0,1,2]],[0])
    except ValueError as exc:
        assert 'SOURCE_REST_ARRAY_DOMAIN' in str(exc)
    else:
        raise AssertionError('Nonfinite geometry unexpectedly admitted')
