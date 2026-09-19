import math
from dataclasses import dataclass
import numpy as np

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import rasterize_visible_face_pixel_counts


@dataclass
class V:
    canonical_mesh_vertex_id:str
    P:tuple


@dataclass
class Mesh:
    vertices:tuple
    faces:tuple


@dataclass
class Camera:
    view_index:int=0
    origin:tuple=(0.0,0.0,-2.0)
    right:tuple=(1.0,0.0,0.0)
    screen_up:tuple=(0.0,-1.0,0.0)
    forward:tuple=(0.0,0.0,1.0)
    half_extent:float=1.0
    resolution:int=32


def test_visible_face_counts_respect_z_occlusion_and_source_mask():
    # Two identical screen-space triangles at different depth; front face wins.
    mesh=Mesh(
        vertices=(
            V("a",(-0.5,-0.5,0.0)),V("b",(0.5,-0.5,0.0)),V("c",(0.0,0.5,0.0)),
            V("d",(-0.5,-0.5,0.5)),V("e",(0.5,-0.5,0.5)),V("f",(0.0,0.5,0.5)),
        ),
        faces=(("a","b","c"),("d","e","f")),
    )
    cam=Camera()
    counts=rasterize_visible_face_pixel_counts(mesh,cam,width=32,height=32)
    assert counts[0]>0
    assert counts[1]==0
    mask=bytes([0]*(32*32))
    masked=rasterize_visible_face_pixel_counts(mesh,cam,width=32,height=32,pixel_mask=mask)
    assert masked==(0,0)


def test_visible_face_counts_detect_motion_exposure():
    mesh=Mesh(
        vertices=(
            V("a",(-0.5,-0.5,0.0)),V("b",(0.5,-0.5,0.0)),V("c",(0.0,0.5,0.0)),
            V("d",(-0.5,-0.5,0.5)),V("e",(0.5,-0.5,0.5)),V("f",(0.0,0.5,0.5)),
        ),
        faces=(("a","b","c"),("d","e","f")),
    )
    cam=Camera()
    rest=rasterize_visible_face_pixel_counts(mesh,cam,width=32,height=32)
    assert rest[1]==0
    posed=np.asarray([v.P for v in mesh.vertices],dtype=float)
    posed[:3,0]+=1.2
    moved=rasterize_visible_face_pixel_counts(mesh,cam,positions=posed,width=32,height=32)
    assert moved[1]>0
