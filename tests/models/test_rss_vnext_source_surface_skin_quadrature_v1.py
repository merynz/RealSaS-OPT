from __future__ import annotations

import pytest
import torch

from models.rss_vnext.training_source_surface_skin_quadrature_v1 import (
    source_surface_area_skin_loss_v1,
)


def mesh(subdivided: bool):
    # Two equal-area source triangle components. Only A is wrong;
    # split A into four faces while B stays one triangle. This changes
    # vertex count but NOT covered source surface area or its target field.
    xyz=[
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
        [2.,0.,0.],[3.,0.,0.],[2.,1.,0.],
    ]
    if subdivided:
        xyz.extend([[.5,0.,0.],[.5,.5,0.],[0.,.5,0.]])
        faces=[[0,6,8],[6,1,7],[8,7,2],[6,7,8],[3,4,5]]
    else:
        faces=[[0,1,2],[3,4,5]]
    rest=torch.tensor([xyz],dtype=torch.float64)
    faces=torch.tensor(faces,dtype=torch.long)
    # Wrong source triangle A; B correct.
    teacher=torch.full((1,len(xyz),2),.5,dtype=torch.float64)
    teacher[0,:3]=torch.tensor([1.,0.])
    if subdivided:
        teacher[0,6:]=torch.tensor([1.,0.])
    student=torch.full_like(teacher,.5).requires_grad_()
    return rest,faces,student,teacher


def test_exact_surface_loss_invariant_to_subdivision_count_but_vertex_mean_is_not():
    coarse=mesh(False)
    fine=mesh(True)
    a=source_surface_area_skin_loss_v1(*coarse)
    b=source_surface_area_skin_loss_v1(*fine,faces_per_chunk=1)
    assert a.loss.detach()==pytest.approx(float(b.loss.detach()),abs=1e-8)
    assert a.qualified_source_face_count==2
    assert b.qualified_source_face_count==5
    assert a.qualified_source_surface_area==pytest.approx(1.)
    assert b.qualified_source_surface_area==pytest.approx(1.)
    expected=.5*.5*(.5**2) # half of surface, SmoothL1(.5)
    assert float(a.loss)==pytest.approx(expected,abs=1e-8)
    vertex_coarse=torch.nn.functional.smooth_l1_loss(coarse[2],coarse[3])
    vertex_fine=torch.nn.functional.smooth_l1_loss(fine[2],fine[3])
    assert float(vertex_fine)>float(vertex_coarse)
    a.loss.backward()
    b.loss.backward()
    assert coarse[2].grad is not None and fine[2].grad is not None
    assert bool(torch.isfinite(coarse[2].grad).all())
    assert coarse[2].grad.abs().sum()>0
    assert a.teacher_only_source_geometry and not a.eligible_for_product_qualification


def test_source_surface_area_uses_only_explicitly_qualified_faces():
    rest,faces,student,teacher=mesh(False)
    mask=torch.tensor([False,True])
    report=source_surface_area_skin_loss_v1(
        rest,faces,student,teacher,source_face_mask=mask,
    )
    assert float(report.loss)==0.
    assert report.qualified_source_face_count==1
    assert report.qualified_source_surface_area==pytest.approx(.5)
    with pytest.raises(ValueError,match="NO_ELIGIBLE_FACES"):
        source_surface_area_skin_loss_v1(
            rest,faces,student,teacher,source_face_mask=torch.zeros_like(mask),
        )


@pytest.mark.parametrize("mutation,code",[
    ("index","NUMERIC_OR_INDEX"),
    ("duplicate","NUMERIC_OR_INDEX"),
    ("face_geometry","DEGENERATE_SOURCE_FACE"),
    ("student_simplex","SIMPLEX"),
    ("teacher_simplex","SIMPLEX"),
    ("wrong_mask","FACE_MASK_DOMAIN"),
    ("zero_faces","DOMAIN"),
])
def test_invalid_source_face_geometry_or_weight_domain_fails_closed(mutation,code):
    rest,faces,student,teacher=mesh(False)
    mask=None
    if mutation=="index":
        faces[0,0]=1000
    elif mutation=="duplicate":
        faces[0,0]=faces[0,1]
    elif mutation=="face_geometry":
        rest[0,2]=rest[0,1]+torch.tensor([.5,0.,0.])
    elif mutation=="student_simplex":
        with torch.no_grad():
            student[0,0]=torch.tensor([1.,1.])
    elif mutation=="teacher_simplex":
        teacher[0,0]=torch.tensor([1.,1.])
    elif mutation=="wrong_mask":
        mask=torch.ones((3,),dtype=torch.bool)
    elif mutation=="zero_faces":
        faces=faces[:0]
    with pytest.raises(ValueError,match=code):
        source_surface_area_skin_loss_v1(
            rest,faces,student,teacher,source_face_mask=mask,
        )


def test_no_grad_into_rest_geometry_or_independent_source_teacher():
    rest,faces,student,teacher=mesh(False)
    rest.requires_grad_(True)
    teacher.requires_grad_(True)
    out=source_surface_area_skin_loss_v1(rest,faces,student,teacher)
    out.loss.backward()
    assert rest.grad is None and teacher.grad is None
    assert student.grad is not None and student.grad.abs().sum()>0.
