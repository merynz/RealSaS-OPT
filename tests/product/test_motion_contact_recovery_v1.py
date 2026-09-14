from __future__ import annotations

from types import SimpleNamespace

from compiler.realsas_compiler_core.motion_contact import (
    ContactWindowIR,
    build_contact_plan,
    build_mage_run_contact_plan,
    measure_contact_sliding,
)
from compiler.realsas_compiler_core.motion_locomotion import build_mage_historical_phase_motion
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
from tests.product.test_mage_topology_motion_deformation_v1 import _fixture


def _synthetic_product(*, rotating: bool):
    root=SimpleNamespace(canonical_joint_id="root",parent_canonical_id=None)
    foot=SimpleNamespace(canonical_joint_id="foot",parent_canonical_id="root")
    skeleton=SimpleNamespace(joints=(root,foot),skeleton_lineage_hash="skeleton")
    keys=(
        SimpleNamespace(time_sec=0.0,rotation_deg=0.0),
        SimpleNamespace(time_sec=1.0,rotation_deg=30.0 if rotating else 0.0),
    )
    track=SimpleNamespace(clip_id="run",canonical_joint_id="root",keys=keys)
    clip=SimpleNamespace(clip_id="run",duration_sec=1.0)
    motion=SimpleNamespace(joint_tracks=(track,),clips=(clip,),motion_state_hash="motion-rot" if rotating else "motion-static")
    mechanical=SimpleNamespace(skeleton=skeleton)
    product=SimpleNamespace(product_state_hash="product",motion_state=motion,mechanical_state=mechanical)
    pivots=[]
    for view in range(8):
        pivots.extend((
            SimpleNamespace(view_index=view,canonical_joint_id="root",raster_xy=(0.0,0.0)),
            SimpleNamespace(view_index=view,canonical_joint_id="foot",raster_xy=(100.0,0.0)),
        ))
    binding=SimpleNamespace(source_product_state_hash="product",binding_set_hash="binding",joint_pivots=tuple(pivots))
    plan=build_contact_plan(
        clip_id="run",source_motion_state_hash=motion.motion_state_hash,skeleton_lineage_hash="skeleton",
        windows=(ContactWindowIR("plant","foot",0.0,1.0,"LEFT"),),
    )
    return product,binding,plan


def test_directional_contact_analyzer_passes_static_plant_and_rejects_visible_sliding():
    product,binding,plan=_synthetic_product(rotating=False)
    report=measure_contact_sliding(product=product,directional_binding=binding,plan=plan,max_allowed_sliding_px=2.0)
    assert report.passed
    assert report.max_sliding_px == 0.0
    assert len(report.measurements)==8
    assert report.metadata["contact_correction_applied"] is False

    product,binding,plan=_synthetic_product(rotating=True)
    report=measure_contact_sliding(product=product,directional_binding=binding,plan=plan,max_allowed_sliding_px=2.0)
    assert not report.passed
    assert report.max_sliding_px > 2.0
    assert report.failed_contact_ids == ("plant",)


def test_mage_contact_plan_is_canonical_role_and_motion_lineage_bound():
    mechanical,_mesh,_mesh_skin=_fixture()
    phase=build_mage_historical_phase_motion(mechanical)
    motion=compile_motion_quality(phase,mechanical)
    first=build_mage_run_contact_plan(mechanical,motion)
    second=build_mage_run_contact_plan(mechanical,motion)
    joint_ids={j.canonical_joint_id for j in mechanical.skeleton.joints}
    assert first.plan_hash==second.plan_hash
    assert first.source_motion_state_hash==motion.motion_state_hash
    assert first.skeleton_lineage_hash==mechanical.skeleton.skeleton_lineage_hash
    assert len(first.windows)==3
    assert all(w.canonical_joint_id in joint_ids for w in first.windows)
    assert first.metadata["contact_lock_claimed"] is False
