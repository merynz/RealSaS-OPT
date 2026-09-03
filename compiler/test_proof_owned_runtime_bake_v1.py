from types import SimpleNamespace

from compiler.realsas_compiler_services.export.runtime_deploy_bake import decode_runtime_deploy_bake
from compiler.realsas_compiler_services.proof.motion_probe import authored_motion_measurement_passes_v1, measure_authored_motion_v1


def _product(*, with_order=False):
    joint=SimpleNamespace(canonical_joint_id="J:0",parent_canonical_id=None,position=(0.,0.,0.)); skeleton=SimpleNamespace(joints=(joint,)); directions=[]
    for view_index in range(8):
        vertices=tuple(SimpleNamespace(canonical_mesh_vertex_id=f"V:{index}",P=point) for index,point in enumerate(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))))
        mesh=SimpleNamespace(vertices=vertices,faces=(("V:0","V:1","V:2"),)); rows=tuple(SimpleNamespace(canonical_mesh_vertex_id=f"V:{index}",influences=(("J:0",1.0),)) for index in range(3))
        component=SimpleNamespace(component_id="body",mesh=mesh,mesh_skin=SimpleNamespace(rows=rows),setup_order=0,default_visible=True)
        directions.append(SimpleNamespace(view_index=view_index,components=(component,)))
    keys=(SimpleNamespace(time_sec=0.,translation_xy=(0.,0.),rotation_deg=0.,scale_xy=(1.,1.),depth_offset=0.),SimpleNamespace(time_sec=.5,translation_xy=(0.,0.),rotation_deg=4.,scale_xy=(1.,1.),depth_offset=0.),SimpleNamespace(time_sec=1.,translation_xy=(0.,0.),rotation_deg=0.,scale_xy=(1.,1.),depth_offset=0.))
    clip=SimpleNamespace(clip_id="idle",clip_kind="PRESET",duration_sec=1.,loop=True); track=SimpleNamespace(clip_id="idle",canonical_joint_id="J:0",keys=keys); order_tracks=(SimpleNamespace(),) if with_order else ()
    return SimpleNamespace(mechanical_state=SimpleNamespace(skeleton=skeleton),directional_renderables=SimpleNamespace(directions=tuple(directions)),motion_state=SimpleNamespace(clips=(clip,),joint_tracks=(track,),order_tracks=order_tracks,visibility_tracks=()))


def test_proof_owned_bake_contains_exact_sampled_frames():
    measurements=measure_authored_motion_v1(_product()); assert authored_motion_measurement_passes_v1(measurements); assert measurements["runtime_bake_status"]=="PASS_STATIC_SETUP_ORDER"; assert measurements["runtime_export_solver_replay"] is False; assert len(measurements["runtime_deploy_bakes"])==1
    decoded=decode_runtime_deploy_bake(measurements["runtime_deploy_bakes"][0]); assert decoded["clip_id"]=="idle"; assert len(decoded["frames"])>=9; assert set(decoded["frames"][0]["mesh_vertices_by_id"])=={f"V{i}:body" for i in range(8)}; assert decoded["frames"][0]["render_order_by_view"]["V0"]==["V0:body"]


def test_unknown_dynamic_order_track_abstains_instead_of_guessing_runtime_semantics():
    measurements=measure_authored_motion_v1(_product(with_order=True)); assert authored_motion_measurement_passes_v1(measurements); assert measurements["runtime_deploy_bakes"]==[]; assert measurements["runtime_bake_status"].startswith("ABSTAIN_")
