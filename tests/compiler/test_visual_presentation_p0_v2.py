from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_domain_v2 import (
    build_domain_binding, evaluate_domain_binding, validate_domain_binding,
    presentation_condition_metrics, OPERATOR_ID, POLICY,
)
from compiler.realsas_compiler_core.visual_depth_v2 import render_visual_depth
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_services.orchestrator.adapters.runtime_v2 import prove_visual_domain_matrix


def domain():
    camera = qualify_camera_v3(dict(origin=[0, 0, -2], right=[1, 0, 0],
                                    screen_up=[0, -1, 0], forward=[0, 0, 1],
                                    half_extent=4, resolution=8), view_id="V0", view_index=0)
    xyz = np.array([[-3., -3, 0], [3, -3, 0], [-3, 3, 0], [3, 3, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    points = np.array([[1., 1], [4, 1], [7, 1], [1, 4], [4, 4], [7, 4],
                       [1, 7], [4, 7], [7, 7]])
    visual = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4],
                       [3, 4, 6], [4, 7, 6], [4, 5, 7], [5, 8, 7]])
    seed = np.full((8, 8), -1, dtype=int)
    owner = seed.copy()
    seed[2, 2] = seed[5, 5] = 0
    owner[2, 2], owner[5, 5] = 0, 1
    binding = build_domain_binding(points_source_xy=points, visual_faces=visual,
        vertex_region_id=np.zeros(len(points), dtype=int), seed_region_labels=seed,
        owner_face_index=owner, mechanical_positions_xyz=xyz, mechanical_faces=faces,
        camera=camera)
    return binding, visual, xyz, camera


def test_chart_field_preserves_rest_and_transports_uniform_motion_and_depth():
    b, f, xyz, camera = domain()
    rest = evaluate_domain_binding(b, visual_faces=f, posed_mechanical_positions_xyz=xyz, camera=camera)
    posed = evaluate_domain_binding(b, visual_faces=f,
        posed_mechanical_positions_xyz=xyz + [0.25, -0.5, 0.75], camera=camera)
    assert np.allclose(rest[:, :2], b["rest_positions"])
    assert np.allclose(posed - rest, [0.25, -0.5, 0.75])
    assert len(b["anchor_vertex"]) < len(b["rest_positions"])


def test_chart_field_rejects_unanchored_components_and_extrapolation():
    b, f, xyz, camera = domain()
    b["anchor_barycentric"][0] = [-1, 1, 1]
    with pytest.raises(QualificationError, match="OPERATOR_INVALID"):
        validate_domain_binding(b, f)


def test_small_unseeded_island_uses_bounded_same_region_support():
    b, f, xyz, camera = domain()
    seed = np.full((8,8), -1, dtype=int); owner=seed.copy()
    seed[2,2]=owner[2,2]=0
    def build(offset):
        points=np.vstack((b["rest_positions"], np.array([[0,0],[2,0],[0,2]])+offset))
        return build_domain_binding(points_source_xy=points,visual_faces=np.vstack((f,[9,10,11])),
            vertex_region_id=np.zeros(12,dtype=int),seed_region_labels=seed,owner_face_index=owner,
            mechanical_positions_xyz=xyz,mechanical_faces=[[0,1,2],[1,3,2]],camera=camera)
    binding=build(20)
    field=evaluate_domain_binding(binding,visual_faces=np.vstack((f,[9,10,11])),
                                  posed_mechanical_positions_xyz=xyz+[1,0,0],camera=camera)
    np.testing.assert_allclose(field[:,:2]-binding["rest_positions"],np.tile([1,0],(12,1)))
    assert np.min(binding["anchor_barycentric"])>=0
    with pytest.raises(QualificationError,match="UNANCHORED_COMPONENT"):build(100)
    b, f, xyz, camera = domain()
    b["anchor_vertex"] = np.array([], dtype=int)
    with pytest.raises(QualificationError, match="OPERATOR_INVALID"):
        validate_domain_binding(b, f)


def test_chart_field_rejects_cross_domain_faces_and_tampered_domain_ids():
    b, f, xyz, camera = domain()
    b["vertex_region_id"][1] = 1
    with pytest.raises(QualificationError, match="TOPOLOGY_INVALID"):
        validate_domain_binding(b, f)
    b, f, xyz, camera = domain()
    b["domain_id"][0] = 9
    with pytest.raises(QualificationError, match="OPERATOR_INVALID"):
        validate_domain_binding(b, f)


def layered(depths=(1., 2.), order=(0, 1)):
    points = np.tile([[1., 1], [6, 1], [1, 6]], (len(depths), 1))
    faces = np.array([[i*3, i*3+1, i*3+2] for i in order])
    uv = np.array([[0, 0]]*3 + [[1, 1]]*(len(points)-3), dtype=float)
    texture = np.zeros((8, 8, 4), dtype=np.uint8)
    texture[0, 0] = [255, 0, 0, 128]
    texture[-1, -1] = [0, 0, 255, 255]
    return render_visual_depth(positions=points, depths=np.repeat(depths, 3),
                              faces=faces, uv=uv, texture=texture, resolution=8)


def test_overlap_uses_canonical_depth_and_is_face_permutation_invariant():
    a, b = layered(), layered(order=(1, 0))
    assert np.array_equal(a.straight_rgba_u8, b.straight_rgba_u8)
    assert a.overlap_pixel_count > 0
    assert a.unresolved_depth_tie_count == 0
    assert tuple(a.straight_rgba_u8[2, 2]) == (128, 0, 127, 255)
    assert a.owner_face_index[2, 2] == 0
    assert b.owner_face_index[2, 2] == 1


def test_depth_ties_and_layer_overflow_are_measured_failures():
    assert layered(depths=(1, 1)).unresolved_depth_tie_count > 0
    assert layered(depths=tuple(range(1,34)), order=tuple(range(33))).fragment_overflow_count > 0


def test_area_and_condition_catch_collapse_with_small_edge_ratios():
    rest = np.array([[0., 0], [1, 0], [0.5, 1]])
    posed = np.array([[0., 0], [1, 0], [0.5, 0.01]])
    metrics = presentation_condition_metrics(rest, posed, [[0, 1, 2]])
    assert metrics["area_collapse_count"] == 1
    assert metrics["condition_failure_count"] == 1
    posed[0, 0] = np.nan
    with pytest.raises(QualificationError, match="NONFINITE"):
        presentation_condition_metrics(rest, posed, [[0, 1, 2]])


def matrix_fixture():
    b, faces, xyz, camera = domain()
    mesh = SimpleNamespace(mesh_lineage_hash="m", faces=[tuple(str(i) for i in row) for row in b["anchor_mechanical_vertices"]], vertices=[
        SimpleNamespace(canonical_mesh_vertex_id=str(i), P=p) for i, p in enumerate(xyz)])
    frame = SimpleNamespace(time_seconds=0., posed_vertex_xyz=[(str(i), p) for i, p in enumerate(xyz)])
    dynamic = SimpleNamespace(dynamic_motion_hash="d", clips=[SimpleNamespace(clip_id="idle", frames=[frame])])
    clip = SimpleNamespace(clip_id="idle", array_prefix="clip_0", frame_count=1)
    projection = SimpleNamespace(visual_deformation_operator_id=OPERATOR_ID,
        visual_deformation_policy_hash=content_sha256(POLICY), mechanical_mesh_binding_hash="m",
        dynamic_motion_binding_hash="d", clips=[clip], views=[])
    arrays = {"clip_0_times": np.array([0.])}
    for i in range(8):
        c = dict(camera.__dict__, view_id=f"V{i}", view_index=i)
        projection.views.append(SimpleNamespace(view_index=i, view_id=f"V{i}", camera=c, source_width=8))
        field = evaluate_domain_binding(b, visual_faces=faces, posed_mechanical_positions_xyz=xyz, camera=camera)
        arrays.update({f"view_{i}_domain_{k}": v for k, v in b.items()})
        arrays[f"view_{i}_faces"] = faces
        arrays[f"clip_0_view_{i}_positions"] = field[None, :, :2]
        arrays[f"clip_0_view_{i}_depths"] = field[None, :, 2]
    return projection, arrays, mesh, dynamic


def test_proof_recomputes_field_and_detects_forged_baked_positions():
    p, a, m, d = matrix_fixture()
    assert prove_visual_domain_matrix(p, a, mesh=m, dynamic=d)["domain_coherence_passed"]
    a["clip_0_view_0_positions"] = a["clip_0_view_0_positions"].copy()
    a["clip_0_view_0_positions"][0, 4, 0] += 0.1
    assert not prove_visual_domain_matrix(p, a, mesh=m, dynamic=d)["domain_coherence_passed"]


def test_slot_attachment_proof_requires_witness_and_rejects_wrong_slot_even_with_good_shape():
    from compiler.realsas_compiler_core.visual_attachment_motion_v1 import (
        OPERATOR_ID as SLOT_OPERATOR, POLICY as SLOT_POLICY,
    )
    p, a, m, d = matrix_fixture()
    p.visual_deformation_operator_id = SLOT_OPERATOR
    p.visual_deformation_policy_hash = content_sha256(SLOT_POLICY)
    p.metadata = {"target_attachments": {"attachments": [{"attachment_index": 1,
        "presentation_motion_model": "TARGET_SLOT_LOCAL_RIGID_2D_CAMERA_TWIST_V1",
        "target_slot_raw_index_fit_only": 0, "vertex_indices": [0,1,2,3]}]}}
    with pytest.raises(QualificationError, match="SEALED_WITNESS_REQUIRED"):
        prove_visual_domain_matrix(p, a, mesh=m, dynamic=d)
    xyz = np.asarray([v.P for v in m.vertices])
    witness = {"vertices": xyz, "presentation_attachment_vertex_owner": np.ones(len(xyz), dtype=int),
        "axis_positions_source": np.array([[0.,0,0]]), "clip_0_canonical_xyz": xyz[None],
        "clip_0_skin_matrices_source": np.eye(4)[None,None]}
    assert prove_visual_domain_matrix(p,a,mesh=m,dynamic=d,attachment_witness=witness)["attachment_slot_motion_passed"]
    # Forged slot movement, perfectly rigid baked triangles: shape alone is insufficient.
    witness["clip_0_skin_matrices_source"][0,0,0,3] = 1
    for vi in range(8):
        a[f"clip_0_view_{vi}_positions"] = a[f"clip_0_view_{vi}_positions"].copy() + [1,0]
    proof = prove_visual_domain_matrix(p,a,mesh=m,dynamic=d,attachment_witness=witness)
    assert proof["area_condition_passed"] and proof["domain_coherence_passed"]
    assert not proof["attachment_slot_motion_passed"]


@pytest.mark.parametrize("missing", ["view", "depth", "time"])
def test_proof_fails_closed_on_incomplete_matrix(missing):
    p, a, m, d = matrix_fixture()
    if missing == "view":
        p.views.pop()
    elif missing == "depth":
        del a["clip_0_view_3_depths"]
    else:
        a["clip_0_times"] = np.array([1.])
    with pytest.raises(QualificationError):
        prove_visual_domain_matrix(p, a, mesh=m, dynamic=d)


def test_motion_blend_proof_rederives_coefficients_even_when_forgery_renders_same_rest_image():
    from compiler.realsas_compiler_core.visual_motion_blend_v1 import OPERATOR_ID as BLEND_OPERATOR, POLICY as BLEND_POLICY
    p,a,m,d=matrix_fixture();p.visual_deformation_operator_id=BLEND_OPERATOR
    p.visual_deformation_policy_hash=content_sha256(BLEND_POLICY);p.metadata={'target_attachments':{'attachments':[]}}
    xyz=np.asarray([v.P for v in m.vertices]);weights=np.tile([1.,0],(len(xyz),1))
    witness={'vertices':xyz,'presentation_attachment_vertex_owner':np.zeros(len(xyz),dtype=int),
        'axis_positions_source':np.zeros((2,3)),'canonical_motion_weights':weights,
        'clip_0_canonical_xyz':xyz[None],'clip_0_skin_matrices_source':np.tile(np.eye(4),(1,2,1,1))}
    for vi in range(8):a[f'view_{vi}_motion_blend_coefficients']=np.tile([1.,0],(9,1))
    proof=prove_visual_domain_matrix(p,a,mesh=m,dynamic=d,attachment_witness=witness)
    assert proof['canonical_pose_palette_passed'] and proof['domain_coherence_passed']
    a['view_0_motion_blend_coefficients'][4]=[0,1]
    with pytest.raises(QualificationError,match='MOTION_COEFFICIENT_DRIFT'):
        prove_visual_domain_matrix(p,a,mesh=m,dynamic=d,attachment_witness=witness)
