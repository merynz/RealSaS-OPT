"""Exercise the Stage45 adapter's relation check independently of V5 replay."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.visual_attachment_depth_v1 import OPERATOR_ID, POLICY
from compiler.realsas_compiler_core.visual_motion_blend_v1 import evaluate_motion_blend
from compiler.realsas_compiler_services.orchestrator.adapters import presentation_research_v4_impl as gate
from test_visual_presentation_pose_v1 import fixture


@dataclass
class Projection:
    visual_deformation_operator_id: str
    visual_deformation_policy_hash: str
    views: list
    clips: list


def test_stage45_rejects_healthy_disconnected_control_even_if_old_replay_passes(monkeypatch):
    geometry, rest, blend, faces = fixture()
    rest, blend = rest[:8], blend[:8]
    actual = evaluate_motion_blend(rest, rest_source_xy=rest[:, :2], coefficients=blend,
        **{k: geometry[k] for k in ("axis_positions_source", "skin_matrices_source", "camera")})
    # Isolate the independent predicate: the prior operator proof is assumed to
    # pass, exactly as in the native/reference counterexample integration test.
    monkeypatch.setattr(gate.control, "_repair_proof", lambda *a: {
        "maximum_attachment_xy_owner_residual": 0., "attachment_slot_motion_passed": True})
    monkeypatch.setattr(gate.control, "_source_camera", lambda view: geometry["camera"])
    witness = {"axis_positions_source": geometry["axis_positions_source"], "axis_parents": geometry["axis_parents"],
               "clip_0_skin_matrices_source": geometry["skin_matrices_source"][None]}
    arrays = {"view_0_rest_positions": rest[:, :2], "view_0_faces": faces,
        "view_0_vertex_attachment_owner": np.zeros(8, dtype=np.int32),
        "view_0_attachment_rest_depths": np.ones(8), "view_0_motion_blend_coefficients": blend,
        "view_0_domain_domain_id": np.array([0]*4 + [1]*4),
        "clip_0_view_0_positions": actual[None, :, :2]}
    projection = Projection(OPERATOR_ID, content_sha256(POLICY), [SimpleNamespace(view_index=0, view_id="V0")],
                            [SimpleNamespace(array_prefix="clip_0", clip_id="broken", loop=False)])
    proof = gate._repair_proof(projection, arrays, {"attachments": {"attachments": []}}, witness, None, None)
    assert proof["area_condition_passed"] is True
    assert proof["connected_palette_relations_passed"] is False
    assert proof["connected_palette_relation_failed_frame_views"] == 1
    assert proof["maximum_parent_child_relation_residual_source_px"] > 20
    assert proof["relational_presentation_passed"] is False
    assert proof["qualified_dynamic_coverage_passed"] is None
