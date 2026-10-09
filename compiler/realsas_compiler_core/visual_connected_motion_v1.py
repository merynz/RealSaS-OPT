"""Versioned research operator for connected body/prop presentation XY.

Depth is still the V5 provisional surface/slot policy. Semantic overlap and
qualified contact/material support remain separate mandatory Stage45 checks.
"""
from .visual_attachment_depth_v1 import POLICY as DEPTH_POLICY
from .visual_presentation_pose_v1 import POLICY as POSE_POLICY

OPERATOR_ID = "SOURCE_CHART_CONNECTED_2D_POSE_WITH_SLOT_OWNED_DEPTH_V6"
POLICY = {**DEPTH_POLICY,
    "schema": "RealSaS.VisualDeformationOperatorPolicy.v6",
    "operator_id": OPERATOR_ID,
    "body_xy_contract": "FROZEN_G_CONNECTED_PRESENTATION_PALETTE",
    "attachment_xy_contract": "SAME_CONNECTED_PALETTE_TARGET_SLOT",
    "connected_pose_policy": POSE_POLICY,
    "source_cut_coordinates_are_qualified_contacts": False,
    "surface_slot_depth_is_qualified_semantic_draw_order": False,
    "missing_relational_qualification": "STAGE45_FAIL_CLOSED",
}
