from __future__ import annotations
import json
from pathlib import Path

from compiler.realsas_compiler_core.mesh.product_policy_v1 import (
    mesh_policy_from_product_policy_document,
)

ROOT=Path(__file__).resolve().parents[2]


def test_frozen_canonical_policy_document_maps_exactly_to_typed_policy():
    payload=json.loads((ROOT/"canonical"/"QUALIFIED_MESH_PRODUCT_POLICY_V1_20260918.json").read_text(encoding="utf-8"))
    policy=mesh_policy_from_product_policy_document(payload)
    assert policy.g1_max_normal_refinement_ratio == 0.125
    assert policy.g1_max_tangential_to_normal_ratio == 0.25
    assert policy.g3_min_angle_deg == 7.5
    assert policy.g3_max_aspect_longest_over_min_altitude == 16.0
    assert policy.g3_min_dynamic_area_ratio == 0.05
    assert policy.g3_max_dynamic_area_ratio == 20.0
    assert policy.g3_max_dynamic_condition_number == 16.0
    by={row.carrier_class:row for row in policy.coverage_thresholds}
    assert by["MESH"].min_recall == 0.97
    assert by["PLANAR"].min_recall == 0.99
    assert len(policy.qualification_policy_lineage_hash)==64
