from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def _c():return json.loads((ROOT/"canonical"/"VNEXT_SINGLE_CARRIER_CONTRACT_V1.json").read_text())
def test_vnext_contract_forbids_dual_mesh_authority_and_hash_only_graphics():
    c=_c();assert c["mechanics"]["canonical_authority_count"]==1;assert c["mechanics"]["independent_visual_mesh_forbidden"] is True;assert c["mechanics"]["canonical_mechanics_minted_after_dynamic_selection"] is True;assert c["graphics"]["binds_only_canonical_M"] is True;assert c["graphics"]["named_body_part_semantics_forbidden"] is True;assert c["runtime"]["graphics_hash_only_binding_forbidden"] is True;assert c["runtime"]["graphics_content_consumption_required"] is True;assert c["runtime"]["triangle_array_order_as_presentation_authority_forbidden"] is True
def test_vnext_contract_keeps_candidate_and_production_dynamic_courts_distinct():
    d=_c()["dynamic_courts"];assert d["must_remain_distinct"] is True;assert d["candidate_selection_probe"]!=d["production_motion_proof"]
def test_vnext_trainer_contract_retires_subject_serial_updates():
    t=_c()["trainer"];assert t["legacy_subject_serial_loop_forbidden"] is True;assert t["expected_fit8_batch_subjects"]==8
