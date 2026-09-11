from __future__ import annotations

import json
from pathlib import Path

from models.arachne.v5.checkpoint_authority_v1 import sealed_arachne_v5_fit1_authority

ROOT = Path(__file__).resolve().parents[2]


def test_arachne_v5_checkpoint_authority_matches_promoted_witness() -> None:
    a = sealed_arachne_v5_fit1_authority()
    witness = json.loads((ROOT / "models/arachne/v5/PROMOTED_MAGE_FIT_WITNESS_V1.json").read_text())
    manifest = json.loads((ROOT / "canonical/ARACHNE_A1_V5_FIT1_EVIDENCE_MANIFEST_V1.json").read_text())

    assert witness["status"] == "FIT1_FROZEN_PROMOTED"
    assert witness["architecture_id"] == a["architecture_id"]
    assert witness["backbone"]["checkpoint_sha256"] == a["backbone_checkpoint_sha256"]
    assert witness["backbone"]["checkpoint_drive_id"] == a["backbone_checkpoint_drive_id"]
    assert witness["decoder"]["delta_checkpoint_sha256"] == a["decoder_delta_sha256"]
    assert witness["decoder"]["delta_checkpoint_drive_id"] == a["decoder_delta_drive_id"]
    assert manifest["checkpoint"]["total_parameters"] == a["total_parameter_count"]
    assert manifest["metrics"]["gsa_row_l1_p95"] == a["fit1_gsa_p95"]
    assert manifest["runtime_guards"]["a0_continuous_field_model_loaded"] is False
    assert witness["claim_boundary"]["unseen_family_claimed"] is False
    assert witness["claim_boundary"]["product_pass_claimed"] is False
