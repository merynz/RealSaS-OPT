import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_closure_contract_keeps_training_closed():
    data = json.loads((ROOT / "canonical" / "CORPUS370_TRAINING_READINESS_CLOSURE_V1.json").read_text())
    assert data["candidate_universe"]["total_candidates"] == 370
    assert data["training_authorized"] is False
    assert data["historical_authority"]["truth139_training_authorized_is_inventory_only"] is True
    assert data["historical_authority"]["historical_112_27_split_inherited"] is False


def test_global_sdf_requires_explicit_solid_authority():
    data = json.loads((ROOT / "canonical" / "CORPUS370_TRAINING_READINESS_CLOSURE_V1.json").read_text())
    sdf = data["global_sdf"]
    assert sdf["requires"] == "EXPLICIT_QUALIFIED_SOLID_ENTITY_OR_ASSEMBLY"
    assert sdf["missing_sign"] == "MASKED_OR_ABSENT__NEVER_ZERO_FILLED"
    assert "NEAREST_COMPONENT_SIGN_AS_GLOBAL_SIGN" in sdf["forbidden"]
