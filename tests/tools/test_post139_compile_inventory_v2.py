from __future__ import annotations

from tools.corpus.post139_compile_inventory_v2 import semantic_state, validate_policy


def row(subject_id: str, *, source_id: str = "source", role: str = "POST139_CANDIDATE") -> dict:
    return {"subject_id": subject_id, "source_id": source_id, "role": role}


def test_semantic_policy_classifies_without_deleting_audit_rows():
    policy = {
        "exclude_subject_ids": ["loadout"],
        "alias_subject_ids": {"blue": "gold"},
        "reference_include_subject_ids_by_source": {"base": ["base-textured"]},
    }
    assert semantic_state(row("loadout"), policy) == ("EXCLUDED_BY_SEMANTIC_POLICY", None)
    assert semantic_state(row("blue"), policy) == ("SEMANTIC_ALIAS_COLLAPSE", "gold")
    assert semantic_state(row("base-flat", source_id="base", role="REFERENCE_BASE139"), policy) == (
        "REFERENCE_VARIANT_NOT_IN_BASE139", None,
    )
    assert semantic_state(row("base-textured", source_id="base", role="REFERENCE_BASE139"), policy) == (
        "ELIGIBLE", None,
    )


def test_policy_validation_fails_closed_on_unknown_and_cross_source_aliases():
    subjects = [row("a", source_id="one"), row("b", source_id="two")]
    policy = {
        "exclude_subject_ids": ["missing"],
        "alias_subject_ids": {"a": "b"},
        "reference_include_subject_ids_by_source": {},
    }
    blockers = validate_policy(subjects, {"sources": []}, policy)
    codes = {item["code"] for item in blockers}
    assert "SUBJECT_POLICY_UNKNOWN_EXCLUSION" in codes
    assert "SUBJECT_POLICY_ALIAS_CROSS_SOURCE" in codes
