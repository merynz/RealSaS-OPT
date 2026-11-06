from __future__ import annotations

"""V7 inherited RigProgram -> Mechanics execution static compatibility court.

This asserts the *gaps*, not that either product or source truth is correct.
"""

from pathlib import Path
import pytest

from tools.mechanics.audit_v7_mechanics_execution_gap_v1 import (
    audit_v7_mechanics_execution_gap_v1,
)


ROOT=Path(__file__).resolve().parents[2]


def test_v7_inherited_v6_operator_and_dof_vocabulary_has_measured_runtime_gap():
    receipt=audit_v7_mechanics_execution_gap_v1(ROOT)
    assert receipt["status"]=="GAP__NO_V7_MECHANICS_PRODUCT_READINESS"
    o=receipt["observations"]
    assert o["producer_constraint_operator_ids"]==(
        "IK","AIM","LIMIT_ROTATION","LIMIT_TRANSLATION",
        "FOLLOW","COPY_TRANSFORM","PATH","STRETCH",
    )
    assert o["canonical_runtime_constraint_operator_ids"]==(
        "IK","AIM","FOLLOW","COPY_TRANSFORM","STRETCH",
    )
    assert o["producer_only_operator_ids"]==(
        "LIMIT_ROTATION","LIMIT_TRANSLATION","PATH",
    )
    assert o["producer_dof_labels"]==(
        "TX","TY","TZ","RX","RY","RZ","SX","SY","SZ",
    )
    assert o["multigoal_analytic_rotational_dof_labels"]==("RX","RY","RZ")
    assert o["producer_dofs_not_executable_in_analytic_multigoal"]==(
        "TX","TY","TZ","SX","SY","SZ",
    )
    assert not o["explicit_typed_anchor_frame_payload_present"]
    assert not o["explicit_typed_relative_bridge_payload_present"]
    assert len(receipt["gaps"])==4
    assert receipt["authority"]["rig_program_execution_product_ready"] is False
    assert receipt["authority"]["training_admission_minted"] is False
    assert receipt["authority"]["default_product_compiler_mutated"] is False


def test_gap_audit_fails_closed_if_source_is_absent_or_cannot_parse(tmp_path: Path):
    with pytest.raises(ValueError,match="SOURCE_FILE_MISSING"):
        audit_v7_mechanics_execution_gap_v1(tmp_path)

    for path in (
        "models/axis/rig_program_model_v6.py",
        "compiler/realsas_compiler_core/rig_constraint_schema_v1.py",
        "compiler/realsas_compiler_core/mechanics_multigoal_solver_v1.py",
    ):
        dest=tmp_path/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text((ROOT/path).read_text(encoding="utf-8"),encoding="utf-8")
    result=audit_v7_mechanics_execution_gap_v1(tmp_path)
    assert result["status"]=="GAP__NO_V7_MECHANICS_PRODUCT_READINESS"
    changed=tmp_path/"models/axis/rig_program_model_v6.py"
    changed.write_text(
        changed.read_text().replace("CONSTRAINT_OPERATORS_V6 = (","UNRECOGNIZED_OPERATORS_V6 = ("),
        encoding="utf-8",
    )
    with pytest.raises(ValueError,match="SYMBOL_MISSING"):
        audit_v7_mechanics_execution_gap_v1(tmp_path)
