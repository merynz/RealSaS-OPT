import pytest

from compiler.realsas_compiler_core.repair_registry import (
    operation_authority_blockers_v1,
    operation_authority_v1,
    resolve_promoted_repair_operation_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    QualifiedJoint,
    QualifiedSkeletonIR,
)
from compiler.realsas_compiler_services.proof.control_basis_repair import (
    ALLOWED_CHANGE_PATHS,
    OPERATION_ID,
    OWNER_ID,
    QUALIFICATION_HASH,
    build_control_basis_single_prune_operation_v1,
    execute_control_basis_single_prune_v1,
)
from compiler.realsas_compiler_services.proof.repair_loop import (
    build_bounded_repair_directives_v1,
)


def _skeleton():
    return QualifiedSkeletonIR(
        joints=(
            QualifiedJoint("root", (0.0, 0.0, 0.0), None),
            QualifiedJoint("mid", (0.0, 1.0, 0.0), "root"),
            QualifiedJoint("tip", (0.0, 2.0, 0.0), "mid"),
            QualifiedJoint("side", (1.0, 1.0, 0.0), "root"),
        ),
        root_id="root",
        qualification_report={"status": "PASS"},
        skeleton_lineage_hash="SK:PARENT",
    )


def _op(control_id="mid"):
    return build_control_basis_single_prune_operation_v1(
        target_signature_ids=("SIG:1",),
        selected_control_id=control_id,
        source_skeleton_hash="SK:PARENT",
    )


def _finding(status="attributed"):
    return {
        "finding_id": "OWNER_ATTR:1",
        "signature_id": "SIG:1",
        "status": status,
        "selected_owner_id": OWNER_ID,
        "automatic_repair_eligible": status == "attributed",
    }


def test_control_basis_prune_operation_matches_promoted_registry_authority():
    op = _op()
    authority = operation_authority_v1(OPERATION_ID)
    assert authority.current_main_status == "CANONICAL_MAINLINE_EXECUTABLE"
    assert authority.qualification_hash == QUALIFICATION_HASH
    assert authority.allowed_change_patterns == ALLOWED_CHANGE_PATHS
    assert operation_authority_blockers_v1(op) == ()
    resolved = resolve_promoted_repair_operation_v1(OPERATION_ID)
    assert resolved is execute_control_basis_single_prune_v1


def test_attributed_control_basis_owner_can_receive_executable_directive():
    op = _op()
    directives = build_bounded_repair_directives_v1(
        attribution_finding=_finding(),
        parent_product_state_hash="PRODUCT:PARENT",
        baseline_measurement_report_hash="MEASURE:PARENT",
        proof_probe_fingerprint="PROBE:SEALED",
        operation_candidates=(op,),
    )
    assert len(directives) == 1
    directive = directives[0]
    assert directive.executable is True
    assert directive.selected_owner_id == OWNER_ID
    assert directive.operation.operation_id == OPERATION_ID

    assert build_bounded_repair_directives_v1(
        attribution_finding=_finding(status="abstained"),
        parent_product_state_hash="PRODUCT:PARENT",
        baseline_measurement_report_hash="MEASURE:PARENT",
        proof_probe_fingerprint="PROBE:SEALED",
        operation_candidates=(op,),
    ) == ()


def test_executor_materializes_exact_one_control_child_and_reparents_children():
    source = _skeleton()
    result = execute_control_basis_single_prune_v1(
        operation=_op("mid"),
        skeleton=source,
    )
    child = result.child_skeleton
    by = {j.canonical_joint_id: j for j in child.joints}

    assert len(source.joints) == 4
    assert len(child.joints) == 3
    assert "mid" not in by
    assert by["tip"].parent_canonical_id == "root"
    assert by["tip"].position == (0.0, 2.0, 0.0)
    assert by["side"].position == (1.0, 1.0, 0.0)
    assert child.skeleton_lineage_hash != source.skeleton_lineage_hash

    receipt = result.receipt
    assert receipt.source_control_count == 4
    assert receipt.pruned_control_count == 3
    assert receipt.removed_control_id == "mid"
    assert receipt.replacement_parent_id == "root"
    assert receipt.reparented_child_ids == ("tip",)

    payload = result.to_dict()
    assert payload["product_authority_minted"] is False
    assert "MIRA_SKIN" in payload["downstream_rederivation_required"]
    assert "EXACT_DYNAMIC_PROOF" in payload["downstream_rederivation_required"]


def test_executor_fails_closed_on_root_selection_and_source_lineage_drift():
    with pytest.raises(QualificationError, match="CONTROL_PRUNE_ROOT_FORBIDDEN"):
        execute_control_basis_single_prune_v1(
            operation=_op("root"),
            skeleton=_skeleton(),
        )

    drifted = QualifiedSkeletonIR(
        joints=_skeleton().joints,
        root_id="root",
        qualification_report={"status": "PASS"},
        skeleton_lineage_hash="SK:OTHER",
    )
    with pytest.raises(
        ValueError,
        match="CONTROL_BASIS_REPAIR_SOURCE_SKELETON_DRIFT",
    ):
        execute_control_basis_single_prune_v1(
            operation=_op("mid"),
            skeleton=drifted,
        )


def test_builder_requires_real_signature_control_and_source_identity():
    with pytest.raises(ValueError, match="TARGET_SIGNATURES_EMPTY"):
        build_control_basis_single_prune_operation_v1(
            target_signature_ids=(),
            selected_control_id="mid",
            source_skeleton_hash="SK:PARENT",
        )
    with pytest.raises(ValueError, match="CONTROL_ID_EMPTY"):
        build_control_basis_single_prune_operation_v1(
            target_signature_ids=("SIG:1",),
            selected_control_id="",
            source_skeleton_hash="SK:PARENT",
        )
    with pytest.raises(ValueError, match="SOURCE_SKELETON_HASH_EMPTY"):
        build_control_basis_single_prune_operation_v1(
            target_signature_ids=("SIG:1",),
            selected_control_id="mid",
            source_skeleton_hash="",
        )
