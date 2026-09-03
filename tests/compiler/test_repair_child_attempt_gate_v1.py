from types import SimpleNamespace as NS

import compiler.realsas_compiler_core.repair_attempt as gate
from compiler.realsas_compiler_core.repair_registry import (
    REPAIR_OPERATION_AUTHORITY,
    RepairOperationAuthorityRecordV1,
    operation_authority_blockers_v1,
)


def _product(state="P0", parent=None, mesh="M0", motion="MO0"):
    mechanical=NS(
        surface=NS(geometry_lineage_hash="S"),
        skeleton=NS(skeleton_lineage_hash="R"),
        skin=NS(skin_lineage_hash="W"),
        mechanical_equivalence_class="THREE_D_EQUIVALENT_MECHANICS",
        full_3d_reconstruction_authority=False,
    )
    comp=NS(
        component_id="body",
        mesh=NS(mesh_lineage_hash=mesh),
        mesh_skin=NS(mesh_skin_lineage_hash="MS"),
        appearance=NS(appearance_lineage_hash="A"),
        setup_order=0,
        coverage_classification="VISIBLE_CORE",
        default_visible=True,
        metadata={},
        completions=(),
    )
    return NS(
        schema_version="RealSaS.CanonicalPuppetGraph.v3",
        representation_class="DIRECTIONAL_2D_2P5D_PUPPET",
        mechanical_equivalence_class="THREE_D_EQUIVALENT_MECHANICS",
        full_3d_reconstruction_authority=False,
        export_contract_version="RealSaS.RuntimePackageIR.v1",
        product_state_hash=state,
        parent_state_hash=parent,
        mechanical_state=mechanical,
        directional_renderables=NS(directions=(NS(view_index=0,camera_binding_hash="CAM",metadata={},components=(comp,)),)),
        capability_contract=NS(capability_contract_hash="CAP"),
        motion_state=NS(motion_state_hash=motion),
        runtime_policy={},
        editable_metadata={},
    )


def _operation(opid="TEST_MESH"):
    return NS(
        operation_id=opid,
        owner_id="mesh",
        operation_family="test_mesh",
        qualification_hash="Q",
        allowed_change_paths=("directional_visual.direction.*.component.*.mesh",),
    )


def _directive(operation=None):
    return NS(
        directive_id="D",
        parent_product_state_hash="P0",
        selected_owner_id="mesh",
        operation=operation or _operation(),
        executable=True,
    )


def _bypass_ontology_only_for_unit_fixture(monkeypatch):
    monkeypatch.setattr(gate, "validate_product_ontology", lambda product: None)


def _install_test_authority(monkeypatch):
    records=dict(REPAIR_OPERATION_AUTHORITY)
    records["TEST_MESH"]=RepairOperationAuthorityRecordV1(
        "TEST_MESH","mesh","test_mesh","CANONICAL_MAINLINE_EXECUTABLE","Q",
        ("directional_visual.direction.*.component.*.mesh",),"test","test-only",
    )
    monkeypatch.setattr("compiler.realsas_compiler_core.repair_registry.REPAIR_OPERATION_AUTHORITY", records)


def test_historical_repair_operations_are_fail_closed():
    assert REPAIR_OPERATION_AUTHORITY
    for oid, rec in REPAIR_OPERATION_AUTHORITY.items():
        op=NS(
            operation_id=oid, owner_id=rec.owner_id, operation_family=rec.operation_family,
            qualification_hash=rec.qualification_hash, allowed_change_paths=rec.allowed_change_patterns,
        )
        assert operation_authority_blockers_v1(op)
        assert rec.current_main_status != "CANONICAL_MAINLINE_EXECUTABLE"


def test_semantic_delta_localizes_mesh_leaf():
    assert gate.product_semantic_delta_v1(_product(),_product("P1","P0",mesh="M1")) == (
        "directional_visual.direction.0.component.body.mesh",
    )


def test_unregistered_operation_cannot_validate_child(monkeypatch):
    _bypass_ontology_only_for_unit_fixture(monkeypatch)
    audit=gate.audit_repair_child_attempt_v1(_product(),_product("P1","P0",mesh="M1"),_directive())
    assert audit.status == "REJECTED"
    assert "repair_operation_not_in_current_authority_registry" in audit.blockers


def test_registered_same_parent_owner_scoped_child_passes_unit_gate(monkeypatch):
    _bypass_ontology_only_for_unit_fixture(monkeypatch)
    _install_test_authority(monkeypatch)
    audit=gate.audit_repair_child_attempt_v1(_product(),_product("P1","P0",mesh="M1"),_directive())
    assert audit.status == "PASS"
    assert audit.actual_changed_paths == ("directional_visual.direction.0.component.body.mesh",)


def test_wrong_parent_or_cross_scope_change_rejected(monkeypatch):
    _bypass_ontology_only_for_unit_fixture(monkeypatch)
    _install_test_authority(monkeypatch)
    wrong_parent=gate.audit_repair_child_attempt_v1(_product(),_product("P1","WRONG",mesh="M1"),_directive())
    assert "repair_child_parent_lineage_mismatch" in wrong_parent.blockers
    cross_scope=gate.audit_repair_child_attempt_v1(_product(),_product("P2","P0",mesh="M1",motion="MO1"),_directive())
    assert cross_scope.status == "REJECTED"
    assert any(x.startswith("repair_child_actual_change_outside_authorized_scope:") for x in cross_scope.blockers)
