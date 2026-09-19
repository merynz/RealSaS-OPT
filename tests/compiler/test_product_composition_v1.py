from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import pytest

from compiler.realsas_compiler_core.observation_authority_v1 import (
    QualifiedObservationViewIR,build_qualified_observation_set,
)
from compiler.realsas_compiler_core.product_composition_v1 import (
    build_composition_set,validate_composition_set,
)
from compiler.realsas_compiler_core.product_authority_v1 import PresentationSlotIR
from compiler.realsas_compiler_core.types import QualificationError


def _fixture():
    observations=build_qualified_observation_set(tuple(
        QualifiedObservationViewIR(
            i,64,64,f"obs-{i}",f"{i+16:064x}",f"{i+32:064x}",f"cam-{i}",
            "PASS",(f"e-{i}",)
        )
        for i in range(8)
    ))
    structure=SimpleNamespace(
        structure_lineage_hash="structure-hash",
        slots=(
            PresentationSlotIR("slot-b","root",1,"att-b",("ORDER",)),
            PresentationSlotIR("slot-a","root",0,"att-a",("ORDER",)),
        ),
    )
    mesh=SimpleNamespace(
        mesh_lineage_hash="mesh-hash",
        qualification_report={"g5_evidence_hash":"g5-hash"},
    )
    state=SimpleNamespace(product_state_hash="state-hash")
    return observations,structure,mesh,state


def test_composition_uses_zbuffer_and_keeps_slot_order_nonphysical():
    obs,structure,mesh,state=_fixture()
    value=build_composition_set(
        mesh=mesh,observation_set=obs,presentation_structure=structure,product_state=state
    )
    assert len(value.views)==8
    assert all(row.slot_order==("slot-a","slot-b") for row in value.views)
    assert all(row.physical_occlusion_rule=="CANONICAL_Z_BUFFER_VISIBLE_OWNER_V1" for row in value.views)
    assert all(row.slot_order_role=="UI_SETUP_ONLY__NOT_PHYSICAL_OCCLUSION" for row in value.views)
    validate_composition_set(
        value,mesh=mesh,observation_set=obs,presentation_structure=structure,product_state=state
    )


def test_composition_rejects_slot_order_as_hidden_physical_occlusion_authority():
    obs,structure,mesh,state=_fixture()
    value=build_composition_set(
        mesh=mesh,observation_set=obs,presentation_structure=structure,product_state=state
    )
    first=replace(value.views[0],slot_order_role="PHYSICAL_OCCLUSION",composition_binding_hash="")
    from compiler.realsas_compiler_core.product_composition_v1 import composition_view_hash,composition_set_hash
    first=replace(first,composition_binding_hash=composition_view_hash(first))
    bad=replace(value,views=(first,*value.views[1:]),composition_set_hash="")
    bad=replace(bad,composition_set_hash=composition_set_hash(bad))
    with pytest.raises(QualificationError,match="SLOT_ORDER_ROLE_DRIFT"):
        validate_composition_set(
            bad,mesh=mesh,observation_set=obs,presentation_structure=structure,product_state=state
        )


def test_composition_is_bound_to_exact_g5_visible_owner_evidence():
    obs,structure,mesh,state=_fixture()
    value=build_composition_set(
        mesh=mesh,observation_set=obs,presentation_structure=structure,product_state=state
    )
    changed=SimpleNamespace(mesh_lineage_hash="mesh-hash",qualification_report={"g5_evidence_hash":"other"})
    with pytest.raises(QualificationError,match="G5_EVIDENCE_DRIFT"):
        validate_composition_set(
            value,mesh=changed,observation_set=obs,presentation_structure=structure,product_state=state
        )
