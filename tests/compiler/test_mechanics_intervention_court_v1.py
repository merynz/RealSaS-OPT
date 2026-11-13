from __future__ import annotations

"""Non-rest mechanical response is independently executable, not source truth."""

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    mint_canonical_mechanics_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalAttachmentAnchorIRV2, _hash_control_program,
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanics_intervention_court_v1 import (
    compile_mechanics_intervention_court_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.platform_worker.mechanics_executor_v1 import (
    execute_mechanics_capability_v1,
)
from tests.compiler.test_mechanics_cas_worker_main_v1 import _fixture


def _parents(with_socket=False):
    _, _, _, _, mesh, rig, skin = _fixture()
    if with_socket:
        anchor = CanonicalAttachmentAnchorIRV2(
            canonical_anchor_id="A:socket",
            parent_canonical_control_id=rig.controls[0].canonical_control_id,
            local_P=(0.0, 0.25, 0.0),
            source_proposal_anchor_id="P:A:test",
        )
        rig = replace(rig, attachment_anchors=(anchor,),
                      control_program_hash="", rig_lineage_hash="")
        rig = replace(rig, control_program_hash=_hash_control_program(rig))
        rig = replace(rig, rig_lineage_hash=canonical_rig_lineage_hash_v2(rig))
    mechanics = mint_canonical_mechanics_v1(mesh, rig, skin)
    return mesh, rig, skin, mechanics


def test_root_drag_moves_actual_skinned_mesh_and_bound_control_socket():
    mesh, rig, skin, mechanics = _parents(with_socket=True)
    court = compile_mechanics_intervention_court_v1(mesh, rig, skin, mechanics)
    assert court.status == "MEASURED_COMPILER_CONTROL_INTERVENTIONS_NOT_SOURCE_BEHAVIOR"
    assert court.control_count == len(rig.controls) == 1
    assert court.anchor_count == 1
    assert court.full_mesh_deformation_probe_count == 1
    assert court.full_mesh_deformation_complete is True
    assert len(court.probes) == 1
    probe = court.probes[0]
    assert probe.control_id == rig.controls[0].canonical_control_id
    assert probe.joint_motion_max > 0
    assert probe.vertex_motion_max > 0
    assert probe.anchor_motion_max > 0
    assert probe.anchor_count == 1
    assert court.observed_joint_motion is True
    assert court.observed_mesh_motion is True
    assert court.court_hash == content_sha256({
        k: v for k, v in court.to_dict().items() if k != "court_hash"
    })
    assert court.source_attachment_contact_qualified is False
    assert court.learned_secondary_motion_qualified is False
    assert court.collision_qualified is False
    assert court.edit_consequence_qualified is False
    assert court.product_revision_minted is False


def test_wrong_parent_and_control_probe_budget_are_fail_closed():
    mesh, rig, skin, mechanics = _parents()
    wrong = replace(mechanics, mesh_lineage_hash="f"*64)
    with pytest.raises(QualificationError,match="CANONICAL_PARENT_LINEAGE_DRIFT"):
        compile_mechanics_intervention_court_v1(mesh, rig, skin, wrong)
    with pytest.raises(QualificationError,match="PROBE_POLICY_OR_BUDGET_INVALID"):
        compile_mechanics_intervention_court_v1(
            mesh, rig, skin, mechanics, max_probed_controls=0,
        )
    with pytest.raises(QualificationError,match="PROBE_POLICY_OR_BUDGET_INVALID"):
        compile_mechanics_intervention_court_v1(
            mesh, rig, skin, mechanics, relative_translation=float("nan"),
        )


def test_mechanics_worker_emits_optional_cas_intervention_without_mutating_rig():
    request, read, put, outputs, mesh, rig, skin = _fixture()
    request["capability_metadata"]["control_intervention_policy"] = {
        "enabled": True, "max_probed_controls": 128,
        "max_full_mesh_deformation_probes": 4,
        "relative_translation": 0.02, "max_constraint_residual": 0.001,
    }
    result = execute_mechanics_capability_v1(
        request, read_object=read, put_bytes=put,
    )
    assert result["status"] == "PASS"
    assert len(result["outputs"]) == len(outputs) == 2
    assert result["outputs"][1]["artifact_type"] == "RealSaS.MechanicsInterventionCourt"
    assert result["outputs"][1]["schema_version"] == "v1"
    assert result["outputs"][1]["content_sha256"] == sha256(outputs[1]).hexdigest()
    receipt = json.loads(outputs[1])
    assert receipt["canonical_rig_lineage_hash"] == rig.rig_lineage_hash
    assert receipt["canonical_skin_lineage_hash"] == skin.skin_lineage_hash
    assert receipt["control_count"] == 1
    assert receipt["observed_mesh_motion"] is True
    assert receipt["source_attachment_contact_qualified"] is False
    assert receipt["learned_secondary_motion_qualified"] is False
    assert result["diagnostics"]["intervention_court_hash"] == receipt["court_hash"]
    assert result["diagnostics"]["product_revision_pass_claimed"] is False
