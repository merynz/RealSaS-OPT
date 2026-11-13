from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    mint_canonical_mechanics_v1,
)
from compiler.realsas_compiler_core.mechanics_dynamic_integrity_court_v1 import (
    compile_mechanics_dynamic_integrity_court_v1,
    validate_mechanics_dynamic_integrity_court_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.platform_worker.mechanics_executor_v1 import (
    execute_mechanics_capability_v1,
)
from compiler.realsas_compiler_services.platform_worker.product_downstream_executor_v1 import (
    GRAPHICS_CONTRACT, MOTION_CONTRACT, PROOF_CONTRACT,
    execute_graphics_capability_v1, execute_motion_capability_v2,
    execute_proof_capability_v1,
)
from tests.compiler.test_mechanics_cas_worker_main_v1 import _fixture
from tests.compiler.test_product_downstream_cas_v1 import _requests


def _canonical():
    _, _, _, _, mesh, rig, skin = _fixture()
    mechanics = mint_canonical_mechanics_v1(mesh, rig, skin)
    return mesh, rig, skin, mechanics


def test_real_nonrest_collision_is_measured_on_every_face_and_control():
    mesh, rig, skin, mechanics = _canonical()
    court = compile_mechanics_dynamic_integrity_court_v1(
        mesh, rig, skin, mechanics,
    )
    assert court.examined_face_count == len(mesh.faces)
    assert court.examined_control_count == len(rig.controls)
    assert court.complete_face_and_control_coverage is True
    assert len(court.probes) == 3 * len(rig.controls)
    assert court.maximum_new_intersection_count >= 0
    assert court.no_new_self_intersections == (
        court.maximum_new_intersection_count == 0
    )
    assert all(p.all_vertices_finite and p.all_socket_positions_finite
               for p in court.probes)
    assert all(len(p.pose_hash) == 64 and len(p.socket_hash) == 64
               for p in court.probes)
    assert court.source_contact_qualified is False
    assert court.learned_secondary_motion_qualified is False
    assert court.native_render_parity_qualified is False
    assert not court.product_revision_minted
    assert validate_mechanics_dynamic_integrity_court_v1(
        court.to_dict(), mesh=mesh, rig=rig,
        skin=skin, mechanics=mechanics,
    ) == court


def test_court_rejects_rehashed_collision_false_pass():
    mesh, rig, skin, mechanics = _canonical()
    receipt = compile_mechanics_dynamic_integrity_court_v1(
        mesh, rig, skin, mechanics,
    ).to_dict()
    forged = deepcopy(receipt)
    forged["no_new_self_intersections"] = True
    forged["maximum_new_intersection_count"] = 0
    forged["probes"][0]["new_intersection_count"] = 0
    # Even an internally consistent content hash is not sufficient.
    forged["court_hash"] = "0" * 64
    with pytest.raises(QualificationError, match="REPLAY_BYTES_OR_METRICS_DRIFT"):
        validate_mechanics_dynamic_integrity_court_v1(
            forged, mesh=mesh, rig=rig,
            skin=skin, mechanics=mechanics,
        )


def test_resource_budget_never_converts_partial_collision_scan_into_pass():
    mesh, rig, skin, mechanics = _canonical()
    if len(mesh.faces) > 1:
        court = compile_mechanics_dynamic_integrity_court_v1(
            mesh, rig, skin, mechanics, max_faces=1,
        )
        assert court.complete_face_and_control_coverage is False
        assert court.no_new_self_intersections is False
        assert court.examined_face_count == 0
        assert court.status == "INCOMPLETE_RESOURCE_BUDGET_NOT_COLLISION_PASS"


def test_mechanics_worker_persists_dynamic_receipt_and_proof_recomputes_it():
    request, read_object, put_bytes, outputs, mesh, rig, skin = _fixture()
    request["capability_metadata"]["dynamic_integrity_policy"] = {
        "enabled": True, "max_faces": 1024, "max_controls": 128,
        "relative_translation": .02, "max_constraint_residual": .001,
    }
    result = execute_mechanics_capability_v1(
        request, read_object=read_object, put_bytes=put_bytes,
    )
    assert result["status"] == "PASS"
    assert len(result["outputs"]) == 2
    row = result["outputs"][1]
    assert row["artifact_type"] == "RealSaS.MechanicsDynamicIntegrityCourt"
    assert result["diagnostics"]["dynamic_integrity_court_hash"]
    assert result["diagnostics"]["source_contact_qualified"] is False
    assert result["diagnostics"]["learned_secondary_motion_qualified"] is False

    cas, mesh, rig, skin, mechanics, req = _requests()
    court = compile_mechanics_dynamic_integrity_court_v1(
        mesh, rig, skin, mechanics,
    )
    court_ref = cas.put(court, "RealSaS.MechanicsDynamicIntegrityCourt", "v1")
    graphics = execute_graphics_capability_v1(
        req("graphics", GRAPHICS_CONTRACT),
        read_object=cas.read_object, put_bytes=cas.put_bytes,
    )
    motion = execute_motion_capability_v2(
        req("motion", MOTION_CONTRACT),
        read_object=cas.read_object, put_bytes=cas.put_bytes,
    )
    proof_req = req("proof", PROOF_CONTRACT)
    proof_req["capability_metadata"]["dynamic_integrity_court_required"] = True
    proof_req["input_artifacts"] += [
        court_ref, graphics["outputs"][0], motion["outputs"][1],
    ]
    proof = execute_proof_capability_v1(
        proof_req, read_object=cas.read_object,
        put_bytes=cas.put_bytes,
    )
    assert proof["status"] == "FAIL"  # other independent witnesses absent
    assert proof["diagnostics"]["dynamic_integrity_recomputed_in_proof"] is True
    assert proof["diagnostics"]["dynamic_integrity_court_hash"] == court.court_hash
    assert proof["diagnostics"]["dynamic_collision_nonregression_qualified"] == (
        court.complete_face_and_control_coverage and court.no_new_self_intersections
    )
    # Tamper a metric and rehash the JSON / CAS: recomputation still catches it.
    fraud = court.to_dict()
    fraud["maximum_new_intersection_count"] = 99
    fraud["court_hash"] = "a" * 64
    fake_ref = cas.put(fraud, "RealSaS.MechanicsDynamicIntegrityCourt", "v1")
    proof_req["input_artifacts"] = [
        fake_ref if r["artifact_type"] == "RealSaS.MechanicsDynamicIntegrityCourt" else r
        for r in proof_req["input_artifacts"]
    ]
    with pytest.raises(QualificationError, match="REPLAY_BYTES_OR_METRICS_DRIFT"):
        execute_proof_capability_v1(
            proof_req, read_object=cas.read_object,
            put_bytes=cas.put_bytes,
        )
