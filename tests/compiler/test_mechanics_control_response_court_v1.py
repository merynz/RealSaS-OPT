from __future__ import annotations

"""Differential mechanics control observability courts (runtime ground truth)."""

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
from compiler.realsas_compiler_core.mechanics_control_response_court_v1 import (
    evaluate_declared_control_response_v1,
)
from compiler.realsas_compiler_core.types import QualificationError

from tests.compiler.test_mechanics_control_solver_v1 import (
    _mesh, _rig, _skin, _mechanics_fixture,
)


def _make():
    mesh=_mesh()
    rig=_rig(mesh)
    skin=_skin(mesh,rig)
    mechanics=_mechanics_fixture(mesh,rig,skin)
    return mesh,rig,skin,mechanics


def _probe(mesh,rig,skin,mechanics,**kwargs):
    return evaluate_declared_control_response_v1(
        mesh,rig,skin,mechanics,
        triangle_vertex_ids=("MV:0","MV:1","MV:2"),
        barycentric=(0.15,0.25,0.60),
        **kwargs,
    )


def test_root_has_measured_effect_and_oracle_graph_does_not_equal_proof():
    mesh,rig,skin,mechanics=_make()
    report=_probe(mesh,rig,skin,mechanics)
    by={x.canonical_control_id:x for x in report.controls}
    assert report.status=="OBSERVED_EFFECTS"
    assert report.graph_status=="AMBIGUOUS"
    assert by["C_ROOT"].status=="OBSERVED_LOCAL_EFFECT"
    assert by["C_ROOT"].measured_max_gain>0.9
    assert by["C_ROOT"].passed_probes>0
    assert all(x.passed_probes+x.failed_probes in (6,12) for x in report.controls)
    assert report.metadata["graphics_authority_claimed"] is False
    assert report.metadata["no_local_effect_does_not_mean_permanently_inert"] is True


def test_pole_declared_reachable_but_straight_rest_may_have_no_local_response():
    mesh,rig,skin,mechanics=_make()
    report=_probe(mesh,rig,skin,mechanics)
    by={x.canonical_control_id:x for x in report.controls}
    assert "C_POLE" in by
    assert by["C_POLE"].graph_support_mass>0.5
    # Local finite differences only, so this must not claim pole is a
    # permanently inert control if the rest limb is straight.
    assert by["C_POLE"].status in ("NO_LOCAL_EFFECT_OBSERVED","INCONCLUSIVE",
                                     "OBSERVED_LOCAL_EFFECT")
    assert report.metadata["local_probes_are_not_source_authored_intent"] is True


def test_unknown_and_unreachable_baselines_fail_closed():
    mesh,rig,skin,mechanics=_make()
    with pytest.raises(QualificationError,match="BASE_CONTROL_UNKNOWN"):
        _probe(mesh,rig,skin,mechanics,baseline_overrides=(
            ControlTransformIRV1("UNDECLARED"),))
    with pytest.raises(QualificationError,match="BASELINE_NOT_SOLVED"):
        _probe(mesh,rig,skin,mechanics,baseline_overrides=(
            ControlTransformIRV1("C_IK",translation=(15.,0.,0.)),))
    with pytest.raises(QualificationError,match="CONTROL_RESOURCE_BOUND"):
        _probe(mesh,rig,skin,mechanics,maximum_controls=1)


def test_measured_control_court_requires_real_triangle_and_valid_policy():
    mesh,rig,skin,mechanics=_make()
    from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_lineage_hash_v1
    broken=replace(mesh,faces=(),mesh_lineage_hash="")
    broken=replace(broken,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(broken))
    broken_rig=replace(rig,mesh_lineage_hash=broken.mesh_lineage_hash)
    broken_skin=replace(skin,mesh_lineage_hash=broken.mesh_lineage_hash)
    broken_mechanics=replace(mechanics,mesh_lineage_hash=broken.mesh_lineage_hash)
    with pytest.raises(QualificationError,match="HIT_NOT_CANONICAL_FACE"):
        evaluate_declared_control_response_v1(
            broken,broken_rig,broken_skin,broken_mechanics,
            triangle_vertex_ids=("MV:0","MV:1","MV:2"),
            barycentric=(0.15,0.25,0.60),
        )
    with pytest.raises(QualificationError,match="PROBE_POLICY_INVALID"):
        _probe(mesh,rig,skin,mechanics,translation_step_world=float("nan"))
