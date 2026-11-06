from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    canonical_mechanics_from_dict_v1, mint_canonical_mechanics_v1,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    canonical_skin_lineage_hash_v2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1, solve_control_pose_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_canonical_mechanics_ir_v1 import (
    _mesh as _seal_mesh, _rig as _seal_rig, _skin as _seal_skin,
)
from tests.compiler.test_mechanics_control_solver_v1 import (
    _mesh as _solve_mesh, _rig as _solve_rig,
)


def _parents():
    mesh = _seal_mesh()
    rig = _seal_rig(mesh)
    skin = _seal_skin(mesh, rig)
    return mesh, rig, skin


def test_main_contract_to_baseline_mechanics_mints_only_rest_qualification():
    mesh, rig, skin = _parents()
    mechanics = mint_canonical_mechanics_v1(mesh, rig, skin)
    assert mechanics.mesh_lineage_hash == mesh.mesh_lineage_hash
    assert mechanics.rig_lineage_hash == rig.rig_lineage_hash
    assert mechanics.skin_lineage_hash == skin.skin_lineage_hash
    assert mechanics.metadata["constraint_execution_semantics"] == "MECHANICS_CONTROL_SOLVER_V1"
    assert mechanics.metadata["runtime_control_pose_solution_uses_same_solver"] is True
    assert mechanics.metadata["motion_authority_claimed"] is False
    assert mechanics.metadata["graphics_authority_claimed"] is False
    assert canonical_mechanics_from_dict_v1(mechanics.to_dict()) == mechanics


def test_mechanics_rejects_independently_sealed_skin_bound_to_wrong_deform_graph():
    mesh, rig, skin = _parents()
    changed = replace(skin, deform_structure_hash="f"*64)
    changed = replace(
        changed, skin_lineage_hash=canonical_skin_lineage_hash_v2(changed)
    )
    with pytest.raises(QualificationError, match="MECHANICS_DEFORM_STRUCTURE_DRIFT"):
        mint_canonical_mechanics_v1(mesh, rig, changed)


def test_mechanics_seal_detects_schedule_tampering():
    mesh, rig, skin = _parents()
    original = mint_canonical_mechanics_v1(mesh, rig, skin).to_dict()
    original["constraint_schedule"] = ["UNDECLARED"]
    with pytest.raises(QualificationError, match="MECHANICS_LINEAGE_HASH_DRIFT"):
        canonical_mechanics_from_dict_v1(original)


def test_old_baseline_pose_solver_rejects_unknown_user_control_instead_of_synthesis():
    mesh = _solve_mesh()
    rig = _solve_rig(mesh)
    with pytest.raises(QualificationError, match="MECHANICS_CONTROL_STATE_UNKNOWN"):
        solve_control_pose_v1(
            rig,
            (ControlTransformIRV1("SYNTHETIC_UNDECLARED", translation=(1.,0.,0.)),),
        )
