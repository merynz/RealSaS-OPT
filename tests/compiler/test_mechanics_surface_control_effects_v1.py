from __future__ import annotations

"""Mechanics control-effect relation courts, independent of learned models."""

from dataclasses import replace
import pytest

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import CanonicalRigControlIRV2
from compiler.realsas_compiler_core.mechanics_surface_control_effects_v1 import (
    surface_control_effect_evidence_v1,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import CanonicalSkinRowIRV2
from compiler.realsas_compiler_core.types import QualificationError

from tests.compiler.test_mechanics_control_solver_v1 import _mesh, _rig, _skin


def _setup():
    mesh=_mesh()
    rig=_rig(mesh)
    skin=_skin(mesh,rig)
    return rig,skin


def _pick(rig,skin,*,minimum_support=1e-8):
    return surface_control_effect_evidence_v1(
        rig,skin,
        triangle_vertex_ids=("MV:0","MV:1","MV:2"),
        barycentric=(0.15,0.25,0.60),
        minimum_support=minimum_support,
    )


def test_root_without_skin_weight_still_reaches_all_deform_descendants():
    rig,skin=_setup()
    result=_pick(rig,skin)
    candidates={r.canonical_control_id:r for r in result.candidates}
    assert result.status=="AMBIGUOUS"
    # ROOT directly targets J0; only one vertex has J0 influence, but the
    # declared subtree includes J1/J2. Its true influence can reach any.
    assert candidates["C_ROOT"].support_mass==pytest.approx(1.0)
    assert candidates["C_ROOT"].effect_paths==("DECLARED_DIRECT_SUBTREE",)
    assert candidates["C_ROOT"].influence_joint_ids==("J0","J1","J2")
    assert result.metadata["graphics_or_layer_authority_claimed"] is False


def test_unskinned_ik_pole_can_influence_surface_via_declared_constraint():
    rig,skin=_setup()
    result=_pick(rig,skin)
    candidates={r.canonical_control_id:r for r in result.candidates}
    assert "C_POLE" in candidates
    pole=candidates["C_POLE"]
    assert pole.control_kind=="IK_POLE"
    assert pole.support_mass==pytest.approx(1.0)
    assert pole.causal_constraint_ids==("Q0",)
    assert pole.effect_paths==("DECLARED_CONSTRAINT_REACHABILITY",)
    assert rig.controls[2].target_canonical_joint_ids==()


def test_no_constraint_means_pole_does_not_gain_invented_authority():
    rig,skin=_setup()
    rig=replace(rig,constraints=())
    result=_pick(rig,skin)
    assert "C_POLE" not in {x.canonical_control_id for x in result.candidates}
    assert result.abstained_control_count==1


def test_bare_unknown_control_type_does_not_synthesize_mechanical_role():
    rig,skin=_setup()
    rig=replace(rig,controls=rig.controls+(
        CanonicalRigControlIRV2("C_UNKNOWN","UNPROVEN_KIND",("J2",),None,
                               (0.,2.,0.),"teacher-provenance"),
    ))
    result=_pick(rig,skin)
    assert "C_UNKNOWN" not in {x.canonical_control_id for x in result.candidates}
    assert result.abstained_control_count==1


def test_pick_fails_closed_on_invalid_skin_and_lineage_drift():
    rig,skin=_setup()
    with pytest.raises(QualificationError,match="LINEAGE_DRIFT"):
        _pick(rig,replace(skin,deform_structure_hash="not_same"))
    with pytest.raises(QualificationError,match="BARYCENTRIC_INVALID"):
        surface_control_effect_evidence_v1(
            rig,skin,triangle_vertex_ids=("MV:0","MV:1","MV:2"),
            barycentric=(0.5,0.5,0.5),
        )
    corrupt=replace(skin,rows=(replace(skin.rows[0],influences=(("J0",0.1),)),)
                    +skin.rows[1:])
    with pytest.raises(QualificationError,match="SKIN_SIMPLEX_INVALID"):
        _pick(rig,corrupt)
    with pytest.raises(QualificationError,match="SURFACE_VERTEX_UNKNOWN"):
        surface_control_effect_evidence_v1(
            rig,skin,triangle_vertex_ids=("MV:0","MV:1","MISSING"),
            barycentric=(0.1,0.3,0.6),
        )


def test_control_with_no_effect_on_hit_abstains_not_hallucinated():
    rig,skin=_setup()
    # FK on J2 can affect a surface at J2, but not one attached to J0.
    extra=CanonicalRigControlIRV2(
        "C_TIP_ONLY","FK",("J2",),None,(0.,2.,0.),"P4",
    )
    rig=replace(rig,controls=rig.controls+(extra,))
    result=surface_control_effect_evidence_v1(
        rig,skin,triangle_vertex_ids=("MV:0","MV:1","MV:2"),
        barycentric=(1.,0.,0.),
    )
    assert "C_TIP_ONLY" not in {x.canonical_control_id for x in result.candidates}
