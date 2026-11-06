from __future__ import annotations

"""Prepared sparse LBS parity/lineage/authority courts; no timing threshold."""

from dataclasses import replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshVertexIR, canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import CanonicalSkinRowIRV2
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1, deform_mesh_lbs_v1, solve_control_pose_v1,
)
from compiler.realsas_compiler_core.mechanics_prepared_skin_v1 import (
    prepare_mechanics_skin_deformer_v1, deform_prepared_mechanics_skin_v1,
)
from compiler.realsas_compiler_core.types import QualificationError

from tests.compiler.test_mechanics_control_solver_v1 import _mesh,_rig,_skin


def _objects():
    mesh=_mesh()
    rig=_rig(mesh)
    skin=_skin(mesh,rig)
    prepared=prepare_mechanics_skin_deformer_v1(mesh,rig,skin)
    return mesh,rig,skin,prepared


def _assert_parity(mesh,rig,skin,prepared,pose):
    original=deform_mesh_lbs_v1(mesh,rig,skin,pose)
    quick=deform_prepared_mechanics_skin_v1(
        prepared,pose,
        expected_mesh_lineage_hash=mesh.mesh_lineage_hash,
        expected_skin_lineage_hash=skin.skin_lineage_hash,
        expected_deform_structure_hash=rig.deform_structure_hash,
    )
    assert quick.canonical_mesh_vertex_ids==original.canonical_mesh_vertex_ids
    assert np.allclose(quick.positions,original.positions,atol=1e-11,rtol=0.)
    assert quick.metadata["visual_rebind_performed"] is False
    assert prepared.metadata["training_authorized"] is False


def test_prepared_canonical_root_and_ik_equal_legacy_lbs():
    mesh,rig,skin,prepared=_objects()
    assert prepared.vertex_count==3
    assert prepared.influence_count==3
    assert prepared.rest_offset_vectors.flags.writeable is False
    assert prepared.influence_weights.flags.writeable is False
    for transforms in (
        (),
        (ControlTransformIRV1("C_ROOT",translation=(1.3,-0.4,0.7)),),
        (ControlTransformIRV1("C_IK",translation=(0.8,-0.5,0.1)),),
    ):
        pose=solve_control_pose_v1(rig,transforms)
        _assert_parity(mesh,rig,skin,prepared,pose)


def test_prepared_multi_influence_on_512_vertices_matches_legacy():
    mesh,rig,_,_= _objects()
    from tests.compiler.test_mechanics_control_solver_v1 import _support
    rng=np.random.default_rng(171)
    verts=tuple(CanonicalMeshVertexIR(
        canonical_mesh_vertex_id=f"V:{i}",support_binding=_support(i),
        component_id="generic",P=tuple(map(float,rng.normal(size=3))),
        source_candidate_vertex_id=f"CV:{i}",
    ) for i in range(512))
    new_mesh=replace(mesh,vertices=verts,faces=(),edges=(),mesh_lineage_hash="")
    new_mesh=replace(new_mesh,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(new_mesh))
    new_rig=replace(rig,mesh_lineage_hash=new_mesh.mesh_lineage_hash)
    rows=[]
    for v in verts:
        weights=rng.dirichlet((0.7,1.3,0.4))
        rows.append(CanonicalSkinRowIRV2(
            v.canonical_mesh_vertex_id,
            tuple((f"J{i}",float(w)) for i,w in enumerate(weights)),
            v.source_candidate_vertex_id,
        ))
    original_skin=_skin(mesh,rig)
    new_skin=replace(original_skin,rows=tuple(rows),
                     mesh_lineage_hash=new_mesh.mesh_lineage_hash)
    prepared=prepare_mechanics_skin_deformer_v1(new_mesh,new_rig,new_skin)
    assert prepared.influence_count==1536
    pose=solve_control_pose_v1(new_rig,(
        ControlTransformIRV1("C_ROOT",translation=(-0.5,0.9,0.3)),))
    _assert_parity(new_mesh,new_rig,new_skin,prepared,pose)
    with pytest.raises(ValueError):
        prepared.influence_weights[0]=0.1


def test_prepared_rejects_lineage_skin_simplex_and_duplicate_influence():
    mesh,rig,skin,prepared=_objects()
    with pytest.raises(QualificationError,match="LINEAGE_DRIFT"):
        prepare_mechanics_skin_deformer_v1(mesh,replace(rig,mesh_lineage_hash="BAD"),skin)
    bad=replace(skin,rows=(
        replace(skin.rows[0],influences=(("J0",0.25),)),*skin.rows[1:]))
    with pytest.raises(QualificationError,match="SKIN_SIMPLEX_INVALID"):
        prepare_mechanics_skin_deformer_v1(mesh,rig,bad)
    duplicate=replace(skin,rows=(
        replace(skin.rows[0],influences=(("J0",0.5),("J0",0.5))),*skin.rows[1:]))
    with pytest.raises(QualificationError,match="SKIN_JOINT_INVALID"):
        prepare_mechanics_skin_deformer_v1(mesh,rig,duplicate)
    with pytest.raises(QualificationError,match="INFLUENCE_RESOURCE_BOUND"):
        prepare_mechanics_skin_deformer_v1(mesh,rig,skin,maximum_influences=2)


def test_prepared_pose_drift_fails_closed_before_deform():
    mesh,rig,skin,prepared=_objects()
    pose=solve_control_pose_v1(rig,())
    with pytest.raises(QualificationError,match="POSE_OR_BINDING_DRIFT"):
        deform_prepared_mechanics_skin_v1(
            prepared,replace(pose,control_program_hash="stale"))
    with pytest.raises(QualificationError,match="POSE_OR_BINDING_DRIFT"):
        deform_prepared_mechanics_skin_v1(
            prepared,pose,expected_skin_lineage_hash="not-the-skin")
    with pytest.raises(QualificationError,match="POSE_JOINT_COVERAGE_DRIFT"):
        deform_prepared_mechanics_skin_v1(
            prepared,replace(pose,joints=pose.joints[:2]))


def test_prepared_live_motion_and_intervention_runtime_match_default_kernel():
    from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
        MechanicsInterventionStepV1,
        evaluate_mechanics_intervention_frame_v1,
        evaluate_mechanics_intervention_sequence_v1,
    )
    from tests.compiler.test_mechanics_control_solver_v1 import _mechanics_fixture
    mesh,rig,skin,prepared=_objects()
    mechanics=_mechanics_fixture(mesh,rig,skin)
    transforms=(ControlTransformIRV1("C_ROOT",translation=(0.8,-0.3,0.25)),)
    old=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,overrides=transforms,
    )
    fast=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,overrides=transforms,prepared_skin=prepared,
    )
    assert old.status==fast.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert np.allclose(old.deformation.positions,fast.deformation.positions,atol=1e-11)
    assert fast.metadata["deformer_kernel"]=="PREPARED_SPARSE_NUMPY_RESEARCH_ONLY"
    assert old.metadata["deformer_kernel"]=="LEGACY_LBS_V1"
    steps=tuple(MechanicsInterventionStepV1(
        time_seconds=float(i)*0.1,
        overrides=(ControlTransformIRV1("C_ROOT",translation=(float(i)*0.1,0.,0.)),),
    ) for i in range(3))
    old_seq=evaluate_mechanics_intervention_sequence_v1(
        mesh,rig,skin,mechanics,steps,
    )
    fast_seq=evaluate_mechanics_intervention_sequence_v1(
        mesh,rig,skin,mechanics,steps,prepared_skin=prepared,
    )
    assert old_seq.status==fast_seq.status
    assert fast_seq.max_vertex_speed==pytest.approx(old_seq.max_vertex_speed,abs=1e-8)
    assert fast_seq.max_vertex_acceleration==pytest.approx(old_seq.max_vertex_acceleration,abs=1e-8)


def test_prepared_skin_cannot_be_written_or_reenabled():
    mesh,rig,skin,prepared=_objects()
    for arr in (
        prepared.rest_offset_vectors,
        prepared.influence_weights,
        prepared.influence_vertex_indices,
        prepared.influence_joint_indices,
    ):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr.setflags(write=True)
