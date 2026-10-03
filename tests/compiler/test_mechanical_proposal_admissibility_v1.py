from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.mesh.mechanical_proposal_admissibility_v1 import (
    MechanicalProposalAdmissibilityGuardV1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR,
    build_mesh_qualification_policy,
)


def _policy():
    return build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.125,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH",0.999,0.999,0.00025,0.0005),
            CarrierCoverageThresholdIR("PLANAR",0.99,0.995,0.0025,0.0025),
        ),
        g3_min_dynamic_area_ratio=0.5,
        g3_max_dynamic_area_ratio=2.0,
        g3_max_dynamic_condition_number=2.0,
    )


def _guard(posed):
    g=MechanicalProposalAdmissibilityGuardV1.__new__(
        MechanicalProposalAdmissibilityGuardV1
    )
    g.rest=np.asarray([
        [0.,0.,0.],
        [1.,0.,0.],
        [0.,1.,0.],
        [1.,1.,0.],
    ],dtype=np.float64)
    g.weights=np.ones((4,1),dtype=np.float64)
    g.vertex_index={str(i):i for i in range(4)}
    g.policy=_policy()
    g.max_edge_ratio=2.0
    g.tolerance=1e-9
    g.posed=np.stack((g.rest.copy(),np.asarray(posed,dtype=np.float64)),axis=0)
    g.probe_ids=("REST","STRESS")
    g._signature_cache={}
    return g


def test_local_mechanical_guard_accepts_equivalent_safe_retriangulation():
    g=_guard([
        [0.,0.,0.],
        [1.,0.,0.],
        [0.,1.,0.],
        [1.,1.,0.],
    ])
    assert g({
        "old_faces":(("0","1","2"),("1","3","2")),
        "new_faces":(("0","1","3"),("0","3","2")),
    }) is True


def test_local_mechanical_guard_rejects_topology_only_deformation_regression():
    # Same rest vertices; only the diagonal changes. Under the frozen stress
    # pose the old local triangulation has ~1.05 normalized severity while the
    # alternate diagonal has ~5.26. Static topology is not mechanically neutral.
    g=_guard([
        [0.44950443,0.63593941,0.0],
        [1.46187726,1.19970229,0.0],
        [0.64843565,1.77010398,0.0],
        [0.67608164,1.06391427,0.0],
    ])
    old=g.signature((("0","1","2"),("1","3","2")))
    new=g.signature((("0","1","3"),("0","3","2")))
    assert new.maximum_severity > old.maximum_severity * 4.0
    assert g({
        "old_faces":(("0","1","2"),("1","3","2")),
        "new_faces":(("0","1","3"),("0","3","2")),
    }) is False


def test_local_mechanical_guard_evaluates_temporary_split_with_exact_lbs():
    # One rigidly transformed joint: the temporary split vertex must be skinned
    # from its midpoint rest position, not guessed from static geometry alone.
    posed=np.asarray([
        [0.2,0.1,0.0],
        [1.2,0.1,0.0],
        [0.2,1.1,0.0],
        [1.2,1.1,0.0],
    ],dtype=np.float64)
    g=_guard(posed)
    I=np.eye(4,dtype=np.float64)
    T=np.eye(4,dtype=np.float64)
    T[:3,3]=np.asarray([0.2,0.1,0.0])
    g.probe_matrices=np.stack((I[None],T[None]),axis=0)
    proposal={
        "operator":"split",
        "old_faces":(("0","3","1"),("3","0","2")),
        "new_faces":(
            ("0","m","1"),("m","3","1"),
            ("3","m","2"),("m","0","2"),
        ),
        "temporary_split_vertex":{
            "id":"m","edge":("0","3"),"fraction":0.5,
        },
    }
    new=g.signature_with_temporary_split(
        proposal["new_faces"],proposal["temporary_split_vertex"]
    )
    assert new.unsafe_face_count==0
    assert g(proposal) is True
