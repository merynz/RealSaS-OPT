from types import SimpleNamespace

import pytest

from compiler.realsas_compiler_core.carrier_query_normal_evidence_v1 import (
    build_carrier_signed_query_normal_evidence_v1,
    carrier_signed_query_normal_evidence_from_dict,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    SurfaceNode,
    SurfaceSupportBinding,
)


def _surface(*, missing_b=False):
    return SimpleNamespace(
        geometry_lineage_hash="s"*64,
        surface_nodes=(
            SurfaceNode(
                "s0",(0.0,0.0,0.0),(0,),(),(),
                derived_normal=(0.0,0.0,1.0),
            ),
            SurfaceNode(
                "s1",(1.0,0.0,0.0),(0,),(),(),
                derived_normal=None if missing_b else (0.0,0.0,1.0),
            ),
        ),
    )


def _candidate(*, seam=False):
    if seam:
        b=SurfaceSupportBinding(
            "SEAM_GEOMETRY_INTERPOLATION",
            (("s0",0.25),("s1",0.75)),
            metadata={
                # Deliberately contradictory mechanical skin support. Query
                # normal evidence must ignore this field.
                "skin_support_coefficients":(("s0",1.0),),
            },
        )
    else:
        b=SurfaceSupportBinding(
            "LOCAL_CONVEX_INTERPOLATION",
            (("s0",0.25),("s1",0.75)),
        )
    return SimpleNamespace(
        candidate_lineage_hash="c"*64,
        surface_binding_hash="s"*64,
        vertices=(
            SimpleNamespace(
                candidate_vertex_id="v0",
                support_binding=SurfaceSupportBinding(
                    "IDENTITY_SURFACE_NODE",(("s0",1.0),)
                ),
                P=(0.0,0.0,0.0),
            ),
            SimpleNamespace(
                candidate_vertex_id="v1",
                support_binding=b,
                P=(0.75,0.0,0.0),
            ),
        ),
    )


def _carrier():
    return SimpleNamespace(
        candidate_mesh_binding_hash="c"*64,
        source_surface_binding_hash="s"*64,
        ordered_vertex_ids=("v0","v1"),
        carrier_evidence_hash="e"*64,
    )


def test_signed_query_normals_transport_geometry_support_and_roundtrip():
    value=build_carrier_signed_query_normal_evidence_v1(
        _candidate(),mechanical_carrier_evidence=_carrier(),surface=_surface()
    )
    assert value.ordered_vertex_ids==("v0","v1")
    assert value.signed_normals==((0.0,0.0,1.0),(0.0,0.0,1.0))
    assert value.normal_valid==(True,True)
    assert value.metadata["skin_support_consumed"] is False
    assert value.metadata["geometry_support_consumed"] is True
    assert carrier_signed_query_normal_evidence_from_dict(value.to_dict())==value


def test_seam_skin_support_is_not_query_normal_authority():
    value=build_carrier_signed_query_normal_evidence_v1(
        _candidate(seam=True),
        mechanical_carrier_evidence=_carrier(),
        surface=_surface(),
    )
    assert value.signed_normals[1]==(0.0,0.0,1.0)
    assert value.metadata["skin_support_consumed"] is False


def test_missing_source_normal_fails_closed_per_query_row():
    value=build_carrier_signed_query_normal_evidence_v1(
        _candidate(),
        mechanical_carrier_evidence=_carrier(),
        surface=_surface(missing_b=True),
    )
    assert value.normal_valid==(True,False)
    assert value.signed_normals[1]==(0.0,0.0,0.0)
    assert value.metadata["invalid_missing_source_normal_count"]==1


def test_exact_lineage_bindings_are_required():
    bad=SimpleNamespace(
        candidate_mesh_binding_hash="z"*64,
        source_surface_binding_hash="s"*64,
        ordered_vertex_ids=("v0","v1"),
        carrier_evidence_hash="e"*64,
    )
    with pytest.raises(
        QualificationError,match="CARRIER_QUERY_NORMAL_CANDIDATE_BINDING_DRIFT"
    ):
        build_carrier_signed_query_normal_evidence_v1(
            _candidate(),mechanical_carrier_evidence=bad,surface=_surface()
        )
