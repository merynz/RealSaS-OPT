import pytest
from compiler.realsas_compiler_core.continuity_socket_bridge_v3 import (
    SocketBridgePolicyV3, qualify_socket_bridge, build_socket_bridge_set,
    assert_socket_bridge_measurement,
)
from compiler.realsas_compiler_core.types import QualificationError


def bridge(**kw):
    base=dict(view_index=0, component_id="BOOK_FOREGROUND", canonical_parent_joint_id="J:right",
              pivot_rest_xy=(10.0,10.0), nearest_foreground_alpha_rest_xy=(12.0,10.0),
              rest_bridge_pixel_count=3, rest_bridge_background_pixel_count=0,
              rest_body_socket_covered=True, foreground_alpha_authority_sha256="a"*64,
              source_component_lineage_hash="b"*64)
    base.update(kw); return qualify_socket_bridge(**base)


def test_bridge_qualifies_without_owner_boundary_semantics():
    b=bridge(); assert b.metadata["owner_raster_adjacency_used_as_mechanical_seam"] is False


def test_open_rest_bridge_fails_closed():
    with pytest.raises(QualificationError, match="REST_NOT_CLOSED"):
        bridge(rest_bridge_background_pixel_count=1)


def test_missing_body_socket_fails_closed():
    with pytest.raises(QualificationError, match="BODY_SOCKET_NOT_COVERED"):
        bridge(rest_body_socket_covered=False)


def test_carrier_quad_cannot_be_occupancy():
    with pytest.raises(QualificationError, match="CARRIER_QUAD_OCCUPANCY_FORBIDDEN"):
        bridge(foreground_carrier_mesh_used_as_occupancy=True)


def test_dynamic_measurement_pass_and_fail():
    p=SocketBridgePolicyV3(); s=build_socket_bridge_set(view_index=0,bridges=(bridge(),),policy=p)
    assert assert_socket_bridge_measurement(bridge_set=s,newly_exposed_pixel_count=1,evaluated_bridge_pixel_count=100,policy=p)==0.01
    with pytest.raises(QualificationError, match="DYNAMIC_BACKGROUND_EXPOSURE"):
        assert_socket_bridge_measurement(bridge_set=s,newly_exposed_pixel_count=3,evaluated_bridge_pixel_count=100,policy=p)


def test_policy_forbids_threshold_relaxation():
    with pytest.raises(QualificationError, match="THRESHOLD_RELAXATION_FORBIDDEN"):
        SocketBridgePolicyV3(max_new_background_fraction=0.03).validate()
