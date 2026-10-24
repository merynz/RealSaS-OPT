from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from compiler.realsas_compiler_core.product_surface_court_v1 import (
    PRODUCT_SURFACE_BUILDER_V1,
    PRODUCT_SURFACE_COMPLETED_V1,
    PRODUCT_SURFACE_OBSERVED_V1,
    PRODUCT_SURFACE_PROFILE_V1,
    PRODUCT_SURFACE_RELATION_V1,
    _nearest_iris_orientation_hints,
    _reprofile_surface,
)
from compiler.realsas_compiler_core.static_mesh_qualification_v2 import (
    build_source_fidelity_policy_binding_v1,
    source_fidelity_policy_binding_hash_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "canonical" / "PRODUCT_SHAPE_SURFACE_COURT_POLICY_V1_20261024.json"
SOURCE_POLICY = ROOT / "canonical" / "STAGE13_REST_REPROJECTION_POLICY_V2_20260919.json"
ADEQUACY_POLICY = ROOT / "canonical" / "STAGE14_SUBSTRATE_ADEQUACY_POLICY_V3_20260929.json"

PRODUCT_POLICY_SHA = "e182023bd59d079ecb747d0812af4e83de7415593c4bca54c753d0f7abfa6f51"
SOURCE_POLICY_SHA = "8edf9934f1d14f45dcb719fcf39691d69c5797f7410c958ae3005ef639a3c214"
ADEQUACY_POLICY_SHA = "bc98d3aa3cd419ca9ae43010e890fc554ab62e57268933afc630aae20c38139b"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _intermediate_surface() -> RiggingSurfaceIR:
    nodes = (
        SurfaceNode(
            surface_id="OLD:0",
            P=(0.0, 0.0, 1.0),
            support_views=(0,),
            provenance_refs=("legacy",),
            source_observation_ids=("obs0",),
            raster_bindings=((0, (10.0, 10.0)),),
            persistence_group_id="OLD:0",
            derived_normal=(0.0, 0.0, 1.0),
            validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": False},
        ),
        SurfaceNode(
            surface_id="OLD:1",
            P=(1.0, 0.0, 1.0),
            support_views=(0,),
            provenance_refs=("legacy",),
            source_observation_ids=("obs0",),
            raster_bindings=((0, (11.0, 10.0)),),
            persistence_group_id="OLD:1",
            derived_normal=(0.0, 0.0, 1.0),
            validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": False},
        ),
        SurfaceNode(
            surface_id="OLD:2",
            P=(0.0, 1.0, 1.0),
            support_views=(),
            provenance_refs=("legacy",),
            source_observation_ids=(),
            raster_bindings=(),
            persistence_group_id="OLD:2",
            derived_normal=(0.0, 0.0, 1.0),
            validity_flags=("MODEL_COMPLETED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": False},
        ),
    )
    relations = (
        SurfaceRelation("OLD:R01", "OLD:0", "OLD:1", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0, {"world_distance": 1.0}),
        SurfaceRelation("OLD:R02", "OLD:0", "OLD:2", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0, {"world_distance": 1.0}),
        SurfaceRelation("OLD:R12", "OLD:1", "OLD:2", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0, {"world_distance": 1.41421356237}),
    )
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=relations,
        geometry_lineage_hash="a" * 64,
        builder_id="legacy-intermediate",
        schema_version="RealSaS.RiggingSurfaceIR.v1",
        metadata={
            "scene_first_signed_geometry": True,
            "Nd_operator": "RealSaS.GSA.ZeroSurfaceTopologyAreaNormal.v2",
            "Nd_operator_sha256": "d" * 64,
            "raster_coordinate_system": "PIXEL_CENTER_XY",
            "resolution": 1024,
            "teacher_truth_used": False,
        },
    )


def test_product_surface_policy_binds_exact_frozen_bytes() -> None:
    assert _sha(POLICY) == PRODUCT_POLICY_SHA
    assert _sha(SOURCE_POLICY) == SOURCE_POLICY_SHA
    assert _sha(ADEQUACY_POLICY) == ADEQUACY_POLICY_SHA
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    assert policy["source_fidelity_policy"]["sha256"] == SOURCE_POLICY_SHA
    assert policy["surface_adequacy_policy"]["sha256"] == ADEQUACY_POLICY_SHA
    assert policy["normal_orientation"]["operator_id"] == "RealSaS.GSA.ZeroSurfaceTopologyAreaNormal.v2"
    assert policy["normal_orientation"]["operator_reuse_scope"] == "NUMERICAL_TOPOLOGY_AREA_NORMAL_ONLY__NO_SIGNED_ZERO_AUTHORITY"


def test_product_surface_reprofile_removes_signed_zero_semantics_and_tensorizes_for_axis() -> None:
    surface = _reprofile_surface(
        _intermediate_surface(),
        shape_content_sha256="1" * 64,
        decoded_topology_sha256="2" * 64,
        iris_checkpoint_sha256="3" * 64,
        tessa_checkpoint_sha256="4" * 64,
        normalization_hash="5" * 64,
        policy_sha256=PRODUCT_POLICY_SHA,
        hint_diagnostics={"hint_role": "ORIENTATION_SIGN_ONLY"},
        adequacy_summary={"status": "PASS"},
    )
    assert surface.builder_id == PRODUCT_SURFACE_BUILDER_V1
    assert surface.metadata["surface_evidence_profile"] == PRODUCT_SURFACE_PROFILE_V1
    assert surface.metadata["scene_first_signed_geometry"] is False
    assert surface.metadata["signed_zero_surface_claimed"] is False
    assert {relation.relation_kind for relation in surface.local_relations} == {PRODUCT_SURFACE_RELATION_V1}
    flags = {flag for node in surface.surface_nodes for flag in node.validity_flags}
    assert PRODUCT_SURFACE_OBSERVED_V1 in flags
    assert PRODUCT_SURFACE_COMPLETED_V1 in flags
    assert "OBSERVED_SIGNED_ZERO_SURFACE" not in flags
    assert "MODEL_COMPLETED_SIGNED_ZERO_SURFACE" not in flags
    serialized = json.dumps(surface.to_dict(), sort_keys=True)
    assert "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR" not in serialized

    tensor = tensorize_rigging_surface_v1(surface, require_scene_first=True)
    assert tensor.node_count == 3
    assert tensor.edge_count == 3
    assert int(tensor.observed.sum()) == 2
    assert int(tensor.completed.sum()) == 1
    assert tensor.relation_kind_vocab == (PRODUCT_SURFACE_RELATION_V1,)
    assert np.isfinite(tensor.positions_normalized).all()
    assert np.isfinite(tensor.normals).all()


def test_iris_orientation_hint_distance_gate_is_fail_closed() -> None:
    decoded = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
    witness = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    normals = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    with pytest.raises(QualificationError, match="PRODUCT_SURFACE_IRIS_HINT_DISTANCE_GATE_FAILED"):
        _nearest_iris_orientation_hints(
            decoded,
            witness,
            normals,
            p95_limit_norm=0.03,
            max_limit_norm=0.08,
        )


def test_direct_source_fidelity_binding_reuses_stage13_thresholds_without_geometry_substrate() -> None:
    document = json.loads(SOURCE_POLICY.read_text(encoding="utf-8"))
    cameras = SimpleNamespace(camera_set_hash="c" * 64, cameras=tuple(range(8)))
    observation = SimpleNamespace(observation_set_hash="o" * 64, views=tuple(range(8)))
    binding = build_source_fidelity_policy_binding_v1(
        cameras=cameras,
        observation=observation,
        policy_document=document,
        policy_document_sha256=SOURCE_POLICY_SHA,
    )
    assert binding.binding_hash == source_fidelity_policy_binding_hash_v1(binding)
    assert binding.policy == document["thresholds"]
    assert binding.metadata["direct_product_binding"] is True
    assert binding.metadata["legacy_geometry_substrate_fabricated"] is False
