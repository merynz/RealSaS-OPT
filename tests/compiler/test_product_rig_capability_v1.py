from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from compiler.realsas_compiler_core.types import SkeletonProposalIR
from compiler.realsas_compiler_services.platform_worker import rig_executor_v1 as rig_executor

AXIS_ARCH = "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1"
SURFACE_HASH = "1" * 64


def _proposal() -> SkeletonProposalIR:
    return SkeletonProposalIR(
        joints=(),
        edges=(),
        surface_binding_hash=SURFACE_HASH,
        model_provenance=AXIS_ARCH,
        metadata={
            "candidate_architecture": AXIS_ARCH,
            "hard_causal_parent_authority": True,
            "compiler_role": "VALIDATE_AND_MATERIALIZE",
            "compiler_parent_reselection_allowed": False,
            "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
        },
    )


def test_product_rig_executor_preserves_axis_proposal_and_compiler_authority(monkeypatch):
    mesh = SimpleNamespace(
        source_evidence_binding_hash=SURFACE_HASH,
        mesh_lineage_hash="2" * 64,
        carrier_evidence_hash="3" * 64,
        carrier_topology_hash="4" * 64,
        carrier_geometry_hash="5" * 64,
    )
    surface = SimpleNamespace(geometry_lineage_hash=SURFACE_HASH)
    tensor = SimpleNamespace(source_surface_hash=SURFACE_HASH, node_count=7)
    proposal = _proposal()
    fake_model = SimpleNamespace(
        propose=lambda *_args, **_kwargs: proposal,
    )
    qualified = SimpleNamespace(
        qualification_report={
            "status": "PASS_REQUIRED_TREE_VALIDATED",
            "selection_search_performed": False,
            "parent_reselection_performed": False,
        }
    )
    rig = SimpleNamespace(
        rig_lineage_hash="6" * 64,
        joints=(object(), object()),
        root_joint_ids=("J:root",),
        to_dict=lambda: {
            "schema_version": "RealSaS.CanonicalRigIR.v1",
            "rig_lineage_hash": "6" * 64,
            "metadata": {"authority": "COMPILER_CANONICAL_RIG"},
        },
    )

    import compiler.realsas_compiler_core.artifact_codec_v2 as codec
    import compiler.realsas_compiler_core.canonical_mesh_ir_v1 as mesh_ir
    import compiler.realsas_compiler_core.canonical_rig_ir_v1 as rig_ir
    import compiler.realsas_compiler_core.required_tree_v1 as required_tree
    import models.axis.runtime_checkpoint_v1 as runtime_checkpoint
    import models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 as tensorization

    monkeypatch.setattr(codec, "rigging_surface_from_dict", lambda _payload: surface)
    monkeypatch.setattr(mesh_ir, "canonical_mesh_from_dict_v1", lambda _payload: mesh)
    monkeypatch.setattr(runtime_checkpoint, "load_axis_runtime_model_v1", lambda *_args, **_kwargs: (fake_model, {"schema": "RealSaS.AXIS.RuntimeCheckpoint.v1"}))
    monkeypatch.setattr(tensorization, "tensorize_rigging_surface_v1", lambda *_args, **_kwargs: tensor)
    seen = {}

    def qualify(_surface, actual_proposal, *, authority_bindings):
        seen["qualified_proposal"] = actual_proposal
        seen["authority_bindings"] = dict(authority_bindings)
        return qualified

    def mint(actual_mesh, *, proposal, qualified_skeleton, feature_evidence_binding_hash):
        seen["mint_mesh"] = actual_mesh
        seen["mint_proposal"] = proposal
        seen["qualified_skeleton"] = qualified_skeleton
        seen["feature_evidence_binding_hash"] = feature_evidence_binding_hash
        return rig

    monkeypatch.setattr(required_tree, "qualify_required_skeleton_tree_v1", qualify)
    monkeypatch.setattr(rig_ir, "mint_canonical_rig_v1", mint)
    monkeypatch.setattr(rig_executor, "_device_from_environment", lambda: "cpu")
    monkeypatch.setattr(
        rig_executor,
        "materialize_model_releases",
        lambda **_kwargs: [
            {
                "family": "AXIS",
                "architecture_id": AXIS_ARCH,
                "model_release_id": "axis-release-1",
                "checkpoint": {
                    "artifact_type": "RealSaS.AXISRuntimeCheckpoint",
                    "schema_version": "v1",
                    "path": "/immutable/axis-runtime.pt",
                    "content_sha256": "7" * 64,
                },
            }
        ],
    )

    objects = {
        "mesh": json.dumps({"schema": "ignored"}).encode(),
        "surface": json.dumps({"schema": "ignored"}).encode(),
    }

    def read_object(ref):
        return objects[ref["storage_key"]]

    def put_bytes(data):
        digest = hashlib.sha256(data).hexdigest()
        return {
            "storage_key": "cas/" + digest,
            "content_sha256": digest,
            "size_bytes": len(data),
        }

    request = {
        "capability_id": "product.rig",
        "kind": "MODULE",
        "execution_id": "exec-rig-1",
        "capability_metadata": {
            "executor_contract": "RealSaS.ProductRigInference.v1",
            "resource_step_limit_max": 256,
        },
        "input_artifacts": [
            {
                "artifact_type": "RealSaS.CanonicalMeshIR",
                "schema_version": "v1",
                "storage_key": "mesh",
            },
            {
                "artifact_type": "RealSaS.RiggingSurfaceIR",
                "schema_version": "v1",
                "storage_key": "surface",
            },
        ],
        "model_releases": [{"family": "AXIS"}],
    }
    result = rig_executor.execute_rig_capability_v1(
        request,
        read_object=read_object,
        put_bytes=put_bytes,
    )

    assert result["status"] == "PASS"
    assert [row["role"] for row in result["outputs"]] == [
        "canonical_rig",
        "rig_structure_proposal",
    ]
    assert result["outputs"][0]["authority_class"] == "COMPILER_CANONICAL_RIG"
    assert result["outputs"][1]["authority_class"] == "LEARNED_RIG_STRUCTURE_PROPOSAL_ONLY"
    assert seen["qualified_proposal"] is proposal
    assert seen["mint_mesh"] is mesh
    assert seen["mint_proposal"] is proposal
    assert seen["qualified_skeleton"] is qualified
    assert seen["feature_evidence_binding_hash"] == SURFACE_HASH
    assert seen["authority_bindings"] == {
        "canonical_mesh": mesh.mesh_lineage_hash,
        "mechanical_carrier_evidence": mesh.carrier_evidence_hash,
        "mechanical_carrier_topology": mesh.carrier_topology_hash,
        "mechanical_carrier_geometry": mesh.carrier_geometry_hash,
    }
    diagnostics = result["diagnostics"]
    assert diagnostics["fit_performed"] is False
    assert diagnostics["frame_observability_claimed"] is False
    assert diagnostics["post_bind_frame_observability_required"] is True
    assert diagnostics["feature_evidence_is_canonical_parent"] is False
    assert diagnostics["canonical_parent"] == "CANONICAL_MESH"
    assert diagnostics["parent_reselection_performed"] is False
    assert diagnostics["product_revision_pass_claimed"] is False
