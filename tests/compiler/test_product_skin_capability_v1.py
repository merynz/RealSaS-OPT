from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from compiler.realsas_compiler_services.platform_worker import skin_executor_v1 as skin_executor

MIRA_ARCH = "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5"
SURFACE_HASH = "1" * 64


def test_product_skin_executor_keeps_mira_proposal_and_compiler_authority(monkeypatch):
    mesh = SimpleNamespace(
        source_evidence_binding_hash=SURFACE_HASH,
        mesh_lineage_hash="2" * 64,
        carrier_evidence_hash="3" * 64,
        carrier_topology_hash="4" * 64,
        carrier_geometry_hash="5" * 64,
        partition_binding_hash="6" * 64,
    )
    rig = SimpleNamespace(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        feature_evidence_binding_hash=SURFACE_HASH,
        rig_lineage_hash="7" * 64,
        structure_qualification_hash="8" * 64,
        joints=(object(), object()),
    )
    surface = SimpleNamespace(geometry_lineage_hash=SURFACE_HASH)
    carrier = SimpleNamespace(
        ordered_vertex_ids=("M:0", "M:1"),
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        topology_hash=mesh.carrier_topology_hash,
        geometry_hash=mesh.carrier_geometry_hash,
    )
    skeleton = SimpleNamespace(
        joints=(SimpleNamespace(canonical_joint_id="J:0"), SimpleNamespace(canonical_joint_id="J:1")),
        skeleton_lineage_hash=rig.structure_qualification_hash,
    )
    proposal = SimpleNamespace(
        to_dict=lambda: {
            "schema_version": "RealSaS.CarrierSkinProposalIR.v1",
            "metadata": {"output_domain": "MECHANICAL_CARRIER_M"},
        }
    )
    qualified = SimpleNamespace(
        qualification_report={
            "status": "PASS_DIRECT_CARRIER_SKIN",
            "simplex_max_abs_residual_before": 0.0,
            "max_row_correction_l1": 0.0,
            "total_correction_l1": 0.0,
        },
        to_dict=lambda: {
            "schema_version": "RealSaS.QualifiedCarrierSkinIR.v1",
            "qualification_report": {"status": "PASS_DIRECT_CARRIER_SKIN"},
        },
    )
    skin = SimpleNamespace(
        skin_lineage_hash="9" * 64,
        rows=(object(), object()),
        to_dict=lambda: {
            "schema_version": "RealSaS.CanonicalSkinIR.v1",
            "skin_lineage_hash": "9" * 64,
            "metadata": {"authority": "COMPILER_CANONICAL_SKIN"},
        },
    )

    import compiler.realsas_compiler_core.artifact_codec_v2 as codec
    import compiler.realsas_compiler_core.canonical_mesh_ir_v1 as mesh_ir
    import compiler.realsas_compiler_core.canonical_rig_ir_v1 as rig_ir
    import compiler.realsas_compiler_core.canonical_skin_ir_v1 as skin_ir
    import compiler.realsas_compiler_core.carrier_skin_v1 as carrier_skin
    import models.mira.carrier_native_v55 as carrier_native
    import models.mira.inference_v55 as inference
    import models.mira.runtime_checkpoint_v1 as runtime_checkpoint

    monkeypatch.setattr(codec, "rigging_surface_from_dict", lambda _payload: surface)
    monkeypatch.setattr(mesh_ir, "canonical_mesh_from_dict_v1", lambda _payload: mesh)
    monkeypatch.setattr(rig_ir, "canonical_rig_from_dict_v1", lambda _payload: rig)
    monkeypatch.setattr(
        runtime_checkpoint,
        "load_mira_runtime_model_v1",
        lambda *_args, **_kwargs: (object(), object(), {"schema": "RealSaS.MIRA.RuntimeCheckpoint.v1"}),
    )
    monkeypatch.setattr(
        inference,
        "infer_dense_wm_v55",
        lambda **_kwargs: (
            [[0.75, 0.25], [0.1, 0.9]],
            carrier,
            skeleton,
            {"joint_ids": ("J:0", "J:1"), "mesh_vertex_count": 2, "gsa_surface_count": 5},
        ),
    )
    seen = {}

    def build_proposal(**kwargs):
        seen["proposal_weights"] = kwargs["weights"]
        seen["proposal_carrier"] = kwargs["carrier"]
        seen["proposal_skeleton"] = kwargs["skeleton"]
        seen["model_provenance"] = kwargs["model_provenance"]
        return proposal

    def qualify(actual_carrier, actual_skeleton, actual_proposal, **policy):
        seen["qualify_carrier"] = actual_carrier
        seen["qualify_skeleton"] = actual_skeleton
        seen["qualify_proposal"] = actual_proposal
        seen["policy"] = dict(policy)
        return qualified

    def mint(actual_mesh, actual_rig, *, qualified_skin, proposal_evidence_refs):
        seen["mint_mesh"] = actual_mesh
        seen["mint_rig"] = actual_rig
        seen["mint_qualified"] = qualified_skin
        seen["mint_refs"] = tuple(proposal_evidence_refs)
        return skin

    monkeypatch.setattr(carrier_native, "proposal_from_dense_wm_v55", build_proposal)
    monkeypatch.setattr(carrier_skin, "qualify_carrier_skin_v1", qualify)
    monkeypatch.setattr(skin_ir, "mint_canonical_skin_v1", mint)
    monkeypatch.setattr(skin_executor, "_device_from_environment", lambda: "cpu")
    monkeypatch.setattr(
        skin_executor,
        "materialize_model_releases",
        lambda **_kwargs: [
            {
                "family": "MIRA",
                "architecture_id": MIRA_ARCH,
                "model_release_id": "mira-release-1",
                "checkpoint": {
                    "artifact_type": "RealSaS.MIRARuntimeCheckpoint",
                    "schema_version": "v1",
                    "path": "/immutable/mira-runtime.pt",
                    "content_sha256": "a" * 64,
                },
            }
        ],
    )

    objects = {
        "mesh": json.dumps({"schema": "ignored"}).encode(),
        "rig": json.dumps({"schema": "ignored"}).encode(),
        "surface": json.dumps({"schema": "ignored"}).encode(),
        "support": json.dumps({"schema_version": "RealSaS.TESSAMaterialSupportFieldIR.v1"}).encode(),
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
        "capability_id": "product.skin",
        "kind": "MODULE",
        "execution_id": "exec-skin-1",
        "capability_metadata": {
            "executor_contract": "RealSaS.ProductSkinInference.v1",
            "decode_chunk": 512,
            "carrier_skin_qualification": {
                "max_simplex_repair_l1": 1e-5,
                "max_total_correction_l1": 1e-4,
                "negative_tolerance": 1e-8,
                "max_influences": 41,
            },
        },
        "input_artifacts": [
            {"artifact_type": "RealSaS.CanonicalMeshIR", "schema_version": "v1", "storage_key": "mesh"},
            {"artifact_type": "RealSaS.CanonicalRigIR", "schema_version": "v1", "storage_key": "rig"},
            {"artifact_type": "RealSaS.RiggingSurfaceIR", "schema_version": "v1", "storage_key": "surface"},
            {
                "artifact_type": "RealSaS.TESSAMaterialSupportFieldIR",
                "schema_version": "v1",
                "storage_key": "support",
                "content_sha256": "b" * 64,
            },
        ],
        "model_releases": [{"family": "MIRA"}],
    }
    result = skin_executor.execute_skin_capability_v1(
        request,
        read_object=read_object,
        put_bytes=put_bytes,
    )

    assert result["status"] == "PASS"
    assert [row["role"] for row in result["outputs"]] == [
        "canonical_skin",
        "skin_influence_proposal",
        "carrier_skin_qualification",
    ]
    assert result["outputs"][0]["authority_class"] == "COMPILER_CANONICAL_SKIN"
    assert result["outputs"][1]["authority_class"] == "LEARNED_SKIN_INFLUENCE_PROPOSAL_ONLY"
    assert seen["proposal_carrier"] is carrier
    assert seen["proposal_skeleton"] is skeleton
    assert seen["qualify_proposal"] is proposal
    assert seen["mint_mesh"] is mesh
    assert seen["mint_rig"] is rig
    assert seen["mint_qualified"] is qualified
    assert seen["mint_refs"] == ("a" * 64, "b" * 64)
    assert seen["policy"] == {
        "max_simplex_repair_l1": 1e-5,
        "max_total_correction_l1": 1e-4,
        "negative_tolerance": 1e-8,
        "max_influences": 41,
    }
    diagnostics = result["diagnostics"]
    assert diagnostics["fit_performed"] is False
    assert diagnostics["teacher_authority_used"] is False
    assert diagnostics["source_skin_runtime_authority"] is False
    assert diagnostics["gsa_skin_field_authority_minted"] is False
    assert diagnostics["semantic_skin_transfer_performed"] is False
    assert diagnostics["feature_evidence_is_canonical_parent"] is False
    assert diagnostics["material_support_is_canonical_parent"] is False
    assert diagnostics["canonical_parents"] == ["CANONICAL_MESH", "CANONICAL_RIG"]
    assert diagnostics["product_revision_pass_claimed"] is False
