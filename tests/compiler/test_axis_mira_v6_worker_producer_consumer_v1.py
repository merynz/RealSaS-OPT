from __future__ import annotations

"""Typed CAS boundary and ModelRelease guards for candidate V2 producers.

This suite does not train models or certify generated Rig/Skin quality.
The positive assertions are real canonical data decoding and correct routing,
not a fabricated successful model inference/ProductRevision.
"""

import asyncio
from dataclasses import replace
import hashlib
import json

import pytest

from compiler.realsas_compiler_core.skin_code_ir_v1 import (
    JointSkinCodeRowIR, SkinCodeProposalIR, seal_skin_code_proposal_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.platform_worker import (
    rig_executor_v2, skin_executor_v2,
)
from tests.compiler.test_mechanics_cas_worker_main_v1 import _fixture


def _cas_objects():
    _, _, _, _, mesh, rig, skin = _fixture()
    code = seal_skin_code_proposal_v1(SkinCodeProposalIR(
        rows=tuple(JointSkinCodeRowIR(j.source_proposal_joint_id, (2, 3)) for j in rig.joints),
        rig_program_proposal_hash=rig.proposal_evidence_hash,
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        carrier_topology_hash=mesh.carrier_topology_hash,
        carrier_geometry_hash=mesh.carrier_geometry_hash,
        codec_architecture_id=skin_executor_v2.MIRA_V6_ARCHITECTURE_ID,
        codec_checkpoint_hash="d" * 64,
        codebook_size=27,
        codes_per_joint=2,
        model_provenance="SYNTHETIC_TEST_ONLY",
    ))
    surface = {
        "schema_version": "RealSaS.RiggingSurfaceIR.v1",
        "surface_nodes": [],
        "local_relations": [],
        "geometry_lineage_hash": mesh.source_evidence_binding_hash,
    }
    objects = {
        "mesh": mesh.to_dict(),
        "rig": rig.to_dict(),
        "code": code.to_dict(),
        "surface": surface,
    }
    blobs = {key: (json.dumps(value, sort_keys=True) + "\n").encode() for key, value in objects.items()}
    refs = {
        "mesh": {"artifact_type": "RealSaS.CanonicalMeshIR", "schema_version": "v1", "storage_key": "mesh"},
        "rig": {"artifact_type": "RealSaS.CanonicalRigIR", "schema_version": "v2", "storage_key": "rig"},
        "code": {"artifact_type": "RealSaS.SkinCodeProposalIR", "schema_version": "v1", "storage_key": "code"},
        "surface": {"artifact_type": "RealSaS.RiggingSurfaceIR", "schema_version": "v1", "storage_key": "surface"},
    }
    published = []

    def read_object(ref):
        return blobs[ref["storage_key"]]

    def put_bytes(data):
        published.append(data)
        return {"storage_key": "cas/example", "size_bytes": len(data), "content_sha256": hashlib.sha256(data).hexdigest()}

    return objects, blobs, refs, read_object, put_bytes, published


def _request(role):
    is_rig = role == "rig"
    return {
        "kind": "MODULE",
        "execution_id": "regression-one",
        "capability_id": "product." + role,
        "model_releases": [],
        "capability_metadata": (
            {
                "executor_contract": rig_executor_v2.RIG_EXECUTOR_CONTRACT_V2,
                "compiler_parent_reselection_allowed": False,
                "compiler_control_synthesis_allowed": False,
                "compiler_constraint_synthesis_allowed": False,
            } if is_rig else {
                "executor_contract": skin_executor_v2.SKIN_EXECUTOR_CONTRACT_V2,
                "compiler_code_search_allowed": False,
                "compiler_code_repair_allowed": False,
            }
        ),
    }


def _exec(role, request, read, put):
    fn = rig_executor_v2.execute_rig_capability_v2 if role == "rig" else skin_executor_v2.execute_skin_capability_v2
    return fn(request, read_object=read, put_bytes=put)


@pytest.mark.parametrize("role,keys", [("rig", ("mesh", "surface")), ("skin", ("mesh", "rig", "code"))])
def test_real_typed_cas_inputs_fail_closed_without_modelrelease(role, keys):
    objects, blobs, refs, read, put, outputs = _cas_objects()
    request = _request(role)
    request["input_artifacts"] = [refs[key] for key in keys]
    expected = "RIG_V2_MODEL_FAMILY_SET_DRIFT" if role == "rig" else "SKIN_V2_MODEL_FAMILY_SET_DRIFT"
    with pytest.raises(RuntimeError, match=expected):
        _exec(role, request, read, put)
    assert outputs == []


@pytest.mark.parametrize("role,keys", [("rig", ("mesh", "surface")), ("skin", ("mesh", "rig", "code"))])
def test_missing_or_relabelled_parent_is_not_silently_converted(role, keys):
    _, _, refs, read, put, outputs = _cas_objects()
    request = _request(role)
    request["input_artifacts"] = [refs[key] for key in keys]
    request["input_artifacts"][0] = dict(request["input_artifacts"][0], schema_version="v2")
    expected = "RIG_V2_INPUT_ARTIFACT_CARDINALITY" if role == "rig" else "SKIN_V2_INPUT_ARTIFACT_CARDINALITY"
    with pytest.raises(RuntimeError, match=expected):
        _exec(role, request, read, put)
    assert outputs == []


def test_rig_v2_refuses_surface_evidence_bound_to_another_mesh():
    obj, blobs, refs, read, put, outputs = _cas_objects()
    obj["surface"]["geometry_lineage_hash"] = "0" * 64
    blobs["surface"] = json.dumps(obj["surface"]).encode()
    req = _request("rig")
    req["input_artifacts"] = [refs["mesh"], refs["surface"]]
    with pytest.raises(RuntimeError, match="RIG_V2_FEATURE_EVIDENCE_MESH_BINDING_DRIFT"):
        _exec("rig", req, read, put)
    assert outputs == []


def test_skin_v2_refuses_rehashed_but_wrong_rig_program_binding():
    obj, blobs, refs, read, put, outputs = _cas_objects()
    # Use the real proposal serializer and reseal semantic hash:
    code = skin_executor_v2._skin_code_from_dict(obj["code"])
    code = seal_skin_code_proposal_v1(replace(
        code, rig_program_proposal_hash="e" * 64, proposal_lineage_hash=""
    ))
    blobs["code"] = json.dumps(code.to_dict()).encode()
    req = _request("skin")
    req["input_artifacts"] = [refs["mesh"], refs["rig"], refs["code"]]
    with pytest.raises(QualificationError, match="SKIN_CODE_RIG_PROGRAM_BINDING_DRIFT"):
        _exec("skin", req, read, put)
    assert outputs == []


def test_skin_v2_refuses_mira_checkpoint_drift_before_model_load(monkeypatch):
    _, _, refs, read, put, outputs = _cas_objects()
    req = _request("skin")
    req["input_artifacts"] = [refs["mesh"], refs["rig"], refs["code"]]
    monkeypatch.setattr(skin_executor_v2, "materialize_model_releases", lambda **kwargs: [{
        "family": "MIRA",
        "architecture_id": skin_executor_v2.MIRA_V6_ARCHITECTURE_ID,
        "checkpoint": {
            "artifact_type": "RealSaS.MIRARuntimeCheckpoint",
            "schema_version": "v2",
            "content_sha256": "a" * 64,
            "path": "/does/not/exist",
        },
    }])
    with pytest.raises(RuntimeError, match="SKIN_V2_CODE_CODEC_CHECKPOINT_DRIFT"):
        _exec("skin", req, read, put)
    assert outputs == []


@pytest.mark.parametrize("role", ("rig", "skin"))
def test_candidate_v2_worker_dispatch_is_explicit_and_keeps_v1(monkeypatch, role):
    pytest.importorskip("temporalio")
    from compiler.realsas_compiler_services.platform_worker import worker
    executor = rig_executor_v2 if role == "rig" else skin_executor_v2
    name = "execute_rig_capability_v2" if role == "rig" else "execute_skin_capability_v2"
    calls = []

    def probe(request, *, read_object, put_bytes):
        calls.append(request["capability_id"])
        return {"capability_id": request["capability_id"], "status": "FAIL",
                "outputs": [], "failure": {"code": "TEST_PROBE_ONLY"}}

    monkeypatch.setattr(executor, name, probe)
    monkeypatch.setenv("REALSAS_ENGINE_PROCESS_CHILD", "1")

    async def inline(fn, value):
        return fn(value)

    monkeypatch.setattr(worker, "_with_heartbeat", inline)
    req = _request(role)
    result = asyncio.run(worker.execute_capability(req))
    assert calls == ["product." + role]
    assert result["status"] == "FAIL"
    assert result["failure"]["code"] == "TEST_PROBE_ONLY"


def test_rig_v2_runtime_passes_exact_mechanics_capability_mask_to_axis(monkeypatch):
    """Execute the real worker through CAS parents up to the generated proposal.

    Only the model is a deliberately stopped stub: no model quality/product PASS
    is claimed, but the actual worker must provide the true Mechanics subset.
    """
    from types import SimpleNamespace
    import torch
    from compiler.realsas_compiler_core.rig_constraint_schema_v1 import RIG_CONSTRAINT_OPERATOR_IDS_V1
    from models.axis import runtime_checkpoint_v2, scene_conditioning_v6

    _, _, refs, read, put, emitted = _cas_objects()
    request = _request("rig")
    request["input_artifacts"] = [refs["mesh"], refs["surface"]]
    request["model_releases"] = [{"family": "AXIS"}]

    monkeypatch.setattr(rig_executor_v2, "materialize_model_releases", lambda **kwargs: [{
        "family": "AXIS",
        "architecture_id": rig_executor_v2.AXIS_V6_ARCHITECTURE_ID,
        "checkpoint": {
            "artifact_type": "RealSaS.AXISRuntimeCheckpoint",
            "schema_version": "v2",
            "content_sha256": "e" * 64,
            "path": "/never-read",
        },
        "model_release_id": "model-probe-only",
    }])
    calls = []

    class ProbeModel:
        config = SimpleNamespace(max_sequence_length=64)

        def generate_events(self, scene_memory, *, tokens_per_joint, max_steps,
                            allowed_constraint_operators=None):
            calls.append(tuple(allowed_constraint_operators or ()))
            assert max_steps == 64
            assert tokens_per_joint == 2
            raise RuntimeError("PROBE_STOP_BEFORE_NEURAL_OUTPUT")

    monkeypatch.setattr(
        runtime_checkpoint_v2, "load_axis_runtime_model_v2",
        lambda path, *, device: (
            ProbeModel(), {
                "skin_codec_binding": {
                    "architecture_id": "fixture-codec",
                    "checkpoint_sha256": "f" * 64,
                    "codebook_size": 16,
                    "tokens_per_joint": 2,
                },
            },
        ),
    )
    monkeypatch.setattr(
        scene_conditioning_v6, "condition_axis_v6_from_rigging_surface",
        lambda surface, *, device: SimpleNamespace(
            scene_memory=torch.zeros((1, 1, 4), device=device),
        ),
    )
    with pytest.raises(RuntimeError, match="PROBE_STOP_BEFORE_NEURAL_OUTPUT"):
        _exec("rig", request, read, put)
    assert calls == [RIG_CONSTRAINT_OPERATOR_IDS_V1]
    assert emitted == []


@pytest.mark.parametrize("corrupt", [3.75, True, -1, 27])
def test_skin_code_court_never_coerces_or_repairs_invalid_discrete_token(corrupt):
    _, _, _, _, mesh, rig, _ = _fixture()
    original = SkinCodeProposalIR(
        rows=tuple(JointSkinCodeRowIR(j.source_proposal_joint_id, (2, 3)) for j in rig.joints),
        rig_program_proposal_hash=rig.proposal_evidence_hash,
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        carrier_topology_hash=mesh.carrier_topology_hash,
        carrier_geometry_hash=mesh.carrier_geometry_hash,
        codec_architecture_id=skin_executor_v2.MIRA_V6_ARCHITECTURE_ID,
        codec_checkpoint_hash="d" * 64, codebook_size=27,
        codes_per_joint=2, model_provenance="SYNTHETIC_TEST_ONLY",
    )
    from compiler.realsas_compiler_core.skin_code_ir_v1 import validate_skin_code_proposal_v1
    forged = seal_skin_code_proposal_v1(replace(
        original,
        rows=(replace(original.rows[0], code_ids=(corrupt, 3)),) + original.rows[1:],
    ))
    with pytest.raises(QualificationError, match="SKIN_CODE_ID_OUT_OF_DOMAIN"):
        validate_skin_code_proposal_v1(
            forged,
            expected_proposal_joint_ids=tuple(j.source_proposal_joint_id for j in rig.joints),
            expected_rig_program_proposal_hash=rig.proposal_evidence_hash,
        )
