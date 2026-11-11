from __future__ import annotations

"""CAS-bound Mechanics rest qualification, explicitly not interactive proof."""

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    canonical_mechanics_from_dict_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_from_dict_v1,
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    _hash_control_program,
    _hash_deform_structure,
    canonical_rig_from_dict_v2,
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    canonical_skin_from_dict_v2,
    canonical_skin_lineage_hash_v2,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.platform_worker.mechanics_executor_v1 import (
    MECHANICS_EXECUTOR_CONTRACT_V1,
    execute_mechanics_capability_v1,
)
from tests.compiler.test_canonical_mechanics_ir_v1 import (
    _mesh as _seal_mesh,
    _rig as _seal_rig,
    _skin as _seal_skin,
)


def _fixture(*, wrong_skin_structure: bool = False):
    # Older direct-object Mechanics fixtures are not valid CAS readback fixtures:
    # the modern canonical decoders enforce extra metadata and true derived hashes.
    # Reconstruct a *synthetic test-only* typed triple without changing authority
    # validators or any source-truth/ModelRelease data.
    mesh = _seal_mesh()
    mesh = replace(
        mesh, mesh_lineage_hash="",
        metadata={
            **mesh.metadata,
            "learned_family_embedded_in_schema": False,
            "dynamic_qualification_claimed": False,
        },
    )
    mesh = replace(mesh, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(mesh))
    rig = _seal_rig(mesh)
    rig = replace(
        rig,
        deform_structure_hash="",
        control_program_hash="",
        rig_lineage_hash="",
    )
    rig = replace(rig, deform_structure_hash=_hash_deform_structure(rig))
    rig = replace(rig, control_program_hash=_hash_control_program(rig))
    rig = replace(rig, rig_lineage_hash=canonical_rig_lineage_hash_v2(rig))
    skin = _seal_skin(mesh, rig)
    # Require roundtrip equivalence *before* the worker is involved.
    assert canonical_mesh_from_dict_v1(mesh.to_dict()) == mesh
    assert canonical_rig_from_dict_v2(rig.to_dict()) == rig
    assert canonical_skin_from_dict_v2(skin.to_dict()) == skin
    if wrong_skin_structure:
        changed = replace(skin, deform_structure_hash="f" * 64)
        skin = replace(
            changed, skin_lineage_hash=canonical_skin_lineage_hash_v2(changed)
        )
    payloads = {
        "mesh": mesh.to_dict(),
        "rig": rig.to_dict(),
        "skin": skin.to_dict(),
    }
    objects = {
        key: (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
        for key, value in payloads.items()
    }
    request = {
        "capability_id": "product.mechanics",
        "kind": "MODULE",
        "capability_metadata": {
            "executor_contract": MECHANICS_EXECUTOR_CONTRACT_V1,
            "rest_qualification": {
                "max_position_drift": 1e-5,
                "max_constraint_residual": 1e-4,
            },
        },
        "model_releases": [],
        "input_artifacts": [
            {
                "artifact_type": "RealSaS.CanonicalMeshIR",
                "schema_version": "v1",
                "storage_key": "mesh",
            },
            {
                "artifact_type": "RealSaS.CanonicalRigIR",
                "schema_version": "v2",
                "storage_key": "rig",
            },
            {
                "artifact_type": "RealSaS.CanonicalSkinIR",
                "schema_version": "v2",
                "storage_key": "skin",
            },
        ],
    }
    outputs: list[bytes] = []

    def read_object(ref):
        return objects[ref["storage_key"]]

    def put_bytes(raw):
        outputs.append(raw)
        return {
            "storage_key": "cas/sha256/test",
            "content_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }

    return request, read_object, put_bytes, outputs, mesh, rig, skin


def test_cas_mechanics_produces_sealed_rest_only_artifact_not_a_product_revision():
    request, read_object, put_bytes, outputs, mesh, rig, skin = _fixture()
    result = execute_mechanics_capability_v1(
        request, read_object=read_object, put_bytes=put_bytes
    )
    assert result["status"] == "PASS"
    assert len(result["outputs"]) == len(outputs) == 1
    ref = result["outputs"][0]
    assert ref["artifact_type"] == "RealSaS.CanonicalMechanicsIR"
    assert ref["schema_version"] == "v1"
    assert ref["content_sha256"] == hashlib.sha256(outputs[0]).hexdigest()

    mechanics = canonical_mechanics_from_dict_v1(json.loads(outputs[0]))
    assert mechanics.mesh_lineage_hash == mesh.mesh_lineage_hash
    assert mechanics.rig_lineage_hash == rig.rig_lineage_hash
    assert mechanics.skin_lineage_hash == skin.skin_lineage_hash
    assert mechanics.metadata["canonical_constraint_solution_rest_pose"] is True
    assert result["diagnostics"]["product_revision_pass_claimed"] is False
    assert result["diagnostics"]["learned_inference_performed"] is False


def test_cas_mechanics_rejects_independently_resealed_wrong_deform_structure():
    request, read_object, put_bytes, outputs, *_ = _fixture(
        wrong_skin_structure=True
    )
    with pytest.raises(QualificationError, match="MECHANICS_DEFORM_STRUCTURE_DRIFT"):
        execute_mechanics_capability_v1(
            request, read_object=read_object, put_bytes=put_bytes
        )
    assert outputs == []


@pytest.mark.parametrize(
    "mutation, code",
    [
        ("contract", "MECHANICS_EXECUTOR_CONTRACT_DRIFT"),
        ("kind", "MECHANICS_CAPABILITY_KIND_DRIFT"),
        ("models", "MECHANICS_LEARNED_MODEL_RELEASE_FORBIDDEN"),
        ("duplicate_rig", "MECHANICS_INPUT_ARTIFACT_CARDINALITY"),
        ("wrong_skin_schema", "MECHANICS_INPUT_ARTIFACT_CARDINALITY"),
    ],
)
def test_cas_mechanics_fails_closed_before_minting(mutation, code):
    request, read_object, put_bytes, outputs, *_ = _fixture()
    if mutation == "contract":
        request["capability_metadata"]["executor_contract"] = "UNDECLARED"
    elif mutation == "kind":
        request["kind"] = "STAGE"
    elif mutation == "models":
        request["model_releases"] = [{"family": "UNAUTHORIZED"}]
    elif mutation == "duplicate_rig":
        request["input_artifacts"].append(dict(request["input_artifacts"][1]))
    elif mutation == "wrong_skin_schema":
        request["input_artifacts"][2]["schema_version"] = "v1"
    with pytest.raises(RuntimeError, match=code):
        execute_mechanics_capability_v1(
            request, read_object=read_object, put_bytes=put_bytes
        )
    assert outputs == []


def test_actual_worker_dispatch_routes_explicit_module_to_mechanics(monkeypatch):
    # The dedicated worker CI installs temporalio; the generic Compiler CI does
    # not. Skip only this adapter smoke outside its owning runtime environment.
    # All direct CAS/lineage/negative tests must still execute in both lanes.
    pytest.importorskip("temporalio")
    from compiler.realsas_compiler_services.platform_worker import worker

    request, read_object, put_bytes, outputs, *_ = _fixture()
    monkeypatch.setenv("REALSAS_ENGINE_PROCESS_CHILD", "1")
    monkeypatch.setattr(worker, "_read_cas_object", read_object)
    monkeypatch.setattr(worker, "_put_cas_bytes", put_bytes)

    async def run_inline(fn, value):
        return fn(value)

    monkeypatch.setattr(worker, "_with_heartbeat", run_inline)
    result = asyncio.run(worker.execute_capability(request))
    assert result["status"] == "PASS"
    assert len(outputs) == 1
    assert result["diagnostics"]["product_revision_pass_claimed"] is False


def test_mechanics_not_prematurely_registered_as_product_ready():
    # Rig/Skin V6 parent artifacts are now selected by the active product catalog.
    # Mechanics has its own remaining runtime and qualification gates; its
    # absence from the registered catalog must not be attributed to V1 parents.
    path = (
        Path(__file__).resolve().parents[2]
        / "platform/internal/capability/contracts/product_capabilities_v1.json"
    )
    semantics = json.loads(path.read_text(encoding="utf-8"))
    assert "product.mechanics" not in semantics["capabilities"]
    assert semantics["capabilities"]["product.rig"]["metadata"]["output_schema_version"] == "v2"
    assert semantics["capabilities"]["product.skin"]["metadata"]["output_schema_version"] == "v2"
