from __future__ import annotations

"""Executable downstream CAS courts, not merely source-file existence checks."""

from hashlib import sha256
import json
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import mint_canonical_mechanics_v1
from compiler.realsas_compiler_core.canonical_motion_ir_v2 import canonical_motion_from_dict_v2
from compiler.realsas_compiler_core.graphics_v1 import build_graphics_evidence_v1
from compiler.realsas_compiler_core.graphics_runtime_v1 import RuntimeGraphicsV1
from compiler.realsas_compiler_services.platform_worker.product_downstream_executor_v1 import (
    GRAPHICS_CONTRACT, MOTION_CONTRACT, PROOF_CONTRACT,
    execute_graphics_capability_v1, execute_motion_capability_v2,
    execute_proof_capability_v1,
)
from tests.compiler.test_mechanics_cas_worker_main_v1 import _fixture
from tests.compiler.test_control_motion_v2 import _proposal


class _Cas:
    def __init__(self):
        self.items: dict[str, bytes] = {}

    def put_bytes(self, raw):
        digest = sha256(raw).hexdigest()
        self.items[digest] = raw
        return {"storage_key":digest,"content_sha256":digest,"size_bytes":len(raw)}

    def put(self, obj, kind, version):
        data = obj.to_dict() if hasattr(obj,"to_dict") else obj
        ref = self.put_bytes((json.dumps(data,sort_keys=True)+"\n").encode())
        return {**ref,"artifact_type":kind,"schema_version":version}

    def read_object(self, ref):
        return self.items[ref["storage_key"]]


def _requests():
    _,_,_,_,mesh,rig,skin = _fixture()
    mechanics = mint_canonical_mechanics_v1(mesh,rig,skin)
    cas = _Cas()
    refs = [
        cas.put(mesh,"RealSaS.CanonicalMeshIR","v1"),
        cas.put(rig,"RealSaS.CanonicalRigIR","v2"),
        cas.put(skin,"RealSaS.CanonicalSkinIR","v2"),
        cas.put(mechanics,"RealSaS.CanonicalMechanicsIR","v1"),
    ]
    evidence = build_graphics_evidence_v1(
        mesh_binding_hash=mesh.mesh_lineage_hash,
        static_relations=(),view_relations=(),
    )
    refs.append(cas.put(evidence,"RealSaS.GraphicsEvidence","v1"))
    refs.append(cas.put(_proposal(rig,mechanics),"RealSaS.ControlMotionClipProposalIR","v1"))
    def request(role,contract):
        return {
            "capability_id":"product."+role,"kind":"MODULE",
            "capability_metadata":{"executor_contract":contract},
            "model_releases":[],"input_artifacts":list(refs),
        }
    return cas,mesh,rig,skin,mechanics,request


def test_graphics_and_motion_are_real_replayed_cas_consumers_and_proof_abstains():
    cas,mesh,rig,skin,mechanics,request = _requests()
    graphics = execute_graphics_capability_v1(
        request("graphics",GRAPHICS_CONTRACT),
        read_object=cas.read_object,put_bytes=cas.put_bytes,
    )
    assert graphics["status"] == "PASS"
    assert [x["artifact_type"] for x in graphics["outputs"]] == [
        "RealSaS.CompiledGraphics","RealSaS.RuntimeGraphics",
    ]
    assert graphics["diagnostics"]["source_appearance_qualified"] is False
    assert graphics["diagnostics"]["product_revision_minted"] is False

    motion = execute_motion_capability_v2(
        request("motion",MOTION_CONTRACT),
        read_object=cas.read_object,put_bytes=cas.put_bytes,
    )
    assert motion["status"] == "PASS"
    assert [x["artifact_type"] for x in motion["outputs"]] == [
        "RealSaS.QualifiedControlMotionIR","RealSaS.CanonicalMotionIR",
    ]
    m = canonical_motion_from_dict_v2(json.loads(
        cas.read_object(motion["outputs"][1])
    ))
    assert m.mechanics_lineage_hash == mechanics.mechanics_lineage_hash
    assert len(m.control_tracks) == 1

    proof_req = request("proof",PROOF_CONTRACT)
    proof_req["input_artifacts"] += [graphics["outputs"][0],motion["outputs"][1]]
    proof = execute_proof_capability_v1(
        proof_req,read_object=cas.read_object,put_bytes=cas.put_bytes,
    )
    assert proof["status"] == "FAIL"
    assert proof["failure"]["code"] == "PRODUCT_PROOF_INDEPENDENT_WITNESSES_MISSING"
    assert proof["outputs"] == []
    assert proof["diagnostics"]["product_pass_authority"] is False
    assert proof["diagnostics"]["product_revision_minted"] is False


def test_graphics_tamper_or_foreign_mesh_cannot_cross_downstream_cas():
    cas,mesh,rig,skin,mechanics,request = _requests()
    req = request("graphics",GRAPHICS_CONTRACT)
    evidence_ref = next(x for x in req["input_artifacts"] if x["artifact_type"]=="RealSaS.GraphicsEvidence")
    raw = json.loads(cas.read_object(evidence_ref))
    raw["mesh_binding_hash"] = "f"*64
    cas.items[evidence_ref["storage_key"]] = json.dumps(raw).encode()
    with pytest.raises(RuntimeError,match="INPUT_CAS_DRIFT"):
        execute_graphics_capability_v1(
            req,read_object=cas.read_object,put_bytes=cas.put_bytes,
        )


def test_motion_inherits_upstream_model_release_without_second_inference():
    # Platform passes the Attempt's model identities to every stage, even when
    # the downstream compiler does not need a second model.
    cas,mesh,rig,skin,mechanics,request = _requests()
    req = request("motion",MOTION_CONTRACT)
    req["model_releases"] = [{"family":"RSSVNEXT"}]
    result = execute_motion_capability_v2(
        req,read_object=cas.read_object,put_bytes=cas.put_bytes,
    )
    assert result["status"] == "PASS"
    assert result["diagnostics"]["product_revision_minted"] is False


def test_motion_cannot_request_its_own_model_family():
    cas,mesh,rig,skin,mechanics,request = _requests()
    req = request("motion",MOTION_CONTRACT)
    req["capability_metadata"]["required_models"] = [
        {"family":"MIRA","architecture_id":"unsupported-independent-worker"}
    ]
    with pytest.raises(RuntimeError,match="CONTRACT_OR_MODEL_DRIFT"):
        execute_motion_capability_v2(
            req,read_object=cas.read_object,put_bytes=cas.put_bytes,
        )
