from __future__ import annotations

import json

import pytest
import torch

from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    canonical_skin_from_dict_v2,
)
from compiler.rss_vnext.workers.rss_vnext_skin_executor_v4 import (
    EXECUTOR_CONTRACT, execute_rss_vnext_skin_capability_v4,
    RIG_TYPE, SKIN_CODE_TYPE,
)
from tests.compiler.test_rss_vnext_raw_canonical_rig_v4 import (
    _positive_fixture, _run as run_rig,
)
from tests.compiler.test_rss_vnext_raw_product_input_shape_v4 import MemoryCAS


def _fixture(tmp_path):
    torch.manual_seed(233)
    torch.set_num_threads(1)
    cas, rig_req, mesh = _positive_fixture(
        tmp_path, with_discrete_skin=True,
    )
    rig_result = run_rig(cas, rig_req)
    assert rig_result["status"] == "PASS"
    meta = {
        "executor_contract": EXECUTOR_CONTRACT,
        "required_models": [],
        "compiler_code_search_allowed": False,
        "compiler_code_repair_allowed": False,
        "post_hoc_skin_transfer_allowed": False,
        "carrier_skin_qualification": {
            "max_simplex_repair_l1": 0.00001,
            "max_total_correction_l1": 0.0001,
            "negative_tolerance": 1e-8,
            "max_influences": 41,
        },
    }
    req = {
        "kind": "MODULE", "capability_id": "product.skin",
        "capability_metadata": meta, "model_releases": [],
        "input_artifacts": rig_req["input_artifacts"] + rig_result["outputs"],
    }
    return cas, req, mesh, rig_result


def _run(cas, req):
    return execute_rss_vnext_skin_capability_v4(
        req, read_object=cas.read_object, put_bytes=cas.put_bytes,
    )


def test_raw_two_images_mesh_rig_exact_fsq_replay_to_canonical_skin(tmp_path):
    cas, req, mesh, rig_result = _fixture(tmp_path)
    result = _run(cas, req)
    assert result["status"] == "PASS"
    assert len(result["outputs"]) == 4
    d = result["diagnostics"]
    assert d["mesh_lineage_hash"] == mesh.mesh_lineage_hash
    assert d["vertex_count"] == len(mesh.vertices) == 3
    assert d["joint_count"] == 1
    assert d["discrete_skin_max_abs_replay_error"] == pytest.approx(0., abs=5e-5)
    assert d["compiler_code_repair_performed"] is False
    assert d["separate_mira_model_release_loaded"] is False
    assert d["contact_and_pose_qualified"] is False
    assert d["product_revision_minted"] is False
    skin = canonical_skin_from_dict_v2(
        json.loads(cas.read_object(result["outputs"][0]))
    )
    assert skin.mesh_lineage_hash == mesh.mesh_lineage_hash
    assert skin.deform_structure_hash == d["deform_structure_hash"]
    assert skin.skin_lineage_hash == d["skin_lineage_hash"]
    assert result["outputs"][0]["artifact_type"] == "RealSaS.CanonicalSkinIR"
    witness = json.loads(cas.read_object(result["outputs"][3]))
    assert witness["embedded_decoder_content_sha256"]
    assert witness["discrete_skin_max_abs_replay_error"] <= 5e-5


def test_raw_skin_rejects_different_canonical_rig_lineage(tmp_path):
    cas, req, mesh, _ = _fixture(tmp_path)
    ref = next(r for r in req["input_artifacts"] if r["artifact_type"] == RIG_TYPE)
    rig = json.loads(cas.read_object(ref))
    rig["mesh_lineage_hash"] = "f" * 64
    forged = {
        **cas.put_bytes(json.dumps(rig).encode()),
        "artifact_type": RIG_TYPE, "schema_version": "v2",
    }
    req["input_artifacts"] = [
        forged if r is ref else r for r in req["input_artifacts"]
    ]
    with pytest.raises(Exception, match="RIG|MESH|LINEAGE|HASH"):
        _run(cas, req)


def test_raw_skin_rejects_forged_skin_code_parent(tmp_path):
    cas, req, _, _ = _fixture(tmp_path)
    ref = next(r for r in req["input_artifacts"] if r["artifact_type"] == SKIN_CODE_TYPE)
    code = json.loads(cas.read_object(ref))
    code["codec_checkpoint_hash"] = "e" * 64
    forged = {
        **cas.put_bytes(json.dumps(code).encode()),
        "artifact_type": SKIN_CODE_TYPE, "schema_version": "v1",
    }
    req["input_artifacts"] = [
        forged if r is ref else r for r in req["input_artifacts"]
    ]
    with pytest.raises(Exception, match="SKIN|CODE|RIG|PARENT|HASH"):
        _run(cas, req)


def test_calibrated_skin_still_delegates_to_existing_v3(monkeypatch):
    import compiler.rss_vnext.workers.rss_vnext_skin_executor_v4 as module
    calls = []
    monkeypatch.setattr(module, "execute_rss_vnext_skin_capability_v3",
        lambda req, **kwargs: (
            calls.append(req) or {"status": "DELEGATED"}
        ),
    )
    result = _run(MemoryCAS(), {
        "kind": "MODULE", "capability_id": "product.skin",
        "capability_metadata": {
            "executor_contract": EXECUTOR_CONTRACT,
            "compiler_code_search_allowed": False,
            "compiler_code_repair_allowed": False,
            "post_hoc_skin_transfer_allowed": False,
        },
        "model_releases": [],
        "input_artifacts": [{
            "artifact_type": "RealSaS.RSSvNextTwoViewSource",
            "schema_version": "v1",
        }],
    })
    assert result["status"] == "DELEGATED"
    assert calls[0]["capability_metadata"]["executor_contract"] == (
        "RealSaS.ProductSkinInference.v3"
    )
