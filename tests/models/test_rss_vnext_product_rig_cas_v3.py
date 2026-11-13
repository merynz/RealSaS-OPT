from __future__ import annotations

"""RSS.vNext Rig CAS boundary, without a synthetic checkpoint PASS."""

from dataclasses import asdict
from hashlib import sha256
import json

import pytest

from compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1 import (
    RSSvNextComputedCandidateFaceV1, RSSvNextComputedCandidateSurfaceV1,
    rss_vnext_computed_surface_from_dict_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.platform_worker.rss_vnext_rig_executor_v3 import (
    EXECUTOR_CONTRACT, execute_rss_vnext_rig_capability_v3,
    _cas, _seal,
)


def _request():
    return {
        "kind": "MODULE", "capability_id": "product.rig",
        "capability_metadata": {
            "executor_contract": EXECUTOR_CONTRACT, "required_models": [],
        },
        "input_artifacts": [], "model_releases": [],
    }


def test_rig_worker_requires_canonical_static_parent_and_never_runs_another_model():
    req = _request()
    with pytest.raises(RuntimeError, match="PARENT_CARDINALITY:RealSaS.RSSvNextTwoViewSource"):
        execute_rss_vnext_rig_capability_v3(
            req, read_object=lambda _: b"", put_bytes=lambda _: {},
        )
    req["model_releases"] = [{"family": "AXIS"}]
    with pytest.raises(RuntimeError, match="MODEL_OR_EXECUTOR_CONTRACT_DRIFT"):
        execute_rss_vnext_rig_capability_v3(
            req, read_object=lambda _: b"", put_bytes=lambda _: {},
        )
    req["model_releases"] = []
    req["capability_metadata"]["required_models"] = [
        {"family": "MIRA"},
    ]
    with pytest.raises(RuntimeError, match="MODEL_OR_EXECUTOR_CONTRACT_DRIFT"):
        execute_rss_vnext_rig_capability_v3(
            req, read_object=lambda _: b"", put_bytes=lambda _: {},
        )


def test_rig_cas_reopens_exact_source_bytes_and_rejects_one_byte_swap():
    raw = b'{"exact":"source"}'
    digest = sha256(raw).hexdigest()
    ref = {"content_sha256":digest, "size_bytes":len(raw),
           "storage_key":"cas/"+digest}
    assert _cas(ref,lambda _:raw) == raw
    with pytest.raises(RuntimeError,match="CAS_BYTES_OR_HASH_DRIFT"):
        _cas(ref,lambda _:raw+b"!")
    with pytest.raises(RuntimeError,match="CAS_RESOURCE_LIMIT"):
        _cas({**ref,"size_bytes":300*1024*1024},lambda _:raw)
    def put(data):
        return {"content_sha256":sha256(data).hexdigest(),
                "size_bytes":len(data),
                "storage_key":"cas/"+sha256(data).hexdigest()}
    output = _seal({"schema":"test"},role="receipt",kind="RealSaS.Test",
                   version="v1",put_bytes=put)
    assert output["role"] == "receipt"
    assert output["authority_class"] == "COMPILER_CANONICAL_RIG_NOT_PRODUCT_REVISION"


def test_computed_face_json_cas_replay_preserves_nested_neural_channels():
    face = RSSvNextComputedCandidateFaceV1(
        candidate_face_index=0,
        candidate_vertex_indices=(0,1,2),
        candidate_component_index=0,
        area=0.125,
        unit_normal_xyz=(0.,0.,1.),
        region_probabilities=(0.7,0.3),
        presentation_role_probabilities=(0.2,0.4,0.4),
        graphics_logits_by_corner=((1.,2.),(3.,4.),(5.,6.)),
        affordance_logits_by_corner=((0.,1.),(2.,3.),(4.,5.)),
        original_rgba_by_corner=(
            ((1.,0.,0.,1.),(0.,1.,0.,1.)),
            ((1.,0.,0.,1.),(0.,1.,0.,1.)),
            ((1.,0.,0.,1.),(0.,1.,0.,1.)),
        ),
        original_projection_valid_by_corner=((True,False),(True,True),(False,True)),
    )
    surface = RSSvNextComputedCandidateSurfaceV1(
        source_inference_evidence_hash="a"*64,
        neural_scene_carrier_sha256="b"*64,
        unified_model_config_sha256="c"*64,
        candidate_mesh_sha256="d"*64,
        topology_token_sequence_sha256="e"*64,
        state="UNQUALIFIED_COMPUTED_SURFACE_CANDIDATE",reason="",
        faces=(face,),
    )
    actual = rss_vnext_computed_surface_from_dict_v1(json.loads(json.dumps(asdict(surface))))
    assert actual == surface
    assert actual.faces[0].original_rgba_by_corner[0][1] == (0.,1.,0.,1.)
    corrupted = json.loads(json.dumps(asdict(surface)))
    corrupted["faces"][0]["original_projection_valid_by_corner"] = None
    with pytest.raises(QualificationError,match="FACE_NESTING_INVALID"):
        rss_vnext_computed_surface_from_dict_v1(corrupted)
    del corrupted["faces"][0]["candidate_component_index"]
    with pytest.raises(QualificationError,match="FACE_FIELDS_DRIFT"):
        rss_vnext_computed_surface_from_dict_v1(corrupted)
