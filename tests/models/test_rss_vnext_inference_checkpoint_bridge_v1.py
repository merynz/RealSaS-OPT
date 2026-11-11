from __future__ import annotations

from dataclasses import replace
from copy import deepcopy

import pytest
import torch
import numpy as np

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,
    seal_inference_evidence_v1,
    validate_inference_evidence_v1,
    candidate_face_region_readout_v1,
)
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextUnifiedCharacterModel, RSSvNextPatchEncoderForTests,
)
from models.rss_vnext.inference_checkpoint_v1 import (
    checkpoint_payload_v1, save_inference_checkpoint_v1, load_inference_checkpoint_v1,
)
from models.rss_vnext.inference_evidence_export_v1 import export_inference_evidence_v1


@pytest.fixture(autouse=True)
def cpu_deterministic():
    torch.manual_seed(17)
    torch.set_num_threads(1)


def _network():
    cfg = RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).eval()


def _images():
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[:, :, 3] = cam[:, :, 7] = cam[:, :, 11] = cam[:, :, 12] = 1.0
    return rgba, cam


def _mesh():
    n = 3
    return {
        "topology_tokens": [128, 129, 4],
        "rest_xyz": [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        "base_xyz": [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        "local_features": [[.1, .2, .3]] * n,
        "faces": [[0, 1, 2]],
        "vertex_components": [0, 0, 0],
        "face_components": [0],
        "normals_xyz": [[0., 0., 1.]] * n,
        "semantic_logits": [[0., 1.]] * n,
        "presentation_role_logits": [[.1, .2, .3, .4]] * n,
        "presentation_role_schema_id": "RSS.vNext.SemanticPresentationRoles.FourWayResearch.v1",
        "affordance_logits": [[.5, .5]] * n,
        "graphics_logits": [[.1, .2, .3, .4, .5]] * n,
        "static_quality_logits": [[0., 0., 0., 0.]] * n,
        "unsigned_distance": [.1] * n,
        "sign_logits": [.7] * n,
        "source_rays": {
            "rgba": [[[.1, .2, .3, .8], [.2, .3, .4, .4]]] * n,
            "valid": [[True, True]] * n,
            "alpha": [[.8, .4]] * n,
            "appearance_authority": "ORIGINAL_RGBA_CAMERA_RAY_NOT_POINT_TEXTURE",
            "depth_authority": "UNKNOWN_NO_SOURCE_DEPTH",
        },
        "source_authority": "LEARNED_PREDICTED_TOPOLOGY",
        "signed_solid_authority": "UNQUALIFIED",
        "canonical_mesh_authority": False,
    }


def _candidate(**override):
    payload = dict(
        checkpoint_sha256="a" * 64,
        observed_front_sha256="b" * 64,
        observed_back_sha256="c" * 64,
        camera_binding_sha256="d" * 64,
        mesh_status="UNQUALIFIED_MESH_CANDIDATE",
        mesh=_mesh(),
    )
    payload.update(override)
    return RSSvNextInferenceEvidenceIRV1(**payload)


def test_checkpoint_roundtrip_has_exact_state_and_no_product_grant(tmp_path):
    net = _network()
    path = tmp_path / "unified.pt"
    digest = save_inference_checkpoint_v1(path, net)
    assert len(digest) == 64
    assert checkpoint_payload_v1(net)["qualification_status"] == "INFERENCE_PACKAGING_ONLY_NOT_PRODUCT_RELEASE"
    loaded = load_inference_checkpoint_v1(
        path, expected_sha256=digest,
        vision_encoder=RSSvNextPatchEncoderForTests(net.cfg.iris.image_token_dim),
        product_mode=False,
    )
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())
    for name, state in net.state_dict().items():
        assert torch.equal(state, loaded.state_dict()[name]), name
    rgba, cam = _images()
    with torch.no_grad():
        a = net(rgba, cam).continuous_surface_proposal
        b = loaded(rgba, cam).continuous_surface_proposal
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_checkpoint_refuses_unpinned_wrong_or_product_test_encoder(tmp_path):
    net = _network()
    target = tmp_path / "test.pt"
    digest = save_inference_checkpoint_v1(target, net)
    with pytest.raises(RuntimeError, match="CHECKPOINT_PIN_REQUIRED"):
        load_inference_checkpoint_v1(target, expected_sha256="not-hash", vision_encoder=net.vision)
    with pytest.raises(RuntimeError, match="CHECKPOINT_BYTES_DRIFT"):
        load_inference_checkpoint_v1(target, expected_sha256="b" * 64, vision_encoder=net.vision)
    with pytest.raises(RuntimeError, match="TEST_VISION_FORBIDDEN_IN_PRODUCT"):
        load_inference_checkpoint_v1(target, expected_sha256=digest, vision_encoder=net.vision)
    with target.open("ab") as fp:
        fp.write(b"tampered")
    with pytest.raises(RuntimeError, match="CHECKPOINT_BYTES_DRIFT"):
        load_inference_checkpoint_v1(
            target, expected_sha256=digest, vision_encoder=net.vision, product_mode=False,
        )


def test_compiler_accepts_exact_mesh_addressed_evidence_without_minting_product():
    value = seal_inference_evidence_v1(_candidate())
    validate_inference_evidence_v1(value, expected_checkpoint_sha256="a" * 64)
    assert value.qualification_status == "UNQUALIFIED_NEURAL_PROPOSAL_ONLY"
    assert not value.mesh["canonical_mesh_authority"]
    with pytest.raises(QualificationError, match="CHECKPOINT_BINDING_DRIFT"):
        validate_inference_evidence_v1(value, expected_checkpoint_sha256="e" * 64)


@pytest.mark.parametrize("change,code", [
    ({"canonical_mesh_authority": True}, "PREMATURE_AUTHORITY"),
    ({"source_authority": "TEACHER_FORCED_TRAINING_ONLY"}, "PREMATURE_AUTHORITY"),
    ({"faces": [[0, 1, 3]]}, "FACE_INVALID"),
    ({"vertex_components": [0, 1, 0]}, "CROSS_COMPONENT_FACE"),
    ({"unsigned_distance": [float("nan"), .1, .1]}, "UDF"),
])
def test_compiler_rejects_invented_authority_and_bad_addresses(change, code):
    mesh = _mesh()
    mesh.update(change)
    with pytest.raises((QualificationError, ValueError), match=code):
        seal_inference_evidence_v1(_candidate(mesh=mesh))


def test_source_ray_ownership_not_surface_texture():
    mesh = _mesh()
    mesh["source_rays"]["depth_authority"] = "KNOWN"
    with pytest.raises(QualificationError, match="SOURCE_RAY_AUTHORITY"):
        seal_inference_evidence_v1(_candidate(mesh=mesh))


def test_hash_drift_and_abstain_are_fail_closed():
    value = seal_inference_evidence_v1(_candidate())
    changed = deepcopy(value.mesh)
    changed["rest_xyz"][0][0] += .2
    with pytest.raises(QualificationError, match="HASH_DRIFT"):
        validate_inference_evidence_v1(replace(value, mesh=changed))
    abstain = seal_inference_evidence_v1(_candidate(
        mesh=None, mesh_status="ABSTAIN", reason="TOPOLOGY_SEQUENCE_INCOMPLETE",
    ))
    assert abstain.mesh is None
    with pytest.raises(QualificationError, match="ABSTAIN_MUST_NOT_PROMOTE"):
        seal_inference_evidence_v1(_candidate(
            mesh=_mesh(), mesh_status="ABSTAIN", reason="TOPOLOGY_SEQUENCE_INCOMPLETE",
        ))


def test_model_source_state_export_and_source_change_firewall():
    net = _network()
    rgba, cam = _images()
    with torch.no_grad():
        out = net(rgba, cam)
    prediction = {
        "state": out.state,
        "mesh_status": "ABSTAIN",
        "mesh_candidate": None,
        "mesh_failure": "TOPOLOGY_SEQUENCE_INCOMPLETE",
        "observed_view_count": 2,
        "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
        "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
    }
    exported = export_inference_evidence_v1(
        prediction, original_rgba=rgba, original_camera_features=cam,
        checkpoint_sha256="a" * 64,
    )
    assert exported.mesh_status == "ABSTAIN"
    assert exported.observed_front_sha256 != exported.observed_back_sha256
    assert exported.mesh is None
    with pytest.raises(ValueError, match="SOURCE_STATE_MISMATCH"):
        export_inference_evidence_v1(
            prediction, original_rgba=rgba.clone() + 0.01,
            original_camera_features=cam, checkpoint_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="REQUIRES_TWO_RGBA"):
        export_inference_evidence_v1(
            prediction, original_rgba=rgba[:, :1],
            original_camera_features=cam, checkpoint_sha256="a" * 64,
        )


def test_compiler_receipt_preserves_addressed_semantic_regions_and_graphics_without_authority():
    net = _network()
    rgba, cam = _images()
    with torch.no_grad():
        out = net(rgba, cam)
    prediction = {
        "state": out.state, "mesh_status": "ABSTAIN",
        "mesh_candidate": None, "mesh_failure": "TOPOLOGY_SEQUENCE_INCOMPLETE",
        "observed_view_count": 2,
        "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
        "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
    }
    evidence = export_inference_evidence_v1(
        prediction, original_rgba=rgba, original_camera_features=cam,
        checkpoint_sha256="a" * 64,
    )
    scene = evidence.scene
    assert len(scene["region_query_xyz"]) == out.state.query_xyz.shape[1]
    assert len(scene["region_membership_logits"]) == len(scene["region_query_xyz"])
    assert len(scene["region_graphics_logits"]) == len(scene["region_query_xyz"])
    assert len(scene["region_graphics_logits"][0]) == 5
    assert len(scene["region_membership_logits"][0]) == net.cfg.region_slots + 1
    assert scene["authority"] == "PREDICTED_FROM_TWO_ORIGINAL_VIEWS"
    assert evidence.qualification_status == "UNQUALIFIED_NEURAL_PROPOSAL_ONLY"
    assert evidence.mesh_status == "ABSTAIN"
    validate_inference_evidence_v1(evidence)


def test_existing_v1_relation_only_receipt_remains_readable_but_not_product_evidence():
    net = _network()
    rgba, cam = _images()
    with torch.no_grad():
        output = net(rgba, cam)
    full = export_inference_evidence_v1(
        {
            "state": output.state, "mesh_status": "ABSTAIN",
            "mesh_candidate": None, "mesh_failure": "TOPOLOGY_SEQUENCE_INCOMPLETE",
            "observed_view_count": 2,
            "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
            "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
        },
        original_rgba=rgba, original_camera_features=cam,
        checkpoint_sha256="a" * 64,
    )
    legacy = deepcopy(full.scene)
    for field_name in (
        "region_query_xyz", "region_membership_logits",
        "region_graphics_logits",
    ):
        del legacy[field_name]
    sealed = seal_inference_evidence_v1(
        replace(full, scene=legacy, evidence_hash=""),
    )
    validate_inference_evidence_v1(sealed)
    assert sealed.qualification_status == "UNQUALIFIED_NEURAL_PROPOSAL_ONLY"
    assert "region_query_xyz" not in sealed.scene


@pytest.mark.parametrize("corruption,error", [
    ("unaddressed", "REGION_ADDRESS_DRIFT"),
    ("missing_unknown", "REGION_UNKNOWN_CHANNEL"),
    ("graphics_width", "GRAPHICS_FIELD_FIVE_CHANNELS"),
    ("self_edge", "RELATION_SELF_EDGE_MASK_INVALID"),
])
def test_semantic_compiler_receipt_rejects_rehashed_unaddressed_or_self_inconsistent_scene(corruption, error):
    net = _network()
    rgba, cam = _images()
    with torch.no_grad():
        output = net(rgba, cam)
    evidence = export_inference_evidence_v1(
        {
            "state": output.state, "mesh_status": "ABSTAIN",
            "mesh_candidate": None, "mesh_failure": "TOPOLOGY_SEQUENCE_INCOMPLETE",
            "observed_view_count": 2,
            "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
            "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
        },
        original_rgba=rgba, original_camera_features=cam,
        checkpoint_sha256="a" * 64,
    )
    scene = deepcopy(evidence.scene)
    if corruption == "unaddressed":
        scene["region_query_xyz"].pop()
    elif corruption == "missing_unknown":
        scene["region_membership_logits"][0].pop()
    elif corruption == "graphics_width":
        scene["region_graphics_logits"][0].pop()
    else:
        scene["region_relation_valid_mask"][0][0] = True
    with pytest.raises(QualificationError, match=error):
        seal_inference_evidence_v1(replace(evidence, scene=scene, evidence_hash=""))


def test_real_checkpoint_two_view_model_to_replayable_compiler_receipt(tmp_path):
    from models.rss_vnext.inference_pipeline_v1 import (
        run_two_view_to_compiler_proposal_v1,
        read_compiler_proposal_receipt_v1,
    )

    net = _network()
    checkpoint = tmp_path / "checkpoint.pt"
    digest = save_inference_checkpoint_v1(checkpoint, net)
    rgba, cam = _images()
    # DINOv2 has 14px patch requirements; a 28px research image is also
    # compatible with the 4px test encoder and with source archive policy.
    rgba = torch.rand(1, 2, 4, 28, 28)
    out_dir = tmp_path / "compile_001"
    receipt = run_two_view_to_compiler_proposal_v1(
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=digest,
        vision_encoder=RSSvNextPatchEncoderForTests(net.cfg.iris.image_token_dim),
        observed_rgba=rgba,
        camera_features=cam,
        output_directory=out_dir,
        max_topology_tokens=16,
        generate_controls=True,
        product_mode=False,
    )
    assert receipt["status"] in ("ABSTAIN", "UNQUALIFIED_MESH_CANDIDATE")
    assert receipt["product_revision_minted"] is False
    again, evidence = read_compiler_proposal_receipt_v1(
        out_dir, expected_checkpoint_sha256=digest,
    )
    assert again == receipt
    assert evidence.mesh_status == receipt["status"]
    with pytest.raises(RuntimeError, match="OUTPUT_REVISION_ALREADY_EXISTS"):
        run_two_view_to_compiler_proposal_v1(
            checkpoint_path=checkpoint, expected_checkpoint_sha256=digest,
            vision_encoder=RSSvNextPatchEncoderForTests(net.cfg.iris.image_token_dim),
            observed_rgba=rgba, camera_features=cam, output_directory=out_dir,
            max_topology_tokens=16, product_mode=False,
        )
    with pytest.raises(RuntimeError, match="RECEIPT_AUTHORITY_DRIFT"):
        read_compiler_proposal_receipt_v1(
            out_dir, expected_checkpoint_sha256="e" * 64,
        )
    with (out_dir / "observed_source_tensors.npz").open("ab") as fh:
        fh.write(b"attack")
    with pytest.raises(RuntimeError, match="ARTIFACT_BYTES_DRIFT"):
        read_compiler_proposal_receipt_v1(
            out_dir, expected_checkpoint_sha256=digest,
        )


def test_reuse_existing_axis_decoder_without_minting_canonical_rig():
    from dataclasses import asdict
    from models.axis.rig_program_model_v6 import RigProgramEventV6, RigTokenKindV6
    from models.rss_vnext.candidate_rig_court_v1 import qualify_candidate_rig_program_v1

    kinds = (
        RigTokenKindV6.BOS,
        RigTokenKindV6.JOINT,
        RigTokenKindV6.FRAMES_START,
        RigTokenKindV6.FRAME,
        RigTokenKindV6.CONTROLS_START,
        RigTokenKindV6.CHAINS_START,
        RigTokenKindV6.CONSTRAINTS_START,
        RigTokenKindV6.ANCHORS_START,
        RigTokenKindV6.SKIN_START,
        *([RigTokenKindV6.SKIN_CODE] * 4),
        RigTokenKindV6.EOS,
    )
    events = []
    for kind in kinds:
        kwargs = {}
        if kind is RigTokenKindV6.JOINT:
            kwargs = {"coord_indices": (128, 128, 128), "pointer_a": -1}
        if kind is RigTokenKindV6.FRAME:
            kwargs = {
                "pointer_a": 0,
                "frame_6d": (1., 0., 0., 0., 1., 0.),
            }
        if kind is RigTokenKindV6.SKIN_CODE:
            kwargs = {"skin_code_id": 7}
        row = asdict(RigProgramEventV6(kind, **kwargs))
        row["kind"] = int(kind)
        events.append(row)

    proposal = seal_inference_evidence_v1(_candidate(control_events=tuple(events)))
    court = qualify_candidate_rig_program_v1(
        proposal, model_config=RSSvNextConfig.tiny_for_test(),
        expected_checkpoint_sha256="a" * 64,
    )
    assert court.product_authority is False
    assert len(court.rig_program.proposal.joints) == 1
    assert len(court.skin_codes.rows) == 1
    assert court.skin_codes.rows[0].code_ids == (7, 7, 7, 7)
    assert court.rig_program.proposal.mesh_lineage_hash == court.candidate_mesh_hash
    with pytest.raises(QualificationError, match="RIG_REQUIRES_VALID_NEURAL_MESH"):
        qualify_candidate_rig_program_v1(
            seal_inference_evidence_v1(_candidate(
                mesh=None, mesh_status="ABSTAIN", reason="NO_VALID_TOPOLOGY",
            )),
            model_config=RSSvNextConfig.tiny_for_test(),
            expected_checkpoint_sha256="a" * 64,
        )


def _mesh_with_addressed_scene_for_region_court():
    net = _network()
    rgba, cam = _images()
    with torch.no_grad():
        out = net(rgba, cam)
    source = export_inference_evidence_v1(
        {
            "state": out.state, "mesh_status": "ABSTAIN",
            "mesh_candidate": None, "mesh_failure": "NO_VALID_MESH",
            "observed_view_count": 2,
            "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
            "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
        },
        original_rgba=rgba, original_camera_features=cam,
        checkpoint_sha256="a" * 64,
    )
    mesh = _mesh()
    slots = len(source.scene["region_relation_valid_mask"])
    mesh["semantic_logits"] = [
        [4., *([0.] * (slots - 1)), -3.],
        [0., *([3.] * (slots - 1)), 0.],
        [0., *([0.] * (slots - 1)), 5.],
    ]
    return mesh, source.scene


def test_mesh_face_region_readout_is_exactly_addressed_and_never_grants_drawable():
    mesh, scene = _mesh_with_addressed_scene_for_region_court()
    evidence = seal_inference_evidence_v1(_candidate(mesh=mesh, scene=scene))
    faces = candidate_face_region_readout_v1(evidence)
    assert len(faces) == 1
    row = faces[0]
    assert row["candidate_face_index"] == 0
    assert row["candidate_vertex_indices"] == (0, 1, 2)
    assert row["candidate_component_id"] == 0
    assert row["candidate_evidence_hash"] == evidence.evidence_hash
    assert row["source_surface_or_drawable_authority"] is False
    assert row["authority"] == "UNQUALIFIED_FACE_POOL_OF_LEARNED_VERTEX_MEMBERSHIP"
    assert len(row["candidate_region_probabilities"]) == len(scene["region_relation_valid_mask"])
    assert 0 < row["unknown_probability"] < 1
    assert sum(row["candidate_region_probabilities"]) + row["unknown_probability"] == pytest.approx(1.)

    # A vertex-order permutation of the same triangle does not change the
    # region vector. It does change the proposal hash, preserving provenance.
    rotated = deepcopy(mesh)
    rotated["faces"] = [[2, 0, 1]]
    other = seal_inference_evidence_v1(_candidate(mesh=rotated, scene=scene))
    after = candidate_face_region_readout_v1(other)[0]
    assert after["candidate_region_probabilities"] == pytest.approx(row["candidate_region_probabilities"])
    assert after["unknown_probability"] == pytest.approx(row["unknown_probability"])
    assert after["candidate_evidence_hash"] != row["candidate_evidence_hash"]


def test_mesh_region_vocab_cannot_drift_from_shared_scene_or_erase_unknown():
    mesh, scene = _mesh_with_addressed_scene_for_region_court()
    # Keep all vertex rows internally consistent, but drop the UNKNOWN
    # column from the entire mesh to exercise cross-head vocabulary binding.
    for row in mesh["semantic_logits"]:
        row.pop()
    with pytest.raises(QualificationError, match="MESH_REGION_VOCABULARY_DRIFT"):
        seal_inference_evidence_v1(_candidate(mesh=mesh, scene=scene))


def test_candidate_face_region_readout_abstains_without_exact_topology_or_addresses():
    mesh, scene = _mesh_with_addressed_scene_for_region_court()
    no_mesh = seal_inference_evidence_v1(_candidate(
        mesh=None, scene=scene, mesh_status="ABSTAIN",
        reason="NO_VALID_MESH",
    ))
    with pytest.raises(QualificationError, match="FACE_REGION_REQUIRES_MESH_PROPOSAL"):
        candidate_face_region_readout_v1(no_mesh)
    legacy = seal_inference_evidence_v1(_candidate(mesh=mesh))
    with pytest.raises(QualificationError, match="FACE_REGION_REQUIRES_ADDRESSED_SCENE"):
        candidate_face_region_readout_v1(legacy)


def _canonical_source_for_checkpoint_tests():
    from compiler.realsas_compiler_core.rss_vnext_source_observation_v1 import (
        canonicalize_two_view_source_v1,
    )
    rgba = np.full((2, 4, 28, 28), 200, dtype=np.uint8)
    rgba[0, 0, 3, 5] = 100
    rgba[0, 3] = 255
    rgba[1, 3] = 230
    cam = np.zeros((2, 13), dtype=np.float32)
    cam[:, 7] = 1.0
    cam[:, 12] = 1.0
    cam[0, 3] = cam[0, 11] = 1.0
    cam[1, 3] = cam[1, 11] = -1.0
    return canonicalize_two_view_source_v1(
        rgba, cam, source_rights_claim_sha256="4" * 64,
    )


def test_canonical_source_pixel_camera_frame_and_rights_claim_are_bound_without_qualification():
    from compiler.realsas_compiler_core.rss_vnext_source_observation_v1 import (
        canonicalize_two_view_source_v1,
    )
    src = _canonical_source_for_checkpoint_tests()
    rgba, cam = src.to_model_tensors()
    assert rgba.shape == (1, 2, 4, 28, 28)
    assert rgba[0, 0, 0, 3, 5].item() == pytest.approx(100 / 255)
    assert torch.equal(cam[0], torch.from_numpy(src.camera_features.copy()))
    assert src.receipt()["source_or_model_rights_verified"] is False
    assert src.receipt()["auxiliary_views_observed"] is False
    assert not src.native_rgba.flags.writeable
    changed = src.native_rgba.copy()
    changed[0, 0, 3, 5] += 1
    other = canonicalize_two_view_source_v1(
        changed, src.camera_features,
        source_rights_claim_sha256=src.source_rights_claim_sha256,
    )
    assert other.source_binding_sha256 != src.source_binding_sha256
    flipped = src.camera_features.copy()
    flipped[1, 3:6] = flipped[0, 3:6]
    flipped[1, 9:12] = flipped[0, 9:12]
    with pytest.raises(QualificationError, match="FRONT_BACK_CAMERA_OPPOSITION_REQUIRED"):
        canonicalize_two_view_source_v1(
            src.native_rgba, flipped,
            source_rights_claim_sha256=src.source_rights_claim_sha256,
        )


def test_canonical_two_view_source_checkpoint_to_compiler_is_replayable_and_never_product_pass(tmp_path):
    from models.rss_vnext.inference_pipeline_v1 import (
        run_canonical_two_view_to_compiler_proposal_v1,
        read_compiler_proposal_receipt_v1,
    )
    source = _canonical_source_for_checkpoint_tests()
    net = _network()
    ckpt = tmp_path / "model.pt"
    sha = save_inference_checkpoint_v1(ckpt, net)
    out = tmp_path / "canonical"
    receipt = run_canonical_two_view_to_compiler_proposal_v1(
        source_observation=source,
        checkpoint_path=ckpt,
        expected_checkpoint_sha256=sha,
        vision_encoder=RSSvNextPatchEncoderForTests(net.cfg.iris.image_token_dim),
        output_directory=out, max_topology_tokens=16,
        generate_controls=True, product_mode=False,
    )
    assert receipt["canonical_source_binding_sha256"] == source.source_binding_sha256
    assert receipt["source_rights_verified"] is False
    recovered, evidence = read_compiler_proposal_receipt_v1(
        out, expected_checkpoint_sha256=sha,
    )
    assert recovered == receipt
    assert evidence.qualification_status == "UNQUALIFIED_NEURAL_PROPOSAL_ONLY"
    assert receipt["product_revision_minted"] is False
    srcfile = out / "original_two_view_source.npz"
    srcfile.write_bytes(srcfile.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="CANONICAL_SOURCE_BYTES_DRIFT"):
        read_compiler_proposal_receipt_v1(out, expected_checkpoint_sha256=sha)


def test_product_inference_cannot_use_raw_two_view_tensors_without_canonical_source(tmp_path):
    from models.rss_vnext.inference_pipeline_v1 import run_two_view_to_compiler_proposal_v1
    rgba, cam = _canonical_source_for_checkpoint_tests().to_model_tensors()
    with pytest.raises(RuntimeError, match="PRODUCT_CANONICAL_SOURCE_REQUIRED"):
        run_two_view_to_compiler_proposal_v1(
            checkpoint_path=tmp_path / "missing.pt",
            expected_checkpoint_sha256="a" * 64,
            vision_encoder=_network().vision,
            observed_rgba=rgba, camera_features=cam,
            output_directory=tmp_path / "no-canon",
            product_mode=True,
        )


def test_neural_axis_v6_event_is_preserved_exactly_at_compiler_evidence_seam():
    from dataclasses import asdict
    from models.axis.rig_program_model_v6 import RigProgramEventV6, RigTokenKindV6
    event = asdict(RigProgramEventV6(RigTokenKindV6.SKIN_CODE, skin_code_id=13))
    event["kind"] = int(RigTokenKindV6.SKIN_CODE)
    value = seal_inference_evidence_v1(_candidate(control_events=(event,)))
    assert value.control_events[0]["skin_code_id"] == 13
    assert value.control_events[0]["kind"] == 17
    assert value.qualification_status == "UNQUALIFIED_NEURAL_PROPOSAL_ONLY"


@pytest.mark.parametrize("mutation,expected", [
    ({"skin_code_id": 4.75}, "CONTROL_INTEGER_FIELD_INVALID"),
    ({"skin_code_id": True}, "CONTROL_INTEGER_FIELD_INVALID"),
    ({"kind": 99}, "CONTROL_EVENT_KIND_INVALID"),
    ({"pointer_a": -3}, "CONTROL_INTEGER_FIELD_INVALID"),
    ({"dof_mask": [0.5]*9}, "CONTROL_DOF_NOT_BINARY"),
    ({"frame_6d": [float("nan")]*6}, "CONTROL_FIELD_INVALID:frame_6d"),
    ({"anchor_parent_kind": 7}, "CONTROL_ANCHOR_PARENT_INVALID"),
    ({"coord_indices": [0, 1]}, "CONTROL_COORD_INVALID"),
])
def test_compiler_neural_control_event_rejects_unsupported_or_mutated_fields(mutation, expected):
    from dataclasses import asdict
    from models.axis.rig_program_model_v6 import RigProgramEventV6, RigTokenKindV6
    base = asdict(RigProgramEventV6(RigTokenKindV6.SKIN_CODE, skin_code_id=13))
    base["kind"] = 17
    base.update(mutation)
    with pytest.raises(QualificationError, match=expected):
        seal_inference_evidence_v1(_candidate(control_events=(base,)))
