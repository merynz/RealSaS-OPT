from __future__ import annotations

from hashlib import sha256

import pytest
import torch
from torch import nn

from models.rss_vnext.training_optimizer_ownership_v1 import (
    qualify_rss_vnext_optimizer_ownership_v1,
)
from models.rss_vnext.training_phase_handoff_v1 import (
    export_rss_vnext_training_phase_handoff_v1,
    import_rss_vnext_training_phase_handoff_v1,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
)
from tests.models.test_rss_vnext_structural_trainer import (
    make_model, make_sample, manifest_hash,
)


ARCH = sha256(b"tiny-architecture").hexdigest()
CORPUS = sha256(b"qualified-corpus").hexdigest()
POLICY = sha256(b"phase-one-objectives").hexdigest()


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(641)
    torch.set_num_threads(1)


def _export(path, model, optimizer, *,
            source="SLRM", target="STRUCTURAL", policy=POLICY, step=2):
    receipt = qualify_rss_vnext_optimizer_ownership_v1(model, optimizer)
    return export_rss_vnext_training_phase_handoff_v1(
        path, model=model, source_kind=source, target_kind=target,
        source_global_step=step, dataset_manifest_sha256=CORPUS,
        source_objective_policy_sha256=policy,
        source_optimizer_binding_sha256=receipt.binding_sha256,
        architecture_id="test-model", architecture_digest=ARCH,
    )


def _import(path, model, optimizer, receipt, *, policy=None, corpus=CORPUS,
            source=None, arch=ARCH):
    return import_rss_vnext_training_phase_handoff_v1(
        path, model=model, optimizer=optimizer,
        target_kind="STRUCTURAL",
        expected_source_kind=source or receipt.source_kind,
        expected_source_objective_policy_sha256=policy or receipt.source_objective_policy_sha256,
        expected_source_optimizer_binding_sha256=receipt.source_optimizer_binding_sha256,
        dataset_manifest_sha256=corpus,
        architecture_id="test-model", architecture_digest=arch,
    )


def test_slrm_to_structural_carries_exact_weights_not_optimizer_moments(tmp_path):
    source = nn.Linear(4, 2)
    source_opt = torch.optim.AdamW(source.parameters(), lr=1e-3)
    source(torch.randn(3, 4)).square().sum().backward()
    source_opt.step()
    assert source_opt.state  # Source has optimizer moments.
    path = tmp_path / "handoff.pt"
    receipt = _export(path, source, source_opt)
    target = nn.Linear(4, 2)
    target_opt = torch.optim.AdamW(target.parameters(), lr=2e-4)
    assert not target_opt.state
    accepted = _import(path, target, target_opt, receipt)
    assert accepted.model_state_sha256 == receipt.model_state_sha256
    assert accepted.source_global_step == 2
    assert all(torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters()))
    assert not target_opt.state
    assert all(p.grad is None for p in target.parameters())


def test_handoff_refuses_origin_drift_before_touching_model(tmp_path):
    source = nn.Linear(4, 2)
    opt = torch.optim.AdamW(source.parameters())
    path = tmp_path / "source.pt"
    receipt = _export(path, source, opt)
    target = nn.Linear(4, 2)
    target_opt = torch.optim.AdamW(target.parameters())
    original = [p.detach().clone() for p in target.parameters()]
    for changed in (
        {"policy": sha256(b"different-loss").hexdigest()},
        {"corpus": sha256(b"different-manifest").hexdigest()},
        {"arch": sha256(b"different-model").hexdigest()},
        {"source": "STRUCTURAL"},
    ):
        with pytest.raises(ValueError, match="PROVENANCE_DRIFT"):
            _import(path, target, target_opt, receipt, **changed)
        assert all(torch.equal(x, p) for x, p in zip(original, target.parameters()))


def test_handoff_refuses_tampered_weights_and_model_abi(tmp_path):
    source = nn.Linear(4, 2)
    source_opt = torch.optim.AdamW(source.parameters())
    path = tmp_path / "weights.pt"
    receipt = _export(path, source, source_opt)
    payload = torch.load(path, weights_only=True)
    payload["model"]["weight"][0, 0] += 10.
    torch.save(payload, path)
    target = nn.Linear(4, 2)
    target_opt = torch.optim.AdamW(target.parameters())
    with pytest.raises(ValueError, match="MODEL_DIGEST_DRIFT"):
        _import(path, target, target_opt, receipt)

    # Re-signing the payload does not let the importer load a different ABI.
    path2 = tmp_path / "abi.pt"
    receipt2 = _export(path2, source, source_opt)
    other = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="MODEL_ABI_DRIFT"):
        _import(path2, other, torch.optim.AdamW(other.parameters()), receipt2)


def test_handoff_refuses_dirty_destination_optimizer_and_unsupported_transition(tmp_path):
    source = nn.Linear(4, 2)
    source_opt = torch.optim.AdamW(source.parameters())
    path = tmp_path / "stage.pt"
    receipt = _export(path, source, source_opt)
    target = nn.Linear(4, 2)
    target_opt = torch.optim.AdamW(target.parameters())
    target(torch.randn(2, 4)).sum().backward()
    target_opt.step()
    with pytest.raises(ValueError, match="DESTINATION_OPTIMIZER_NOT_FRESH"):
        _import(path, target, target_opt, receipt)
    with pytest.raises(ValueError, match="TRANSITION_NOT_ALLOWED"):
        _export(path, source, source_opt, target="MIRA_CODEC")
    with pytest.raises(ValueError, match="SOURCE_POLICY_INVALID"):
        _export(path, source, source_opt, step=0)


def test_structural_phase_transition_restarts_optimizer_and_binds_resume_origin(tmp_path):
    source_model = make_model()
    source = RSSvNextStructuralTrainer(
        source_model, torch.optim.AdamW(source_model.parameters(), lr=1e-3),
        dataset_manifest_sha256=manifest_hash(),
    )
    source.train_step(make_sample(source_model))
    handoff_path = tmp_path / "structural-handoff.pt"
    receipt = source.export_phase_handoff(handoff_path)
    assert receipt.source_kind == "STRUCTURAL"

    next_model = make_model()
    next_policy = RSSvNextTrainingWeights(
        topology=1., motion_preference=0., source_semantics=0.,
        flexicubes_surface=0.,
    )
    next_stage = RSSvNextStructuralTrainer(
        next_model, torch.optim.AdamW(next_model.parameters(), lr=1e-4),
        dataset_manifest_sha256=manifest_hash(), weights=next_policy,
    )
    accepted = next_stage.import_phase_handoff(
        handoff_path,
        expected_source_kind="STRUCTURAL",
        expected_source_objective_policy_sha256=source.objective_policy_sha256,
        expected_source_optimizer_binding_sha256=source.optimizer_ownership.binding_sha256,
    )
    assert accepted.model_state_sha256 == receipt.model_state_sha256
    assert next_stage.global_step == 0
    assert not next_stage.optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(
        source_model.parameters(), next_model.parameters(),
    ))
    next_stage.train_step(make_sample(next_model))
    resume_path = tmp_path / "next-stage-resume.pt"
    next_stage.save_checkpoint(resume_path, sampler_cursor=4)

    independent_model = make_model()
    independent = RSSvNextStructuralTrainer(
        independent_model, torch.optim.AdamW(independent_model.parameters()),
        dataset_manifest_sha256=manifest_hash(), weights=next_policy,
    )
    with pytest.raises(ValueError, match="CHECKPOINT_PHASE_ORIGIN_DRIFT"):
        independent.load_checkpoint(resume_path)
    independent.import_phase_handoff(
        handoff_path,
        expected_source_kind="STRUCTURAL",
        expected_source_objective_policy_sha256=source.objective_policy_sha256,
        expected_source_optimizer_binding_sha256=source.optimizer_ownership.binding_sha256,
    )
    assert independent.load_checkpoint(resume_path) == 4
    assert independent.global_step == 1
