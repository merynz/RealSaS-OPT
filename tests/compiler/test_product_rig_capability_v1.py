from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
AXIS_ARCH = "RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1"
CHECKPOINT_SCHEMA = "RealSaS.AXISBFSDiscreteXYZV541.DeployableCheckpoint.v1"
AXIS_INPUT_ADAPTER = "RealSaS.GeppettoRiggingSurfaceTensorization.v1"


def test_rig_executor_module_import_does_not_load_torch():
    script = r'''
import builtins
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise RuntimeError("TORCH_IMPORT_FORBIDDEN_DURING_RIG_EXECUTOR_IMPORT")
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
from compiler.realsas_compiler_services.platform_worker import rig_executor_v1
assert rig_executor_v1.RIG_EXECUTOR_CONTRACT_V1 == "RealSaS.ProductRigCompile.v1"
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def _deployable_payload(model):
    return {
        "schema": CHECKPOINT_SCHEMA,
        "architecture_id": AXIS_ARCH,
        "config_hash": str(model.config.config_hash),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "runtime_diffusion_retired": True,
        "legacy_diffusion_parameters_present": False,
        "hard_causal_parent_authority": True,
        "compiler_role": "VALIDATE_AND_MATERIALIZE",
        "compiler_parent_reselection_allowed": False,
        "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
        "coord_bins_per_axis": 256,
        "full_runtime_state": True,
        "optimizer_state": None,
        "teacher_runtime_authority": False,
        "historical_fit_provenance": {
            "base_checkpoint_sha256": "a" * 64,
            "v53_progress_sha256": "b" * 64,
            "v541_composite_sha256": "c" * 64,
            "historical_three_part_runtime_dependency": False,
        },
    }


def test_axis_v541_deployable_checkpoint_is_one_strict_retired_runtime_state(tmp_path):
    torch = pytest.importorskip("torch")
    from models.axis.axis_bfs_discrete_xyz_v541 import (
        AXISBFSDiscreteXYZV541,
        ForbiddenRuntimeDiffusionV541,
    )
    from models.axis.deployable_checkpoint_v541 import (
        load_axis_v541_deployable_checkpoint_v1,
    )

    model = AXISBFSDiscreteXYZV541().cpu().float().retire_legacy_diffusion_runtime()
    payload = _deployable_payload(model)
    assert not any(name.startswith("diffusion.") for name in payload["model_state_dict"])
    path = tmp_path / "axis-v541-deployable.pt"
    torch.save(payload, path)

    loaded, checked = load_axis_v541_deployable_checkpoint_v1(
        path, device=torch.device("cpu")
    )
    assert checked["schema"] == CHECKPOINT_SCHEMA
    assert isinstance(loaded.diffusion, ForbiddenRuntimeDiffusionV541)
    assert loaded.diffusion.invocation_count == 0
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert set(loaded.state_dict()) == set(payload["model_state_dict"])


def test_axis_v541_deployable_checkpoint_rejects_hidden_diffusion_state(tmp_path):
    torch = pytest.importorskip("torch")
    from models.axis.axis_bfs_discrete_xyz_v541 import AXISBFSDiscreteXYZV541
    from models.axis.deployable_checkpoint_v541 import (
        load_axis_v541_deployable_checkpoint_v1,
    )

    model = AXISBFSDiscreteXYZV541().cpu().float().retire_legacy_diffusion_runtime()
    payload = _deployable_payload(model)
    payload["model_state_dict"]["diffusion.hidden_dependency"] = torch.zeros(1)
    path = tmp_path / "axis-v541-invalid.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="AXIS_V541_DEPLOYABLE_DIFFUSION_STATE_FORBIDDEN"):
        load_axis_v541_deployable_checkpoint_v1(path, device=torch.device("cpu"))


def test_rig_executor_keeps_axis_proposal_internal_and_emits_one_compiler_rig(monkeypatch):
    from compiler.realsas_compiler_core import artifact_codec_v2, canonical_mesh_ir_v1
    from compiler.realsas_compiler_core import required_tree_v1
    from compiler.realsas_compiler_services.platform_worker import model_binding
    from compiler.realsas_compiler_services.platform_worker import rig_executor_v1 as rig
    from models.axis import deployable_checkpoint_v541
    from models.geppetto.reference_strength_v1 import rigging_surface_tensorization_v1

    mesh = SimpleNamespace(
        source_evidence_binding_hash="s" * 64,
        mesh_lineage_hash="m" * 64,
        carrier_evidence_hash="e" * 64,
        carrier_topology_hash="t" * 64,
        carrier_geometry_hash="g" * 64,
        static_qualification_hash="q" * 64,
    )
    surface = SimpleNamespace(geometry_lineage_hash="s" * 64)
    surface_tensor = SimpleNamespace(
        source_surface_hash="s" * 64,
        tensorization_hash="x" * 64,
        node_count=4,
    )
    proposal = SimpleNamespace(
        metadata={
            "candidate_architecture": AXIS_ARCH,
            "hard_causal_parent_authority": True,
            "compiler_parent_reselection_allowed": False,
            "diffusion_runtime_authority": False,
        },
        to_dict=lambda: {"schema_version": "RealSaS.SkeletonProposalIR.v1", "proposal": "internal"},
    )
    skeleton_payload = {
        "schema_version": "RealSaS.QualifiedSkeletonIR.v1",
        "skeleton_lineage_hash": "k" * 64,
        "joints": [{"id": "J:0"}],
    }
    skeleton = SimpleNamespace(
        skeleton_lineage_hash="k" * 64,
        joints=(object(),),
        to_dict=lambda: dict(skeleton_payload),
    )
    axis_ref = SimpleNamespace(
        checkpoint_path=Path("/tmp/axis.pt"),
        model_release_id="axis-release",
        behavior_identity_sha256="b" * 64,
        checkpoint_content_sha256="c" * 64,
    )

    monkeypatch.setattr(
        rig,
        "materialize_model_releases",
        lambda **kwargs: [{"family": "AXIS"}],
    )
    monkeypatch.setattr(rig, "_device_from_environment", lambda: "cpu")
    monkeypatch.setattr(
        model_binding,
        "require_materialized_model_release",
        lambda *args, **kwargs: axis_ref,
    )
    monkeypatch.setattr(canonical_mesh_ir_v1, "canonical_mesh_from_dict_v1", lambda payload: mesh)
    monkeypatch.setattr(artifact_codec_v2, "rigging_surface_from_dict", lambda payload: surface)
    monkeypatch.setattr(artifact_codec_v2, "qualified_skeleton_from_dict", lambda payload: skeleton)
    monkeypatch.setattr(
        rigging_surface_tensorization_v1,
        "tensorize_rigging_surface_v1",
        lambda value, require_scene_first=True: surface_tensor,
    )

    class FakeModel:
        def propose(self, value, *, resource_step_limit, diffusion_sample_steps):
            assert value is surface_tensor
            assert resource_step_limit == 4
            assert diffusion_sample_steps is None
            return proposal

    monkeypatch.setattr(
        deployable_checkpoint_v541,
        "load_axis_v541_deployable_checkpoint_v1",
        lambda path, *, device: (FakeModel(), {"schema": CHECKPOINT_SCHEMA}),
    )

    def qualify(value_surface, value_proposal, *, authority_bindings):
        assert value_surface is surface
        assert value_proposal is proposal
        assert authority_bindings["canonical_mesh_lineage_hash"] == mesh.mesh_lineage_hash
        assert authority_bindings["axis_model_release_id"] == "axis-release"
        assert authority_bindings["axis_input_adapter_contract"] == AXIS_INPUT_ADAPTER
        assert authority_bindings["axis_input_surface_evidence_hash"] == surface.geometry_lineage_hash
        return skeleton

    monkeypatch.setattr(required_tree_v1, "qualify_required_skeleton_tree_v1", qualify)

    request = {
        "capability_id": "product.rig",
        "owner_module_id": "product.rig",
        "kind": "MODULE",
        "execution_id": "exec-rig",
        "capability_metadata": {
            "executor_contract": "RealSaS.ProductRigCompile.v1",
            "canonical_parent_role": "mesh",
            "axis_input_adapter_contract": AXIS_INPUT_ADAPTER,
            "axis_input_adapter_is_canonical_authority": False,
            "axis_input_adapter_replaceable": True,
            "surface_dependency_is_compatibility_evidence_only": True,
            "axis_checkpoint_schema": CHECKPOINT_SCHEMA,
            "maximum_joint_steps": 256,
            "compiler_parent_reselection_allowed": False,
            "proposal_authority_artifact_emitted": False,
            "canonical_mesh_mutation_allowed": False,
            "post_bind_frame_observability_claimed": False,
            "frame_observability_deferred_to_mechanics": True,
        },
        "input_artifacts": [
            {
                "artifact_type": "RealSaS.CanonicalMeshIR",
                "schema_version": "v1",
                "content_sha256": "d" * 64,
            },
            {
                "artifact_type": "RealSaS.RiggingSurfaceIR",
                "schema_version": "v1",
                "content_sha256": "e" * 64,
            },
        ],
        "model_releases": [{"family": "AXIS"}],
    }
    sealed_payloads = []

    def read_object(ref):
        return b"{}"

    def put_bytes(data):
        sealed_payloads.append(json.loads(data.decode("utf-8")))
        return {
            "storage_key": "cas/rig",
            "content_sha256": "f" * 64,
            "size_bytes": len(data),
        }

    result = rig.execute_rig_capability_v1(
        request,
        read_object=read_object,
        put_bytes=put_bytes,
    )
    assert result["status"] == "PASS"
    assert len(result["outputs"]) == 1
    assert result["outputs"][0]["role"] == "canonical_rig"
    assert result["outputs"][0]["artifact_type"] == "RealSaS.QualifiedSkeletonIR"
    assert result["outputs"][0]["authority_class"] == "COMPILER_CANONICAL_RIG"
    assert sealed_payloads == [skeleton_payload]
    assert result["diagnostics"]["canonical_parent_role"] == "mesh"
    assert result["diagnostics"]["axis_input_adapter_contract"] == AXIS_INPUT_ADAPTER
    assert result["diagnostics"]["axis_input_adapter_is_canonical_authority"] is False
    assert result["diagnostics"]["axis_input_adapter_replaceable"] is True
    assert result["diagnostics"]["surface_dependency_is_compatibility_evidence_only"] is True
    assert result["diagnostics"]["proposal_authority_artifact_emitted"] is False
    assert result["diagnostics"]["compiler_parent_reselection_performed"] is False
    assert result["diagnostics"]["canonical_mesh_mutated"] is False
    assert result["diagnostics"]["post_bind_frame_observability_claimed"] is False
    assert result["diagnostics"]["frame_observability_deferred_to_mechanics"] is True


def test_axis_v541_historical_fit_lineage_collapses_to_one_deployable_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from models.axis.axis_bfs_discrete_xyz_v541 import AXISBFSDiscreteXYZV541
    from models.axis.deployable_checkpoint_v541 import (
        load_axis_v541_deployable_checkpoint_v1,
        seal_axis_v541_deployable_checkpoint_v1,
    )

    historical = AXISBFSDiscreteXYZV541().cpu().float()
    full_state = historical.state_dict()
    coord_names = {name for name in full_state if name.startswith("coord_token_delta.")}
    assert coord_names
    base_state = {
        name: value.detach().cpu().clone()
        for name, value in full_state.items()
        if name not in coord_names
    }
    base = tmp_path / "GEPPETTO_KNIGHT_CHECKPOINT.pt"
    torch.save(
        {"model": base_state, "config_hash": str(historical.config.config_hash)},
        base,
    )
    base_sha = sha256(base.read_bytes()).hexdigest()

    v53_key = next(name for name in base_state if not name.startswith("diffusion."))
    v53 = tmp_path / "AXIS41_TREE_GRAMMAR_PROGRESS_V53.pt"
    torch.save(
        {
            "schema": "RealSaS.AXIS41TreeGrammarProgress.v5_3",
            "base_checkpoint_sha256": base_sha,
            "trainable_delta": {v53_key: base_state[v53_key].clone()},
        },
        v53,
    )
    v53_sha = sha256(v53.read_bytes()).hexdigest()

    composite = tmp_path / "AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_COMPOSITE_V541.pt"
    torch.save(
        {
            "schema": "RealSaS.AXIS41DiscreteXYZComposite.v5_4_1",
            "architecture": AXIS_ARCH,
            "base_checkpoint_sha256": base_sha,
            "v53_progress_sha256": v53_sha,
            "hard_causal_parent_authority": True,
            "compiler_tree_reselection": False,
            "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
            "diffusion_runtime_authority": False,
            "trainable_model_delta": {
                name: full_state[name].detach().cpu().clone() for name in coord_names
            },
        },
        composite,
    )

    output = tmp_path / "AXIS_V541_DEPLOYABLE.pt"
    manifest = seal_axis_v541_deployable_checkpoint_v1(
        base_checkpoint_path=base,
        v53_progress_path=v53,
        v541_composite_path=composite,
        output_path=output,
    )
    assert manifest["schema"] == CHECKPOINT_SCHEMA
    assert manifest["checkpoint_content_sha256"] == sha256(output.read_bytes()).hexdigest()
    loaded, payload = load_axis_v541_deployable_checkpoint_v1(
        output, device=torch.device("cpu")
    )
    assert payload["historical_fit_provenance"] == {
        "base_checkpoint_sha256": base_sha,
        "v53_progress_sha256": v53_sha,
        "v541_composite_sha256": sha256(composite.read_bytes()).hexdigest(),
        "historical_three_part_runtime_dependency": False,
    }
    assert not any(name.startswith("diffusion.") for name in loaded.state_dict())
    assert set(loaded.state_dict()) == set(payload["model_state_dict"])
