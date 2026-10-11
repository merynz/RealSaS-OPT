"""Real qualification + byte seals after the three scheduling contractions."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    ModelFitExecutionIR,
    model_fit_execution_hash,
)
from compiler.realsas_compiler_core.types import (
    RiggingSurfaceIR,
    SurfaceNode,
    QualificationError,
)
from compiler.realsas_compiler_services.orchestrator.adapters import (
    learned_mechanics_v3 as mechanics,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    write_ir,
    sha256_file,
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.model_execution_common_v2 import (
    seal_model_checkpoint,
)


def install(ctx, stage, value, name):
    output = write_ir(ctx["run_root"] / name, value, authority_class="TEST_FIXTURE")
    ctx["ledger"]["stages"].append({"id": stage, "status": "PASS", "outputs": [output]})


def setup(tmp_path):
    ctx = {"run_root": tmp_path, "ledger": {"stages": []}}
    surface = RiggingSurfaceIR(
        (SurfaceNode("s0", (0.0, 0.0, 0.0), (0, 1), (), ()),), (), "fixture-surface"
    )
    candidate = SimpleNamespace(
        vertices=tuple(
            SimpleNamespace(candidate_vertex_id=f"v{i}", P=p)
            for i, p in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        ),
        faces=(("v0", "v1", "v2"),),
        candidate_lineage_hash="c" * 64,
        surface_binding_hash=surface.geometry_lineage_hash,
    )
    carrier = build_mechanical_carrier_evidence_v1(
        candidate,
        static_qualification=SimpleNamespace(
            candidate_mesh_binding_hash="c" * 64,
            surface_addressing_binding_hash="a" * 64,
            qualification_hash="q" * 64,
        ),
        surface_addressing=SimpleNamespace(
            candidate_mesh_binding_hash="c" * 64, addressing_hash="a" * 64
        ),
    )
    install(ctx, "15_RIGGING_SURFACE_QUALIFIED", surface, "surface.json")
    install(ctx, "19_STATIC_CANONICAL_MESH_QUALIFIED", carrier, "carrier.json")
    return ctx, surface, carrier


def execution(ctx, carrier, lane, proposal, policy):
    root = ctx["run_root"]
    checkpoint = root / f"{lane}.bin"
    checkpoint.write_bytes(b"sealed checkpoint " + lane.encode())
    result = root / f"{lane}_result.json"
    result.write_text("{}")
    path = root / f"{lane}_proposal.json"
    path.write_text(json.dumps(proposal))
    value = ModelFitExecutionIR(
        lane,
        "fixture",
        "p" * 64,
        "m" * 64,
        tuple(
            {
                "mechanical_carrier_evidence": carrier.carrier_evidence_hash,
                "mechanical_carrier_topology": carrier.topology_hash,
                "mechanical_carrier_geometry": carrier.geometry_hash,
            }.items()
        ),
        str(result),
        sha256_file(result),
        str(checkpoint),
        sha256_file(checkpoint),
        str(result),
        sha256_file(result),
        str(path),
        sha256_file(path),
        policy,
        {},
        "",
    )
    return replace(value, execution_hash=model_fit_execution_hash(value))


def qualify_axis(ctx, surface, carrier):
    proposal = {
        "schema_version": "RealSaS.SkeletonProposalIR.v1",
        "surface_binding_hash": surface.geometry_lineage_hash,
        "joints": [
            {
                "proposal_id": "root",
                "position": [0.0, 0.0, 0.0],
                "root_score": 1.0,
                "confidence": 1.0,
                "support_surface_ids": ["s0"],
                "metadata": {
                    "hard_causal_parent_authority": True,
                    "discrete_xyz_authority": True,
                    "coord_bins_per_axis": 256,
                },
            }
        ],
        "edges": [],
        "metadata": {
            "hard_causal_parent_authority": True,
            "compiler_role": "VALIDATE_AND_MATERIALIZE",
            "compiler_parent_reselection_allowed": False,
            "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
            "diffusion_runtime_authority": False,
            "conditional_residual_diffusion": False,
        },
    }
    fit = execution(ctx, carrier, "GEPPETTO", proposal, {})
    install(ctx, "27_GEPPETTO_FIT", fit, "axis_execution.json")
    result = mechanics.qualify_skeleton_stage(ctx)
    ctx["ledger"]["stages"].append(
        {"id": "28_SKELETON_QUALIFIED", "status": "PASS", "outputs": result["outputs"]}
    )
    return fit, result


def assert_same_seal(ctx, fit, output, qualified_hash, old_stage):
    seal, old_output = seal_model_checkpoint(
        ctx,
        execution=fit,
        qualified_output_binding_hash=qualified_hash,
        stage_id=old_stage,
    )
    assert Path(output["path"]).read_bytes() == Path(old_output["path"]).read_bytes()
    assert output["sha256"] == old_output["sha256"]


def test_checkpoint_payload_bytes_are_preserved_and_skeleton_gate_still_runs(tmp_path):
    ctx, surface, carrier = setup(tmp_path)
    fit, result = qualify_axis(ctx, surface, carrier)
    skeleton = stage_output_payload(
        ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
    )
    seal = next(
        o
        for o in result["outputs"]
        if o["schema"] == "RealSaS.ModelCheckpointSealIR.v1"
    )
    assert_same_seal(
        ctx,
        fit,
        seal,
        skeleton["skeleton_lineage_hash"],
        "29_GEPPETTO_CHECKPOINT_SEALED",
    )
    Path(fit.checkpoint_path).write_bytes(b"drift")
    with pytest.raises(QualificationError, match="MODEL_CHECKPOINT_BYTES_DRIFT"):
        mechanics.qualify_skeleton_stage(ctx)


def test_skin_qualification_and_checkpoint_seal_share_one_producer(tmp_path):
    ctx, surface, carrier = setup(tmp_path)
    qualify_axis(ctx, surface, carrier)
    skeleton = stage_output_payload(
        ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
    )
    jid = skeleton["root_id"]
    proposal = {
        "schema_version": "RealSaS.CarrierSkinProposalIR.v1",
        "influences": [
            {"carrier_vertex_id": v, "canonical_joint_id": jid, "weight": 1.0}
            for v in carrier.ordered_vertex_ids
        ],
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "carrier_topology_hash": carrier.topology_hash,
        "carrier_geometry_hash": carrier.geometry_hash,
        "skeleton_binding_hash": skeleton["skeleton_lineage_hash"],
        "metadata": {
            "output_domain": "MECHANICAL_CARRIER_M",
            "semantic_skin_transfer_performed": False,
            "gsa_skin_field_authority_minted": False,
            "direct_simplex_readout": True,
            "source_skin_runtime_authority": False,
            "mesh_mutation_invalidates_prediction": True,
        },
    }
    policy = {
        "max_simplex_repair_l1": 0.0,
        "max_total_correction_l1": 0.0,
        "negative_tolerance": 1e-12,
        "max_influences": 1,
    }
    fit = execution(ctx, carrier, "ARACHNE", proposal, policy)
    install(ctx, "31_ARACHNE_FIT", fit, "mira_execution.json")
    result = mechanics.qualify_skin_stage(ctx)
    ctx["ledger"]["stages"].append(
        {"id": "32_SKIN_QUALIFIED", "status": "PASS", "outputs": result["outputs"]}
    )
    skin = stage_output_payload(
        ctx, "32_SKIN_QUALIFIED", "RealSaS.QualifiedCarrierSkinIR.v1"
    )
    seal = next(
        o
        for o in result["outputs"]
        if o["schema"] == "RealSaS.ModelCheckpointSealIR.v1"
    )
    assert_same_seal(
        ctx, fit, seal, skin["skin_lineage_hash"], "33_ARACHNE_CHECKPOINT_SEALED"
    )
    assert all(row["influences"] == [[jid, 1.0]] for row in skin["rows"])
    Path(fit.result_path).write_text('{"drift":true}')
    with pytest.raises(QualificationError, match="MODEL_CHECKPOINT_RESULT_BYTES_DRIFT"):
        mechanics.qualify_skin_stage(ctx)
