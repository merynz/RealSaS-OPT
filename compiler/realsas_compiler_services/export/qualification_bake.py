from __future__ import annotations

"""Proof-to-deploy handoff for qualification-owned motion frames."""

from compiler.realsas_compiler_core.v4 import require_current_proof_bundle
from compiler.realsas_compiler_services.proof.motion_bake import (
    QualificationOwnedMotionBakeIR,
    assert_motion_bake_binding,
    deploy_frame_dicts,
)
from .runtime_deploy_bake import encode_runtime_deploy_bake


def require_motion_bake_proven_by_product_bundle(product, proof_bundle, bake: QualificationOwnedMotionBakeIR):
    require_current_proof_bundle(product, proof_bundle, require_pass=True)
    reports = [r for r in proof_bundle.domain_reports if r.proof_domain == "MOTION"]
    if len(reports) != 1 or reports[0].status != "PASS":
        raise ValueError("RUNTIME_DEPLOY_BAKE_REQUIRES_PASS_MOTION_DOMAIN")
    report = reports[0]
    assert_motion_bake_binding(bake, source_product_state_hash=product.product_state_hash, proof_plan_hash=report.proof_plan_hash)
    hashes = dict(report.metadata.get("qualification_owned_bake_hashes") or {})
    single = str(report.metadata.get("qualification_owned_bake_hash") or "")
    if hashes:
        if hashes.get(bake.clip_id) != bake.bake_hash:
            raise ValueError("RUNTIME_DEPLOY_BAKE_NOT_BOUND_BY_MOTION_PROOF")
    elif single != bake.bake_hash:
        raise ValueError("RUNTIME_DEPLOY_BAKE_NOT_BOUND_BY_MOTION_PROOF")
    if report.metadata.get("export_solver_replay_forbidden") is not True:
        raise ValueError("RUNTIME_EXPORT_REQUIRES_NO_SOLVER_REPLAY_CONTRACT")
    return report


def encode_proven_motion_bake(product, proof_bundle, bake: QualificationOwnedMotionBakeIR) -> dict:
    require_motion_bake_proven_by_product_bundle(product, proof_bundle, bake)
    return encode_runtime_deploy_bake(
        clip_id=bake.clip_id,
        duration_seconds=bake.duration_seconds,
        fps=bake.fps,
        loop=bake.loop,
        sample_times_seconds=tuple(frame.time_seconds for frame in bake.frames),
        frames=deploy_frame_dicts(bake),
        evaluator_semantic_version=bake.evaluator_semantic_version,
        sampling_policy=bake.sampling_policy,
    )
