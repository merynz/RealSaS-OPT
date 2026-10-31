from __future__ import annotations

"""Fail-closed execution wrapper for the modular product Proof evaluator.

The authority schemas and domain kernels live in ``modular_product_proof_v1``.
This wrapper only ensures domain consequences are reported as typed FAIL rows
instead of escaping before a ProductProofBundleIR can be minted.
"""

from dataclasses import replace
from typing import Callable, Iterable

from .canonical_graphics_ir_v1 import CanonicalGraphicsIRV1
from .canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .canonical_motion_ir_v2 import CanonicalMotionIRV2
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .modular_product_proof_v1 import (
    MODULAR_PRODUCT_PROOF_EVALUATOR_V1,
    MODULAR_PRODUCT_STATE_SCHEMA_V1,
    REQUIRED_PROOF_DOMAINS_V1,
    _bind_domain_report,
    _canonical_motions,
    _directional_visual_measurements,
    _mechanical_structure_measurements,
    _mesh_quality_measurements,
    _motion_and_deformation_measurements,
    _runtime_consumption_measurements,
    modular_product_state_hash_v1,
)
from .types import QualificationError
from .v4 import proof_bundle_hash
from .v4_types import ProductProofBundleIR

Json = dict[str, object]


def _require_authority(
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
) -> None:
    if rig.metadata.get("authority") != "COMPILER_CANONICAL_RIG":
        raise QualificationError("MODULAR_PROOF_RIG_AUTHORITY_METADATA_DRIFT")
    if skin.metadata.get("authority") != "COMPILER_CANONICAL_SKIN":
        raise QualificationError("MODULAR_PROOF_SKIN_AUTHORITY_METADATA_DRIFT")


def _motion_domain(
    *,
    index: int,
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    motions: tuple[CanonicalMotionIRV2, ...],
) -> dict:
    measured = _motion_and_deformation_measurements(
        mesh,
        rig,
        skin,
        mechanics,
        motions,
    )
    return measured[index]


def evaluate_modular_product_proof_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    mechanics: CanonicalMechanicsIRV1,
    graphics: CanonicalGraphicsIRV1,
    motions: Iterable[CanonicalMotionIRV2],
) -> ProductProofBundleIR:
    motions = _canonical_motions(motions)
    _require_authority(rig, skin)
    state_hash = modular_product_state_hash_v1(
        mesh,
        rig,
        skin,
        mechanics,
        graphics,
        motions,
    )

    evaluators: dict[str, Callable[[], dict]] = {
        "MECHANICAL_STRUCTURE": lambda: _mechanical_structure_measurements(
            mesh,
            rig,
            skin,
            mechanics,
        ),
        "MESH_QUALITY": lambda: _mesh_quality_measurements(mesh),
        "DEFORMATION": lambda: _motion_domain(
            index=1,
            mesh=mesh,
            rig=rig,
            skin=skin,
            mechanics=mechanics,
            motions=motions,
        ),
        "DIRECTIONAL_VISUAL": lambda: _directional_visual_measurements(graphics, mesh),
        "MOTION": lambda: _motion_domain(
            index=0,
            mesh=mesh,
            rig=rig,
            skin=skin,
            mechanics=mechanics,
            motions=motions,
        ),
        "RUNTIME_CONSUMPTION": lambda: _runtime_consumption_measurements(
            state_hash,
            mesh,
            rig,
            skin,
            mechanics,
            graphics,
            motions,
        ),
    }
    reports = tuple(
        _bind_domain_report(state_hash, domain, evaluators[domain])
        for domain in REQUIRED_PROOF_DOMAINS_V1
    )
    overall = "PASS" if all(report.status == "PASS" for report in reports) else "FAIL"
    value = ProductProofBundleIR(
        source_product_state_hash=state_hash,
        required_domains=REQUIRED_PROOF_DOMAINS_V1,
        domain_reports=reports,
        overall_status=overall,
        proof_bundle_hash="",
        metadata={
            "product_state_schema": MODULAR_PRODUCT_STATE_SCHEMA_V1,
            "evaluator": MODULAR_PRODUCT_PROOF_EVALUATOR_V1,
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "rig_lineage_hash": rig.rig_lineage_hash,
            "skin_lineage_hash": skin.skin_lineage_hash,
            "mechanics_lineage_hash": mechanics.mechanics_lineage_hash,
            "graphics_lineage_hash": graphics.graphics_lineage_hash,
            "motion_lineage_hashes": [motion.motion_lineage_hash for motion in motions],
            "legacy_canonical_puppet_graph_projection_used": False,
            "product_pass_authority": "PRODUCT_PROOF_BUNDLE_IR_V1",
        },
    )
    return replace(value, proof_bundle_hash=proof_bundle_hash(value))


__all__ = ["evaluate_modular_product_proof_v1"]
