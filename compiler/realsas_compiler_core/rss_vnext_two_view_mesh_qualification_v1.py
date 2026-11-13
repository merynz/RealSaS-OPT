from __future__ import annotations

"""Two *observed* camera mesh admission, not eight invented source views.

Original front/back rasters constrain an inferred TESSA 3D carrier but do not
uniquely determine depth, hidden surfaces or deformation. This court may mint
CanonicalMeshIR with clear model-generated geometry provenance only when the
candidate mesh passes *both* real-source silhouette tests, conditioning and
exact topology/addressability. It does not grant attachment/rig/graphics/motion
or ProductRevision authority; those are independent downstream qualifications.
"""

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from .canonical_mesh_ir_v1 import CanonicalMeshIR, mint_canonical_mesh_v1
from .hashing import content_sha256
from .mechanical_carrier_evidence_v1 import build_mechanical_carrier_evidence_v1
from .product_authority_v1 import (
    MeshQualificationPolicyIR, validate_mesh_qualification_policy,
)
from .rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1, validate_inference_evidence_v1,
)
from .rss_vnext_neural_mesh_proposal_basis_v1 import (
    RSSvNextNeuralMeshCandidateBasisV1,
)
from .rss_vnext_source_evidence_binding_v1 import (
    RSSvNextSourceEvidenceBindingCourtV1,
)
from .rss_vnext_source_observation_v1 import RSSvNextTwoViewSourceObservationV1
from .rss_vnext_two_view_raster_diagnostic_v1 import (
    RSSvNextTwoViewRasterDiagnosticV1,
    compile_rss_vnext_two_view_raster_diagnostic_v1,
    validate_rss_vnext_two_view_raster_diagnostic_v1,
)
from .static_mesh_qualification_v1 import relation_parent_quality_report_v1
from .surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR, build_surface_addressing,
    static_mesh_qualification_hash,
)
from .types import QualificationError


@dataclass(frozen=True)
class RSSvNextTwoViewStaticPolicyV1:
    # Explicit versioned calibration inputs; the release must freeze and sign
    # these independently of inference. No silent use of Stage13's 8-view P999.
    min_recall: float
    min_precision: float
    max_largest_coherent_hole_fraction: float
    max_interior_uncovered_fraction: float
    min_component_recall: float
    max_silhouette_edge_p95_px: float
    source_alpha_threshold: int = 128
    schema: str = "RealSaS.RSSvNext.TwoObservedViewMeshPolicy.v1"

    def validate(self) -> None:
        if (
            type(self) is not RSSvNextTwoViewStaticPolicyV1
            or self.schema != "RealSaS.RSSvNext.TwoObservedViewMeshPolicy.v1"
            or self.source_alpha_threshold != 128
        ):
            raise QualificationError("RSS_VNEXT_TWO_VIEW_STATIC_POLICY_SCHEMA_OR_ALPHA_DRIFT")
        values = (
            self.min_recall, self.min_precision,
            self.max_largest_coherent_hole_fraction,
            self.max_interior_uncovered_fraction,
            self.min_component_recall,
        )
        if any(type(x) not in (float, int) or not math.isfinite(x)
               or not 0 <= x <= 1 for x in values):
            raise QualificationError("RSS_VNEXT_TWO_VIEW_STATIC_POLICY_PROBABILITY_DOMAIN")
        if (
            self.min_recall <= 0 or self.min_precision <= 0
            or self.min_component_recall <= 0
            or type(self.max_silhouette_edge_p95_px) not in (float, int)
            or not math.isfinite(self.max_silhouette_edge_p95_px)
            or self.max_silhouette_edge_p95_px <= 0
        ):
            raise QualificationError("RSS_VNEXT_TWO_VIEW_STATIC_POLICY_THRESHOLD_INVALID")


@dataclass(frozen=True)
class RSSvNextTwoViewMeshAdmissionV1:
    status: str
    canonical_mesh: CanonicalMeshIR | None
    static_qualification: StaticCanonicalMeshQualificationIR | None
    source_raster_receipt_sha256: str
    policy_hash: str
    per_view: tuple[dict[str, Any], ...]
    blockers: tuple[str, ...]
    schema: str = "RealSaS.RSSvNext.TwoObservedViewMeshAdmission.v1"
    source_3d_depth_observed: bool = False
    source_hidden_surfaces_observed: bool = False
    source_rights_verified: bool = False
    learned_geometry_accepted_as_source_truth: bool = False
    appearance_authority: bool = False
    mechanical_attachment_authority: bool = False
    product_revision_minted: bool = False


def qualify_rss_vnext_two_view_mesh_v1(
    *, source: RSSvNextTwoViewSourceObservationV1,
    evidence: RSSvNextInferenceEvidenceIRV1,
    source_identity_court: RSSvNextSourceEvidenceBindingCourtV1,
    diagnostic: RSSvNextTwoViewRasterDiagnosticV1,
    neural_basis: RSSvNextNeuralMeshCandidateBasisV1,
    static_policy: RSSvNextTwoViewStaticPolicyV1,
    mesh_policy: MeshQualificationPolicyIR,
) -> RSSvNextTwoViewMeshAdmissionV1:
    static_policy.validate()
    validate_mesh_qualification_policy(mesh_policy)
    validate_inference_evidence_v1(
        evidence, expected_checkpoint_sha256=evidence.checkpoint_sha256,
    )
    validate_rss_vnext_two_view_raster_diagnostic_v1(diagnostic)
    if (
        type(source) is not RSSvNextTwoViewSourceObservationV1
        or type(neural_basis) is not RSSvNextNeuralMeshCandidateBasisV1
        or evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE"
        or evidence.mesh is None
        or neural_basis.inference_evidence_hash != evidence.evidence_hash
        or neural_basis.original_mesh_sha256 != content_sha256(evidence.mesh)
        or neural_basis.canonical_mesh_authority is not False
        or neural_basis.source_fidelity_authority is not False
        or neural_basis.neural_surface.metadata.get("source_surface_qualified") is not False
        or neural_basis.candidate.metadata.get("source_geometry_was_not_independently_observed") is not True
    ):
        raise QualificationError("RSS_VNEXT_TWO_VIEW_NEURAL_SOURCE_3D_AUTHORITY_DRIFT")
    if (
        diagnostic.source_binding_sha256 != source.source_binding_sha256
        or diagnostic.inference_evidence_hash != evidence.evidence_hash
        or diagnostic.checkpoint_sha256 != evidence.checkpoint_sha256
        or diagnostic.original_mesh_sha256 != neural_basis.original_mesh_sha256
        or diagnostic.state != "MEASURED_TWO_VIEW_SILHOUETTE_NOT_SOURCE_FIDELITY"
        or len(diagnostic.views) != 2
    ):
        raise QualificationError("RSS_VNEXT_TWO_VIEW_RASTER_PARENT_OR_COVERAGE_DRIFT")

    # Recompute independently from the original RGBA/cameras and canonical
    # evidence. A caller-controlled JSON report can be hash-consistent yet
    # wrong; a report hash is not source authority. No invented third view.
    recomputed = compile_rss_vnext_two_view_raster_diagnostic_v1(
        source=source, evidence=evidence,
        source_evidence_binding_court=source_identity_court,
    )
    if diagnostic != recomputed:
        raise QualificationError("RSS_VNEXT_TWO_VIEW_RASTER_REPLAY_MISMATCH")

    candidate = neural_basis.candidate
    addressing = build_surface_addressing(candidate)
    if (
        len(addressing.vertex_addresses) != len(candidate.vertices)
        or len(addressing.face_address_ids) != len(candidate.faces)
    ):
        raise QualificationError("RSS_VNEXT_TWO_VIEW_SURFACE_ADDRESSABILITY_INCOMPLETE")
    quality = relation_parent_quality_report_v1(candidate, mesh_policy)
    policy_hash = content_sha256({
        "raster_policy": asdict(static_policy),
        "mesh_policy_hash": mesh_policy.qualification_policy_lineage_hash,
    })
    results: list[dict[str, Any]] = []
    for i, row in enumerate(diagnostic.views):
        if (
            type(row) is not dict
            or row.get("view_index") != i
            or row.get("view_id") != ("FRONT", "BACK")[i]
            or row.get("original_source_rgba_sha256") != source.view_pixel_sha256[i]
            or not isinstance(row.get("geometry"), dict)
        ):
            raise QualificationError("RSS_VNEXT_TWO_VIEW_RASTER_VIEW_DOMAIN_DRIFT")
        try:
            item = row["geometry"]["rest"]["alpha_comparison"]["alpha_ge_128"]
            metrics = dict(item["metrics"])
        except (KeyError, TypeError, ValueError) as exc:
            raise QualificationError("RSS_VNEXT_TWO_VIEW_RASTER_MEASUREMENT_MISSING") from exc
        if item.get("source_threshold_has_foreground") is not True:
            raise QualificationError("RSS_VNEXT_TWO_VIEW_NO_OBSERVED_CHARACTER")
        acceptance = (
            float(metrics["recall"]) >= static_policy.min_recall
            and float(metrics["precision"]) >= static_policy.min_precision
            and float(metrics["largest_coherent_hole_fraction"])
            <= static_policy.max_largest_coherent_hole_fraction
            and float(metrics["interior_uncovered_fraction"])
            <= static_policy.max_interior_uncovered_fraction
            and float(metrics["minimum_eligible_component_recall"])
            >= static_policy.min_component_recall
            and float(metrics["silhouette_edge_p95_px"])
            <= static_policy.max_silhouette_edge_p95_px
        )
        results.append({
            "view_index": i, "view_id": row["view_id"],
            "original_source_rgba_sha256": source.view_pixel_sha256[i],
            "observed_2d_silhouette_qualified": bool(acceptance),
            "measurement": metrics,
        })
    passed = (
        int(quality["degenerate_face_count"]) == 0
        and int(quality["policy_violating_face_count"]) == 0
        and all(x["observed_2d_silhouette_qualified"] for x in results)
    )
    blocked = (
        () if passed else ("OBSERVED_TWO_VIEW_FIDELITY_OR_MESH_CONDITIONING_FAILED",)
    )
    if not passed:
        return RSSvNextTwoViewMeshAdmissionV1(
            "FAIL", None, None, diagnostic.diagnostic_hash,
            policy_hash, tuple(results), blocked,
        )
    # Reuse the EXISTING CanonicalMeshIR constructor and carrier court.
    # The qualification report is deliberately explicit about what passed:
    # *observed* 2D raster fidelity, not depth uniqueness or hidden geometry.
    source_gate = content_sha256({
        "source_binding_sha256": source.source_binding_sha256,
        "source_observed_front": source.view_pixel_sha256[0],
        "source_observed_back": source.view_pixel_sha256[1],
        "diagnostic_hash": diagnostic.diagnostic_hash,
        "policy_hash": policy_hash,
        "qualification_scope": "TWO_OBSERVED_SILHOUETTES_ONLY",
    })
    report = {
        "status": "PASS_STATIC_CANONICAL_CARRIER",
        "static_quality_policy_passed": True,
        "static_quality_min_angle_deg": quality["min_angle_deg"],
        "static_quality_max_aspect": quality["max_aspect_longest_over_min_altitude"],
        "static_quality_policy_violating_face_count": 0,
        "source_fidelity_qualification_passed": True,
        "source_fidelity_scope": "ONLY_THE_TWO_ORIGINAL_OBSERVED_SOURCE_RASTERS",
        "source_observed_depth_qualified": False,
        "source_hidden_surface_truth_claimed": False,
        "source_geometry_was_not_independently_observed": True,
        "silhouette_is_geometry_authority_not_graphics": True,
        "presentation_domain_required_for_mesh_admission": False,
        "demo_geometry_lineage": False,
        "product_authority_claimed": False,
        "per_view": tuple(results),
    }
    static = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash=addressing.addressing_hash,
        appearance_domain_binding_hash="",
        geometry_gate_binding_hash=source_gate,
        partition_binding_hash=candidate.partition_binding_hash,
        qualification_report=report,
        qualification_hash="",
        metadata={
            "qualification": "TWO_OBSERVED_RASTER_CAMERAS_ONLY",
            "depth_nullspace_not_disambiguated": True,
            "inference_3d_basis_is_model_generated": True,
            "appearance_binding_requires_independent_graphics": True,
            "source_rights_verified": False,
            "policy_hash": policy_hash,
        },
    )
    static = replace(static, qualification_hash=static_mesh_qualification_hash(static))
    carrier = build_mechanical_carrier_evidence_v1(
        candidate, static_qualification=static, surface_addressing=addressing,
    )
    mesh = mint_canonical_mesh_v1(
        candidate, static_qualification=static, carrier_evidence=carrier,
        proposal_evidence_refs=(
            evidence.evidence_hash,
            neural_basis.basis_hash,
            source_gate,
            diagnostic.diagnostic_hash,
        ),
    )
    return RSSvNextTwoViewMeshAdmissionV1(
        "PASS_OBSERVED_TWO_VIEW_STATIC_CARRIER", mesh, static,
        diagnostic.diagnostic_hash, policy_hash, tuple(results), (),
    )


__all__ = [
    "RSSvNextTwoViewStaticPolicyV1", "RSSvNextTwoViewMeshAdmissionV1",
    "qualify_rss_vnext_two_view_mesh_v1",
]
