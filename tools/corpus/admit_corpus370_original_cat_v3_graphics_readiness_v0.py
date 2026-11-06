#!/usr/bin/env python3
"""Cat V3 PRIVATE original-source court -> generic Graphics readiness bridge.

This adapter knows an immutable source experiment, not the synthesis algorithm.
No inference inputs, artist ownership, original pixels or training targets leave
the private runner. Replays the independent original court on sealed bytes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.graphics_source_readiness_v0 import (
    GraphicsSourceClaimV0, GraphicsAdmittedArtifactV0,
    SOURCE_AUTHORITY, audit_graphics_source_readiness_v0,
)
from tools.corpus.court_drawable_capability_pose_transport_v3 import run as independent_court

SOURCE_SHA = "1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
RENDER_SHA = "b2bb4fa9fc0b777bb8418aebc729123616b774b55eeb215ba17651e7f3448e0d"
RENDER_MANIFEST_SHA = "89dcb4d516e5579e766b19bd2d25eb31cbfaaa64f27ce34632405d99ce6874bc"
INDEPENDENT_COURT_SHA = "0171e9ec836d7b3c44ec64abcc59f712ebd33a87c2ad4edf6188976dc08e272c"
REQUIRED_SOURCE_FACTS = (
    "AUTHORED_NATIVE_SOURCE_POSE",
    "SOURCE_UV_EDIT_CONSEQUENCE_ACROSS_POSES",
)
OPERATIONS = (
    "POSE_AWARE_SOURCE_UV", "INDEPENDENT_ATLAS_EDITS",
    "LOCAL_DEPTH_SOURCE_OVER", "HIDDEN_EDIT_STATE_PERSISTENCE",
)
NO_AUTHORITY = ("graphics_owner_authorized", "iris_training_authorized",
                "tessa_training_authorized", "materialization_authorized",
                "training_authorized")


def file_sha(path: Path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(value, name):
    if not value:
        raise ValueError("CAT_GRAPHICS_READINESS_V0_" + name)


def build_from_recomputed_court_v0(court: dict, *, court_sha: str) -> dict:
    """Source-specific mapping. Does NOT trust a caller without byte replay.

    This pure helper is separately testable, but only main() may use it for an
    original-source claim after replay + digest checks on exact source inputs.
    """
    require(court_sha == INDEPENDENT_COURT_SHA, "ORIGINAL_COURT_HASH_DRIFT")
    require(court.get("status") ==
        "PASS_ORIGINAL_CAT_POSE_UV_EDIT_PHYSICAL_SOURCE__FIXED_SCREEN_CLASS_VIOLATED",
        "PHYSICAL_COURT_NOT_PASS")
    require(court.get("schema") ==
        "RealSaS.IndependentCatPoseUVTransportSourceCourt.v3",
        "INDEPENDENT_COURT_SCHEMA_DRIFT")
    require(court.get("source_original_blender_physical_sample_sha256") == RENDER_SHA
        and court.get("source_original_blender_physical_render_manifest_sha256") ==
            RENDER_MANIFEST_SHA
        and court.get("actual_source_original_full_physical_renders") == 64
        and court.get("source_original_pose_scope") == "AUTHORED_TEACHER_ORDINAL_0_AND_14_ONLY"
        and court.get("source_original_edited_atlas_operation_scope") ==
            "THREE_FIXED_ANONYMOUS_UV_ADDRESSES"
        and court.get("source_uv_surface_attached_edit_native_source_status") ==
            "SATISFIED_PHYSICAL_ORIGINAL_SOURCE_WITNESS_ONLY",
        "PHYSICAL_SOURCE_SCOPE_OR_LINEAGE_DRIFT")
    require(court.get("source_missing_evidence_obligations") == []
        and court.get("source_original_pose_world_max_displacement", 0) > 0
        and court.get("source_edit_observed_pose0_by_operation") == [59303, 57005, 1207]
        and court.get("source_edit_observed_pose14_by_operation") == [57800, 57952, 1207]
        and court.get("measured_original_physical_source", {}).get(
            "fixed_screen_pose14_total_mismatched_sample_incidents") == 40878
        and all(court.get(f) is False for f in NO_AUTHORITY)
        and court.get("original_drawable_artist_partition") == "UNKNOWN_NOT_MEASURED",
        "SOURCE_CONSEQUENCE_OR_AUTHORITY_DRIFT")
    # An exact V3 source experiment defines one operation scope, not all
    # future views, poses, alpha policies, edit demands or student subjects.
    scope_hash = content_sha256({
        "source_sha256": SOURCE_SHA,
        "sample_sha256": RENDER_SHA,
        "independent_court_sha256": INDEPENDENT_COURT_SHA,
        "pose_scope": court["source_original_pose_scope"],
        "edit_scope": court["source_original_edited_atlas_operation_scope"],
        "shader": court["source_original_shader_domain"],
        "camera_and_samples": "V0,V2,V4,V6__OFFSETS_0_16",
    })
    rows = tuple(GraphicsSourceClaimV0(
        fact=fact, source_graphics_binding_hash=SOURCE_SHA,
        scope_hash=scope_hash,
        record_hash=content_sha256({
            "fact": fact,
            "scope_hash": scope_hash,
            "source_physical_court_sha256": INDEPENDENT_COURT_SHA,
            "source_samples_sha256": RENDER_SHA,
        }),
        evidence_artifact_hash=RENDER_SHA,
        independent_court_hash=INDEPENDENT_COURT_SHA,
        origin=SOURCE_AUTHORITY, observation="OBSERVED",
    ) for fact in REQUIRED_SOURCE_FACTS)
    verified = GraphicsAdmittedArtifactV0(
        source_graphics_binding_hash=SOURCE_SHA, scope_hash=scope_hash,
        independent_court_hash=INDEPENDENT_COURT_SHA,
        admitted_record_hashes=frozenset(row.record_hash for row in rows),
    )
    report = audit_graphics_source_readiness_v0(
        source_graphics_binding_hash=SOURCE_SHA,
        required_operations=OPERATIONS,
        claims=rows, admitted_artifacts={RENDER_SHA: verified},
    )
    require(report.per_operation_source_observability["POSE_AWARE_SOURCE_UV"] ==
            "OBSERVED_IN_ADMITTED_SOURCE_SCOPE"
        and all(report.per_operation_source_observability[op] == "UNKNOWN"
                for op in OPERATIONS if op != "POSE_AWARE_SOURCE_UV"
        ), "OUT_OF_SCOPE_OPERATION_PROMOTED")
    return {
        "schema": "RealSaS.OriginalCatV3SourceGraphicsReadiness.v0",
        "source_scope": "CAT_NATIVE_ORIGINAL_2POSE_4VIEW_SOURCE_EDIT_ONLY",
        "independently_recomputed_source_court_sha256": court_sha,
        "physical_source_sample_sha256": RENDER_SHA,
        "graphics_readiness": report.to_dict(),
        "source_original_artist_ownership": "UNKNOWN",
        "compiled_puppet_physical_parity": "UNKNOWN",
        "student_training_authorized": False,
        "model_inference_input_admitted": False,
    }


def requalify(root: Path, out: Path):
    root = Path(root)
    sample = root / "render/PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_SAMPLES_V3.npz"
    manifest = root / "render/PRIVATE_CAT_NATIVE_POSE_UV_EDIT_CYCLES_V3.json"
    require(file_sha(sample) == RENDER_SHA
            and file_sha(manifest) == RENDER_MANIFEST_SHA,
            "PHYSICAL_SOURCE_BYTES_SHA_DRIFT")
    recomputed_path = root / "out/PRIVATE_INDEPENDENT_CAT_POSE_UV_TRANSPORT_V3.json"
    independent_court(root, recomputed_path)
    require(file_sha(recomputed_path) == INDEPENDENT_COURT_SHA,
            "RECOMPUTED_INDEPENDENT_COURT_SHA_DRIFT")
    court = json.loads(recomputed_path.read_text())
    result = build_from_recomputed_court_v0(court, court_sha=file_sha(recomputed_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "ORIGINAL_CAT_V3_SOURCE_EVIDENCE_READINESS_REQUALIFIED",
        "scope": result["source_scope"],
        "observability": result["graphics_readiness"]["per_operation_source_observability"],
        "unresolved": result["graphics_readiness"]["missing_obligations"],
        "source_puppet_parity": "UNKNOWN",
        "owner_truth": "UNKNOWN",
        "training_authorized": False,
    }, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    requalify(args.root, args.out)


if __name__ == "__main__":
    main()
