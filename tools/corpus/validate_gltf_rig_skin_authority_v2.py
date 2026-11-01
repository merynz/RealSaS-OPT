#!/usr/bin/env python3
from __future__ import annotations

"""Independent court for the V2 glTF rig/skin source authority candidate.

The independent implementation remains SourceGLTF from the V1 court.  V2 changes
only the producer-side rotation semantic: source quaternions are normalized before
TRS matrix construction.  We deliberately reuse the independent V1 reconstruction
rather than the V2 producer helper so the court stays implementation-independent.
"""

import argparse
import json
from pathlib import Path

from tools.corpus import validate_gltf_rig_skin_authority_v1 as v1


PARENT_SCHEMA = "RealSaS.GltfRigSkinAuthority.v2"
PARENT_STATUS = "PASS_GLTF_RIG_SKIN_SOURCE_AUTHORITY_V2_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA = "RealSaS.GltfRigSkinAuthorityCourt.v2"
STATUS = "PASS_GLTF_RIG_SKIN_SOURCE_AUTHORITY_V2_INDEPENDENT_COURT__IMPORT_EQUIVALENCE_REQUIRED"


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("GLTF_RIG_SKIN_V2_COURT_PARENT")
    policy = dict(manifest.get("policy") or {})
    if policy.get("rotation_quaternion_policy") != "FINITE_NONZERO_SOURCE_QUATERNION_NORMALIZED_TO_UNIT_BEFORE_MATRIX_CONSTRUCTION":
        raise ValueError("GLTF_RIG_SKIN_V2_COURT_ROTATION_POLICY")
    rotation = dict(manifest.get("rotation_semantics") or {})
    if rotation.get("normalization_applied_before_matrix_construction") is not True:
        raise ValueError("GLTF_RIG_SKIN_V2_COURT_ROTATION_SEMANTICS")

    # Translate only the declared schema/status so the already-independent V1
    # court implementation can reconstruct source semantics with SourceGLTF.
    surrogate = dict(manifest)
    surrogate["schema"] = v1.PARENT_SCHEMA
    surrogate["status"] = v1.PARENT_STATUS
    payload = v1.validate(surrogate)
    payload["schema"] = SCHEMA
    payload["status"] = STATUS
    payload["independent_method"] = {
        **dict(payload.get("independent_method") or {}),
        "producer_v2_rotation_helper_reused": False,
        "rotation_semantics": "SOURCE_GLTF_QUATERNION_INTERPRETED_AS_UNIT_ROTATION_BY_INDEPENDENT_SOURCE_READER",
        "v1_failure_class_repaired": "GLTF_RIG_SKIN_COURT_ARRAY_MISMATCH:joint_local_matrix",
    }
    payload["authority"] = {
        **dict(payload.get("authority") or {}),
        "v1_artifacts_mutated": False,
        "source_rig_skin_court_passed": True,
        "importer_equivalence_required": True,
        "family_authority_minted": False,
        "training_authorized": False,
    }
    payload.pop("court_identity_sha256", None)
    payload["court_identity_sha256"] = v1.digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text())
    payload = validate(manifest)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "court_identity_sha256": payload["court_identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
