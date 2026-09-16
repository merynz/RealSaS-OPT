from __future__ import annotations

"""Fail-closed TEST_SUBJECT_001 launcher for the exact sealed D0 FIX1 axis contract.

The underlying materializer remains a generic subject fixture. This launcher is the
Mage TEST_SUBJECT_001 authority boundary: it refuses any axis JSON except the exact
D0 FIX1 artifact and preserves the artifact's explicit distinction between a usable
diagnostic axis/sign contract and an unclaimed D1 semantic truth.
"""

import json
import math
from hashlib import sha256
from pathlib import Path


EXPECTED_D0_AXIS_CONTRACT_SHA256 = "8bfaad13ae4f0c7750756f96dd276246bd09d37c7fb51c0e6c17936721a6c43f"
EXPECTED_D0_AXIS_SCHEMA = "RealSaS.D0MotionAxisContract.v1"
EXPECTED_D0_AXIS_STATUS = "PASS__GEOMETRY_DERIVED_AXIS_CONTRACT_MICROPOSE_QUALIFIED"
EXPECTED_D0_JOINT_AXIS_COUNT = 20


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("TEST_SUBJECT_001_D0_EXPECTED_JSON_OBJECT")
    return value


def validate_d0_fix1_payload_v1(raw: dict) -> dict:
    """Validate semantic boundary and explicit per-joint sign fields.

    This does not promote the D0 artifact to semantic truth. It only proves that the
    exact diagnostic contract is structurally safe to feed to the D1 adapter while
    Runtime-v3 remains runtime_qualified=False.
    """

    if raw.get("schema") != EXPECTED_D0_AXIS_SCHEMA:
        raise RuntimeError("TEST_SUBJECT_001_D0_SCHEMA_DRIFT")
    if raw.get("status") != EXPECTED_D0_AXIS_STATUS:
        raise RuntimeError("TEST_SUBJECT_001_D0_STATUS_DRIFT")
    if raw.get("micro_pose_all_passed") is not True:
        raise RuntimeError("TEST_SUBJECT_001_D0_MICRO_POSE_NOT_PASS")
    if raw.get("axis_semantics_are_explicit_contract_choices") is not True:
        raise RuntimeError("TEST_SUBJECT_001_D0_EXPLICIT_AXIS_SEMANTICS_REQUIRED")
    if raw.get("semantic_truth_claimed") is not False:
        raise RuntimeError("TEST_SUBJECT_001_D0_SEMANTIC_TRUTH_BOUNDARY_DRIFT")
    if raw.get("human_semantic_approval_required_before_D1_semantic_claim") is not True:
        raise RuntimeError("TEST_SUBJECT_001_D0_HUMAN_SEMANTIC_APPROVAL_BOUNDARY_DRIFT")
    if raw.get("product_pass_claimed") is not False:
        raise RuntimeError("TEST_SUBJECT_001_D0_PRODUCT_PASS_BOUNDARY_DRIFT")

    rows = raw.get("joint_axes")
    if not isinstance(rows, list) or len(rows) != EXPECTED_D0_JOINT_AXIS_COUNT:
        raise RuntimeError("TEST_SUBJECT_001_D0_JOINT_AXIS_COUNT_DRIFT")

    seen: set[str] = set()
    sign_counts = {"plus": 0, "minus": 0}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("TEST_SUBJECT_001_D0_AXIS_ROW_INVALID")
        joint_id = str(row.get("canonical_joint_id") or "")
        if not joint_id or joint_id in seen:
            raise RuntimeError("TEST_SUBJECT_001_D0_AXIS_JOINT_DUPLICATE_OR_EMPTY")
        seen.add(joint_id)

        axis = row.get("axis_xyz")
        if not isinstance(axis, list) or len(axis) != 3:
            raise RuntimeError(f"TEST_SUBJECT_001_D0_AXIS_VECTOR_INVALID:{joint_id}")
        try:
            axis_values = tuple(float(x) for x in axis)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"TEST_SUBJECT_001_D0_AXIS_VECTOR_INVALID:{joint_id}") from exc
        if not all(math.isfinite(x) for x in axis_values) or math.sqrt(sum(x * x for x in axis_values)) <= 1.0e-12:
            raise RuntimeError(f"TEST_SUBJECT_001_D0_AXIS_VECTOR_INVALID:{joint_id}")

        # Critical: the adapter historically had a +1 fallback. TEST_SUBJECT_001 is
        # forbidden from reaching that fallback; every sealed row must carry a sign.
        if "legacy_scalar_to_semantic_sign" not in row:
            raise RuntimeError(f"TEST_SUBJECT_001_D0_EXPLICIT_SIGN_REQUIRED:{joint_id}")
        try:
            sign = float(row["legacy_scalar_to_semantic_sign"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"TEST_SUBJECT_001_D0_EXPLICIT_SIGN_INVALID:{joint_id}") from exc
        if not math.isfinite(sign) or sign not in (-1.0, 1.0):
            raise RuntimeError(f"TEST_SUBJECT_001_D0_EXPLICIT_SIGN_INVALID:{joint_id}")
        sign_counts["plus" if sign > 0 else "minus"] += 1

        if not str(row.get("role") or "").strip():
            raise RuntimeError(f"TEST_SUBJECT_001_D0_ROLE_REQUIRED:{joint_id}")
        if not str(row.get("positive_rotation_semantic") or "").strip():
            raise RuntimeError(f"TEST_SUBJECT_001_D0_POSITIVE_ROTATION_SEMANTIC_REQUIRED:{joint_id}")

    if sign_counts != {"plus": 17, "minus": 3}:
        raise RuntimeError(f"TEST_SUBJECT_001_D0_SIGN_DISTRIBUTION_DRIFT:{sign_counts}")

    return {
        "schema": EXPECTED_D0_AXIS_SCHEMA,
        "status": EXPECTED_D0_AXIS_STATUS,
        "joint_axis_count": len(rows),
        "explicit_sign_count": len(rows),
        "plus_sign_count": sign_counts["plus"],
        "minus_sign_count": sign_counts["minus"],
        "semantic_truth_claimed": False,
        "human_semantic_approval_required_before_D1_semantic_claim": True,
        "product_pass_claimed": False,
    }


def validate_sealed_d0_fix1_v1(path: Path) -> dict:
    path = Path(path).resolve()
    digest = _sha(path)
    if digest != EXPECTED_D0_AXIS_CONTRACT_SHA256:
        raise RuntimeError("TEST_SUBJECT_001_D0_AXIS_CONTRACT_SHA_DRIFT")
    report = validate_d0_fix1_payload_v1(_load_json(path))
    return {**report, "axis_contract_sha256": digest}


def materialize(args) -> dict:
    axis_report = validate_sealed_d0_fix1_v1(Path(args.axis_contract))

    # Lazy import keeps the seal validator independently testable on CPU and makes
    # this file an authority wrapper rather than a fork of the product materializer.
    from experiments.playback_stack_v1 import materialize_test_subject_001_v1 as subject

    report = subject.materialize(args)
    if report.get("axis_contract_file_sha256") != EXPECTED_D0_AXIS_CONTRACT_SHA256:
        raise RuntimeError("TEST_SUBJECT_001_D0_REPORT_BINDING_DRIFT")
    if report.get("runtime_qualified") is not False:
        raise RuntimeError("TEST_SUBJECT_001_D0_DIAGNOSTIC_MUST_REMAIN_RUNTIME_UNQUALIFIED")
    return {**report, "d0_fix1_seal": axis_report}


def main() -> None:
    from experiments.playback_stack_v1 import materialize_test_subject_001_v1 as subject

    args = subject._parser().parse_args()
    report = materialize(args)
    print("TEST_SUBJECT_001_D0_FIX1_SEAL_PASS")
    print(json.dumps({
        "archive_path": report["package"]["archive_path"],
        "archive_sha256": report["package"]["archive_sha256"],
        "axis_contract_sha256": report["d0_fix1_seal"]["axis_contract_sha256"],
        "runtime_qualified": report["runtime_qualified"],
        "founder_visual_pass_claimed": report["founder_visual_pass_claimed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
