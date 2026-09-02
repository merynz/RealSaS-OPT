from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "RealSaS.ArchitectureFreezeGate.v1"

# Only source modules that define current generic learner architecture/training semantics.
# Historical fit scripts, canonical reports and parity fixtures are deliberately outside
# this source-manifest boundary; they may name witnesses as provenance but cannot define
# current architecture.
CURRENT_GENERIC_SOURCE_FILES = (
    "experiments/iris_reprojection_v2_20260831/q_domain_v2.py",
    "experiments/iris_reprojection_v2_20260831/foundation_adapter_v2.py",
    "experiments/iris_reprojection_v2_20260831/dinov2_foundation_v2.py",
    "experiments/iris_reprojection_v2_20260831/q_descriptor_sampler_v2.py",
    "experiments/iris_reprojection_v2_20260831/evidence_field_v2.py",
    "experiments/iris_reprojection_v2_20260831/model_v2.py",
    "experiments/iris_reprojection_v2_20260831/loss_v2.py",
    "experiments/iris_reprojection_v2_20260831/train_v2.py",
    "experiments/iris_reprojection_v2_20260831/eval_v2.py",
    "experiments/geppetto_arachne_r6_20260901/geppetto_conditioning_v2.py",
    "experiments/geppetto_arachne_r6_20260901/geppetto_candidate_v2.py",
    "experiments/geppetto_arachne_r6_20260901/geppetto_loss_v2.py",
    "experiments/geppetto_arachne_r6_20260901/geppetto_train_v2.py",
    "experiments/geppetto_arachne_r6_20260901/geppetto_eval_v2.py",
    "experiments/geppetto_arachne_r6_20260901/skinfield_codec_v2.py",
    "experiments/geppetto_arachne_r6_20260901/arachne_v2.py",
    "experiments/geppetto_arachne_r6_20260901/arachne_loss_v2.py",
    "experiments/geppetto_arachne_r6_20260901/arachne_train_v2.py",
    "experiments/geppetto_arachne_r6_20260901/arachne_eval_v2.py",
)

# Family/witness literals that have appeared in historical fitting work. They are legal
# in reports/fixtures but forbidden in current generic architecture source.
FORBIDDEN_FAMILY_LITERALS = (
    "mage",
    "asset_96b983142e9fcd29ecf52f48",
    "asset_0004e640bea23f87618cad9e",
    "cf898585da33fab50c724d31",
)

REQUIRED_PRE_FREEZE_ARTIFACTS = (
    "canonical/DINO_TOKEN_PARITY_V1_SEAL_20260902.json",
    "canonical/GENERIC_SOURCE_COMPLETION_CLOSURE_20260902.md",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_manifest(repo_root: Path) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    violations: list[str] = []
    for rel in CURRENT_GENERIC_SOURCE_FILES:
        p = repo_root / rel
        if not p.is_file():
            violations.append(f"MISSING_SOURCE:{rel}")
            continue
        raw = p.read_text(encoding="utf-8")
        low = raw.lower()
        for literal in FORBIDDEN_FAMILY_LITERALS:
            if literal.lower() in low:
                violations.append(f"FAMILY_LITERAL:{literal}:{rel}")
        hashes[rel] = sha256_file(p)
    return hashes, violations


def verify_prerequisites(repo_root: Path) -> list[str]:
    violations: list[str] = []
    for rel in REQUIRED_PRE_FREEZE_ARTIFACTS:
        p = repo_root / rel
        if not p.is_file():
            violations.append(f"MISSING_PREREQUISITE:{rel}")
    parity = repo_root / "canonical/DINO_TOKEN_PARITY_V1_SEAL_20260902.json"
    if parity.is_file():
        d = json.loads(parity.read_text(encoding="utf-8"))
        if d.get("status") != "PASS_RECOVERED_HISTORICAL_TOKEN_PARITY_AUTHORITY":
            violations.append("DINO_PARITY_NOT_PASS")
        if d.get("family_selection_authorized") is not False:
            violations.append("PARITY_SEAL_ILLEGALLY_AUTHORIZES_FAMILY_SELECTION")
    return violations


def build_freeze_candidate(repo_root: Path) -> dict:
    hashes, source_violations = source_manifest(repo_root)
    violations = source_violations + verify_prerequisites(repo_root)
    ordered = [{"path": p, "sha256": hashes[p]} for p in sorted(hashes)]
    fingerprint = hashlib.sha256(
        json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "status": "PASS_SOURCE_ELIGIBLE_FOR_FREEZE" if not violations else "FAIL_SOURCE_NOT_FREEZABLE",
        "family_selection_authorized": False,
        "generic_source_count": len(ordered),
        "generic_source_fingerprint_sha256": fingerprint,
        "generic_source_manifest": ordered,
        "violations": violations,
        "rule": "ARCHITECTURE_FIRST__FAMILY_SELECTION_ONLY_AFTER_EXPLICIT_FREEZE_SEAL",
    }


def require_family_selection_authority(repo_root: Path, seal_path: str = "canonical/ARCHITECTURE_FREEZE_V1.json") -> dict:
    """Hard gate that every future family selector must call before reading candidates."""
    p = repo_root / seal_path
    if not p.is_file():
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__ARCHITECTURE_FREEZE_SEAL_MISSING")
    seal = json.loads(p.read_text(encoding="utf-8"))
    if seal.get("schema") != "RealSaS.ArchitectureFreezeSeal.v1":
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__FREEZE_SCHEMA_DRIFT")
    if seal.get("status") != "PASS_ARCHITECTURE_FROZEN" or seal.get("family_selection_authorized") is not True:
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__ARCHITECTURE_NOT_FROZEN")
    current = build_freeze_candidate(repo_root)
    if current["status"] != "PASS_SOURCE_ELIGIBLE_FOR_FREEZE":
        raise RuntimeError(f"FAMILY_SELECTION_BLOCKED__SOURCE_GATE_FAIL:{current['violations']}")
    expected = seal.get("generic_source_fingerprint_sha256")
    if current["generic_source_fingerprint_sha256"] != expected:
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__SOURCE_CHANGED_AFTER_FREEZE")
    return seal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check-selection-authority", action="store_true")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    if args.check_selection_authority:
        seal = require_family_selection_authority(root)
        print(json.dumps({"status": "PASS_FAMILY_SELECTION_AUTHORIZED", "seal": seal}, indent=2, sort_keys=True))
        return
    result = build_freeze_candidate(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS_SOURCE_ELIGIBLE_FOR_FREEZE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
