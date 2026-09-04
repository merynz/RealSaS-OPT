from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


SCHEMA = "RealSaS.ArchitectureFreezeGate.v2"
SEAL_SCHEMA = "RealSaS.ArchitectureFreezeSeal.v2"
SOURCE_SCOPE_ID = "LIVING_MODELS_COMPILER_RUNTIME_AND_EXECUTION_CLOSURE_V2"
SELECTION_SCOPE_ID = "TEXTURED_PREFIT_TRUTH_AND_BLINDED_FIT8_SELECTION_V2"
DEFAULT_SEAL_PATH = "canonical/ARCHITECTURE_FREEZE_V2.json"

LIVING_PYTHON_ROOTS = (
    "models",
    "compiler/realsas_compiler_core",
    "compiler/realsas_compiler_services",
)
RUNTIME_ROOTS = (
    "runtime/reference_v4",
    "runtime/realsas_cpp",
)
RUNTIME_SOURCE_SUFFIXES = frozenset({".py", ".c", ".cc", ".cpp", ".h", ".hpp"})
RUNTIME_SOURCE_NAMES = frozenset({"CMakeLists.txt"})
VISIBLE_EXECUTION_SOURCE = "compiler/vendor/realsas_synthesis/canonical_graph_optimizer.py"
VENDOR_EXECUTION_CLOSURE = "compiler/vendor/realsas_v05_current_execution_closure.b64"
SELECTION_APPARATUS_FILES = (
    "experiments/family_selection_v1/prefit_observation_authority_v1.py",
    "experiments/family_selection_v1/prefit_family_truth_eligibility_v1.py",
    "experiments/family_selection_v1/post_freeze_family_selector_v1.py",
    "experiments/single_family_e2e_v1/data_manifest_v1.py",
)
REQUIRED_PRE_FREEZE_ARTIFACTS = (
    "canonical/DINO_TOKEN_PARITY_V1_SEAL_20260902.json",
    "canonical/LEARNED_ARTIFACT_REGISTRY_V1_20260904.json",
    "canonical/MWB2_DIRECTIONAL_BEHAVIORAL_CLOSURE_20260903.md",
)
FORBIDDEN_FAMILY_NAMES = ("mage",)
FORBIDDEN_EXACT_IDENTIFIERS = (
    "asset_96b983142e9fcd29ecf52f48",
    "asset_0004e640bea23f87618cad9e",
    "cf898585da33fab50c724d31",
)
EXPECTED_VISIBLE_OPTIMIZER_SHA256 = "b2fddb64753ca783e298be4f1078c70b9667fa9931c67de738f955976e2587c1"
EXPECTED_VISIBLE_OPTIMIZER_BYTES = 53544
EXPECTED_VENDOR_RAW_SHA256 = "3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _family_literal_violations(raw: str, rel: str) -> list[str]:
    low = raw.lower()
    out: list[str] = []
    for identifier in FORBIDDEN_EXACT_IDENTIFIERS:
        if identifier.lower() in low:
            out.append(f"FAMILY_IDENTIFIER:{identifier}:{rel}")
    for name in FORBIDDEN_FAMILY_NAMES:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", low):
            out.append(f"FAMILY_NAME:{name}:{rel}")
    return out


def _walk_python(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file() and "__pycache__" not in p.parts)


def _walk_runtime(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and (p.suffix.lower() in RUNTIME_SOURCE_SUFFIXES or p.name in RUNTIME_SOURCE_NAMES)
    )


def _walk_vendor_closure(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def _manifest_for_paths(repo_root: Path, paths: list[Path], *, inspect_family_literals: bool) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    violations: list[str] = []
    seen: set[str] = set()
    for path in sorted(paths):
        rel = path.relative_to(repo_root).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        if not path.is_file():
            violations.append(f"MISSING_SOURCE:{rel}")
            continue
        if inspect_family_literals:
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                violations.append(f"NON_UTF8_LIVING_SOURCE:{rel}")
            else:
                violations.extend(_family_literal_violations(raw, rel))
        rows.append({"path": rel, "sha256": sha256_file(path)})
    return rows, violations


def _fingerprint(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def living_source_manifest(repo_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    source_paths: list[Path] = []
    violations: list[str] = []
    for rel in LIVING_PYTHON_ROOTS:
        root = repo_root / rel
        if not root.is_dir():
            violations.append(f"MISSING_LIVING_ROOT:{rel}")
        source_paths.extend(_walk_python(root))
    for rel in RUNTIME_ROOTS:
        root = repo_root / rel
        if not root.is_dir():
            violations.append(f"MISSING_RUNTIME_ROOT:{rel}")
        source_paths.extend(_walk_runtime(root))
    visible = repo_root / VISIBLE_EXECUTION_SOURCE
    if not visible.is_file():
        violations.append(f"MISSING_VISIBLE_EXECUTION_SOURCE:{VISIBLE_EXECUTION_SOURCE}")
    else:
        source_paths.append(visible)
        raw = visible.read_bytes()
        if len(raw) != EXPECTED_VISIBLE_OPTIMIZER_BYTES:
            violations.append(f"VISIBLE_OPTIMIZER_SIZE_DRIFT:{len(raw)}")
        got = hashlib.sha256(raw).hexdigest()
        if got != EXPECTED_VISIBLE_OPTIMIZER_SHA256:
            violations.append(f"VISIBLE_OPTIMIZER_SHA_DRIFT:{got}")
    vendor_root = repo_root / VENDOR_EXECUTION_CLOSURE
    vendor_paths = _walk_vendor_closure(vendor_root)
    if not vendor_paths:
        violations.append(f"MISSING_VENDOR_EXECUTION_CLOSURE:{VENDOR_EXECUTION_CLOSURE}")
    rows, source_violations = _manifest_for_paths(repo_root, source_paths, inspect_family_literals=True)
    vendor_rows, _ = _manifest_for_paths(repo_root, vendor_paths, inspect_family_literals=False)
    rows.extend(vendor_rows)
    rows = sorted(rows, key=lambda row: row["path"])
    violations.extend(source_violations)
    manifest = vendor_root / "manifest.json"
    if manifest.is_file():
        obj = json.loads(manifest.read_text(encoding="utf-8"))
        if obj.get("scope") != "CURRENT_IRIS_TO_COMPILER_EXECUTION_CLOSURE":
            violations.append("VENDOR_EXECUTION_SCOPE_DRIFT")
        if obj.get("raw_sha256") != EXPECTED_VENDOR_RAW_SHA256:
            violations.append(f"VENDOR_EXECUTION_RAW_SHA_DRIFT:{obj.get('raw_sha256')}")
        records = {str(row.get("path")): row for row in obj.get("records", [])}
        optimizer = records.get("realsas_synthesis/canonical_graph_optimizer.py")
        if not isinstance(optimizer, dict) or optimizer.get("sha256") != EXPECTED_VISIBLE_OPTIMIZER_SHA256:
            violations.append("VISIBLE_OPTIMIZER_NOT_BOUND_TO_VENDOR_MANIFEST")
    else:
        violations.append("VENDOR_EXECUTION_MANIFEST_MISSING")
    return rows, violations


def selection_apparatus_manifest(repo_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    paths: list[Path] = []
    violations: list[str] = []
    for rel in SELECTION_APPARATUS_FILES:
        path = repo_root / rel
        if not path.is_file():
            violations.append(f"MISSING_SELECTION_APPARATUS:{rel}")
        paths.append(path)
    rows, source_violations = _manifest_for_paths(repo_root, paths, inspect_family_literals=False)
    violations.extend(source_violations)
    return rows, violations


def verify_prerequisites(repo_root: Path) -> list[str]:
    violations: list[str] = []
    for rel in REQUIRED_PRE_FREEZE_ARTIFACTS:
        if not (repo_root / rel).is_file():
            violations.append(f"MISSING_PREREQUISITE:{rel}")
    parity = repo_root / "canonical/DINO_TOKEN_PARITY_V1_SEAL_20260902.json"
    if parity.is_file():
        obj = json.loads(parity.read_text(encoding="utf-8"))
        if obj.get("status") != "PASS_RECOVERED_HISTORICAL_TOKEN_PARITY_AUTHORITY":
            violations.append("DINO_PARITY_NOT_PASS")
        if obj.get("family_selection_authorized") is not False:
            violations.append("PARITY_SEAL_ILLEGALLY_AUTHORIZES_FAMILY_SELECTION")
    registry = repo_root / "canonical/LEARNED_ARTIFACT_REGISTRY_V1_20260904.json"
    if registry.is_file():
        obj = json.loads(registry.read_text(encoding="utf-8"))
        if obj.get("schema") != "RealSaS.LearnedArtifactRegistry.v1":
            violations.append("LEARNED_REGISTRY_SCHEMA_DRIFT")
        if obj.get("fit_authorized_now") is not False:
            violations.append("LEARNED_REGISTRY_FIT_ALREADY_AUTHORIZED")
        stages = obj.get("current_stages", {})
        if not isinstance(stages, dict) or not stages:
            violations.append("LEARNED_REGISTRY_CURRENT_STAGES_MISSING")
        else:
            for name, stage in stages.items():
                if stage.get("state") != "UNTRAINED" or stage.get("checkpoint") is not None:
                    violations.append(f"LEARNED_STAGE_NOT_PREFIT_EMPTY:{name}")
    mwb = repo_root / "canonical/MWB2_DIRECTIONAL_BEHAVIORAL_CLOSURE_20260903.md"
    if mwb.is_file() and "PASS_MWB2_DIRECTIONAL_BEHAVIORAL_CLOSED" not in mwb.read_text(encoding="utf-8"):
        violations.append("MWB2_DIRECTIONAL_CLOSURE_NOT_PASS")
    return violations


def build_freeze_candidate(repo_root: Path) -> dict:
    living, living_violations = living_source_manifest(repo_root)
    selection, selection_violations = selection_apparatus_manifest(repo_root)
    violations = living_violations + selection_violations + verify_prerequisites(repo_root)
    return {
        "schema": SCHEMA,
        "status": "PASS_SOURCE_ELIGIBLE_FOR_FREEZE" if not violations else "FAIL_SOURCE_NOT_FREEZABLE",
        "family_selection_authorized": False,
        "living_source_scope_id": SOURCE_SCOPE_ID,
        "living_source_count": len(living),
        "living_source_fingerprint_sha256": _fingerprint(living),
        "living_source_manifest": living,
        "selection_apparatus_scope_id": SELECTION_SCOPE_ID,
        "selection_apparatus_count": len(selection),
        "selection_apparatus_fingerprint_sha256": _fingerprint(selection),
        "selection_apparatus_manifest": selection,
        "vendor_execution_raw_sha256": EXPECTED_VENDOR_RAW_SHA256,
        "visible_optimizer_sha256": EXPECTED_VISIBLE_OPTIMIZER_SHA256,
        "violations": violations,
        "rule": "LIVING_ARCHITECTURE_AND_SELECTION_APPARATUS_FIRST__FAMILY_SELECTION_ONLY_AFTER_EXPLICIT_V2_SEAL",
    }


def require_family_selection_authority(repo_root: Path, seal_path: str = DEFAULT_SEAL_PATH) -> dict:
    path = repo_root / seal_path
    if not path.is_file():
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__ARCHITECTURE_FREEZE_V2_SEAL_MISSING")
    seal = json.loads(path.read_text(encoding="utf-8"))
    if seal.get("schema") != SEAL_SCHEMA:
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__FREEZE_V2_SCHEMA_DRIFT")
    if seal.get("status") != "PASS_ARCHITECTURE_FROZEN" or seal.get("family_selection_authorized") is not True:
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__ARCHITECTURE_NOT_FROZEN")
    if int(seal.get("scientific_fit_steps_at_freeze", -1)) != 0:
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__FREEZE_WAS_NOT_PREFIT")
    current = build_freeze_candidate(repo_root)
    if current["status"] != "PASS_SOURCE_ELIGIBLE_FOR_FREEZE":
        raise RuntimeError(f"FAMILY_SELECTION_BLOCKED__SOURCE_GATE_FAIL:{current['violations']}")
    if current["living_source_fingerprint_sha256"] != seal.get("living_source_fingerprint_sha256"):
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__LIVING_SOURCE_CHANGED_AFTER_FREEZE")
    if current["selection_apparatus_fingerprint_sha256"] != seal.get("selection_apparatus_fingerprint_sha256"):
        raise RuntimeError("FAMILY_SELECTION_BLOCKED__SELECTION_APPARATUS_CHANGED_AFTER_FREEZE")
    return seal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--check-selection-authority", action="store_true")
    parser.add_argument("--seal-path", default=DEFAULT_SEAL_PATH)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    if args.check_selection_authority:
        seal = require_family_selection_authority(root, args.seal_path)
        print(json.dumps({"status": "PASS_FAMILY_SELECTION_AUTHORIZED", "seal": seal}, indent=2, sort_keys=True))
        return
    result = build_freeze_candidate(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS_SOURCE_ELIGIBLE_FOR_FREEZE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
