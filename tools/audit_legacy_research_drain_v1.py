#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from typing import Iterable

SCHEMA = "RealSaS.LegacyResearchBranchDrainPathInventory.v1"

A = "A_REUSABLE_CURRENT_IMPLEMENTATION"
B = "B_CURRENT_EXPERIMENT_EXECUTION_ONLY"
C = "C_HISTORICAL_SCIENTIFIC_PROVENANCE"
D = "D_SUPERSEDED_IMPLEMENTATION"
E = "E_BRANCH_OWNED_OPERATIONAL_STATE"
F = "F_OBSOLETE_BRANCH_WORKFLOW"
G = "G_PRODUCT_RELEVANT_CANDIDATE"
H = "H_UNRESOLVED_DEPENDENCY"

ALLOWED = {A, B, C, D, E, F, G, H}

EXACT: dict[str, tuple[str, str]] = {
    "AGENTS.md": (E, "legacy branch continuation instructions are not operational authority"),
    "CURRENT_PLAN.md": (E, "legacy branch continuation plan is replaced by Platform state"),
    "CURRENT_STATE.md": (E, "legacy branch current-state file is replaced by Platform state"),
    "SYSTEM_INDEX.md": (E, "branch-local system index must not overwrite canonical main wholesale"),
    "models/iris/topology/head_v1.py": (H, "existing main file changed off-main; requires semantic diff review"),
    "compiler/realsas_compiler_core/graphics_v1.py": (A, "current CompiledGraphicsV1 contract is consumed by runtime Graphics V1"),
    "compiler/realsas_compiler_core/graphics_runtime_v1.py": (A, "materializes behaviorally consumable RuntimeGraphicsV1"),
    "compiler/realsas_compiler_core/runtime_graphics_binary_v1.py": (A, "serializes RuntimeGraphicsV1 into mandatory graphics.bin runtime content"),
    "compiler/realsas_compiler_core/graphics_v2.py": (G, "new evidence/compiler contract lacks proven V2 native runtime consumption"),
    "compiler/realsas_compiler_core/mechanical_candidate_v1.py": (G, "compiler-facing candidate source requires current product-consumer review"),
    "runtime/realsas_cpp/src/graphics_v1.hpp": (A, "native Graphics V1 runtime contract"),
    "runtime/realsas_cpp/src/runtime_v2_graphics_reference.cpp": (A, "native RuntimePackage V2 Graphics V1 behavioral consumer"),
    "runtime/realsas_cpp/tests/graphics_v1_contract_test.cpp": (A, "protects native Graphics V1 runtime contract"),
    "tests/compiler/test_graphics_v2.py": (G, "protects Graphics V2 compiler semantics but not native runtime consumption"),
    "tests/compiler/test_vnext_graphics_native_e2e.py": (A, "behavioral proof that Graphics V1 changes native raster output"),
    "tests/compiler/test_vnext_single_carrier_graphics_v1.py": (A, "protects current single-M Graphics V1 path"),
    "models/iris/vnext/contracts_v1.py": (A, "active IRIS vNext configuration dependency of structural bridge V3/V4"),
    "models/iris/vnext/presentation_evidence_v1.py": (A, "active role-free presentation evidence dependency of bridge V4"),
    "models/iris/vnext/structure_v3.py": (A, "active scene/surface bridge reused by bridge V4"),
    "models/iris/vnext/structure_v4.py": (A, "active IRIS-to-TESSA V3 presentation-conditioned bridge"),
    "models/tessa/v2/config_v2.py": (A, "active decoder configuration dependency reused by TESSA V3"),
    "models/tessa/v2/model_v2.py": (A, "active decoder and typed-memory implementation reused by TESSA V3"),
    "models/tessa/v2/objective_v2.py": (A, "active structural objective reused by TESSA V3 and current evaluator"),
    "models/tessa/v2/generation_v2.py": (A, "active fail-closed grammar/generation primitive used by current evaluator"),
    "models/tessa/v2/windowed_incremental_generation_v2.py": (A, "active exact incremental decode/parity primitive used by current evaluator"),
    "models/tessa/v3/__init__.py": (A, "active TESSA V3 package surface"),
    "models/tessa/v3/config_v3.py": (A, "active matched-capacity tri-stream configuration"),
    "models/tessa/v3/model_v3.py": (A, "active TESSA V3 structuralizer built on V2 decoder/objective stack"),
    "tests/iris/test_presentation_evidence_v1.py": (A, "protects active role-free presentation evidence"),
    "tests/iris/test_structure_bridge_v3.py": (A, "protects active scene/surface bridge"),
    "tests/iris/test_structure_bridge_v4.py": (A, "protects active presentation-conditioned bridge"),
    "tests/iris/test_tessa_generation_v2.py": (A, "protects active generation grammar"),
    "tests/iris/test_tessa_incremental_generation_v2.py": (A, "protects active incremental decode semantics"),
    "tests/iris/test_tessa_structuralizer_v2.py": (A, "protects active V2 decoder reused by V3"),
    "tests/iris/test_tessa_structuralizer_v3.py": (A, "protects active V3 structuralizer"),
    "tools/notebooks/build_iris_vnext_fit8_a100_preflight_v1.py": (
        D,
        "pre-training A100 preflight-only notebook handoff from 2026-10-16; current A/B is post-training with frozen checkpoints and later execution contracts",
    ),
    "tools/verify_native_runtime_source_seal_v2.py": (
        A,
        "generic append-only native runtime source-lineage verifier; validates runtime/realsas_cpp seal closure without minting authority",
    ),
}


def _run(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _is_ab_execution(path: str) -> bool:
    name = Path(path).name.lower()
    return (
        "structural_evidence_ab" in name
        or "structural-evidence-ab" in name
        or "tessa_v3_structural_evidence_ab" in name
    )


def classify(path: str) -> tuple[str, str]:
    if path in EXACT:
        return EXACT[path]

    if path.startswith(".github/workflows/"):
        return F, "legacy branch workflow; rebuild only a generic main-based gate when still required"

    if path.startswith("docs/research/"):
        return C, "dated research plan/result/decision material is provenance, not continuation authority"

    if path.startswith("canonical/"):
        name = Path(path).name
        if name == "COLAB_RESEARCH_WORKER_UX_V1.json":
            return E, "legacy worker/branch UX policy is superseded by Platform-owned execution"
        return C, "legacy canonical research/binding/seal record retained as immutable provenance"

    if path.startswith("models/iris/vnext/"):
        name = Path(path).name
        if name.startswith("cohort_v") or name in {
            "h_rep_probe_v1.py",
            "optimization_v2.py",
            "teacher_truth_v1.py",
            "topology_teacher_v1.py",
        }:
            return H, "research/model utility requires dependency and supersession review"
        return H, "IRIS vNext implementation requires active import/consumer closure before recovery"

    if path.startswith("models/tessa/v2/"):
        name = Path(path).name
        if name in {
            "ab_court_v2.py",
            "court_v2.py",
            "experiment_v2.py",
            "global_attention_v2.py",
            "model_global_v2.py",
            "training_corpus_v2.py",
            "incremental_generation_v2.py",
        }:
            return H, "TESSA V2 auxiliary implementation requires current-consumer/supersession review"
        return H, "TESSA V2 source requires dependency closure before recovery"

    if path.startswith("models/tessa/v3/"):
        return H, "TESSA V3 source requires dependency closure before recovery"

    if path.startswith("compiler/") or path.startswith("runtime/"):
        return G, "compiler/runtime delta requires product-consumer and qualification review"

    if path.startswith("tests/compiler/"):
        return G, "compiler test follows product-relevant candidate until consumer closure is resolved"

    if path.startswith("tests/iris/"):
        if _is_ab_execution(path):
            return B, "test is specific to the current Structural-Evidence A/B execution/binding"
        return H, "IRIS/TESSA test must follow the disposition of the source contract it protects"

    if path.startswith("tests/repository/"):
        return H, "repository/source-gate test requires review after branch-workflow authority is removed"

    if path.startswith("tools/iris_vnext/"):
        if _is_ab_execution(path):
            return B, "tool is specific to the current Structural-Evidence A/B execution/evaluation"
        return H, "research tool requires generic-vs-experiment-only and supersession review"

    raise KeyError(path)


def _blob_sha(repo: Path, ref: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", f"{ref}:{path}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _changed_paths(repo: Path, base: str, head: str) -> Iterable[tuple[str, str]]:
    raw = _run(repo, "diff", "--name-status", "--find-renames", f"{base}...{head}")
    for line in raw.splitlines():
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R"):
            if len(fields) != 3:
                raise RuntimeError(f"unexpected rename row: {line}")
            yield status, fields[2]
        else:
            if len(fields) != 2:
                raise RuntimeError(f"unexpected diff row: {line}")
            yield status, fields[1]


def build_inventory(repo: Path, base: str, head: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    unclassified: list[str] = []
    for status, path in _changed_paths(repo, base, head):
        try:
            disposition, rationale = classify(path)
        except KeyError:
            disposition = "UNCLASSIFIED"
            rationale = "no deterministic drain rule matched"
            unclassified.append(path)
        rows.append(
            {
                "path": path,
                "git_status": status,
                "head_blob_sha": _blob_sha(repo, head, path),
                "disposition": disposition,
                "rationale": rationale,
            }
        )

    counts = Counter(str(row["disposition"]) for row in rows)
    return {
        "schema": SCHEMA,
        "base": base,
        "head": head,
        "base_commit": _run(repo, "rev-parse", base).strip(),
        "head_commit": _run(repo, "rev-parse", head).strip(),
        "changed_path_count": len(rows),
        "classification_counts": dict(sorted(counts.items())),
        "unclassified_paths": unclassified,
        "complete_path_classification": not unclassified,
        "entries": rows,
    }


def self_test() -> None:
    cases = {
        "CURRENT_STATE.md": E,
        ".github/workflows/structural_evidence_ab_execution_v1.yml": F,
        "canonical/STRUCTURAL_EVIDENCE_AB_PREREGISTRATION_V2.json": C,
        "docs/research/IRIS_SHARED_STRUCTURAL_EVIDENCE_PLAN_20261020.md": C,
        "models/iris/vnext/structure_v4.py": A,
        "models/iris/vnext/cohort_v9.py": H,
        "models/tessa/v2/model_v2.py": A,
        "models/tessa/v3/model_v3.py": A,
        "compiler/realsas_compiler_core/graphics_v2.py": G,
        "runtime/realsas_cpp/src/graphics_v1.hpp": A,
        "tests/iris/test_structural_evidence_ab_execution_v1.py": B,
        "tools/iris_vnext/evaluate_structural_evidence_ab_directional_screen_v1.py": B,
        "tools/iris_vnext/audit_fit8_h_geo_p1_v1.py": H,
        "tools/notebooks/build_iris_vnext_fit8_a100_preflight_v1.py": D,
        "tools/verify_native_runtime_source_seal_v2.py": A,
    }
    for path, expected in cases.items():
        actual, _ = classify(path)
        if actual != expected:
            raise AssertionError(f"{path}: expected {expected}, got {actual}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("legacy drain classifier self-test: PASS")
        if not args.base and not args.head:
            return 0

    if not args.base or not args.head:
        parser.error("--base and --head are required unless only --self-test is used")

    repo = args.repo_root.resolve()
    inventory = build_inventory(repo, args.base, args.head)
    payload = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "base_commit": inventory["base_commit"],
        "head_commit": inventory["head_commit"],
        "changed_path_count": inventory["changed_path_count"],
        "classification_counts": inventory["classification_counts"],
        "unclassified_paths": inventory["unclassified_paths"],
    }, indent=2, sort_keys=True))
    return 0 if inventory["complete_path_classification"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
