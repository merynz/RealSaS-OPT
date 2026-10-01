from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "canonical" / "MAINLINE_EXECUTION_PLAN_V2.json"

STAGE_RE = re.compile(r"^\d{2}_[A-Z0-9_]+$")
QUALITY_TERMS = (
    "target_edge_px",
    "min_visible_pixels",
    "max_iterations",
    "resolution",
    "threshold",
    "tolerance",
    "epsilon",
    "stress_angle",
    "max_edge_ratio",
    "min_angle",
    "max_aspect",
    "subdivision",
    "refine",
    "sample_count",
    "tile_resolution",
)
RISK_TERMS = (
    "PASS_DEMO_ONLY",
    "DEMO_ONLY",
    "fallback",
    "legacy",
    "historical",
    "research_only",
    "skip",
    "continue",
    "best_effort",
    "approx",
    "heuristic",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def module_path(module: str) -> Path | None:
    rel = Path(*module.split("."))
    p = ROOT / rel.with_suffix(".py")
    if p.is_file():
        return p
    p = ROOT / rel / "__init__.py"
    if p.is_file():
        return p
    return None


def function_map(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    out = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
    return out


def called_local_names(node: ast.AST, fnames: set[str]) -> set[str]:
    out = set()
    for x in ast.walk(node):
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in fnames:
            out.add(x.func.id)
    return out


def function_closure(module_file: Path, entry: str):
    src = module_file.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(module_file))
    fmap = function_map(tree)
    if entry not in fmap:
        return src, tree, fmap, set()
    seen = set()
    stack = [entry]
    while stack:
        name = stack.pop()
        if name in seen or name not in fmap:
            continue
        seen.add(name)
        stack.extend(sorted(called_local_names(fmap[name], set(fmap))))
    return src, tree, fmap, seen


def literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def node_source(src: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(src, node) or ""
    except Exception:
        return ""


def collect_stage_reads(src: str, fmap: dict, closure: set[str]):
    explicit = []
    all_stage_literals = []
    run_artifact_literals = []

    for name in sorted(closure):
        fn = fmap[name]
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                callee = ""
                if isinstance(n.func, ast.Name):
                    callee = n.func.id
                elif isinstance(n.func, ast.Attribute):
                    callee = n.func.attr
                if callee in {"stage_output_payload", "_stage_output_payload"}:
                    vals = [literal_string(a) for a in n.args]
                    stage = next((v for v in vals if v and STAGE_RE.fullmatch(v)), None)
                    if stage:
                        explicit.append({
                            "stage_id": stage,
                            "function": name,
                            "line": getattr(n, "lineno", None),
                            "source": node_source(src, n)[:400],
                        })
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                value = n.value
                if STAGE_RE.fullmatch(value):
                    all_stage_literals.append({
                        "stage_id": value,
                        "function": name,
                        "line": getattr(n, "lineno", None),
                    })

        fsrc = node_source(src, fn)
        if '["run_root"]' in fsrc and '"artifacts"' in fsrc:
            for m in re.finditer(r'["\'](\d{2}_[A-Z0-9_]+)["\']', fsrc):
                run_artifact_literals.append({
                    "stage_id": m.group(1),
                    "function": name,
                })

    return explicit, all_stage_literals, run_artifact_literals


def transitive_ancestors(stage_id: str, by_id: dict[str, dict]) -> set[str]:
    out = set()
    stack = list(by_id[stage_id].get("depends_on", []))
    while stack:
        x = str(stack.pop())
        if x in out:
            continue
        out.add(x)
        stack.extend(by_id[x].get("depends_on", []))
    return out


def topo_check(stages: list[dict]):
    by_id = {s["id"]: s for s in stages}
    indeg = {s["id"]: 0 for s in stages}
    children = defaultdict(list)
    for s in stages:
        for d in s.get("depends_on", []):
            indeg[s["id"]] += 1
            children[d].append(s["id"])
    q = deque(sorted((x for x, d in indeg.items() if d == 0), key=lambda x: by_id[x]["ordinal"]))
    order = []
    while q:
        x = q.popleft()
        order.append(x)
        for y in children[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                q.append(y)
    return len(order) == len(stages), order


def text_matches(path: Path, terms):
    src = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for lineno, line in enumerate(src.splitlines(), 1):
        low = line.lower()
        for term in terms:
            if term.lower() in low:
                rows.append({"line": lineno, "term": term, "text": line.strip()[:500]})
                break
    return rows


def inspect_information_contracts():
    findings = []

    gsa = ROOT / "compiler/realsas_compiler_core/substrate/scene_first_signed.py"
    if gsa.is_file():
        src = gsa.read_text(encoding="utf-8")
        face_to_edge = (
            "_adaptive_voxel_compact" in src
            and "inverse[f" in src
            and "return cp,cn,edges" in src.replace(" ", "")
        )
        if face_to_edge:
            findings.append({
                "id": "INFO_GSA_FACE_INCIDENCE_COLLAPSE",
                "severity": "CRITICAL",
                "class": "INFORMATION_LOSS",
                "evidence": "dense faces are used to derive compact edges; compact face/cell incidence is not returned by _adaptive_voxel_compact",
                "design_question": "What exact compact 2-simplex/cell incidence must GSA preserve without over-authorizing product mesh topology?",
            })

    cm = ROOT / "compiler/realsas_compiler_core/canonical_mesh_candidate_v1.py"
    if cm.is_file():
        src = cm.read_text(encoding="utf-8")
        if "combinations(neighbors[a], 2)" in src and "safe_edges" in src:
            findings.append({
                "id": "INFO_STAGE18_CLIQUE_TO_FACE_RECONSTRUCTION",
                "severity": "CRITICAL",
                "class": "INFORMATION_INVENTION",
                "evidence": "Stage18 relation baseline reconstructs triangular faces from 3-cliques of pairwise GSA relations",
                "design_question": "Should a face require preserved GSA cell incidence, a constrained discretizer proof, or another explicit higher-order surface witness?",
            })

    v = ROOT / "compiler/realsas_compiler_services/orchestrator/adapters/v2_architecture.py"
    if v.is_file():
        src = v.read_text(encoding="utf-8")
        if '"35_DYNAMIC_MECHANICAL_MESH_QUALIFIED"' in src and "repair_candidate_path" in src:
            findings.append({
                "id": "DAG_STAGE18_READS_STAGE35_REPAIR_SIDE_CHANNEL",
                "severity": "CRITICAL",
                "class": "DECLARED_DAG_VIOLATION",
                "evidence": "Stage18 reads Stage35 repair artifacts directly from run_root while Stage35 declares Stage18 as an upstream dependency",
                "design_question": "Represent repair as explicit immutable epoch/iteration input rather than an undeclared downstream filesystem back-edge.",
            })
        if "target_edge_px=16" in src:
            findings.append({
                "id": "OPT_VISUAL_MESH_TARGET_EDGE_16_UNCALIBRATED",
                "severity": "HIGH",
                "class": "OPTIMIZATION_ASSUMPTION",
                "evidence": "source-owned visual mesh builder is invoked with hard-coded target_edge_px=16",
                "design_question": "Determine optimal/adaptive resolution from source curvature/silhouette/deformation and latency budget before freezing a product value.",
            })

    vm = ROOT / "compiler/realsas_compiler_core/visual_mesh_arap_v1.py"
    if vm.is_file():
        src = vm.read_text(encoding="utf-8")
        if "bound_faces[vertex_index]" in src or "bound_faces" in src:
            findings.append({
                "id": "VISUAL_BINDING_VERTEX_LOCAL_AFFINE_DOMAIN",
                "severity": "HIGH",
                "class": "COMPOSITION_INVARIANT_GAP",
                "evidence": "visual binding stores mechanical face ownership per visual vertex; per-visual-face single affine-domain coherence requires an explicit invariant/gate",
                "design_question": "Decide whether one visual triangle must be contained in one mechanical affine domain or whether a stronger continuous field construction is preferred.",
            })

    st = ROOT / "compiler/realsas_compiler_core/mesh/skin_topology_compatibility_v1.py"
    if st.is_file():
        src = st.read_text(encoding="utf-8")
        if "SEAM_CUT_BY_UNSAFE_FACE_REMOVAL" in src:
            findings.append({
                "id": "REPAIR_SKIN_TOPOLOGY_FACE_DELETE_ONLY",
                "severity": "HIGH",
                "class": "REPAIR_QUALITY_GAP",
                "evidence": "current Stage35 skin-topology repair deletes unsafe faces and declares local_cdt_required=False",
                "design_question": "Choose the optimal topology repair operator (vertex duplication, seam insertion, constrained local retriangulation, etc.) before coding.",
            })

    return findings


def main():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    stages = list(plan["stages"])
    by_id = {str(s["id"]): s for s in stages}
    ordinal = {str(s["id"]): int(s["ordinal"]) for s in stages}

    is_dag, order = topo_check(stages)
    stage_audit = []
    findings = inspect_information_contracts()

    for stage in stages:
        stage_id = str(stage["id"])
        module, _, entry = str(stage["adapter"]).partition(":")
        path = module_path(module)
        row = {
            "stage_id": stage_id,
            "ordinal": int(stage["ordinal"]),
            "adapter": stage["adapter"],
            "declared_dependencies": list(map(str, stage.get("depends_on", []))),
            "adapter_file": None if path is None else str(path.relative_to(ROOT)),
            "explicit_stage_reads": [],
            "stage_literals": [],
            "hidden_artifact_stage_literals": [],
            "undeclared_explicit_reads": [],
            "back_edge_references": [],
        }
        if path is None:
            row["error"] = "ADAPTER_MODULE_FILE_MISSING"
            stage_audit.append(row)
            continue

        src, tree, fmap, closure = function_closure(path, entry)
        explicit, literals, artifact_literals = collect_stage_reads(src, fmap, closure)
        row["explicit_stage_reads"] = explicit
        row["stage_literals"] = literals
        row["hidden_artifact_stage_literals"] = artifact_literals

        declared = set(map(str, stage.get("depends_on", [])))
        ancestors = transitive_ancestors(stage_id, by_id)
        for r in explicit:
            dep = r["stage_id"]
            if dep != stage_id and dep not in declared:
                row["undeclared_explicit_reads"].append(r)
                findings.append({
                    "id": f"DAG_UNDECLARED_STAGE_READ__{stage_id}__{dep}",
                    "severity": "CRITICAL" if dep not in ancestors else "HIGH",
                    "class": "DECLARED_DAG_DEPENDENCY_GAP",
                    "stage": stage_id,
                    "referenced_stage": dep,
                    "evidence": r,
                    "design_question": "Declare the real data dependency or remove the read; fingerprints/cache must bind every consumed authority.",
                })
            if dep in ordinal and ordinal[dep] > ordinal[stage_id]:
                row["back_edge_references"].append(r)
                findings.append({
                    "id": f"DAG_BACK_EDGE__{stage_id}__{dep}",
                    "severity": "CRITICAL",
                    "class": "DAG_BACK_EDGE",
                    "stage": stage_id,
                    "referenced_stage": dep,
                    "evidence": r,
                    "design_question": "Move this into an explicit repair epoch/controller; a single static DAG cannot contain a downstream-to-upstream data edge.",
                })

        for r in artifact_literals:
            dep = r["stage_id"]
            if dep == stage_id:
                continue
            if dep in ordinal and ordinal[dep] > ordinal[stage_id]:
                findings.append({
                    "id": f"DAG_HIDDEN_ARTIFACT_BACK_EDGE__{stage_id}__{dep}",
                    "severity": "CRITICAL",
                    "class": "FILESYSTEM_SIDE_CHANNEL_BACK_EDGE",
                    "stage": stage_id,
                    "referenced_stage": dep,
                    "evidence": r,
                    "design_question": "Eliminate undeclared filesystem coupling; make repair lineage an explicit input to a new immutable iteration.",
                })
        stage_audit.append(row)

    scan_roots = [
        ROOT / "compiler/realsas_compiler_core",
        ROOT / "compiler/realsas_compiler_services/orchestrator/adapters",
        ROOT / "models/iris",
        ROOT / "models/geppetto",
        ROOT / "models/arachne",
    ]
    optimization_scan = []
    risk_scan = []
    for base in scan_roots:
        for path in sorted(base.rglob("*.py")):
            q = text_matches(path, QUALITY_TERMS)
            if q:
                optimization_scan.append({
                    "path": str(path.relative_to(ROOT)),
                    "matches": q[:80],
                })
            r = text_matches(path, RISK_TERMS)
            if r:
                risk_scan.append({
                    "path": str(path.relative_to(ROOT)),
                    "matches": r[:80],
                })

    report = {
        "schema": "RealSaS.V2ArchitectureIntegrityAudit.v1",
        "status": "AUDIT_ONLY__NO_REPAIR_APPLIED",
        "repo_head": "",
        "plan_sha256": sha256_file(PLAN),
        "declared_graph": {
            "stage_count": len(stages),
            "is_dag": bool(is_dag),
            "topological_order": order,
        },
        "finding_count": len(findings),
        "findings": findings,
        "stage_dependency_audit": stage_audit,
        "optimization_and_quality_knob_scan": optimization_scan,
        "fallback_demo_legacy_risk_scan": risk_scan,
        "audit_policy": {
            "repairs_applied": False,
            "thresholds_changed": False,
            "optimization_values_selected": False,
            "design_before_implementation_required": True,
            "performance_contract": "canonical/PRODUCT_COMPILE_PERFORMANCE_CONTRACT_V1_20260927.json",
        },
    }

    try:
        import subprocess
        report["repo_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        report["repo_head"] = "UNKNOWN"

    out = ROOT / "canonical" / "V2_ARCHITECTURE_INTEGRITY_AUDIT_V1_20260928.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("ARCH_AUDIT_HEAD", report["repo_head"])
    print("ARCH_AUDIT_DECLARED_DAG", report["declared_graph"]["is_dag"])
    print("ARCH_AUDIT_FINDINGS", len(findings))
    for row in findings:
        print("ARCH_FINDING", json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
