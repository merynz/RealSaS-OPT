from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
WORKFLOW_DIR = ROOT / ".github" / "workflows"

META_PATTERNS = {
    "preregistration": re.compile(r"^\*\*Preregistration:\*\*\s*`([^`]+)`\s*$", re.M),
    "workflow": re.compile(r"^\*\*Workflow:\*\*\s*`([^`]+)`\s*$", re.M),
    "run": re.compile(r"^\*\*Run:\*\*\s*`?(\d+)`?\s*$", re.M),
    "status": re.compile(r"^\*\*Status:\*\*\s*`([^`]+)`\s*$", re.M),
}
WORKFLOW_NAME_RE = re.compile(r"^name:\s*([^#\n]+?)\s*$", re.M)
PYTHON_PATH_RE = re.compile(r"(?:python|python3)\s+([^\s\\]+\.py)")


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def added_commit(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    out = git("log", "--follow", "--diff-filter=A", "--format=%H", "--", rel)
    commits = [line.strip() for line in out.splitlines() if line.strip()]
    if not commits:
        raise RuntimeError(f"no addition commit found for {rel}")
    return commits[-1]


def is_ancestor(older: str, newer: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", older, newer],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0


def workflow_index() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        match = WORKFLOW_NAME_RE.search(text)
        if match:
            found[match.group(1).strip().strip("'\"")] = path
    return found


def github_run(run_id: int) -> dict:
    repo = os.environ.get("GITHUB_REPOSITORY", "merynz/RealSaS-OPT")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for Actions-run provenance validation")
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "realsas-research-provenance-gate-v1",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def parse_result(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    for key, pattern in META_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            raise RuntimeError(f"{path.name}: missing required metadata field {key}")
        meta[key] = match.group(1)
    return meta


def source_scripts(workflow_path: Path) -> list[Path]:
    text = workflow_path.read_text(encoding="utf-8")
    scripts: list[Path] = []
    for raw in PYTHON_PATH_RE.findall(text):
        clean = raw.strip("'\"")
        candidate = ROOT / clean
        if candidate.exists() and candidate.suffix == ".py":
            scripts.append(candidate)
    return sorted(set(scripts))


def audit_one(result_path: Path, workflows: dict[str, Path]) -> dict:
    meta = parse_result(result_path)
    prereg_path = result_path.parent / meta["preregistration"]
    if not prereg_path.is_file():
        raise RuntimeError(f"{result_path.name}: preregistration file not found: {prereg_path.name}")

    workflow_name = meta["workflow"]
    workflow_path = workflows.get(workflow_name)
    if workflow_path is None:
        raise RuntimeError(f"{result_path.name}: workflow named {workflow_name!r} not found")

    scripts = source_scripts(workflow_path)
    if not scripts:
        raise RuntimeError(f"{result_path.name}: workflow has no directly referenced Python probe")

    run_id = int(meta["run"])
    run = github_run(run_id)
    run_head = str(run.get("head_sha") or "")
    if not run_head:
        raise RuntimeError(f"{result_path.name}: run {run_id} has no head_sha")
    git("cat-file", "-e", f"{run_head}^{{commit}}")

    actual_name = str(run.get("name") or "")
    if actual_name != workflow_name:
        raise RuntimeError(
            f"{result_path.name}: run {run_id} workflow mismatch: expected {workflow_name!r}, got {actual_name!r}"
        )
    if run.get("conclusion") != "success":
        raise RuntimeError(f"{result_path.name}: run {run_id} conclusion is {run.get('conclusion')!r}, not success")

    result_commit = added_commit(result_path)
    sources = [prereg_path, workflow_path, *scripts]
    source_rows = []
    for source in sources:
        commit = added_commit(source)
        if not is_ancestor(commit, run_head):
            raise RuntimeError(
                f"{result_path.name}: source {source.relative_to(ROOT)} was not committed before run head {run_head}"
            )
        source_rows.append(
            {
                "path": source.relative_to(ROOT).as_posix(),
                "addition_commit": commit,
            }
        )

    if not is_ancestor(run_head, result_commit):
        raise RuntimeError(
            f"{result_path.name}: run head {run_head} is not an ancestor of result commit {result_commit}"
        )

    return {
        "result": result_path.relative_to(ROOT).as_posix(),
        "status": meta["status"],
        "run_id": run_id,
        "workflow": workflow_name,
        "run_head_sha": run_head,
        "run_event": run.get("event"),
        "run_conclusion": run.get("conclusion"),
        "result_addition_commit": result_commit,
        "sources": source_rows,
        "provenance_closed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    results = sorted(EXP_DIR.glob("*_RESULT_V1.md"))
    if not results:
        raise RuntimeError("no *_RESULT_V1.md records found")

    workflows = workflow_index()
    audited = []
    for path in results:
        audited.append(audit_one(path, workflows))

    report = {
        "schema": "RealSaS.ResearchProvenanceGate.v1",
        "experiment_dir": EXP_DIR.relative_to(ROOT).as_posix(),
        "audited_result_count": len(audited),
        "all_provenance_closed": all(row["provenance_closed"] for row in audited),
        "records": audited,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PROVENANCE_GATE_FAIL: {exc}", file=sys.stderr)
        raise
