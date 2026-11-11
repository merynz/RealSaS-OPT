#!/usr/bin/env python3
"""Read-only live RSS.vNext / Truth Foundry work ledger from GitHub Issues.

Task metadata and dependency graph are versioned source. Operational status is
NOT the repository JSON: it is read from GitHub issue open/closed states, and
completion is checked only when a documented evidence receipt exists.

No mutation, issue closure, source truth admission or model promotion occurs here.
Requires network or --snapshot for offline/synthetic tests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "canonical/RSS_VNEXT_WORK_LEDGER_INDEX_V1.json"
API_HOST = "https://api.github.com"
CLOSE_HEADING = re.compile(r"(?mi)^## Completion Evidence\s*$")
HASH = re.compile(r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{64}|[0-9a-fA-F]{40})(?![0-9a-fA-F])")
URL = re.compile(r"https://(?:github\.com|drive\.google\.com|api\.github\.com)/[^\s)>\]]+", re.I)
COMPLETION_TAG = "RSS_LEDGER_COMPLETION_V1"
MAX_PAGES = 20


def load_index(path: pathlib.Path = DEFAULT_INDEX) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "RealSaS.RSSvNextWorkLedgerIndex.v1":
        raise ValueError("LEDGER_INDEX_SCHEMA_DRIFT")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 25:
        raise ValueError("LEDGER_TASK_COUNT_DRIFT")
    ids = [t["id"] for t in tasks]
    nums = [t["issue_number"] for t in tasks]
    if len(set(ids)) != len(ids) or len(set(nums)) != len(nums):
        raise ValueError("LEDGER_TASK_ID_DUPLICATE")
    if {t["scope"] for t in tasks} != {"architecture", "truth", "operations"}:
        raise ValueError("LEDGER_SCOPE_DRIFT")
    for task in tasks:
        if task["id"] in task["depends_on"] or any(d not in ids for d in task["depends_on"]):
            raise ValueError("LEDGER_DEPENDENCY_INVALID:" + task["id"])
    if data.get("completion_evidence_marker") != COMPLETION_TAG:
        raise ValueError("LEDGER_COMPLETION_MARKER_DRIFT")
    return data


def fetch_issues(repo: str, expected_numbers: set[int], token: str | None = None) -> dict[int, dict]:
    """Paginate GitHub Issues, including closed ones; never infer missing=OPEN."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("INVALID_REPOSITORY")
    found: dict[int, dict] = {}
    for page in range(1, MAX_PAGES + 1):
        query = urllib.parse.urlencode({"state": "all", "per_page": 100, "page": page})
        url = f"{API_HOST}/repos/{repo}/issues?{query}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "RSSvNext-ReadOnlyLedger/1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=18) as response:
                page_data = json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"GITHUB_LIVE_LEDGER_READ_FAILED:{type(exc).__name__}") from exc
        if not isinstance(page_data, list):
            raise ValueError("GITHUB_LEDGER_RESPONSE_NOT_LIST")
        for record in page_data:
            if "pull_request" in record:
                continue
            number = record.get("number")
            if number in expected_numbers:
                found[number] = record
        if len(found) == len(expected_numbers):
            return found
        if len(page_data) < 100:
            break
    missing = sorted(expected_numbers - set(found))
    raise RuntimeError("LIVE_LEDGER_ISSUE_NOT_FOUND:" + ",".join(map(str, missing)))


def evaluate_issue(index_task: dict, issue: dict) -> dict:
    task_id = index_task["id"]
    expected_number = index_task["issue_number"]
    if issue.get("number") != expected_number:
        raise ValueError("LIVE_LEDGER_ISSUE_NUMBER_MISMATCH:" + task_id)
    title = issue.get("title") or ""
    if not title.startswith(f"[RSS.vNext/{task_id}]"):
        raise ValueError("LIVE_LEDGER_ISSUE_TITLE_MISMATCH:" + task_id)
    state = issue.get("state")
    reason = issue.get("state_reason")
    body = issue.get("body") or ""
    if state not in ("open", "closed"):
        raise ValueError("LIVE_LEDGER_INVALID_STATE:" + task_id)
    receipt = ""
    head = CLOSE_HEADING.search(body)
    if head:
        receipt = body[head.end():]
    evidence_ok = (
        bool(head)
        and COMPLETION_TAG in receipt
        and bool(HASH.search(receipt))
        and bool(URL.search(receipt))
    )
    completed = state == "closed" and reason == "completed" and evidence_ok
    if completed:
        effective = "COMPLETE_WITH_RECEIPT"
    elif state == "open":
        effective = "OPEN"
    elif reason == "not_planned":
        effective = "CLOSED_NOT_PLANNED"
    else:
        effective = "CLOSED_UNVERIFIED"
    return {
        "id": task_id,
        "number": expected_number,
        "scope": index_task["scope"],
        "title": index_task["title"],
        "depends_on": index_task["depends_on"],
        "github_state": state,
        "github_state_reason": reason,
        "completion_receipt_syntax_present": evidence_ok,
        "checked": completed,
        "status": effective,
        "url": issue.get("html_url") or f"https://github.com/merynz/RealSaS-OPT/issues/{expected_number}",
    }


def evaluate(index: dict, issue_by_number: dict[int, dict]) -> dict:
    tasks = [evaluate_issue(t, issue_by_number[t["issue_number"]]) for t in index["tasks"]]
    counter = Counter(t["status"] for t in tasks)
    return {
        "schema": "RealSaS.RSSvNextLiveWorkLedgerReadback.v1",
        "repository": index["repository"],
        "hub_issue": index["issue_hub"],
        "source_of_truth": "LIVE_GITHUB_ISSUES",
        "architecture_design_frozen": index["architecture_source"]["design_frozen"],
        "total": len(tasks),
        "checked_completed": sum(t["checked"] for t in tasks),
        "open": counter["OPEN"],
        "closed_unverified": counter["CLOSED_UNVERIFIED"],
        "closed_not_planned": counter["CLOSED_NOT_PLANNED"],
        "tasks": tasks,
    }


def markdown(report: dict) -> str:
    lines = [
        "# RSS.vNext — Live Architecture + Truth Ledger",
        "",
        f"Source: https://github.com/{report['repository']}/issues/{report['hub_issue']}  ",
        f"**Checked {report['checked_completed']} / {report['total']}** | OPEN {report['open']} | "
        f"CLOSED_UNVERIFIED {report['closed_unverified']} | NOT_PLANNED {report['closed_not_planned']}",
        "",
        "Checked means the GitHub issue is closed-as-completed and has a syntactically valid "
        "completion receipt. It does *not* mean that this script independently re-ran science.",
    ]
    for scope, header in [
        ("architecture", "Architecture / implementation"),
        ("truth", "Truth Foundry / admission"),
        ("operations", "Source recovery / operations"),
    ]:
        lines.extend(["", "## " + header, ""])
        for item in report["tasks"]:
            if item["scope"] != scope:
                continue
            check = "x" if item["checked"] else " "
            dep = ", ".join(item["depends_on"])
            suffix = f" — {item['status']}"
            if dep:
                suffix += f" · deps: {dep}"
            lines.append(f"- [{check}] [{item['id']}: {item['title']}]({item['url']}){suffix}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=pathlib.Path, default=DEFAULT_INDEX)
    parser.add_argument("--snapshot", type=pathlib.Path, default=None,
                        help="Offline test fixture with GitHub /issues-like records; never claims live status")
    parser.add_argument("--format", choices=["md", "json"], default="md")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    index = load_index(args.index)
    if args.snapshot:
        data = json.loads(args.snapshot.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "issues" in data:
            data = data["issues"]
        if not isinstance(data, list):
            raise ValueError("SNAPSHOT_EXPECTS_GITHUB_ISSUE_LIST")
        issues = {r["number"]: r for r in data}
        required = {t["issue_number"] for t in index["tasks"]}
        if required - set(issues):
            raise ValueError("SNAPSHOT_INCOMPLETE_NO_FABRICATED_OPEN_STATE")
        origin = "OFFLINE_SNAPSHOT__NOT_LIVE"
    else:
        issues = fetch_issues(index["repository"], {t["issue_number"] for t in index["tasks"]},
                              os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
        origin = "LIVE_GITHUB_API"
    report = evaluate(index, issues)
    report["read_origin"] = origin
    report["read_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    output = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.format == "json" else markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"RSS_VNEXT_LEDGER_READ_FAILED:{exc}", file=sys.stderr)
        raise SystemExit(2)
