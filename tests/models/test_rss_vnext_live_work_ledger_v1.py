"""Live ledger contract tests: synthetic issues only; no GitHub/API writes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.ledger import rss_vnext_status as ledger


class RSSvNextLiveLedgerTests(unittest.TestCase):
    def setUp(self):
        self.index = ledger.load_index()
        self.task = self.index["tasks"][0]
        self.issue = {
            "number": self.task["issue_number"],
            "title": f"[RSS.vNext/{self.task['id']}] {self.task['title']}",
            "state": "open",
            "state_reason": None,
            "html_url": f"https://github.com/merynz/RealSaS-OPT/issues/{self.task['issue_number']}",
            "body": "Task pending. PR #397 is not done.",
        }

    def test_exact_25_issue_registry_and_frozen_design_not_implementation(self):
        self.assertEqual(len(self.index["tasks"]), 25)
        self.assertEqual(self.index["issue_hub"], 399)
        self.assertEqual([t["issue_number"] for t in self.index["tasks"]], list(range(400, 425)))
        self.assertTrue(self.index["architecture_source"]["design_frozen"])
        self.assertFalse(self.index["architecture_source"]["implementation_proven"])
        self.assertEqual(self.index["work_status_authority"], "LIVE_GITHUB_ISSUES__NOT_STATIC_JSON")

    def test_open_issue_is_unchecked_even_with_receipt(self):
        self.issue["body"] += "\n## Completion Evidence\nRSS_LEDGER_COMPLETION_V1\n" + (
            "https://github.com/merynz/RealSaS-OPT/actions/runs/1234\n"
            "SHA 0123456789abcdef0123456789abcdef01234567\n"
        )
        result = ledger.evaluate_issue(self.task, self.issue)
        self.assertEqual(result["status"], "OPEN")
        self.assertFalse(result["checked"])

    def test_closed_without_evidence_is_not_completed(self):
        self.issue.update(state="closed", state_reason="completed")
        result = ledger.evaluate_issue(self.task, self.issue)
        self.assertEqual(result["status"], "CLOSED_UNVERIFIED")
        self.assertFalse(result["checked"])

    def test_closed_receipt_missing_sha_is_not_completed(self):
        self.issue.update(state="closed", state_reason="completed")
        self.issue["body"] = (
            "## Completion Evidence\nRSS_LEDGER_COMPLETION_V1\n"
            "https://github.com/merynz/RealSaS-OPT/actions/runs/1234\n"
        )
        self.assertFalse(ledger.evaluate_issue(self.task, self.issue)["checked"])

    def test_closed_receipt_with_exact_sha_and_link_is_checked(self):
        self.issue.update(state="closed", state_reason="completed")
        self.issue["body"] = (
            "## Completion Evidence\nRSS_LEDGER_COMPLETION_V1\n"
            "Reviewed and individually verified evidence SHA:\n"
            "0123456789abcdef0123456789abcdef01234567\n"
            "https://github.com/merynz/RealSaS-OPT/actions/runs/1234\n"
        )
        result = ledger.evaluate_issue(self.task, self.issue)
        self.assertTrue(result["checked"])
        self.assertEqual(result["status"], "COMPLETE_WITH_RECEIPT")

    def test_not_planned_issue_does_not_count(self):
        self.issue.update(state="closed", state_reason="not_planned")
        self.issue["body"] = (
            "## Completion Evidence\nRSS_LEDGER_COMPLETION_V1\n"
            "0123456789abcdef0123456789abcdef01234567\n"
            "https://github.com/merynz/RealSaS-OPT/actions/runs/1234\n"
        )
        self.assertEqual(ledger.evaluate_issue(self.task, self.issue)["status"], "CLOSED_NOT_PLANNED")

    def test_mismatched_issue_identity_fails_closed(self):
        self.issue["title"] = "[RSS.vNext/A02] Replacing a task"
        with self.assertRaisesRegex(ValueError, "TITLE_MISMATCH"):
            ledger.evaluate_issue(self.task, self.issue)

    def test_snapshot_never_simulates_live_or_missing_issues(self):
        synthetic = [{
            "number": t["issue_number"],
            "title": f"[RSS.vNext/{t['id']}] {t['title']}",
            "state": "open",
            "state_reason": None,
            "body": "",
        } for t in self.index["tasks"]]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snap = root / "issues.json"
            out = root / "status.json"
            snap.write_text(json.dumps(synthetic), encoding="utf-8")
            self.assertEqual(ledger.main(["--snapshot", str(snap), "--format", "json", "--output", str(out)]), 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["read_origin"], "OFFLINE_SNAPSHOT__NOT_LIVE")
            self.assertEqual((result["total"], result["checked_completed"]), (25, 0))
            self.assertIn("- [ ]", ledger.markdown(result))
            snap.write_text(json.dumps(synthetic[:-1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SNAPSHOT_INCOMPLETE"):
                ledger.main(["--snapshot", str(snap)])

    def test_task_dependencies_are_known_and_acyclic(self):
        items = {t["id"]: t for t in self.index["tasks"]}
        visited = set()
        active = set()

        def walk(name):
            if name in active:
                raise AssertionError("dependency cycle: " + name)
            if name in visited:
                return
            active.add(name)
            for dep in items[name]["depends_on"]:
                walk(dep)
            active.remove(name)
            visited.add(name)

        for tid in items:
            walk(tid)
        self.assertEqual(len(visited), 25)


if __name__ == "__main__":
    unittest.main()
