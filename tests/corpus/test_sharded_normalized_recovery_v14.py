"""Synthetic source-locked V14 shard contracts; not a real-source court."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.corpus.truth_foundry.evidence_v14 import sharded_normalized_recovery_v14 as v14
from tools.corpus.truth_foundry.evidence_v14 import bounded_direct_copy_v14 as transport


class V14RecoveryContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "queue.csv"
        self.inventory = self.root / "inventory.csv.gz"
        with self.queue.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source_key", "candidate_id"])
            w.writeheader()
            for i in range(1527):
                w.writerow({"source_key": f"S:{i:04d}",
                            "candidate_id": f"quaternius:{i:024x}"})
        with gzip.open(self.inventory, "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "source_key", "pool", "master_v11_source_recovery_status"])
            w.writeheader()
            for i in range(1527):
                state = ("SHA_VERIFIED_RECOVERY_RECEIPT" if i < 726
                         else ("PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED"
                               if i < 861 else "PATH_ONLY_UNVERIFIED"))
                w.writerow({"source_key": f"S:{i:04d}", "pool":"MASTER3993",
                            "master_v11_source_recovery_status":state})

    def test_disjoint_exact_726_and_exclude_135_holdouts(self):
        _, queue, eligible, holdouts = v14.all_rows(self.queue,self.inventory)
        self.assertEqual((len(queue),len(eligible),len(holdouts)),(1527,726,135))
        for count in (2, 8, 16):
            shards = [v14.assigned(eligible,index,count) for index in range(count)]
            keys = [r["source_key"] for group in shards for r in group]
            self.assertEqual(len(keys),726)
            self.assertEqual(len(set(keys)),726)
            self.assertTrue(all(k.startswith("S:") and int(k[2:]) < 726 for k in keys))

    def test_plan_is_hash_bound_and_no_holdout_paths(self):
        dest = self.root / "plan"
        v14.plan(SimpleNamespace(queue=str(self.queue),
                                 baseline_inventory=str(self.inventory),
                                 out=str(dest),shard_index=3,shard_count=8))
        report = v14.read_json(dest / "SHARD_PLAN_V14.json")
        paths = (dest/"paths.txt").read_text().splitlines()
        self.assertEqual(len(paths),report["assigned_source_count"]*2)
        self.assertEqual(report["protected_holdouts_excluded"],135)
        self.assertFalse(any("source_original" in p or "raw.fbx" in p for p in paths))
        self.assertTrue(all(p.endswith(("/normalized.npz","/source_structure.json")) for p in paths))
        self.assertEqual(v14.sha(dest/"queue_shard.csv"),report["subset_queue_sha256"])
        with self.queue.open("a") as f:
            f.write("DRIFT\n")
        self.assertNotEqual(v14.sha(self.queue),report["queue_input_sha256"])

    def test_merge_rejects_missing_shards(self):
        with self.assertRaises(FileNotFoundError):
            v14.merge(SimpleNamespace(shards=str(self.root / "missing"),
                out=str(self.root/"merge"), queue=str(self.queue),
                baseline_inventory=str(self.inventory), shard_count=8))

    def test_invalid_identity_rejected_before_download(self):
        with self.queue.open() as f:
            rows=list(csv.DictReader(f))
        rows[0]["candidate_id"]="../escape"
        with self.queue.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["source_key","candidate_id"])
            w.writeheader()
            w.writerows(rows)
        with self.assertRaises(AssertionError):
            v14.all_rows(self.queue,self.inventory)



class V14DriveQuotaSafety(unittest.TestCase):
    def test_reasons_distinguish_quota_from_permission_without_private_paths(self):
        cases = (
            (b'Error 403: userRateLimitExceeded /secret/location', "USER_RATE_LIMIT_EXCEEDED", "RATE_OR_QUOTA"),
            (b'Error 403: rateLimitExceeded /secret/location', "PROJECT_RATE_LIMIT_EXCEEDED", "RATE_OR_QUOTA"),
            (b'Error 403: downloadQuotaExceeded /secret/location', "DOWNLOAD_QUOTA_EXCEEDED", "RATE_OR_QUOTA"),
            (b'Error 403: storageQuotaExceeded /secret/location', "STORAGE_QUOTA_EXCEEDED", "RATE_OR_QUOTA"),
            (b'Error 403: insufficientFilePermissions /secret/location', "EXPLICIT_ACCESS_DENIED", "EXPLICIT_ACCESS_DENIED"),
            (b'Error 403: forbidden /secret/location', "HTTP_403_UNSPECIFIED", "HTTP_403_UNSPECIFIED"),
        )
        for raw, reason, category in cases:
            with self.subTest(reason=reason, category=category):
                self.assertEqual(transport.safe_drive_reason(raw), reason)
                self.assertEqual(transport.classify(raw), category)
                self.assertNotIn("secret", transport.safe_drive_reason(raw))

    def test_rate_limit_never_retries_same_file(self):
        with tempfile.TemporaryDirectory() as temp:
            item = "quaternius_" + "a" * 24 + "/normalized.npz"
            process = SimpleNamespace(returncode=1,
                stderr=b'Google Drive error 403: userRateLimitExceeded',
                stdout=b'')
            with mock.patch.object(transport.subprocess, "run", return_value=process) as caller:
                result = transport.fetch(item, "gdrive:private", Path(temp), 20, 2)
            self.assertEqual(result["status"], "RATE_OR_QUOTA")
            self.assertEqual(result["drive_error_reason"], "USER_RATE_LIMIT_EXCEEDED")
            self.assertEqual(caller.call_count, 1)
            self.assertFalse((Path(temp)/item).exists())

    def test_path_allowlist_rejects_source_original_and_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            for item in ("../escape/normalized.npz", "quaternius_" + "a"*24 + "/original.fbx"):
                with self.subTest(item=item), self.assertRaises(ValueError):
                    transport.fetch(item, "gdrive:private", Path(temp), 20, 1)


if __name__=="__main__":
    unittest.main()
