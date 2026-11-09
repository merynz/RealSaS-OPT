"""Synthetic source-locked V14 shard contracts; not a real-source court."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.corpus.truth_foundry.evidence_v14 import sharded_normalized_recovery_v14 as v14


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
        rows=list(csv.DictReader(self.queue.open()))
        rows[0]["candidate_id"]="../escape"
        with self.queue.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["source_key","candidate_id"])
            w.writeheader()
            w.writerows(rows)
        with self.assertRaises(AssertionError):
            v14.all_rows(self.queue,self.inventory)


if __name__=="__main__":
    unittest.main()
