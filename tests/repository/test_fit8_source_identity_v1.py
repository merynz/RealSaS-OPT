from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.corpus.verify_fit8_sources_v1 import ROOT, safe_relative, verify


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Fit8SourceIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.selection = json.loads((ROOT / "canonical/FIT8_DIVERSITY_SELECTION_V1.json").read_text())
        packs = {}
        source_bytes = {}
        for row in self.selection["selected"]:
            src = row["source"]
            data = row["subject_key"].encode()
            src["sha256"], src["byte_size"] = sha(data), len(data)
            source_bytes[src["corpus_relative_path"]] = data
            packs[row["pack_id"]] = {
                "id": row["pack_id"], "status": "SEALED_RAW", "provider": row["creator"],
                "official_url": src["official_page"], "source_snapshot_sha256": src["source_page_sha256"],
                "license": src["license"], "drive_folder_id": src["locator"]["pack_folder_id"],
            }
        manifest = json.dumps({"corpus_id": self.selection["raw_corpus"]["corpus_id"],
                               "results": list(packs.values())}).encode()
        ledger = "".join(f"{sha(data)}  {path}\n" for path, data in source_bytes.items()).encode()
        material = b"synthetic corpus root material\n"
        expected = self.selection["raw_corpus"]
        expected.update(manifest_sha256=sha(manifest), file_ledger_sha256=sha(ledger),
                        root_sha256=sha(material))
        seal = {"corpus_id": expected["corpus_id"], "manifest_sha256": sha(manifest),
                "file_ledger_sha256": sha(ledger), "corpus_root_sha256": sha(material)}
        self.bundle = self.root / "audit.zip"
        with zipfile.ZipFile(self.bundle, "w") as z:
            z.writestr("manifest\\", b"")
            z.writestr("manifest\\MANIFEST.json", manifest)
            z.writestr("manifest\\FILE_HASHES.sha256", ledger)
            z.writestr("manifest\\CORPUS_SEAL.json", json.dumps(seal))
            z.writestr("manifest\\CORPUS_SEAL_MATERIAL.txt", material)
            for path, data in source_bytes.items():
                z.writestr(path.replace("/", "\\"), data)
        expected["audit_package_sha256"] = sha(self.bundle.read_bytes())
        self.selection_path = self.root / "selection.json"
        self.write_selection()

    def write_selection(self) -> None:
        self.selection_path.write_text(json.dumps(self.selection))

    def test_exact_bytes_do_not_claim_import_or_render(self) -> None:
        receipt = verify(self.selection_path, self.bundle, None)
        self.assertEqual(receipt["verified_source_count"], 8)
        self.assertFalse(receipt["import_executed"])
        self.assertFalse(receipt["render_executed"])
        self.assertFalse(receipt["product_pass_claimed"])

    def test_changed_local_source_cannot_hide_behind_archive(self) -> None:
        path = self.root / self.selection["selected"][0]["source"]["corpus_relative_path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "SOURCE_BYTE_HASH"):
            verify(self.selection_path, self.bundle, self.root)

    def test_corrupt_audit_bundle_fails_before_reading(self) -> None:
        with self.bundle.open("ab") as f:
            f.write(b"drift")
        with self.assertRaisesRegex(ValueError, "AUDIT_BUNDLE_HASH"):
            verify(self.selection_path, self.bundle, None)

    def test_relabeling_source_hash_does_not_change_ledger_authority(self) -> None:
        self.selection["selected"][0]["source"]["sha256"] = "0" * 64
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "SOURCE_LEDGER_HASH"):
            verify(self.selection_path, self.bundle, None)

    def test_duplicate_source_and_admission_claim_fail(self) -> None:
        original = copy.deepcopy(self.selection)
        self.selection["selected"][1]["source"]["sha256"] = self.selection["selected"][0]["source"]["sha256"]
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "DUPLICATE_SOURCE"):
            verify(self.selection_path, self.bundle, None)
        self.selection = original
        self.selection["execution_ready"] = True
        self.write_selection()
        with self.assertRaisesRegex(ValueError, "IDENTITY_IS_NOT_ADMISSION"):
            verify(self.selection_path, self.bundle, None)

    def test_external_paths_are_not_resolved(self) -> None:
        for path in ("../raw/asset", "/raw/asset", "C:\\raw\\asset", "raw//asset"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_relative(path)

    def test_committed_receipt_binds_selection_and_eight_sources(self) -> None:
        raw = (ROOT / "canonical/FIT8_DIVERSITY_SELECTION_V1.json").read_bytes()
        roster = json.loads(raw)
        receipt = json.loads((ROOT / "canonical/FIT8_SOURCE_IDENTITY_AUDIT_20261013.json").read_text())
        self.assertEqual(receipt["selection_sha256"], sha(raw))
        self.assertEqual(receipt["sources"], [{"subject_key": row["subject_key"],
                         "corpus_relative_path": row["source"]["corpus_relative_path"],
                         "sha256": row["source"]["sha256"], "byte_size": row["source"]["byte_size"]}
                        for row in roster["selected"]])


if __name__ == "__main__":
    unittest.main()
