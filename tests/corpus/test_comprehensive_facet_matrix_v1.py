"""Truth Foundry 264-facet inventory must remain provenance-bound and fail-closed."""
from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from tools.corpus.truth_foundry.comprehensive_facet_matrix_v1 import (
    IMPORT, load_catalog, run,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "canonical/REALSAS_TRUTH_FOUNDRY_COMPREHENSIVE_FACETS_V1.json"


class ComprehensiveFacetMatrixRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, self.facets = load_catalog(CATALOG)
        numeric = {f["preexisting_numeric_observation_field"] for f in self.facets
                   if f.get("preexisting_numeric_observation_field")}
        self.source = self.root / "source.csv"
        self.headers = (
            ["corpus", "asset_id", "source_sha256", "historical_split",
             "physical_census_loaded", "training_authorized", "materialization_authorized"]
            + sorted(numeric)
        )
        self.rows = [
            dict.fromkeys(self.headers, ""),
            dict.fromkeys(self.headers, ""),
            dict.fromkeys(self.headers, ""),
        ]
        for i, row in enumerate(self.rows):
            row.update({
                "corpus": "MASTER3993" if i < 2 else "CORE370",
                "asset_id": "asset_" + str(i), "source_sha256": chr(97+i)*64,
                "historical_split": "FIT" if i != 2 else "CORE370_UNRESOLVED",
                "physical_census_loaded": "True" if i == 0 else "False",
                "training_authorized": "False", "materialization_authorized": "False",
            })
            if i == 0:
                for field in numeric:
                    row[field] = "0"
                row["mesh_disconnected_components"] = "44"
                row["observed_bone_count"] = "65"
                row["skin_rows_sampled"] = "520"
        self.write_input()

    def write_input(self):
        with self.source.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=self.headers)
            w.writeheader()
            w.writerows(self.rows)

    def produce(self, name="out"):
        return run(self.source, CATALOG, self.root/name,
                   expected_master=2, expected_core=1)

    def test_catalog_is_exhaustive_across_heads_and_domains(self):
        obj, facets = load_catalog(CATALOG)
        self.assertEqual(obj["group_count"], 14)
        self.assertGreaterEqual(len(facets), 250)
        self.assertEqual(len(facets), len({x["id"] for x in facets}))
        heads = {h for f in facets for h in f["consumer_heads"]}
        self.assertEqual(heads, {"IRIS_GEOMETRY","IRIS_SEMANTIC",
                                 "IRIS_GRAPHICS","TESSA","AXIS","MIRA"})
        groups = {f["group"] for f in facets}
        for needed in ("identity_rights_lineage","semantic_source_regions",
                       "graphics_drawable_causality","caa_canonical_artifact",
                       "compiler_mechanics_products","quality_qualification"):
            self.assertIn(needed, groups)
        self.assertTrue(all(f["training_authorized"] is False for f in facets))

    def test_real_import_count_is_hint_not_semantics(self):
        s = self.produce()
        self.assertEqual(s["source_record_count"], 3)
        self.assertEqual(s["asset_facet_slots"], 3*len(self.facets))
        self.assertEqual(s["group_review_tasks_open"], 3*14)
        self.assertEqual(s["master_physical_census_bound"], 1)
        self.assertEqual(s["independent_target_court_pass_minted"], 0)
        self.assertEqual(s["source_face_semantic_labels_minted"], 0)
        self.assertEqual(s["graphics_drawable_pass_minted"], 0)
        with gzip.open(self.root/"out"/"ASSET_FACET_EVIDENCE_MATRIX_V1.jsonl.gz","rt") as fp:
            first = json.loads(next(fp))
            second = json.loads(next(fp))
        self.assertEqual(first["facet_total"], len(self.facets))
        self.assertEqual(first["facet_evidence"]["mesh_object_vs_connected_island"]["observed_value"],44)
        self.assertEqual(first["facet_evidence"]["joint_parent_hierarchy_and_cycles"]["observed_value"],65)
        self.assertEqual(first["facet_evidence"]["skin_sparse_influence_index_weight"]["observed_value"],520)
        self.assertEqual(first["facet_evidence"]["native_ik_fk_constraint_graph"]["observation_state"], IMPORT)
        self.assertEqual(first["facet_evidence"]["native_ik_fk_constraint_graph"]["independent_court_state"],"UNKNOWN")
        self.assertEqual(first["facet_evidence"]["source_face_concept_review"]["observation_state"],"UNKNOWN_UNREVIEWED")
        self.assertEqual(first["facet_evidence"]["full_keep_hide_isolate_render_pairs"]["supervision_status"],"NOT_ADMITTED")
        self.assertEqual(second["facet_evidence"]["joint_parent_hierarchy_and_cycles"]["observed_value"],None)
        self.assertTrue(all(x["training_mask_valid"] is False for x in first["facet_evidence"].values()))

    def test_same_frozen_input_gives_identical_gzip_sha(self):
        a = self.produce("out_a")
        b = self.produce("out_b")
        self.assertEqual(a["file_sha256"]["ASSET_FACET_EVIDENCE_MATRIX_V1.jsonl.gz"],
                         b["file_sha256"]["ASSET_FACET_EVIDENCE_MATRIX_V1.jsonl.gz"])

    def test_forged_training_authority_is_rejected(self):
        self.rows[0]["training_authorized"] = "True"
        self.write_input()
        with self.assertRaisesRegex(ValueError,"INPUT_ALREADY_AUTHORIZED"):
            self.produce()

    def test_imported_missing_count_fails_not_unknown(self):
        self.rows[0]["observed_bone_count"] = ""
        self.write_input()
        with self.assertRaisesRegex(ValueError,"MISSING_OR_NONNUMERIC_IMPORT"):
            self.produce()

    def test_core_import_claim_fails(self):
        self.rows[2]["physical_census_loaded"] = "True"
        self.write_input()
        with self.assertRaisesRegex(ValueError,"CORE_FALSE_PHYSICAL_CENSUS"):
            self.produce()

    def test_source_digest_tamper_fails(self):
        with self.assertRaisesRegex(ValueError,"FROZEN_SOURCE_MATRIX_SHA"):
            run(self.source, CATALOG, self.root/"out",
                expected_master=2, expected_core=1,
                expected_source_sha="0"*64)

    def test_no_dup_source(self):
        self.rows[1]["asset_id"] = "asset_0"
        self.write_input()
        with self.assertRaisesRegex(ValueError,"SOURCE_DUPLICATE_OR_BLANK"):
            self.produce()


if __name__ == "__main__":
    unittest.main()
