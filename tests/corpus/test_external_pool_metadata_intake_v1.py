"""Regression: metadata discovery is not independent source/admission truth."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.corpus.truth_foundry.external_pool_metadata_intake_v1 import (
    load_catalog, original_sha_from_record, parse_indexes, run
)

ROOT = Path(__file__).resolve().parents[2]
CAT = ROOT/"canonical/REALSAS_TRUTH_FOUNDRY_EXTERNAL_POOL_DERIVABILITY_V1.json"


class ExternalPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_all_six_heads_and_derivation_gates(self):
        x=load_catalog(CAT)
        self.assertFalse(x["training_admission"])
        self.assertEqual(len(x["pools"]),5)
        self.assertEqual(sum(len(p["per_head"]) for p in x["pools"]),30)
        pp={p["id"]:p for p in x["pools"]}
        self.assertEqual(pp["ARTICULATION_XL2"]["per_head"]["TESSA"]["status"],
                         "PUBLISHED_NATIVE")
        self.assertEqual(pp["PARTVERSE_XL"]["per_head"]["IRIS_SEMANTIC"]["status"],
                         "SOURCE_AND_BUILDER")
        self.assertEqual(pp["ANYMATE"]["per_head"]["IRIS_GRAPHICS"]["status"],
                         "SOURCE_RECOVERY")
        self.assertEqual(pp["UNIML3D"]["per_head"]["AXIS"]["status"],
                         "PUBLISHED_NATIVE")
        for p in pp.values():
            self.assertEqual(len(p["per_head"]),6)
            self.assertTrue(all(v["requires"] for v in p["per_head"].values()))

    def test_sha_exact_match_requires_64_hex(self):
        self.assertEqual(original_sha_from_record({"uuid":"1"*64}),"")
        self.assertEqual(original_sha_from_record({"sha256":"A"*64}),"a"*64)
        self.assertEqual(original_sha_from_record({"sha256":"hello"}),"")

    def test_only_sha_crosswalk_no_name_match(self):
        pool={"id":"SAMPLE","metadata_files":["meta.csv"]}
        d=self.root/"data";d.mkdir()
        (d/"meta.csv").write_text("uuid,sha256,name\nid1,"+"a"*64+",person\n"
                                  "id2,,person\n",encoding="utf-8")
        rows,summaries=parse_indexes(pool,"e"*40,d,{"a"*64:["MASTER3993:asset_1"]})
        self.assertEqual(summaries[0]["rows"],2)
        self.assertEqual(summaries[0]["sha_match_candidates_not_deduplicated"],1)
        self.assertEqual(rows[0]["master_exact_sha_candidate_match_count"],1)
        self.assertEqual(rows[1]["master_exact_sha_candidate_match_count"],0)
        self.assertFalse(rows[0]["train_eligible"])

    def test_fake_remote_pinned_metadata_and_zero_admission(self):
        entries={
            "yfdeng/Anymate":{"README.md":"# AnyMate\n"},
            "Seed3D/Articulation-XL2.0":{
                "README.md":"# XL\n",
                "meta_Articulation_XL_2.0.csv":"uuid,source,vertex_count,face_count,joint_count,bone_count,category_label,fileType,fileIdentifier\nidA,src,100,20,3,4,animal,glb,path\n",
            },
            "dscdyc/partversexl":{
                "README.md":"# PartVerse\n",
                "metadata.csv":"sha256,name\n"+"a"*64+",animal\n",
                "train.csv":"uid\np0\n", "val.csv":"uid\np1\n",
            },
            "Linzhan/UniML3D":{
                "README.md":"# UniML3D\n",
                "clips.csv":"clip,object_type,category\nclip1,s1,avian\n",
                "skeletons.csv":"object_type,category,num_joints\ns1,avian,8\n",
            }
        }
        class API:
            def repo_info(self,repo_id,repo_type,files_metadata=False):
                assert repo_type=="dataset"
                return SimpleNamespace(sha="b"*40)
        def downloader(*,repo_id,filename,repo_type,revision,local_dir):
            self.assertEqual(revision,"b"*40)
            self.assertEqual(repo_type,"dataset")
            p=Path(local_dir)/filename;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(entries[repo_id][filename],encoding="utf-8")
            return str(p)
        result=run(CAT,self.root/"out",api=API(),downloader=downloader)
        self.assertEqual(result["pool_repos_intaken"],4)
        self.assertEqual(result["provider_metadata_candidate_rows"],5)
        self.assertEqual(result["semantic_regions_reviewed"],0)
        self.assertEqual(result["source_rights_qualified"],0)
        self.assertFalse(result["training_authorized"])
        self.assertTrue((self.root/"out"/"SUMMARY_V1.json").is_file())

    def test_invalid_source_rights_cannot_enter_catalog(self):
        obj=load_catalog(CAT)
        obj["pool_admission"]=True
        modified=self.root/"bad.json";modified.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,"CATALOG_AUTHORITY"):
            load_catalog(modified)


if __name__=="__main__":
    unittest.main()
