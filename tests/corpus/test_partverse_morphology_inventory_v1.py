"""Candidate subject filtering and actual PartVerse face/GLB evidence; no semantic PASS."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.corpus.truth_foundry.partverse_morphology_triage_v1 import (
    classify, read_caption, run as triage_run, source_uid
)
from tools.corpus.truth_foundry.partverse_part_geometry_inventory_v1 import (
    one_source, run as geometry_run
)


class PartVerseMorphologyAndGeometryTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_weak_text_is_not_semantic_or_morphology_pass(self):
        a=classify("A stylized wolf creature with wings and claws.")
        self.assertEqual(a[0],"ANIMAL_CREATURE")
        self.assertEqual(a[1],"STYLIZED_CLAIM")
        b=classify("A wooden sofa with cat-eared dragon ornament.")
        self.assertEqual(b[0],"FURNITURE_ARCHITECTURE")
        self.assertEqual(b[2],"NONCHARACTER_DEFER_OR_REVIEW")
        self.assertEqual(read_caption("")[1],"MISSING")
        self.assertEqual(source_uid("https://sketchfab.com/3d-models/"+"a"*32),"a"*32)
        self.assertEqual(source_uid("https://evil.example/3d-models/"+"b"*32),"")

    def test_split_court_blocks_val_and_furniture(self):
        ids=["a"*32,"b"*32,"c"*32,"d"*32,"e"*32]
        rows=[
            ("A stylized cartoon dragon creature.","a"*64),
            ("A stylized robot with arms and legs.","b"*64),
            ("A photorealistic human man with realistic skin.","c"*64),
            ("A wooden chair with cushions.","d"*64),
            ("A low-poly medieval knight game character.","e"*64),
        ]
        meta=self.root/"metadata.csv"
        with meta.open("w",newline="",encoding="utf-8") as fp:
            w=csv.writer(fp);w.writerow(["sha256","file_identifier","aesthetic_score","captions"])
            for uid,(caption,sh) in zip(ids,rows):
                w.writerow([sh,"https://sketchfab.com/3d-models/"+uid,"6.4",json.dumps([caption])])
        train=self.root/"train.csv"; val=self.root/"val.csv"
        train.write_text("part_id\n"+"\n".join(ids[:4])+"\n",encoding="utf-8")
        val.write_text("part_id\n"+ids[-1]+"\n",encoding="utf-8")
        result=triage_run(meta,self.root/"triage",max_selected=2500,
                          provider_train=train,provider_val=val)
        self.assertEqual(result["record_count"],5)
        self.assertEqual(result["provider_split_counts"]["PROVIDER_VAL_SEALED"],1)
        self.assertEqual(result["selected_with_protected_val_leak"],0)
        self.assertEqual(result["source_rights_pass"],0)
        self.assertEqual(result["visual_morphology_court_pass"],0)
        with (self.root/"triage"/"PARTVERSE_CANDIDATE_DOWNLOAD_QUEUE_V1.csv").open() as f:
            queue=list(csv.DictReader(f))
        self.assertTrue(all(r["provider_split"]=="PROVIDER_TRAIN_CANDIDATE" for r in queue))
        self.assertNotIn("c"*32,{r["partverse_uid"] for r in queue})
        self.assertNotIn("d"*32,{r["partverse_uid"] for r in queue})
        self.assertNotIn("e"*32,{r["partverse_uid"] for r in queue})
        self.assertFalse(any(r["head_training_admitted"]=="True" for r in queue))

    def test_face0_unlisted_vs_geometry_count_independent(self):
        import trimesh
        uid="a"*32
        src=self.root/"staging"/"sampled_annotation_data"/uid
        src.mkdir(parents=True)
        mesh=trimesh.creation.box()
        mesh.export(src/(uid+"_segmented.glb"))
        lab={str(i):(0 if i<4 else 1) for i in range(len(mesh.faces))}
        (src/(uid+"_face2label.json")).write_text(json.dumps(lab))
        (src/(uid+"_info.json")).write_text(json.dumps({
            "ordered_face_label":[1],
            "ordered_faceid":[list(range(4,len(mesh.faces)))],
        }))
        row={"source_sha256_provider_claim":"a"*64,
             "object_type_candidate":"HUMANOID_CHARACTER",
             "style_candidate":"STYLE_UNVERIFIED",
             "provider_split":"PROVIDER_TRAIN_CANDIDATE"}
        result=one_source(src,row)
        self.assertEqual(result["triangle_face_count"],len(mesh.faces))
        self.assertTrue(result["face_count_equal_triangles"])
        self.assertEqual(result["part_id_count_including_zero"],2)
        self.assertTrue(result["part_info_face_ids_subset_consistent"])
        self.assertEqual(result["face_count_label0_unlisted_by_info"],4)
        self.assertEqual(result["original_source_sha_bytes_parity_court"],"UNKNOWN")
        self.assertEqual(result["anatomical_part_concept_court"],"UNKNOWN")
        self.assertFalse(result["semantic_head_training_admitted"])
        queue=self.root/"queue.csv"
        with queue.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=["partverse_uid","source_sha256_provider_claim",
                                         "object_type_candidate","style_candidate","provider_split"])
            w.writeheader();w.writerow({"partverse_uid":uid,**row})
        summary=geometry_run(self.root/"staging",queue,self.root/"out",max_bundle_uids=100)
        self.assertEqual(summary["selected_candidates"],1)
        self.assertEqual(summary["processed_glb_face_count_contiguity_matches"],1)
        self.assertEqual(summary["face_count_missing_in_provider_info_despite_face2label"],4)
        self.assertEqual(summary["bundle_count"],1)
        self.assertEqual(summary["head_semantic_training_pass"],0)
        self.assertTrue((self.root/"out"/"bundle_shards"/"PARTVERSE_SOURCE_ANNOTATIONS_000.tar.gz").exists())

    def test_refuse_heldout_queue_for_geometry_intake(self):
        q=self.root/"q.csv"
        q.write_text("partverse_uid,provider_split\n"+"a"*32+",PROVIDER_VAL_SEALED\n")
        with self.assertRaisesRegex(ValueError,"HELDOUT_SPLIT"):
            geometry_run(self.root,q,self.root/"out")


if __name__=="__main__":
    unittest.main()
