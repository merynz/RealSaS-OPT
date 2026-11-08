"""Small synthetic tar.gz proves PartVerse archive selection is NOT label admission."""
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.corpus.truth_foundry.partverse_archive_probe_v1 import (
    inspect_tar_stream, parse_face2label, safe_archive_path
)


def tar_bytes(files):
    fp=io.BytesIO()
    with tarfile.open(fileobj=fp,mode="w:gz") as archive:
        for name,b in files.items():
            info=tarfile.TarInfo(name);info.size=len(b)
            archive.addfile(info,io.BytesIO(b))
    fp.seek(0)
    return fp


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)

    def test_face2label_valid_multi_instance(self):
        result=parse_face2label(json.dumps({"0":0,"1":0,"2":1,"3":2}).encode())
        self.assertEqual(result["labeled_faces"],4)
        self.assertEqual(result["unique_part_ids"],3)

    def test_malformed_labels_are_fail_closed(self):
        with self.assertRaises(ValueError):
            parse_face2label(json.dumps({"0":"wing","1":"head"}).encode())
        with self.assertRaises(ValueError):
            parse_face2label(json.dumps({"-1":1}).encode())

    def test_archive_path_traversal_rejected(self):
        self.assertFalse(safe_archive_path("../evil"))
        self.assertFalse(safe_archive_path("/absolute/file"))
        self.assertTrue(safe_archive_path("anno_infos/abc123456789/abc123456789_face2label.json"))

    def test_sampled_archive_is_unqualified_and_bounded(self):
        uid="abc123456789abcdef"
        f=tar_bytes({
            f"anno_infos/{uid}/{uid}_face2label.json":b'{"0":1,"1":1,"2":2}',
            f"anno_infos/{uid}/{uid}_segmented.glb":b"GLB-example-bytes",
            f"anno_infos/{uid}/{uid}_info.json":b'{"bboxes":[]}',
            f"anno_infos/{uid}/unimportant.txt":b"ignored",
            f"../out_of_root.bin":b"do not copy"
        })
        s=inspect_tar_stream(f,self.out,max_members=100,max_glbs=1)
        self.assertEqual(s["face2label_records_parsed"],1)
        self.assertEqual(s["sampled_source_uids"],1)
        self.assertEqual(s["source_face_semantic_pass"],0)
        self.assertEqual(s["drawable_layer_pass"],0)
        self.assertFalse(s["training_authorized"])
        self.assertFalse(s["full_archive_processed"])
        self.assertFalse((self.out/"out_of_root.bin").exists())
        self.assertTrue((self.out/"sampled_annotation_data"/uid/(uid+"_face2label.json")).exists())
        self.assertEqual(s["statistics"]["unsafe_paths_skipped"],1)


if __name__=="__main__":
    unittest.main()
