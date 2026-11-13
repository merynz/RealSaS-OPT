"""Synthetic repeatable PartVerse processed-part evidence contract."""
import io
import csv
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
import sys

import trimesh
sys.path.insert(0,str(Path(__file__).parent))
from tools.corpus.truth_foundry.semantic_part_geometry_v1 import inspect, run, sha_file

UID='a'*32


def fixture():
    m=trimesh.creation.box()
    buff=io.BytesIO()
    m.export(buff,file_type='glb')
    labels={str(i):(0 if i<4 else 1) for i in range(len(m.faces))}
    info={'ordered_face_label':[1],
          'ordered_faceid':[list(range(4,len(m.faces)))]}
    return buff.getvalue(),json.dumps(labels).encode(),json.dumps(info).encode(),len(m.faces)


class PartEvidenceTest(unittest.TestCase):
    def test_per_region_area_and_label_zero_are_observations_not_truth(self):
        glb,labels,info,faces=fixture()
        a,parts,edges=inspect(glb,labels,info,UID,{'provider_split':'PROVIDER_TRAIN_CANDIDATE'},'0'*64)
        self.assertEqual(a['processed_glb_triangle_count'],faces)
        self.assertEqual(a['processed_glb_part_count'],2)
        self.assertEqual(a['processed_info_unlisted_face_count'],4)
        self.assertTrue(a['processed_info_listed_parts_exact_face_sets_match'])
        self.assertEqual(sum(p['part_triangle_count'] for p in parts),faces)
        self.assertAlmostEqual(sum(p['part_area_fraction'] for p in parts),1.0)
        self.assertTrue(all(p['part_anatomical_or_functional_concept']=='UNKNOWN' for p in parts))
        self.assertTrue(all(not p['training_mask_valid'] for p in parts))
        self.assertFalse(a['training_authorized'])
        self.assertTrue(all(not x['training_mask_valid'] for x in edges))

    def test_partial_or_corrupt_face_map_is_rejected(self):
        glb,labels,info,faces=fixture()
        obj=json.loads(labels);obj.pop('0')
        with self.assertRaisesRegex(ValueError,'FACE_COUNT_AND_CONTIGUITY_MISMATCH'):
            inspect(glb,json.dumps(obj).encode(),info,UID,{},'0'*64)
        obj=json.loads(labels);obj['0']=-1
        with self.assertRaisesRegex(ValueError,'NEGATIVE_FACE_OR_LABEL'):
            inspect(glb,json.dumps(obj).encode(),info,UID,{},'0'*64)

    def test_archive_sha_and_explicit_missing_receipt(self):
        glb,labels,info,faces=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'bundles').mkdir()
            path=root/'bundles/PARTVERSE_SOURCE_ANNOTATIONS_000.tar.gz'
            with tarfile.open(path,'w:gz') as tf:
                for ending,data in [('_segmented.glb',glb),('_face2label.json',labels),('_info.json',info)]:
                    name='source_annotation/'+UID+'/'+UID+ending
                    t=tarfile.TarInfo(name);t.size=len(data)
                    tf.addfile(t,io.BytesIO(data))
            queue=root/'selection.csv'
            with queue.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['partverse_uid','provider_split']);w.writeheader()
                w.writerow({'partverse_uid':UID,'provider_split':'PROVIDER_TRAIN_CANDIDATE'})
                w.writerow({'partverse_uid':'b'*32,'provider_split':'PROVIDER_TRAIN_CANDIDATE'})
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'schema':'RealSaS.PartVersePartGeometryInventory.v1',
                'input_queue_sha256':sha_file(queue),
                'bundles':[{'file':path.name,'source_uids':1,'sha256':sha_file(path)}],
                'missing_selected_source_uids':1,'staged_source_uids':1}))
            with self.assertRaisesRegex(ValueError,'UNRECEIPTED_MISSING_SOURCES'):
                run(root/'bundles',manifest,queue,root/'out')
            result=run(root/'bundles',manifest,queue,root/'out',allow_receipted_missing=True)
            self.assertEqual(result['processed_mesh_label_geometry_observation_count'],1)
            self.assertEqual(result['missing_provider_records'],1)
            self.assertEqual(result['per_part_source_region_observations'],2)
            self.assertFalse(result['training_authorized'])
            self.assertEqual(result['anatomical_semantic_concept_pass'],0)
            wrong=json.loads(manifest.read_text());wrong['bundles'][0]['sha256']='0'*64
            manifest.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'BUNDLE_SHA_MISMATCH'):
                run(root/'bundles',manifest,queue,root/'out',allow_receipted_missing=True)

    def test_protected_provider_split_blocks_knowledge_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'q.csv';m=root/'manifest.json'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['partverse_uid','provider_split']);w.writeheader();
                w.writerow({'partverse_uid':UID,'provider_split':'PROVIDER_VAL_SEALED'})
            m.write_text(json.dumps({'schema':'RealSaS.PartVersePartGeometryInventory.v1'}))
            with self.assertRaisesRegex(ValueError,'PROTECTED_PROVIDER_SPLIT'):
                run(root,m,p,root/'out')

if __name__=='__main__': unittest.main()