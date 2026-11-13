"""UniML3D bounded downloadable canonical source is not independently admitted truth."""
import csv
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.corpus.truth_foundry.selective_glb_intake_v1 import (
    RARE, candidates_from_queue, glb_structure, run, stream_asset, sha,
)


def make_glb():
    doc={'asset':{'version':'2.0'},
         'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1}]}],
         'accessors':[{'count':4},{'count':6}],
         'skins':[{'joints':[0,1,2]}],
         'nodes':[{}, {}, {}],
         'materials':[{}],
         'images':[{'bufferView':0, 'mimeType':'image/png'}],
         'textures':[{'source':0}]}
    js=json.dumps(doc).encode();js+=b' ' *((4-len(js)%4)%4)
    return struct.pack('<4sIIII',b'glTF',2,20+len(js),len(js),0x4e4f534a)+js


class FakeResponse:
    def __init__(self, payload):
        self.content=payload;self.headers={'Content-Length':str(len(payload))}
        self.status_code=200
    def __enter__(self):return self
    def __exit__(self,*_):return False
    def raise_for_status(self):return None
    def iter_content(self,n):yield self.content


class FakeSession:
    def __init__(self,payload):self.payload=payload
    def get(self,url,stream,timeout,headers):
        assert '/resolve/' in url and url.endswith('.glb')
        return FakeResponse(self.payload)


class SelectiveUniML3DTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.queue=self.root/'q.csv'
        with self.queue.open('w',newline='') as fp:
            w=csv.DictWriter(fp,fieldnames=['dataset','object_type','category','num_joints','num_clips',
                                           'source_rights_court','training_authorized'])
            w.writeheader()
            for i in range(2500):
                w.writerow({'dataset':'objaverse','object_type':f'fixture_{i:07d}_glb',
                  'category':RARE[i%len(RARE)] if i<180 else 'bipedal',
                  'num_joints':4,'num_clips':1,'source_rights_court':'UNKNOWN','training_authorized':False})

    def test_rare_before_bipedal_and_balance(self):
        selected=candidates_from_queue(self.queue,72,sha(self.queue))
        self.assertTrue(all(r['category'] in RARE for r in selected))
        from collections import Counter
        groups=Counter(r['category'] for r in selected)
        self.assertTrue(all(groups[cat]==12 for cat in RARE))
        self.assertEqual(len({r['object_type'] for r in selected}),72)

    def test_source_sha_tamper_protected(self):
        with self.assertRaisesRegex(ValueError,'FROZEN_SOURCE_QUEUE_SHA_MISMATCH'):
            candidates_from_queue(self.queue,3,'0'*64)

    def test_glb_native_structural_fields_not_semantic_labels(self):
        p=self.root/'asset.glb';p.write_bytes(make_glb())
        struct_info=glb_structure(p)
        self.assertTrue(struct_info['glb_valid'])
        self.assertEqual(struct_info['glb_mesh_count'],1)
        self.assertEqual(struct_info['glb_triangle_indices_count'],2)
        self.assertEqual(struct_info['glb_skin_count'],1)
        self.assertEqual(struct_info['glb_skin_joint_count'],3)
        self.assertEqual(struct_info['glb_embedded_image_count'],1)

    def test_small_true_download_is_never_admitted(self):
        summary=run(self.queue,self.root/'out',revision='b'*40,
                    expected_sha=sha(self.queue),max_selected=6,max_total_mib=3,max_file_mib=1,
                    session=FakeSession(make_glb()))
        self.assertEqual(summary['selected_subjects'],6)
        self.assertEqual(summary['glb_structure_pass_count'],6)
        self.assertEqual(summary['rights_qualified_count'],0)
        self.assertEqual(summary['semantic_face_labels_qualified'],0)
        self.assertFalse(summary['training_authorized'])
        with (self.root/'out'/'UNIML3D_SOURCE_GLB_EVIDENCE_V1.csv').open() as fp:
            rows=list(csv.DictReader(fp))
        self.assertTrue(all(r['download_sha256'] for r in rows))
        self.assertTrue(all(r['source_original_byte_sha_verified']=='False' for r in rows))

    def test_per_file_cap_does_not_write_partial(self):
        p=self.root/'a.glb'
        status=stream_asset('https://huggingface.co/datasets/Linzhan/UniML3D/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/canonical_assets/objaverse/test.glb',p,20,
                            session=FakeSession(make_glb()))
        self.assertEqual(status,'OVERSIZE_HEADER')
        self.assertFalse(p.exists())
        self.assertFalse(p.with_suffix('.part').exists())

if __name__=='__main__':unittest.main()