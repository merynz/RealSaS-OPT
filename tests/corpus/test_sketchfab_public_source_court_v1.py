import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from sketchfab_public_source_court_v1 import parse_public_record,run,sha

UID='a'*32

def item(uid=UID):
    return dict(partverse_uid=uid,source_sha256_provider_claim='b'*64,
       source_url='https://sketchfab.com/3d-models/'+uid,
       provider_split='PROVIDER_TRAIN_CANDIDATE',object_type_candidate='ANIMAL_CREATURE',
       style_candidate='STYLE_UNVERIFIED')

class FakeSession:
    def __init__(self,status=200):self.status=status
    def get(self,url,timeout,headers):
        assert url=='https://api.sketchfab.com/v3/models/'+UID
        doc={'uid':UID,'license':{'slug':'by','fullName':'Attribution'},'name':'Dragon',
             'isDownloadable':True,'faceCount':123,'animationCount':1,'user':{'uid':'author'}}
        class Response:
            status_code=self.status
            def json(self):return doc
        return Response()

class TestPublicCourt(unittest.TestCase):
    def test_public_license_is_not_training_pass(self):
        r=parse_public_record(item(),200,{'uid':UID,'license':{'slug':'by-nc'},
                                               'isDownloadable':True,'name':'dragon'})
        self.assertEqual(r['license_slug'],'by-nc')
        self.assertEqual(r['commercial_training_license_court'],'UNKNOWN__NEEDS_ORIGINAL_LICENSE_REVIEW')
        self.assertFalse(r['head_training_authorized'])
        self.assertEqual(r['source_original_bytes_sha_court'],'UNKNOWN')
    def test_unknown_model_never_becomes_negative_truth(self):
        r=parse_public_record(item(),404,None)
        self.assertEqual(r['api_status'],'PUBLIC_LOOKUP_FAILED_OR_RECORD_MISMATCH')
        self.assertEqual(r['source_visual_morphology_court'],'UNKNOWN')
    def test_real_queue_bound_to_sha_and_source_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);queue=root/'queue.csv'
            with queue.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(item()));w.writeheader();w.writerow(item())
            s=run(queue,root/'out',expected_queue_sha=sha(queue),max_requests=2,session=FakeSession())
            self.assertEqual(s['source_public_model_metadata_found'],1)
            self.assertEqual(s['independently_verified_commercial_training_rights'],0)
            self.assertEqual(s['semantic_head_training_pass'],0)
            self.assertFalse(s['training_authorized'])
            with self.assertRaisesRegex(ValueError,'SOURCE_QUEUE_SHA_CHANGED'):
                run(queue,root/'out',expected_queue_sha='0'*64,session=FakeSession())
            with queue.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(item()));w.writeheader();x=item();x['source_url']='https://evil.example/123';w.writerow(x)
            with self.assertRaisesRegex(ValueError,'SOURCE_IDENTIFIER_URL_MISMATCH'):
                run(queue,root/'out',session=FakeSession())

if __name__=='__main__':unittest.main()