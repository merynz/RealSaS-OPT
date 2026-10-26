import hashlib
from pathlib import Path
import tempfile
import unittest
from tools.ops.inspect_truth_inputs_v1 import inspect


class InspectionTests(unittest.TestCase):
    def test_recovers_exact_bytes_never_same_filename_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); authority=root/'authority'; platform=root/'platform'
            source=authority/'teachers'; source.mkdir(parents=True)
            (source/'a').write_bytes(b'bad')
            digest=hashlib.sha256(b'yes').hexdigest()
            (source/digest).write_bytes(b'yes')
            req={'schema':'RealSaS.TruthInputInspectionRequest.v1','read_only':True,
                 'search_roots':[{'root':'authority','relative':'teachers'}],
                 'artifacts':[{'file_name':'a','sha256':digest,'size_bytes':3}]}
            result=inspect(req,authority_root=authority,platform_root=platform,output=root/'out')
            self.assertEqual(result['status'],'RECOVERED')
            self.assertEqual(len(result['mismatched_candidates']),1)
            self.assertFalse(result['qualification_minted'])
            self.assertEqual((root/'out'/result['recovered'][0]['recovered_relative_path']).read_bytes(),b'yes')
            (source/digest).unlink()
            (source/digest).symlink_to(root/'out'/result['recovered'][0]['recovered_relative_path'])
            result=inspect(req,authority_root=authority,platform_root=platform,output=root/'out2')
            self.assertEqual(result['status'],'INCOMPLETE')

    def test_search_path_escape_and_write_request_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            req={'schema':'RealSaS.TruthInputInspectionRequest.v1','read_only':True,
                 'search_roots':[{'root':'authority','relative':'../outside'}],'artifacts':[]}
            with self.assertRaisesRegex(ValueError,'ROOT_ESCAPE'):
                inspect(req,authority_root=root/'a',platform_root=root/'p',output=root/'o')
            req['read_only']=False
            with self.assertRaisesRegex(ValueError,'READ_ONLY'):
                inspect(req,authority_root=root/'a',platform_root=root/'p',output=root/'o')
