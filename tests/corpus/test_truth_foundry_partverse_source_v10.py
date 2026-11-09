import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2] / 'tools' / 'corpus' / 'truth_foundry' / 'evidence_v10'
SPEC = importlib.util.spec_from_file_location('v10_source_court', ROOT / 'truth_foundry_partverse_source_court_v10.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

class GraphMathTests(unittest.TestCase):
    def test_isolated(self):
        self.assertEqual(M.connected(4,[]),(4,1))
    def test_chain(self):
        self.assertEqual(M.connected(5,[(0,1),(1,2)]),(3,3))
    def test_cycle(self):
        self.assertEqual(M.connected(4,[(0,1),(1,2),(2,0)]),(2,3))
    def test_no_nodes(self):
        self.assertEqual(M.connected(0,[]),(0,0))
    def test_false_court_boolean_fails_closed(self):
        with self.assertRaises(AssertionError):
            M.flag_for({'training_mask_valid':'True'},'training_mask_valid',{'False'})
    def test_sha_digest(self):
        import hashlib,tempfile
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'geometry')
            f.flush()
            self.assertEqual(M.digest(pathlib.Path(f.name)),hashlib.sha256(b'geometry').hexdigest())

if __name__=='__main__': unittest.main()