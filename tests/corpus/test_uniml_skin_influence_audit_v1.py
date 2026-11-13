import json
import struct
import tempfile
import unittest
from pathlib import Path
from tools.corpus.truth_foundry.uniml_skin_influence_audit_v1 import glb,decode,inspect


def make_glb(out, joint_id=1, weight=0.75, bad_sparse=False):
    j=bytes([0,joint_id,0,0,1,0,0,0])
    w=struct.pack('<8f',1-weight,weight,0,0,1.,0,0,0)
    ibm=struct.pack('<32f',*([1.,0,0,0,0,1.,0,0,0,0,1.,0,0,0,0,1.]*2))
    binblock=j+w+ibm
    data={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(binblock)}],
          'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':len(j)},
                         {'buffer':0,'byteOffset':len(j),'byteLength':len(w)},
                         {'buffer':0,'byteOffset':len(j)+len(w),'byteLength':len(ibm)}],
          'accessors':[{'bufferView':0,'count':2,'type':'VEC4','componentType':5121},
                       {'bufferView':1,'count':2,'type':'VEC4','componentType':5126},
                       {'bufferView':2,'count':2,'type':'MAT4','componentType':5126}],
          'meshes':[{'primitives':[{'attributes':{'JOINTS_0':0,'WEIGHTS_0':1}}]}],
          'skins':[{'joints':[1,2],'inverseBindMatrices':2}],
          'nodes':[{'mesh':0,'skin':0},{'name':'joint_A'},{'name':'joint_B'}]}
    if bad_sparse:data['accessors'][1]['sparse']={'count':1}
    txt=json.dumps(data).encode('utf-8');txt+=b' ' * (-len(txt)%4)
    binblock+=b'\0' * (-len(binblock)%4)
    blob=struct.pack('<4sII',b'glTF',2,12+8+len(txt)+8+len(binblock))
    blob+=struct.pack('<I4s',len(txt),b'JSON')+txt
    blob+=struct.pack('<I4s',len(binblock),b'BIN\0')+binblock
    Path(out).write_bytes(blob)


class InfluenceCourtTests(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory();self.addCleanup(self.dir.cleanup)
        self.path=Path(self.dir.name)/'subject.glb'

    def test_real_joint_palette_weight_data_preserves_unknown(self):
        make_glb(self.path)
        r,j=inspect(self.path)
        self.assertTrue(r['actual_numeric_skin_values_observed'])
        self.assertEqual(r['skinned_vertex_count'],2)
        self.assertEqual(r['valid_weight_vertex_count'],2)
        self.assertEqual(r['canonical_inverse_bind_count_matches'],1)
        self.assertEqual(r['positive_vertex_joint_assignments'],3)
        self.assertEqual(len(j),2)
        self.assertEqual(j[1]['canonical_node_name_unreviewed'],'joint_B')
        self.assertFalse(r['mira_training_admitted'])
        self.assertFalse(j[1]['semantic_joint_class_approved'])

    def test_invalid_palette_detected_without_fake_valid(self):
        make_glb(self.path,joint_id=9)
        r,_=inspect(self.path)
        self.assertEqual(r['out_of_skin_joint_palette_count'],1)
        self.assertEqual(r['valid_weight_vertex_count'],1)
        self.assertFalse(r['axis_training_admitted'])

    def test_unnormalized_weights_detected(self):
        make_glb(self.path,weight=2.0)
        r,_=inspect(self.path)
        self.assertEqual(r['nonfinite_or_negative_weight_count'],1)
        self.assertEqual(r['valid_weight_vertex_count'],1)
        self.assertFalse(r['mira_training_admitted'])

    def test_sparse_fails_closed(self):
        make_glb(self.path,bad_sparse=True)
        r,_=inspect(self.path)
        self.assertEqual(r['skinned_vertex_count'],0)
        self.assertEqual(r['unsupported_processed_primitive_count'],1)
        self.assertFalse(r['actual_numeric_skin_values_observed'])

    def test_reader_rejects_accessor_boundary_crossing(self):
        make_glb(self.path)
        data,binary=glb(self.path)
        data['bufferViews'][1]['byteLength']=2
        with self.assertRaisesRegex(ValueError,'BUFFER_VIEW_OOB'):
            decode(data,binary,1)

if __name__=='__main__':unittest.main()