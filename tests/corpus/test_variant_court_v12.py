"""Negative controls for legacy imported-variant evidence (no student leakage)."""
import tempfile, pathlib, json, unittest, sys, csv
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/"tools/corpus/truth_foundry/evidence_v12"))
from variant_court_v12 import parse_subject

class VariantCourtTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=pathlib.Path(self.temp.name)
        self.row={'source_key':'MASTER3993:asset_x','candidate_id':'provider:abcd','provider':'provider','original_sha_claim':'abcd'}
        d=self.base
        (d/'admission.json').write_text(json.dumps({'candidate_id':'provider:abcd','canonical_asset_id':'asset_x','source_identity':{'source_sha256':'abcd','source_url':'https://example.com/original'},'legal':{'attribution_required':True,'decision':'PASS'},'decision':'ADMIT'}))
        (d/'source_structure.json').write_text(json.dumps({'mesh_objects':[{'name':'Head','vertex_start':0,'vertex_count':5,'face_count':3}], 'bone_names':['J0'],'bone_count':1,'armature':'r'}))
        (d/'technical_audit.json').write_text(json.dumps({'skin_metrics':{'row_sum_mean':1.,'row_sum_p99_abs_err':0.,'zero_row_fraction':0.},'rig_metrics':{'bone_count':1},'skin_pass':True}))
        (d/'normalization_meta.json').write_text(json.dumps({'center':[0,0,0],'scale':1.0}))
    def test_exact_match_is_legacy_observation_only(self):
        s,meshes,bones=parse_subject(self.row,self.base)
        self.assertEqual(s['inspection_state'],'FOUR_DOCUMENTS_OBSERVED');self.assertFalse(s['source_original_bytes_rehashed']);self.assertFalse(s['training_authorized'])
        self.assertEqual(s['historical_variant_vertex_count'],5);self.assertEqual(len(meshes),1);self.assertEqual(len(bones),1)
        self.assertEqual(set(s['facet_candidates']),{'original_asset_provider_locator','attribution_obligations','source_vertex_face_count','source_name_only_teacher_hint_isolation','skin_weight_normalization','zero_weight_vertices_and_unbound_mesh','canonical_frame_transform'})
    def test_identity_drift_quarantines_all_facets(self):
        p=self.base/'admission.json';j=json.loads(p.read_text());j['candidate_id']='provider:other';p.write_text(json.dumps(j))
        s,_,_=parse_subject(self.row,self.base)
        self.assertEqual(s['inspection_state'],'CONFLICT_QUARANTINED');self.assertEqual(s['facet_candidates'],{})
    def test_bone_count_drift_quarantines(self):
        p=self.base/'technical_audit.json';j=json.loads(p.read_text());j['rig_metrics']['bone_count']=2;p.write_text(json.dumps(j))
        s,_,_=parse_subject(self.row,self.base)
        self.assertEqual(s['inspection_state'],'CONFLICT_QUARANTINED');self.assertIn('TECHNICAL_VS_STRUCTURE_BONE_COUNT_DRIFT',s['diagnostic_issues'])
    def test_no_variant_is_not_partial(self):
        for f in list(self.base.iterdir()):f.unlink()
        s,meshes,bones=parse_subject(self.row,self.base)
        self.assertEqual(s['inspection_state'],'NO_VARIANT_RECEIPT_FOUND');self.assertEqual((meshes,bones),([],[]))

if __name__=='__main__':unittest.main()