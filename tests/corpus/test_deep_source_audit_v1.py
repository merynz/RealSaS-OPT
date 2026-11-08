"""Synthetic fail-closed regression; no private source assets or rights claimed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from tools.corpus.truth_foundry.deep_source_audit_v1 import (
    MORPH_CLASS, STAGES, NUMERIC_FIELDS, analyze_one, _mesh_modularity, fail, run
)


class DeepSourceAuditTest(unittest.TestCase):
    def test_morphology_classification_is_approval_only(self):
        for name in ('HUMAN','HUMANOID_NONHUMAN','QUADRUPED_MAMMAL','DINOSAUR',
                     'ARTHROPOD_MULTILEGGED','TANK_GROUND_VEHICLE','AIRCRAFT',
                     'PROP_TOOL_WEAPON','UNKNOWN'):
            self.assertIn(name, MORPH_CLASS)
        self.assertGreaterEqual(len(STAGES), 12)

    def test_one_mesh_many_islands_not_drawable(self):
        self.assertIn('NOT_DRAWABLES', _mesh_modularity(1, 44))
        self.assertIn('NOT_ANATOMY', _mesh_modularity(1, 1))
        self.assertIn('MULTI_OBJECT', _mesh_modularity(2, 6))
        self.assertEqual(_mesh_modularity(0, 0),'NO_IMPORTED_MESH')

    def sample(self):
        aid='asset_test000'; src='a'*64; candidate='test:source'; ext='.glb'
        counts={k:0 for k in NUMERIC_FIELDS}
        counts.update({'source_scene_node_count':1,'mesh_object_count':1,
                       'mesh_disconnected_components':3,'polygon_count':2,
                       'vertex_count':6,'mesh_boundary_edges':6,
                       'material_count':1,'skin_rows_sampled':0})
        full={
            'schema':'RealSaS.Corpus4008Wave1SourceRigSceneObservation.v1',
            'asset_id':aid,'source_sha256':src,'candidate_id':candidate,
            'source_format':ext,'numeric_census':counts,
            'training_authorized':False,'semantic_region_truth_courted':False,
            'original_source_scene':{
                'imported_scene':{
                    'imported_nodes':[{'object_type':'MESH','name_teacher_only':'human_LEFT_ARM',
                                       'parent_type':'OBJECT','parent_node_index':None,'modifiers':[]}],
                    'meshes':[{'vertex_count':6,'polygon_count':2,
                               'material_slots':1,'scene_node_index':0,'vertex_groups':0,
                               'skin_rows_sampled':0,'skin_rows_without_any_groups':6,
                               'index_topology':{'source_index_component_count':3,
                                                 'boundary_edge_count':6,'nonmanifold_edge_count':0}}],
                    'armatures':[]},
                'material_and_texture_inventory':{'images_teacher_only':[]}},
            'imported_rig_and_animation_control':{'actions':[],'imported_constraint_type_counts':{},'imported_camera_count':0}
        }
        summary={k:full[k] for k in ('schema','asset_id','source_sha256','candidate_id','source_format','numeric_census')}
        summary.update({'training_authorized':False,'commercial_ml_rights_courted':False,
                        'receipt_gz_sha256':'b'*64})
        row={'asset_id':aid,'source_sha256':src,'candidate_id':candidate,
             'source_format':ext,'historical_split':'FIT','census_summary_sha256':'c'*64,
             **{k:str(v) for k,v in counts.items()}}
        return row,summary,full

    def test_rich_structure_is_observation_not_semantic_or_drawable(self):
        row,s,f=self.sample()
        detail,tasks=analyze_one(row,s,f,'b'*64,'c'*64)
        self.assertEqual(detail['counts']['mesh_disconnected_components'],3)
        self.assertEqual(detail['structural_class'],'ONE_OBJECT_MULTI_DISCONNECTED_ISLANDS__NOT_DRAWABLES')
        self.assertIn('human_LEFT_ARM',detail['privileged_source_metadata_teacher_only']['scene_node_names'])
        self.assertEqual(detail['morphology'],'UNKNOWN_NEEDS_VISUAL_SOURCE_COURT')
        self.assertIsNone(detail['drawable_count'])
        self.assertFalse(detail['training_authorized'])
        self.assertTrue(all(r['independent_court_state']=='UNKNOWN' for r in tasks))
        self.assertIn('connected_islands_semantic_versus_drawable_test',{t['stage'] for t in tasks})

    def test_imported_ik_zero_does_not_mean_original_lacks_controls(self):
        row,s,f=self.sample()
        detail,_=analyze_one(row,s,f,'b'*64,'c'*64)
        self.assertEqual(detail['counts']['observed_ik_constraint_count'],0)
        self.assertIn('UNKNOWN',detail['native_control_functionality'])

    def test_tampered_full_receipt_source_refused(self):
        row,s,f=self.sample();f['source_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'SUMMARY_FULL_MISMATCH'):
            analyze_one(row,s,f,'b'*64,'c'*64)

    def test_tampered_summary_counts_refused(self):
        row,s,f=self.sample();s['numeric_census']=dict(s['numeric_census']);s['numeric_census']['vertex_count']=7
        with self.assertRaisesRegex(ValueError,'CENSUS_SOURCE_MISMATCH'):
            analyze_one(row,s,f,'b'*64,'c'*64)

    def test_tampered_gzip_sha_refused(self):
        row,s,f=self.sample()
        with self.assertRaisesRegex(ValueError,'GZIP_SHA'):
            analyze_one(row,s,f,'e'*64,'c'*64)

    def test_must_have_real_full_master_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'physical.csv').write_text('asset_id,census_state\nasset_foo,IMPORTED_NUMERIC_OBSERVATION_ONLY\n')
            with self.assertRaisesRegex(ValueError,'MASTER_3993_PHYSICAL_ROSTER'):
                run(root/'physical.csv',tuple(root / f'w{i}' for i in range(4)),root/'out')


if __name__ == '__main__':
    unittest.main()