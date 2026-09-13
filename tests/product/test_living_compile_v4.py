from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path

from product.living_compile.authoring import save_user_layer
from product.living_compile.common import LivingCompileError
from product.living_compile.runtime import runtime_clips, runtime_frame
from product.living_compile.scene import build_scene
from product.living_compile.server import LivingCompileApplication


def writej(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def fixture(root: Path):
    nodes=[]
    pts=[(-.4,-.4,0),(.4,-.4,0),(0,.4,0)]
    for i,p in enumerate(pts):
        nodes.append({'surface_id':f'S{i}','P':p,'raster_bindings':[[v,[100+300*(p[0]+.5),100+300*(p[1]+.5)]] for v in range(8)]})
    joints=[
      {'canonical_joint_id':'ROOT','position':[0,-.2,0],'parent_canonical_id':None,'support_surface_ids':['S0','S1']},
      {'canonical_joint_id':'CHILD','position':[0,.2,0],'parent_canonical_id':'ROOT','support_surface_ids':['S2']},
    ]
    dirs=[]
    for v in range(8):
        vertices=[{'canonical_mesh_vertex_id':f'M{i}','P':p,'support_binding':{'coefficients':[[f'S{i}',1.0]]}} for i,p in enumerate(pts)]
        rows=[{'canonical_mesh_vertex_id':f'M{i}','influences':[['ROOT',.5],['CHILD',.5]]} for i in range(3)]
        dirs.append({'view_index':v,'camera_binding_hash':f'cam{v}','components':[{'component_id':'BODY','mesh':{'vertices':vertices,'faces':[['M0','M1','M2']]},'mesh_skin':{'rows':rows,'transfer_method':'TEST'},'component_state_hash':f'c{v}'}]})
    product={
        'schema_version':'RealSaS.CanonicalPuppetGraph.v3','product_state_hash':'p'*64,
        'representation_class':'DIRECTIONAL_2D_2P5D_PUPPET','full_3d_reconstruction_authority':False,
        'mechanical_state':{
            'surface':{'surface_nodes':nodes,'geometry_lineage_hash':'s'*64},
            'skeleton':{'joints':joints,'skeleton_lineage_hash':'g'*64},
        },
        'directional_renderables':{'directions':dirs,'directional_visual_state_hash':'v'*64},
        'motion_state':{'clips':[{'clip_id':'IDLE','clip_kind':'idle','duration_sec':1.0,'loop':True,'metadata':{}}],}
    }
    proof={'schema_version':'RealSaS.ProductProofBundleIR.v1','source_product_state_hash':'p'*64,'overall_status':'PASS','proof_bundle_hash':'q'*64,'domain_reports':[{'proof_domain':'MOTION','status':'PASS','failure_signatures':[],'owner_attribution':[],'domain_proof_hash':'m'*64}]}
    bake={'schema_version':'RealSaS.QualificationOwnedMotionBakeIR.v1','source_product_state_hash':'p'*64,'proof_plan_hash':'r'*64,'clip_id':'IDLE','duration_seconds':1.0,'fps':1.0,'loop':True,'bake_hash':'b'*64,'frames':[{'time_seconds':0.0,'mesh_vertices_by_id':[['V0:BODY',[[130,130],[370,130],[250,370]]]],'render_order_by_view':[['V0',['V0:BODY']]]},{'time_seconds':1.0,'mesh_vertices_by_id':[['V0:BODY',[[140,130],[380,130],[260,370]]]],'render_order_by_view':[['V0',['V0:BODY']]]}]}
    projections=[{
        'source_product_state_hash':'p'*64,'view_index':v,'camera_binding_hash':f'cam{v}','surface_lineage_hash':'s'*64,
        'affine_rows':[[300,0,0,250],[0,300,0,250]],'source_surface_ids':['S0','S1','S2'],'correspondence_count':3,
        'affine_rank':3,'rms_residual_px':0.0,'p95_residual_px':0.0,'max_residual_px':0.0,'raster_span_px':1.0,
        'p95_residual01':0.0,'max_residual01':0.0,'max_joint_affine_hull_residual01':0.0,'policy_hash':'h'*64,
        'qualification_report':{'status':'PASS'},'projection_binding_hash':f'{v:064x}','schema_version':'RealSaS.DirectionalViewProjectionBindingIR.v1'
    } for v in range(8)]
    pivots=[]
    for v in range(8):
        pivots.extend([
            {'view_index':v,'canonical_joint_id':'ROOT','raster_xy':[250,190],'projection_binding_hash':f'{v:064x}','schema_version':'RealSaS.DirectionalJointPivotIR.v1'},
            {'view_index':v,'canonical_joint_id':'CHILD','raster_xy':[250,310],'projection_binding_hash':f'{v:064x}','schema_version':'RealSaS.DirectionalJointPivotIR.v1'},
        ])
    directional_binding={
        'source_product_state_hash':'p'*64,'surface_lineage_hash':'s'*64,'skeleton_lineage_hash':'g'*64,
        'directional_visual_state_hash':'v'*64,'projections':projections,'joint_pivots':pivots,
        'binding_set_hash':'d'*64,'policy_hash':'h'*64,'schema_version':'RealSaS.DirectionalJointViewBindingSetIR.v1',
        'metadata':{'authority':'COMPILER_QUALIFIED_DERIVED_BINDING'},
    }
    writej(root/'puppet/canonical_puppet_graph_v3.json',product)
    writej(root/'proof/product_proof_bundle_ir.json',proof)
    writej(root/'proof/motion_bakes/IDLE.json',bake)
    writej(root/'renderables/directional_joint_view_binding_set_ir.json',directional_binding)


class LivingCompileV4Tests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(); self.root=Path(self.t.name)/'bundle'; self.root.mkdir(); fixture(self.root)
    def tearDown(self): self.t.cleanup()
    def test_scene_is_v4_and_raster_derived(self):
        s=build_scene(self.root)
        self.assertEqual(s['schema_version'],'RealSaS.LivingCompileScene.v4')
        self.assertEqual(s['counts']['views'],8)
        self.assertEqual(s['puppet']['meshes'][0]['vertices'][0],[130.0,130.0])
        self.assertFalse(s['authority']['canonical_product_mutated_by_ui'])
    def test_rig_overlay_uses_qualified_directional_pivot_not_support_centroid(self):
        s=build_scene(self.root)
        root=next(x for x in s['puppet']['rig']['controls'] if x['view_id']=='V0' and x['control_id']=='ROOT')
        self.assertEqual(root['anchor'],[250.0,190.0])
        self.assertEqual(root['metadata']['anchor_authority'],'QUALIFIED_DIRECTIONAL_JOINT_VIEW_BINDING')
        self.assertEqual(s['authority']['rig_overlay_authority'],'QUALIFIED_DIRECTIONAL_JOINT_VIEW_BINDING')
    def test_missing_directional_binding_withholds_rig_instead_of_guessing(self):
        (self.root/'renderables/directional_joint_view_binding_set_ir.json').unlink()
        s=build_scene(self.root)
        self.assertEqual(s['puppet']['rig']['controls'],[])
        self.assertEqual(s['puppet']['rig']['edges'],[])
        self.assertIn('DIRECTIONAL_BINDING_NOT_EXPORTED__RIG_OVERLAY_WITHHELD',s['proof']['warnings'])
        self.assertEqual(s['authority']['rig_overlay_authority'],'WITHHELD_MISSING_QUALIFIED_BINDING')
    def test_runtime_uses_proof_bake(self):
        self.assertEqual(runtime_clips(self.root)['clips'][0]['authority'],'QUALIFICATION_OWNED_MOTION_BAKE')
        f=runtime_frame(self.root,'IDLE',.5)
        self.assertEqual(f['frame']['mesh_vertices_by_id']['V0:BODY'][0],[135.0,130.0])
    def test_proof_block_keeps_static_scene_visible_but_runtime_withheld(self):
        proof_path=self.root/'proof/product_proof_bundle_ir.json'
        proof=json.loads(proof_path.read_text())
        proof['overall_status']='ABSTAIN'
        writej(proof_path,proof)
        s=build_scene(self.root)
        self.assertFalse(s['proof']['passed'])
        self.assertTrue(s['puppet']['meshes'])
        self.assertTrue(s['puppet']['rig']['controls'])
        r=runtime_clips(self.root)
        self.assertEqual(r['clips'],[])
        self.assertFalse(r['preview_available'])
        self.assertEqual(r['authority'],'RUNTIME_PREVIEW_WITHHELD_UNTIL_CURRENT_PASS_PROOF')
        with self.assertRaises(LivingCompileError): runtime_frame(self.root,'IDLE',.5)
    def test_authoring_v3_is_sibling_and_validates_topology(self):
        before=(self.root/'puppet/canonical_puppet_graph_v3.json').read_bytes()
        out=save_user_layer(self.root,{'view_id':'V0','rig_topology_edits':[{'operation':'add_bone','bone_id':'user:tip','parent_id':'CHILD','anchor_xy':[250,400]}],'ik_constraints':[{'constraint_id':'IK:0','start_bone_id':'ROOT','end_bone_id':'CHILD','target_bone_id':'user:tip'}],'animation_edits':[{'clip_id':'USER','duration_seconds':1,'tracks':[{'bone_id':'CHILD','keys':[{'time_seconds':0,'rotation_degrees':0,'translation_xy':[0,0],'scale_xy':[1,1],'curve':'linear'},{'time_seconds':1,'rotation_degrees':20,'translation_xy':[0,0],'scale_xy':[1,1],'curve':'cubic'}]}]}]})
        self.assertTrue(out['requires_dynamic_revalidation'])
        self.assertEqual(before,(self.root/'puppet/canonical_puppet_graph_v3.json').read_bytes())
        self.assertTrue(Path(out['path']).is_file())
    def test_static_shell_present(self):
        app=LivingCompileApplication.default(); data,ctype=app.static_bytes('/')
        self.assertIn(b'Living Compile',data)
        self.assertEqual(ctype,'text/html')

if __name__=='__main__': unittest.main()
