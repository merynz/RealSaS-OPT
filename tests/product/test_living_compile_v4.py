from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path

from product.living_compile.authoring import save_user_layer
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
    product={'schema_version':'RealSaS.CanonicalPuppetGraph.v3','product_state_hash':'p'*64,'representation_class':'DIRECTIONAL_2D_2P5D_PUPPET','full_3d_reconstruction_authority':False,'mechanical_state':{'surface':{'surface_nodes':nodes},'skeleton':{'joints':joints}},'directional_renderables':{'directions':dirs},'motion_state':{'clips':[{'clip_id':'IDLE','clip_kind':'idle','duration_sec':1.0,'loop':True,'metadata':{}}]}}
    proof={'schema_version':'RealSaS.ProductProofBundleIR.v1','source_product_state_hash':'p'*64,'overall_status':'PASS','proof_bundle_hash':'q'*64,'domain_reports':[{'proof_domain':'MOTION','status':'PASS','failure_signatures':[],'owner_attribution':[],'domain_proof_hash':'m'*64}]}
    bake={'schema_version':'RealSaS.QualificationOwnedMotionBakeIR.v1','source_product_state_hash':'p'*64,'proof_plan_hash':'r'*64,'clip_id':'IDLE','duration_seconds':1.0,'fps':1.0,'loop':True,'bake_hash':'b'*64,'frames':[{'time_seconds':0.0,'mesh_vertices_by_id':[['V0:BODY',[[130,130],[370,130],[250,370]]]],'render_order_by_view':[['V0',['V0:BODY']]]},{'time_seconds':1.0,'mesh_vertices_by_id':[['V0:BODY',[[140,130],[380,130],[260,370]]]],'render_order_by_view':[['V0',['V0:BODY']]]}]}
    writej(root/'puppet/canonical_puppet_graph_v3.json',product)
    writej(root/'proof/product_proof_bundle_ir.json',proof)
    writej(root/'proof/motion_bakes/IDLE.json',bake)


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
    def test_runtime_uses_proof_bake(self):
        self.assertEqual(runtime_clips(self.root)['clips'][0]['authority'],'QUALIFICATION_OWNED_MOTION_BAKE')
        f=runtime_frame(self.root,'IDLE',.5)
        self.assertEqual(f['frame']['mesh_vertices_by_id']['V0:BODY'][0],[135.0,130.0])
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
