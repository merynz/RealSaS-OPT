from dataclasses import replace
import unittest
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.graphics_evidence_v2 import (
    CanonicalAddressV2,RayHitEvidenceV2,RayStackEvidenceV2,AdmittedRayArtifactV2,
)
from compiler.realsas_compiler_core.graphics_v2 import compile_ray_graphics_v2


class RayGraphicsTests(unittest.TestCase):
    def ray(self,key='r',faces=(0,1),alphas=(.5,.5),**kwargs):
        hits=tuple(RayHitEvidenceV2(CanonicalAddressV2('a'*64,face,(.2,.3,.5),'b'*64),float(i+1),alpha,(1.,0.,0.))
                   for i,(face,alpha) in enumerate(zip(faces,alphas)))
        row=RayStackEvidenceV2('c'*64,key,0,hits,True,'SOURCE_OVER',1e-9,'d'*64,'e'*64,True,'NO_AUTHORED_ORDER__DEPTH_QUALIFIED')
        return replace(row,**kwargs)

    def compile(self,rows,admitted=None,**kwargs):
        if admitted is None:admitted={'d'*64:AdmittedRayArtifactV2('e'*64,frozenset(r.record_hash() for r in rows))}
        return compile_ray_graphics_v2(canonical_mesh_binding_hash='a'*64,canonical_mesh_skin_binding_hash='f'*64,
                source_graphics_binding_hash='1'*64,face_count=3,evidence=rows,
                admitted_artifacts=admitted,address_witnesses={'b'*64},**kwargs)

    def test_transparent_front_keeps_rear_contribution(self):
        result=self.compile([self.ray()]);comp=result.rays[0]['compositing']
        self.assertEqual(comp['incoming_transmittance'],[1.,.5])
        self.assertEqual(comp['contribution'],[.5,.25])
        self.assertEqual(comp['alpha'],.75)
        self.assertEqual(comp['premultiplied_linear_rgb'],[.75,0,0])

    def test_crossing_orders_survive_without_dropping_cycle_edges(self):
        rows=[self.ray('ab'),self.ray('ba',faces=(1,0))]
        result=self.compile(rows)
        self.assertEqual(result.scopes[0]['order_constraints'],[(0,1),(1,0)])
        self.assertEqual(result.scopes[0]['edges_removed'],0)
        self.assertEqual(result.scopes[0]['order_status'],'LOCAL_INTERLEAVING_REQUIRED')

    def test_unknown_and_tie_abstain_without_fabricating_visibility(self):
        unknown=self.compile([self.ray(alphas=(None,.5))]).rays[0]['compositing']
        self.assertFalse(unknown['alpha_valid']);self.assertEqual(unknown['reason'],'UNKNOWN_ALPHA')
        row=self.ray();row=replace(row,hits=(row.hits[0],replace(row.hits[1],depth=1.)))
        result=self.compile([row]);comp=result.rays[0]['compositing']
        self.assertTrue(comp['alpha_valid']);self.assertEqual(comp['alpha'],.75)
        self.assertFalse(comp['composite_valid']);self.assertEqual(result.rays[0]['order_constraints'],[])
        self.assertEqual(result.scopes[0]['unobserved_visibility'],'UNKNOWN')

    def test_valid_flag_and_hash_alone_cannot_mint_admission(self):
        with self.assertRaisesRegex(QualificationError,'ARTIFACT_UNADMITTED'):
            self.compile([self.ray()],admitted={})
        row=self.ray();admitted={'d'*64:AdmittedRayArtifactV2('e'*64,frozenset([row.record_hash()]))}
        tampered=replace(row,hits=(replace(row.hits[0],effective_alpha=.7),row.hits[1]))
        with self.assertRaisesRegex(QualificationError,'RECORD_NOT_IN_QUALIFIED'):
            self.compile([tampered],admitted=admitted)

    def test_exact_mesh_witness_duplicate_and_numerics_fail_closed(self):
        row=self.ray()
        bad=replace(row,hits=(replace(row.hits[0],address=replace(row.hits[0].address,mesh_hash='2'*64)),row.hits[1]))
        with self.assertRaisesRegex(QualificationError,'CANONICAL_M_DRIFT'):self.compile([bad])
        with self.assertRaisesRegex(QualificationError,'DUPLICATE_RAY'):self.compile([row,row])
        bad=replace(row,hits=(replace(row.hits[0],address=replace(row.hits[0].address,binding_witness_hash='2'*64)),row.hits[1]))
        with self.assertRaisesRegex(QualificationError,'WITNESS_UNADMITTED'):self.compile([bad])

    def test_empty_complete_ray_is_transparent_and_input_order_deterministic(self):
        empty=self.ray('empty',faces=(),alphas=())
        a=self.compile([empty,self.ray()]);b=self.compile([self.ray(),empty])
        self.assertEqual(a.graph_hash,b.graph_hash)
        self.assertEqual(a.rays[0]['compositing']['alpha'],0)
        self.assertTrue(a.rays[0]['compositing']['composite_valid'])

    def test_unknown_artistic_order_cannot_be_replaced_by_physical_depth(self):
        row=self.ray(source_artistic_order_status='UNSUPPORTED_AUTHORED_ORDER')
        result=self.compile([row]);comp=result.rays[0]['compositing']
        self.assertEqual(result.rays[0]['order_constraints'],[])
        self.assertEqual(comp['reason'],'SOURCE_ORDER_UNQUALIFIED')
        self.assertFalse(comp['composite_valid'])
        self.assertTrue(comp['alpha_valid'])


if __name__=='__main__':unittest.main()
