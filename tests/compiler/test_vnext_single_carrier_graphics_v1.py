from __future__ import annotations
import pytest
from compiler.realsas_compiler_core.graphics_runtime_v1 import build_runtime_graphics_v1
from compiler.realsas_compiler_core.graphics_v1 import GraphicsStaticRelationEvidenceV1,GraphicsViewRelationEvidenceV1,build_graphics_evidence_v1,compile_graphics_v1
from compiler.realsas_compiler_core.mechanical_candidate_v1 import AdmittedMeshCandidateV1,MeshCandidateRefV1,build_candidate_mechanics_probe_v1,build_mesh_candidate_set_v1,select_canonical_mechanics_v1
from compiler.realsas_compiler_core.types import QualificationError

def _c(i,h,p):return MeshCandidateRefV1(i,h,"scene","tokens-"+i,p)
def test_candidate_set_mints_no_canonical_authority_until_dynamic_selection():
    s=build_mesh_candidate_set_v1((_c("m0","mesh0",.2),_c("m1","mesh1",.1),_c("m2","mesh2",.3),_c("m3","mesh3",.4)),source_scene_binding_hash="scene");assert s.metadata["single_canonical_mesh_authority_minted"] is False;adm=(AdmittedMeshCandidateV1("m0","mesh0","static0",.2),AdmittedMeshCandidateV1("m1","mesh1","static1",.1));p=(build_candidate_mechanics_probe_v1(candidate_id="m0",mesh_binding_hash="mesh0",skeleton_binding_hash="g0",skin_binding_hash="w0",dynamic_penalty=.03,passed=True),build_candidate_mechanics_probe_v1(candidate_id="m1",mesh_binding_hash="mesh1",skeleton_binding_hash="g1",skin_binding_hash="w1",dynamic_penalty=.01,passed=False));x=select_canonical_mechanics_v1(s,adm,p);assert (x.selected_candidate_id,x.canonical_mesh_binding_hash,x.canonical_skeleton_binding_hash,x.canonical_skin_binding_hash)==("m0","mesh0","g0","w0");assert x.metadata["authority_class"]=="SOLE_CANONICAL_MECHANICS"
def test_premature_candidate_canonical_claim_fails_closed():
    bad=MeshCandidateRefV1("m0","mesh0","scene","tokens",0.,metadata={"canonical":True})
    with pytest.raises(QualificationError,match="PREMATURE_CANONICAL"):build_mesh_candidate_set_v1((bad,_c("m1","mesh1",.1)),source_scene_binding_hash="scene")
def test_graphics_compiles_role_free_order_and_runtime_consumes_content():
    faces=((0,1,2),(3,4,5),(6,7,8));static=(GraphicsStaticRelationEvidenceV1(0,1,0.,1.,.9,.8,.8,.9,"s01"),GraphicsStaticRelationEvidenceV1(1,2,0.,1.,0.,.8,.8,.9,"s12"));views=[]
    for v in range(8):views.extend((GraphicsViewRelationEvidenceV1(v,0,1,.95,.95,1.,1.,.95,f"v{v}-01"),GraphicsViewRelationEvidenceV1(v,1,2,.90,.95,1.,1.,.90,f"v{v}-12"),GraphicsViewRelationEvidenceV1(v,2,0,.70,.80,1.,1.,.70,f"v{v}-20")))
    e=build_graphics_evidence_v1(mesh_binding_hash="mesh",static_relations=static,view_relations=tuple(views));g=compile_graphics_v1(canonical_mesh_binding_hash="mesh",canonical_mesh_skin_binding_hash="mesh-skin",source_graphics_binding_hash="source-graphics",faces=faces,evidence=e,mechanical_attachment_evidence={(0,1):{"persistent_contact":.95,"parent_mechanical_domain_id":"joint:hand","anchor_face_index":0,"barycentric_anchor":(.2,.3,.5),"evidence_ref":"mechanical-contact"}});assert g.metadata["categorical_semantics_used"] is False and g.metadata["single_canonical_mesh_authority"] is True and g.metadata["independent_visual_mesh_allowed"] is False;assert len(g.fragments)==3 and len(g.attachments)==1;assert all(v.metadata["triangle_array_order_is_presentation_authority"] is False for v in g.views);r=build_runtime_graphics_v1(g,face_count=3);assert r.face_fragment_index==(0,1,2) and len(r.views)==8 and r.metadata["hash_only_binding_forbidden"] is True and r.metadata["behaviorally_consumable"] is True;assert all(len(v.topological_fragment_order)==3 for v in r.views)
def test_graphics_cannot_bind_a_second_visual_mesh():
    e=build_graphics_evidence_v1(mesh_binding_hash="proposal-A",static_relations=(GraphicsStaticRelationEvidenceV1(0,1,0.,1.,0.,0.,0.,1.),),view_relations=())
    with pytest.raises(QualificationError,match="CANONICAL_MESH_BINDING_DRIFT"):compile_graphics_v1(canonical_mesh_binding_hash="canonical-B",canonical_mesh_skin_binding_hash="skin",source_graphics_binding_hash="source",faces=((0,1,2),(3,4,5)),evidence=e)
