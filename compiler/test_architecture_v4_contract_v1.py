from __future__ import annotations
import unittest
from dataclasses import replace
from realsas_compiler_core.bundle_routes import route_for
from realsas_compiler_core.mesh_binding import mesh_lineage_hash, mesh_skin_lineage_hash
from realsas_compiler_core.types import (
    SurfaceNode, RiggingSurfaceIR, QualifiedJoint, QualifiedSkinRow, QualifiedSkinIR,
    SurfaceSupportBinding, QualifiedMeshVertex, QualifiedEditableMeshIR,
    QualifiedMeshSkinRow, QualifiedMeshSkinIR, QualificationError,
)
from realsas_compiler_core.v4_types import (
    QualifiedSkeletonIRV2, AppearanceCornerBinding, CapabilityRequirement, MotionClipIR,
    JointTransformKeyIR,
)
from realsas_compiler_core.v4 import (
    qualified_skeleton_v2_lineage_hash, mechanical_state_hash, build_mechanical_state,
    build_appearance_binding, build_renderable_component, build_directional_renderable,
    build_directional_renderable_set, build_capability_contract,
    make_single_family_e2e_capability_contract, build_joint_track, build_motion_state,
    assemble_product_v3, bind_proof_plan, bind_measurement_report, bind_domain_proof,
    bind_product_proof_bundle, require_current_proof_bundle, project_runtime_package_v3,
)


class V4ContractTests(unittest.TestCase):
    def setUp(self):
        self.surface=RiggingSurfaceIR((
            SurfaceNode("S0",(0.,0.,0.),(0,1,2,3,4,5,6,7),("p0",),("o0",)),
            SurfaceNode("S1",(1.,0.,0.),(0,1,2,3,4,5,6,7),("p1",),("o1",)),
            SurfaceNode("S2",(0.,1.,0.),(0,1,2,3,4,5,6,7),("p2",),("o2",)),
        ),geometry_lineage_hash="SURFACE")
        sk=QualifiedSkeletonIRV2(
            joints=(QualifiedJoint("J0",(0.,0.,0.),None,("S0",),"P0"),),
            deform_root_ids=("J0",),assembly_root_binding={},
            qualification_report={"passed":True},skeleton_lineage_hash=""
        )
        self.skeleton=replace(sk,skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(sk))
        self.skin=QualifiedSkinIR(
            rows=tuple(QualifiedSkinRow(s, (("J0",1.0),),0.,0.) for s in ("S0","S1","S2")),
            surface_binding_hash="SURFACE",skeleton_binding_hash=self.skeleton.skeleton_lineage_hash,
            qualification_report={"passed":True},skin_lineage_hash="SKIN"
        )
        self.mechanical=build_mechanical_state(self.surface,self.skeleton,self.skin)

    def mesh_for(self,view):
        verts=(
            QualifiedMeshVertex("MV0",(0.,0.,0.),SurfaceSupportBinding("IDENTITY_SURFACE_NODE",(("S0",1.0),)),"C0"),
            QualifiedMeshVertex("MV1",(1.,0.,0.),SurfaceSupportBinding("IDENTITY_SURFACE_NODE",(("S1",1.0),)),"C1"),
            QualifiedMeshVertex("MV2",(0.,1.,0.),SurfaceSupportBinding("IDENTITY_SURFACE_NODE",(("S2",1.0),)),"C2"),
        )
        mesh=QualifiedEditableMeshIR(verts,(("MV0","MV1","MV2"),),(("MV0","MV1"),("MV1","MV2"),("MV2","MV0")),"SURFACE",view,f"CAM:{view}",{"passed":True},"")
        mesh=replace(mesh,mesh_lineage_hash=mesh_lineage_hash(mesh))
        rows=tuple(QualifiedMeshSkinRow(f"MV{i}",(("J0",1.0),),((f"S{i}",1.0),),0.,0.) for i in range(3))
        ms=QualifiedMeshSkinIR(rows,"SURFACE",self.skeleton.skeleton_lineage_hash,"SKIN",mesh.mesh_lineage_hash,"IDENTITY",{"passed":True},"")
        ms=replace(ms,mesh_skin_lineage_hash=mesh_skin_lineage_hash(ms))
        return mesh,ms

    def build_visual(self):
        dirs=[]
        for view in range(8):
            mesh,ms=self.mesh_for(view)
            corners=tuple(
                AppearanceCornerBinding(0,i,(float(i==1),float(i==2)),view,(0.5,0.5),f"OBS:{view}:{i}","OBSERVED_LOCAL","",1.0)
                for i in range(3)
            )
            app=build_appearance_binding(target_view_index=view,mesh_binding_hash=mesh.mesh_lineage_hash,camera_binding_hash=mesh.camera_binding_hash,corner_bindings=corners)
            comp=build_renderable_component(component_id="body",view_index=view,mesh=mesh,mesh_skin=ms,appearance=app,setup_order=0,coverage_classification="VISIBLE_CORE")
            dirs.append(build_directional_renderable(view_index=view,camera_binding_hash=f"CAM:{view}",components=(comp,)))
        return build_directional_renderable_set(tuple(dirs))

    def motion(self, payload="CLIP", moved_deg=20.0):
        clip=MotionClipIR("idle","PRESET",payload,("BASE_LBS",),"preset://idle",1.0,True)
        track=build_joint_track("JT0","idle","J0",(
            JointTransformKeyIR(0.0),
            JointTransformKeyIR(0.5,translation_xy=(0.02,0.0),rotation_deg=moved_deg,scale_xy=(1.0,1.0),depth_offset=0.05),
            JointTransformKeyIR(1.0),
        ))
        return build_motion_state((clip,),(track,))

    def product(self):
        visual=self.build_visual()
        caps=make_single_family_e2e_capability_contract(
            base_lbs_hash="LBS",mesh_hash="MESH",skinning_hash="SKINNING",visual_hash=visual.directional_visual_state_hash,
            preset_motion_hash="PRESET",runtime_hash="RUNTIME",policy_hash="POLICY"
        )
        return assemble_product_v3(self.mechanical,visual,caps,self.motion())

    def pass_bundle(self, product):
        reports=[]
        required=sorted(set(d for r in product.capability_contract.requirements if r.activation=="REQUIRED" for d in r.required_proof_domains))
        for domain in required:
            plan=bind_proof_plan(product,proof_domain=domain,operator_policy_hashes=("OP",),probe_specification={"probe":domain})
            meas=bind_measurement_report(product,plan,measurements={"ok":True})
            reports.append(bind_domain_proof(product,plan,meas,status="PASS"))
        return bind_product_proof_bundle(product,tuple(reports))

    def test_exact_eight_directions(self):
        visual=self.build_visual()
        bad=replace(visual,directions=visual.directions[:-1])
        caps=build_capability_contract("x",(CapabilityRequirement("BASE_LBS","REQUIRED","a","p",("DEFORMATION",)),))
        with self.assertRaisesRegex(QualificationError,"EXACTLY_8"):
            assemble_product_v3(self.mechanical,bad,caps,build_motion_state(()))

    def test_product_hash_changes_on_appearance_change(self):
        product_a=self.product(); visual=product_a.directional_renderables; d0=visual.directions[0]; c0=d0.components[0]
        changed_app=replace(c0.appearance,atlas_payload_hash="NEW",appearance_lineage_hash="")
        from realsas_compiler_core.v4 import appearance_lineage_hash, component_state_hash, direction_state_hash, directional_visual_state_hash
        changed_app=replace(changed_app,appearance_lineage_hash=appearance_lineage_hash(changed_app))
        changed_c=replace(c0,appearance=changed_app,component_state_hash=""); changed_c=replace(changed_c,component_state_hash=component_state_hash(changed_c))
        changed_d=replace(d0,components=(changed_c,),direction_state_hash=""); changed_d=replace(changed_d,direction_state_hash=direction_state_hash(changed_d))
        changed_visual=replace(visual,directions=(changed_d,)+visual.directions[1:],directional_visual_state_hash="")
        changed_visual=replace(changed_visual,directional_visual_state_hash=directional_visual_state_hash(changed_visual))
        product_b=assemble_product_v3(product_a.mechanical_state,changed_visual,product_a.capability_contract,product_a.motion_state)
        self.assertNotEqual(product_a.product_state_hash,product_b.product_state_hash)

    def test_proof_is_sibling_and_stale_after_product_change(self):
        product=self.product(); bundle=self.pass_bundle(product); self.assertEqual(bundle.overall_status,"PASS")
        old_hash=product.product_state_hash
        changed=assemble_product_v3(product.mechanical_state,product.directional_renderables,product.capability_contract,self.motion(payload="CLIP2",moved_deg=35.0))
        self.assertEqual(product.product_state_hash,old_hash); self.assertNotEqual(changed.product_state_hash,old_hash)
        with self.assertRaisesRegex(QualificationError,"STALE"): require_current_proof_bundle(changed,bundle,require_pass=True)

    def test_directional_artifact_routes_do_not_collide(self):
        visual=self.build_visual()
        self.assertNotEqual(route_for(visual.directions[0]).filename,route_for(visual.directions[1]).filename)
        self.assertNotEqual(route_for(visual.directions[0].components[0]).filename,route_for(visual.directions[1].components[0]).filename)

    def test_missing_required_proof_domain_abstains(self):
        product=self.product(); required=sorted(set(d for r in product.capability_contract.requirements if r.activation=="REQUIRED" for d in r.required_proof_domains)); reports=[]
        for domain in required[:-1]:
            plan=bind_proof_plan(product,proof_domain=domain,operator_policy_hashes=("OP",),probe_specification={"probe":domain})
            meas=bind_measurement_report(product,plan,measurements={"ok":True}); reports.append(bind_domain_proof(product,plan,meas,status="PASS"))
        self.assertEqual(bind_product_proof_bundle(product,tuple(reports)).overall_status,"ABSTAIN")

    def test_required_preset_motion_must_change_a_canonical_joint(self):
        visual=self.build_visual()
        caps=make_single_family_e2e_capability_contract(base_lbs_hash="LBS",mesh_hash="MESH",skinning_hash="SKINNING",visual_hash=visual.directional_visual_state_hash,preset_motion_hash="PRESET",runtime_hash="RUNTIME",policy_hash="POLICY")
        static=self.motion(moved_deg=0.0)
        static_track=replace(static.joint_tracks[0],keys=(JointTransformKeyIR(0.0),JointTransformKeyIR(0.5),JointTransformKeyIR(1.0)),track_hash="")
        from realsas_compiler_core.v4 import track_hash, motion_state_hash
        static_track=replace(static_track,track_hash=track_hash(static_track))
        static=replace(static,joint_tracks=(static_track,),motion_state_hash=""); static=replace(static,motion_state_hash=motion_state_hash(static))
        with self.assertRaisesRegex(QualificationError,"NO_EFFECTIVE_JOINT_MOTION"): assemble_product_v3(self.mechanical,visual,caps,static)

    def test_appearance_must_cover_every_face_corner(self):
        mesh,ms=self.mesh_for(0)
        app=build_appearance_binding(target_view_index=0,mesh_binding_hash=mesh.mesh_lineage_hash,camera_binding_hash=mesh.camera_binding_hash,corner_bindings=(AppearanceCornerBinding(0,0,(0.,0.),0,(0.,0.),"OBS","OBSERVED_LOCAL"),))
        comp=build_renderable_component(component_id="body",view_index=0,mesh=mesh,mesh_skin=ms,appearance=app,setup_order=0,coverage_classification="VISIBLE_CORE")
        from realsas_compiler_core.v4 import validate_renderable_component
        with self.assertRaisesRegex(QualificationError,"COVERAGE_INCOMPLETE"): validate_renderable_component(comp,self.mechanical)

    def test_motion_contract_has_no_3d_rigid_transform_fields(self):
        key=JointTransformKeyIR(0.25,translation_xy=(1.0,2.0),rotation_deg=15.0,scale_xy=(1.1,0.9),depth_offset=0.2)
        payload=key.to_dict()
        self.assertEqual(set(payload),{"time_sec","translation_xy","rotation_deg","scale_xy","depth_offset"})
        self.assertNotIn("rotation_xyzw",payload)
        self.assertNotIn("translation",payload)

    def test_full_3d_reconstruction_authority_is_forbidden(self):
        bad=replace(self.mechanical,full_3d_reconstruction_authority=True,mechanical_state_hash="")
        bad=replace(bad,mechanical_state_hash=mechanical_state_hash(bad))
        visual=self.build_visual()
        caps=make_single_family_e2e_capability_contract(base_lbs_hash="LBS",mesh_hash="MESH",skinning_hash="SKINNING",visual_hash=visual.directional_visual_state_hash,preset_motion_hash="PRESET",runtime_hash="RUNTIME",policy_hash="POLICY")
        with self.assertRaisesRegex(QualificationError,"FULL_3D_RECONSTRUCTION"):
            assemble_product_v3(bad,visual,caps,self.motion())

    def test_runtime_declares_directional_puppet_not_3d_reconstruction(self):
        product=self.product(); bundle=self.pass_bundle(product)
        runtime=project_runtime_package_v3(product,bundle,manifest={"name":"fixture"},runtime_payload_ref="runtime://fixture")
        self.assertEqual(runtime.manifest["representation_class"],"DIRECTIONAL_2D_2P5D_PUPPET")
        self.assertEqual(runtime.manifest["mechanical_equivalence_class"],"THREE_D_EQUIVALENT_MECHANICS")
        self.assertFalse(runtime.manifest["full_3d_reconstruction_authority"])


if __name__=="__main__": unittest.main()
