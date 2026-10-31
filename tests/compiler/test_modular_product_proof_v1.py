from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    AppearanceTextureIR,
    CompleteAppearanceAssetIR,
    complete_appearance_asset_hash,
)
from compiler.realsas_compiler_core.canonical_graphics_ir_v1 import mint_canonical_graphics_v1
from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import mint_canonical_mechanics_v1
from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,
    CanonicalMeshVertexIR,
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_motion_ir_v2 import (
    CanonicalControlMotionKeyIRV2,
    CanonicalControlMotionTrackIRV2,
    CanonicalMotionIRV2,
    canonical_motion_lineage_hash_v2,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
    _hash_control_program,
    _hash_deform_structure,
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2,
    CanonicalSkinRowIRV2,
    canonical_skin_lineage_hash_v2,
)
from compiler.realsas_compiler_core.graphics_v1 import (
    CompiledGraphicsV1,
    GraphicsFragmentV1,
    GraphicsViewStateV1,
    compiled_graphics_hash_v1,
)
from compiler.realsas_compiler_core.modular_product_proof_eval_v1 import (
    evaluate_modular_product_proof_v1,
)
from compiler.realsas_compiler_core.modular_product_proof_v1 import (
    REQUIRED_PROOF_DOMAINS_V1,
    require_current_modular_proof_bundle_v1,
)
from compiler.realsas_compiler_core.types import SurfaceSupportBinding


def _support(index: int) -> SurfaceSupportBinding:
    return SurfaceSupportBinding(
        mode="EXACT_SURFACE_NODE",
        coefficients=((f"S:{index}", 1.0),),
        metadata={},
    )


def _mesh() -> CanonicalMeshIR:
    points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    value = CanonicalMeshIR(
        vertices=tuple(
            CanonicalMeshVertexIR(
                canonical_mesh_vertex_id=f"MV:{index}",
                support_binding=_support(index),
                component_id="body",
                P=point,
                source_candidate_vertex_id=f"CV:{index}",
            )
            for index, point in enumerate(points)
        ),
        faces=(("MV:0", "MV:1", "MV:2"),),
        edges=(("MV:0", "MV:1"), ("MV:1", "MV:2"), ("MV:0", "MV:2")),
        source_evidence_binding_hash="1" * 64,
        partition_binding_hash="2" * 64,
        carrier_policy_binding_hash="3" * 64,
        static_qualification_hash="4" * 64,
        carrier_evidence_hash="5" * 64,
        carrier_topology_hash="6" * 64,
        carrier_geometry_hash="7" * 64,
        proposal_evidence_refs=("8" * 64,),
        source_candidate_lineage_hash="9" * 64,
        mesh_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_MESH",
            "birth_boundary": "STATIC_MESH_QUALIFIED",
            "geometry_authority": "COMPILER_STATIC_MESH_QUALIFICATION",
            "learned_family_embedded_in_schema": False,
            "learned_proposals_are_upstream_evidence_only": True,
            "dynamic_qualification_claimed": False,
            "rig_authority_claimed": False,
            "skin_authority_claimed": False,
            "motion_authority_claimed": False,
            "material_support_is_geometry_authority": False,
            "repair_after_mesh_birth_requires_new_mesh_lineage": True,
        },
    )
    return replace(value, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(value))


def _rig(mesh: CanonicalMeshIR) -> CanonicalRigIRV2:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    value = CanonicalRigIRV2(
        joints=(CanonicalDeformJointIRV2("J0", (0.0, 0.0, 0.0), None, "PJ0"),),
        local_frames=(CanonicalJointFrameIRV2("J0", identity, ("RX", "RY", "RZ")),),
        controls=(
            CanonicalRigControlIRV2(
                "C_ROOT",
                "ROOT",
                ("J0",),
                None,
                (0.0, 0.0, 0.0),
                "PC0",
            ),
        ),
        chains=(),
        constraints=(),
        attachment_anchors=(),
        deform_root_joint_ids=("J0",),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        feature_evidence_binding_hash="a" * 64,
        proposal_evidence_hash="b" * 64,
        rig_program_qualification_hash="c" * 64,
        deform_structure_hash="",
        control_program_hash="",
        rig_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_RIG"},
    )
    value = replace(value, deform_structure_hash=_hash_deform_structure(value))
    value = replace(value, control_program_hash=_hash_control_program(value))
    return replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v2(value))


def _skin(mesh: CanonicalMeshIR, rig: CanonicalRigIRV2) -> CanonicalSkinIRV2:
    value = CanonicalSkinIRV2(
        rows=tuple(
            CanonicalSkinRowIRV2(
                canonical_mesh_vertex_id=vertex.canonical_mesh_vertex_id,
                influences=(("J0", 1.0),),
                source_carrier_vertex_id=vertex.source_candidate_vertex_id,
            )
            for vertex in mesh.vertices
        ),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        carrier_skin_qualification_hash="d" * 64,
        skin_code_proposal_hash="e" * 64,
        skin_codec_architecture_id="RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0",
        skin_codec_checkpoint_hash="f" * 64,
        skin_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_SKIN"},
    )
    return replace(value, skin_lineage_hash=canonical_skin_lineage_hash_v2(value))


def _appearance() -> CompleteAppearanceAssetIR:
    textures = tuple(
        AppearanceTextureIR(
            direction_index=index,
            direction_id=f"V{index}",
            transport_png_path=f"V{index}.png",
            transport_png_sha256=(f"{index:x}" * 64)[:64],
            width=8,
            height=8,
        )
        for index in range(8)
    )
    value = CompleteAppearanceAssetIR(
        compile_seal_binding_hash="1" * 64,
        candidate_mesh_binding_hash="2" * 64,
        surface_addressing_binding_hash="3" * 64,
        appearance_domain_binding_hash="4" * 64,
        output_direction_set_binding_hash="5" * 64,
        textures=textures,
        uv_npz_path="uv.npz",
        uv_npz_sha256="6" * 64,
        provenance_npz_path="provenance.npz",
        provenance_npz_sha256="7" * 64,
        atlas_layout={"layout": "fixture"},
        asset_hash="",
        metadata={"source_wins": True},
    )
    return replace(value, asset_hash=complete_appearance_asset_hash(value))


def _graphics(mesh, skin, mechanics):
    appearance = _appearance()
    fragment = GraphicsFragmentV1(
        fragment_id="GFX:body",
        face_indices=(0,),
        support_band_pixels=0,
        metadata={"canonical_mesh_only": True},
    )
    presentation = CompiledGraphicsV1(
        canonical_mesh_binding_hash=mesh.mesh_lineage_hash,
        canonical_mesh_skin_binding_hash=skin.skin_lineage_hash,
        source_graphics_binding_hash=appearance.asset_hash,
        graphics_evidence_binding_hash="8" * 64,
        fragments=(fragment,),
        views=tuple(
            GraphicsViewStateV1(index, (fragment.fragment_id,), ())
            for index in range(8)
        ),
        attachments=(),
        graph_hash="",
        metadata={
            "authority_class": "COMPILED_GRAPHICS",
            "single_canonical_mesh_authority": True,
            "independent_visual_mesh_allowed": False,
            "source_pixels_win_where_observed": True,
        },
    )
    presentation = replace(
        presentation,
        graph_hash=compiled_graphics_hash_v1(presentation),
    )
    return mint_canonical_graphics_v1(
        mesh,
        skin,
        mechanics,
        presentation=presentation,
        appearance=appearance,
    )


def _motion(rig, mechanics, *, zero_quaternion: bool = False):
    end_quaternion = (0.0, 0.0, 0.0, 0.0) if zero_quaternion else (1.0, 0.0, 0.0, 0.0)
    value = CanonicalMotionIRV2(
        canonical_clip_id="MOTION:RUN",
        display_name="Run",
        intent="RUN",
        duration_seconds=1.0,
        loop=True,
        control_tracks=(
            CanonicalControlMotionTrackIRV2(
                canonical_motion_track_id="MT:ROOT",
                canonical_control_id="C_ROOT",
                keys=(
                    CanonicalControlMotionKeyIRV2(
                        0.0,
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0),
                    ),
                    CanonicalControlMotionKeyIRV2(
                        1.0,
                        (0.1, 0.0, 0.0),
                        end_quaternion,
                        (1.0, 1.0, 1.0),
                    ),
                ),
                source_proposal_track_id="PMT:ROOT",
            ),
        ),
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
        source_proposal_hash="9" * 64,
        motion_qualification_hash="a" * 64,
        motion_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_MOTION",
            "editable_motion_space": "ANIMATOR_CONTROLS",
            "direct_joint_track_authority": False,
            "derived_joint_bakes_allowed": True,
            "constraint_execution_owner": "MECHANICS",
        },
    )
    return replace(value, motion_lineage_hash=canonical_motion_lineage_hash_v2(value))


def _product(*, broken_motion: bool = False):
    mesh = _mesh()
    rig = _rig(mesh)
    skin = _skin(mesh, rig)
    mechanics = mint_canonical_mechanics_v1(mesh, rig, skin)
    graphics = _graphics(mesh, skin, mechanics)
    motion = _motion(rig, mechanics, zero_quaternion=broken_motion)
    return mesh, rig, skin, mechanics, graphics, (motion,)


def test_modular_product_proof_runs_all_shipping_domains_and_passes():
    product = _product()
    proof = evaluate_modular_product_proof_v1(*product)
    assert proof.overall_status == "PASS"
    assert proof.required_domains == REQUIRED_PROOF_DOMAINS_V1
    assert tuple(report.proof_domain for report in proof.domain_reports) == REQUIRED_PROOF_DOMAINS_V1
    assert all(report.status == "PASS" for report in proof.domain_reports)
    assert proof.metadata["legacy_canonical_puppet_graph_projection_used"] is False
    require_current_modular_proof_bundle_v1(*product, proof, require_pass=True)


def test_runtime_motion_failure_is_preserved_as_domain_fail_instead_of_escaping():
    product = _product(broken_motion=True)
    proof = evaluate_modular_product_proof_v1(*product)
    assert proof.overall_status == "FAIL"
    by_domain = {report.proof_domain: report for report in proof.domain_reports}
    assert by_domain["MOTION"].status == "FAIL"
    assert by_domain["DEFORMATION"].status == "FAIL"
    assert by_domain["MESH_QUALITY"].status == "PASS"
    assert by_domain["DIRECTIONAL_VISUAL"].status == "PASS"
    assert by_domain["MOTION"].owner_attribution[0]["owner"] == "MOTION"
    assert by_domain["DEFORMATION"].owner_attribution[0]["owner"] == "MECHANICS"
