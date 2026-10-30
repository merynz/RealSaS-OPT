from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    AppearanceTextureIR,
    CompleteAppearanceAssetIR,
    complete_appearance_asset_hash,
)
from compiler.realsas_compiler_core.canonical_graphics_ir_v1 import (
    canonical_graphics_from_dict_v1,
    mint_canonical_graphics_v1,
)
from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    CanonicalMechanicsIRV1,
    canonical_mechanics_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,
    CanonicalMeshVertexIR,
    canonical_mesh_lineage_hash_v1,
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
from compiler.realsas_compiler_core.types import QualificationError, SurfaceSupportBinding


def _support(index: int) -> SurfaceSupportBinding:
    return SurfaceSupportBinding(
        mode="EXACT_SURFACE_NODE",
        coefficients=((f"S:{index}", 1.0),),
    )


def _mesh() -> CanonicalMeshIR:
    vertices = tuple(
        CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=f"MV:{index}",
            support_binding=_support(index),
            component_id="component:0",
            P=point,
            source_candidate_vertex_id=f"CV:{index}",
        )
        for index, point in enumerate(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    value = CanonicalMeshIR(
        vertices=vertices,
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
        metadata={"authority": "COMPILER_CANONICAL_MESH"},
    )
    return replace(value, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(value))


def _skin(mesh: CanonicalMeshIR) -> CanonicalSkinIRV2:
    value = CanonicalSkinIRV2(
        rows=tuple(
            CanonicalSkinRowIRV2(
                canonical_mesh_vertex_id=vertex.canonical_mesh_vertex_id,
                influences=(("J:root", 1.0),),
                source_carrier_vertex_id=vertex.source_candidate_vertex_id,
            )
            for vertex in mesh.vertices
        ),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash="a" * 64,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        carrier_skin_qualification_hash="b" * 64,
        skin_code_proposal_hash="c" * 64,
        skin_codec_architecture_id="RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0",
        skin_codec_checkpoint_hash="d" * 64,
        skin_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_SKIN"},
    )
    return replace(value, skin_lineage_hash=canonical_skin_lineage_hash_v2(value))


def _mechanics(mesh: CanonicalMeshIR, skin: CanonicalSkinIRV2) -> CanonicalMechanicsIRV1:
    value = CanonicalMechanicsIRV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash="e" * 64,
        deform_structure_hash=skin.deform_structure_hash,
        control_program_hash="f" * 64,
        skin_lineage_hash=skin.skin_lineage_hash,
        constraint_schedule=(),
        executable_operator_ids=(),
        rest_pose_hash="1" * 64,
        qualification_hash="2" * 64,
        mechanics_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_MECHANICS"},
    )
    return replace(
        value,
        mechanics_lineage_hash=canonical_mechanics_lineage_hash_v1(value),
    )


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
        compile_seal_binding_hash="3" * 64,
        candidate_mesh_binding_hash="4" * 64,
        surface_addressing_binding_hash="5" * 64,
        appearance_domain_binding_hash="6" * 64,
        output_direction_set_binding_hash="7" * 64,
        textures=textures,
        uv_npz_path="uv.npz",
        uv_npz_sha256="8" * 64,
        provenance_npz_path="provenance.npz",
        provenance_npz_sha256="9" * 64,
        atlas_layout={"layout": "fixture"},
        asset_hash="",
        metadata={"source_wins": True},
    )
    return replace(value, asset_hash=complete_appearance_asset_hash(value))


def _presentation(mesh: CanonicalMeshIR, skin: CanonicalSkinIRV2, appearance: CompleteAppearanceAssetIR) -> CompiledGraphicsV1:
    fragment = GraphicsFragmentV1(
        fragment_id="GFX:fixture",
        face_indices=(0,),
        support_band_pixels=0,
        metadata={"canonical_mesh_only": True},
    )
    value = CompiledGraphicsV1(
        canonical_mesh_binding_hash=mesh.mesh_lineage_hash,
        canonical_mesh_skin_binding_hash=skin.skin_lineage_hash,
        source_graphics_binding_hash=appearance.asset_hash,
        graphics_evidence_binding_hash="a" * 64,
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
    return replace(value, graph_hash=compiled_graphics_hash_v1(value))


def test_canonical_graphics_unifies_caa_and_presentation_under_one_visual_authority():
    mesh = _mesh()
    skin = _skin(mesh)
    mechanics = _mechanics(mesh, skin)
    appearance = _appearance()
    presentation = _presentation(mesh, skin, appearance)

    graphics = mint_canonical_graphics_v1(
        mesh,
        skin,
        mechanics,
        presentation=presentation,
        appearance=appearance,
    )

    assert graphics.metadata["authority"] == "COMPILER_CANONICAL_GRAPHICS"
    assert graphics.metadata["graphics_owns_all_visual_presentation"] is True
    assert graphics.metadata["caa_is_graphics_appearance_sublayer"] is True
    assert graphics.metadata["visibility_occlusion_owner"] == "GRAPHICS"
    assert graphics.metadata["draw_order_owner"] == "GRAPHICS"
    assert graphics.metadata["mechanics_may_not_decide_draw_order"] is True
    assert graphics.metadata["motion_may_not_decide_draw_order"] is True
    assert canonical_graphics_from_dict_v1(graphics.to_dict()) == graphics


def test_canonical_graphics_fails_closed_when_presentation_is_not_bound_to_caa_asset():
    mesh = _mesh()
    skin = _skin(mesh)
    mechanics = _mechanics(mesh, skin)
    appearance = _appearance()
    presentation = _presentation(mesh, skin, appearance)
    presentation = replace(
        presentation,
        source_graphics_binding_hash="0" * 64,
        graph_hash="",
    )
    presentation = replace(
        presentation,
        graph_hash=compiled_graphics_hash_v1(presentation),
    )

    with pytest.raises(QualificationError, match="PRESENTATION_APPEARANCE_BINDING_DRIFT"):
        mint_canonical_graphics_v1(
            mesh,
            skin,
            mechanics,
            presentation=presentation,
            appearance=appearance,
        )
