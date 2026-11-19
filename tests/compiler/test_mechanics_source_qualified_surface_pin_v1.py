from __future__ import annotations

"""Real canonical Mechanics+XPBD consumer for typed native open pin programs.

Source/Compiler receipts are independently fabricated *test fixtures* only.
This is NOT a learned cloth, license audit, native annotation or ProductProof.
"""

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetSessionV1,
)
from compiler.realsas_compiler_core.mechanics_source_qualified_surface_pin_v1 import (
    PIN_ADMISSION, TRANSPORT_ADMISSION,
    SourceQualifiedOpenSurfacePinProgramV1,
    prepare_source_qualified_open_surface_xpbd_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    XPBDSurfacePolicyV1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_xpbd_surface_editor_v1 import _editor


def _evidence(program):
    from compiler.realsas_compiler_core.mechanics_source_qualified_surface_pin_v1 import (
        _admit_open_patch,
    )
    mesh=_editor()[0].mesh
    _,canonical_hem_sha256=_admit_open_patch(
        mesh,canonical_component_id=program.canonical_component_id,
        selected_face_indices=program.canonical_face_indices,
    )
    native={
        "admission_status": PIN_ADMISSION,
        "source_qualification_key": program.source_qualification_key,
        "source_asset_sha256": program.source_asset_sha256,
        "dataset_manifest_sha256": program.dataset_manifest_sha256,
        "native_open_boundary_sha256": program.native_open_boundary_sha256,
        "native_pin_witness_sha256": program.native_pin_witness_sha256,
        "host_only_intervention_sha256": program.host_only_intervention_sha256,
        "candidate_mstar_topology_address_sha256":
            program.candidate_mstar_topology_address_sha256,
        "source_face_pin_transport_sha256":
            program.source_face_pin_transport_sha256,
        "program_witness_sha256": program.witness_hash(),
        "canonical_component_id": program.canonical_component_id,
        "host_canonical_chain_id": program.host_canonical_chain_id,
        "source_rights_independently_verified": True,
        "native_open_hem_and_pins_independently_verified": True,
        "heldout_host_only_intervention_confirmed": True,
        "generated_views_used_as_source_truth": False,
        "student_slot_ids_used_as_source_truth": False,
        "material_coefficients_claimed_as_source_truth": False,
    }
    return (
        {program.source_qualification_key:native},
        {program.source_face_pin_transport_sha256:{
            "admission_status":TRANSPORT_ADMISSION,
            "canonical_mesh_lineage_hash":program.canonical_mesh_lineage_hash,
            "source_candidate_lineage_hash":_editor()[0].mesh.source_candidate_lineage_hash,
            "candidate_mstar_topology_address_sha256":
                program.candidate_mstar_topology_address_sha256,
            "source_asset_sha256":program.source_asset_sha256,
            "native_pin_witness_sha256":program.native_pin_witness_sha256,
            "native_open_boundary_sha256":program.native_open_boundary_sha256,
            "canonical_open_boundary_edges_sha256":canonical_hem_sha256,
            "source_to_canonical_hem_transport_verified":True,
            "source_face_pin_transport_sha256":
                program.source_face_pin_transport_sha256,
            "independently_qualified_final_mstar_address":True,
        }},
    )


def _case(*,pins=("MV:0",),source_pins=("NATIVE:0",)):
    bundle,old_domain,old_policy=_editor()
    program=SourceQualifiedOpenSurfacePinProgramV1(
        source_asset_sha256="a"*64,
        dataset_manifest_sha256="b"*64,
        source_qualification_key="source-open-native-pinned-asset",
        candidate_mstar_topology_address_sha256="c"*64,
        canonical_mesh_lineage_hash=bundle.mesh.mesh_lineage_hash,
        canonical_component_id=bundle.mesh.vertices[0].component_id,
        host_canonical_chain_id=bundle.rig.chains[0].canonical_chain_id,
        source_native_face_indices=(7,),
        canonical_face_indices=(0,),
        source_native_face_corner_barycentric=(
            ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),
        ),
        source_native_pin_vertex_ids=source_pins,
        canonical_pin_vertex_ids=pins,
        native_open_boundary_sha256="d"*64,
        native_pin_witness_sha256="e"*64,
        host_only_intervention_sha256="f"*64,
        source_face_pin_transport_sha256="",
    )
    program=replace(
        program,source_face_pin_transport_sha256=program.transport_hash(),
    )
    native,transport=_evidence(program)
    args=dict(
        mesh=bundle.mesh,rig=bundle.rig,skin=bundle.skin,
        mechanics=bundle.mechanics,program=program,
        trusted_native_pin_registry=native,
        trusted_compiler_transport_registry=transport,
    )
    return bundle,old_domain,old_policy,args


def test_native_source_pin_program_enters_actual_editor_xpbd_without_minting_truth():
    bundle,old_domain,_,args=_case()
    activation=prepare_source_qualified_open_surface_xpbd_v1(**args)
    assert activation.domain==old_domain
    assert activation.domain.dynamic_vertex_ids==("MV:1","MV:2")
    assert activation.domain.pinned_vertex_ids==("MV:0",)
    assert activation.product_release_admission is False
    assert activation.source_material_truth_claimed is False
    assert len(activation.qualification_receipt_sha256)==64
    editor=ControllablePuppetSessionV1(bundle)
    with pytest.raises(QualificationError,match="DT_WITHOUT_POLICY"):
        editor.frame(dt_seconds=1./60.)
    # The explicit user/compiler activation remains separate from source truth.
    editor.enable_declared_xpbd_surface(
        activation.domain,XPBDSurfacePolicyV1(
            domain_hash=activation.domain.domain_hash,
        ),
    )
    initial=editor.frame()
    moved=editor.frame(dt_seconds=1./60.)
    assert moved.xpbd_surface_state.ticks==2
    assert moved.deformed.canonical_mesh_vertex_ids==(
        initial.deformed.canonical_mesh_vertex_ids
    )
    assert moved.deformed.positions!=initial.deformed.positions
    assert moved.deformed.positions[0]==initial.deformed.positions[0]
    # Same Graphics/Mechanics output surface: not a second cloth mesh.
    assert moved.meshes!=initial.meshes
    assert moved.product_revision_id==initial.product_revision_id


def test_two_host_anchored_pins_keep_one_free_dynamic_particle():
    bundle,_,_,args=_case(
        pins=("MV:0","MV:1"),
        source_pins=("NATIVE:0","NATIVE:1"),
    )
    accepted=prepare_source_qualified_open_surface_xpbd_v1(**args)
    assert accepted.domain.dynamic_vertex_ids==("MV:2",)
    assert accepted.domain.pinned_vertex_ids==("MV:0","MV:1")
    editor=ControllablePuppetSessionV1(bundle)
    editor.enable_declared_xpbd_surface(
        accepted.domain,XPBDSurfacePolicyV1(
            domain_hash=accepted.domain.domain_hash,
        ),
    )
    assert editor.frame(dt_seconds=1./60.).xpbd_surface_state.ticks==2


def test_no_native_pin_heldout_intervention_or_mstar_transport_no_authority():
    _,_,_,args=_case()
    native=args["trusted_native_pin_registry"]
    transport=args["trusted_compiler_transport_registry"]
    key=args["program"].source_qualification_key
    trans=args["program"].source_face_pin_transport_sha256
    with pytest.raises(QualificationError,match="RECEIPTS_REQUIRED"):
        prepare_source_qualified_open_surface_xpbd_v1(
            **{**args,"trusted_native_pin_registry":{}},
        )
    with pytest.raises(QualificationError,match="RECEIPTS_REQUIRED"):
        prepare_source_qualified_open_surface_xpbd_v1(
            **{**args,"trusted_compiler_transport_registry":{}},
        )
    for field,value in (
        ("heldout_host_only_intervention_confirmed",False),
        ("generated_views_used_as_source_truth",True),
        ("source_rights_independently_verified",False),
        ("native_open_hem_and_pins_independently_verified",False),
    ):
        modified={key:{**native[key],field:value}}
        with pytest.raises(QualificationError,match="RECEIPT_DRIFT"):
            prepare_source_qualified_open_surface_xpbd_v1(
                **{**args,"trusted_native_pin_registry":modified},
            )
    with pytest.raises(QualificationError,match="RECEIPT_DRIFT"):
        prepare_source_qualified_open_surface_xpbd_v1(
            **{**args,"trusted_compiler_transport_registry":{
                trans:{**transport[trans],
                       "independently_qualified_final_mstar_address":False}
            }},
        )
    # An attested source OPEN hem cannot be reused with a different M* hem.
    with pytest.raises(QualificationError,match="RECEIPT_DRIFT"):
        prepare_source_qualified_open_surface_xpbd_v1(
            **{**args,"trusted_compiler_transport_registry":{
                trans:{**transport[trans],
                       "canonical_open_boundary_edges_sha256":"0"*64}
            }},
        )


def test_reindexed_faces_and_pin_order_cannot_be_replayed_under_stale_hash():
    _,_,_,args=_case()
    p=args["program"]
    with pytest.raises(QualificationError,match="ADDRESS_OR_HASH_DRIFT"):
        prepare_source_qualified_open_surface_xpbd_v1(
            **{**args,"program":replace(p,source_native_face_indices=(8,))},
        )
    with pytest.raises(QualificationError,match="ADDRESS_OR_HASH_DRIFT"):
        prepare_source_qualified_open_surface_xpbd_v1(
            **{**args,"program":replace(p,canonical_pin_vertex_ids=("MV:2",))},
        )
    with pytest.raises(QualificationError,match="ADDRESS_OR_HASH_DRIFT"):
        prepare_source_qualified_open_surface_xpbd_v1(
            **{**args,"program":replace(
                p,candidate_mstar_topology_address_sha256="0"*64,
            )},
        )


def test_released_host_and_closed_body_are_not_free_open_garment():
    from compiler.realsas_compiler_core.mechanics_source_qualified_surface_pin_v1 import (
        _admit_open_patch,
    )
    bundle,_,_,args=_case()
    p=args["program"]
    foreign=replace(
        p,host_canonical_chain_id="UNBOUND",
    )
    foreign=replace(
        foreign,source_face_pin_transport_sha256=foreign.transport_hash(),
    )
    new_native,new_transport=_evidence(foreign)
    with pytest.raises(QualificationError,match="HOST_CHAIN_NOT_IN_CANONICAL_RIG"):
        prepare_source_qualified_open_surface_xpbd_v1(
            **{**args,"program":foreign,
               "trusted_native_pin_registry":new_native,
               "trusted_compiler_transport_registry":new_transport},
        )
    # A closed two-sided triangle has no open boundary.
    closed=replace(bundle.mesh,faces=(
        bundle.mesh.faces[0],
        tuple(reversed(bundle.mesh.faces[0])),
    ))
    with pytest.raises(QualificationError,match="OPEN_HEM_MISSING"):
        _admit_open_patch(
            closed,canonical_component_id=p.canonical_component_id,
            selected_face_indices=(0,1),
        )


def test_unknown_native_pin_is_not_verified_negative_or_auto_activation():
    _,_,_,args=_case()
    p=args["program"]
    missing=replace(p,source_native_pin_vertex_ids=())
    with pytest.raises(QualificationError,match="ADDRESS_OR_HASH_DRIFT"):
        prepare_source_qualified_open_surface_xpbd_v1(
            **{**args,"program":missing},
        )


def test_one_native_face_can_parent_multiple_refined_mstar_children():
    from compiler.realsas_compiler_core.mechanics_source_qualified_surface_pin_v1 import (
        _validate_native_face_corners_v1,
    )
    _,_,_,args=_case()
    p=args["program"]
    # Two triangles split from ONE native face. Positive winding and all
    # barycentric children are exact addresses, not nearest-vertex guesses.
    refined=replace(
        p,source_native_face_indices=(7,7),
        canonical_face_indices=(10,11),
        source_native_face_corner_barycentric=(
            ((1.,0.,0.),(.5,.5,0.),(.5,0.,.5)),
            ((.5,.5,0.),(0.,1.,0.),(.5,0.,.5)),
        ),
    )
    _validate_native_face_corners_v1(refined)
    assert refined.transport_hash()!=p.transport_hash()
    bad=replace(
        refined,source_native_face_corner_barycentric=(
            refined.source_native_face_corner_barycentric[0],
            ((1.,0.,0.),(1.,0.,0.),(0.,1.,0.)),
        ),
    )
    with pytest.raises(QualificationError,match="FACE_BARYCENTRIC_INVALID"):
        _validate_native_face_corners_v1(bad)
    wrong=replace(refined,source_native_face_corner_barycentric=())
    with pytest.raises(QualificationError,match="FACE_BARYCENTRIC_CARDINALITY"):
        _validate_native_face_corners_v1(wrong)
