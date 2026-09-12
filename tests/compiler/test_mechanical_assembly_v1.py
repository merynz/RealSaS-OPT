from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.component_attachment import (
    ComponentEvidenceIR,
    qualify_component_set_v1,
)
from compiler.realsas_compiler_core.mechanical_assembly import (
    build_qualified_mechanical_assembly,
    qualified_mechanical_assembly_hash,
    validate_qualified_mechanical_assembly,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    QualifiedJoint,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
)
from compiler.realsas_compiler_core.v4 import (
    build_mechanical_state,
    qualified_skeleton_v2_lineage_hash,
)
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2


def _fixture():
    surface = RiggingSurfaceIR(
        (
            SurfaceNode("S_BODY", (0.0, 0.0, 0.0), tuple(range(8)), ("p",), ("o",)),
            SurfaceNode("S_HAT", (0.0, 1.0, 0.0), tuple(range(8)), ("p",), ("o",)),
        ),
        geometry_lineage_hash="SURFACE",
    )
    sk0 = QualifiedSkeletonIRV2(
        joints=(
            QualifiedJoint("J_ROOT", (0.0, 0.0, 0.0), None, ("S_BODY",), "P0"),
            QualifiedJoint("J_HEAD", (0.0, 1.0, 0.0), "J_ROOT", ("S_HAT",), "P1"),
        ),
        deform_root_ids=("J_ROOT",),
        assembly_root_binding={},
        qualification_report={"passed": True},
        skeleton_lineage_hash="",
    )
    skeleton = replace(sk0, skeleton_lineage_hash=qualified_skeleton_v2_lineage_hash(sk0))
    skin = QualifiedSkinIR(
        rows=(
            QualifiedSkinRow("S_BODY", (("J_ROOT", 1.0),), 0.0, 0.0),
            QualifiedSkinRow("S_HAT", (("J_HEAD", 1.0),), 0.0, 0.0),
        ),
        surface_binding_hash="SURFACE",
        skeleton_binding_hash=skeleton.skeleton_lineage_hash,
        qualification_report={"passed": True},
        skin_lineage_hash="SKIN",
    )
    mechanical = build_mechanical_state(surface, skeleton, skin)
    components = qualify_component_set_v1(
        (
            ComponentEvidenceIR(
                component_id="body",
                source_component_refs=("src:body",),
                observation_views=tuple(range(8)),
                surface_ids=("S_BODY",),
                proposed_mechanical_class="DEFORMABLE_COMPONENT",
                detachability_class="FIXED_COMPONENT",
                source_geometry_hash="GEO_BODY",
                skin_or_deformer_lineage_hash="SKIN",
            ),
            ComponentEvidenceIR(
                component_id="hat",
                source_component_refs=("src:hat",),
                observation_views=tuple(range(8)),
                surface_ids=("S_HAT",),
                proposed_mechanical_class="RIGID_SKINNED_COMPONENT",
                proposed_parent_joint_id="J_HEAD",
                detachability_class="DETACHABILITY_UNKNOWN",
                bind_state_authority_hash="BIND_HAT",
                source_geometry_hash="GEO_HAT",
                skin_or_deformer_lineage_hash="SKIN",
            ),
        ),
        known_surface_ids={"S_BODY", "S_HAT"},
        known_joint_ids={"J_ROOT", "J_HEAD"},
        surface_lineage_hash=surface.geometry_lineage_hash,
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
    )
    return mechanical, components


def test_component_authority_is_hash_bound_to_mechanical_state():
    mechanical, components = _fixture()
    assembly = build_qualified_mechanical_assembly(mechanical, components)
    assert assembly.assembly_lineage_hash == qualified_mechanical_assembly_hash(assembly)
    assert assembly.component_set.component_set_lineage_hash


def test_component_semantic_change_changes_assembly_lineage():
    mechanical, components = _fixture()
    a = build_qualified_mechanical_assembly(mechanical, components)
    changed_component = replace(
        components.components[1],
        detachability_class="FIXED_COMPONENT",
        component_lineage_hash="",
    )
    from compiler.realsas_compiler_core.component_attachment import (
        qualified_component_lineage_hash,
        qualified_component_set_lineage_hash,
    )
    changed_component = replace(
        changed_component,
        component_lineage_hash=qualified_component_lineage_hash(changed_component),
    )
    changed_set = replace(
        components,
        components=(components.components[0], changed_component),
        component_set_lineage_hash="",
    )
    changed_set = replace(
        changed_set,
        component_set_lineage_hash=qualified_component_set_lineage_hash(changed_set),
    )
    b = build_qualified_mechanical_assembly(mechanical, changed_set)
    assert a.assembly_lineage_hash != b.assembly_lineage_hash


def test_surface_lineage_drift_fails_closed():
    mechanical, components = _fixture()
    bad_set = replace(components, surface_lineage_hash="OTHER", component_set_lineage_hash="")
    from compiler.realsas_compiler_core.component_attachment import qualified_component_set_lineage_hash
    bad_set = replace(bad_set, component_set_lineage_hash=qualified_component_set_lineage_hash(bad_set))
    value = build_qualified_mechanical_assembly(mechanical, components)
    bad = replace(value, component_set=bad_set, assembly_lineage_hash="")
    bad = replace(bad, assembly_lineage_hash=qualified_mechanical_assembly_hash(bad))
    with pytest.raises(QualificationError, match="SURFACE_LINEAGE_MISMATCH"):
        validate_qualified_mechanical_assembly(bad)
