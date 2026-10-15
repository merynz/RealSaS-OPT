from __future__ import annotations

from tools.corpus.fit8_full_truth_schema_v1 import (
    DETERMINISTIC_DERIVED,
    QUALIFIED_DERIVED,
    REQUIRED,
    REQUIRED_WITH_VALIDITY,
    TeacherChannelSpecV1,
)

SCHEMA_VERSION = "RealSaS.FIT8FExtractionTruthExtension.v1"
STATUS = "RESEARCH_EXTENSION_NOT_YET_PROMOTED_TO_FULL_TRUTH_BUNDLE_V2"

# This extension is deliberately separate from FIT8FullTruthTeacherBundle.v1.
# It records extraction-domain truth discovered after exact FIT8 local topology
# courts. Promotion into a future bundle version requires an explicit migration.
F_EXTRACTION_CHANNELS_V1 = (
    TeacherChannelSpecV1(
        "extraction_scale_manifest",
        "extraction_multiscale",
        "json",
        "[L]",
        "PREREGISTERED_CANONICAL_CELL_LATTICE_PITCHES_AND_FRAMES",
        ("F",),
        DETERMINISTIC_DERIVED,
        availability=REQUIRED,
        notes="No single 64/96/128 grid is ontology; h/pitch is explicit teacher context.",
    ),
    TeacherChannelSpecV1(
        "cell_surface_support_dimension",
        "extraction_cell",
        "ragged",
        "{L}[C,*]",
        "EXACT_CLIPPED_SURFACE_AFFINE_DIMENSION_UNDER_EXPLICIT_SPATIAL_POLICY",
        ("F",),
        QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Separates AREA_SUPPORT, LINE_SUPPORT, POINT_TOUCH and NONE. Only positive-area support instantiates a surface patch.",
    ),
    TeacherChannelSpecV1(
        "cell_supported_patch_partition",
        "extraction_cell_patch",
        "ragged",
        "{L}[C,P,*]",
        "QUALIFIED_LOCAL_SHEET_PARTITION_FROM_POSITIVE_LENGTH_SHARED_SOURCE_EDGE_SUPPORT",
        ("F",),
        QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Scale-local geometric sheet identity; never equate with object/material/owner/component. Refines but never merges local_patch_lower_bound.",
    ),
    TeacherChannelSpecV1(
        "cell_edge_crossing_set",
        "extraction_cell_edge",
        "ragged",
        "{L}[C,12,*]",
        "CANONICALIZED_AUTHORED_SURFACE_CROSSINGS_WITH_SOURCE_BINDING_NORMAL_AND_ORIENTATION_VALIDITY",
        ("F",),
        QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Variable crossing cardinality is preserved; one crossing per grid edge is not universal truth.",
    ),
    TeacherChannelSpecV1(
        "cell_face_strand_set",
        "extraction_cell_face",
        "ragged",
        "{L}[C,6,*]",
        "QUALIFIED_AUTHORED_TRIANGLE_PLANE_STRAND_GRAPH",
        ("F",),
        QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Carries boundary arcs, closed face loops, open interior endpoints and branch/non-manifold strand topology.",
    ),
    TeacherChannelSpecV1(
        "face_strand_edge_crossing_binding",
        "extraction_cell_boundary_incidence",
        "ragged",
        "{L}[C,6,*]",
        "WORLD_SPACE_AND_SOURCE_LINEAGE_BINDING_OF_FACE_STRAND_ENDPOINTS_TO_SPECIFIC_CELL_EDGE_CROSSINGS",
        ("F",),
        QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Cube-edge ID alone is insufficient when one edge contains multiple authored crossings; corner multi-edge equivalence remains explicit.",
    ),
    TeacherChannelSpecV1(
        "cell_authored_edge_incidence",
        "extraction_cell_authored_edge",
        "ragged",
        "{L}[C,*]",
        "QUALIFIED_SOURCE_CARRIER_EDGE_INCIDENCE_CLIPPED_TO_CELL",
        ("F",),
        QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Distinguishes manifold interior edges, authored open boundaries and authored non-manifold incidence; preserves POINT_TOUCH versus POSITIVE_LENGTH.",
    ),
    TeacherChannelSpecV1(
        "cell_neighbor_strand_continuation",
        "extraction_neighbor_face_incidence",
        "ragged",
        "{L}[Eneighbor,*]",
        "QUALIFIED_CROSS_CELL_CONTINUATION_FROM_SHARED_BOUNDARY_CROSSING_AND_SOURCE_LINEAGE",
        ("F",),
        QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="No global persistent patch-ID ontology; continuation is local boundary incidence.",
    ),
    TeacherChannelSpecV1(
        "cell_local_topology_qualification",
        "extraction_cell_patch_topology",
        "ragged",
        "{L}[C,P]",
        "CLIPPED_PRIMAL_TOPOLOGY_COURT_WITH_BOUNDARY_NONMANIFOLD_ORIENTABILITY_AND_GENUS_VALIDITY",
        ("F",),
        QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Topology claims abstain when preconditions fail. Boundary-explainable genus-0 annuli are not automatically a new interior primitive.",
    ),
    TeacherChannelSpecV1(
        "cell_topology_residual_target",
        "extraction_residual",
        "ragged",
        "{L}[C,*]",
        "DIFFERENCE_BETWEEN_QUALIFIED_LOCAL_TRUTH_AND_BASE_F_REPRESENTATION_CAPACITY",
        ("F",),
        QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Reasons remain compositional: extra edge crossing, face loop/open endpoint/branch, authored open/nonmanifold boundary, or future interior-topology residual. Unknown cases abstain.",
    ),
)


def validate_f_extraction_truth_extension_v1() -> None:
    names = []
    for channel in F_EXTRACTION_CHANNELS_V1:
        channel.validate()
        names.append(channel.name)
        if "F" not in channel.consumed_by:
            raise ValueError(f"FIT8_F_EXTRACTION_CHANNEL_NOT_CONSUMED_BY_F:{channel.name}")
    if len(names) != len(set(names)):
        raise ValueError("FIT8_F_EXTRACTION_CHANNEL_DUPLICATE")


validate_f_extraction_truth_extension_v1()


__all__ = [
    "F_EXTRACTION_CHANNELS_V1",
    "SCHEMA_VERSION",
    "STATUS",
    "validate_f_extraction_truth_extension_v1",
]
