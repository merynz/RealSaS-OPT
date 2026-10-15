from tools.corpus.fit8_f_extraction_truth_extension_v1 import (
    F_EXTRACTION_CHANNELS_V1,
    STATUS,
    validate_f_extraction_truth_extension_v1,
)
from tools.corpus.fit8_full_truth_schema_v1 import TEACHER_CHANNELS_V1


def test_extension_is_explicitly_not_silent_bundle_v1_mutation():
    assert STATUS == "RESEARCH_EXTENSION_NOT_YET_PROMOTED_TO_FULL_TRUTH_BUNDLE_V2"
    base_names = {channel.name for channel in TEACHER_CHANNELS_V1}
    extension_names = {channel.name for channel in F_EXTRACTION_CHANNELS_V1}
    assert not (base_names & extension_names)


def test_extraction_truth_is_variable_cardinality_and_scale_explicit():
    by_name = {channel.name: channel for channel in F_EXTRACTION_CHANNELS_V1}
    assert by_name["extraction_scale_manifest"].shape == "[L]"
    for name in (
        "cell_supported_patch_partition",
        "cell_edge_crossing_set",
        "cell_face_strand_set",
        "face_strand_edge_crossing_binding",
        "cell_authored_edge_incidence",
        "cell_topology_residual_target",
    ):
        assert "ragged" == by_name[name].dtype
        assert "K=4" not in by_name[name].notes


def test_grid_local_patch_identity_is_not_redefined_as_semantic_owner_truth():
    by_name = {channel.name: channel for channel in F_EXTRACTION_CHANNELS_V1}
    patch = by_name["cell_supported_patch_partition"]
    assert patch.consumed_by == ("F",)
    assert "never equate with object/material/owner/component" in patch.notes


def test_extension_schema_validates():
    validate_f_extraction_truth_extension_v1()
