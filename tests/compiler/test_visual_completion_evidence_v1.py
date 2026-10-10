from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_completion_evidence_v1 import (
    inspect_visual_completion_field, source_visual_boundary_evidence,
    validate_source_visual_boundary_evidence,
)
from compiler.realsas_compiler_core.visual_material_render_v1 import (
    VisualRasterQualificationError, render_visual_material,
)


def _asset(tmp_path, bad_donor=False):
    path = tmp_path / "completion.npz"
    donor = np.full((8, 2), -3, dtype=np.int16)
    if bad_donor:
        donor[0, 0] = 0
    np.savez_compressed(path, sample_positions=np.zeros((2, 3)),
        sample_face_index=np.array([0, 1]), rgba=np.full((8, 2, 4), 255, dtype=np.uint8),
        provenance=np.full((8, 2), 4, dtype=np.uint8), source_view=donor)
    return SimpleNamespace(asset_hash="a" * 64, metadata={
        "canonical_completion_field_bound": True,
        "canonical_completion_field_path": str(path),
        "canonical_completion_field_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_completion_field_schema": "RealSaS.CAACompileArrays.v3",
        "canonical_completion_support_domain": "CANONICAL_SURFACE_ADDRESSING",
        "canonical_completion_field_compile_hash": "c" * 64,
        # Optimistic flags cannot produce support or a material consumer.
        "hidden_layer_support_qualified": True,
    })


def test_colours_on_canonical_surface_do_not_qualify_hidden_visual_layers(tmp_path):
    asset = _asset(tmp_path)
    field = inspect_visual_completion_field(asset)
    assert field["provenance_counts"]["4"] == 16
    assert field["validation_read"] is True
    assert field["transport_sample_count"] == 0
    evidence = source_visual_boundary_evidence(asset=asset,
        appearance=SimpleNamespace(qualification_hash="b" * 64),
        presentation=SimpleNamespace(set_hash="d" * 64), completion=field)
    validate_source_visual_boundary_evidence(evidence)
    assert evidence["full_visual_acceptance_passed"] is False
    assert evidence["relationships"][0]["required_relation_count"] is None
    assert evidence["relationships"][2]["status"] == "PRODUCED_CANONICAL_FIELD_WITHOUT_VISUAL_TRANSPORT"


def test_changed_completion_bytes_are_rejected_at_the_consumer(tmp_path):
    asset = _asset(tmp_path)
    with open(asset.metadata["canonical_completion_field_path"], "ab") as stream:
        stream.write(b"drift")
    with pytest.raises(QualificationError, match="BYTES_DRIFT"):
        inspect_visual_completion_field(asset)


def test_resealed_invalid_completion_provenance_is_rejected(tmp_path):
    with pytest.raises(QualificationError, match="PROVENANCE_DRIFT"):
        inspect_visual_completion_field(_asset(tmp_path, bad_donor=True))


@pytest.mark.parametrize("reseal", [False, True])
def test_green_flag_cannot_replace_missing_relation_producer_and_consumer(tmp_path, reseal):
    asset = _asset(tmp_path)
    evidence = source_visual_boundary_evidence(asset=asset,
        appearance=SimpleNamespace(qualification_hash="b" * 64),
        presentation=None, completion=inspect_visual_completion_field(asset))
    evidence = deepcopy(evidence)
    evidence["full_visual_acceptance_passed"] = True
    if reseal:
        evidence.pop("evidence_hash")
        evidence["evidence_hash"] = content_sha256(evidence)
    with pytest.raises(QualificationError, match="HASH_DRIFT|UNIMPLEMENTED_CAPABILITY_CLAIM"):
        validate_source_visual_boundary_evidence(evidence)


@pytest.mark.parametrize("depths,expected", [([1., 1.], "tie"), ([1., 2., 3., 4., 5.], "overflow")])
def test_raster_failure_distinguishes_order_ambiguity_and_layer_capacity(depths, expected):
    rest = np.array([[0., 0.], [7., 0.], [0., 7.]])
    positions = np.tile(rest, (len(depths), 1))
    faces = np.arange(len(positions)).reshape(-1, 3)
    texture = np.full((8, 8, 4), 255, dtype=np.uint8)
    with pytest.raises(VisualRasterQualificationError) as failure:
        render_visual_material(positions=positions, depths=np.repeat(depths, 3),
            faces=faces, uv=positions / 7, texture=texture,
            provenance=np.zeros((8, 8), dtype=np.uint8),
            source_view=np.zeros((8, 8), dtype=np.int16), view_index=0, resolution=8)
    d = failure.value.diagnostics
    assert (d["unresolved_depth_tie_count"] > 0) == (expected == "tie")
    assert (d["fragment_overflow_count"] > 0) == (expected == "overflow")
    examples = d["depth_tie_examples"] if expected == "tie" else d["fragment_overflow_examples"]
    assert examples and "pixel_xy" in examples[0]
