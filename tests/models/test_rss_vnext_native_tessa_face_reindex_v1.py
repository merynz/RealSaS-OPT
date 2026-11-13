from __future__ import annotations

"""Source-native → TESSA teacher face reindex tests (no contact oracle)."""
from dataclasses import asdict,replace
import pytest
import torch
import numpy as np

from models.rss_vnext.attachment_face_correspondence_v1 import mesh_topology_digest_v1
from tests.models.test_rss_vnext_attachment_face_correspondence_v1 import _tokens
from tests.models.test_rss_vnext_inference_checkpoint_bridge_v1 import _network,_images
from tools.truth_foundry.native_tessa_face_reindex_v1 import (
    reindex_source_faces_to_tessa_teacher_v1,
    read_native_tessa_face_reindex_candidate_v1,
)


@pytest.fixture(autouse=True)
def cpu_deterministic():
    torch.manual_seed(681)
    torch.set_num_threads(1)


def _native(*,reverse_winding=True):
    # This fixture is a test-only source native mesh with two independent
    # SOURCE components, but its *face list order* differs from TESSA's.
    # It carries NO source socket, attachment or intervention truth.
    xyz=np.array([
        [-.35,-.35,0],[-.15,-.35,0],[-.35,-.15,0],
        [.15,.15,0],[.35,.15,0],[.15,.35,0],
    ],dtype=np.float64)
    face1=(3,5,4) if reverse_winding else (3,4,5)
    faces=np.array([face1,(0,1,2)],dtype=np.int64)
    comps=np.array([1,0],dtype=np.int64)
    net=_network().eval()
    tokens=tuple(map(int,_tokens(net).tolist()))
    return net,xyz,faces,comps,tokens


def _run(*,reverse_winding=True):
    net,xyz,faces,comps,tokens=_native(
        reverse_winding=reverse_winding,
    )
    result=reindex_source_faces_to_tessa_teacher_v1(
        source_vertices_normalized=xyz,
        source_faces=faces,
        source_face_component_indices=comps,
        tessa_teacher_token_ids=tokens,
        coordinate_bins=net.cfg.tessa.coordinate_bins,
    )
    return net,result


def test_bijective_source_face_reindex_survives_tessa_component_and_face_reordering():
    net,r=_run()
    assert r.source_face_to_tessa_face==(1,0)
    assert r.tessa_face_to_source_face==(1,0)
    assert r.source_face_components==(1,0)
    assert r.tessa_face_components==(0,1)
    assert r.source_corner_to_tessa_corner==((0,2,1),(0,1,2))
    assert r.source_face_winding_parity==(1,0)
    assert r.authority=="STRUCTURAL_FACE_IDENTITY_IN_EXACT_TESSA_QUANTIZED_DOMAIN_ONLY"
    assert r.trainer_admission_qualified is False
    assert not any((r.source_contact_qualified,
                    r.native_rig_attachment_qualified,
                    r.source_mesh_geometry_qualified,
                    r.source_drawable_qualified))


def test_source_reindex_digest_equals_real_student_mesh_topology_digest():
    net,r=_run()
    rgba,cam=_images()
    with torch.no_grad():
        state,_,_=net.encode_observations(rgba,cam)
        _,_,_,_,tokens=_native()
        actual=net.build_mesh_candidate(
            state,tokens,source_authority="TEACHER_FORCED_TRAINING_ONLY",
        )
    assert r.tessa_mesh_topology_digest==mesh_topology_digest_v1(actual)
    assert actual.topology_token_ids==tokens
    assert actual.faces.shape==(2,3)


def test_winding_flip_is_recorded_but_not_qualified_as_source_orientation():
    net,r=_run(reverse_winding=False)
    assert r.source_face_to_tessa_face==(1,0)
    assert r.source_face_winding_parity==(0,0)
    assert r.source_corner_to_tessa_corner==((0,1,2),(0,1,2))
    assert not r.source_mesh_geometry_qualified


def test_same_component_quantization_collision_is_quarantined_not_nearest_matched():
    net,xyz,faces,comps,tokens=_native()
    # Place the two vertex IDs 3 and 5 at one quantized TESSA location.
    xyz[5]=xyz[3]+np.array([0.00002,0.,0.])
    with pytest.raises(ValueError,match="QUANTIZATION_VERTEX_COLLISION"):
        reindex_source_faces_to_tessa_teacher_v1(
            source_vertices_normalized=xyz,source_faces=faces,
            source_face_component_indices=comps,
            tessa_teacher_token_ids=tokens,
            coordinate_bins=net.cfg.tessa.coordinate_bins,
        )


def test_different_components_may_have_same_quantized_coordinates():
    net,xyz,faces,comps,tokens=_native()
    # Source has no source owner correspondence at this same position,
    # but per-component coordinate reuse is permitted by the grammar.
    xyz[3]=xyz[0]
    # Student component-1 still expects its own original coordinates,
    # so this mismatch must fail rather than treating overlapping geometry
    # as a valid matching source triangle.
    with pytest.raises(ValueError,match="NATIVE_TESSA_TRIANGLE_KEY_BIJECTION_MISSING"):
        reindex_source_faces_to_tessa_teacher_v1(
            source_vertices_normalized=xyz,source_faces=faces,
            source_face_component_indices=comps,
            tessa_teacher_token_ids=tokens,
            coordinate_bins=net.cfg.tessa.coordinate_bins,
        )


def test_same_face_count_with_different_mesh_cannot_borrow_source_labels():
    net,xyz,faces,comps,tokens=_native()
    xyz[4,0]=.10 # remains inside normalized range but different TESSA bin
    with pytest.raises(ValueError,match="NATIVE_TESSA_TRIANGLE_KEY_BIJECTION_MISSING"):
        reindex_source_faces_to_tessa_teacher_v1(
            source_vertices_normalized=xyz,source_faces=faces,
            source_face_component_indices=comps,
            tessa_teacher_token_ids=tokens,
            coordinate_bins=net.cfg.tessa.coordinate_bins,
        )


def test_missing_or_relabelled_component_does_not_preserve_native_face_index():
    net,xyz,faces,comps,tokens=_native()
    with pytest.raises(ValueError,match="SOURCE_FACE_OR_COMPONENT_NOT_INDEXED"):
        reindex_source_faces_to_tessa_teacher_v1(
            source_vertices_normalized=xyz,source_faces=faces,
            source_face_component_indices=np.array([0,2],dtype=np.int64),
            tessa_teacher_token_ids=tokens,
            coordinate_bins=net.cfg.tessa.coordinate_bins,
        )
    with pytest.raises(ValueError,match="NATIVE_TESSA_TRIANGLE_KEY_BIJECTION_MISSING"):
        reindex_source_faces_to_tessa_teacher_v1(
            source_vertices_normalized=xyz,source_faces=faces,
            source_face_component_indices=np.array([0,1],dtype=np.int64),
            tessa_teacher_token_ids=tokens,
            coordinate_bins=net.cfg.tessa.coordinate_bins,
        )


def test_invalid_normalization_and_duplicate_source_face_rejected():
    net,xyz,faces,comps,tokens=_native()
    xyz[0,0]=.9
    with pytest.raises(ValueError,match="SOURCE_NOT_ALREADY_IN_TESSA_NORMALIZED_FRAME"):
        reindex_source_faces_to_tessa_teacher_v1(
            source_vertices_normalized=xyz,source_faces=faces,
            source_face_component_indices=comps,
            tessa_teacher_token_ids=tokens,
            coordinate_bins=net.cfg.tessa.coordinate_bins,
        )
    net,xyz,faces,comps,tokens=_native()
    faces[0]=(3,3,5)
    with pytest.raises(ValueError,match="SOURCE_FACE_OR_COMPONENT_NOT_INDEXED"):
        reindex_source_faces_to_tessa_teacher_v1(
            source_vertices_normalized=xyz,source_faces=faces,
            source_face_component_indices=comps,
            tessa_teacher_token_ids=tokens,
            coordinate_bins=net.cfg.tessa.coordinate_bins,
        )


def test_candidate_roundtrip_preserves_hash_without_source_authority():
    net,r=_run()
    loaded=read_native_tessa_face_reindex_candidate_v1(asdict(r))
    assert loaded==r
    data=asdict(r)
    data["source_face_to_tessa_face"]=(0,1)
    with pytest.raises(ValueError,match="CANDIDATE_HASH_OR_AUTHORITY_DRIFT"):
        read_native_tessa_face_reindex_candidate_v1(data)
    data=asdict(r)
    data["trainer_admission_qualified"]=True
    with pytest.raises(ValueError,match="CANDIDATE_HASH_OR_AUTHORITY_DRIFT"):
        read_native_tessa_face_reindex_candidate_v1(data)
    data=asdict(r)
    data["schema"]="SyntheticCertifiedProduct"
    with pytest.raises(ValueError,match="CANDIDATE_HASH_OR_AUTHORITY_DRIFT"):
        read_native_tessa_face_reindex_candidate_v1(data)


def test_teacher_change_is_not_overridden_by_same_source_index_map():
    net,xyz,faces,comps,tokens=_native()
    # Wrong coordinate bins make even the same token values a different
    # geometry. Reindex must reject rather than inherit source addresses.
    # The TESSA grammar's special IDs depend on coordinate_bins; replaying
    # a 128-bin teacher in a 256-bin decoder must fail at the BOS grammar.
    with pytest.raises(ValueError,match="TESSA_DECODE_BOS_MISSING"):
        reindex_source_faces_to_tessa_teacher_v1(
            source_vertices_normalized=xyz,source_faces=faces,
            source_face_component_indices=comps,
            tessa_teacher_token_ids=tokens,
            coordinate_bins=256,
        )
