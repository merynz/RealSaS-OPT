"""Byte parity cannot admit unowned painter-order overlaps."""
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image
import pytest

from compiler.realsas_compiler_services.orchestrator.adapters import runtime_v2 as rt


def scene(tmp_path, *, overlap=True, transparent=False, reverse=False):
    texture = np.zeros((8, 8, 4), dtype=np.uint8)
    texture[:, :4] = (255, 0, 0, 255)
    texture[:, 4:] = (0, 0, 255, 0 if transparent else 255)
    path = tmp_path / 'source.png'
    Image.fromarray(texture).save(path)
    rest = np.array([[1.,1.],[6.,1.],[1.,6.],[1.,1.],[6.,1.],[1.,6.]])
    if not overlap:
        rest[3:] = [[6.,1.],[6.,6.],[1.,6.]]
    faces = np.array([[0,1,2],[3,4,5]])
    if reverse:
        faces = faces[::-1].copy()
    uv = np.array([[0.,0.]]*3+[[1.,0.]]*3)
    arrays = {'view_0_uv':uv, 'view_0_faces':faces, 'view_0_rest_positions':rest,
              'clip_0_view_0_positions':rest[None]}
    view = NS(view_index=0, view_id='V0', visual_vertex_count=6, visual_face_count=2,
              texture_path=str(path),source_width=8, source_height=8,camera={'resolution':8})
    clip = NS(array_prefix='clip_0',clip_id='test',frame_count=1)
    projection = NS(projection_hash='projection',views=(view,),clips=(clip,))
    reference = rt._source_owned_visual_reference_frame(projection,arrays,clip=clip,view=view,frame_index=0)
    return projection,arrays,reference


def test_overlap_is_detected_even_when_orientation_and_stretch_pass(tmp_path):
    _,arrays,a = scene(tmp_path)
    _,_,b = scene(tmp_path,reverse=True)
    assert a.unresolved_overlap_pixel_count > 0
    assert a.unresolved_overlap_pixel_count == b.unresolved_overlap_pixel_count
    assert a.maximum_pixel_contributor_count == 2
    assert np.any(a.straight_rgba_u8 != b.straight_rgba_u8)
    metrics = rt._visual_mesh_motion_metrics(arrays['view_0_rest_positions'],arrays['view_0_rest_positions'],arrays['view_0_faces'])
    assert metrics['flipped_triangle_count'] == metrics['edge_gt_4_count'] == 0


@pytest.mark.parametrize('kwargs',[{'overlap':False},{'transparent':True}])
def test_shared_edges_and_transparent_samples_do_not_create_collision(tmp_path,kwargs):
    _,_,ref = scene(tmp_path,**kwargs)
    assert ref.unresolved_overlap_pixel_count == 0
    assert ref.maximum_pixel_contributor_count == 1


@pytest.mark.parametrize('overlap',[True,False])
def test_stage45_parity_does_not_override_missing_depth_authority(tmp_path,monkeypatch,overlap):
    projection,arrays,ref = scene(tmp_path,overlap=overlap)
    archive=tmp_path/'test.rss'; archive.write_bytes(b'fixture')
    package=NS(package_hash='package',archive_path=str(archive),archive_sha256=rt.sha256_file(archive),
        metadata={'presentation_geometry_mode':'SOURCE_OWNED_VISUAL_PRESENTATION_V1','mechanical_mesh_render_authority':False})
    playback=NS(package_binding_hash='package',projection_binding_hash='projection',native_player_sha256='player',playback_hash='playback')
    monkeypatch.setattr(rt,'_native_player',lambda ctx:(tmp_path/'player','player'))
    monkeypatch.setattr(rt,'_projection_arrays',lambda p:arrays)
    monkeypatch.setattr(rt,'_native_parallel_workers',lambda ctx:1)
    paths=[tmp_path/x for x in ('rgba','prov','owner')]
    for path,array in zip(paths,(ref.straight_rgba_u8,ref.provenance_code,ref.owner_face_index.astype('<i4'))):
        path.write_bytes(array.tobytes())
    monkeypatch.setattr(rt,'_run_native_many',lambda **kw:[(*paths,'renderer=REALSAS_V2_SOURCE_OWNED_VISUAL_2D')])
    monkeypatch.setattr(rt,'write_ir',lambda *a,**kw:{})
    result=rt._prove_source_owned_visual_dynamic_integrity({'run_root':tmp_path,'stage':{'id':'45'}},projection=projection,package=package,playback=playback)
    assert result['status'] == ('FAIL' if overlap else 'PASS')
    if overlap:
        q=result['diagnostics']['qualification_report']
        assert q['native_reference_byte_parity_passed']
        assert not q['visual_occlusion_passed']
        assert 'SOURCE_VISUAL_DEPTH_ORDER_AUTHORITY_MISSING' in result['blockers']


def test_old_pass_receipt_without_occlusion_evidence_is_rejected():
    from dataclasses import replace
    from compiler.realsas_compiler_core.runtime_visual_authority_v1 import (
        SourceOwnedVisualDynamicIntegrityV1IR, source_owned_visual_dynamic_integrity_hash,
        validate_source_owned_visual_dynamic_integrity,
    )
    from compiler.realsas_compiler_core.types import QualificationError
    value=SourceOwnedVisualDynamicIntegrityV1IR(
        package_binding_hash='a'*64,projection_binding_hash='b'*64,native_playback_binding_hash='c'*64,
        evaluated_frame_view_count=1,rendered_visible_pixel_count=10,empty_frame_view_count=0,
        flipped_triangle_count=0,edge_gt_4_count=0,edge_gt_10_count=0,maximum_p95_edge_ratio=1.,
        maximum_edge_ratio=1.,native_reference_mismatch_pixel_count=0,
        direct_source_provenance_mismatch_pixel_count=0,
        qualification_report={'status':'PASS_SOURCE_OWNED_VISUAL_DYNAMIC_INTEGRITY'},integrity_hash='')
    value=replace(value,integrity_hash=source_owned_visual_dynamic_integrity_hash(value))
    with pytest.raises(QualificationError,match='OCCLUSION_EVIDENCE_MISSING_OR_FAILED'):
        validate_source_owned_visual_dynamic_integrity(value)
    value=replace(value,qualification_report={**value.qualification_report,'visual_occlusion_passed':True,
        'unresolved_overlap_pixel_count':0,'occlusion_scope':'SAMPLED_ALPHA_POSITIVE_PIXEL_COLLISIONS'})
    value=replace(value,integrity_hash=source_owned_visual_dynamic_integrity_hash(value))
    validate_source_owned_visual_dynamic_integrity(value)
