from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.fit_observation_v1.master_aligned_source_appearance_v1 import (
    assert_geometry_identity,
    render_asset,
)


def _fixture(tmp_path: Path):
    geom=tmp_path/'primary_geometry.npz'
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    faces=np.array([[0,1,2]],np.int32)
    np.savez_compressed(geom,vertices_source=vertices,vertices=vertices,faces=faces,vertex_normals=np.array([[0,0,1]]*3,np.float32))

    app=tmp_path/'appearance.npz'
    np.savez_compressed(app,vertices_source=vertices,faces=faces,
                        face_uv=np.array([[[0,0],[1,0],[0,1]]],np.float32),
                        face_uv_valid=np.array([1],np.uint8),face_material=np.array([0],np.int32))
    pkg=tmp_path/'pkg'; pkg.mkdir(); tex=pkg/'hero.png'
    Image.new('RGBA',(4,4),(20,120,220,255)).save(tex)
    struct=tmp_path/'appearance.json'
    struct.write_text(json.dumps({'materials':[{'name':'hero','base_color':[.1,.2,.3,1],'image_name':'hero.png','image_path':None}]}),encoding='utf-8')

    renders=tmp_path/'renders'; center=512*1024+512
    for vi in range(8):
        vd=renders/f'V{vi}'; vd.mkdir(parents=True)
        np.savez_compressed(vd/'raster_authority.npz',pixel_linear_index=np.array([center],np.int64),triangle_id=np.array([0],np.int32),barycentric_uv=np.array([[1/3,1/3]],np.float32),resolution=np.array([1024,1024]),origin=np.array(['TOP_LEFT']))
        (vd/'camera.json').write_text(json.dumps({'yaw_deg':vi*45}),encoding='utf-8')
    return geom,app,pkg,struct,renders


def test_master_raster_is_the_only_pixel_authority(tmp_path: Path):
    geom,app,pkg,struct,renders=_fixture(tmp_path); out=tmp_path/'out'
    manifest=render_asset(asset_id='asset_x',candidate_id='candidate_x',source_sha256='a'*64,
                          source_file=pkg/'source.fbx',source_package=pkg,canonical_geometry=geom,
                          appearance_npz=app,appearance_json=struct,canonical_render_root=renders,
                          output_root=out,min_texture_coverage=.99)
    assert manifest['rerasterized'] is False
    assert manifest['render_geometry_authority']=='EXISTING_MASTER_RASTER_AUTHORITY'
    assert manifest['raster_authority']=='MASTER_SOURCE_TEXTURED_RGBA'
    assert manifest['textured_foreground_fraction']==1.0
    for vi in range(8):
        with Image.open(out/f'V{vi}'/'RGBA.png') as im:
            arr=np.asarray(im)
            assert im.mode=='RGBA' and im.size==(1024,1024)
            assert int((arr[:,:,3]>0).sum())==1
            assert tuple(arr[512,512,:3])==(20,120,220)


def test_source_geometry_must_match_master_exactly(tmp_path: Path):
    geom,app,*_=_fixture(tmp_path)
    with np.load(app,allow_pickle=False) as x:
        data={k:x[k] for k in x.files}
    data['vertices_source']=data['vertices_source'].copy(); data['vertices_source'][0,0]=1e-3
    bad=tmp_path/'bad.npz'; np.savez_compressed(bad,**data)
    with pytest.raises(RuntimeError,match='SOURCE_CANONICAL_GEOMETRY_DRIFT'):
        assert_geometry_identity(geom,bad)
